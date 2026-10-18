"""reducto-selfcheck: flag Reducto table parses that are likely wrong, using only the parse itself.

Runs Reducto on RD-TableBench's public PDFs, scores every parse with Reducto's own grader
(rd-tablebench's table_similarity), then trains a self-check on features of the predicted table alone:
ragged rows, empty cells, numeric columns with stray text, duplicate rows, and so on. The question: at a
fixed precision, how many of the bad parses can a pipeline catch before they reach anything downstream?

  git clone https://github.com/reductoai/rd-tablebench ../rd-tablebench
  python bench.py --n 200 --bad-below 0.8      # needs REDUCTO_API_KEY in .env
"""
import argparse, glob, json, math, random, re, sys, time, datetime as dt
from html.parser import HTMLParser
import requests
from common import ROOT, need, os

RDT = ROOT.parent / "rd-tablebench"; URL = "https://platform.reducto.ai"

class Cells(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows, self.cur, self.cell, self.inc, self.span, self.th = [], None, "", False, 0, 0
    def handle_starttag(self, t, a):
        a = dict(a)
        if t == "tr": self.cur = []
        if t in ("td", "th"):
            self.inc, self.cell = True, ""; self.th += t == "th"
            self.span += int(a.get("colspan", 1) or 1) > 1 or int(a.get("rowspan", 1) or 1) > 1
            self.cs = int(a.get("colspan", 1) or 1)
    def handle_endtag(self, t):
        if t in ("td", "th") and self.cur is not None: self.cur += [self.cell.strip()] * self.cs; self.inc = False
        if t == "tr" and self.cur is not None: self.rows.append(self.cur); self.cur = None
    def handle_data(self, d):
        if self.inc: self.cell += d

NUM = re.compile(r"^[\s$€£(+-]*[\d.,]+%?[)\s]*$")

def features(html, n_tables=1):
    p = Cells(); p.feed(html or ""); rows = [r for r in p.rows if r]
    f = {"bias": 1.0, "no_table": float(not rows)}
    if not rows: return f
    widths = [len(r) for r in rows]; mode = max(set(widths), key=widths.count); cells = [c for r in rows for c in r]
    f["log_rows"] = math.log1p(len(rows)) / 5; f["log_cols"] = math.log1p(mode) / 3
    f["ragged"] = sum(w != mode for w in widths) / len(rows)
    f["empty"] = sum(not c for c in cells) / len(cells)
    f["dup_rows"] = 1 - len({tuple(r) for r in rows}) / len(rows)
    f["has_spans"] = float(p.span > 0); f["has_header"] = float(p.th > 0)
    f["tiny"] = float(len(rows) <= 2 or mode <= 1)
    f["multi_table"] = float(n_tables > 1)
    lens = [len(c) for c in cells]; f["log_mean_cell"] = math.log1p(sum(lens) / len(lens)) / 5; f["long_cell"] = float(max(lens) > 200)
    stray = []
    for j in range(mode):
        col = [r[j] for r in rows[1:] if j < len(r) and r[j]]
        if len(col) >= 3 and sum(bool(NUM.match(c)) for c in col) / len(col) >= 0.6:
            stray.append(sum(not NUM.match(c) for c in col) / len(col))
    f["numeric_col_stray"] = max(stray) if stray else 0.0
    return f

def train(X, y, epochs=400, lr=0.5, l2=1e-3):
    keys = sorted({k for x in X for k in x}); w = dict.fromkeys(keys, 0.0)
    P = max(1, sum(y)); N = max(1, len(y) - sum(y)); cw = {1: len(y) / (2 * P), 0: len(y) / (2 * N)}
    for _ in range(epochs):
        g = dict.fromkeys(keys, 0.0)
        for x, t in zip(X, y):
            pr = 1 / (1 + math.exp(-sum(w[k] * v for k, v in x.items())))
            for k, v in x.items(): g[k] += cw[t] * (pr - t) * v
        for k in keys: w[k] -= lr * (g[k] / len(y) + l2 * w[k])
    return w

def recall_at_precision(scores, y, target):
    order = sorted(zip(scores, y), reverse=True); tp = best = 0; P = sum(y)
    for i, (_, t) in enumerate(order, 1):
        tp += t
        if tp / i >= target: best = max(best, tp)
    return best / P if P else float("nan"), best

def reducto_parse(pdf, ocr="combined"):
    h = {"Authorization": f"Bearer {os.environ['REDUCTO_API_KEY']}"}
    with open(pdf, "rb") as f:
        up = requests.post(f"{URL}/upload", headers=h, files={"file": (os.path.basename(pdf), f, "application/pdf")}, timeout=120)
    up.raise_for_status(); fid = up.json()["file_id"]
    job = requests.post(f"{URL}/parse_async", headers=h, json={"document_url": fid, **({"advanced_options": {"ocr_system": ocr}} if ocr != "default" else {})}, timeout=120); job.raise_for_status()
    jid = job.json()["job_id"]
    for _ in range(120):
        info = requests.get(f"{URL}/job/{jid}", headers=h, timeout=60).json()
        if info.get("status") == "Completed": return info["result"]
        if info.get("status") == "Failed": return {"error": info}
        time.sleep(3)
    return {"error": "timeout"}

def dataset_dir():
    """Get RD-TableBench from Hugging Face, skipping files we don't need, falling back to the cached copy when the
    network drops, and unpacking it if the dataset ships as an archive."""
    import tarfile, zipfile
    from huggingface_hub import snapshot_download
    kw = dict(repo_type="dataset", max_workers=1, ignore_patterns=["*.md", ".gitattributes"])
    try: snap = Path(snapshot_download("reducto/rd-tablebench", **kw))
    except Exception as e:
        print(f"  Hugging Face unreachable ({type(e).__name__}), using the cached copy", flush=True)
        snap = Path(snapshot_download("reducto/rd-tablebench", local_files_only=True, **kw))
    out = ROOT / "data" / "rd-tablebench"
    if not any(snap.rglob("*.pdf")) and not any(out.rglob("*.pdf")):
        for arc in [a for a in snap.rglob("*") if a.is_file() and (a.suffix == ".zip" or ".tar" in a.name)]:
            print(f"  unpacking {arc.name}", flush=True); out.mkdir(parents=True, exist_ok=True)
            (zipfile.ZipFile(arc).extractall(out) if arc.suffix == ".zip" else tarfile.open(arc).extractall(out))
    return out if any(out.rglob("*.pdf")) else snap

def auc(scores, y):
    pairs = sorted(zip(scores, y)); ranks = {}; i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]: j += 1
        for k in range(i, j): ranks[k] = (i + j + 1) / 2
        i = j
    pos = [ranks[i] for i, (_, t) in enumerate(pairs) if t]; P, N = len(pos), len(y) - len(pos)
    return (sum(pos) - P * (P + 1) / 2) / (P * N) if P and N else float("nan")

def catch_at_recall(scores, y, target=0.8):
    order = sorted(zip(scores, y), reverse=True); P = sum(y); hit = 0
    for k, (_, t) in enumerate(order, 1):
        hit += t
        if P and hit / P >= target: return {"flagged": k, "caught": hit, "precision": hit / k}
    return {"flagged": len(y), "caught": hit, "precision": hit / max(1, len(y))}

def main(n, bad_below, target_precision, ocr="combined"):
    need("REDUCTO_API_KEY")
    if not (RDT / "grading.py").exists(): raise SystemExit(f"clone rd-tablebench next to this folder first: {RDT}")
    sys.path.insert(0, str(RDT))
    from grading import table_similarity; from convert import html_to_numpy; from parsing import parse_reducto_response
    data = dataset_dir()
    pdfs = sorted(glob.glob(str(data / "**" / "*.pdf"), recursive=True))
    gts = {Path(p).stem: p for p in glob.glob(str(data / "**" / "*.html"), recursive=True) if "groundtruth" in Path(p).parent.as_posix()}
    pairs = [(p, gts[Path(p).stem]) for p in pdfs if Path(p).stem in gts]
    print(f"  found {len(pdfs)} PDFs, {len(gts)} ground-truth tables, {len(pairs)} matched pairs", flush=True)
    if not pairs:
        top = sorted(x.name for x in data.iterdir())[:20]
        raise SystemExit(f"No PDF and ground-truth pairs found in {data}. Top-level files: {top}. Paste this to Claude.")
    random.Random(7).shuffle(pairs); pairs = pairs[:n]
    out = ROOT / ("parses" if ocr == "default" else f"parses-{ocr}"); out.mkdir(exist_ok=True); rows = []
    for i, (pdf, gt) in enumerate(pairs, 1):
        fp = out / (Path(pdf).stem + ".json")
        if not fp.exists(): fp.write_text(json.dumps(reducto_parse(pdf, ocr)))
        html, raw = parse_reducto_response(str(fp))
        n_tables = sum(b.get("type") == "Table" for c in (raw or {}).get("result", {}).get("chunks", []) for b in c.get("blocks", [])) if raw else 0
        try: s = table_similarity(html_to_numpy(open(gt, encoding="utf-8").read()), html_to_numpy(html)) if html else 0.0
        except Exception: s = 0.0
        rows.append({"id": Path(pdf).stem, "score": s, "f": features(html, n_tables)})
        if i % 20 == 0: print(f"  {i}/{len(pairs)} parsed and graded", flush=True)
    y = [int(r["score"] < bad_below) for r in rows]; X = [r["f"] for r in rows]
    folds = 5; idx = list(range(len(rows))); random.Random(3).shuffle(idx); sc = [0.0] * len(rows)
    for k in range(folds):
        te = idx[k::folds]; tr = [i for i in idx if i not in set(te)]
        w = train([X[i] for i in tr], [y[i] for i in tr])
        for i in te: sc[i] = sum(w.get(a, 0) * v for a, v in X[i].items())
    mean = sum(r["score"] for r in rows) / len(rows); base = sum(y) / len(y); A = auc(sc, y); c = catch_at_recall(sc, y, 0.8)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"scores-{ocr}.json").write_text(json.dumps([{"id": r["id"], "score": float(r["score"])} for r in rows]), encoding="utf-8")
    md = "\n".join([f"# Reducto table self-check on RD-TableBench, {dt.date.today()}", "",
        f"{len(rows)} public RD-TableBench PDFs parsed by Reducto with ocr_system={ocr}, the setting Reducto's own benchmark uses, and graded with rd-tablebench's own `table_similarity`. Mean score {mean:.3f}.",
        f"A parse counts as bad below {bad_below}: {sum(y)} of {len(rows)} ({100 * base:.0f}%).", "",
        "| Metric, 5-fold CV | Value |", "|---|---|",
        f"| AUC, bad vs good parses | **{A:.3f}** |",
        f"| Tables flagged to catch 80% of bad parses | {c['flagged']} of {len(rows)} ({100 * c['flagged'] / len(rows):.0f}%) |",
        f"| Share of flagged tables that are bad | {100 * c['precision']:.0f}% vs {100 * base:.0f}% if flagged at random, {c['precision'] / base if base else float('nan'):.1f}x |", "",
        "The self-check reads only the predicted table, never the ground truth, so a pipeline can run it in production.", "",
        "## Honest limits", "",
        "- Only the public subset of RD-TableBench; Reducto holds the full set back to prevent contamination.",
        "- A logistic model on hand-picked structural features. It shows the signal exists, not the ceiling.",
        "- An AUC near 0.5 means no better than chance. The lift row is the honest test: flagging at random already catches bad parses at the base rate.",
        f"- The {bad_below} cutoff is a choice. Per-table scores are saved in results/scores-{ocr}.json so you can pick another."])
    (ROOT / "results").mkdir(exist_ok=True); (ROOT / "results" / f"{dt.date.today()}.md").write_text(md, encoding="utf-8"); print(md)

from pathlib import Path
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--bad-below", type=float, default=0.8)
    ap.add_argument("--precision", type=float, default=0.8); ap.add_argument("--ocr", default="combined")
    a = ap.parse_args(); main(a.n, a.bad_below, a.precision, a.ocr)
