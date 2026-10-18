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

def reducto_parse(pdf):
    h = {"Authorization": f"Bearer {os.environ['REDUCTO_API_KEY']}"}
    with open(pdf, "rb") as f:
        up = requests.post(f"{URL}/upload", headers=h, files={"file": (os.path.basename(pdf), f, "application/pdf")}, timeout=120)
    up.raise_for_status(); fid = up.json()["file_id"]
    job = requests.post(f"{URL}/parse_async", headers=h, json={"document_url": fid}, timeout=120); job.raise_for_status()
    jid = job.json()["job_id"]
    for _ in range(120):
        info = requests.get(f"{URL}/job/{jid}", headers=h, timeout=60).json()
        if info.get("status") == "Completed": return info["result"]
        if info.get("status") == "Failed": return {"error": info}
        time.sleep(3)
    return {"error": "timeout"}

def main(n, bad_below, target_precision):
    need("REDUCTO_API_KEY")
    if not (RDT / "grading.py").exists(): raise SystemExit(f"clone rd-tablebench next to this folder first: {RDT}")
    sys.path.insert(0, str(RDT))
    from grading import table_similarity; from convert import html_to_numpy; from parsing import parse_reducto_response
    from huggingface_hub import snapshot_download
    data = Path(snapshot_download("reducto/rd-tablebench", repo_type="dataset"))
    pdfs = sorted(glob.glob(str(data / "**" / "*.pdf"), recursive=True))
    gts = {Path(p).stem: p for p in glob.glob(str(data / "**" / "*.html"), recursive=True)}
    pairs = [(p, gts[Path(p).stem]) for p in pdfs if Path(p).stem in gts]
    random.Random(7).shuffle(pairs); pairs = pairs[:n]
    out = ROOT / "parses"; out.mkdir(exist_ok=True); rows = []
    for i, (pdf, gt) in enumerate(pairs, 1):
        fp = out / (Path(pdf).stem + ".json")
        if not fp.exists(): fp.write_text(json.dumps(reducto_parse(pdf)))
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
    rec, caught = recall_at_precision(sc, y, target_precision)
    mean = sum(r["score"] for r in rows) / len(rows)
    md = "\n".join([f"# Reducto table self-check on RD-TableBench, {dt.date.today()}", "",
        f"{len(rows)} public RD-TableBench PDFs parsed by Reducto and graded with rd-tablebench's own `table_similarity`. Mean score {mean:.3f}.",
        f"A parse counts as bad below {bad_below}: {sum(y)} of {len(rows)}.", "",
        "| Metric | Value |", "|---|---|",
        f"| Bad parses caught at {int(100 * target_precision)}% precision, 5-fold CV | **{100 * rec:.0f}%** ({caught} of {sum(y)}) |", "",
        "The self-check reads only the predicted table, never the ground truth, so a pipeline can run it in production.", "",
        "## Honest limits", "",
        "- Only the public subset of RD-TableBench; Reducto holds the full set back to prevent contamination.",
        "- A logistic model on hand-picked structural features. It shows the signal exists, not the ceiling.",
        f"- The {bad_below} cutoff is a choice. The repo reports the curve logic so you can pick another."])
    (ROOT / "results").mkdir(exist_ok=True); (ROOT / "results" / f"{dt.date.today()}.md").write_text(md, encoding="utf-8"); print(md)

from pathlib import Path
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--bad-below", type=float, default=0.8)
    ap.add_argument("--precision", type=float, default=0.8); a = ap.parse_args(); main(a.n, a.bad_below, a.precision)
