# reducto-selfcheck

Reducto-selfcheck: flag Reducto table parses that are likely wrong, using only the parse itself.

Runs Reducto on RD-TableBench's public PDFs, scores every parse with Reducto's own grader
(rd-tablebench's table_similarity), then trains a self-check on features of the predicted table alone:
ragged rows, empty cells, numeric columns with stray text, duplicate rows, and so on. The question: at a
fixed precision, how many of the bad parses can a pipeline catch before they reach anything downstream?

## Results

<!-- results:start -->
# Reducto today vs the outputs shipped in RD-TableBench, 2026-10-18

100 public RD-TableBench tables, graded with rd-tablebench's own table_similarity against the human ground truth. Today's API run with ocr_system=combined, the setting in Reducto's public benchmark code.

| Tables | Count | Outputs shipped in the benchmark | Today's API |
|---|---|---|---|
| All | 100 | 0.896 | 0.788 |
| Came back as Key Value blocks, no Table block | 4 | 0.946 | 0.000 |
| All others | 96 | 0.894 | 0.821 |
| All others, ground truth independent of Reducto's old output | 85 | 0.881 | 0.811 |
| Ground truth byte-identical to Reducto's old output | 12 | 1.000 | 0.826 |

The shipped outputs score 0.896, matching Reducto's published 90.2%, which calibrates the grading pipeline.

## Honest limits

- Reducto's published results reflect its configuration on the run date, and its public harness covers only part of the pipeline. A setting not in the public code may close the gap.
- 100 tables from the public 1,000. Per-table scores are in results/old-vs-today.json.
- An earlier version of this tool paired each table with a file from the wrong folder, since provider outputs share names with the ground truth. Every number here uses the groundtruth folder only.
<!-- results:end -->

## Run it

```
pip install -r requirements.txt
cp .env.example .env    # keys: REDUCTO_API_KEY
git clone https://github.com/reductoai/rd-tablebench ../rd-tablebench
python bench.py --n 200
```

Every run writes `results/<date>.md` with the numbers and an honest-limits section.

## License

MIT
