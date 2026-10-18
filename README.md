# reducto-selfcheck

Reducto-selfcheck: flag Reducto table parses that are likely wrong, using only the parse itself.

Runs Reducto on RD-TableBench's public PDFs, scores every parse with Reducto's own grader
(rd-tablebench's table_similarity), then trains a self-check on features of the predicted table alone:
ragged rows, empty cells, numeric columns with stray text, duplicate rows, and so on. The question: at a
fixed precision, how many of the bad parses can a pipeline catch before they reach anything downstream?

## Results

<!-- results:start -->
_Run it to populate._
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
