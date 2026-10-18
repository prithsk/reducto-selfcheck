import bench
good = "<table><tr><th>Item</th><th>Q1</th></tr><tr><td>Rev</td><td>1,200</td></tr><tr><td>Cost</td><td>(300)</td></tr><tr><td>Net</td><td>900</td></tr></table>"
bad = "<table><tr><td>Item</td><td>Q1</td><td></td></tr><tr><td>Rev 1,200</td></tr><tr><td>Rev 1,200</td></tr><tr><td>Net</td><td>see note</td><td></td></tr></table>"
fg, fb = bench.features(good), bench.features(bad)
assert fg["ragged"] == 0 and fg["has_header"] == 1 and fg["numeric_col_stray"] == 0, fg
assert fb["ragged"] > 0 and fb["dup_rows"] > 0 and fb["empty"] > 0, fb
assert bench.features("")["no_table"] == 1.0
assert bench.features('<table><tr><td colspan="2">A</td></tr><tr><td>1</td><td>2</td></tr></table>')["ragged"] == 0
r, c = bench.recall_at_precision([0.9, 0.8, 0.7, 0.1], [1, 1, 0, 1], 0.8); assert c == 2 and abs(r - 2 / 3) < 1e-9
print("all tests passed")
