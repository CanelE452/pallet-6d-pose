# Byte preservation during publication

The first publication attempt stopped before committing because Git reported a final blank line in the already hash-frozen `posefix_contract_math.py` and CRLF in the CSV-writer-generated `CONFIRMATION_CAPTURE_TEMPLATE.csv`.

Neither artifact was edited to satisfy formatting: changing the math file's bytes would invalidate implementation/model-panel bindings without changing any mathematics. The publisher keeps normal whitespace checks on every other path, and allows only CRLF/final-blank-line formatting for these two explicit files using per-command Git options. It does not change repository/global Git configuration or disable trailing-space checks generally. The initial publication exception remains recorded. No model, prediction, result, or PDF was changed by this publication adjustment.
