# HISTORICAL DEV161 snapshot — do not evaluate

This directory preserves the exact pre-QA evaluator inputs. It is historical
evidence only and contains the 21 labels excluded by the 2026-08-27 GT audit.

`eval_manifest.json` is byte-identical to the original 302,071-byte file:

```text
sha256  272b13d2df90b2184c2df9e65e06f61f9f41ebbb03fca429ea0c78fd45b3b745
items   161 (56 + 105)
status  SUPERSEDED_BY_GT_QA; NOT_AN_ACTIVE_EVALUATION_POPULATION
```

The runnable parent `eval_manifest.json` is the clean 140-item replacement.
Do not copy metrics from this snapshot into the v2 paper tables.

The original generator sources are retained byte-for-byte with the suffix
`.historical.txt`. The non-Python suffix is intentional: the old generator's
hard-coded output targets the active parent and must never be executed.
