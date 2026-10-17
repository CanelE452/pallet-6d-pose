# Reviewed publication V2

The original experimental commands remain in [REPRODUCE.md](REPRODUCE.md).
This supplement records the final packaging attempt after independent review
clarified the difference between fixed RGB queries and the candidate's infinite
supporting-line extension. Predictions, metrics, references, model settings,
training runs and method selection did not change.

The first private report remains at `$RUN_PRIVATE/report_v1`. Its complete
public bytes, exact source preimages and receipts are preserved under
`$RUN_PRIVATE/publication_attempt1_preserved`. The reviewed report uses new
receipt paths so the first attempt is never overwritten.

With the original private inputs and completed sealed outputs available, the
actual final command is:

```bash
python -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.finalize \
  --run-dir "$RUN_PRIVATE" \
  --report-dir "$RUN_PRIVATE/report_v2" \
  --independent-receipt "$RUN_PRIVATE/independent_v2/INDEPENDENT_CHECK.json" \
  --gallery-receipt "$RUN_PRIVATE/gallery_v3/GALLERY_RECEIPT.json" \
  --mechanism-dir "$RUN_PRIVATE/mechanism_v1" \
  --cost-ledger "$RUN_PRIVATE/ROOT_COST_SCOPE_REVIEW_V2.json" \
  --publication-receipt-path "$RUN_PRIVATE/PUBLICATION_RECEIPT_V2.json" \
  --final-receipt-path "$RUN_PRIVATE/FINALIZATION_RECEIPT_V2.json" \
  --output-dir "$SOURCE_ROOT/_docs/experiments/pallet_statewise_end_to_end_recovery_v1" \
  --final-GO
```

The controller's `REPORT_PUBLICATION_REVIEW_REPAIR_V2` stage runs this command.
`campaign resume --run-dir "$RUN_PRIVATE"` checks the current runtime stage
proofs and resumes only unfinished work. Repeating a completed finalization
checks the existing V2 report and publication receipts. It does not train,
generate candidates or score the evaluation set again.

The small public results cannot reproduce private RGB, cached networks or full
pose banks by themselves. The original data requirements and hashes remain
explicit in the original guide and source bindings.
