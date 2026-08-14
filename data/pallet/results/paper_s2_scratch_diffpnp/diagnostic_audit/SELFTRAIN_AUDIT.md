# PAPER_S2 existing self-training audit

This report uses only existing run configuration and pseudo-label history.
No evaluation image, final-test annotation, or final-test frame was opened.

## Round metadata

| run | R1 accepted | R1 rate | R2 accepted | R2 rate |
|---|---:|---:|---:|---:|
| ransac_loo night | 64 | 12.8% | 169 | 33.8% |
| ransac_loo combined | 151 | 10.1% | 360 | 24.0% |
| ransac_loo + flip combined | 110 | 7.3% | 241 | 16.1% |

Each history shows fresh pseudo-label extraction in R2, but the run
configuration executes two rounds unconditionally. There is no independent
R1 gate satisfying the current protocol.

## Validity decision

- The acceptance counts/rates are usable as training-process metadata.
- Historical R0/R1/R2 performance is not paper-valid: the training pool
  shares sessions with filter-val, while the historical outside evaluator
  included sealed `capturepallet07` and `capturepallet09`.
- The old 36-frame manual aggregate is from PL-pool `capturepallet11`, not
  strict filter-validation.
- No nested 0/10/25/50/75/100% quantity sweep or equal-count
  top/middle/bottom/random quality sweep exists.
- Therefore no self-training checkpoint is selected and no performance
  curve from those contaminated evaluations is used in the present audit.

## BLOCKED

```text
BLOCKED:
필요한 항목: session-independent real-unlabeled pool and strict N=87 validation, plus nested quantity/equal-count quality manifests
현재 확인한 위치: data/pallet/results/ralph_selftrain/*/{config.yaml,training_history.json}
시도한 명령: paper_s2_selftrain_history_audit.py (metadata-only aggregation)
실패 원인: existing pool/evaluation membership is contaminated; required sweeps do not exist
대체로 수행한 진단: per-round PL total/accepted/rate/PnP-fail/filter-fail/loss/runtime aggregation
이 blocker가 전체 결론에 미치는 영향: existing self-training gains cannot support a paper claim or choose the final model
```

Machine-readable rows: `data/pallet/results/paper_s2_scratch_diffpnp/diagnostic_audit/self_training_rounds.csv`

Acceptance-only figure: `data/pallet/results/paper_s2_scratch_diffpnp/diagnostic_audit/self_training_acceptance_curve.png`
