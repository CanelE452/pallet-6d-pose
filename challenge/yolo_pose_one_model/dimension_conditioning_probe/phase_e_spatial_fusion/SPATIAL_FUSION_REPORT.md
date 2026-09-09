# Phase-E spatial fusion report

This run is irrevocably exploratory. It preserves the completed Phase-C result and cannot authorize promotion, full-YOLO training, wrapper integration, or deployment.

## Source-only lock

- Locked primary method: `S1_SPATIAL_CONCAT`
- Selection reason: `CONCAT_DEFAULT_COMPLEX_METHOD_NOT_2PP_BETTER`
- Method score: mean selected-epoch synthetic-DEV balanced accuracy over seeds 0/1/2.
- Reported artifact per arm: median-DEV seed, tie lower seed.
- Synthetic TEST and reused real DEV were re-accessed only after the durable Phase-E lock.

## Frozen E-gates

| Gate | Passed |
|---|---:|
| E1 | False |
| E2 | False |
| E3 | False |
| E4 | True |
| E5 | False |
| E6 | False |
| E7 | True |
| E8 | True |
| E9 | False |

Overall E1-E9 pass: `False`.

Synthetic TEST uses each arm's locked median artifact. Real and E-gates use only the locked primary median and S0 median; only the primary receives shuffled/wrong-dimension real controls. Losing conditioned methods were not evaluated on real in Phase E.

Wood pose remains blocked and no multishape-generalization claim is made.

Maximum possible verdict: `EXPLORATORY_SPATIAL_FUSION_DIAGNOSTIC_COMPLETE`.
