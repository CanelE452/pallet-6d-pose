# Claim–evidence matrix

| Claim | Evidence | Estimand / N / units | Uncertainty | Novelty relation | Remaining check | Permissible manuscript wording |
|---|---|---|---|---|---|---|
| P lowers developed2D residual | P_VS_R0_PAIRED + full-precision predictions | mean seed pooled median difference,311 matched/319 total frames,2756 points,px | -0.7108 [-1.1737,-0.4111] session95% | empirical signal, not new expectation | independent sessions/QA | On reused development data the fixed refiner reduced pooled median error. |
| Detector outputs are preserved | original P inference audits; D actual3008-image checks; runtime parity | boxes/scores/order/top1/center | exact tested identity | implementation contract | portability beyond fixed R0 | The tested wrapper preserves the detector outputs. |
| Same-evidence direct control | D_PROTOCOL_LOCK,tests,D3 checkpoints |19,450 vs18,962 params;8×221×32×2 samples | P−D -0.5254 [-0.9800,-0.2101] | bounded readout/supervision comparison | convergence; formal prior | P has a lower development median than this fixed direct control. |
| Geometry change | canonical MAIN per-frame outputs | translation median reduction7.439mm;319 poses per existing model | exploratory intervals separate | sensing evaluation | independent metrology | Geometry-reference errors changed by the reported amount. |
| Runtime | RUNTIME_PANEL | desktop actual images,26×5/model | sample median/mean/P90; no device-general CI | system cost | Jetson/export if required | Desktop overhead was measured under the stated conditions. |
| Formal prior superiority | none | NOT_YET_MEASURED | none | not established | PoseFix actual trained comparator | No claim currently permitted. |
| Independent generalization | none | NEW_CONFIRMATION_DATA_REQUIRED | none | not established | new captures and blinded labels | No independent confirmation has been conducted. |

Avoid SOTA, first-ever, arbitrary pallet, unseen-instance, model-agnostic, all-metric superiority, Jetson real-time and forklift safety. Related-method links and limits are in PRIOR_ART_MATRIX.md. No journal acceptance inference is made.
