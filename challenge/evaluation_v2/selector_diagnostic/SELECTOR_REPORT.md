# DEV_POS140 W/D selector diagnostic

Status: **POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR**

This is a development diagnostic, never a FINAL result. The frozen selector
configuration was not changed after observing these values.

## Gate

- overall: `0.5928571428571429` (required >= 0.95)
- NIGHT: `0.4642857142857143` (required >= 0.90)
- minimum session: `0.3333333333333333` (required >= 0.85)
- tail dominance: `False`
- blocked reason: `SELECTOR_TAIL_DOMINANCE_GATE_FAILED`

## Session accuracy

| session | N | correct | accuracy |
|---|---:|---:|---:|
| `eval_cad` | 18 | 18 | 1.000000 |
| `eval_night08` | 12 | 4 | 0.333333 |
| `eval_night09` | 16 | 9 | 0.562500 |
| `eval_noapril` | 12 | 12 | 1.000000 |
| `eval_outside` | 22 | 14 | 0.636364 |
| `eval_pallet07` | 27 | 12 | 0.444444 |
| `eval_pallet09` | 33 | 14 | 0.424242 |

## Contract

Selection used only predicted 9-keypoint coordinates, camera intrinsics,
fixed physical dimensions (1.10/0.11/1.30 m), and frozen scoring constants.
GT parity and canonical equivalence-class poses were opened only after all
selection decisions had completed.

A FAIL is not tuned away on DEV. Only source/synthetic-only redesign may be
proposed before a new pre-registered diagnostic.
