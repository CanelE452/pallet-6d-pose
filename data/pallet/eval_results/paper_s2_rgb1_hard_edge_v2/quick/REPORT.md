# paper_s2_rgb1_hard_edge_v2 — quick evaluation

Protocol locks: RGB single frame, input 400, belief 50, init ep57, existing Stage-B data only.
Selection: synthetic + filterval123. handannot17 is report-only. Sealed final-test sessions were not enumerated.

## Synthetic checkpoints

| checkpoint | arm | det% | front | rear | corner | legacy h8 | safe h8 | UC% | safe yield% | safe accept% | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_ep57 | baseline | 89.2 | 7.54 | 8.41 | 7.81 | 8.26 | 8.29 | 1.4 | 34.2 | 52.2 | 152 |
| hard_edge_tail_ep0058 | hard_edge_tail | 89.8 | 7.56 | 8.41 | 7.82 | 8.31 | 8.29 | 1.8 | 35.0 | 53.0 | 150 |
| hard_edge_tail_ep0059 | hard_edge_tail | 89.8 | 7.54 | 8.39 | 7.83 | 8.25 | 8.33 | 1.8 | 34.0 | 52.2 | 154 |
| hard_edge_tail_ep0060 | hard_edge_tail | 89.8 | 7.54 | 8.43 | 7.81 | 8.28 | 8.17 | 1.6 | 35.2 | 53.2 | 149 |

## Per-arm synthetic selection

- `hard_edge_tail`: `hard_edge_tail_ep0060`; synthetic_guard=True

## Real development and secondary check

### baseline_ep57

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.79 | 19.04 | 13.55 | 18.69 | 16.48 | 23/24.7 | 5.7 | 44.7 | 58.3 | 55.6 | - | 15 |
| handannot17 | 23.5 | 5.21 | 8.24 | 6.24 | 8.75 | 7.52 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

### hard_edge_tail_ep0058

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.75 | 18.95 | 13.46 | 18.88 | 16.57 | 24/25.8 | 8.1 | 48.0 | 76.9 | 53.3 | - | 12 |
| handannot17 | 23.5 | 5.15 | 8.06 | 6.13 | 8.66 | 7.47 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

### hard_edge_tail_ep0059

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.75 | 18.91 | 13.46 | 18.34 | 16.87 | 23/24.7 | 6.5 | 47.2 | 66.7 | 51.1 | - | 14 |
| handannot17 | 23.5 | 5.15 | 7.98 | 6.13 | 8.62 | 7.42 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

### hard_edge_tail_ep0060

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.77 | 18.86 | 13.39 | 18.41 | 16.87 | 22/23.7 | 6.5 | 48.8 | 61.5 | 48.9 | - | 13 |
| handannot17 | 23.5 | 5.19 | 7.96 | 6.17 | 8.62 | 7.43 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

## Decision

- BEST_ARM=`baseline`
- BEST_CKPT=`/home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB/net_epoch_0057.pth`
- reason: baseline retained: no scoped checkpoint met the pre-registered count-preservation and UC-improvement limits
- rank_sums: `{}`
- rejected: `{"hard_edge_tail_ep0058": ["safe_gross=21>20", "uc_main=24>18"], "hard_edge_tail_ep0059": ["safe_gross=22>20", "uc_main=23>18"], "hard_edge_tail_ep0060": ["filterval_guard", "safe_gross=23>20", "uc_main=22>18"]}`

W/D learned prior: disabled; ambiguous candidates are rejected.

handannot17 did not participate in this decision.

Scoped-training preservation policy: `{"max_detection_drop_frames": 2, "max_legacy_pnp_drop_frames": 2, "max_safe_accept_drop_frames": 3, "max_safe_gross_increase_frames": 0, "max_uc_eligible_drop_frames": 5, "min_safe_good_delta_frames": 0, "min_uc_improvement_frames": 5}`
Filterval exact counts: `{"baseline_ep57": {"detection": 94, "legacy_pnp": 94, "safe_accept": 55, "safe_good": 7, "safe_gross": 20, "uc_eligible": 93, "uc_main": 23}, "hard_edge_tail_ep0058": {"detection": 94, "legacy_pnp": 94, "safe_accept": 59, "safe_good": 10, "safe_gross": 21, "uc_eligible": 93, "uc_main": 24}, "hard_edge_tail_ep0059": {"detection": 94, "legacy_pnp": 94, "safe_accept": 58, "safe_good": 8, "safe_gross": 22, "uc_eligible": 93, "uc_main": 23}, "hard_edge_tail_ep0060": {"detection": 94, "legacy_pnp": 94, "safe_accept": 60, "safe_good": 8, "safe_gross": 23, "uc_eligible": 93, "uc_main": 22}}`
