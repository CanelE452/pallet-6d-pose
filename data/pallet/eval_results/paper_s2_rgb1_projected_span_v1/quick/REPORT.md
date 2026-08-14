# paper_s2_rgb1_projected_span_v1 — quick evaluation

Protocol locks: RGB single frame, input 400, belief 50, init ep57, existing Stage-B data only.
Selection: synthetic + filterval123. handannot17 is report-only. Sealed final-test sessions were not enumerated.

## Synthetic checkpoints

| checkpoint | arm | det% | front | rear | corner | legacy h8 | safe h8 | UC% | safe yield% | safe accept% | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_ep57 | baseline | 89.2 | 7.54 | 8.41 | 7.81 | 8.26 | 8.29 | 1.4 | 34.2 | 52.4 | 152 |
| span_fusion_ep0058 | span_fusion | 89.2 | 7.48 | 8.37 | 7.79 | 8.2 | 8.11 | 1.4 | 35.2 | 52.0 | 154 |
| span_fusion_ep0059 | span_fusion | 89.2 | 7.46 | 8.36 | 7.77 | 8.23 | 8.14 | 1.4 | 35.4 | 52.6 | 151 |
| span_fusion_ep0060 | span_fusion | 89.2 | 7.46 | 8.36 | 7.8 | 8.22 | 8.13 | 1.4 | 34.8 | 52.0 | 154 |
| span_tail_ep0058 | span_tail | 89.8 | 7.49 | 8.41 | 7.81 | 8.27 | 8.3 | 1.8 | 35.4 | 53.2 | 149 |
| span_tail_ep0059 | span_tail | 89.8 | 7.49 | 8.37 | 7.78 | 8.24 | 8.3 | 1.8 | 34.6 | 52.8 | 152 |
| span_tail_ep0060 | span_tail | 89.8 | 7.49 | 8.38 | 7.77 | 8.23 | 8.25 | 1.6 | 34.8 | 52.8 | 152 |

## Per-arm synthetic selection

- `span_fusion`: `span_fusion_ep0058`; synthetic_guard=True
- `span_tail`: `span_tail_ep0060`; synthetic_guard=True

## Real development and secondary check

### baseline_ep57

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.79 | 19.04 | 13.55 | 18.69 | 17.11 | 23/24.7 | 5.7 | 50.4 | 58.3 | 40.0 | - | 15 |
| handannot17 | 23.5 | 5.21 | 8.24 | 6.24 | 8.75 | 7.52 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

### span_fusion_ep0058

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.78 | 19.02 | 13.57 | 18.68 | 17.07 | 23/24.7 | 5.7 | 50.4 | 58.3 | 40.0 | - | 15 |
| handannot17 | 23.5 | 5.09 | 8.19 | 6.18 | 8.68 | 7.46 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

### span_fusion_ep0059

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.78 | 19.03 | 13.55 | 18.69 | 17.01 | 24/25.8 | 5.7 | 52.0 | 58.3 | 37.8 | - | 14 |
| handannot17 | 23.5 | 5.09 | 8.16 | 6.16 | 8.67 | 7.45 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

### span_fusion_ep0060

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.78 | 19.02 | 13.54 | 18.38 | 17.05 | 24/25.8 | 6.5 | 51.2 | 61.5 | 40.0 | - | 14 |
| handannot17 | 23.5 | 5.08 | 8.13 | 6.13 | 8.65 | 7.42 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

### span_tail_ep0058

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.73 | 18.94 | 13.43 | 18.09 | 17.04 | 25/26.9 | 7.3 | 53.7 | 69.2 | 35.6 | - | 13 |
| handannot17 | 23.5 | 5.13 | 8.05 | 6.1 | 8.64 | 7.44 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

### span_tail_ep0059

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.73 | 18.91 | 13.44 | 18.19 | 17.14 | 24/25.8 | 8.1 | 52.8 | 71.4 | 35.6 | - | 14 |
| handannot17 | 23.5 | 5.13 | 7.97 | 6.09 | 8.58 | 7.38 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

### span_tail_ep0060

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.72 | 18.86 | 13.36 | 18.28 | 17.32 | 23/24.7 | 6.5 | 52.8 | 61.5 | 35.6 | - | 14 |
| handannot17 | 23.5 | 5.14 | 7.95 | 6.1 | 8.59 | 7.38 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

## Decision

- BEST_ARM=`baseline`
- BEST_CKPT=`/home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB/net_epoch_0057.pth`
- reason: baseline retained: no scoped checkpoint met the pre-registered count-preservation and UC-improvement limits
- rank_sums: `{}`
- rejected: `{"span_fusion_ep0058": ["uc_main=23>18"], "span_fusion_ep0059": ["safe_gross=28>27", "uc_main=24>18"], "span_fusion_ep0060": ["uc_main=24>18"], "span_tail_ep0058": ["safe_gross=29>27", "uc_main=25>18"], "span_tail_ep0059": ["safe_gross=29>27", "uc_main=24>18"], "span_tail_ep0060": ["safe_gross=29>27", "uc_main=23>18"]}`

W/D learned prior: disabled; ambiguous candidates are rejected.

handannot17 did not participate in this decision.

Scoped-training preservation policy: `{"max_detection_drop_frames": 2, "max_legacy_pnp_drop_frames": 2, "max_safe_accept_drop_frames": 3, "max_safe_gross_increase_frames": 0, "max_uc_eligible_drop_frames": 5, "min_safe_good_delta_frames": 0, "min_uc_improvement_frames": 5}`
Filterval exact counts: `{"baseline_ep57": {"detection": 94, "legacy_pnp": 94, "safe_accept": 62, "safe_good": 7, "safe_gross": 27, "uc_eligible": 93, "uc_main": 23}, "span_fusion_ep0058": {"detection": 94, "legacy_pnp": 94, "safe_accept": 62, "safe_good": 7, "safe_gross": 27, "uc_eligible": 93, "uc_main": 23}, "span_fusion_ep0059": {"detection": 94, "legacy_pnp": 94, "safe_accept": 64, "safe_good": 7, "safe_gross": 28, "uc_eligible": 93, "uc_main": 24}, "span_fusion_ep0060": {"detection": 94, "legacy_pnp": 94, "safe_accept": 63, "safe_good": 8, "safe_gross": 27, "uc_eligible": 93, "uc_main": 24}, "span_tail_ep0058": {"detection": 94, "legacy_pnp": 94, "safe_accept": 66, "safe_good": 9, "safe_gross": 29, "uc_eligible": 93, "uc_main": 25}, "span_tail_ep0059": {"detection": 94, "legacy_pnp": 94, "safe_accept": 65, "safe_good": 10, "safe_gross": 29, "uc_eligible": 93, "uc_main": 24}, "span_tail_ep0060": {"detection": 94, "legacy_pnp": 94, "safe_accept": 65, "safe_good": 8, "safe_gross": 29, "uc_eligible": 93, "uc_main": 23}}`
