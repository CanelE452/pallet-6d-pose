# paper_s2_rgb1_frozen_fusion_v1 — quick evaluation

Protocol locks: RGB single frame, input 400, belief 50, init ep57, existing Stage-B data only.
Selection: synthetic + filterval123. handannot17 is report-only. Sealed final-test sessions were not enumerated.

## Synthetic checkpoints

| checkpoint | arm | det% | front | rear | corner | legacy h8 | safe h8 | UC% | safe yield% | safe accept% | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_ep57 | baseline | 89.2 | 7.54 | 8.41 | 7.81 | 8.26 | 8.29 | 1.4 | 34.2 | 52.4 | 152 |
| mask_extent_frozen_ep0058 | mask_extent_frozen | 89.2 | 7.48 | 8.39 | 7.77 | 8.21 | 8.14 | 1.4 | 35.0 | 52.4 | 152 |
| mask_extent_frozen_ep0059 | mask_extent_frozen | 89.2 | 7.5 | 8.4 | 7.78 | 8.24 | 8.24 | 1.4 | 35.2 | 53.0 | 149 |
| mask_extent_frozen_ep0060 | mask_extent_frozen | 89.2 | 7.51 | 8.42 | 7.78 | 8.23 | 8.15 | 1.4 | 34.8 | 52.6 | 150 |

## Per-arm synthetic selection

- `mask_extent_frozen`: `mask_extent_frozen_ep0058`; synthetic_guard=True

## Real development and secondary check

### baseline_ep57

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.79 | 19.04 | 13.55 | 18.69 | 17.11 | 23/24.7 | 5.7 | 50.4 | 58.3 | 40.0 | - | 15 |
| handannot17 | 23.5 | 5.21 | 8.24 | 6.24 | 8.75 | 7.52 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

### mask_extent_frozen_ep0058

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.81 | 19.02 | 13.52 | 18.68 | 17.14 | 23/24.7 | 5.7 | 51.2 | 58.3 | 37.8 | - | 15 |
| handannot17 | 23.5 | 5.29 | 8.19 | 6.28 | 8.77 | 7.57 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

### mask_extent_frozen_ep0059

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.78 | 19.01 | 13.51 | 18.69 | 17.17 | 25/26.9 | 6.5 | 52.0 | 61.5 | 37.8 | - | 14 |
| handannot17 | 23.5 | 5.27 | 8.16 | 6.26 | 8.76 | 7.54 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

### mask_extent_frozen_ep0060

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.77 | 19.01 | 13.5 | 18.71 | 17.18 | 24/25.8 | 6.5 | 52.0 | 61.5 | 37.8 | - | 14 |
| handannot17 | 23.5 | 5.29 | 8.16 | 6.26 | 8.75 | 7.54 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

## Decision

- BEST_ARM=`baseline`
- BEST_CKPT=`/home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB/net_epoch_0057.pth`
- reason: baseline retained: no frozen-fusion checkpoint met the pre-registered count-preservation and UC-improvement limits
- rank_sums: `{}`
- rejected: `{"mask_extent_frozen_ep0058": ["safe_gross=28>27", "uc_main=23>18"], "mask_extent_frozen_ep0059": ["safe_gross=28>27", "uc_main=25>18"], "mask_extent_frozen_ep0060": ["safe_gross=28>27", "uc_main=24>18"]}`

W/D learned prior: disabled; ambiguous candidates are rejected.

handannot17 did not participate in this decision.

Frozen-fusion preservation policy: `{"max_detection_drop_frames": 2, "max_legacy_pnp_drop_frames": 2, "max_safe_accept_drop_frames": 3, "max_safe_gross_increase_frames": 0, "max_uc_eligible_drop_frames": 5, "min_safe_good_delta_frames": 0, "min_uc_improvement_frames": 5}`
Filterval exact counts: `{"baseline_ep57": {"detection": 94, "legacy_pnp": 94, "safe_accept": 62, "safe_good": 7, "safe_gross": 27, "uc_eligible": 93, "uc_main": 23}, "mask_extent_frozen_ep0058": {"detection": 94, "legacy_pnp": 94, "safe_accept": 63, "safe_good": 7, "safe_gross": 28, "uc_eligible": 93, "uc_main": 23}, "mask_extent_frozen_ep0059": {"detection": 94, "legacy_pnp": 94, "safe_accept": 64, "safe_good": 8, "safe_gross": 28, "uc_eligible": 93, "uc_main": 25}, "mask_extent_frozen_ep0060": {"detection": 94, "legacy_pnp": 94, "safe_accept": 64, "safe_good": 8, "safe_gross": 28, "uc_eligible": 93, "uc_main": 24}}`
