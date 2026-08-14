# paper_s2_rgb1_improve_v1 — quick evaluation

Protocol locks: RGB single frame, input 400, belief 50, init ep57, existing Stage-B data only.
Selection: synthetic + filterval123. handannot17 is report-only. Sealed final-test sessions were not enumerated.

## Synthetic checkpoints

| checkpoint | arm | det% | front | rear | corner | legacy h8 | safe h8 | UC% | safe yield% | safe accept% | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_ep57 | baseline | 89.2 | 7.54 | 8.41 | 7.81 | 8.26 | 8.29 | 1.4 | 34.2 | 52.4 | 152 |
| control_ep0058 | control | 89.6 | 7.69 | 8.55 | 8.02 | 8.39 | 8.13 | 1.8 | 34.8 | 50.6 | 152 |
| control_ep0059 | control | 89.8 | 7.54 | 8.37 | 7.61 | 8.03 | 7.94 | 3.1 | 35.6 | 52.0 | 150 |
| control_ep0060 | control | 88.8 | 7.69 | 8.52 | 7.77 | 8.05 | 8.17 | 2.7 | 34.8 | 51.0 | 160 |
| control_ep0061 | control | 89.8 | 7.99 | 8.92 | 8.07 | 8.58 | 8.6 | 1.6 | 34.0 | 52.8 | 148 |
| control_ep0062 | control | 89.4 | 7.38 | 8.4 | 7.61 | 7.94 | 7.75 | 1.3 | 35.0 | 50.0 | 155 |
| control_ep0063 | control | 88.8 | 7.85 | 8.35 | 7.97 | 8.27 | 8.23 | 0.9 | 35.4 | 52.2 | 147 |
| border_target_ep0058 | border_target | 91.4 | 7.62 | 8.45 | 7.9 | 8.18 | 8.25 | 1.5 | 37.0 | 55.2 | 152 |
| border_target_ep0059 | border_target | 91.6 | 7.86 | 8.54 | 7.76 | 8.55 | 8.54 | 2.6 | 34.8 | 55.6 | 150 |
| border_target_ep0060 | border_target | 92.6 | 8.25 | 8.65 | 8.16 | 8.47 | 8.52 | 1.5 | 36.2 | 53.8 | 154 |
| border_target_ep0061 | border_target | 92.8 | 7.89 | 8.57 | 7.88 | 8.37 | 8.34 | 2.0 | 35.8 | 55.4 | 150 |
| border_target_ep0062 | border_target | 91.0 | 7.87 | 8.53 | 7.99 | 8.42 | 8.12 | 1.6 | 37.2 | 54.8 | 149 |
| border_target_ep0063 | border_target | 90.8 | 7.94 | 8.82 | 8.04 | 8.45 | 8.45 | 1.6 | 36.2 | 55.2 | 151 |
| mask_extent_ep0058 | mask_extent | 90.0 | 7.55 | 8.49 | 7.83 | 8.2 | 8.11 | 1.6 | 35.2 | 51.6 | 156 |
| mask_extent_ep0059 | mask_extent | 90.6 | 7.68 | 8.48 | 7.84 | 8.28 | 8.18 | 1.8 | 35.6 | 51.8 | 153 |
| mask_extent_ep0060 | mask_extent | 89.0 | 7.9 | 8.64 | 8.03 | 8.26 | 8.24 | 1.8 | 34.4 | 51.8 | 152 |
| mask_extent_ep0061 | mask_extent | 89.8 | 7.54 | 8.4 | 7.9 | 8.33 | 8.23 | 1.4 | 34.2 | 51.4 | 154 |
| mask_extent_ep0062 | mask_extent | 88.8 | 7.88 | 8.54 | 7.85 | 8.42 | 8.25 | 1.6 | 34.4 | 50.4 | 154 |
| mask_extent_ep0063 | mask_extent | 88.8 | 7.81 | 8.61 | 7.84 | 8.16 | 8.0 | 1.1 | 36.0 | 50.8 | 156 |
| uncertainty_ep0058 | uncertainty | 88.8 | 7.48 | 8.49 | 7.78 | 8.07 | 7.97 | 1.8 | 31.4 | 46.6 | 179 |
| uncertainty_ep0059 | uncertainty | 89.8 | 7.6 | 8.49 | 7.66 | 8.23 | 8.22 | 2.2 | 33.2 | 50.0 | 165 |
| uncertainty_ep0060 | uncertainty | 88.4 | 7.93 | 8.39 | 7.93 | 8.16 | 8.2 | 2.0 | 32.8 | 48.8 | 166 |
| uncertainty_ep0061 | uncertainty | 90.0 | 7.9 | 8.6 | 7.96 | 8.47 | 8.26 | 2.2 | 35.2 | 51.2 | 156 |
| uncertainty_ep0062 | uncertainty | 88.6 | 7.6 | 8.31 | 7.84 | 8.17 | 8.13 | 1.1 | 31.6 | 49.0 | 161 |
| uncertainty_ep0063 | uncertainty | 89.2 | 7.85 | 8.86 | 8.25 | 8.56 | 8.53 | 1.1 | 33.0 | 48.0 | 160 |
| full_ep0058 | full | 92.2 | 7.99 | 8.64 | 8.17 | 8.49 | 8.54 | 2.4 | 32.8 | 52.0 | 173 |
| full_ep0059 | full | 92.6 | 7.62 | 8.61 | 7.84 | 8.41 | 8.52 | 2.4 | 35.0 | 54.0 | 161 |
| full_ep0060 | full | 91.6 | 7.76 | 8.49 | 7.86 | 8.24 | 8.21 | 1.8 | 34.6 | 51.8 | 161 |
| full_ep0061 | full | 91.4 | 7.68 | 8.53 | 8.03 | 8.39 | 8.48 | 1.3 | 34.4 | 51.8 | 167 |
| full_ep0062 | full | 90.6 | 7.93 | 8.76 | 7.96 | 8.39 | 8.31 | 1.6 | 33.8 | 52.0 | 158 |
| full_ep0063 | full | 91.0 | 8.1 | 9.01 | 8.24 | 8.65 | 8.58 | 1.3 | 34.8 | 51.4 | 168 |

## Per-arm synthetic selection

- `border_target`: `border_target_ep0058`; synthetic_guard=True
- `control`: `control_ep0063`; synthetic_guard=True
- `full`: `full_ep0060`; synthetic_guard=True
- `mask_extent`: `mask_extent_ep0063`; synthetic_guard=True
- `uncertainty`: `uncertainty_ep0059`; synthetic_guard=True

## Real development and secondary check

### baseline_ep57

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.4 | 8.79 | 19.04 | 13.55 | 18.69 | 17.11 | 23/24.7 | 5.7 | 50.4 | 58.3 | 40.0 | - | 15 |
| handannot17 | 23.5 | 5.21 | 8.24 | 6.24 | 8.75 | 7.52 | 1/25.0 | 11.8 | 11.8 | 66.7 | 100.0 | - | 0 |

### border_target_ep0058

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 67.5 | 9.46 | 17.77 | 14.11 | 19.23 | 18.54 | 19/22.9 | 5.7 | 52.0 | 58.3 | 25.6 | - | 12 |
| handannot17 | 41.2 | 7.0 | 7.86 | 7.0 | 10.75 | 7.1 | 1/14.3 | 17.6 | 23.5 | 100.0 | 100.0 | - | 0 |

### control_ep0063

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 67.5 | 8.75 | 15.95 | 11.8 | 18.35 | 16.41 | 15/18.3 | 8.9 | 45.5 | 73.3 | 42.1 | - | 12 |
| handannot17 | 23.5 | 6.23 | 9.46 | 7.28 | 16.85 | 7.68 | 0/0.0 | 11.8 | 11.8 | 100.0 | 100.0 | - | 0 |

### full_ep0060

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 70.7 | 9.24 | 19.39 | 11.7 | 17.85 | 14.74 | 18/20.7 | 8.1 | 48.8 | 66.7 | 41.0 | 0.372 | 12 |
| handannot17 | 47.1 | 7.06 | 9.31 | 7.2 | 9.3 | 7.74 | 0/0.0 | 17.6 | 23.5 | 75.0 | - | 0.309 | 0 |

### mask_extent_ep0063

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 71.5 | 10.22 | 17.59 | 12.78 | 18.5 | 15.01 | 10/11.4 | 8.9 | 43.1 | 91.7 | 59.5 | - | 16 |
| handannot17 | 17.6 | 7.65 | 8.26 | 7.65 | 8.31 | 8.36 | 0/0.0 | 11.8 | 11.8 | 66.7 | - | - | 0 |

### uncertainty_ep0059

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 74.8 | 9.46 | 20.97 | 14.3 | 22.6 | 19.68 | 21/23.1 | 3.3 | 47.2 | 44.4 | 45.1 | 0.393 | 19 |
| handannot17 | 29.4 | 7.47 | 8.92 | 7.47 | 11.7 | 7.62 | 1/20.0 | 11.8 | 11.8 | 100.0 | 100.0 | 0.686 | 0 |

## Decision

- BEST_ARM=`baseline`
- BEST_CKPT=`/home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB/net_epoch_0057.pth`
- reason: baseline retained after synthetic+filterval rank comparison
- rank_sums: `{"baseline_ep57": 0}`
- rejected: `{"border_target": ["filterval_guard"], "control": ["filterval_guard"], "full": ["filterval_guard"], "mask_extent": ["filterval_guard"], "uncertainty": ["filterval_guard"]}`

W/D learned prior: disabled; ambiguous candidates are rejected.

handannot17 did not participate in this decision.
