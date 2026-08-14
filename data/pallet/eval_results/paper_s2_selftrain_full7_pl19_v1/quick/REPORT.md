# paper_s2_selftrain_full7_pl19_v1 — quick evaluation

Protocol locks: RGB single frame, input 400, belief 50, init ep57, configured training provenance verified.
Selection: synthetic + filterval105. handannot17 (n=16) is report-only. Sealed final-test sessions were not enumerated.
Real membership: canonical n=140, evaluated n=121, training exclusions=19.

## Synthetic checkpoints

| checkpoint | arm | det% | front | rear | corner | legacy h8 | safe h8 | UC% | safe yield% | safe accept% | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_ep57 | baseline | 89.2 | 7.54 | 8.41 | 7.81 | 8.26 | 8.29 | 1.4 | 34.2 | 52.2 | 152 |
| filtered_st_ep0058 | filtered_st | 89.6 | 7.86 | 8.72 | 8.28 | 8.7 | 8.57 | 2.3 | 32.0 | 52.2 | 155 |
| filtered_st_ep0059 | filtered_st | 89.2 | 8.0 | 8.74 | 8.31 | 8.63 | 8.63 | 1.6 | 33.0 | 51.6 | 152 |
| filtered_st_ep0060 | filtered_st | 89.2 | 7.94 | 8.88 | 8.24 | 8.7 | 8.71 | 1.8 | 33.2 | 53.8 | 147 |

## Per-arm synthetic selection

- `filtered_st`: `filtered_st_ep0058`; synthetic_guard=True

## Real development and secondary check

### baseline_ep57

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 72.4 | 8.79 | 19.85 | 13.5 | 18.69 | 16.5 | 20/26.7 | 2.9 | 40.0 | 37.5 | 58.3 | - | 13 |
| handannot17 | 18.8 | 3.2 | 8.58 | 5.25 | 9.74 | 7.76 | 1/33.3 | 6.2 | 6.2 | 50.0 | 100.0 | - | 0 |

### filtered_st_ep0058

| set | det% | front | rear | corner | legacy h8 | safe h8 | UC n/% | safe yield% | safe accept% | good retention% | bad reject% | sigma/error rho | WD ambiguous rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filterval | 76.2 | 10.57 | 22.3 | 15.31 | 21.29 | 18.67 | 20/25.3 | 1.0 | 47.6 | 25.0 | 48.8 | - | 9 |
| handannot17 | 18.8 | 3.52 | 8.89 | 4.71 | 10.26 | 8.19 | 0/0.0 | 6.2 | 6.2 | 100.0 | 100.0 | - | 0 |

## Decision

- BEST_ARM=`baseline`
- BEST_CKPT=`/home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB/net_epoch_0057.pth`
- reason: baseline retained after synthetic+filterval rank comparison
- rank_sums: `{"baseline_ep57": 0}`
- rejected: `{"filtered_st": ["filterval_guard"]}`

W/D learned prior: disabled; ambiguous candidates are rejected.

handannot17 did not participate in this decision.
