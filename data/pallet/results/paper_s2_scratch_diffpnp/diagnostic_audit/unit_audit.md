# PAPER_S2 geometry unit audit

- generated: `2026-07-28T07:07:00.308947+00:00`
- commit: `0baa6dfc2ba850dd498f59b74e42663828d166c7`
- checkpoint: `/home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB/net_epoch_0057.pth`

## Gate summary

| gate | result |
|---|---:|
| Global soft-argmax known Gaussian | PASS |
| PAPER_S2 local soft-argmax/covariance | PASS |
| Legacy global soft-argmax on DOPE value maps (T=1.0) | FAIL |
| PAPER_S2 local soft-argmax on DOPE value maps (T=0.1) | PASS |
| Legacy BPnP oracle | PASS |
| Legacy BPnP finite difference | FAIL (5.429e-01) |
| PAPER_S2 DiffPnP3D oracle | PASS |
| PAPER_S2 DiffPnP3D finite difference | PASS (1.014e-09) |
| PAPER_S2 NaN guard backward | PASS |

The legacy BPnP result and PAPER_S2 DiffPnP3D result are intentionally
reported separately: the canonical ep57 checkpoint used the latter.
Legacy operational failures: `['fractional_center', 'low_peak_background', 'negative_raw_output', 'rotated_45deg', 'single_gaussian_center', 'x_elongated', 'y_elongated']`.
PAPER_S2 local operational failures: `[]`.
