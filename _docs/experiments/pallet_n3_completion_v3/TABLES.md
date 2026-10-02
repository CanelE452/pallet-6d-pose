# N3 completion tables

`x` = missing/blocked/unmeasured; `NA` = protocol-defined not applicable; numeric `0` = measured zero.

### Reused DEV319: base to N3

| Backbone | Method | Seeds | 2D median (px) | 2D P90 (px) | PCK10 (fraction) | E_sym | T median (cm) | R median (deg) | Yaw median (deg) | Pose coverage | PCK5 (fraction) | PCK20 (fraction) | Penalty P90 (px) | T P90 (cm) | R P90 (deg) | Yaw P90 (deg) | IoU3D median | ADDsym AUC | Frames | Matched frames | GT corners | Observed corners | Pose frames |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO | Base | 1 | 6.721 | 43.89 | 0.6343 | 0.04952 | 7.897 | 2.539 | 1.316 | 1 | 0.3629 | 0.8007 | 61.71 | 40.53 | 86.53 | 86.24 | 0.5943 | 0.3766 | 319 | 311 | 2499 | 2445 | 319 |
| YOLO | N3 seed mean | 3 | 5.778 | 42.13 | 0.6859 | 0.04842 | 7.068 | 2.07 | 1.134 | 1 | 0.4287 | 0.8243 | 61.72 | 37.81 | 85.92 | 85.67 | 0.6309 | 0.4122 | 319 | 311 | 2499 | 2445 | 319 |
| DOPE | Base | 1 | 12.57 | 51.28 | 0.2417 | 0.3087 | 10.05 | 3.53 | 2.281 | 0.6583 | 0.05202 | 0.537 | 800 | 68.34 | 80.58 | 80.37 | 0.433 | 0.1818 | 319 | 233 | 2499 | 1797 | 210 |
| DOPE | N3 seed mean | 3 | 7.469 | 53.57 | 0.4434 | 0.3055 | 8.356 | 3.051 | 1.798 | 0.6583 | 0.2264 | 0.5805 | 800 | 63.6 | 82.17 | 82.01 | 0.5728 | 0.2332 | 319 | 233 | 2499 | 1797 | 210 |
| ResNet-18 | Base | 1 | 8.223 | 62.64 | 0.5226 | 0.1119 | 9.739 | 4.342 | 2.031 | 1 | 0.2809 | 0.7035 | 248.4 | 97.54 | 87.41 | 87.08 | 0.5243 | 0.334 | 319 | 292 | 2499 | 2291 | 319 |
| ResNet-18 | N3 seed mean | 3 | 7.085 | 62.37 | 0.5722 | 0.111 | 9.134 | 3.832 | 1.874 | 1 | 0.3415 | 0.7159 | 249.6 | 100.3 | 87.47 | 87.22 | 0.5511 | 0.357 | 319 | 292 | 2499 | 2291 | 319 |

_Note: N3 is the arithmetic mean of seed-level statistics, not an ensemble. Absolute backbone ranking is not valid because conditional match sets differ._

### DOPE / ResNet-18 DEV319 per-seed results

| Backbone | Method | Seed | 2D median (px) | 2D P90 (px) | PCK10 (fraction) | E_sym | T median (cm) | R median (deg) | Yaw median (deg) | Pose coverage | PCK5 (fraction) | PCK20 (fraction) | Penalty P90 (px) | T P90 (cm) | R P90 (deg) | Yaw P90 (deg) | IoU3D median | ADDsym AUC | Frames | Matched frames | GT corners | Observed corners | Pose frames |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DOPE | Base | base | 12.57 | 51.28 | 0.2417 | 0.3087 | 10.05 | 3.53 | 2.281 | 0.6583 | 0.05202 | 0.537 | 800 | 68.34 | 80.58 | 80.37 | 0.433 | 0.1818 | 319 | 233 | 2499 | 1797 | 210 |
| DOPE | N3 | 1 | 7.57 | 54.68 | 0.4414 | 0.3055 | 8.447 | 3.136 | 1.815 | 0.6583 | 0.2205 | 0.5794 | 800 | 64.52 | 82.66 | 82.57 | 0.5793 | 0.2327 | 319 | 233 | 2499 | 1797 | 210 |
| DOPE | N3 | 2 | 7.43 | 53.13 | 0.4442 | 0.3055 | 8.273 | 3.091 | 1.753 | 0.6583 | 0.2289 | 0.5798 | 800 | 62.51 | 82.7 | 82.6 | 0.5763 | 0.2335 | 319 | 233 | 2499 | 1797 | 210 |
| DOPE | N3 | 3 | 7.408 | 52.9 | 0.4446 | 0.3054 | 8.348 | 2.925 | 1.826 | 0.6583 | 0.2297 | 0.5822 | 800 | 63.77 | 81.16 | 80.86 | 0.5627 | 0.2335 | 319 | 233 | 2499 | 1797 | 210 |
| ResNet-18 | Base | base | 8.223 | 62.64 | 0.5226 | 0.1119 | 9.739 | 4.342 | 2.031 | 1 | 0.2809 | 0.7035 | 248.4 | 97.54 | 87.41 | 87.08 | 0.5243 | 0.334 | 319 | 292 | 2499 | 2291 | 319 |
| ResNet-18 | N3 | 1 | 7.091 | 62.57 | 0.5714 | 0.1111 | 9.069 | 3.807 | 1.847 | 1 | 0.3385 | 0.7143 | 248.5 | 99.64 | 87.52 | 87.2 | 0.5463 | 0.3545 | 319 | 292 | 2499 | 2291 | 319 |
| ResNet-18 | N3 | 2 | 7.15 | 62.14 | 0.5702 | 0.1111 | 9.21 | 3.905 | 1.943 | 1 | 0.3401 | 0.7147 | 250.4 | 101 | 87.5 | 87.25 | 0.546 | 0.3542 | 319 | 292 | 2499 | 2291 | 319 |
| ResNet-18 | N3 | 3 | 7.013 | 62.4 | 0.575 | 0.1109 | 9.124 | 3.784 | 1.833 | 1 | 0.3457 | 0.7187 | 250 | 100.3 | 87.39 | 87.21 | 0.561 | 0.3625 | 319 | 292 | 2499 | 2291 | 319 |

_Note: Base is listed once per backbone; N3 seeds are independent fits, never an ensemble._

### DOPE / ResNet-18 paired session-bootstrap deltas (N3 - Base)

| Backbone | Seed | Sessions | Resamples | Bootstrap seed | Median delta (px) | Median delta CI95 | P90 delta (px) | P90 delta CI95 | PCK10 delta (pp) | PCK10 delta CI95 | E_sym delta | E_sym delta CI95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DOPE | 1 | 13 | 10000 | 20260917 | -5.001 | [-5.757671329468234, -4.182964723586372] | 3.396 | [-4.919370293025964, 3.8796703969038373] | 19.97 | [14.713382986773704, 24.92485275951693] | -0.003172 | [-0.003999700017957903, -0.0022883241894971255] |
| DOPE | 2 | 13 | 10000 | 20260917 | -5.141 | [-5.851155357795893, -4.323854280435686] | 1.853 | [-4.895927055560795, 3.5289968840256365] | 20.25 | [14.88514896940402, 25.176865951844697] | -0.003259 | [-0.004093931310762836, -0.0023872803366903495] |
| DOPE | 3 | 13 | 10000 | 20260917 | -5.162 | [-5.880822706213129, -4.379267022178284] | 1.617 | [-5.100279963685492, 3.176501274775134] | 20.29 | [14.944922547332187, 25.220478838533516] | -0.00327 | [-0.004079162665569764, -0.002424515297666165] |
| ResNet-18 | 1 | 13 | 10000 | 20260917 | -1.132 | [-1.9303893974149928, -0.8588222685627179] | -0.06386 | [-1.5274039726509472, 2.4060589827604844] | 4.882 | [2.9464872648886695, 7.27969889208895] | -0.000766 | [-0.0010950696562846188, -0.0005126815539911824] |
| ResNet-18 | 2 | 13 | 10000 | 20260917 | -1.073 | [-1.8352235134497914, -0.8310054589689057] | -0.4976 | [-1.6673465275657813, 2.142042623892928] | 4.762 | [3.207940413839374, 6.656945699180095] | -0.0007757 | [-0.0010690162354426093, -0.0005277469758786492] |
| ResNet-18 | 3 | 13 | 10000 | 20260917 | -1.209 | [-2.0872646173880507, -0.9848026681010191] | -0.2413 | [-2.0427715455314446, 2.0086973974254896] | 5.242 | [3.406744053935282, 7.396158055615054] | -0.0009307 | [-0.0012809502195610654, -0.0006541894909101072] |

_Note: 10,000 whole-session paired resamples, seed 20260917. Negative is favorable except PCK10, where positive is favorable; no multiplicity adjustment._

### Actual whole-object symmetry target activation

| Backbone | Usable source rows | Non-identity rows | Non-identity fraction | Seed1 exposures | Seed1 non-identity | Seed2 exposures | Seed2 non-identity | Seed3 exposures | Seed3 non-identity |
|---|---|---|---|---|---|---|---|---|---|
| DOPE | 44063 | 134 | 0.003041 | 96000 | 290 | 96000 | 290 | 96000 | 289 |
| ResNet-18 | 55806 | 0 | 0 | 96000 | 0 | 96000 | 0 | 96000 | 0 |

_Note: The objective is wired for both backbones, but target-branch activation is empirical. Zero means measured zero, not x._

### Trained N3 dimension-path sensitivity with fixed visual evidence

| Backbone | Seed | Audit | Visual inputs identical | Base logits identical | Changed logits | Max absolute delta | Checkpoint SHA-256 |
|---|---|---|---|---|---|---|---|
| DOPE | 1 | PASS | true | true | 1776 | 6.182 | e856aa58ab6dd15d… |
| DOPE | 2 | PASS | true | true | 1776 | 9.929 | df4e4d98cc98fe80… |
| DOPE | 3 | PASS | true | true | 1776 | 12.85 | 8b407a0161167033… |
| ResNet-18 | 1 | PASS | true | true | 1776 | 8.358 | 114fc574943868a5… |
| ResNet-18 | 2 | PASS | true | true | 1776 | 7.872 | 23359f6550f1cc4a… |
| ResNet-18 | 3 | PASS | true | true | 1776 | 5.559 | e2ad85b9397c5beb… |

_Note: Only registered W,D,H changes. This proves the trained metadata path is active, not the causal accuracy gain of dimensions._

### YOLO controlled N0/N1/N2/N3 ablation

| Method | Dimensions | Symmetry supervision | 2D median (px) | 2D P90 (px) | PCK10 (fraction) | E_sym | T median (cm) | R median (deg) | Yaw median (deg) | Pose coverage | PCK5 (fraction) | PCK20 (fraction) | Penalty P90 (px) | T P90 (cm) | R P90 (deg) | Yaw P90 (deg) | IoU3D median | ADDsym AUC | Frames | Matched frames | GT corners | Observed corners | Pose frames |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N0: replay local refiner | false | false | 5.944 | 42.87 | 0.6753 | 0.04868 | x | x | x | x | 0.4162 | 0.8178 | 61.71 | x | x | x | x | x | 319 | 311 | 2499 | 2445 | x |
| N1: symmetry only | false | true | 5.921 | 42.51 | 0.6756 | 0.04867 | x | x | x | x | 0.4182 | 0.8157 | 61.6 | x | x | x | x | x | 319 | 311 | 2499 | 2445 | x |
| N2: dimensions only | true | false | 5.778 | 42.46 | 0.6859 | 0.04842 | 7.011 | 2.09 | 1.142 | 1 | 0.4299 | 0.8238 | 61.91 | 38.11 | 85.82 | 85.61 | 0.6335 | 0.4126 | 319 | 311 | 2499 | 2445 | 319 |
| N3: dimensions + symmetry | true | true | 5.778 | 42.13 | 0.6859 | 0.04842 | 7.068 | 2.07 | 1.134 | 1 | 0.4287 | 0.8243 | 61.72 | 37.81 | 85.92 | 85.67 | 0.6309 | 0.4122 | 319 | 311 | 2499 | 2445 | 319 |

_Note: N0/N1/N2/N3 use the locked eight-corner evaluator. Differences must be read metric by metric; tiny decimal changes are not a universal gain._

### YOLO controlled ablation contrasts

| Contrast | Role | Median delta (px) | P90 delta (px) | PCK10 delta (pp) | E_sym delta |
|---|---|---|---|---|---|
| N2 - N0 | dimension contribution under fixed supervision | -0.1662 | -0.4128 | 1.054 | -0.0002611 |
| N1 - N0 | symmetry-only contribution | -0.02285 | -0.3589 | 0.02668 | -9.408e-06 |
| N3 - N2 | increment from symmetry with dimensions | 0.0004884 | -0.3257 | 0 | -2.794e-06 |
| N3 - N1 | increment from dimensions with symmetry | -0.1429 | -0.3797 | 1.027 | -0.0002545 |

_Note: Each value is the first named method minus the second. Negative is favorable for error metrics and positive is favorable for PCK10._

### YOLO cap, damage, and recovery audit (seed-statistic mean)

| Method | Output cap | 2D median (px) | 2D P90 (px) | PCK10 | Initial inside cap | Initial outside cap | Cap hits | Corners improved | Corners unchanged | Corners worsened | Frames improved | Frames unchanged | Frames worsened | <5 to >10 px | >20 to <10 px | Move median (px) | Move P90 (px) | Cap violations | Bound violations | Comparable frames | Comparable corners |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N2 | 1% | 5.778 | 42.46 | 0.6859 | 1464 | 981 | 67.67 | 1660 | 0 | 785.3 | 268.3 | 0 | 42.67 | 0 | 0.3333 | 1.839 | 5.873 | 0 | 0 | 311 | 2445 |
| N2 | 2% | 5.76 | 42.71 | 0.6892 | 1957 | 488 | 1 | 1659 | 0 | 786.3 | 267.7 | 0 | 43.33 | 0 | 3.667 | 1.839 | 5.873 | 0 | 0 | 311 | 2445 |
| N2 | none | 5.76 | 42.71 | 0.6892 | NA | NA | NA | 1659 | 0 | 786.3 | 267.7 | 0 | 43.33 | 0 | 3.667 | 1.839 | 5.873 | 0 | 0 | 311 | 2445 |
| N3 | 1% | 5.778 | 42.13 | 0.6859 | 1464 | 981 | 64 | 1653 | 0 | 792 | 265.3 | 0 | 45.67 | 0.3333 | 0.6667 | 1.817 | 5.871 | 0 | 0 | 311 | 2445 |
| N3 | 2% | 5.762 | 42.62 | 0.6885 | 1957 | 488 | 2 | 1652 | 0 | 793 | 264.7 | 0 | 46.33 | 0.3333 | 3 | 1.817 | 5.871 | 0 | 0 | 311 | 2445 |
| N3 | none | 5.762 | 42.62 | 0.6885 | NA | NA | NA | 1652 | 0 | 793 | 264.7 | 0 | 46.33 | 0.3333 | 3 | 1.817 | 5.871 | 0 | 0 | 311 | 2445 |

_Note: Counts are arithmetic means of three seed-level counts and may therefore be fractional. The fixed base symmetry branch is retained for damage/recovery diagnosis._

### YOLO paired 2D-to-pose direction audit at the locked 1% cap

| Method | Seed | Paired frames | 2D improved | 2D worsened | T median delta (cm) | T improved | T unchanged | T worsened | R median delta (deg) | R improved | R unchanged | R worsened | Yaw median delta (deg) | Yaw improved | Yaw unchanged | Yaw worsened | 2D+ / T+ | 2D+ / T- | 2D- / T+ | 2D- / T- | New pose failures | Pose recoveries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N2 | 1 | 311 | 274 | 37 | -0.2806 | 191 | 0 | 120 | -0.1409 | 202 | 0 | 109 | -0.02705 | 167 | 0 | 144 | 171 | 103 | 20 | 17 | 0 | 0 |
| N2 | 2 | 311 | 261 | 50 | -0.3756 | 203 | 0 | 108 | -0.1561 | 208 | 0 | 103 | -0.04935 | 172 | 0 | 139 | 182 | 79 | 21 | 29 | 0 | 0 |
| N2 | 3 | 311 | 270 | 41 | -0.2293 | 181 | 0 | 130 | -0.1854 | 212 | 0 | 99 | -0.05257 | 174 | 0 | 137 | 165 | 105 | 16 | 25 | 0 | 0 |
| N3 | 1 | 311 | 273 | 38 | -0.2751 | 195 | 0 | 116 | -0.157 | 205 | 0 | 106 | -0.03634 | 167 | 0 | 144 | 175 | 98 | 20 | 18 | 0 | 0 |
| N3 | 2 | 311 | 258 | 53 | -0.3694 | 200 | 0 | 111 | -0.1565 | 209 | 0 | 102 | -0.02512 | 171 | 0 | 140 | 180 | 78 | 20 | 33 | 0 | 0 |
| N3 | 3 | 311 | 265 | 46 | -0.2722 | 186 | 0 | 125 | -0.1656 | 209 | 0 | 102 | -0.05627 | 170 | 0 | 141 | 165 | 100 | 21 | 25 | 0 | 0 |

_Note: Deltas are candidate minus base; negative T/R/yaw deltas are favorable. These are paired descriptive counts, separate from session-bootstrap corner CIs._

### YOLO whole-package comparator audit

| Method | Role | Seeds | 2D median (px) | 2D P90 (px) | PCK10 (fraction) | E_sym | T median (cm) | R median (deg) | Yaw median (deg) | Pose coverage | PCK5 (fraction) | PCK20 (fraction) | Penalty P90 (px) | T P90 (cm) | R P90 (deg) | Yaw P90 (deg) | IoU3D median | ADDsym AUC | Frames | Matched frames | GT corners | Observed corners | Pose frames |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0: no refiner | initial estimate | 1 | 6.721 | 43.89 | 0.6343 | 0.04952 | 7.897 | 2.539 | 1.316 | 1 | 0.3629 | 0.8007 | 61.71 | 40.53 | 86.53 | 86.24 | 0.5943 | 0.3766 | 319 | 311 | 2499 | 2445 | 319 |
| P: local distribution | dimension-free refiner | 3 | 5.938 | 42.63 | 0.6749 | 0.04868 | 7.153 | 2.154 | 1.151 | 1 | 0.4175 | 0.8181 | 61.64 | 38.66 | 85.99 | 85.82 | 0.6367 | 0.408 | 319 | 311 | 2499 | 2445 | 319 |
| D: direct regression | output/loss package | 3 | 6.504 | 42.99 | 0.6437 | 0.04921 | 7.597 | 2.274 | 1.206 | 1 | 0.3828 | 0.8111 | 61.62 | 41.36 | 86.05 | 85.78 | 0.6168 | 0.392 | 319 | 311 | 2499 | 2445 | 319 |
| L: line structure | structure package | 3 | 6.146 | 42.98 | 0.6687 | 0.04887 | 7.548 | 2.233 | 1.175 | 1 | 0.4071 | 0.8139 | 60.55 | 38.38 | 86.23 | 85.86 | 0.6282 | 0.402 | 319 | 311 | 2499 | 2445 | 319 |
| PoseFix-style | image-pose refiner package | 3 | 5.561 | 43.91 | 0.6871 | 0.0483 | 6.951 | 2.028 | 1.047 | 1 | 0.4452 | 0.8193 | 61.13 | 41.68 | 85.99 | 85.6 | 0.6232 | 0.4214 | 319 | 311 | 2499 | 2445 | 319 |
| N3: dimensions + symmetry | proposed package | 3 | 5.778 | 42.13 | 0.6859 | 0.04842 | 7.068 | 2.07 | 1.134 | 1 | 0.4287 | 0.8243 | 61.72 | 37.81 | 85.92 | 85.67 | 0.6309 | 0.4122 | 319 | 311 | 2499 | 2445 | 319 |

_Note: D/L/PoseFix and N3 differ in inputs, output parameterization, loss, or training budget. This is a whole-package comparison, not a single-factor causal ablation._

### Update alternatives: safe HELDOUT128 and blocked DEV319 rows

| Population | Method | Evidence role | Student RGB overlap | Teacher session overlap | Independent TEST | Selection history | 2D median (px) | 2D P90 (px) | PCK10 (fraction) | E_sym | T median (cm) | R median (deg) | Yaw median (deg) | Pose coverage | PCK5 (fraction) | PCK20 (fraction) | Penalty P90 (px) | T P90 (cm) | R P90 (deg) | Yaw P90 (deg) | IoU3D median | ADDsym AUC | Frames | Matched frames | GT corners | Observed corners | Pose frames |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| predeclared HELDOUT128 plastic; 7 recording groups; reused DEV, not independent test | R0 | reused development safe cohort; not independent TEST | 0 | 0 | false | historically selected on reused plastic194 DEV | 9.565 | 40.9 | 0.4914 | 0.08598 | 9.353 | 3.53 | 2.756 | 1 | 0.2081 | 0.7269 | 70.63 | 79.12 | 88.73 | 88.53 | 0.5865 | 0.338 | 128 | 120 | 985 | 931 | 128 |
| predeclared HELDOUT128 plastic; 7 recording groups; reused DEV, not independent test | Synthetic-only update | reused development safe cohort; not independent TEST | 0 | 0 | false | historically selected on reused plastic194 DEV | 9.741 | 41.78 | 0.4863 | 0.08597 | 9.203 | 3.507 | 2.59 | 1 | 0.2061 | 0.7279 | 70.29 | 78.61 | 88.68 | 88.34 | 0.5948 | 0.3389 | 128 | 120 | 985 | 931 | 128 |
| predeclared HELDOUT128 plastic; 7 recording groups; reused DEV, not independent test | Raw pseudo-label student | reused development safe cohort; not independent TEST | 0 | 0 | false | historically selected on reused plastic194 DEV | 9.961 | 41.96 | 0.4751 | 0.08617 | 9.226 | 3.942 | 2.626 | 1 | 0.203 | 0.7259 | 70.14 | 77.88 | 89.05 | 88.6 | 0.5808 | 0.3347 | 128 | 120 | 985 | 931 | 128 |
| predeclared HELDOUT128 plastic; 7 recording groups; reused DEV, not independent test | Corrected pseudo-label student | reused development safe cohort; not independent TEST | 0 | 0 | false | historically selected on reused plastic194 DEV | 8.954 | 42.09 | 0.5147 | 0.0854 | 9.186 | 3.766 | 2.55 | 1 | 0.2467 | 0.7391 | 70.36 | 78.94 | 88.81 | 88.66 | 0.5928 | 0.359 | 128 | 120 | 985 | 931 | 128 |
| predeclared HELDOUT128 plastic; 7 recording groups; reused DEV, not independent test | R0 + N3 seed mean | reused development safe cohort; not independent TEST | 0 | 0 | false | historically selected on reused plastic194 DEV | 8.29 | 40.09 | 0.5624 | 0.08444 | 8.117 | 2.9 | 2.266 | 1 | 0.2653 | 0.7641 | 69.95 | 73.29 | 88.08 | 87.95 | 0.6111 | 0.367 | 128 | 120 | 985 | 931 | 128 |
| DEV319 | Synthetic-only update | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| DEV319 | Raw pseudo-label student | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| DEV319 | Corrected pseudo-label student | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |

_Note: HELDOUT128 is a predeclared exposure-safe reused-development cohort. DEV319 update cells stay x because a common exposure contract is unavailable._

### GREEN0918_119 square audit (manual declared)

| Backbone | Method | Seeds | 2D median (px) | 2D P90 (px) | PCK10 (fraction) | E_sym | T median (cm) | R median (deg) | Yaw median (deg) | Pose coverage | PCK5 (fraction) | PCK20 (fraction) | Penalty P90 (px) | T P90 (cm) | R P90 (deg) | Yaw P90 (deg) | IoU3D median | ADDsym AUC | Frames | Matched frames | GT corners | Observed corners | Pose frames |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO | Base | 1 | 5.526 | 11.44 | 0.8522 | 0.01654 | x | x | x | x | 0.4219 | 0.9668 | 11.63 | x | x | x | x | x | 119 | 118 | 602 | 597 | x |
| YOLO | OLD_P seed mean | 3 | 4.991 | 10.09 | 0.8904 | 0.01565 | x | x | x | x | 0.4956 | 0.9729 | 10.33 | x | x | x | x | x | 119 | 118 | 602 | 597 | x |
| YOLO | N2 seed mean | 3 | 4.949 | 9.882 | 0.8959 | 0.01559 | x | x | x | x | 0.5011 | 0.9729 | 10.14 | x | x | x | x | x | 119 | 118 | 602 | 597 | x |
| YOLO | N3 seed mean | 3 | 5.004 | 9.932 | 0.8926 | 0.01562 | x | x | x | x | 0.4928 | 0.974 | 10.33 | x | x | x | x | x | 119 | 118 | 602 | 597 | x |
| DOPE | Base | 1 | 11.75 | 30.23 | 0.3671 | 0.1097 | x | x | x | x | 0.0897 | 0.7508 | 85.52 | x | x | x | x | x | 119 | 109 | 602 | 549 | x |
| DOPE | N3 seed mean | 3 | 7.037 | 28.24 | 0.6462 | 0.1052 | x | x | x | x | 0.2763 | 0.7913 | 84.34 | x | x | x | x | x | 119 | 109 | 602 | 549 | x |
| ResNet-18 | Base | 1 | 6.584 | 17.07 | 0.7243 | 0.02876 | x | x | x | x | 0.3189 | 0.9103 | 18.73 | x | x | x | x | x | 119 | 117 | 602 | 592 | x |
| ResNet-18 | N3 seed mean | 3 | 5.891 | 15.87 | 0.7835 | 0.02789 | x | x | x | x | 0.3887 | 0.9097 | 18.4 | x | x | x | x | x | 119 | 117 | 602 | 592 | x |

_Note: One fixed dimension vector; independent 3D pose is x and dimension effect is not identifiable._

### Six N3 fits

| Backbone | Seed | Status | Steps | Exposures | Elapsed (s) | Trainable params | Dimensions to N3 | Symmetry supervision | Dimensions to base | Checkpoint SHA-256 |
|---|---|---|---|---|---|---|---|---|---|---|
| DOPE | 1 | COMPLETE | 6000 | 96000 | 1115 | 23331 | true | true | false | e856aa58ab6dd15d… |
| DOPE | 2 | COMPLETE | 6000 | 96000 | 1117 | 23331 | true | true | false | df4e4d98cc98fe80… |
| DOPE | 3 | COMPLETE | 6000 | 96000 | 1110 | 23331 | true | true | false | 8b407a0161167033… |
| ResNet-18 | 1 | COMPLETE | 6000 | 96000 | 532.4 | 23331 | true | true | false | 114fc574943868a5… |
| ResNet-18 | 2 | COMPLETE | 6000 | 96000 | 533.9 | 23331 | true | true | false | 23359f6550f1cc4a… |
| ResNet-18 | 3 | COMPLETE | 6000 | 96000 | 532.8 | 23331 | true | true | false | e2ad85b9397c5beb… |

### Synthetic-only fixed selection

| Backbone | Seed | Temperature | Lambda | Image-diagonal cap | Real outcome used |
|---|---|---|---|---|---|
| DOPE | 1 | 1 | 1 | 0.01 | false |
| DOPE | 2 | 1 | 1 | 0.01 | false |
| DOPE | 3 | 1 | 1 | 0.01 | false |
| ResNet-18 | 1 | 1 | 1 | 0.01 | false |
| ResNet-18 | 2 | 1 | 1 | 0.01 | false |
| ResNet-18 | 3 | 1 | 1 | 0.01 | false |

_Note: Temperature is fixed on synthetic calibration; real outcome used must be false._

### Locked RTX runtime

| Backbone | Path | Median (ms) | P90 (ms) | FPS | Params | Peak allocated (B) |
|---|---|---|---|---|---|---|
| DOPE | Base E2E | 62.74 | 77.59 | 15.94 | 50267350 | 359025152 |
| DOPE | Base + N3 E2E | 65.81 | 80.93 | 15.2 | 50290681 | 359025152 |
| DOPE | N3 only | 2.838 | 3.465 | 352.4 | 23331 | 263436288 |
| ResNet-18 | Base E2E | 8.89 | 9.573 | 112.5 | 15374665 | 103647744 |
| ResNet-18 | Base + N3 E2E | 11.59 | 12.75 | 86.27 | 15397996 | 103647744 |
| ResNet-18 | N3 only | 2.632 | 3.63 | 380 | 23331 | 92359168 |

### Offline lifter case study

| Method | Frames | Available | Fresh | Longest missing (s) | Yaw step median (deg) | Yaw step P90 (deg) | Independent accuracy |
|---|---|---|---|---|---|---|---|
| R0 | 846 | 0.9835 | 0.9835 | 4.036 | 0.2082 | 0.8286 | x |
| N3_seed1 | 846 | 0.9835 | 0.9835 | 4.036 | 0.2295 | 0.788 | x |

_Note: Visible state is unknown; position/yaw accuracy is x._

### YOLO material and occlusion subgroups

| Group type | Group | Method | Seeds | 2D median (px) | 2D P90 (px) | PCK10 (fraction) | E_sym | T median (cm) | R median (deg) | Yaw median (deg) | Pose coverage | Frames | Matched frames | GT corners | Observed corners | Pose frames |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| material | plastic | Base | 1 | 7.509 | 43.64 | 0.6008 | 0.06484 | 10.47 | 2.45 | 1.349 | 1 | 194 | 186 | 1513 | 1459 | 194 |
| material | plastic | OLD_P seed mean | 3 | 6.397 | 42.25 | 0.646 | 0.06384 | 9.703 | 2.099 | 1.227 | 1 | 194 | 186 | 1513 | 1459 | 194 |
| material | plastic | N2 seed mean | 3 | 6.216 | 42 | 0.6572 | 0.06355 | 9.259 | 2.045 | 1.257 | 1 | 194 | 186 | 1513 | 1459 | 194 |
| material | plastic | N3 seed mean | 3 | 6.203 | 41.7 | 0.6568 | 0.06354 | 9.363 | 2.017 | 1.244 | 1 | 194 | 186 | 1513 | 1459 | 194 |
| material | wood | Base | 1 | 6.124 | 45.54 | 0.6856 | 0.02575 | 4.204 | 2.65 | 1.236 | 1 | 125 | 125 | 986 | 986 | 125 |
| material | wood | OLD_P seed mean | 3 | 5.371 | 45.85 | 0.7194 | 0.02516 | 3.85 | 2.331 | 1.066 | 1 | 125 | 125 | 986 | 986 | 125 |
| material | wood | N2 seed mean | 3 | 5.243 | 44.47 | 0.7299 | 0.02494 | 3.805 | 2.194 | 1.064 | 1 | 125 | 125 | 986 | 986 | 125 |
| material | wood | N3 seed mean | 3 | 5.274 | 44.87 | 0.7306 | 0.02496 | 3.739 | 2.202 | 0.9984 | 1 | 125 | 125 | 986 | 986 | 125 |
| occlusion | clean | Base | 1 | 7.258 | 17.49 | 0.6594 | 0.01025 | 2.93 | 1.611 | 0.6333 | 1 | 29 | 29 | 229 | 229 | 29 |
| occlusion | clean | OLD_P seed mean | 3 | 5.523 | 13.57 | 0.7817 | 0.008473 | 2.982 | 1.548 | 0.6109 | 1 | 29 | 29 | 229 | 229 | 29 |
| occlusion | clean | N2 seed mean | 3 | 5.572 | 12.45 | 0.7904 | 0.008109 | 2.806 | 1.499 | 0.5721 | 1 | 29 | 29 | 229 | 229 | 29 |
| occlusion | clean | N3 seed mean | 3 | 5.527 | 12.31 | 0.7977 | 0.008041 | 2.731 | 1.498 | 0.5578 | 1 | 29 | 29 | 229 | 229 | 29 |
| occlusion | moderate | Base | 1 | 9.117 | 24.79 | 0.589 | 0.02942 | 5.862 | 1.903 | 1.36 | 1 | 20 | 20 | 146 | 146 | 20 |
| occlusion | moderate | OLD_P seed mean | 3 | 8.026 | 22.59 | 0.6416 | 0.02797 | 5.51 | 1.687 | 1.191 | 1 | 20 | 20 | 146 | 146 | 20 |
| occlusion | moderate | N2 seed mean | 3 | 7.571 | 22.03 | 0.6575 | 0.02744 | 5.419 | 1.69 | 1.18 | 1 | 20 | 20 | 146 | 146 | 20 |
| occlusion | moderate | N3 seed mean | 3 | 7.512 | 22.21 | 0.6438 | 0.02745 | 5.474 | 1.671 | 1.131 | 1 | 20 | 20 | 146 | 146 | 20 |
| occlusion | severe | Base | 1 | 11.42 | 53.42 | 0.4049 | 0.1281 | 13.87 | 64.96 | 30.52 | 1 | 79 | 71 | 610 | 556 | 79 |
| occlusion | severe | OLD_P seed mean | 3 | 10.38 | 53.07 | 0.4377 | 0.1272 | 15.41 | 21.34 | 12.68 | 1 | 79 | 71 | 610 | 556 | 79 |
| occlusion | severe | N2 seed mean | 3 | 10.09 | 52.55 | 0.4557 | 0.1269 | 15.95 | 17.73 | 11.08 | 1 | 79 | 71 | 610 | 556 | 79 |
| occlusion | severe | N3 seed mean | 3 | 10.07 | 52.28 | 0.4546 | 0.1269 | 15.97 | 17.64 | 11.1 | 1 | 79 | 71 | 610 | 556 | 79 |
| occlusion | unclassified | Base | 1 | 5.342 | 52.79 | 0.7272 | 0.0251 | 7.634 | 1.881 | 1.026 | 1 | 191 | 191 | 1514 | 1514 | 191 |
| occlusion | unclassified | OLD_P seed mean | 3 | 4.815 | 53.89 | 0.7576 | 0.0245 | 6.423 | 1.792 | 0.865 | 1 | 191 | 191 | 1514 | 1514 | 191 |
| occlusion | unclassified | N2 seed mean | 3 | 4.612 | 53.01 | 0.7655 | 0.02427 | 6.161 | 1.709 | 0.8797 | 1 | 191 | 191 | 1514 | 1514 | 191 |
| occlusion | unclassified | N3 seed mean | 3 | 4.652 | 53.11 | 0.7662 | 0.02428 | 6.072 | 1.729 | 0.8748 | 1 | 191 | 191 | 1514 | 1514 | 191 |

_Note: Occlusion labels cover only the declared subset; unclassified is retained._
