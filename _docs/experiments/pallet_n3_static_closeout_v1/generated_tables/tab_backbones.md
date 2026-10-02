| Backbone | Path | Median px | P90 px | PCK10 % | T med cm | R med deg | Matched frames | Observed corners | Pose frames | Pose % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo | Base | 6.721 | 43.890 | 63.425 | 7.897 | 2.539 | 311 | 2445 | 319 | 100.000 |
| yolo | N3 | 5.778 | 42.134 | 68.587 | 7.068 | 2.070 | 311 | 2445 | 319 | 100.000 |
| dope | Base | 12.570 | 51.282 | 24.170 | 10.046 | 3.530 | 233 | 1797 | 210 | 65.831 |
| dope | N3 | 7.469 | 53.570 | 44.338 | 8.356 | 3.051 | 233 | 1797 | 210 | 65.831 |
| ResNet18 RGB 10ep CONSTANT-fold | Base | 8.223 | 62.638 | 52.261 | 9.739 | 4.342 | 292 | 2291 | 319 | 100.000 |
| ResNet18 RGB 10ep CONSTANT-fold | N3 | 7.085 | 62.371 | 57.223 | 9.134 | 3.832 | 292 | 2291 | 319 | 100.000 |

Three separately trained N3 heads per backbone, identical additional6000-update budget. Bases RGB only; N3 image features plus WDH and symmetry supervision. All319 and2499 full-reference denominator retained. Different base training budgets and conditional supports prevent causal backbone ranking.
