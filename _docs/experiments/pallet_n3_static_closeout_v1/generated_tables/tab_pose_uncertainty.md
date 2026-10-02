| Backbone | Metric | Mean delta | CI95 low | CI95 high | Seed SD | Seed min | Seed max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yolo | translation_cm_median | -0.829 | -1.653 | -0.195 | 0.293 | -1.107 | -0.523 |
| yolo | translation_cm_P90 | -2.721 | -20.493 | 5.710 | 0.367 | -3.118 | -2.394 |
| yolo | rotation_deg_median | -0.468 | -0.886 | -0.091534 | 0.041178 | -0.511 | -0.429 |
| yolo | rotation_deg_P90 | -0.608 | -9.538 | -0.071090 | 0.037384 | -0.650 | -0.577 |
| yolo | yaw_deg_median | -0.182 | -0.607 | -0.051822 | 0.024751 | -0.210 | -0.162 |
| yolo | yaw_deg_P90 | -0.571 | -18.475 | 0.132 | 0.056073 | -0.616 | -0.508 |
| dope | translation_cm_median | -1.690 | -2.796 | -0.786 | 0.087371 | -1.773 | -1.599 |
| dope | translation_cm_P90 | -4.739 | -19.564 | 4.923 | 1.016 | -5.829 | -3.818 |
| dope | rotation_deg_median | -0.479 | -0.893 | -0.196 | 0.111 | -0.605 | -0.394 |
| dope | rotation_deg_P90 | 1.590 | -1.659 | 7.847 | 0.881 | 0.573 | 2.118 |
| dope | yaw_deg_median | -0.483 | -0.667 | 0.007165 | 0.039121 | -0.528 | -0.455 |
| dope | yaw_deg_P90 | 1.645 | -1.664 | 7.780 | 0.995 | 0.496 | 2.233 |
| resnet18 | translation_cm_median | -0.604 | -2.077 | 0.202 | 0.071250 | -0.670 | -0.529 |
| resnet18 | translation_cm_P90 | 2.759 | -20.988 | 3.960 | 0.695 | 2.091 | 3.479 |
| resnet18 | rotation_deg_median | -0.510 | -0.714 | -0.022118 | 0.064131 | -0.558 | -0.437 |
| resnet18 | rotation_deg_P90 | 0.060633 | -0.520 | 0.432 | 0.069497 | -0.019040 | 0.109 |
| resnet18 | yaw_deg_median | -0.156 | -0.361 | 0.194 | 0.060026 | -0.198 | -0.087616 |
| resnet18 | yaw_deg_P90 | 0.143 | -0.533 | 0.346 | 0.024453 | 0.123 | 0.170 |

N3 minus Base; difference of medians/quantiles, not median paired difference. Session bootstrap10000 seed20260917, same resamples across seeds; mean formed within each draw. Reused-DEV posthoc analysis, no multiplicity correction, geometry-derived reference.
