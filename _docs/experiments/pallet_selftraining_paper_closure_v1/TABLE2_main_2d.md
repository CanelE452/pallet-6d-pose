# Main 2D comparison on 128 images and 985 supported corners. PCK includes detection failures; medians and P90 are conditional on matched detection. LR5 was historically DEV-selected.

| Arm | PCK5 % | PCK10 % | PCK20 % | Med px | P90 px |
| --- | --- | --- | --- | --- | --- |
| R0 | 20.81 | 49.14 | 72.69 | 9.565 | 40.902 |
| RAW_LR5 | 20.30 | 47.51 | 72.59 | 9.961 | 41.958 |
| REF_LR5 | 24.67 | 51.47 | 73.91 | 8.954 | 42.085 |
