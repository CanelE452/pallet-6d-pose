T4 PnP availability uses frozen cached predictions and native K/registered cuboids only. EPnP and LM use identical detected indices; all eight depths must be positive. No RANSAC/LOO/flip screening, reference coordinates, or neural forwards. The rate measures numerical availability, not pose accuracy.

Primary own-kind and shared comparisons

| kind | model | n_frames | frames_detected_ge6 | frames_PnP_possible | PnP_possible_rate |
|---|---|---|---|---|---|
| PLASTIC | R0_before_selftraining | 128 | 92 | 92 | 0.71875 |
| WOOD | R0_before_selftraining | 45 | 37 | 37 | 0.8222222222222222 |
| GREEN | R0_before_selftraining | 119 | 116 | 116 | 0.9747899159663865 |
| GREEN | GREEN_LOO_flip_R1 | 119 | 117 | 117 | 0.9831932773109243 |
| GREEN | GREEN_LOO_flip_R2 | 119 | 117 | 117 | 0.9831932773109243 |
| PLASTIC | PLASTIC_LOO_flip_R1 | 128 | 99 | 99 | 0.7734375 |
| PLASTIC | PLASTIC_LOO_flip_R2 | 128 | 119 | 119 | 0.9296875 |
| WOOD | WOOD_LOO_flip_R1 | 45 | 38 | 38 | 0.8444444444444444 |
| WOOD | WOOD_LOO_flip_R2 | 45 | 40 | 40 | 0.8888888888888888 |
| PLASTIC | SYNTHETIC_CONTROL_R1 | 128 | 78 | 78 | 0.609375 |
| WOOD | SYNTHETIC_CONTROL_R1 | 45 | 34 | 34 | 0.7555555555555555 |
| GREEN | SYNTHETIC_CONTROL_R1 | 119 | 112 | 112 | 0.9411764705882353 |
| PLASTIC | SYNTHETIC_CONTROL_R2 | 128 | 95 | 95 | 0.7421875 |
| WOOD | SYNTHETIC_CONTROL_R2 | 45 | 37 | 37 | 0.8222222222222222 |
| GREEN | SYNTHETIC_CONTROL_R2 | 119 | 114 | 114 | 0.957983193277311 |
| GREEN | GREEN_LOO_only_R1 | 119 | 116 | 116 | 0.9747899159663865 |
| GREEN | GREEN_LOO_only_R2 | 119 | 117 | 117 | 0.9831932773109243 |
| PLASTIC | PLASTIC_LOO_only_R1 | 128 | 99 | 99 | 0.7734375 |
| PLASTIC | PLASTIC_LOO_only_R2 | 128 | 108 | 108 | 0.84375 |
| WOOD | WOOD_LOO_only_R1 | 45 | 38 | 38 | 0.8444444444444444 |
| WOOD | WOOD_LOO_only_R2 | 45 | 39 | 39 | 0.8666666666666667 |

Auxiliary cross-kind comparisons

| kind | model | n_frames | frames_detected_ge6 | frames_PnP_possible | PnP_possible_rate |
|---|---|---|---|---|---|
| PLASTIC | GREEN_LOO_flip_R1 | 128 | 99 | 99 | 0.7734375 |
| WOOD | GREEN_LOO_flip_R1 | 45 | 34 | 34 | 0.7555555555555555 |
| PLASTIC | GREEN_LOO_flip_R2 | 128 | 113 | 113 | 0.8828125 |
| WOOD | GREEN_LOO_flip_R2 | 45 | 35 | 35 | 0.7777777777777778 |
| WOOD | PLASTIC_LOO_flip_R1 | 45 | 35 | 35 | 0.7777777777777778 |
| GREEN | PLASTIC_LOO_flip_R1 | 119 | 115 | 115 | 0.9663865546218487 |
| WOOD | PLASTIC_LOO_flip_R2 | 45 | 35 | 35 | 0.7777777777777778 |
| GREEN | PLASTIC_LOO_flip_R2 | 119 | 118 | 118 | 0.9915966386554622 |
| PLASTIC | WOOD_LOO_flip_R1 | 128 | 89 | 89 | 0.6953125 |
| GREEN | WOOD_LOO_flip_R1 | 119 | 116 | 116 | 0.9747899159663865 |
| PLASTIC | WOOD_LOO_flip_R2 | 128 | 111 | 111 | 0.8671875 |
| GREEN | WOOD_LOO_flip_R2 | 119 | 117 | 117 | 0.9831932773109243 |
| PLASTIC | GREEN_LOO_only_R1 | 128 | 101 | 101 | 0.7890625 |
| WOOD | GREEN_LOO_only_R1 | 45 | 34 | 34 | 0.7555555555555555 |
| PLASTIC | GREEN_LOO_only_R2 | 128 | 116 | 116 | 0.90625 |
| WOOD | GREEN_LOO_only_R2 | 45 | 35 | 35 | 0.7777777777777778 |
| WOOD | PLASTIC_LOO_only_R1 | 45 | 35 | 35 | 0.7777777777777778 |
| GREEN | PLASTIC_LOO_only_R1 | 119 | 116 | 116 | 0.9747899159663865 |
| WOOD | PLASTIC_LOO_only_R2 | 45 | 34 | 34 | 0.7555555555555555 |
| GREEN | PLASTIC_LOO_only_R2 | 119 | 117 | 117 | 0.9831932773109243 |
| PLASTIC | WOOD_LOO_only_R1 | 128 | 99 | 99 | 0.7734375 |
| GREEN | WOOD_LOO_only_R1 | 119 | 114 | 114 | 0.957983193277311 |
| PLASTIC | WOOD_LOO_only_R2 | 128 | 111 | 111 | 0.8671875 |
| GREEN | WOOD_LOO_only_R2 | 119 | 116 | 116 | 0.9747899159663865 |
