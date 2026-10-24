All ten declared risks, fixed higher-risk direction. Values are tied-rank AUC, with missing-input and class counts retained in the accompanying JSON. E/D is teacher external E1 versus D0. No numerical cuts or new model decisions.

OLD_NATIVE / TRAIN

| Declared input signal | q>3 D | q>3 E | available mistake D | available mistake E | known failure D | known failure E | E/D all | E/D ≤3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-detector confidence | 0.562 | 0.600 | 0.645 | NA | 0.858 | 1.000 | 0.571 | 0.527 |
| 1-native keypoint confidence | 0.547 | 0.545 | 0.560 | NA | 0.824 | 1.000 | 0.592 | 0.587 |
| local invalid fraction | 0.510 | 0.506 | 0.500 | NA | 0.800 | 1.000 | 0.500 | 0.500 |
| central32 invalid fraction | 0.510 | 0.506 | 0.500 | NA | 0.800 | 1.000 | 0.500 | 0.500 |
| context invalid fraction | 0.510 | 0.506 | 0.500 | NA | 0.800 | 1.000 | 0.500 | 0.500 |
| input unavailable | 0.510 | 0.506 | 0.500 | NA | 0.894 | 1.000 | 0.501 | 0.500 |
| native bounds invalid | 0.510 | 0.506 | 0.500 | NA | 0.894 | 1.000 | 0.501 | 0.500 |
| query outside bbox / scale | 0.575 | 0.593 | 0.448 | NA | 0.645 | 0.279 | 0.522 | 0.482 |
| global edge concurrence RMS | 0.529 | 0.611 | 0.703 | NA | 0.881 | 1.000 | 0.528 | 0.464 |
| native heldout edge RMS | 0.522 | 0.571 | 0.470 | NA | 0.788 | 1.000 | 0.546 | 0.511 |

OLD_NATIVE / VALIDATION

| Declared input signal | q>3 D | q>3 E | available mistake D | available mistake E | known failure D | known failure E | E/D all | E/D ≤3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-detector confidence | 0.639 | 0.693 | 0.620 | 0.720 | 0.620 | 0.720 | 0.561 | 0.490 |
| 1-native keypoint confidence | 0.544 | 0.555 | 0.543 | 0.477 | 0.543 | 0.477 | 0.583 | 0.576 |
| local invalid fraction | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| central32 invalid fraction | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| context invalid fraction | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| input unavailable | 0.500 | 0.500 | 0.500 | 0.500 | 0.575 | 0.542 | 0.507 | 0.500 |
| native bounds invalid | 0.500 | 0.500 | 0.500 | 0.500 | 0.575 | 0.542 | 0.507 | 0.500 |
| query outside bbox / scale | 0.394 | 0.273 | 0.410 | 0.521 | 0.410 | 0.521 | 0.436 | 0.551 |
| global edge concurrence RMS | 0.645 | 0.618 | 0.560 | 0.589 | 0.560 | 0.589 | 0.532 | 0.538 |
| native heldout edge RMS | 0.611 | 0.605 | 0.537 | 0.511 | 0.537 | 0.511 | 0.523 | 0.495 |

CONTINUOUS_NATIVE / TRAIN

| Declared input signal | q>3 D | q>3 E | available mistake D | available mistake E | known failure D | known failure E | E/D all | E/D ≤3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-detector confidence | 0.679 | 0.634 | 0.813 | 0.729 | 0.841 | 0.779 | 0.553 | 0.556 |
| 1-native keypoint confidence | 0.574 | 0.592 | 0.796 | 0.647 | 0.829 | 0.718 | 0.574 | 0.559 |
| local invalid fraction | 0.510 | 0.513 | 0.578 | 0.493 | 0.648 | 0.594 | 0.504 | 0.500 |
| central32 invalid fraction | 0.506 | 0.506 | 0.519 | 0.498 | 0.600 | 0.598 | 0.501 | 0.500 |
| context invalid fraction | 0.511 | 0.510 | 0.557 | 0.495 | 0.631 | 0.595 | 0.502 | 0.500 |
| input unavailable | 0.505 | 0.503 | 0.500 | 0.500 | 0.629 | 0.667 | 0.500 | 0.500 |
| native bounds invalid | 0.505 | 0.503 | 0.500 | 0.500 | 0.629 | 0.667 | 0.500 | 0.500 |
| query outside bbox / scale | 0.523 | 0.447 | 0.527 | 0.300 | 0.496 | 0.300 | 0.514 | 0.570 |
| global edge concurrence RMS | 0.607 | 0.553 | 0.747 | 0.604 | 0.783 | 0.657 | 0.516 | 0.534 |
| native heldout edge RMS | 0.634 | 0.569 | 0.688 | 0.603 | 0.731 | 0.674 | 0.504 | 0.519 |

CONTINUOUS_NATIVE / VALIDATION

| Declared input signal | q>3 D | q>3 E | available mistake D | available mistake E | known failure D | known failure E | E/D all | E/D ≤3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-detector confidence | 0.659 | 0.658 | 0.746 | 0.446 | 0.755 | 0.446 | 0.573 | 0.555 |
| 1-native keypoint confidence | 0.543 | 0.613 | 0.723 | 0.364 | 0.733 | 0.364 | 0.618 | 0.590 |
| local invalid fraction | 0.502 | 0.500 | 0.509 | 0.500 | 0.526 | 0.500 | 0.499 | 0.500 |
| central32 invalid fraction | 0.501 | 0.500 | 0.500 | 0.500 | 0.517 | 0.500 | 0.499 | 0.500 |
| context invalid fraction | 0.502 | 0.500 | 0.518 | 0.500 | 0.534 | 0.500 | 0.499 | 0.500 |
| input unavailable | 0.501 | 0.500 | 0.500 | 0.500 | 0.517 | 0.500 | 0.499 | 0.500 |
| native bounds invalid | 0.501 | 0.500 | 0.500 | 0.500 | 0.517 | 0.500 | 0.499 | 0.500 |
| query outside bbox / scale | 0.517 | 0.480 | 0.511 | 0.313 | 0.503 | 0.313 | 0.503 | 0.536 |
| global edge concurrence RMS | 0.639 | 0.699 | 0.731 | 0.616 | 0.738 | 0.616 | 0.530 | 0.447 |
| native heldout edge RMS | 0.600 | 0.614 | 0.670 | 0.654 | 0.678 | 0.654 | 0.491 | 0.455 |
