# Dataset composition

DEV는 controlled 173장, FINAL positive는 annotation과 QA eligibility를 모두
충족한 frame만 포함한다. ALL_AVAILABLE은 각 DEV/FINAL union을 image SHA256으로
deduplicate한 보조 population이며 held-out FINAL이 아니다. 조건은 서로 중복될 수
있고 metric은 evaluation 전까지 `—`다.

## Experiment 6 condition table

```text
Population      Condition         N   pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCall↑
──────────────────────────────────────────────────────────────────────────────────────────────
DEV             Plastic         128      —        —       —         —       —       —        —
DEV             Wood             45      —        —       —         —       —       —        —
DEV             DAY             100      —        —       —         —       —       —        —
DEV             NIGHT            28      —        —       —         —       —       —        —
DEV             Occlusion         0      —        —       —         —       —       —        —
DEV             Truncation       28      —        —       —         —       —       —        —
DEV             Far / small       0      —        —       —         —       —       —        —
FINAL           Plastic           0      —        —       —         —       —       —        —
FINAL           Wood              0      —        —       —         —       —       —        —
FINAL           DAY               0      —        —       —         —       —       —        —
FINAL           NIGHT             0      —        —       —         —       —       —        —
FINAL           Occlusion         0      —        —       —         —       —       —        —
FINAL           Truncation        0      —        —       —         —       —       —        —
FINAL           Far / small       0      —        —       —         —       —       —        —
ALL_AVAILABLE   Plastic         128      —        —       —         —       —       —        —
ALL_AVAILABLE   Wood             45      —        —       —         —       —       —        —
ALL_AVAILABLE   DAY             100      —        —       —         —       —       —        —
ALL_AVAILABLE   NIGHT            28      —        —       —         —       —       —        —
ALL_AVAILABLE   Occlusion         0      —        —       —         —       —       —        —
ALL_AVAILABLE   Truncation       28      —        —       —         —       —       —        —
ALL_AVAILABLE   Far / small       0      —        —       —         —       —       —        —
```

## Experiment 7 split composition

```text
Population      Object                Frames   Sessions    DAY   NIGHT    Dimensions    Occlusion   Truncation
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
DEV             Plastic                  128          7    100      28  1.1×0.11×1.3 m            0           19
DEV             Wood                      45          2      0       0  0.8×0.14×0.59 m            0            9
DEV             Combined positive        173          9    100      28             —            0           28
DEV             Negative                2689          1      0       0             —            0            0
FINAL           Plastic                    0          0      0       0  1.1×0.11×1.3 m            0            0
FINAL           Wood                       0          0      0       0  0.8×0.14×0.59 m            0            0
FINAL           Combined positive          0          0      0       0             —            0            0
FINAL           Negative                   0          0      0       0             —            0            0
ALL_AVAILABLE   Plastic                  128          7    100      28  1.1×0.11×1.3 m            0           19
ALL_AVAILABLE   Wood                      45          2      0       0  0.8×0.14×0.59 m            0            9
ALL_AVAILABLE   Combined positive        173          9    100      28             —            0           28
ALL_AVAILABLE   Negative                2688          1      0       0             —            0            0
```

```text
Population      Condition       Frames
──────────────────────────────────────
DEV             Clean                0
DEV             Occlusion            0
DEV             Truncation          28
DEV             Far / small          0
DEV             Low angle            0
DEV             Mid angle            0
DEV             High angle           0
FINAL           Clean                0
FINAL           Occlusion            0
FINAL           Truncation           0
FINAL           Far / small          0
FINAL           Low angle            0
FINAL           Mid angle            0
FINAL           High angle           0
ALL_AVAILABLE   Clean                0
ALL_AVAILABLE   Occlusion            0
ALL_AVAILABLE   Truncation          28
ALL_AVAILABLE   Far / small          0
ALL_AVAILABLE   Low angle            0
ALL_AVAILABLE   Mid angle            0
ALL_AVAILABLE   High angle           0
```

DEV negative의 frozen membership은 2689행을 유지한다. ALL_AVAILABLE negative는
known duplicate image membership을 SHA256으로 합쳐 현재 2688 unique image다.
