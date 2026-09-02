# Table M1 — Main method comparison

population `PAPER_EVAL`.  N 은 manifest 에서 읽는다.  세 행 모두 **같은
evaluator·같은 319/2689·같은 metric 정의**로 채점했다 — 별도 채점기를
만들면 행끼리 비교가 성립하지 않는다.

```text
Method                                N_pos  N_neg  corner↓   det↑  AP50-95↑   AUROC↑   FPR95↓   R med↓    yaw↓
────────────────────────────────────────────────────────────────────────────────────────────────────────────────
DOPE (same-data backbone control)       319   2689   10.083  0.737    0.3412   0.9903   0.0409        —       —
YOLO26n-Pose (synthetic-only)           319   2689    4.420  0.975    0.7688   0.9921   0.0417        —       —
Proposed                                319   2689    4.180  0.984    0.7585   0.9953   0.0283        —       —
```

`R med` 와 `yaw` 는 `POSE_METRICS_STATUS = BLOCKED` 이라 비워 둔다.
2D 개선을 6D pose 개선이라고 쓰지 않는다.

## DOPE 행의 비대칭 — 각주로 반드시 남긴다

DOPE 에는 box head 가 없다.  AP 와 IoU@0.5 매칭에 필요한 box 는 **검출된
cuboid 코너의 bounding box** 로 유도했다.  YOLO 의 box 는 학습된 예측이므로
`AP50-95` 열의 두 값은 같은 양이 아니다.  score 도 DOPE 는 belief peak,
YOLO 는 box confidence라 `AUROC`/`FPR95` 의 척도가 서로 다르다.

직접 비교가 성립하는 열은 **corner 와 det** 이다 — 둘 다 GT 의 2D keypoint 와
IoU 만 쓰고 모델 고유 출력 형식에 의존하지 않는다.

DOPE 추론은 reflect-padding 을 썼다.  plain squash 로 돌리면 truncation·근접에서
체계적으로 과소검출되어 DOPE 를 부당하게 나쁘게 만든다.

## 아직 없는 행

```text
SingleShotPose   INCOMPATIBLE   저장소에 구현이 없다
PVNet            NEEDS_TRAIN    구현 자산은 있으나 과거 negative 결과가 있다
Real-FT          NEEDS_AUDIT    PAPER_EVAL 과의 학습 중복을 먼저 감사해야 한다
```

근거는 `_docs/paper/EXTERNAL_BASELINE_AUDIT.md`.  억지 wrapper 로 숫자를
만들지 않는다.
