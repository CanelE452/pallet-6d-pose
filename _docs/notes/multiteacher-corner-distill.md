# 여러 교사 + 국소 코너 증류 (multiteacher_corner_distill_v1)

계열 코드: `multiteacher_corner_distill_v1`
착수: 2026-09-05

## 1. 제안

**가설.** 합성 데이터만으로 학습한 단일 교사(이하 기준 모델 R0, YOLO26n-Pose)는 실제 영상 일부에서
keypoint 를 구조적으로 크게 틀린다. 이 오차는 기존 geometry filter / confidence / consistency 로
걸러지지 않고, 같은 교사의 hard pseudo-coordinate 를 다시 학생에게 먹이면 교사의 편향이 그대로 전달된다.
서로 다른 representation 을 가진 여러 교사와 실제 RGB 의 국소 코너 증거를 쓰면
그 편향을 넘는 pseudo supervision 을 만들 수 있는가?

**방법 (게이트 순서, 싼 것 먼저).**
- Gate A 여러 교사 상보성 감사 (학습 0) — oracle-best-teacher 상한이 최고 단일 교사보다 실제로 위인가.
- Gate B 국소 코너 증거 감사 (학습 0) — R0 예측 주변 실제 RGB 에 더 정확한 코너 후보가 존재하는가.
- Gate C 국소 코너 전문가 짧은 파일럿 학습 — B 가 통과할 때만.
- Gate D 융합 pseudo-target 품질 감사 → D2 단일 학생 증류 (900 optimizer update).
- Gate E 도메인 편향 진단 → AdaBN / residual adapter.

**판정 지표.** 2D: detection coverage, pooled keypoint median px, p90 px, gross20, gross40.
6D: PoseCov, rotation median, yaw median, translation cm, IoU3D, symmetry-aware ADD AUC.
불확실성: frame bootstrap + recording/session-cluster bootstrap.
게이트별 통과 임계는 `METHOD_LOCK.json` 에 결과를 보기 전에 고정한다.

**예상 실패 모드.**
- 여러 모델이 같은 방향으로 틀린다 (상보성 없음) → Gate A 에서 종료.
- 정답 후보는 있는데 GT 없이 고를 신호가 없다 (oracle only) → Gate B/C 로는 못 넘어간다.
- 국소 좌표는 좋아지는데 학생에게 전달되지 않는다.

**중단 기준.** Gate A 실패 → 여러 교사 융합 학습 금지. Gate B 실패 → 국소 전문가 학습 금지.
Gate C 실패 → adapter 로 구조하지 않는다. 융합 target 품질이 R0 보다 나쁘면 학생 학습 금지.

**누수 경고.** 평가에 쓰는 PAPER_EVAL 319 장은 이미 반복 사용된 development set 이다.
결과가 좋아져도 held-out / confirmed / final / SOTA 로 부르지 않는다. 상태는 DEVELOPMENT_METHOD_SIGNAL.

## 2. 결과

(진행 중 — 게이트별로 이어 쓴다)
