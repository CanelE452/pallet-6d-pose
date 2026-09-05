# Paper-adjacent technical audits

**이 폴더는 paper-facing 문서가 아니다.** 논문 본문·표·주장의 정본은
`_docs/paper/final/` 이다. 여기 있는 문서는 기술 상태를 기록한 감사 자료이고,
논문에 필요한 결론은 `final/` 에 반영된 형태로만 인용한다.

## 목적

```text
POSE_METRIC_READINESS.md         6D pose metric 이 왜 열리지 않는지, 해제 조건은 무엇인지
EXTERNAL_BASELINE_AUDIT.md       외부 baseline(DOPE / SSP / PVNet) 의 비교 가능 여부
SELF_TRAINING_TRACK_AUDIT.md     YOLO26 self-training 트랙 착수 감사 (학습 0회 시점)
PALLET_POSE_LITERATURE_AUDIT.md  선행 pallet pose 연구 11편이 논문에 보고한 값과 비교 가능 여부
```

## 지금 알아야 할 것

```text
POSE_METRICS_STATUS = BLOCKED
```

rotation · translation · yaw · ADD · ADD-S · 3D IoU · 5cm5deg · 6D pose AUC 는
논문 어디에도 성능 문장으로 쓰지 않는다. blocker 는 라벨링 미완이 아니라 **알고리즘**
이다 — 측정된 최고 axis selector 가 0.65, gate 는 0.95 라 추가 어노테이션으로는
열리지 않는다.

pipeline 서술은 허용된다: *"2D keypoints are consumed by a PnP solver."*
성능 주장은 금지된다: *"our method improves 6D pose."*

외부 baseline 은 DOPE 만 같은 evaluator 로 채점됐고, 그것도 box head 가 없어
box 를 코너에서 유도했다. 그래서 최종 Table 1 에서 DOPE 는 **ranking 열 없이**
architecture reference panel 로만 등장한다.
