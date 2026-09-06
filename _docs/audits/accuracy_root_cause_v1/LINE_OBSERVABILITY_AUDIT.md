# line 관측성 감사 — 왜 측정하지 않고 멈췄는가

작성 2026-09-06 · HEAD `2e5ec0e` · **진단 미실행. 그 사유를 기록한다.**

지시문 §12 는 GT line segment 를 따라 image evidence(gradient magnitude, 방향 일치,
평행 offset 대조군)를 재라고 했다. **재지 않았다.** 재면 안 되는 이유가 §11 에서 나왔기
때문이고, 없는 측정을 있는 것처럼 만들지 않는다.

---

## 판정

```
LINE_OBSERVABILITY_DIAGNOSTIC = NOT_RUN
사유 = TARGET_VISIBILITY_NOT_IDENTIFIABLE  (§11 게이트에서 정지)
§13 게이트 Q1/Q2 = 판정 불가 (실패가 아니라 측정 불가)
→ LINE / HOUGH TRAINING = STOP  (지시문 §35 의 정지 기준)
```

## 왜 멈췄나

[확인] `HOUGH_IMPLEMENTATION_AUDIT.md` 가 코드로 확정한 것:
GT line 은 `line_feature_capacity_v2.py:307-318` 에서 **projected cuboid 의 두 코너를
잇고 Liang-Barsky 로 image 경계만** 자른다. 12-edge target 은 명시적 amodal
(`instance_edge_topology.py:341-343`). `physical_edge_query.py:49-51` 이 스스로 적는다 —
"the paper dataset carries no trustworthy per-edge visibility".

즉 네 범주 중 `OUTSIDE_IMAGE` 만 구분되고

```
1 PHYSICAL_VISIBLE_EDGE     구분 불가
2 PHYSICAL_OCCLUDED_EDGE    구분 불가
3 VIRTUAL_CUBOID_EDGE       구분 불가
4 OUTSIDE_IMAGE             구분 가능
```

1·2·3 을 가르지 못하는 target 위에서 "target line 에 image evidence 가 있는가" 를 재면,
평행 offset 대조군을 아무리 붙여도 **무엇의 evidence 인지 모르는 수치**가 나온다.
그 수치로 §13 의 Q1/Q2 를 판정하면 게이트가 무의미해진다.

## 다시 열 수 있는 조건 — 이번 감사가 새로 찾은 것

[확인] `gt_v2` 어노테이션(평가자가 읽는 `data/evaluation/pallet_eval_v1/**/annotations/`)에는
**코너별** `visibility` (0/1/2) · `in_frame` · `reason` (`visible`/`occluded`/`truncated`) 이
실제로 들어 있다. 319 프레임 실측 분포: visible 1,570 · unknown 666 · occluded 520 · truncated 43.
(★`GT_TRUST_AUDIT.md` 의 `TARGET_VISIBILITY_NOT_IDENTIFIABLE` 은
`challenge/data/01_real/` 아래 **legacy 사본** 기준이다. 두 사본이 다르다.)

[추정][미검증] 코너별 가시성에서 **간선별** 가시성을 유도하는 것은 가능해 보인다
(양 끝 코너가 모두 visible 이고 같은 가시 면에 속하면 그 간선은 physical visible 후보).
그러나 이건 유도지 라벨이 아니고, 자기 가림 판정에는 면 방향까지 필요하다.
이 유도 규칙을 세우고 사람이 표본 검수하는 것이 line 트랙을 다시 여는 **선행 조건**이다.

## §14 선행연구 조사 — 실행하지 않았다, 그리고 그게 맞다

```
EXTERNAL_SURVEY_NOT_RUN
사유 = 지시문 §14 는 "새로운 line model 이 열릴 경우에만" 조사하라고 조건을 걸었다.
       §13 게이트가 Q1/Q2 판정 불가로 멈췄으므로 그 조건이 성립하지 않는다.
```

따라서 이 감사는 line 표현에 대한 **어떤 novel 주장도 하지 않는다.**
게이트를 통과해 line model 을 열게 되면 그때 CVPR/ICCV/ECCV · TPAMI/IJCV ·
CoRL/ICRA/IROS 를 대상으로 pixel-wise voting 6D pose / edge-based 6D pose /
deep Hough transform / occlusion-robust keypoint localization / symmetry-aware pose
를 조사하고, 저장소의 기존 서베이(`_docs/survey/survey-6d-pose-estimation.md`)를 먼저 읽는다.

## ★새 증거 — 기존 hybrid 는 2D 코너를 아예 건드리지 않는다

2026-09-06 추가 실측. `data/pallet/results/model_compare/HYBRID_POINT_LINE_PER_FRAME.csv`
의 B1(base point) 대 P1(+ Direct-Hough) 를 지표별로 짝지어 비교했다. [확인]

```
지표      비교 가능 n   두 arm 이 완전히 동일한 프레임   중앙 차이
corner        93              93  (100%)              +0.0000 px
R             93               0                      -0.7029 deg
t             93               0                      -0.0017 m
adds          93               0                      -0.0047
iou           93              10                      +0.0319
```

**corner 오차가 93/93 프레임에서 소수점까지 같다.** 즉 이 hybrid 는 pose 해(solve)만
바꾸고 **2D keypoint 층은 입력 그대로 통과시킨다.**

이 감사가 확정한 병목은 **2D 코너의 위치추정**이다(`FAILURE_DECOMPOSITION.md`).
그렇다면 기존 hybrid 계열은 구조상 **병목에 닿을 수 없다** — 성능이 안 올라서가 아니라
그 층을 계산하지 않기 때문이다. 이건 §10 의 "누적하지 않는다" 와는 **다른 층위의**
독립적인 반대 증거다.

## 다시 열더라도 남는 반대 증거

[확인] `EDGE_LOCALIZATION_REQUIREMENT` 는 line 이 유용하려면 각도 오차 ≤1° 를 요구하는데,
`LINE_FEATURE_CAPACITY` 에서 **진짜 along-line sampling 을 하는 계열 A** 조차
3.8~3.9° 에서 정체했다. 누적을 학습 그래프에 넣기 전에 feature 에서 orientation 이
읽히지 않는다는 뜻이라, 누적 가설에 가장 직접적으로 불리하다.

[확인] 그리고 이번 감사가 독립적으로 같은 방향의 증거를 만들었다 —
`SELECTIVE_RISK_AUDIT.md` 에서 예측 코너로 만든 기하 신호(공간대각 교점·면대각 교점·
연결선 길이 변동)의 실패 예측 AUROC 가 0.585~0.649 다.
R0 가 틀릴 때도 코너들끼리는 정합한 육면체를 이룬다. 구조가 깨지는 실패가 아니므로,
**구조(line·edge)를 더 잘 보는 것이 이 실패를 고칠 것이라는 전제 자체가 약하다.**

## 결론

line/edge/voting representation(선택지 E)은 **이번 예산의 후보에서 제외한다.**
"기존 Direct-Hough 실패가 누적 가설을 반증하지 못했다" 는 것은 맞다(Q8 = 구현 family 만
반증). 그러나 (i) target 정의가 아직 서 있지 않고, (ii) 계열 A 가 각도 요건에서 4배 모자라며,
(iii) 실패가 구조 붕괴형이 아니라는 독립 증거가 있다.
세 가지가 동시에 해결되기 전에 line 학습을 시작하면 이미 닫힌 실험의 반복이 된다.
