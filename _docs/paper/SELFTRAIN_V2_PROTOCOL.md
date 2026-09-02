# Self-training V2 — protocol

V1 은 실패/진단 baseline 으로 **그대로 보존**한다.  V2 는 별도 트랙이다.

```text
V1 (보존, 수정 금지)                       V2 (신규)
scripts/self_training_yolo/                scripts/self_training_yolo/v2/
challenge/.../paper_selftrain_v1/          challenge/.../paper_selftrain_v2/
data/pallet/results/paper_selftrain_v1/    data/pallet/results/paper_selftrain_v2/
PSEUDOLABEL_FILTER_LOCK.json               SELFTRAIN_V2_METHOD_LOCK.json
SELFTRAIN_EXPOSURE_LOCK.json
CORNER_REGRESSION_CAUSES.md
```

`direct_3dof` 트랙과 square-field pallet 데이터는 V2 논문 실험에 넣지 않는다.

## 1. 모집단의 역할

**중요.**  PAPER_EVAL 319 는 이미 다음에 쓰였다: visibility review · failure
analysis · NME 분석 · contact sheet · axis-permutation 분석 · 그리고 **V2 방법
설계 자체**.

따라서 V2 에 대해서는 confirmation set 이 아니다.

```text
V2_METHOD_DEVELOPMENT_POPULATION = PAPER_EVAL_319
V2_FINAL_CONFIRMATION_POPULATION = UNOPENED
```

파일 이름은 바꾸지 않는다 — 기존 provenance 를 깨지 않기 위해서다.  역할 변경은
이 문서와 `SELFTRAIN_V2_METHOD_LOCK.json` 이 기록한다.

**V2 의 최종 성능 주장을 PAPER_EVAL 319 하나에서 하지 않는다.**

## 2. 연구 질문

```text
Q1  real domain pseudo-label 을 쓰면서 synthetic-only 의 keypoint localisation 을
    보존 또는 개선할 수 있는가?
Q2  box 신뢰도와 keypoint 신뢰도를 분리하면 V1 의 detection 이득을 유지하면서
    localisation 열화를 막을 수 있는가?
Q3  frame-level geometry rejection 보다 per-keypoint geometry masking 이 더
    유효한가?
Q4  near-square 시점에서 hard semantic pseudo-label 을 억제하고 exact synthetic
    replay 를 균형 있게 주면 90 도 축 순열이 줄어드는가?
Q5  그 결과가 손대지 않은 confirmation set 에서도 재현되는가?
```

## 3. V1 과 무엇이 다른가

```text
V1   frame -> confidence -> geometry score -> frame 전체 ACCEPT/REJECT
     ACCEPT 면 box + 9 keypoints 를 모두 pseudo supervision 으로 쓴다

V2   frame -> detection confidence -> box supervision 결정
     동시에 keypoint 마다 conf / removal residual / flip residual / ambiguity 로
     KEEP / IGNORE 를 따로 정한다
```

좋은 detection 프레임 안에 나쁜 keypoint 가 일부 있다고 프레임을 통째로 버리지
않는다.  반대로 box confidence 가 높다고 모든 keypoint 를 믿지 않는다.

## 4. 왜 이 설계인가 — V1 진단이 지목한 원인

`_docs/paper/CORNER_REGRESSION_CAUSES.md` 와
`_docs/paper/generated/AXIS_FAILURES.md` 에서:

```text
frame-level accept/reject 가 너무 거칠다        pool 272 중 13 장(4.8%)만 바뀐다
box quality 와 keypoint quality 를 한 gate 로   difficult viewpoint 의 detection
                                                신호까지 버린다
noisy keypoint 를 frame 통째로 학습             M4 에서 PASS + gross 58 장
ambiguity-prone viewpoint 를 학습셋에서 제거    near-square 비율 21.6% -> 1.6%
synthetic replay 가 그 coverage 를 보존 못 함   축 supervision 이 사라진다
```

## 5. Arm (§14)

```text
V2-A CONF25       box confidence filtering + pseudo fraction 0.25
                  기존 frame-level hard keypoint label + 원래 synthetic replay
V2-B KP-MASK      V2-A + per-keypoint conf/removal/flip masking
V2-C AMBIG-MASK   V2-B + q >= 0.75 프레임의 semantic corner hard label 제거
                  (box/centroid 는 유지)
V2-D FULL         V2-C + synthetic viewpoint-balanced replay (B0/B1/B2 동일 노출)
```

**V2-D 를 Proposed 로 고정한다.**  결과를 보고 A/B/C 중 좋은 것을 Proposed 라고
바꾸지 않는다.

## 6. Primary metric (§16)

corner 지표는 검출된 프레임만 포함하므로 R0 와 adaptation model 이 서로 다른
프레임을 본다.  V2 에서는 localisation primary 를 바꾼다.

```text
COMMON-DETECTED PAIRED NME
  R0 와 비교 모델이 **둘 다** 같은 GT pallet 을 IoU >= 0.5 로 검출한 프레임만
  그 프레임의 같은 supervised keypoint 만
  NME = Euclidean keypoint error / projected cuboid diagonal
```

raw pixel error 는 secondary 로 유지한다.  detection 은 **별도의 primary
endpoint** 다 — 두 축을 한 숫자로 억지로 합치지 않는다.

```text
Coverage  detection rate
Geometry  paired common-frame NME
```

## 7. Axis permutation metric (§17)

프레임마다 `IDENTITY / YAW90 / YAW180 / YAW270 / MIRROR / MISLOCATED` 를 판정하고
**yaw90 + yaw270 비율**을 보고한다.  `q >= 0.75` subset 을 따로 보고한다.

판정은 **최대 코너 오차**로 한다.  median 은 90 도 순열의 이봉분포에 속는다
(`AXIS_FAILURES.md` 참조).

## 8. DEV success gate (§19)

```text
1  Night detection      V2-D >= R0
2  Day detection        R0 대비 파국적 열화 없음
3  Common NME           V2-D < R0
4  Common NME           V2-D < V2-A
5  q>=0.75 axis perm    V2-D < V2-A
6  전체 axis perm       V2-D <= R0
7  detection 만 좋아지고 localisation 이 다시 악화하면 FAIL
```

통계는 paired frame bootstrap, 가능한 곳은 session-cluster bootstrap.
**DEV 에서 CI 가 0 을 포함해도 방향/효과 크기로 final 을 진행할 수 있으나, 이를
paper confirmation 이라 부르지 않는다.**

FAIL 이면 threshold/q/LR/fraction sweep 을 하지 않는다.
`V2_METHOD_STATUS = FAILED` 로 저장하고 untouched final set 을 열지 않는다.

## 9. 주장 회복 조건 (§27)

```text
LEVEL 1  V2-D detection > R0  AND  V2-D common NME < R0
LEVEL 2  + V2-D common NME < V2-A  AND  axis perm < V2-A
LEVEL 3  + POSE_METRICS_STATUS = READY  AND  pose metric 개선
```

LEVEL 3 이 아니면 제목·초록에서 6D performance improvement 를 주장하지 않는다.

## 10. 알려진 제약 — visibility=0 은 순수한 ignore 가 아니다

`data/pallet/results/paper_selftrain_v2/KEYPOINT_MASK_CONTRACT.json` 에 gradient
측정 결과가 있다.

```text
box supervision 은 mask 와 무관        box gradient 비 = 1.000
pose/RLE 항은 masked point 에서 0      확인
masked point 의 좌표는 무시됨          확인
그러나 keypoint objectness 는 살아 있다  masked 에서도 kpt branch gradient > 0
```

`kpt_shape[-1] == 3` 이라 `bce_pose(pred_kpt[..., 2], kpt_mask)` 가 masked point 에
**"보이지 않음" 을 적극 학습**시킨다.  V2 의 mask 는 순수한 ignore 가 아니라
"이 코너는 신뢰할 수 없다" 는 negative visibility supervision 을 겸한다.

배포가 `kp_conf >= 0.5` 를 쓰므로 **DEV 에서 kp_conf 분포를 감시 지표로 본다.**
`kobj` 가중치는 바꾸지 않는다 — V2 의 변경은 training data / supervision contract
뿐이다(§11).
