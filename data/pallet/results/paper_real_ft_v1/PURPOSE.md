# REAL_FT_V1 — pseudo-label 대신 real manual GT 를 같은 예산에 넣는다

**[소비처]** 논문 self-training 절.  V1~V5 + FAST teacher 의 여섯 음성 결과 다음에
오는 positive control 이다.  결과에 따라 논문 주장이 갈린다.

**[문장]** *"동일한 노출 예산에서 pseudo-label 을 real manual GT 402 장으로 바꾸면
YOLO26n-Pose 의 keypoint localisation 이 source-only R0 를 넘는다."*

이 한 문장이 참이면 논문은 **"synthetic pretraining + minimal real supervision"** 으로
전환한다.  거짓이면 localisation 병목이 **라벨 품질이 아니라 다른 곳**에 있다는 강한
증거가 되고, 그 자체가 여섯 음성 결과를 설명하는 결론이 된다.

**어느 쪽이든 논문에 들어간다.**  그래서 착수한다.

## 왜 이 설계인가

V1~V5 는 전부 *어떤 pseudo-label 을 고를까* 를 바꿨고 전부 실패했다.  다섯 트랙의
수렴점은 "teacher 가 모르는 것을 선택으로 만들어낼 수 없다" 였다.  그렇다면 남은
질문은 하나다 — **teacher 를 사람으로 바꾸면 되는가.**

`SELFTRAIN_EXPOSURE_LOCK` 을 **한 글자도 바꾸지 않는다.**  init, lr, 업데이트 수,
synthetic replay 멤버십, augmentation, seed 가 전부 같다.  1440 개 슬롯에 들어가는
라벨의 출처만 pseudo → real manual GT 로 바뀐다.  그래서 결과 차이를 라벨 품질로
귀속할 수 있다.

## 판단 지표 (사전 고정 — 결과를 보고 고치지 않는다)

주 지표는 **paired NME** 다.  V1~V5 와 같은 evaluator, 같은 PAPER_EVAL 319.

```
G1  ALL paired NME  <  R0            ★주 판정
G2  Night NME      <=  R0
G3  NME p90        <=  R0            꼬리 — V1~V5 가 전부 여기서 죽었다
G4  gross20        <=  R0
G5  Day detection  >=  0.95          파국 방지
G6  Night detection >= R0
```

G1 이 실패하면 **REAL_FT_V1 = FAILED** 이고, 그때 결론은
"라벨 품질은 이 병목의 원인이 아니다" 다.  그 결론도 논문에 쓴다.

## 누수 방지 — 이게 이 실험의 생명이다

```
학습 라벨   challenge/data/01_real/live_capture_gt/   402 장
평가        PAPER_EVAL 319                            (V1~V5 와 동일)
```

측정으로 확인한 분리 (착수 전 실행):

- **픽셀 sha256 대조: 중복 0 장.**  402 장 전부 PAPER_EVAL 319 와 다른 이미지다.
- 물체가 다르다 — 학습 세션은 전부 `plastic_standard_110x110x15`,
  PAPER_EVAL 의 `wood_183705`/`wood_184309` 는 wood 다.
- 해상도가 다르다 — 학습 640x480 CALIBRATED,
  PAPER_EVAL wood 세션 1280x720 `legacy_import` `imported_read_only_source`.
- 저장소 자체 계약이 이미 갈라놓았다 — live_capture_gt 는 전부 `split: "train"`,
  PAPER_EVAL 은 `objects[0].split == "eval"` 정본에서 온다.

## 이 실험이 스스로에게 금지하는 것

- PAPER_EVAL 결과를 보고 epoch·lr·mixing fraction·필터를 고치지 않는다.
  checkpoint 는 **last.pt 고정**, 선택 없음.
- PAPER_EVAL 을 학습·검증 어느 쪽에도 넣지 않는다.
- 실패하면 같은 평가셋에서 REAL_FT_V2 를 설계하지 않는다.
  (`SINGLE_FRAME_SELFTRAINING_DEV_BUDGET_EXHAUSTED = true` 를 그대로 따른다.)
- 학습 데이터는 전부 plastic 이므로 **wood 는 cross-object 일반화**다.
  둘을 합친 수치만 보고하지 않고 반드시 나눠 적는다.

## 조건부 2 번째 arm

G1 이 **PASS 일 때만** 대조군을 돌린다.

```
REAL_IMG_PSEUDO_CONTROL   같은 402 장 이미지, 라벨만 R0 의 자기 예측(pseudo)
```

이게 없으면 "real 이미지를 봐서 좋아진 것"과 "사람 라벨이라 좋아진 것"을 못 가른다.
G1 이 FAIL 이면 이 arm 은 의미가 없으므로 돌리지 않는다.

## 미확인 상태로 착수하지 않는 것

`live_capture_gt` 는 프레임당 3~5 개 코너가 `extrapolated_mask` 로 외삽이고
`pose_status` 가 전부 `UNCONFIRMED_SIGNED_AXIS` 다.  이 팔레트는 footprint 가
정사각(110x110)이라 W/D 축이 원리적으로 모호하다 — 즉 **우리가 고치려는 90° yaw
순열이 라벨 자체에 들어 있을 수 있다.**

`3d-expert` 의 규약 검증(C1/C2/C3)이 끝나기 전에는 학습을 시작하지 않는다.
`USABLE_AS_TRAINING_LABELS = NO` 면 이 실험은 착수하지 않고 그 사실을 기록한다.

## 외부 선행사례

**미조사.**  "합성→실물 keypoint 전이에서 소량 real 라벨의 효율" 에 대한 외부
문헌을 확인하지 않았다.  이 설계는 전적으로 이 저장소의 측정에서 나왔다.
