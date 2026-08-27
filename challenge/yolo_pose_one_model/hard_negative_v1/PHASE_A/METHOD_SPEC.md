# HARD_NEGATIVE_V1 — METHOD SPEC

작성 2026-08-27. **결과를 보기 전에 고정한다.** 이 문서를 결과 보고 고치지 않는다.

## 0. 질문

기존 YN(random synthetic negative 9K)은 negative score 를 낮췄지만 positive score 도
같이 낮춰 high-recall 을 악화시켰다. 이번 질문은 그래서 이것이다:

> easy negative 전체에 BCE 를 거는 대신, **model-mined hard negative + hard-anchor
> focused supervision** 을 쓰면 positive recall 을 보존하면서 FP 를 줄일 수 있는가?

"negative supervision 자체가 나쁜가" 를 다시 묻는 것이 아니다.

## 1. 착수 전 검증 (실측 완료)

```
Y0 vanilla last.pt sha256   37f904b975db3e95297af5acb51f6e99360f4b59245cef04d0511af3f5a189b1  ✔일치
commit                      96ddf1967ecee2759e5d36578a84f2e4eb021efe                          ✔일치
NEGATIVE_SYNTH_V1 train     9,000장  paper_release/negative/extracted/negative_synth_v1_train/rgb  ✔
G38 positive                38,002                                                            ✔
N_HARD = round(38,002x0.05) 1,900
```

## 2. ★ spec 원문에서 바뀐 것 — 채점 체크포인트

원 spec §1 은 `PRIMARY = vanilla Y0` 이고 "Y0E 대체 금지" 였다. 사용자가 마지막에
"Y0E 에서" 를 요청해 충돌이 생겼고, **사용자 결정으로 "둘 다 채점 후 비교"** 로 정했다.

```
채점 A   Y0  vanilla   runs_posecls_g38/Y26_G38_Y0_VANILLA_30EP_SEED42/weights/last.pt
채점 B   Y0E          runs_neg_g38/Y0E/weights/best.pt   sha 59771d5c…
비교     두 HARD_NEG1900 membership 의 교집합/Jaccard 를 내고, 그 결과로 본 membership 확정
```

두 membership 이 거의 같으면 선택 자체가 무의미하므로 spec 원문대로 **Y0 를 본안**으로
쓴다. 크게 갈리면 사용자에게 다시 묻는다. 이 규칙을 결과 보기 전에 고정한다.

## 3. Hardness 정의

```
H(x) = max raw one2one pallet class logit   (sigmoid 이전)
동률 = sha256(image_id) 오름차순
```

`preds["one2one"]["scores"]` 에서 직접 읽는다. **post-NMS 출력을 쓰지 않는다** —
억제된 뒤의 그림이라 mining 근거가 못 된다.

anchor 레벨은 concat 순서(P3 6400 → P4 1600 → P5 400, 합 8400)의 누적합으로 정한다.

## 4. ★ 전처리 선택과 사유

```
채택   LetterBox(640)          학습 파이프라인과 동일
기각   PAD=100 reflect + 640   release 배포계약(실사진용)
```

여기서 묻는 것은 "이 이미지를 **학습에 넣었을 때** 모델이 헷갈리는가" 이므로 학습 때
보는 형태로 재야 한다. 배포계약 전처리로 재면 다른 질문에 답하게 된다.

## 5. 금지 준수

```
optimizer 0 · backward 0 · model.train() 0 · fuse 0
BCE gradient 는 해석적으로만 — negative target y=0 에서 dL/dz = sigmoid(z)
requires_grad_(False) 로 전 파라미터 고정
```

`fuse` 금지는 memory `p26-inference-path-not-the-rejection-factor` 의 "fuse 가
one2many 를 지우는 함정" 때문이다.

## 6. GATE A — 결과 보기 전 고정

top1900 에서:

```
A.  mean(max_conf) >= 0.10
B.  fraction(max_conf >= 0.10) >= 0.25

하나라도 만족  ->  HARD_NEGATIVE_POOL_HAS_SIGNAL      (Phase B 진행)
둘 다 실패     ->  EXISTING_NEGATIVE_POOL_TOO_EASY    (30ep 학습 STOP,
                                                       새 pool 설계 제안만 하고 종료)
```

이 gate 는 **source synthetic 만으로** 결정한다. real negative 사용 금지.

## 7. real negative 2,689 의 지위

SECONDARY diagnosis only. 같은 자로 분포만 내서 나란히 보여준다.
**membership 변경 금지 · gate 판정 사용 금지.**
이미 여러 번 본 DEV-negative 라 최종 untouched test 가 아니다.

## 8. Phase B 예고 (Phase A GO 일 때만)

3-arm matched screen, 10ep. 30ep 은 gate 통과 후 winner 하나만.

```
HC_POSREPEAT1900        G38 38,002 + positive repeat 1,900   stock loss   (노출 대조)
HM_HARDNEG1900_STOCK    G38 38,002 + HARD_NEG1900            stock loss   (mining 효과)
HF_HARDNEG1900_FOCALNEG 같은 membership                       focal-neg    (loss 효과)
```

공통: 같은 pretrained `yolo26n-pose.pt` init · seed 42 · batch 32 · imgsz 640 · SGD
lr0 .01 lrf .01 cos_lr · warmup 3 · patience 0 · single_cls · fliplr/flipud 0 ·
deterministic · architecture 변경 0.

### λ_neg 는 real 로 고르지 않는다

source-only calibration batch(positive 30 + hard-negative 2)를 고정하고,
`lambda_neg=1` 에서 negative focal cls gradient norm / positive stock cls gradient norm
= r 을 재서 `lambda_neg = 0.10 / r` 로 **한 번** 계산한다. 이 값을 여기 기록하고
10ep 결과 보기 전에 고정한다. real128/real-negative 보고 수정 금지.

### ★ 편차 2 — lambda_neg 보정 regime (2026-08-27, 사용자 결정)

spec 13 은 "training 전에 source-only calibration batch 를 고정" 이라고만 했고 어느
체크포인트에서 재는지는 정하지 않았다. 실측 결과 **시점에 따라 1,600배 갈렸다**:

```
regime                              r            lambda_neg
init (COCO pretrained, head 재초기화)  2.43e-07     412,330
Y0   (30ep 학습이 도달하는 상태)        3.91e-04       255.6
```

init 에서는 모델이 negative 에 거의 반응하지 않는다 — anchor 의 99.86% 가 p<0.01 이고
그 focal weight 가 6.1e-08 이다. 그 상태에서 10% 를 맞추면 lambda 가 폭주하고, 학습이
진행돼 모델이 negative 에 확신을 갖는 순간 발산한다.

**채택: Y0 regime, lambda_neg = 255.626.** Y0 는 mining 에 이미 쓴 source-only
체크포인트라 real 을 건드리지 않고, 학습 전 고정이라는 조건도 지킨다.

⚠️ 남는 한계: **r 은 학습 중 크게 변한다.** 어떤 고정 lambda 도 어느 시점에선 어긋난다.
spec 설계가 "r 이 대체로 일정" 을 전제했으나 실측은 그렇지 않다. 10ep 결과를 읽을 때
이 점을 감안할 것.

### mosaic — 방법의 일부

YN 에서 negative 의 19.7% 가 mosaic 으로 positive 와 섞인 것이 확인됐다. 이번엔
**sample-type dependent mosaic** 을 쓴다: negative 가 base sample 이면 mosaic OFF,
positive 는 기존 recipe 그대로. negative 200 probe 에서 instance count = 0 이
100% 인지 확인한다.

## 9. 평가셋의 지위

```
G38 val 1,998        synthetic
real positive DEV128 ★이번 screen 선택용 DEV. 최종 paper test 아님
real negative 2,689  ★동상
```

최종 논문 수치는 나중에 untouched real test 가 필요하다.
