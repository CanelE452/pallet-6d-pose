# 아키텍처 — YOLO/PnP 기준선과 동결된 공간 패리티 후보

```text
PAPER_METHOD_STATUS = NOT_LOCKED
SPATIAL_PARITY_CANDIDATE_STATUS = DIAGNOSTIC_WINNER_NOT_ADOPTED
ACTIVE_PAPER_SELECTOR = PREDICTION_ONLY_BASELINE
PAPER_SELECTOR_GATE_STATUS = FAIL
```

## 논문 대상 시스템 경계

현재 논문 처리 흐름은 동결된 YOLO/PnP 기준선을 그대로 사용한다. YOLO 자세
모델이 팔레트를 검출하면, 평가 경로는 최종 검출 중 신뢰도가 가장 높은 항목과
순서가 정해진 9개의 `camera_dynamic_0123_v4` 키포인트를 선택한다. 객체
레지스트리는 지정된 객체의 정확한 물리 치수를 제공한다. 기존 예측 전용
단축/장축 선택기는 예측 키포인트, 카메라 내부 파라미터, 레지스트리 치수와 동결된
선택기 상수를 입력받아 PnP 가설을 선택한다. 아래의 Phase-E 공간 패리티 헤드는
동결된 개발 후보이다. 논문 기준선, 배포 래퍼 또는 전체 YOLO 체크포인트에
연결되어 있지 않다.

```text
활성 기준선

RGB 이미지
   │
   ▼
동결 YOLO one2one 자세 추론
   └── 최고 신뢰도 최종 검출: 상자 + 신뢰도 + 키포인트 9개  ───┐
카메라 내부 파라미터 ──────────────────────────────────────────┤
object_type ──► 형상 레지스트리 ──► 정확한 (X,Y,Z) ────────────┤
동결된 선택기 상수 ────────────────────────────────────────────┘
                                             │
                                             ▼
                                예측 전용 단축/장축 선택기
                                             │
                                      선택된 축 가설
                                             │
                                             ▼
                     키포인트 9개 + 내부 파라미터 + 정확한 XYZ + 가설
                                             │
                                             ▼
                                 객체별 PnP + 대칭성 처리
```

정확한 레지스트리 치수는 서로 다른 두 역할을 하므로 혼동해서는 안 된다. 원시 값
`(X,Y,Z)`는 PnP 형상을 정의한다. 반면 학습된 패리티 후보는 합성 TRAIN
통계로 표준화한 크기 불변 벡터
`[log(x/g), log(y/g), log(z/g), log(x/z)]`를 입력받으며, 여기서
`g=(xyz)^(1/3)`이다. 합성 진단 행의 XYZ는 감사된 자산 메타데이터에서,
실제 데이터 행의 XYZ는 객체 레지스트리에서 얻는다. 학습된 조건부 후보를 채택하지
않았더라도 PnP에서의 치수 역할은 기준선의 일부로 유지된다.

## 평가된 공간 패리티 후보

Phase E는 YOLO 체크포인트를 동결한 채, 이미지 분기를 Phase C에서 사용한 단일
64-D 셀로 축약하지 않고 선택된 검출 원천 셀 주변의 국소 이웃을 유지한다.

```text
동결 YOLO 활성 one2one 분류 경로
   │
   └── 정확한 검출 원천 위치 (수준, 행, 열)
          │
          └── 7×7×64 분류 끝에서 두 번째 패치
                 ├── 값 ─► Linear(64,64) + SiLU
                 │            + 위치 임베딩 49개
                 │            + 원천 수준 임베딩 3개 ───────┐
                 └── 경계 유효성 mask ──────────────────────┤
                                                            ▼
                                               마스킹 평균 || 마스킹 최댓값
                                                            │
                                                            ▼
                                               공간 특징 128-D ───┐
26-D 정규화 키포인트 형상 ────────────────────────────────────────┤
4-D TRAIN 정규화 치수 로그 비율 ──────────────────────────────────┘  결합 = 158-D
                                                                           ▼
                                                                   MLP 158→128→32→2
                                                                           │
                                                               SHORT_FRONT / LONG_FRONT
                                                                           │
                                                             정확한 레지스트리 치수 + PnP
```

패치 채널은 유효한 합성 TRAIN 토큰만으로 정규화한다. 키포인트와 치수
통계는 Phase-C 합성 TRAIN 통계를 재사용한다. 경계 셀에는 0 패딩을 적용하고
mask 처리한다. 모든 유효 검출 행에서 패치 중앙은 float16으로 저장하기 전에 기존
단일 셀 64-D 특징과 정확히 일치한다. 유효하지 않은 검출은 패리티 헤드에 넣지 않고
기권으로 처리한다.

선택된 후보는 `S1_SPATIAL_CONCAT`이다. 치수는 후단 4-D 슬롯을 통해 분류기에
입력되지만 YOLO나 키포인트 헤드를 조절하지 않는다. 따라서 이 실험을
“dimension-conditioned YOLO”라고 부르는 것은 타당하지 않다.

## 통제된 융합 탐색

모든 방법은 토큰 인코더, 위치/수준 임베딩, 집계된 공간 표현,
키포인트 특징과 `[128,32]` 분류기를 공유한다. FiLM과 교차 어텐션은 같은
후단 결합 기준선에 더하는 잔차 구성요소이며, 선택된 모델에서 동시에
활성화되는 모듈이 아니다.

```text
실험군                            치수 경로                                                                   3개 시드 합성 DEV BA         고정 중앙값 합성 TEST 정확도
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
S0_SPATIAL_NO_DIMS                어댑터 없음; 후단 슬롯 0                                                    84.8571%                     83.6057%
S1_SPATIAL_CONCAT [선택]          후단 4-D 결합만 사용                                                        86.7949%                     86.7459%
S2_SPATIAL_FILM                   S1 + 채널별 FiLM; 최종 투영 0 초기화                                        86.3609%                     85.8841%
S3_SPATIAL_CROSS_ATTENTION        S1 + 4-head 치수 질의 어텐션; 잔차 투영 0 초기화                            86.7217%                     86.3200%
```

사전 등록 규칙은 S1을 기본값으로 두고, 추가 복잡성을 감수하려면 S2 또는 S3가
S1을 절대값 기준 최소 `2%p` 넘어야 한다고 정했다. 고정된 선택
지표에서 FiLM은 S1 대비 `−0.4340 pp`, 어텐션은 `−0.0732 pp`였으므로 S1을
원천 데이터 기준으로 고정했다. 어텐션의 TEST AUROC가 조금 더 높았다는 이유로 DEV 전용
결정을 뒤집지 않았다.

## 개발 증거와 아키텍처 결정

같은 고정 YOLO 설정에서 국소 공간 이웃은 기존 단일 셀 후단 결합 헤드보다 더
나은 *진단 후보*이다. 다만 Phase-C 실패가 Phase E를 시작하게 했으므로 이는
대응 조건을 맞춘 확증 검사가 아니라 단계 간 기술적 비교이다.

```text
동결 패리티 헤드                                   COMMON128 정답      NIGHT 정답       Restricted ADD-S AUC     Oracle 여유 회수율
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Phase C 단일 셀 B3_IMAGE_KP_DIMS                   77/128              11/28            0.273602                 2.56%
Phase E 7×7 공간 S1_SPATIAL_CONCAT [선택]          82/128              14/28            0.296480                 49.87%
```

이 개선만으로 해당 후보가 논문 방법의 자격을 얻는 것은 아니다. COMMON128에서
S1은 대응되는 치수 미사용 공간 실험군보다 정답 프레임이 16개 많았지만, 동결된 요구 조건
`122/128`에 비해 `82/128`에 그쳤다. NIGHT는 요구 조건 `26/28`에 비해
`14/28`이었고 S0의 `21/28`보다도 낮았다. 올바른 치수는 값이 100% 변경된
합성 셔플보다 10,095개 행 중 정답 158개(`1.565 pp`)만 앞서 요구 조건
505개에 미달했고, 오라클 회수율도 요구 조건 `70%`에 비해 `49.87%`였다. 따라서
E1, E2, E3, E5, E6는 실패했고 지연 시간 E9는 실행하지 않았다.

논문에서 안전하게 사용할 수 있는 결론은 다음과 같다.

> 적응적으로 재사용된 고정 헤드 DEV 진단에서 7×7 국소 공간 이웃을 유지하면
> 단일 원천 셀보다 패리티와 자세 회수 성능이 향상되었다. 사전 등록된 3개 시드
> 합성 DEV 선택 규칙에서 FiLM과 치수 질의 교차 어텐션 모두 단순
> 후단 결합을 넘지 못했다. 이 후보는 사전 등록된 실제 데이터, 치수 사용, 오라클 회수
> 요구 조건을 충족하지 못해 채택되지 않았다.

이를 제안 아키텍처, 배포된 선택기, 다중 형상 결과 또는 논문 FINAL
결과라고 부르는 것은 타당하지 않다. 동결된 YOLO는 현재 합성 TEST에 앞단
노출되어 있었고, COMMON128은 이전 진단에서 이미 사용되었으며, wood 자세는 여전히
차단되어 있고, 손대지 않은 FINAL 모집단은 존재하지 않는다. 실행 후 감사에서는
고정 범위 주장도 더 좁혀졌다. TEST 지표와 실제 GT는 선택에 영향을 주지 않았지만,
캐시와 전체 합성 label 배열은 고정 전에 미리 불러왔다. 이러한 한계는
본 결과가 탐색적이라는 점을 더욱 분명히 한다.

동결된 증거:

- [Phase-E 프로토콜](../../challenge/yolo_pose_one_model/dimension_conditioning_probe/phase_e_spatial_fusion/PRE_REGISTERED_PROTOCOL.md)
- [Phase-E 수치 요약](../../challenge/yolo_pose_one_model/dimension_conditioning_probe/phase_e_spatial_fusion/SPATIAL_FUSION_SUMMARY.json)
- [Phase-E 실행 후 감사](../../challenge/yolo_pose_one_model/dimension_conditioning_probe/phase_e_spatial_fusion/POST_RUN_AUDIT.md)

## 클린 스타트 60k 대응 확장 — 챌린지 전용

별도의 2026-08-30 확장에서는 새 클린 스타트 G38+P0+TEX 체크포인트를 대상으로
고정 캐시 비교를 반복했다. 이는 논문 기준선을 변경하지 않으며, 실패한 실제 데이터
Phase-E gate도 복구하지 않는다. 혼합 4,020행 DEV는 이미 열어 보았고 독립적인 TEST
대응 모집단이 없으므로 챌린지 트랙 근거에 해당한다.

이 확장은 초기 S0/S1 클린 스타트 실행에 없었던 세 가지 아키텍처 대조군을 추가했다.
정확한 중앙 셀 `C1_CENTER_CONCAT`, S2 FiLM, S3 치수 질의 교차 어텐션이다.
D0 치수 전용과 K0 키포인트 전용은 누출/입력 종류 대조군으로 학습했다.
모든 새 헤드는 정확히 같은 S0/S1 3개 시드 샘플링 해시와 고정된 최적화 예산을
사용했으며, S1/S2/S3의 공통 가중치와 0단계 logit은 동일했다.

```text
C1 중앙 대조군:  패치 토큰 24 → 공유 토큰 인코더
                 + 중앙 위치 + 원천 수준 임베딩
                 → 중앙 || 중앙 → KP26 || dims4 → 공유 분류기

S1 공간:         전체 토큰 49개 → 마스킹 평균 || 최댓값
                 → KP26 || dims4 → 공유 분류기
```

고정된 산출물에서 S1은 C1보다 `+1.308 pp` 높았고, 대응 시나리오 클러스터 95% 구간은
`[+0.511,+2.125] pp`였다. 이는 이 진단 범위에서 국소 패치를 유지할 근거를
제공한다. 어텐션은 고정된 원시 BA가 가장 높았지만(`90.743%`), 3개 시드 평균은 S1
대비 `+0.277 pp`에 불과했고 고정된 대응 구간은 0을 포함했다. FiLM도 평균을
개선하지 못했다. 둘 다 동결된 `S1 + 2 pp` 복잡성 규칙을 통과하지 못했으므로,
아키텍처 결정은 FiLM이나 어텐션이 아니라 단순 후단 결합으로 유지한다.

이 확장에서는 사후 헤드만 학습했으며 YOLO 파라미터 갱신은 0이었다. 이를
종단 간 치수 조건화라고 부를 수 없고, S1을 실제 데이터 지도학습 목표
모델과 비교할 수도 없으며, 제안 방법 또는 FINAL 행을 채울 수도 없다. 수치 근거는
[확장 보고서](../../challenge/yolo_pose_one_model/spatial_concat_scratch/architecture_extension/ARCHITECTURE_EXTENSION_REPORT.md)와
[요약](../../challenge/yolo_pose_one_model/spatial_concat_scratch/architecture_extension/ARCHITECTURE_EXTENSION_SUMMARY.json)에 있다.

## Historical DOPE comparison — private pre-bottleneck pathway

> **LEGACY DOPE COMPARISON / NOT ADOPTED YOLO PAPER PIPELINE**
>
> This document preserves the 2026-08-17 two-head DOPE architecture experiment.
> Its E3 decision, 5cm5deg values, and “canonical” language are historical within
> that experiment. The current paper pipeline is YOLO-based, and
> `PAPER_METHOD_STATUS = NOT_LOCKED`; nothing below defines the adopted YOLO
> architecture or fills the proposed-method table row. Current contract:
> [`PAPER_EVALUATION_STATUS_20260828.md`](PAPER_EVALUATION_STATUS_20260828.md)
> and [`PAPER_CLAIM_LOCK.md`](PAPER_CLAIM_LOCK.md).
> The current real benchmark is multi-shape; nothing in this DOPE-era document
> defines wood geometry or lets wood inherit the plastic symmetry contract.

Historical paper-facing write-up of the 2026-08-17 DOPE two-head design and the
evidence for every choice within that experiment. Numbers here are copied from
result JSONs under
`data/pallet/results/paper_s2_multihead/`; the diagnostic narrative lives in
`_docs/audits/MULTIHEAD_FAILURE_DIAG.md` and is pointed at rather than repeated.

**Historical scope:** this file was canonical for the DOPE experiment only. The
older architecture notes under `_docs/method/` and `_docs/models/` are from the
v8 generation and may contradict it; neither source overrides the active YOLO
paper contract.

### The claim, in one sentence

> Corner estimation benefits from a private late pathway rooted **before** the
> line-specific 128-channel bottleneck; capacity added **after** that bottleneck
> does not reproduce the gain.

Two things are bundled in "private pathway rooted before the bottleneck", and the
experiments separate them from a third:

```
(1) corner and line want different late representations
(2) the corner branch needs the 256-channel early features, not the line
    branch's 128-channel output
(3) the corner branch simply needs more parameters
```

**(3) is measured and rejected.**  (1) and (2) are supported jointly and are *not*
separated from each other by the experiments run so far; the write-up says so
wherever the claim appears.

### The architecture

```
RGB 400x400
 │
 ├─ VGG19 features[0:19]                     FROZEN, 2,325,568 params
 │     └─ early feature   (B, 256, 50, 50)
 │
 ├─ late copy L  = vgg[19:27]                trainable, 5,014,912
 │     └─ F50_line       (B, 128, 50, 50)
 │           └─ 12 fixed-role queries -> Direct Hough -> 12 joint P(theta, rho)
 │
 └─ late copy C  = vgg[19:27]                trainable, 5,014,912
       └─ F50_corner     (B, 128, 50, 50)
             └─ DOPE belief stages 4-6 -> 9 channels (8 corners + centroid)

total trainable 22,661,532
```

Both late copies are initialised from the same weights, so at step 0 the two
branches are bit-identical and the line output equals the line-only baseline's.
No head is randomly initialised: the belief stages come from the checkpoint the
whole line stage already builds on (`weights/paper_s2/paper_s2_pdg/A1/epoch_003.pth`,
a `DopeNetwork(numSeg=1)` that already contains belief, affinity and seg heads).

#### Line preservation is structural, not empirical

Because the line branch owns its late copy and the early trunk is frozen, the
line branch receives exactly the gradients it would receive with no corner head
at all.  Measured over 25,000 steps from scratch, two seeds:

```
                    angle     offset      CIGM
A0 line-only  s1   2.2051     0.9693    1.6364
E3 two-head   s1   2.2051     0.9693    1.6364
A0 line-only  s2   2.3360     1.0331    1.6680
E3 two-head   s2   2.3360     1.0331    1.6680
```

Identical to four decimals.  This is a property of the wiring, so it should be
stated as a guarantee rather than as a result that happened to come out well.

### Evidence for each design decision

#### Why not fully shared (one late block, both losses)

A fully shared two-head model does **not** improve the line branch and slightly
degrades it, on every budget and seed measured:

```
A1 vs A0 line angle    25,000 steps   -3.10% (s1)   -4.93% (s2)
E1 vs E0 line angle     3,000 steps   -3.87% (s1)   -5.02% (s2)
```

Four of four negative.  The mechanism is **not** a directional gradient conflict:
cosine between the line and corner gradients on the shared block is positive at
every checkpoint on both seeds (+0.13 to +0.25, negative-batch fraction 0.03 to
0.14).  What the audit found instead is that the corner gradient reaching the
shared block collapses during training:

```
||g_corner|| / ||g_line||   step 0:  21.3      step 25,000:  5.9e-04
```

`lambda_corner` was fixed from the step-0 ratio, so by the end the corner loss
contributes about 2e-05 of the line gradient to the shared trunk.  A static
multitask weight calibrated on step-0 gradient norms measures the initialisation,
not the training.

#### Why not stop-grad (shared late block, corner reads it detached)

Stop-grad preserves the line exactly -- and it is free, so it is the natural
baseline -- but its corner branch is clearly worse than a private late block:

```
E3 vs E2, 3,000 steps, paired frame bootstrap, 10,000 resamples
corner    +19.54%  CI [+15.67, +25.90]  P 1.000   (seed 1)
          +15.54%  CI [ +9.68, +19.29]  P 1.000   (seed 2)
PATH-C R  +20.00%  CI [+12.07, +27.32]  P 1.000
          +23.07%  CI [+11.45, +30.14]  P 0.999
line        0.00%  CI [ +0.00,  +0.00]
```

#### Why the gain is not just capacity — the E4 control

E3 adds 5,014,912 trainable parameters over stop-grad.  E4 grants the same budget
but places it **after** the line bottleneck:

```
frozen early -> late L -> F50 -+-> line head              (identical to A0)
                               |
                               +-> detach -> capacity block -> corner head
```

The block is matched to 0.005% (5,015,168 against 5,014,912) and is a
zero-initialised residual, so at step 0 the corner head sees exactly what it sees
under stop-grad.  Verified before training: line output bit-identical to the
line-only arm; `L_line` gradient into the corner side exactly 0.000e+00 and
`L_corner` into the line side exactly 0.000e+00; twenty-step replay identical.

```
3,000 steps, two seeds, paired frame bootstrap

E4 vs E2 (capacity only)   corner   -1.24%  CI [-7.54, +3.83]  P 0.301
                                    -0.14%  CI [-7.06, +5.18]  P 0.419
E3 vs E4 (same budget,     corner  +20.53%  CI [+15.67, +25.90]  P 1.000
          different place)         +15.66%  CI [ +9.40, +21.57]  P 1.000
                           PATH-C R +13.83%  CI [ +7.08, +23.26]  P 1.000
                                    +14.73%  CI [ +5.35, +22.10]  P 0.999
```

Five million parameters after the bottleneck buy nothing.  The same five million
before it buy 16-21% corner accuracy and 14-15% rotation.  That is the sentence
the architecture is chosen on.

The geometry moves the same way -- E4 is worse than the fully-shared baseline on
the dominant pose-driving mode, while E3 is better than it:

```
front_rear_shift        A1 @25k  0.9543 / 0.9981
                        E3 @25k  0.8355 / 0.8544     better
                        E4 @ 3k  1.3865 / 1.4721     worse
affine_scale_isotropic  A1 @25k  0.9523 / 0.9631
                        E3 @25k  0.9703 / 0.9691     shrinkage roughly halved
                        E4 @ 3k  0.9462 / 0.9106     shrinkage deepened
```

(E4 is a 3,000-step arm and E3 a 25,000-step arm, so this comparison is
directional; the matched-budget comparison is in `CAPACITY_CONTROL_RESULT.md`.)

#### What E3 actually improves

Against the fully-shared model at the same 25,000-step budget, two seeds:

```
corner localisation   +18.86% / +15.84%
PATH-C rotation       + 7.63% / + 6.55%
PATH-C translation    +18.67% / +19.02%
5cm5deg               + 6.84pp / + 4.29pp   (0.0781 -> 0.1465, 0.0938 -> 0.1367)
line                    0.000% (structural)
```

And the improvement is in the *systematic* geometry, not only the pixel error:

```
measure                  A1 @25k        E3 @25k
height_ratio          0.7822/0.9098   0.9528/0.8775
front_area_ratio      0.7545/0.8836   0.8433/0.8674
affine_scale_isotropic 0.9523/0.9631  0.9703/0.9691
centroid_shift        0.4966/0.4957   0.4062/0.4170
front_rear_shift      0.9543/0.9981   0.8355/0.8544
nonaffine_rms         0.6362/0.6320   0.5594/0.5265
```

This matters because the corner residual damages pose far more than its magnitude
suggests: a 1.0-cell corner error moves rotation by 14.0 degrees where isotropic
noise of the same size moves it by 5.6.  The bottleneck is the structure of the
error, not its size, so an architecture that reduces `front_rear_shift` and the
isotropic scale bias is doing something a smaller MSE would not.

### Historical DOPE limitations recorded for that paper draft

1. **The mechanism is not fully separated.**  "Private pathway rooted before the
   bottleneck" bundles a task-specific representation with access to 256 early
   channels instead of 128.  E4 rules out capacity; it does not tell those two
   apart.  Do not write "late specialization" as if it were isolated.
2. **The 5.0M duplication is real cost.**  22.66M trainable against 17.65M.
   Justified here by +17% corner and +7% rotation at zero line cost, but it is a
   cost.
3. **Rotation p90 is the one unstable number.**  E3 against E4 gives +20.19% on
   seed 1 and -22.39% on seed 2.  Medians agree; the tail does not.
4. **Seed variance is large in the line branch and small in the corner branch**
   (15-19% against 0.5% at 6,000 steps).  Any line-side claim below about 20% at
   short budgets needs seed averaging; corner-side claims do not.
5. **Synthetic only.**  Everything above is `v2_prod40k_clean_merged`, whose
   elevation distribution (8% below 8 degrees) is nearly the inverse of the real
   captures (94%).  No real-transfer claim is made from these numbers.

### What was tried and rejected

Recorded so the historical DOPE draft could say the design space was searched
rather than guessed.

```
fully-shared two-head (A1/E1)     line degrades, corner gains nothing
+ visible-mask auxiliary (A2)     REJECT: offset -7.87%, CIGM -6.02% vs A1,
                                  worse on both seeds; the mask head trains fine
                                  (IoU 0.605 -> 0.74) and simply does not help
stop-grad two-head (E2)           line free, corner clearly worse than E3
capacity after bottleneck (E4)    no corner gain at all
CIGM as the fusion path           direct corners beat CIGM corners 74-76% of the
                                  time; oracle-min headroom only 7-9%
native joint point+line solver    Huber-robust least squares over corner
                                  reprojection and line incidence together,
                                  lambda calibrated on the dev split.  Re-run on
                                  E3's own predictions rather than the shared
                                  model's, it improves rotation on both seeds
                                  (+8.2% / +5.3%) but only seed 1 keeps
                                  translation and 5cm5deg; seed 2 loses 9.9% and
                                  6.8pp.  Two-of-two was the pre-registered bar,
                                  so this is not adopted.  The defect is
                                  identified: lambda is selected on rotation
                                  median alone, which is the axis the line term
                                  is good at, so the selection over-weights the
                                  lines and translation pays.  Recorded rather
                                  than patched, because the threshold must not be
                                  changed after seeing the result.
dense vector voting, sparse edge, affinity association, centre-conditioned
offset, side-face anchors         all previously measured negative, see
                                  `pvnet-dense-vector-voting-negative-result`
```

### What the line branch contributes to pose: orientation, and only orientation

The two heads are justified above by corner accuracy and by the pose the corners
drive.  The reverse direction -- feeding the line predictions back into the pose
-- was tested three times, and the three attempts separate cleanly.

The full `(theta, rho)` constraint qualified neither time it was tried:

```
                                     seed 1                 seed 2
native joint point+line             R +8.2  t +5.0  pass   R +5.3  t  -9.9  fail
same, on scale-corrected corners    R +5.7  t +6.1  pass   R +4.3  t  -9.3  fail
```

The third attempt removed `rho` from the pose objective and kept the line's
orientation.  This is not a new head or a new prediction -- the existing joint
residual puts both projected endpoints of an edge on the predicted line, and the
half-difference of those two values is algebraically free of the offset:

```
(da + db)/2  = offset of the edge midpoint      carries rho
(da - db)/2  = (L/2) * sin(delta)               rho cancels exactly
```

Rotation then improves everywhere, by two to four times the full-line margin:

```
                        ALL rotation              V<8 rotation
seed 1  dev            +23.98%  CI [+16.1,+30.2]  +35.90%  CI [+26.2,+47.6]
seed 1  confirmation   +24.56%  CI [+14.5,+33.1]  +39.33%  CI [+29.2,+49.0]
seed 2  dev            +15.76%  CI [+10.5,+21.4]  +22.34%  CI [+14.3,+31.6]
seed 2  confirmation   +16.30%  CI [+11.0,+21.5]  +18.90%  CI [+12.0,+28.1]
```

Twenty subset-by-seed-by-population combinations, and the confidence interval
excludes zero in all of them.  Translation is a different story: its interval
contains zero in seven of eight, so the orientation term neither helps nor
provably harms it.  And `rho` is confirmed as what was doing the damage -- on
seed 2 the full-line solver halves 5cm5deg (0.1367 to 0.0684) while the
orientation-only solver raises it (0.1367 to 0.1504), with the confirmation
population reproducing both.

The pre-registered two-seed gate still fails, because seed 1 loses 3.9% of
translation and 1.4pp of 5cm5deg.  That failure is traced, not mysterious: the
selection rule was given a safety filter on translation but still picked the
*smallest rotation* among the survivors, and rotation falls monotonically with
the weight, so seed 1 was pushed to the edge of the locked grid.  Seed 2's
grid-edge candidate was caught by the filter and it passed on both populations.
Re-choosing the weight after seeing that would no longer be a pre-registered
test, so `THETA_ONLY_LINE_USEFUL` stands at false and re-selection needs its own
registration.

So the honest claim within the historical DOPE experiment is narrower than
"the line branch improves pose" and much narrower than "the line branch is
useless": **the line branch
carries orientation information that transfers to rotation robustly, largest
where corners are truncated, and its offset channel is what has repeatedly
damaged translation.**  Full numbers:
`data/pallet/results/paper_s2_multihead/THETA_ONLY_SOLVER_RESULT.md`.

### Open axis, not an architecture question

The largest remaining lever on translation is the corner configuration's
**per-frame isotropic scale**: restoring it to ground truth improves translation
by 31-33% and adds 3.3-3.5pp of 5cm5deg, more than any line formulation
contributes anywhere in this study.

That lever is now known to be **not recoverable from the network's own outputs**.
Ridge regression of the per-frame factor, fitted on the seen split and read once
on dev, reaches R^2 0.13-0.17 against a pre-registered bar of 0.30, and which
feature block wins flips between seeds.  Applying the prediction makes pose worse:
every corrected arm loses 5cm5deg against no correction at all (-7.0pp, -4.3pp),
and a constant factor loses too.

The mechanism is simple and was worth stating in that DOPE draft. Translation
from PnP is
nearly proportional to the corner configuration's scale, so a multiplicative
correction trades bias for variance.  Two heads' worth of training has already
pushed the median bias down to about 2% (E3 halves A1's 4-5% shrinkage), so an
R^2 = 0.13 estimate injects more variance than the bias it removes.  The oracle
works because it is exact, not because correction is the right shape of fix.

So the next step is not a better scale predictor and not another backbone.  It is
a representation and loss that keep the scale from drifting in the first place.
Full numbers: `data/pallet/results/paper_s2_multihead/FINAL_2HEAD_POSE_QUALIFICATION.md`.
