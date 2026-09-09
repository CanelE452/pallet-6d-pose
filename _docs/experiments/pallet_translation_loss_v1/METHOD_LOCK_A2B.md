# METHOD_LOCK — A2b (LC_DERIVED_STABLE_MECHANISM_BASELINE)

> 상태: **정의 동결. 학습 미착수.**
> 이 문서가 A2b 에 대한 정본이다. `METHOD_LOCK_DRAFT.md` 의 arm 표와 충돌하면
> **이 문서가 이긴다** (아래 §0 참조).

## 0. ★ 이름 충돌을 먼저 정리한다

내가 쓴 `METHOD_LOCK_DRAFT.md` 초안의 `A2b` 와 지시받은 `A2b` 는 **서로 다른 것**을
가리켰다. 봉합하지 않고 재배정한다.

```
초안 A2b (폐기된 이름)  = A1 + L_TR 의 프레임 가중을 등방 스칼라로만 적용
                          -> 이름을 A2iso 로 바꾼다. 지금 실행하지 않는다.
정본 A2b (이 문서)      = CONTROL + prior-art / LC-derived pose-sensitivity term
                          = LC_DERIVED_STABLE_MECHANISM_BASELINE
```

초안의 `A2b` 는 `L_TR` 의 방향 가중이 실재하는지 가리는 하위 ablation 이고,
지금 묻는 질문("sensitivity 를 넣으면 병목이 움직이나")에 답하지 않는다.
`A2iso` 는 A2b 가 신호를 낸 뒤에야 의미가 생긴다.

## 1. 이 실험이 답해야 하는 질문 — 하나뿐

```
기존 keypoint loss 에 pose/translation sensitivity 정보를 추가했을 때,
같은 synthetic data 와 같은 모델에서
PAPER_EVAL 319 의 translation / depth 가 실제로 줄어드는가?
```

데이터 분포는 동시에 바꾸지 않는다.

## 2. 표기 — 정본 (2026-09-06 정정, freeze)

```
A2b = LC_DERIVED_STABLE_MECHANISM_BASELINE
```

★이전 표기 `PRIOR_ART_MECHANISM_BASELINE` 은 **더 이상 쓰지 않는다.**
구현 중 LC 원식의 수치 퇴화를 하나 발견해 안정화 가드를 넣었기 때문에,
이 arm 은 공식 구현의 재현이 아니다.

```
mean_we -> 0  ->  robust clipping threshold -> 0  ->  weight 전체 0
              ->  H ~= ridge * I  ->  prior 폭발  ->  잘 맞춘 표본일수록 loss 증가
```

따라서 다음 이름을 **쓰지 않는다**.

```
EXACT_LINEAR_COVARIANCE      금지
PRIOR_ART_REPRODUCTION       금지
ours / proposed / novel      금지
```

보고서·논문에 반드시 함께 적는 문장:

```
- Linear-Covariance Loss 의 핵심 아이디어와 covariance propagation 구조를 따른다.
- 공식 코드를 복사하지 않고 자체 구현했다 (원 저장소에 LICENSE 파일이 없다).
- YOLO26 pose 에 맞게 sparse 8-corner correspondence 로 제한했다.
- Ceres 경로를 사용하지 않는다 (GT 선형화라 닫힌 형태로 대체).
- numerical degeneracy 를 막는 stabilization guard 가 추가됐다.
- 따라서 exact reproduction 이 아니라 LC-derived mechanism baseline 이다.
```

A2b 결과가 좋아져도 **우리 proposed method 의 성능으로 부르지 않는다.**

## 3. A2b 정의 — 무엇을 그대로 따르는가

공식 구현 `github.com/fulliu/lc` 의 `lib/cov_mixed.py::Loss_cov_mixed` 와
`losses.py::sparse_pose_loss` 를 **읽고** 수식을 옮긴다.
**코드는 복사하지 않는다** — 저장소에 LICENSE 파일이 없어(LICENSE/.md/.txt 전부 404)
선언된 라이선스가 없고, 기본값은 all rights reserved 다.

그대로 따르는 것:

```
1. 잔차 기준선     err = pred_2d - project(K, R_gt, t_gt, pts3d)      GT 앵커
2. 잔차 clamp      clamp_error(err, max_err_len = 32)  (벡터 길이, detach 된 배율)
3. M 의 대각       cov = twice_huber(|err|, mean|err| * rel_thresh)   rel_thresh = 3
                   twice_huber(a, d) = d(2a-d) if a>d else a^2       좌표 단위 element-wise
4. 가중치          w = twice_huber(inv_std_pred, delta)  w_e_thresh = 4
                   inv_std_pred = 1 / sigmoid(kpts_sigma)   ★ 이 모델에 실재한다
5. pose 공분산     Sigma_xi = A M A^T,   A = 가중 PnP 의 d(xi)/d(pts2d)
6. 표현 사영       xform_3d — pose 를 8 개 3D bbox corner 위치(24 성분)로 밀어낸다
7. cov 항          loss_cov_3d: 24 대각을 corner 별 3 개씩 묶어 합 -> sqrt -> 8 개 평균
8. linear 항       delta = J_alt A r 를 corner 별 3-vector norm -> 8 개 평균
9. prior 항        같은 loss_fn 을 prior_update_cov 에 적용
10. 최종           L = log(prior) + 0.5 * (cov + linear) / prior
```

## 4. 무엇을 단순화하는가 (전부 열거)

```
S1  weighted PnP Jacobian 을 Ceres(`lib/pnp`, `pnp_auto`) 없이 닫힌 형태로 쓴다.
    GT pose 에서의 선형화라 반복 해가 필요 없다:
        A     = (J^T W J)^-1 J^T W ,  W = diag(w)
        prior = (J^T W J)^-1               (단위 잔차분산 가정)
    J 는 이미 구현·검증된 `pnp_jacobian.py` 다
    (finite difference median 5.2e-10 / p99 1.8e-9, JACOBIAN_VERIFICATION.json).
    LC 의 A = -H^-1 d2nll/dy dx 는 가중 최소제곱에서 이 식과 같다.

S2  dense(xyz/NOC) 분기는 쓰지 않는다. 이 모델은 sparse keypoint 라
    `sparse_pose_loss` 경로만 해당한다.

S3  diameter 정규화를 쓰지 않는다 (LC 는 optional). 팔레트는 치수가 프레임마다
    달라 diameter 로 나누면 arm 간 비교가 물체 크기와 섞인다.

S4  LC 의 Laplace NLL keypoint 항으로 base loss 를 갈아끼우지 않는다.
    base 는 ultralytics `PoseLoss26` (OKS + RLE) 그대로 두고 A2b 항만 더한다.
    -> 두 arm 의 차이가 이 항 하나로 국한된다.

S5  8 corner 만 쓴다. index 8(centroid)은 독립 3D 대응이 아니므로 제외한다
    (기존 keypoint loss 에서는 그대로 유지).

S6  `functorch.jacfwd` 대신 J_alt 를 해석적으로 쓴다.
    p_i = R X_i + t 이므로  dp_i/d(dr) = -skew(R X_i),  dp_i/d(dt) = I.
    이것은 `pnp_jacobian.py` 의 dPdxi 와 같은 블록이라 이미 검증돼 있다.
```

## 5. ★ translation only 로 제한했는가 — 아니다. 그 이유

**제한하지 않았다.** A2b 는 LC 원형대로 `xform_3d`(8 corner in 3D) 사영을 쓴다.

이유를 숨기지 않고 적는다. `LOSS_DESIGN.md` §2 에서 유도했듯

```
L_TR = sqrt( trace( Sigma_xi 의 translation 3x3 블록 ) )
```

이다. 즉 **LC 의 Sigma_xi 를 translation 블록으로 사영하기만 하면 그것이 곧
현재의 `L_TR`** 이다. A2b 를 translation-only 로 제한하면
`A2b == L_TR`(가중·robust 설정 차이만 남음)이 되어, "prior-art baseline 으로
먼저 확인한다" 는 이번 실험의 목적이 무너진다.

따라서:

```
A2b  = LC 원형 (8 corner in 3D 사영).  translation-only 제한 없음.
L_TR = 그 Sigma_xi 를 translation 블록으로 제한한 변형.  이번에 실행하지 않는다.
```

A2b 가 depth/translation 을 움직이면 그것은 **LC 가 이미 하는 일**이고,
그 위에 팔레트 고유 문제를 특정한 뒤에야 `L_TR` 을 제안으로 올린다.

## 6. CONTROL 정의

```
CONTROL = 같은 init(R0 checkpoint) + 같은 data + 같은 steps
        + symmetry/keypoint baseline (per-sample C1/C2, C4 는 데이터에 없음)
        + pose-sensitivity term 없음  (lambda = 0)
A2b     = CONTROL + §3~§5 의 LC-derived term
```

`lambda = 0` 일 때 A2b 코드 경로가 CONTROL 과 **구성적으로 동일**해야 한다
(테스트로 강제: `test_lambda_zero_matches_control`).

★ **R0 의 7.90 cm 와만 비교하지 않는다.** 추가 학습 자체의 효과를 제거하려면
같은 예산의 matched continuation control 이 필수다. 표에는 control 과 A2b 를
나란히 놓고, R0 는 참고 열로만 둔다.

## 7. 동일하게 고정하는 것

```
training data     현 Stage A manifest (g38_legacy_v1v2_p0_tex20k, train 55,980)
NEW SYNTHETIC     0
square 추가       0
C4 ratio 변경     0        (계약상 C4 = 0 프레임. 바꿀 것 자체가 없다)
REAL GT TRAIN     0
self-training     0
solver 변경       0        (평가는 SQPnP + RefineLM 기존 경로)
architecture      동일
initialization    동일     R0 best.pt  sha256 970a0913b38ed4c9...
augmentation      동일     geometry-changing OFF (translate/scale/mosaic=0), color 만
training steps    동일
```

augmentation 을 끄므로 R0 와의 직접 causal comparison 이 아니라
**matched-loss screen** 이다. 두 arm 에 동일 적용한다.

## 8. 예산

```
5 epoch matched continuation screen
```

결과를 본 뒤 epoch 을 늘려 성공시키지 않는다. full 60 epoch 로 시작하지 않는다.

## 9. lambda 선택

학습 전 calibration batch 에서
`||dL_A2b/d pred_kpt|| / ||dL_kp/d pred_kpt|| ~= 0.05` 가 되도록 한 번 정하고
screen 내내 고정한다. `[추정][미검증]`. 결과를 보고 sweep 하지 않는다.
값은 실행 시 이 문서에 append 한다.

## 10. 사전등록 screen gate `[추정][미검증]` — 결과 보기 전 동결

```
A2b vs matched CONTROL

depth median        >= 5% 개선
AND translation median >= 5% 개선
AND rotation median 악화  < 5%
AND 2D corner median 악화 < 5%
AND detection coverage 에 의미 있는 손상 없음
```

5% 는 논문 성공 기준이 아니라 **이 mechanism 을 더 연구할 가치가 있는지** 보는
screen threshold 다.

★ **전체만 좋아지고 plastic 이 그대로면 충분한 성공으로 보지 않는다.**
R0 기준 plastic t 10.47 / depth 9.19 cm, wood t 4.20 / depth 3.69 cm 로
plastic 이 병목의 대부분이다.

★ CI 가 0 을 포함해도 **"LC 계열 전체 실패"라고 쓰지 않는다.**
"현재 예산에서 신호가 확립되지 않았다" 로만 쓴다.

## 11. 보고할 metric

primary: translation median, depth median.
반드시 함께: translation p90 · depth p90 · lateral median ·
rotation median/p90 · yaw median · 2D corner median/p90 · detection coverage.

층: ALL / plastic / wood / Low / Far / Clean / Occlusion.
subgroup 은 서로 겹치므로 **독립 causal factor 로 주장하지 않는다.**

## 12. 결과 분기 (사전 고정)

```
Case 1  depth AND translation 개선   -> POSE_SENSITIVITY_MECHANISM_SUPPORTED
        단 여기서 바로 L_TR 을 proposed 로 돌리지 않는다. LC 가 이미 해결한 것과
        팔레트에 남은 것을 먼저 분해한다.
Case 2  depth 개선 / translation 전체 악화 -> 어느 성분(lateral/rotation)이
        악화되는지 확인. lambda sweep 으로 바로 가지 않는다.
Case 3  2D 개선 / depth 그대로 -> 7.52 cm 가 correspondence magnitude 로 설명되지
        않는다. calibration · physical dimensions · pose GT · sim-real geometry 를
        본다. 새 loss 항을 더 붙이지 않는다.
Case 4  아무것도 개선 안 됨 -> POSE_SENSITIVITY_LOSS_NO_SCREEN_SIGNAL 로 종료.
        L_TR 계열을 확장하지 않는다. 다음 후보는 matched capacity 또는
        targeted data/representation.
```

## 13. 이번 단계에서 실행 금지

```
A0 / A1 / A2 / A3 full Stage A     금지
A2iso (구 A2b, 등방 대조군)         금지 — A2b 신호 확인 후
새 synthetic 생성                   금지
C4 square 비율 변경                 금지
self-training                       금지
DiffPnP 재실행                      금지
```

## 14. 남아 있는 상위 충돌 (해소되지 않음)

`_docs/audits/accuracy_root_cause_v1/FINAL_DECISION.md` 의 `DO_NOT_RUN` 은
**"새 loss 항"** 을 금지한다. A2b 는 새 loss 항을 붙이므로 그 목록에 걸린다.
사용자가 A2b only 진행을 지시했으므로 진행하되, 이 충돌은 기록으로 남긴다.
A2b 가 Case 4 로 끝나면 그 금지가 옳았다는 증거가 하나 더 쌓이는 것이다.

---

# 학습 전 최종 추가 계약 (2026-09-06 freeze)

## 15. geometry-changing augmentation 계약

geometry transform 을 K 에 정확히 반영하는 구현이 **검증된 적이 없으므로**,
이번 screen 은 geometry augmentation 을 전부 끈다.

```
mosaic = 0   mixup = 0
degrees = 0  translate = 0  scale = 0  shear = 0  perspective = 0
fliplr = 0   flipud = 0
```

색상/appearance augmentation(HSV 등)은 기존 recipe 를 유지하되
**CONTROL 과 A2b 에 동일하게** 적용한다.
CONTROL 도 똑같이 끈다. 따라서 primary comparison 은 `R0 vs A2b` 가 아니라
**`matched CONTROL vs matched A2b`** 이고, R0 는 reference 열일 뿐이다.

R0 의 실제 recipe(참고, `args.yaml` 실측):
`mosaic 0.3 · translate 0.1 · scale 0.25 · degrees 0 · shear 0 · perspective 0 ·
fliplr 0 · flipud 0 · mixup 0 · hsv_h 0.015 · hsv_s 0.5 · hsv_v 0.35 · erasing 0.4`
→ **mosaic/translate/scale 셋을 0 으로 바꾼다.** 나머지는 그대로.

## 16. batch geometry wiring smoke — GPU 전 필수

학습 직전, 실제 dataloader batch 를 **100 batch** 읽는다. optimizer step 은 하지 않는다.
각 positive instance 에서 side-table geometry 로 GT corner 를 투영하고
batch keypoint 와 비교한다.

보고: `N instances · median · p95 · p99 · max` 잔차 px,
`bad>0.1px · bad>0.5px · bad>1.0px` 건수.

```
[추정][미검증] wiring gate
p99 <= 0.1 px  AND  max <= 0.5 px
```

실패하면 **GPU training STOP.** loss 를 억지로 고치지 않고
어떤 image transform 이 geometry 계약을 깨는지 찾는다.

## 17. train / val geometry leakage

side table 은 60,000 전체를 담아도 되지만, 학습 중 loss lookup 은
**TRAIN 55,980 만** 접근해야 한다. 코드에 assert 를 넣는다.

```
필수:  val_lookup_count_during_train = 0
       missing_train_lookup          = 0
       unexpected_sample             = 0
보고:  side_table_total · train_lookup_count
```

## 18. matched CONTROL 은 같은 코드 경로

CONTROL 은 stock trainer 를 따로 부르지 않는다. A2b 와 **완전히 같은**
custom trainer / model / dataloader / geometry lookup 을 쓰고 `lambda_lc = 0` 만
다르게 한다. 목적은 `CUSTOM TRAINER 효과` 와 `LC TERM 효과` 의 분리다.
`test_lambda_zero_matches_control` 로 parity 를 강제한다.

## 19. paired matched screen — batch 순서 동일

independent seed 실험이 **아니다**. 같은 init / sample order / batch /
augmentation realization / optimizer / LR / scheduler / batch size / imgsz /
optimizer update 수를 쓴다. LC loss 안에 RNG 를 넣지 않는다.

첫 100 batch 의 `(batch_index, sample_id list)` SHA256 을 저장하고
`FIRST_100_BATCH_HASH_EQUAL = YES` 를 확인한다.
이 실험을 "2 independent replicates" 라고 부르지 않는다 — **paired matched screen** 이다.

## 20. lambda calibration 계약

real PAPER_EVAL 을 λ 선택에 **쓰지 않는다.** synthetic TRAIN 에서 고정
calibration subset 을 만든다.

```
N = 512 positive instances
층화: source_asset · elevation · projected size · aspect ratio (한 구간 쏠림 방지)
멤버십은 결과 보기 전에 LC_CALIBRATION_MANIFEST.txt 로 저장
```

frozen init 에서 optimizer step 없이 forward 하고

```
g_base = dL_keypoint / d pred_kpt_xy
g_lc   = dL_A2b_raw  / d pred_kpt_xy
r_i    = ||g_base_i||_2 / (||g_lc_i||_2 + eps)
lambda_lc = 0.05 * median(r_i over valid instances)
```

`0.05` 는 `[추정][미검증]` screen 고정값. 한 번 계산 후 freeze,
real result 를 보고 다시 고르지 않는다.

보고 필수: `N_total · N_valid · base_grad_norm p10/p50/p90/max ·
raw_lc_grad_norm p10/p50/p90/max · ratio p10/p50/p90 · lambda_lc ·
scaled_lc/base p10/p50/p90 · NaN · Inf · zero-gradient` 및
`LAMBDA_FROZEN_BEFORE_REAL_EVAL = YES`.

```
[추정][미검증] calibration sanity gate — 하나라도 걸리면 학습 시작 안 함
NaN > 0  또는  Inf > 0
또는 valid instances < 90%
또는 scaled_lc/base p90 > 0.5      (소수 singular geometry 가 gradient 지배 방지)
```

## 21. LC conditioning 집계

매 샘플 로그가 아니라 집계 히스토그램만 저장한다.

```
cond(J^T W J) p50/p90/p99 · rank deficient % · ridge fallback % · valid geometry %
visible corner count 분포
Low / Far 층에서 별도 집계 (ill-conditioning 이 그 층에 몰리는지)
```

## 22. checkpoint 선택 — last, best 아님

```
epochs = 5 · early stopping OFF · checkpoint = LAST epoch (epoch5 / last.pt)
```

`best.pt` 를 쓰지 않는다. 이번 목적은 validation fitness 기반 model selection 이
아니라 "같은 5 epoch update 에서 LC 항 하나가 무엇을 바꾸는가" 이다.

## 23. sigma escape 검사 (§14 위험)

LC 가 예측 불확실성을 가중으로 쓰므로, 모델이 keypoint 를 고치는 대신
`sigma 증가 -> inverse std 감소 -> LC penalty 감소` 로 도망칠 수 있다.
CONTROL 대 A2b 에서 `kpts_sigma p10/p50/p90` 을 전체 및 visible corner 에서 비교한다.
A2b 의 sigma 가 크게 증가하면 `UNCERTAINTY_ESCAPE_SUSPECTED` 로 표시한다.
**사전 임계가 없으므로 기술 통계로만 보고하고, 결과를 본 뒤 임계를 만들어
PASS/FAIL 하지 않는다.**

## 24. verdict 사전 고정

```
PASS      POSE_SENSITIVITY_MECHANISM_SUPPORTED
          primary gate 통과 AND plastic 에서도 개선 방향
PARTIAL   POSE_SENSITIVITY_PARTIAL_SIGNAL       예) depth 개선, total t/rotation trade-off
NO SIGNAL POSE_SENSITIVITY_LOSS_NO_SCREEN_SIGNAL
INVALID   SCREEN_INVALID                        geometry wiring 실패 · batch mismatch ·
                                                NaN · sigma escape · CONTROL parity 실패
```

`INVALID` 를 method failure 로 부르지 않는다.
`plastic t/depth 가 전혀 움직이지 않으면 overall gate 통과해도 STRONG_PASS 금지.`

## 25. PASS 여도 지지하지 않는 것

PASS 는 "pose-sensitive / covariance-aware correspondence supervision 이
현재 pallet translation 병목을 움직일 수 있다" 까지만 지지한다.
`우리 loss 가 novel 하다` · `pallet-specific loss 가 필요하다` ·
`C4 데이터를 25% 넣어야 한다` 는 **지지하지 않는다.**

NO SIGNAL 이면 A2 / square ratio / C4 synthetic 으로 넘어가지 않고
`matched capacity` 또는 `targeted data/representation` 을 검토한다.

---

METHOD_LOCK_A2B_FROZEN_UTC = 2026-09-06T13:31:10Z
CONTENT_SHA256_AT_FREEZE = 96b5272fd847a4d0421b18ac4b536b4a098e1d7412d2fa807db5b34010eb405a
(위 해시는 이 두 줄을 제외한 본문에 대한 것이다.)
