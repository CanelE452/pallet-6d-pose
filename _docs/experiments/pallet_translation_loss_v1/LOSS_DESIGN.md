# Loss 설계 — L_LC 기준선과 L_TR 제안

## 0. 공통 규약 (여기서 한 번 고정한다)

```
R(xi) = expm(skew(dr)) @ R_gt      dr 은 카메라 프레임
t(xi) = t_gt + dt                  dt 은 카메라 프레임
xi    = [drx, dry, drz, dtx, dty, dtz]
J     = d vec(u) / d xi            u = 8 cuboid corner 의 투영, (16, 6)
```

`dP/d(dr) = -skew(R X)` 다. `-skew(R X + t)` 가 아니다 — 이 오류로 첫 구현이
finite-difference 게이트에서 걸렸다.

**index 8(centroid)은 translation-risk 에서 제외한다.** 독립적인 3D corner
대응이 아니므로 PnP 민감도의 독립 correspondence 로 세지 않는다.
기존 keypoint loss 에서는 8 번을 그대로 유지한다.

## 1. 기준선 L_LC (Linear-Covariance 충실 구현)

`PRIOR_WORK.md` 의 식 (8)(9)(11)(17) 을 그대로 따른다.

```
M    = diag(r o r)                 r 은 잔차, Huber 적응 적용
C    = A M A^T                     A = -H^-1 d2nll/dy dx  (가중 최소제곱 특수해)
E_cov= (1/8) sum_i sqrt( sum_{j in corner i} C_jj )
```

pose 표현은 원문대로 **3D 상 8 corner (24 성분)**. 축별 가중 없음.
목적: 제안이 기존 pose-aware correspondence loss 를 이기는지 볼 기준.

## 2. 제안 L_TR — 그리고 그것이 사실 무엇인지

```
A     = pinv(J)                    SVD, rcond = 1e-6 (감사 결과 무감응)
A_t   = A[3:6, :]                  translation 행
s_j   = || A_t[:, j] ||_2^2
L_TR  = sqrt( sum_j s_j rho(r_j)^2 + eps ) / s_meter
```

`rho` = robust residual (Huber, LC 와 동일 계열), `s_meter` = 1 m 정규화.

★ 정직하게 적으면, `sum_j s_j r_j^2` 는 정확히

```
sum_j ( sum_{k in {tx,ty,tz}} A[k,j]^2 ) r_j^2  =  trace( (A M A^T)_translation block )
```

이다. 즉 **L_TR = sqrt(trace(C_tt))** 이고, LC 의 C 를 그대로 쓰되 식 (11) 의
사영을 "3D corner 단위 묶음" 에서 "translation 3x3 블록" 으로 바꾼 것이다.
가중치 없는 A = pinv(J) 는 LC 의 A 의 최소제곱 특수해다.

depth 전용 변형은 `L_TR_depth = sqrt(C_zz)` 로 같은 틀 안에 있다.
**첫 버전에서 depth 에 임의의 10 배를 주지 않는다** — 가중은 Jacobian 이 정한다.

## 3. 이름

이 값을 "실제 translation error" 라고 부르지 않는다. 정확한 이름은
**local translation-sensitivity surrogate** 다. GT pose 주변 1 차 근사이고,
corner median >= 10px 구간에서 linearised-vs-actual Spearman 이 0.998 -> 0.814
로 실제로 열화한다.

## 4. symmetry 와 하나의 가설을 공유 (§19, COPE 원칙)

```
g* = argmin_g [ L_kp(g) + λ_p L_presence(g) + λ_R L_RLE(g) + λ_geo L_geo(g) ]
```
샘플 하나에 permutation 하나. 좌표·visibility·supervision mask·source weight 가
**같은 permutation 으로 함께** 움직인다. 항마다 다른 대칭을 고르면 안 된다.

계약상 실제 후보 집합은 (Stage 0A 결과):

```
C1  train 19,268 (34.4%)   permutation 1 개 = identity
C2  train 36,712 (65.6%)   {identity, P180}
C4  train      0           확인된 90 도 동치 asset 이 없고, exact-square 도 1 장뿐
```

즉 **Stage A 에서 C4 분기는 한 번도 실행되지 않는다.** 구현은 하되 테스트로만
검증하고, 실행 경로로는 죽어 있다는 사실을 METHOD_LOCK 에 못 박는다.

## 5. λ_geo 선택 (§32)

학습 전 synthetic calibration batch 에서

```
|| d L_geo / d pred_kpt ||  /  || d L_kp / d pred_kpt ||  ~= 0.05
```

가 되도록 λ_geo 를 한 번 정하고 **screen 내내 고정**한다. 결과를 보고 sweep 하지
않는다. 이 0.05 는 `[추정][미검증]` 이며 METHOD_LOCK 에 값과 함께 기록한다.

memory `aggregate-scale-statistic-does-not-transfer-to-pose` 의 교훈대로
λ 를 loss **값** 으로 잡지 않는다 — gradient norm 비로 잡는다.

## 6. augmentation (§22-23)

R0 의 실제 설정은 `translate=0.1, scale=0.25, mosaic=0.3,
degrees=0, shear=0, perspective=0, fliplr=0` 이다 [확인, args.yaml].
`translate / scale / mosaic` 셋 다 K 와 2D GT 의 정합을 깬다.

Stage A 는 **Option B** 를 택한다 — geometry-changing augmentation 을 끄고
color/appearance 만 쓴다. 모든 arm 에 동일 적용한다. 따라서 R0 와의 직접
causal comparison 이 아니라 **matched-loss screen** 이다.
`mosaic = 0` 은 모든 arm 공통 (§23).

## 7. 테스트 목록 (§24)

```
test_c1_only_identity                        test_translation_jacobian_vs_finite_difference  [PASS]
test_c2_preserves_rectangular_cuboid         test_translation_risk_zero_at_gt
test_c4_rejected_for_rectangular_geometry    test_translation_risk_increases_with_residual
test_c4_preserves_exact_square               test_translation_risk_gradient_finite
test_center_index_fixed                      test_translation_risk_no_nan_near_singular_view
test_one_permutation_shared_across_all_terms test_lambda_geo_zero_matches_symmetry_baseline
test_visibility_mask_permuted                test_source_weight_permuted
```

금지토큰류 검사는 AST 로 한다 (memory `forbidden-token-tests-must-use-ast`, 4 회 반복 실수).
