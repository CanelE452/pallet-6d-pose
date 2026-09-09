# 선행연구 — 원문 확인분

## Linear-Covariance Loss (ICCV 2023)

Fulin Liu, Yinlin Hu, Mathieu Salzmann. arXiv:2303.11516.
공식 코드 <https://github.com/fulliu/lc> (README 상 명시. **라이선스 미확인 —
코드를 복사하지 않았고, 아래 수식만 참고해 자체 구현한다.**)

[확인] arXiv HTML 본문에서 직접 확인한 수식:

```
implicit function theorem 으로 GT pose 주변 선형화
    y   = y_gt + A(z,w) r_gt                       (6)
    A   = dy/dx = dg(x,z,w)/dx |_{x=x_p}           (7)
    A   = -H^-1 d2 nll(y)/dy dx |_{y=y_gt}         (35, supp.)   H = nll 의 Hessian

잔차 공분산은 대각으로 가정
    M   = diag{ r_gt o r_gt }                      (9)
pose 공분산으로 전파
    C   = A M A^T                                  (8)

loss 는 C 의 대각을 **3D 점 단위로 묶어** 합한다
    E_cov = (1/8) sum_{i=1..8} sqrt( sum_{j=3i-2}^{3i} C_jj )     (11)
    L_LC  = log(E_prior) + 0.5 (E_cov + E_linear) / E_prior       (17)
```

[확인] 세 가지가 이번 설계에 직접 걸린다.

1. pose 표현이 **3D 상의 8 개 bounding box corner (24 성분)** 이다. rotation 과
   translation 을 분리하지 않고, 축별 가중치도 없다.
2. 잔차에는 **Huber 를 적응적으로** 적용한다 (§4.2).
3. **depth(광축 방향 translation)가 lateral 보다 민감하다는 논의가 없다.**
   원문에 해당 분석이 존재하지 않는다.

## COPE (WACV 2023)

Thalhammer et al. arXiv:2208.08807.

[확인] symmetry 처리 방식:

```
L_key      = min_{s in S_i} L_reg( y_hat, s y )
L_rot/tra  = L_reg( y_hat, I(S) y )       I(S) = L_key 를 최소화한 s 의 indicator
```

즉 **하나의 symmetry 가설을 keypoint 항과 pose 항이 공유한다.** 항마다 다른
대칭을 고르면 안 된다는 것이 요지다. 이번 §19 의 공유 g* 규칙은 COPE 의 이 원칙을
그대로 따르는 것이며, 새로운 것이 아니다.

## ★ 이번 제안의 정확한 위치 — 조합에 그친다

`LOSS_DESIGN.md` 에서 유도하듯, 제안한 `L_TR` 은

```
L_TR = sqrt( trace( C_tt ) )        C_tt = C 의 translation 3x3 블록
```

이다. 즉 **Linear-Covariance 의 C = A M A^T 를 그대로 쓰되, 식 (11) 의
"3D corner 단위 묶음" 을 "translation 블록" 으로 바꾼 것**이다
(A 는 가중치 없는 최소제곱 특수해 A = pinv(J)).

지시문 §44 의 규칙 — "두 개의 단순 조합이면 novel 이라고 부르지 않는다" — 을
그대로 적용하면, **현재 형태의 L_TR 은 novel 이 아니다.** LC 의 pose-space 사영을
바꾼 변형이고, symmetry 공유 규칙은 COPE 그대로다.

기여 후보로 남는 것은 결과가 나온 뒤에만 논의할 다음뿐이다.

```
[추정][미검증]
thin pallet geometry + 혼재 C1/C2 + 저앙각 edge-on 레짐에서,
전체 pose 공분산 대신 translation 블록만 쓰는 것이
LC 원형보다 실제로 더 낫다는 증거.
```

이것이 확인되기 전에는 "새 loss" 라고 부르지 않는다.

## 출처
- <https://arxiv.org/abs/2303.11516> · <https://arxiv.org/html/2303.11516>
- <https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Linear-Covariance_Loss_for_End-to-End_Learning_of_6D_Pose_Estimation_ICCV_2023_paper.pdf>
- <https://arxiv.org/abs/2208.08807>
- <https://openaccess.thecvf.com/content/WACV2023/papers/Thalhammer_COPE_End-to-End_Trainable_Constant_Runtime_Object_Pose_Estimation_WACV_2023_paper.pdf>
- <https://github.com/fulliu/lc>

---

## 부록 — 공식 구현 재확인 (2026-09-06, A2b 착수 전)

`github.com/fulliu/lc` 를 다시 열어 **코드 수준**으로 확인한 것. 코드는 복사하지
않았고 수식만 옮겼다.

### 라이선스 — 선언 없음

`LICENSE` / `LICENSE.md` / `LICENSE.txt` 전부 404 다(main 브랜치, HTTP 코드로 확인).
선언된 라이선스가 없으므로 기본값은 all rights reserved 다.
→ **코드 복사 금지.** 수식만 읽고 자체 구현한다.

### `lib/cov_mixed.py::Loss_cov_mixed` 실제 흐름 [확인]

```
err_2d        = pts2d_out - project(K, pts3d, R_gt, t_gt)
error_clamped = clamp_error(err_2d, max_err_len = 32)        # 벡터 길이 clamp, detach 배율
weights, cov  = robust_weights_cov(inv_std2d, error_clamped, rel_thresh=3, w_e_thresh=4)
    cov       = twice_huber(|err|, mean|err| * 3)            # M 의 대각
    weights   = twice_huber(inv_std_pred, delta_inv_std)     # 학습된 점별 가중
jac_pts2update, prior_update_cov = weighted_pnp_jac_wrt_pts2d(..., weights, with_cov=True)
update_cov    = sym( jac_d * cov @ jac_d^T )                 # = A M A^T, (6,6)
jac_up2alter  = d(xform_3d)/d(xi)                            # 8 corner in 3D, (24,6)
alter_cov     = diag( J_alt (A M A^T) J_alt^T )
cov_err       = mean_i sqrt( sum of 3 comps of corner i )
delta         = J_alt (A r);  linear_err = mean_i ||delta_i||
loss_pose     = log(prior_error) + 0.5 * (cov_err + linear_err) / prior_error
```

논문 식과 일치하되 코드에서 처음 드러난 것 세 가지:

1. `twice_huber(a,d) = d(2a-d) if a>d else a^2` — M 의 대각은 **좌표 단위로
   Huber 화된 제곱 잔차**이고, 임계는 상수가 아니라 **인스턴스 평균 |err| × 3** 이다.
2. `weights` 는 네트워크가 낸 `1/std` 를 다시 Huber 화한 값이고, M 이 아니라
   **PnP Jacobian 의 가중**으로 들어간다. M 과 W 는 별개다.
3. `xform_3d` 가 기본이고 `cov_2d` 는 옵션이다. 즉 **pose 공분산을 3D corner
   변위로 밀어낸 뒤** loss 를 만든다.

### 이 모델에서 쓸 수 있는가 — 그렇다 [확인]

`ultralytics 8.4.60` 의 `PoseLoss26` 은 `preds["kpts_sigma"]` 를 받아
`pred_kpts` 채널 3:5 에 `sigmoid` 로 붙인다(RLE 경로). R0 의 `args.yaml` 에
`rle: 1.0` 이므로 이 sigma 가 실재한다.
→ LC 의 `inv_std2d = 1/pts2d_std` 를 **대체물 없이 그대로** 얻을 수 있다.

`weighted_pnp_jac_wrt_pts2d` 는 Ceres 확장(`lib/pnp/setup_ceres.py`)에 기대지만,
GT pose 에서의 선형화라 반복 해가 필요 없고 `A = (J^T W J)^-1 J^T W` 로 닫힌다.
`J` 와 `J_alt` 는 이미 `pnp_jacobian.py` 에 구현·finite-difference 검증돼 있다.

무엇을 그대로 따르고 무엇을 단순화하는지의 전체 목록은
**`METHOD_LOCK_A2B.md` §3~§5** 에 있다.
