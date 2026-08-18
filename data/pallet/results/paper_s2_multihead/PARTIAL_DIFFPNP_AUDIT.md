# PARTIAL DIFFPNP AUDIT — GATE A

```
PARTIAL_DIFFPNP_SUPPORTED = False        (CASE A0-1)
→ partial GT oracle 미실행, partial pose screen 미실행, 새 partial solver 구현 금지
```

training 0. `Deep_Object_Pose/train/diffpnp3d_loss.py` 를 읽어 판정했다.

---

## 질문

기존 `DiffPnP3DLoss` 의 `mask` 가 **corner-level correspondence masking** 을 지원해서
V<8 프레임에도 pose supervision 을 줄 수 있는가?

## 답 — `mask` 는 frame-level 이고, solve 가 끝난 뒤에만 쓰인다

선언된 shape 부터 frame-level 이다.

```
:297   mask (B,) bool(pnp_valid_3d ∧ V8)
```

`forward` 전체(:295~:520)에서 `mask` 가 등장하는 위치는 5곳뿐이고, 그중 **실제 계산에
영향을 주는 것은 한 줄**이다.

```
:434   valid = mask.bool() & depth_ok & nan_ok & cond_ok      ← 유일한 계산 경로
:474   "skip_depth":  ...                                     진단 로그
:475   "skip_nan":    ...                                     진단 로그
:476   "skip_cond":   ...                                     진단 로그
:477   "gated_out":   ...                                     진단 로그
```

`:434` 는 `per_frame` 손실을 **GN solve 이후에** 프레임 단위로 0 으로 만드는 게이트다.
solve 자체에는 관여하지 않는다.

## solver 내부에는 corner-level 마스킹이 존재하지 않는다

```
:109  _project_batch(rvec, tvec, X, K)     mask 인자 없음
:118  _jac_batch(rvec, tvec, X, K)         mask 인자 없음 → (B, 2N, 6) 전 행 사용
```

GN 루프는 8 코너 전부를 무조건 넣는다.

```
obs = pred_xy_safe.reshape(B, 2*N, 1)            8 코너 전부
r   = uv.reshape(B, 2*N, 1) - obs                8 코너 전부, 가중치 동일
J   = _jac_batch(...)          (B, 2N, 6)        8 코너 전부
JtJ = J^T J                                      무효 코너가 그대로 정규방정식에 들어감
```

## corner-level 마스킹이 없어서 막히는 지점 (전부 고쳐야 함)

```
1. GN residual        r = uv - obs        2N 행 전부, 무효 코너 제외 불가
2. GN Jacobian        (B, 2N, 6)          행 제거 불가 → JtJ / 조건수 오염
3. depth guard        depth_ok = (P_pred[:,:,2] > 0).all(dim=1)
                      → 8 코너 **전부** 양의 depth 를 요구
4. 3D corner Huber    per_frame_geometry = _huber(d3d).mean(dim=1)
                      → 8 코너 평균, 무효 코너가 손실에 기여
5. span/PCA 항        covariance = gt_centered^T gt_centered / float(N)
                      amax-amin over dim=1 → 8 코너로 footprint 를 잰다
6. fit coverage       _pnp_fit_coverage(uv_pred, observed_xy, ...) 도 전 코너
```

여섯 군데 전부에 correspondence 마스킹을 넣는 것은 **새 partial PnP solver 를 쓰는
것과 같다.** 브리프가 금지한 작업이고, 중단 기준 1 에 해당한다.

## 최소 correspondence 요건

명시적 최소 개수 체크는 코드에 없다. 대신 `cond_ok = isfinite(cond) & (cond < 1e8)` 이
사후적으로 걸러낸다. 즉 부족한 correspondence 는 "거부" 되는 게 아니라 **오염된 채
풀리고 조건수로만 사후 탈락**한다 — partial 감독에 그대로 쓰면 무효 코너의 위치가
pose 를 끌어당긴 뒤에야 걸러진다는 뜻이다.

## LocalSoftArgmax validity handling

`LocalSoftArgmax2D.forward` 는 belief map 전체에서 argmax 를 잡고 7×7 창으로 sub-pixel
좌표를 낸다. **화면 밖 코너에 대한 validity 개념이 없다** — 존재하지 않는 코너에도
반드시 어떤 좌표를 돌려준다. 그래서 V<8 프레임을 그냥 넣으면 무효 코너가
"그럴듯한 좌표" 로 GN 에 들어간다. 이것이 위 6개 항목보다 상류의 문제다.

## 판정

```
PARTIAL_DIFFPNP_SUPPORTED = False
```

브리프 CASE A0-1 대로 여기서 종료한다. partial GT oracle(PHASE 2.3)과 partial pose
screen(PHASE 2.5)은 **실행하지 않는다** — eligibility 가 없는 상태에서 oracle 을 돌리면
solver 의 성질이 아니라 오염의 성질을 재게 된다.

## 이것이 기존 판정에 주는 의미

```
POSE_AWARE_CORNER_GAIN = False 의 적용범위는 좁혀진 채로 유지된다:
  "V8-only DiffPnP supervision did not establish full-pose gains"
V<8 supervision 이 도움이 되는지는 **여전히 미검증**이며, 기존 코드로는 깨끗하게
검증할 수 없다. 이건 pose-aware 감독에 대한 부정적 증거가 아니라 측정 불가다.
```
