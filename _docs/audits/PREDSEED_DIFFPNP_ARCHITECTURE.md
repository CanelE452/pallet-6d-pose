# Solver

기존 `diffpnp3d_loss.py` 의 `_project_batch` / `_jac_batch` / `rodrigues_batch` 를 재사용하고
public helper 하나만 추가했다: `refine_pose_from_predicted_seed`.
신규 architecture 파일 없음.

```
observed_xy    canonical decoder 출력, 원본 image pixel, N=9 (centroid 포함)
object_points  canonical 3D, N=9
K              frame 별 원본 intrinsics
R_seed,t_seed  canonical OpenCV PnP 가 실제 반환한 pose (재계산 아님, detach)
steps 4  damping 0.001  delta_clip 0.5  cond_max 1e+08
```

기존 training-time DiffPnP 와의 차이:
local 7x7 soft-argmax 대신 **canonical decoder 좌표**, GT-pose init 대신 **predicted seed**.

step 별로 projection/Jacobian → normal equation → damping → finite/condition 검사 →
delta clip → update → observed residual 재계산.  residual 이 증가하면 그 update 를 reject 하고
이전 pose 유지.  backtracking parameter sweep 없음.

health guard(finite / det(R)≈+1 / positive depth / condition / residual 비증가)
하나라도 실패하면 seed pose 를 그대로 반환하고 사유를 기록한다.

## Phase D 검증 (실행 전 통과)

- exact pose + exact projection → update ~0, observed residual < 1e-6 [확인]
- 작은 perturbation → observed residual 감소, pose 오차 감소, positive depth 유지 [확인]
- observed_xy 까지 gradient finite [확인]
- signature·source 에 R_gt / t_gt / projected_gt / GT loader 없음 [확인]
- NaN observation → seed 그대로 반환 [확인]
