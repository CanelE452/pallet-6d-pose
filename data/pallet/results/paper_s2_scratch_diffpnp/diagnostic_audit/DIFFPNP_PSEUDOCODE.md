# PAPER_S2 DiffPnP3D pseudocode

This pseudocode is a line-by-line semantic description of the current
implementation, with the canonical ep57 options substituted where stated. It
separates:

- the canonical outer training loss;
- the current NaN-safe unrolled-GN DiffPnP3D module;
- the inactive legacy BPnP module;
- the different evaluation decoder and solver.

Source anchors:

- outer training:
  `Deep_Object_Pose/train/train.py:142-196,278-302,442-473,527-535`
- data/targets:
  `Deep_Object_Pose/common/utils_dataset.py:432-485,542-672,720-775`
- local decode and DiffPnP:
  `Deep_Object_Pose/train/diffpnp3d_loss.py:40-146,225-480`
- target geometry:
  `Deep_Object_Pose/train/diffpnp3d_loss.py:501-524`

## Symbols and conventions

```text
B                  batch size, canonical 12
C_bel              9 = 8 corners + centroid
C_aff              16 = 8 corner-to-centroid 2-D vectors
H_in, W_in          400, 400
H_bel, W_bel        50, 50
N                  8 PnP corners

image              float32 [B, 3, 400, 400], ImageNet normalized
belief_gt          float64 [B, 9, 50, 50]
affinity_gt        float32 [B, 16, 50, 50]
belief_mask        float32 [B, 9]
affinity_mask      float32 [B, 16]
mask_gt            float32 [B, 1, 50, 50]
mask_valid         float32 [B]

belief[s]          float32 [B, 9, 50, 50], s=1..6
affinity[s]        float32 [B, 16, 50, 50], s=1..6
seg[j]             float32 [B, 1, 50, 50] logits, j=1..2

X                  float32 [B, 8, 3], camera-facing object corners
K                  float32 [B, 3, 3], original-pixel camera intrinsics
R_gt               float32 [B, 3, 3], object-to-camera rotation
t_gt               float32 [B, 3], object-to-camera translation
diag               float32 [B], object 3-D diagonal
pnp_mask           bool [B]

belief coordinates x,y:
    [0,50) x [0,50)
original coordinates x,y:
    [0,640) x [0,480)
belief -> original:
    x *= 640/50 = 12.8
    y *= 480/50 = 9.6
```

The loader first squashes `640x480 -> 400x400` and then transforms target
coordinates `400x400 -> 50x50`. The separate `50 -> original` conversion above
therefore exactly inverts the two axis-specific target scales for active
canonical frames.

## 1. Build one training sample

```python
def build_sample(image_rgb, annotation, audit_entry):
    # Deep_Object_Pose/common/utils_dataset.py:432-449
    corners = annotation.object.projected_cuboid       # original pixels, order unchanged
    keypoints9 = corners + [annotation.object.centroid]

    # Deep_Object_Pose/common/utils_dataset.py:451-472
    # Deep_Object_Pose/common/heatmap_refinement.py:45-64
    keypoint_valid9 = annotation.pseudo_keypoint_valid or ones(9)
    belief_channel_mask = float32(keypoint_valid9)
    affinity_channel_mask = float32(
        repeat_each(keypoint_valid9[:8] & keypoint_valid9[8], 2)
    )

    # Deep_Object_Pose/common/utils_dataset.py:474-485
    eligible_before_aug = (
        audit_entry exists
        and audit_entry.pnp_valid_3d
        and audit_entry.V8
    )

    # Deep_Object_Pose/common/utils_dataset.py:542-580
    image_400, keypoints_400, optional_mask_400 = albumentations(
        resize(image_rgb, 400, 400),          # anisotropic squash
        keypoints9,
        optional_real_RLE_mask,
        rotate = not eligible_before_aug,     # active DiffPnP frames skip rotation
        photometric_augmentation = True,
    )

    # Deep_Object_Pose/common/utils_dataset.py:583-603
    keypoints_50, optional_mask_50 = resize_targets(
        keypoints_400, optional_mask_400, 50, 50
    )

    # Deep_Object_Pose/common/utils_belief.py:127-175
    # NumPy zeros has default float64.
    belief_gt = float64[9, 50, 50]
    for channel in 0..8:
        if full_support_of_radius(2 * sigma) is inside grid:
            draw_truncated_at_2sigma_gaussian(belief_gt[channel], keypoints_50[channel])
        else:
            belief_gt[channel] = 0

    # Deep_Object_Pose/common/utils_belief.py:97-124
    affinity_gt = float32[16, 50, 50]  # corner -> centroid unit-vector fields

    # Deep_Object_Pose/common/utils_dataset.py:720-734
    if a real RLE mask survived the same spatial transform:
        mask_gt = float32(optional_mask_50 > 0)[None, :, :]
        mask_valid = float32(1)
    else:
        mask_gt = zeros_float32([1, 50, 50])
        mask_valid = float32(0)

    # Deep_Object_Pose/common/utils_dataset.py:736-775
    pnp_active = (
        eligible_before_aug
        and audit_geometry.img_wh == (640, 480)
        and every(keypoints_50[:8], distance_from_border >= 2 * sigma)
    )

    if pnp_active:
        X, K, R_gt, t_gt, diag = float32(audit_geometry)
        pnp_mask = float32(1)
    else:
        # Fixed shapes are retained for default batch collation and batched solve.
        X, K, R_gt, t_gt, diag = WELL_CONDITIONED_FALLBACK
        pnp_mask = float32(0)

    return {
        "img": imagenet_normalize(float32(image_400)),
        "beliefs": belief_gt,
        "affinities": affinity_gt,
        "belief_channel_mask": belief_channel_mask,
        "affinity_channel_mask": affinity_channel_mask,
        "pvnet_mask": mask_gt,
        "pvnet_mask_valid": mask_valid,
        "diffpnp_X": X,
        "diffpnp_K": K,
        "diffpnp_R": R_gt,
        "diffpnp_t": t_gt,
        "diffpnp_diag": diag,
        "diffpnp_valid": pnp_mask,
    }
```

`X` is reconstructed from the audited dimensions and `best_sym`:

```python
# Deep_Object_Pose/train/diffpnp3d_loss.py:501-524
X0 = canonical_box(W, D, H)                    # float64 [8,3]
X = X0 @ SYMS[audit_entry.best_sym].T
```

The loader casts these geometry arrays to float32.

## 2. Six-stage network forward

```python
# Deep_Object_Pose/common/models.py:43-95,167-217
feature = VGG19_pretrained_false_plus_custom_layers(image)
# feature: float32 [B,128,50,50]

belief[1]   = belief_head_1(feature)           # [B,9,50,50]
affinity[1] = affinity_head_1(feature)         # [B,16,50,50]

for s in 2..6:
    stage_input = concat(belief[s-1], affinity[s-1], feature)
    # stage_input: [B,153,50,50]
    belief[s]   = belief_head_s(stage_input)
    affinity[s] = affinity_head_s(stage_input)

# Deep_Object_Pose/common/models.py:234-238
seg[1] = seg_head_1(feature)                   # [B,1,50,50] logits
seg[2] = seg_head_2(concat(seg[1], feature))   # [B,1,50,50] logits
```

Canonical ep57 has no mask-belief fusion and no corner-quality head.

## 3. Base and mask losses

```python
def channel_masked_mse(pred, target, channel_mask):
    # Deep_Object_Pose/common/heatmap_refinement.py:67-99
    # mask is cast to pred dtype, target is not.
    mask = float32(channel_mask).reshape(B, C, 1, 1)
    squared_error = (pred - target) ** 2
    # belief path arithmetic is float64 because belief_gt is float64.
    return mean(squared_error * mask) / clamp_min(mean(mask), tiny_float32)


# Deep_Object_Pose/train/train.py:172-196
loss_belief = float32_scalar(0)
loss_affinity = float32_scalar(0)

for s in 1..6:
    # += is in-place into a float32 accumulator. The float64 belief-MSE result
    # is cast back to float32 here.
    loss_belief += channel_masked_mse(
        belief[s], belief_gt, belief_channel_mask
    )
    loss_affinity += channel_masked_mse(
        affinity[s], affinity_gt, affinity_channel_mask
    )

loss_base = loss_belief + loss_affinity


# Deep_Object_Pose/train/train.py:278-302
if sum(mask_valid) > 0:
    loss_seg_raw = 0
    for j in 1..2:
        per_pixel = BCE_WITH_LOGITS(seg[j], mask_gt, reduction="none")
        per_frame = mean(per_pixel, dims=(channel, height, width))
        loss_seg_raw += sum(per_frame * mask_valid) / sum(mask_valid)
    loss_mask = 0.01 * loss_seg_raw
else:
    loss_mask = 0
```

`channel_masked_mse` is a valid-element average across the whole batch. A frame
with fewer valid channels receives less total influence than a frame with more
valid channels; frames are not normalized independently.

## 4. Local differentiable keypoint decode

```python
def local_softargmax_7x7(belief8):
    # Deep_Object_Pose/train/diffpnp3d_loss.py:59-101
    # belief8: float32 [B,8,50,50]

    flat = reshape(belief8, [B, 8, 2500])
    peak_value, peak_index = max(flat, dim=cell)
    # peak_index is an integer discrete choice: no gradient through selection.

    peak_y = peak_index // 50
    peak_x = peak_index % 50

    offsets = cartesian_product(range(-3, +4), range(-3, +4))  # 49 entries
    window_y = clamp(peak_y + offsets.y, 0, 49)
    window_x = clamp(peak_x + offsets.x, 0, 49)
    gathered = gather(flat, window_y * 50 + window_x)          # [B,8,49]

    weights = softmax(gathered / 0.1, dim=window_cell)
    x_grid = sum(weights * float(window_x), dim=window_cell)
    y_grid = sum(weights * float(window_y), dim=window_cell)

    pred_xy_original = stack([
        x_grid * 12.8,
        y_grid * 9.6,
    ])                                                         # [B,8,2]

    with no_grad:
        # Diagnostic only. This dictionary is discarded by
        # Deep_Object_Pose/train/train.py:461.
        var_x = sum(weights * (window_x - x_grid) ** 2)
        var_y = sum(weights * (window_y - y_grid) ** 2)
        cov_xy = sum(weights * (window_x - x_grid) * (window_y - y_grid))
        sigma = sqrt(var_x + var_y)
        second_peak = max(flat after masking gathered indices)
        ratio = peak_value / (max(second_peak, 0) + 1e-6)

    return pred_xy_original, {
        # peak_value was produced outside no_grad and retains max-value grad,
        # but the caller discards this dictionary.
        "peak": peak_value,
        "second": stop_gradient(second_peak),
        "var_x": stop_gradient(var_x),
        "var_y": stop_gradient(var_y),
        "cov_xy": stop_gradient(cov_xy),
        "sigma": stop_gradient(sigma),
        "ratio": stop_gradient(ratio),
    }
```

If the predicted argmax is near a border, clamping repeats the same edge cell
within the 49 gathered entries. The softmax then counts the repeated value
multiple times.

## 5. Projection and approximate Jacobian

```python
def project_batch(rvec, tvec, X, K):
    # Deep_Object_Pose/train/diffpnp3d_loss.py:109-115
    R = differentiable_rodrigues(rvec)                  # [B,3,3]
    P_cam = X @ transpose(R) + tvec[:, None, :]        # [B,8,3]
    uvw = P_cam @ transpose(K)
    z_safe = clamp_min(uvw.z, 1e-6)
    uv = uvw.xy / z_safe
    return uv, P_cam


def jacobian_batch(rvec, tvec, X, K):
    # Deep_Object_Pose/train/diffpnp3d_loss.py:118-146
    # Analytic camera-projection blocks using the current transformed points.
    # No finite-difference operation and no OpenCV call.
    R = differentiable_rodrigues(rvec)
    P_cam = X @ transpose(R) + tvec[:, None, :]

    d_pixel_d_point = perspective_jacobian(P_cam, K)   # [B,8,2,3]
    d_point_d_rvec = skew_style_rotation_block(P_cam - tvec[:, None, :])
    d_point_d_tvec = identity3

    J = concat(
        d_pixel_d_point @ d_point_d_rvec,
        d_pixel_d_point @ d_point_d_tvec,
        dim=parameter,
    )
    return reshape(J, [B,16,6])
```

The implementation uses no lens-distortion coefficients.

## 6. Current NaN-safe DiffPnP3D forward

```python
def diffpnp3d_loss(pred_xy, X, K, R_gt, t_gt, diag, external_mask):
    # pred_xy: float32 [B,8,2], original pixels
    # Deep_Object_Pose/train/diffpnp3d_loss.py:295-480

    # ---- Current pre-solve NaN/Inf fix: lines 301-309 ----
    pred_finite = all(isfinite(pred_xy), dims=(corner, xy))       # bool [B]
    pred_xy_safe = nan_to_num(pred_xy, nan=0, posinf=0, neginf=0)
    observations = reshape(pred_xy_safe, [B,16,1])
    # Finite values are unchanged. Replaced entries have zero local gradient.

    # ---- Constant GT initialization: lines 287-314 ----
    with stop_gradient_semantics:
        rvec = scipy_Rotation_from_matrix(detach(R_gt).cpu()).as_rotvec()
        tvec = detach(t_gt)
    rvec = move_back_to_device_and_dtype(rvec)

    I6 = identity(6, dtype=pred_xy.dtype)
    condition = nan[B]

    # ---- Four differentiable local updates: lines 317-333 ----
    repeat 4 times:
        uv, _ = project_batch(rvec, tvec, X, K)
        residual = reshape(uv, [B,16,1]) - observations
        J = jacobian_batch(rvec, tvec, X, K)             # [B,16,6]
        A = transpose(J) @ J + 1e-3 * I6                 # [B,6,6]
        rhs = transpose(J) @ residual                    # [B,6,1]
        delta = solve(A, rhs).squeeze(-1)                # [B,6]

        norm = clamp_min(l2_norm(delta), 1e-9)
        delta *= min(1, 0.5 / norm)                      # trust-region clip

        rvec = rvec - delta.rotation
        tvec = tvec - delta.translation

        with no_grad:
            eigenvalues = eigvalsh(A)
            condition = max(eigenvalues) / clamp_min(min(eigenvalues), 1e-12)

    # ---- Canonical geometry term: lines 335-341 ----
    R_pred = differentiable_rodrigues(rvec)
    P_pred = X @ transpose(R_pred) + tvec[:, None, :]    # [B,8,3]

    P_gt = detach(X @ transpose(R_gt) + t_gt[:, None, :])
    normalized_corner_error = (
        l2_norm(P_pred - P_gt, dim=xyz) / clamp_min(diag[:, None], 1e-6)
    )
    per_frame_geometry = mean(
        huber(normalized_corner_error, delta=0.05),
        dim=corner,
    )

    # ---- Present in current source, disabled for canonical ep57 ----
    # Deep_Object_Pose/train/diffpnp3d_loss.py:343-417
    per_frame_undercoverage = projected_span_and_depth_undercoverage(...)
    per_frame_fit_coverage = pnp_projection_coverage_of_detached_observations(...)

    # Canonical inner weights:
    geometry_weight = 1.0
    undercoverage_weight = 0.0
    fit_coverage_weight = 0.0
    per_frame = per_frame_geometry

    # ---- Post-solve frame guards: lines 419-438 ----
    depth_ok = all(P_pred.z > 0, dim=corner)
    numeric_ok = (
        pred_finite
        and isfinite(per_frame)
        and all(isfinite(span_ratio))
        and isfinite(tz_ratio)
    )
    condition_ok = isfinite(condition) and condition < 1e8

    valid = bool(external_mask) and depth_ok and numeric_ok and condition_ok
    per_frame_safe = where(valid, per_frame, 0)
    n_valid = max(sum(valid), 1)
    loss = sum(per_frame_safe) / n_valid

    return loss, detached_scalar_diagnostics
```

The fixed-shape fallback frames still pass through the batched algebra when
another frame in the batch is active. Their placeholder geometry is designed
to remain well conditioned, and `external_mask=False` removes them from the
loss reduction.

The pre-solve fix is necessary because the final `where(valid, ...)` is not by
itself a safe NaN barrier for either linear algebra or backward. The regression
case is at
`scripts/stage0/paper_s2_geometry_unit_audit.py:831-859`; its asserted test is
`challenge/tests/test_paper_s2_geometry_unit_audit.py:27-32`.

## 7. Canonical total loss and gradient routes

```python
# Deep_Object_Pose/train/train.py:442-473
if any(diffpnp_valid):
    pred_xy, diagnostic_confidence = local_softargmax_7x7(belief[6][:, :8])
    # diagnostic_confidence is assigned to "_conf" and never used.
    raw_diffpnp, info = diffpnp3d_loss(
        pred_xy, X, K, R_gt, t_gt, diag, bool(diffpnp_valid)
    )
    diffpnp_ramp = min(1.0, global_step / 1000.0)
    loss_diffpnp = 0.005 * diffpnp_ramp * raw_diffpnp
else:
    loss_diffpnp = 0

loss_total = loss_base + loss_mask + loss_diffpnp

assert isfinite(loss_total)     # scalar-only guard
loss_total.backward()
optimizer.step()
global_step += 1
```

Gradient routing:

```text
six-stage belief MSE
  -> belief heads 1..6
  -> affinity heads 1..5 through later-stage recurrence
  -> shared VGG

six-stage affinity MSE
  -> affinity heads 1..6
  -> belief heads 1..5 through later-stage recurrence
  -> shared VGG

two-stage mask BCE
  -> segmentation heads 1..2
  -> shared VGG
  -X-> no direct belief hard gate/fusion in ep57

DiffPnP3D
  -> final belief head 6
  -> belief heads 1..5 through recurrence
  -> affinity heads 1..5 through recurrence
  -> shared VGG
  -X-> final affinity head 6
  -X-> GT R/t/X/K/diag
  -X-> local covariance/confidence diagnostics
```

`global_step` is initialized to zero after checkpoint loading
(`Deep_Object_Pose/train/train.py:1198-1206,1402`). Thus Stage B begins a fresh
1000-step DiffPnP ramp.

## 8. Legacy BPnP pseudocode: inactive and distinct

The following is the `--geo_loss` path, which is **off** in the ep57 header.

```python
def legacy_geometric_loss(pred_belief9, gt_belief9, epoch, warmup):
    # Deep_Object_Pose/train/geo_loss.py:84-143
    pred_kp = global_softargmax(pred_belief9, temperature=1.0)
    gt_kp = global_softargmax(gt_belief9, temperature=1.0)

    total = 1.0 * mse(pred_kp, gt_kp)
    total += 0.5 * projected_face_diagonal_midpoint_loss(pred_kp[:8])

    if epoch >= warmup:
        try:
            pred_pose, pred_valid = legacy_bpnp(pred_kp[:8])
            gt_pose, gt_valid = legacy_bpnp(detach(gt_kp[:8]))
            valid = pred_valid * gt_valid

            if any(valid):
                total += 0.1 * reprojection_loss(...)
                total += 0.1 * volume_loss(...)   # rigid-invariant; zero-gradient
                total += 0.1 * ADD_loss(...)
        except Exception:
            pass                                # silently removes PnP terms

    return total


class legacy_bpnp(torch.autograd.Function):
    def forward(kp2d, fixed_kp3d, fixed_K):
        # Deep_Object_Pose/train/geo_loss_bpnp.py:116-147
        kp2d_cpu = detach(kp2d).cpu().numpy()
        for frame:
            success, rvec, tvec = cv2.solvePnP(
                fixed_kp3d,
                kp2d_cpu[frame],
                fixed_K,
                distortion=None,
                flags=SOLVEPNP_EPNP,
            )
        save_tensors_for_backward(...)
        return pose, valid

    def backward(grad_pose):
        # Deep_Object_Pose/train/geo_loss_bpnp.py:149-177
        J = approximate_projection_jacobian(detach(saved_pose))
        inverse = pinv(J.T @ J + max(configured_damping, 1e-3) * I)
        grad_kp2d = J @ inverse @ detach(grad_pose)
        grad_kp2d *= valid
        grad_kp2d = clamp(nan_to_num(grad_kp2d), -10, +10)
        return grad_kp2d
```

This legacy forward/backward pair has a recorded finite-difference relative
error of `0.5428977425` at epsilon `0.01`. A sweep over epsilon
`0.1, 0.05, 0.01, 0.005, 0.001` stays between approximately `0.537` and
`0.545`; its best result is `0.5368470071`, still well above the `1e-2` gate.
The mismatch is therefore not a single-step-size artifact. The current
unrolled-GN DiffPnP3D audit records `1.01439358e-9`; the two methods must not
share one result label.

## 9. Canonical evaluation pseudocode: intentionally different

Checkpoint selection reuses `diffpnp3d_q1_eval.py` verbatim
(`scripts/stage0/diffpnp3d_val_select.py:4-10,91-101`).

```python
def canonical_eval_frame(image_640x480, ep57_state):
    # scripts/stage0/diffpnp3d_q1_eval.py:55-69
    model = DopeNetwork(numSeg=0)
    load_state_dict(ep57_state, strict=False)  # ignores saved segmentation keys

    input_400 = imagenet_normalize(
        cv2.resize(image_640x480, (400, 400))
    )
    belief6 = model(input_400).belief[-1]      # [9,50,50]

    # scripts/data_prep/eval/filter_pr_camfacing.py:124-156
    for channel in 0..8:
        if max(belief6[channel]) < 0.3:
            keypoint[channel] = MISSING
            continue

        smoothed = gaussian_filter(belief6[channel], sigma=2)
        candidates = four_neighbor_local_maxima(smoothed, threshold=0.3)
        peak = candidate_with_largest_raw_belief_value(candidates)
        keypoint_grid[channel] = (
            raw_weighted_centroid(belief6[channel], window=11, center=peak)
            + 0.4395
        )

        keypoint_original[channel] = (
            keypoint_grid[channel].x * 12.8,
            keypoint_grid[channel].y * 9.6,
        )

    if at_least_6_of_9_correspondences_exist:
        candidates = []
        for dims in [(W,D,H), (D,W,H)]:
            candidates += solve_with_IPPE_EPNP_SQPNP_manual_seeds_symmetry_LM(
                available_corners_and_optional_centroid,
                original_K,
                dims,
            )
        pose = rank_by_strict_invariants_then_reprojection(candidates)
    else:
        pose = failure

    return pose
```

Training and evaluation agree on the 400-square input preprocessing and the
axis-specific inverse scale. They do not agree on the keypoint decoder, missing
point policy, centroid use, pose initialization, solver, discrete hypotheses,
or final objective.

The current paper evaluator may also call a fail-closed `>=7` safe solver
(`scripts/stage0/paper_s2_rgb1_eval.py:519-536`;
`challenge/scripts/annotate_pnp.py:1034-1123`). That solver can use learned
scalar uncertainty, but canonical ep57 has no corner-quality head and therefore
provides no uncertainty.

## 10. Covariance status in executable terms

```python
# What exists:
coords, conf = local_softargmax_7x7(belief)
conf["cov_xy"]       # computed correctly
conf["var_x"]        # computed correctly
conf["var_y"]        # computed correctly
requires_grad(conf) == False

# What canonical training does:
pred_xy, _conf = local_softargmax_7x7(belief)
discard(_conf)
diffpnp3d_loss(pred_xy, ...)

# What canonical evaluation does:
keypoints = scipy_smooth_nms_weighted_centroid(belief)
pose = unweighted_deployment_solver(keypoints, ...)

# What is not implemented for ep57:
Sigma_2x2 = differentiable_heatmap_covariance(...)
mahalanobis_residual = r.T @ inverse(Sigma_2x2) @ r
weighted_unrolled_GN_using_Sigma_2x2(...)
full_covariance_deployment_solver(...)
```

Therefore the covariance unit gate is a **diagnostic-only** verification. It
must not be counted as an ep57 model contribution or ablation.
