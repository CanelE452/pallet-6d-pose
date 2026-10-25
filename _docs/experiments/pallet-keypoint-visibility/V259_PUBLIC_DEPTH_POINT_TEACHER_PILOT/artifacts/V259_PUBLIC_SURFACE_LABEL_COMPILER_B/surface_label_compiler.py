"""Pure NumPy public front-surface AUX labels; no I/O, models or pose API.

Labels are policy-derived states at r, never native gold or a model abstention.
Emulated q SAM pixel presence is a separate observation and receives no D/E label.
The caller must separately bind the original policy bytes to POLICY_SHA256.
"""
import numpy as np

POLICY_SHA256 = '793d580f602768c02ddebedf174f6d84d6b1d0dcbc04e71e33cb6afedf72f3e8'
EXPECTED_POLICY = {'created_UTC': '2026-10-25T13:22:57.696437+00:00', 'purpose': 'Conservative public surface-point AUX label prototype, not native gold and not a model fit.', 'basis': 'Explicit user authorization of open real data; first public unoccluded geometry metrology only, no current/held/SourceDEV model scores. Frozen before decoding second occluded pilot.', 'pose_interpretation': 'T object-to-camera as publisher 3D-model description; inverse also retained in geometry report. No optimization or bias correction.', 'parameters': {'radius_pixels': 5, 'observed_window_max_minus_min_m': 0.04, 'render_window_max_minus_min_m': 0.04, 'D_abs_observed_minus_render_at_r_max_m': 0.05, 'E_observed_minus_render_at_r_max_m': -0.15, 'reference_grid_stride': 8, 'reference_grid_offset': 4, 'q_offsets_xy': [[0, 0], [3, 0], [-3, 0], [0, 3], [0, -3], [6, 0], [-6, 0], [0, 6], [0, -6]], 'visual_local_L_pixels': 96}, 'label_definition': {'D': 'Full11x11 valid observed/render support and stable depth windows, full11x11 SAM target support, abs(center observed-render)<=0.05m.', 'E': 'Same stable valid depth/render windows, full11x11 SAM background, center observed-render<=-0.15m.', 'U': 'Everything else, with exclusion reasons; not a correct model abstention.', 'SELF': 'Not inferred from mask-negative. These reference samples use the front rendered target surface, not hidden mesh vertices.', 'r_vs_q': 'Label at reference r and SAM presence at emulated q must be stored separately. q offsets are public emulation, not fresh v61/initializer inference. No relabeling of r D/E as query-pixel gold.', 'native': 'Generic surfaceAUX, no new eight-slot native gold or physical native-support requirement.'}, 'no_auto_training_or_adoption': True, 'no_actual_private223_or_protected_case_payloads': True, 'no_tuning_after_pilot_label_counts': True, 'raw_depth_residuals_preserved_no_translation_shift': True}


def _same_typed(actual, expected):
    if type(actual) is not type(expected): return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_same_typed(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same_typed(a, b) for a, b in zip(actual, expected))
    return actual == expected


def validate_policy(policy):
    """Reject any added/removed/changed policy field, including numeric type."""
    if not _same_typed(policy, EXPECTED_POLICY):
        raise ValueError('Exact frozen V259 policy object required; no threshold or policy overrides')
    return policy['parameters']


def _inputs(observed_m, rendered_m, sam_bool):
    if not all(isinstance(a, np.ndarray) and a.ndim == 2 for a in (observed_m, rendered_m, sam_bool)):
        raise ValueError('Explicit two-dimensional NumPy depth/depth/SAM arrays required')
    shape = observed_m.shape
    if min(shape) <= 0 or rendered_m.shape != shape or sam_bool.shape != shape:
        raise ValueError('Same nonempty HxW shape required')
    if observed_m.dtype not in (np.dtype('float32'), np.dtype('float64')) or rendered_m.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Observed/rendered depths must be floating meters, not integer encoded depth')
    if sam_bool.dtype != np.dtype(bool): raise ValueError('SAM bool only; no implicit thresholding')
    return shape


def compile_surface_labels(observed_m, rendered_m, sam_bool, policy):
    """Return labels[int8 H,W], raw residual, boolean maps and JSON-ready counts.

    Both depth windows must be fully positive finite and separately stable.
    All thresholds are inclusive and use uncorrected FP64 differences of the
    supplied floating-meter values. No rounding tolerance or bias is inserted.
    The full11x11 square is required; edges remain U. Reason maps can overlap.
    """
    p = validate_policy(policy)
    shape = _inputs(observed_m, rendered_m, sam_bool)
    observed = observed_m.astype(np.float64, copy=False)
    rendered = rendered_m.astype(np.float64, copy=False)
    with np.errstate(invalid='ignore'):
        residual = observed - rendered
    radius = p['radius_pixels']; side = 2 * radius + 1
    keys = ('full_window_in_image', 'observed_positive_finite', 'render_positive_finite',
            'observed_stable', 'render_stable', 'sam_all_target', 'sam_all_background')
    masks = {k: np.zeros(shape, dtype=bool) for k in keys}
    if shape[0] >= side and shape[1] >= side:
        inner = (slice(radius, shape[0]-radius), slice(radius, shape[1]-radius))
        masks['full_window_in_image'][inner] = True
        window = np.lib.stride_tricks.sliding_window_view
        obs_w = window(observed, (side, side))
        ren_w = window(rendered, (side, side))
        sam_w = window(sam_bool, (side, side))
        obs_valid = (np.isfinite(obs_w) & (obs_w > 0)).all(axis=(-2, -1))
        ren_valid = (np.isfinite(ren_w) & (ren_w > 0)).all(axis=(-2, -1))
        with np.errstate(invalid='ignore'):
            obs_range = obs_w.max(axis=(-2, -1)) - obs_w.min(axis=(-2, -1))
            ren_range = ren_w.max(axis=(-2, -1)) - ren_w.min(axis=(-2, -1))
        masks['observed_positive_finite'][inner] = obs_valid
        masks['render_positive_finite'][inner] = ren_valid
        masks['observed_stable'][inner] = obs_valid & (obs_range <= p['observed_window_max_minus_min_m'])
        masks['render_stable'][inner] = ren_valid & (ren_range <= p['render_window_max_minus_min_m'])
        masks['sam_all_target'][inner] = sam_w.all(axis=(-2, -1))
        masks['sam_all_background'][inner] = ~sam_w.any(axis=(-2, -1))
    inside = masks['full_window_in_image']
    depth = inside & masks['observed_positive_finite'] & masks['render_positive_finite'] & masks['observed_stable'] & masks['render_stable']
    d_close = np.abs(residual) <= p['D_abs_observed_minus_render_at_r_max_m']
    e_closer = residual <= p['E_observed_minus_render_at_r_max_m']
    d = depth & masks['sam_all_target'] & d_close
    e = depth & masks['sam_all_background'] & e_closer
    if np.any(d & e): raise AssertionError('D/E masks must be disjoint')
    labels = np.full(shape, -1, dtype=np.int8)
    labels[d] = 0; labels[e] = 1
    masks.update(depth_eligible=depth, D=d, E=e, known=d | e)
    reasons = {
      'image_edge': ~inside,
      'observed_invalid': inside & ~masks['observed_positive_finite'],
      'render_invalid': inside & ~masks['render_positive_finite'],
      'observed_discontinuity': inside & masks['observed_positive_finite'] & ~masks['observed_stable'],
      'render_discontinuity': inside & masks['render_positive_finite'] & ~masks['render_stable'],
      'sam_window_mixed': inside & ~masks['sam_all_target'] & ~masks['sam_all_background'],
      'D_residual_not_close': depth & masks['sam_all_target'] & ~d_close,
      'E_residual_not_closer': depth & masks['sam_all_background'] & ~e_closer,
    }
    union = np.zeros(shape, dtype=bool)
    for reason in reasons.values(): union |= reason
    if not np.array_equal(union, labels == -1): raise AssertionError('Every U needs at least one exclusion reason')
    counts = {'height': int(shape[0]), 'width': int(shape[1]), 'total_pixels': int(labels.size),
      'labels': {'D': int(d.sum()), 'E': int(e.sum()), 'U': int((labels == -1).sum())},
      'eligibility': {k: int(v.sum()) for k, v in masks.items()},
      'reasons': {k: int(v.sum()) for k, v in reasons.items()},
      'reason_maps_overlap_do_not_sum_as_U_denominator': True,
      'SELF_inferred': False, 'U_is_model_abstention': False,
      'native_gold': False, 'raw_residual_bias_correction_applied': False,
      'policy_sha256': POLICY_SHA256}
    return {'labels': labels, 'raw_center_residual_m': residual,
            'eligibility': masks, 'reasons': reasons, 'counts': counts}


def reference_grid_xy(shape_hw, policy):
    """All fixed public reference grid positions, y-major; retain edge U rows."""
    p = validate_policy(policy)
    if not isinstance(shape_hw, (tuple, list)) or len(shape_hw) != 2 or any(type(x) is not int or x <= 0 for x in shape_hw):
        raise ValueError('Positive integer H,W required')
    h, w = shape_hw
    xs = np.arange(p['reference_grid_offset'], w, p['reference_grid_stride'], dtype=np.int64)
    ys = np.arange(p['reference_grid_offset'], h, p['reference_grid_stride'], dtype=np.int64)
    x, y = np.meshgrid(xs, ys, indexing='xy')
    return np.stack((x, y), axis=-1).reshape(-1, 2)


def _compiled_shape(compiled):
    labels = compiled['labels']
    if not isinstance(labels, np.ndarray) or labels.ndim != 2 or labels.dtype != np.int8 or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError('Compiled int8 D0/E1/Uminus1 map required')
    if compiled['raw_center_residual_m'].shape != labels.shape: raise ValueError('Raw r residual shape')
    for family in ('eligibility', 'reasons'):
        if not all(isinstance(a, np.ndarray) and a.dtype == bool and a.shape == labels.shape for a in compiled[family].values()):
            raise ValueError('Compiled bool maps must share H,W')
    return labels.shape


def reference_grid_records(compiled, policy):
    """JSON-ready r-only rows. Nonfinite raw residual is null, not corrected."""
    grid = reference_grid_xy(_compiled_shape(compiled), policy)
    state = {-1: 'U', 0: 'D', 1: 'E'}
    records = []
    for i, (x, y) in enumerate(grid):
        x, y = int(x), int(y); label = int(compiled['labels'][y, x])
        residual = float(compiled['raw_center_residual_m'][y, x])
        records.append({'reference_grid_index': i, 'r_xy': [x, y],
          'r_derived_label': label, 'r_derived_state': state[label],
          'r_depth_eligible': bool(compiled['eligibility']['depth_eligible'][y, x]),
          'r_known': label >= 0,
          'r_raw_observed_minus_render_m': residual if np.isfinite(residual) else None,
          'r_raw_residual_finite': bool(np.isfinite(residual)),
          'r_exclusion_reasons': [k for k, value in compiled['reasons'].items() if value[y, x]],
          'native_slot': None, 'SELF_state': 'UNKNOWN'})
    return records


def emulated_query_records(compiled, sam_bool, policy):
    """Grid×9 fixed q offsets; q receives SAM presence only, never r D/E gold."""
    p = validate_policy(policy); h, w = _compiled_shape(compiled)
    if not isinstance(sam_bool, np.ndarray) or sam_bool.shape != (h, w) or sam_bool.dtype != bool:
        raise ValueError('Original bool SAM map shape required')
    records = []
    for reference in reference_grid_records(compiled, policy):
        x, y = reference['r_xy']
        for k, (dx, dy) in enumerate(p['q_offsets_xy']):
            qx, qy = x + dx, y + dy
            in_image = 0 <= qx < w and 0 <= qy < h
            records.append(dict(reference,
              record_id='r_y%d_x%d_q%d' % (y, x, k), q_offset_index=k,
              q_offset_xy=[dx, dy], q_xy=[qx, qy], q_inside_image=in_image,
              q_sam_target_present=bool(sam_bool[qy, qx]) if in_image else None,
              q_pixel_DE_gold=None, r_label_copied_as_query_label=False,
              q_is_public_emulation=True, q_is_actual_initializer=False,
              q_local_L_pixels=p['visual_local_L_pixels'],
              U_is_model_abstention=False))
    return records
