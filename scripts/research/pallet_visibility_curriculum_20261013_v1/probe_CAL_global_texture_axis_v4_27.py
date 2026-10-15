"""Predeclared CAL-only global texture-axis feasibility probe, never a runtime cue.

``plan`` reads source files only. Both ``prepare`` and ``probe`` require an
explicit root GO and the immutable plan. No model inference, pose optimization,
S/U local point query, TEST/remainder image, or TEST/remainder reference is used.
Reference-versus-transposed-patch discrimination is an artificial control, not
proof of physical WD phase generalization. All CAL46 rows, including abstentions
and the original two native incompatibilities, remain in the report.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import common as C


BASE = C.DOC / 'nonsevere_v1'
BOUNDARY = BASE / 'statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
OUTPUT = BOUNDARY / 'CAL_global_texture_axis_probe_v4_27'
CAL_PARENT = BOUNDARY / 'shared_edge_partial_b_eval_v4_7'
MASKS = BOUNDARY / 'image_evidence_v4_6'
LABELS = BASE / 'OOF_pose_v1/ISOLATED_CAL_SCORING.json'
ALIGN_SOURCE = Path(__file__).with_name('statewise_uncertainty_calibration_v4.py')

RECIPE = dict(
    schema='CAL_GLOBAL_TEXTURE_AXIS_FEASIBILITY_RECIPE_V4_27', CAL=46,
    original_native_compatible_expected=44, original_sessions_expected=4,
    population='all CAL46; incompatible2 and every abstention retained',
    reference='derived CAL body pose and registered physical XYZ; not independent metrology',
    physical_plane='physical -Y top face, centered registered model; original mean/Q and body center explicit',
    geometry_matching_tolerance_m=1e-6, rotation_tolerance=1e-7,
    top_face_front_angle_min_deg=2., minimum_projected_area_px2=256.,
    grid_long_side_nodes=192, minimum_grid_side_nodes=24,
    grid='isotropic physical X/Z spacing; rows +Z, columns +X',
    interior_margin_fraction=.10, mask_occupancy_threshold=.5,
    minimum_foreground_fraction=.60, minimum_gradient_samples=256,
    minimum_gradient_rms_gray_per_grid_px=1./255., minimum_tensor_anisotropy=.10,
    gradient='central differences with all four neighboring pixels in foreground/interior/canvas',
    features=['log((Gz+1e-12)/(Gx+1e-12))', 'periodicity_strength_Z-minus-X'],
    periodicity='masked Pearson correlation by displacement, positive interior peak prominence / 2',
    periodicity_lag_min_nodes=4, periodicity_lag_max_nodes=64,
    periodicity_max_axis_fraction=1./3., periodicity_min_pairs=128,
    periodicity_min_pair_fraction=.25,
    template_features=['tanh(log_gradient_ratio/2)', 'periodicity_strength_Z-minus-X'],
    template='session-equal mean of eligible frame feature vectors; squared Euclidean distance',
    minimum_template_frames=2, minimum_template_sessions=2, score_tie_tolerance=1e-12,
    controls=['exact X/Z grid transpose', '180-degree grid rotation (C2)'],
    LOGO='each training centroid excludes every frame in held capture session',
    runtime_session_or_frame_ID_feature=False, material_rule=False, scene3_X_rule=False,
    eligibility_or_feature_parameters_fitted=False, calibrated_likelihood=False,
    synthetic_axis_control_is_independent_GT_phase_classification=False,
    single_wood_session_is_material_generalization_evidence=False,
    S_or_U_raw_2D_local_RGB_queries=0, new_NN_or_pose_optimizer_calls=0,
    TEST_or_REMAINDER_images_or_references=0, default_adoption=False,
    historical_CAL_exposure=True, upstream_full_CAL_fitted_masks_are_conditional=True,
)


class Abstain(ValueError):
    """A retained frame has insufficient or unregistered top-face evidence."""


def _digest(value):
    return hashlib.sha256(json.dumps(C.finite(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve earlier probe artifacts: ' + str(path))
    C.write(path, value)


def _rotation(value, name):
    a = np.asarray(value, dtype=float)
    if (a.shape != (3, 3) or not np.isfinite(a).all()
            or not np.allclose(a.T @ a, np.eye(3), atol=RECIPE['rotation_tolerance'], rtol=0)
            or not np.isclose(np.linalg.det(a), 1., atol=RECIPE['rotation_tolerance'], rtol=0)):
        raise Abstain('INVALID_PROPER_ROTATION_' + name)
    return a


def physical_top_face(geometry, truth):
    """Only 3D geometry/reference fields; no states, 2D KP, or image is an input.

    Body coordinates are centered by construction. The registered model may
    have a nonzero native origin. Its mean is subtracted before placing it at
    the explicitly reconstructed reference body center.
    """
    xyz = np.asarray(geometry['native_xyz'], dtype=float)
    if xyz.shape != (8, 3) or not np.isfinite(xyz).all():
        raise Abstain('INVALID_NATIVE_CUBOID')
    Q = _rotation(geometry['physical_registration_Q'], 'Q')
    R = _rotation(truth['R'], 'REFERENCE_PHYSICAL')
    body_R = _rotation(truth['body_R'], 'REFERENCE_BODY')
    dims_body = np.asarray(truth['body_xyz'], dtype=float)
    t = np.asarray(truth['t'], dtype=float)
    K = np.asarray(geometry['K'], dtype=float)
    if (dims_body.shape != (3,) or not np.isfinite(dims_body).all() or np.any(dims_body <= 0)
            or t.shape != (3,) or not np.isfinite(t).all() or K.shape != (3, 3)
            or not np.isfinite(K).all() or abs(np.linalg.det(K)) < 1e-12):
        raise Abstain('INVALID_REFERENCE_OR_INTRINSICS')
    physical = xyz @ Q
    center = physical.mean(axis=0)
    centered = physical - center
    dims = np.ptp(centered, axis=0)
    expected = C.cuboid(*dims)
    body = C.cuboid(*dims_body)
    physical_body = body @ (body_R.T @ R)
    gaps = []
    for other in (expected, physical_body):
        distance = np.linalg.norm(centered[:, None] - other[None], axis=-1)
        ii, jj = linear_sum_assignment(distance)
        gaps.append(float(np.max(distance[ii, jj])))
    if np.any(dims <= 0) or max(gaps) > RECIPE['geometry_matching_tolerance_m']:
        raise Abstain('PHYSICAL_XYZ_OR_UNORDERED_REFERENCE_REGISTRATION_UNCONFIRMED')
    if dims[1] >= min(dims[0], dims[2]):
        raise Abstain('PHYSICAL_Y_IS_NOT_CONFIRMED_HEIGHT_AXIS')
    # Source convention shared with top_plane_normal_factor_v4_16: top normal -Y.
    top = np.array([[-dims[0]/2, -dims[1]/2, -dims[2]/2],
                    [dims[0]/2, -dims[1]/2, -dims[2]/2],
                    [dims[0]/2, -dims[1]/2, dims[2]/2],
                    [-dims[0]/2, -dims[1]/2, dims[2]/2]])
    reference_center = body.mean(axis=0) @ body_R.T + t
    camera = top @ R.T + reference_center
    if np.any(camera[:, 2] <= 1e-9):
        raise Abstain('TOP_FACE_NONPOSITIVE_DEPTH')
    top_center_camera = top.mean(axis=0) @ R.T + reference_center
    cosine = float((R @ np.array([0., -1., 0.])) @ (-top_center_camera) / np.linalg.norm(top_center_camera))
    if cosine <= np.sin(np.deg2rad(RECIPE['top_face_front_angle_min_deg'])):
        raise Abstain('TOP_FACE_BACK_FACING_OR_GRAZING')
    homogeneous = camera @ K.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-9):
        raise Abstain('INVALID_HOMOGENEOUS_TOP_PROJECTION')
    quad = homogeneous[:, :2] / homogeneous[:, 2:]
    area = abs(float(cv2.contourArea(quad.astype(np.float32))))
    if area < RECIPE['minimum_projected_area_px2']:
        raise Abstain('TOP_FACE_PIXEL_AREA_TOO_SMALL')
    step = float(max(dims[0], dims[2]) / (RECIPE['grid_long_side_nodes'] - 1))
    nx = int(np.floor(dims[0] / step + 1e-8)) + 1
    nz = int(np.floor(dims[2] / step + 1e-8)) + 1
    if min(nx, nz) < RECIPE['minimum_grid_side_nodes']:
        raise Abstain('METRIC_GRID_SHORT_SIDE_TOO_SMALL')
    return dict(projected_quad_xy=quad, dimensions_physical_xyz_m=dims,
        native_mean=xyz.mean(axis=0), registered_physical_mean=center,
        reference_body_center_camera_m=reference_center, reference_R_physical=R,
        native_Q=Q, unordered_geometry_gaps_m=gaps, top_front_cosine=cosine,
        projected_top_area_px2=area, grid_hw=[nz, nx], metric_step_m=step,
        full_face_metric_lengths_xz_m=[float(dims[0]), float(dims[2])])


def rectify_global_top(image_bgr, pallet_mask, geometry, truth):
    """Warp the entire reference plane, never centered on an S/U keypoint."""
    image = np.asarray(image_bgr)
    mask = np.asarray(pallet_mask)
    hw = tuple(int(v) for v in geometry['image_hw'])
    if (image.shape != (*hw, 3) or image.dtype != np.uint8 or mask.shape != hw
            or not np.isfinite(mask).all()):
        raise Abstain('IMAGE_OR_MASK_SHAPE_OR_VALUES_INVALID')
    face = physical_top_face(geometry, truth)
    nz, nx = face['grid_hw']
    # A common physical spacing is maintained: the last row/column can stop
    # less than one sample before the true face boundary instead of stretching.
    lx, lz = face['full_face_metric_lengths_xz_m']
    dest = np.array([[0., 0.], [lx/face['metric_step_m'], 0.],
                     [lx/face['metric_step_m'], lz/face['metric_step_m']],
                     [0., lz/face['metric_step_m']]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(np.asarray(face['projected_quad_xy'], np.float32), dest)
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-15:
        raise Abstain('TOP_HOMOGRAPHY_SINGULAR')
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float) / 255.
    patch = cv2.warpPerspective(grey, H, (nx, nz), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    foreground = cv2.warpPerspective((mask >= RECIPE['mask_occupancy_threshold']).astype(np.uint8),
        H, (nx, nz), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) != 0
    canvas = cv2.warpPerspective(np.ones(hw, dtype=np.uint8), H, (nx, nz),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) != 0
    # Derivative stencils must also remain inside the actual image.
    canvas = cv2.erode(canvas.astype(np.uint8), np.ones((3, 3), np.uint8)) != 0
    xx = np.arange(nx) * face['metric_step_m'] / lx
    zz = np.arange(nz) * face['metric_step_m'] / lz
    margin = RECIPE['interior_margin_fraction']
    interior = ((zz[:, None] >= margin) & (zz[:, None] <= 1-margin)
                & (xx[None, :] >= margin) & (xx[None, :] <= 1-margin))
    fraction = float(np.sum(foreground & canvas & interior) / max(1, int(interior.sum())))
    if fraction < RECIPE['minimum_foreground_fraction']:
        raise Abstain('INSUFFICIENT_TOP_FOREGROUND_OR_CANVAS_COVERAGE')
    return patch, foreground & canvas & interior, dict(C.finite(face),
        image_to_metric_grid_H=H.tolist(), foreground_interior_fraction=fraction,
        mask_is_frozen_global_occupancy_not_S_or_U_visibility=True)


def _axis_periodicity(grey, valid, axis):
    length = grey.shape[axis]
    maximum = min(RECIPE['periodicity_lag_max_nodes'],
                  int(np.floor(length * RECIPE['periodicity_max_axis_fraction'])))
    minimum_pairs = max(RECIPE['periodicity_min_pairs'],
                        RECIPE['periodicity_min_pair_fraction'] * int(valid.sum()))
    points = []
    for lag in range(1, maximum + 1):
        first = [slice(None), slice(None)]
        second = first.copy()
        first[axis] = slice(None, -lag)
        second[axis] = slice(lag, None)
        first, second = tuple(first), tuple(second)
        ok = valid[first] & valid[second]
        count = int(ok.sum())
        correlation = None
        if count >= minimum_pairs:
            a, b = grey[first][ok], grey[second][ok]
            a, b = a-a.mean(), b-b.mean()
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denominator > 1e-12:
                correlation = float(np.clip(a @ b / denominator, -1., 1.))
        points.append(dict(lag_nodes=lag, pairs=count, correlation=correlation))
    strength, period = 0., None
    for i in range(1, len(points)-1):
        c = points[i]['correlation']
        before, after = points[i-1]['correlation'], points[i+1]['correlation']
        if (points[i]['lag_nodes'] < RECIPE['periodicity_lag_min_nodes']
                or c is None or before is None or after is None or c <= 0
                or c <= before or c < after):
            continue
        history = [p['correlation'] for p in points[:i] if p['correlation'] is not None]
        prominence = max(0., (c-min(history))/2.)
        if prominence > strength:
            strength, period = float(prominence), points[i]['lag_nodes']
    return dict(strength=strength, period_nodes=period, correlations=points)


def texture_features(grey, valid, *, metric_step_m=1.):
    """Pure global patch descriptors; rows +Z and columns +X. No IDs or states."""
    grey = np.asarray(grey, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if (grey.ndim != 2 or grey.shape != valid.shape or min(grey.shape) < 3
            or not np.isfinite(grey).all() or not np.isfinite(metric_step_m) or metric_step_m <= 0):
        raise Abstain('INVALID_RECTIFIED_GLOBAL_PATCH')
    stencil = (valid[1:-1, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
               & valid[:-2, 1:-1] & valid[2:, 1:-1])
    count = int(stencil.sum())
    if count < RECIPE['minimum_gradient_samples']:
        raise Abstain('INSUFFICIENT_GLOBAL_GRADIENT_STENCILS')
    dx = ((grey[1:-1, 2:]-grey[1:-1, :-2])/2)[stencil]
    dz = ((grey[2:, 1:-1]-grey[:-2, 1:-1])/2)[stencil]
    rms = float(np.sqrt(np.mean(dx**2+dz**2)))
    if rms < RECIPE['minimum_gradient_rms_gray_per_grid_px']:
        raise Abstain('INSUFFICIENT_GLOBAL_TEXTURE_ENERGY')
    tensor_px = np.array([[np.mean(dx**2), np.mean(dx*dz)],
                          [np.mean(dx*dz), np.mean(dz**2)]])
    eigen = np.linalg.eigvalsh(tensor_px)
    anisotropy = float((eigen[1]-eigen[0])/max(1e-12, eigen.sum()))
    if anisotropy < RECIPE['minimum_tensor_anisotropy']:
        raise Abstain('GLOBAL_TEXTURE_GRADIENT_ISOTROPIC')
    tensor = tensor_px/float(metric_step_m)**2
    gx, gz = float(tensor[0, 0]), float(tensor[1, 1])
    ratio = float(np.log((gz+1e-12)/(gx+1e-12)))
    x, z = _axis_periodicity(grey, valid, 1), _axis_periodicity(grey, valid, 0)
    difference = float(z['strength']-x['strength'])
    return dict(log_gradient_energy_Z_over_X=ratio,
        periodicity_strength_Z_minus_X=difference, gradient_energy_X=gx, gradient_energy_Z=gz,
        gradient_tensor_physical_XZ=tensor.tolist(), gradient_rms_gray_per_grid_px=rms,
        tensor_anisotropy=anisotropy, gradient_samples=count,
        periodicity_X=x, periodicity_Z=z,
        template_vector=[float(np.tanh(ratio/2.)), difference],
        interpretation='positive Z variation suggests X-long stripes; gradient axis is not slat length axis')


def describe_controls(grey, valid, *, metric_step_m=1.):
    """Both controls are exact raster transformations; no new RGB observation."""
    reference = texture_features(grey, valid, metric_step_m=metric_step_m)
    swapped = texture_features(np.asarray(grey).T, np.asarray(valid).T, metric_step_m=metric_step_m)
    c2 = texture_features(np.asarray(grey)[::-1, ::-1], np.asarray(valid)[::-1, ::-1], metric_step_m=metric_step_m)
    expected = -np.asarray(reference['template_vector'])
    if (not np.allclose(swapped['template_vector'], expected, atol=1e-9, rtol=0)
            or not np.allclose(c2['template_vector'], reference['template_vector'], atol=1e-9, rtol=0)):
        raise ValueError('Axis-exchange/C2 descriptor invariance failed')
    return dict(reference=reference, grid_axis_exchange=swapped, C2_grid_rotation=c2,
        exact_grid_axis_exchange_not_second_physical_pose_render=True)


def fit_template(records, *, heldout_session=None):
    records = list(records)
    if any(r.get('partition') != 'CAL' for r in records):
        raise ValueError('CAL-only template; no TEST or remainder record')
    training = [r for r in records if r['session'] != heldout_session]
    eligible = [r for r in training if r['supported']]
    sessions = sorted({r['session'] for r in eligible})
    model = dict(heldout_session=heldout_session, training_ids=[r['id'] for r in training],
        eligible_training_ids=[r['id'] for r in eligible], eligible_training_sessions=sessions,
        sufficient=False, template_vector=None, feature_or_eligibility_parameter_fit=False)
    if len(eligible) < RECIPE['minimum_template_frames'] or len(sessions) < RECIPE['minimum_template_sessions']:
        return model
    means = [np.mean([r['controls']['reference']['template_vector'] for r in eligible if r['session'] == s], axis=0)
             for s in sessions]
    model.update(sufficient=True, template_vector=np.mean(means, axis=0).tolist(),
                 session_mean_vectors={s:v.tolist() for s, v in zip(sessions, means)})
    return model


def compare_template(record, model):
    if not record['supported'] or not model['sufficient']:
        return dict(id=record['id'], session=record['session'], verdict='ABSTAIN',
                    reason=record.get('reason', 'INSUFFICIENT_TRAINING_TEMPLATE'))
    mean = np.asarray(model['template_vector'])
    r, s = (np.asarray(record['controls'][k]['template_vector'])
            for k in ('reference', 'grid_axis_exchange'))
    costs = [float(np.sum((v-mean)**2)) for v in (r, s)]
    delta = costs[1]-costs[0]
    tolerance = RECIPE['score_tie_tolerance']
    verdict = 'REFERENCE' if delta > tolerance else 'AXIS_EXCHANGE' if delta < -tolerance else 'TIE'
    return dict(id=record['id'], session=record['session'], verdict=verdict,
                reference_distance2=costs[0], exchanged_distance2=costs[1], margin=delta,
                artificial_axis_control_not_independent_phase_accuracy=True)


def _require_GO(root_go):
    if root_go is not True:
        raise PermissionError('Explicit root GO required before any CAL input, reference or image IO')


def write_plan(output=OUTPUT):
    """Source-only operation: does not read any experiment metadata or data."""
    output = Path(output).resolve()
    paths = [Path(__file__).resolve(), Path(C.__file__).resolve(), ALIGN_SOURCE,
             Path(__file__).with_name('test_probe_CAL_global_texture_axis_v4_27.py')]
    sources = {str(p.relative_to(C.ROOT)): C.sha(p) for p in paths}
    plan = dict(schema='CAL_GLOBAL_TEXTURE_AXIS_ANALYSIS_PLAN_V4_27', recipe=deepcopy(RECIPE),
        source_sha256=sources, intended_inputs=dict(
            CAL_observation_journal=str((CAL_PARENT/'CAL_PREDICTIONS.jsonl.gz').relative_to(C.ROOT)),
            CAL_prediction_seal=str((CAL_PARENT/'CAL_PREDICTION_SEAL.json').relative_to(C.ROOT)),
            masks_records=str((MASKS/'RECORDS.json').relative_to(C.ROOT)),
            masks_seal=str((MASKS/'SEAL.json').relative_to(C.ROOT)),
            isolated_CAL_references=str(LABELS.relative_to(C.ROOT))),
        first_stage='prepare after root GO: whitelist all CAL46 geometry/K/image/mask bindings; no reference numeric decode or pixels',
        second_stage='probe after same explicit GO and sealed preparation: CAL references and global top-face patches only',
        unresolved='physical axes are inherited reference metadata, not independently certified real-object part axes',
        no_scene3_direction_rule_or_runtime_likelihood=True, actual_CAL_numeric_or_image_reads_before_plan=0,
        actual_CAL_prepare_or_probe_executed=False, model_or_default_changed=False)
    _write_new(output/'ANALYSIS_PLAN.json', plan)
    for relative, digest in sources.items():
        dest = output/'code_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise FileExistsError('Preserve previous snapshot')
        shutil.copyfile(C.ROOT/relative, dest)
        if C.sha(dest) != digest:
            raise ValueError('Source snapshot changed')
    _write_new(output/'PLAN_SEAL.json', dict(plan_sha256=C.sha(output/'ANALYSIS_PLAN.json'),
        source_sha256=sources, actual_CAL_numeric_or_image_reads=0,
        root_GO_required_before_CAL_prepare_and_probe=True))
    return dict(output=str(output), plan_sha256=C.sha(output/'ANALYSIS_PLAN.json'))


def _verify_plan(output):
    output = Path(output).resolve()
    plan, seal = (C.read(output/name) for name in ('ANALYSIS_PLAN.json', 'PLAN_SEAL.json'))
    if plan['recipe'] != RECIPE or seal['plan_sha256'] != C.sha(output/'ANALYSIS_PLAN.json'):
        raise ValueError('Fixed pre-CAL analysis plan required')
    for relative, digest in plan['source_sha256'].items():
        if C.sha(C.ROOT/relative) != digest or C.sha(output/'code_snapshot'/relative) != digest:
            raise ValueError('Probe source binding changed')
    return plan


def prepare(output=OUTPUT, *, root_go=False):
    _require_GO(root_go)  # Must remain ahead of every input read and output existence check.
    output = Path(output).resolve()
    plan = _verify_plan(output)
    if (output/'PREPARATION_SEAL.json').exists():
        raise FileExistsError('Preserve previous CAL preparation')
    journal = CAL_PARENT/'CAL_PREDICTIONS.jsonl.gz'
    seal = C.read(CAL_PARENT/'CAL_PREDICTION_SEAL.json')
    if C.sha(journal) != seal['predictions_sha256']:
        raise ValueError('Frozen CAL-only observation bytes changed')
    maskseal, metadata = C.read(MASKS/'SEAL.json'), C.read(MASKS/'RECORDS.json')
    if (maskseal['records'] != 95 or maskseal['GT_or_pose_labels_read']
            or maskseal['records_sha256'] != C.sha(MASKS/'RECORDS.json')):
        raise ValueError('Original GT-free fixed masks required')
    masks_by = {r['id']:r for r in metadata}
    selected = []
    for row in C.rows(journal):
        if row['partition'] != 'CAL':
            raise ValueError('CAL-only input journal; reject other partition before geometry lookup')
        g, meta = row['runtime']['initial_geometry'], masks_by[row['id']]
        if meta['partition'] != 'CAL' or meta['image_sha256'] != row['image_sha256']:
            raise ValueError('CAL image/mask identity mismatch')
        selected.append(dict(id=row['id'], partition='CAL', session=row['session'],
            image=row['image'], image_sha256=row['image_sha256'],
            geometry={key:deepcopy(g[key]) for key in ('native_xyz', 'physical_registration_Q', 'K', 'image_hw')},
            mask_npz=str((MASKS/meta['mask_npz']).relative_to(C.ROOT)),
            mask_file_sha256=meta['mask_file_sha256']))
    if len(selected) != RECIPE['CAL'] or len({r['id'] for r in selected}) != 46 or len({r['session'] for r in selected}) != 4:
        raise ValueError('Unchanged all46/four-session population required')
    paths = [journal, CAL_PARENT/'CAL_PREDICTION_SEAL.json', MASKS/'SEAL.json', MASKS/'RECORDS.json', LABELS]
    bindings = {str(p.relative_to(C.ROOT)):C.sha(p) for p in paths}
    for r in selected:
        for path, digest in ((r['image'], r['image_sha256']), (r['mask_npz'], r['mask_file_sha256'])):
            if C.sha(C.ROOT/path) != digest:
                raise ValueError('Frozen selected CAL RGB/mask bytes changed')
            bindings[path] = digest
    _write_new(output/'CAL_INPUTS.json', dict(rows=selected))
    _write_new(output/'PROTOCOL.json', dict(schema='CAL_GLOBAL_TEXTURE_AXIS_PREFLIGHT_PROTOCOL_V4_27',
        recipe=RECIPE, ids=[r['id'] for r in selected], source_sha256=plan['source_sha256'], input_sha256=bindings,
        reference_numeric_values_or_images_decoded=False, nonCAL_mask_metadata_is_not_an_image_or_reference=True,
        CAL_input_fields='3D geometry/K/image identity only; no raw or observed keypoint/state fields'))
    _write_new(output/'PREPARATION_SEAL.json', dict(rows=46,
        plan_seal_sha256=C.sha(output/'PLAN_SEAL.json'), protocol_sha256=C.sha(output/'PROTOCOL.json'),
        CAL_inputs_sha256=C.sha(output/'CAL_INPUTS.json'), before_probe_reference_and_image_decode=True))
    return dict(prepared=46, output=str(output))


def _count_verdicts(rows):
    return {name:sum(r['verdict'] == name for r in rows)
            for name in ('REFERENCE', 'AXIS_EXCHANGE', 'TIE', 'ABSTAIN')}


def probe(output=OUTPUT, *, root_go=False):
    _require_GO(root_go)
    output = Path(output).resolve()
    _verify_plan(output)
    if (output/'PROBE_SEAL.json').exists() or (output/'PROBE_REPORT.json').exists():
        raise FileExistsError('Preserve every completed or partial probe')
    protocol, seal = (C.read(output/name) for name in ('PROTOCOL.json', 'PREPARATION_SEAL.json'))
    if (seal['protocol_sha256'] != C.sha(output/'PROTOCOL.json')
            or seal['CAL_inputs_sha256'] != C.sha(output/'CAL_INPUTS.json')
            or seal['plan_seal_sha256'] != C.sha(output/'PLAN_SEAL.json')):
        raise ValueError('Sealed all-CAL preparation required')
    for relative, digest in protocol['input_sha256'].items():
        if C.sha(C.ROOT/relative) != digest:
            raise ValueError('Frozen CAL input changed')
    inputs = C.read(output/'CAL_INPUTS.json')['rows']
    if len(inputs) != 46 or [r['id'] for r in inputs] != protocol['ids'] or any(r['partition'] != 'CAL' for r in inputs):
        raise ValueError('Exactly every CAL46 input is required')
    from . import statewise_uncertainty_calibration_v4 as ALIGN
    labels = C.read(LABELS)['rows']
    if set(labels) != set(protocol['ids']):
        raise ValueError('Isolated exact CAL references only')
    records, patch_bindings = [], {}
    for index, item in enumerate(inputs):
        label = labels[item['id']]
        if label['metadata']['partition'] != 'CAL':
            raise ValueError('Reject non-CAL label before target lookup')
        alignment = ALIGN._reference_alignment({'runtime':{'initial_geometry':item['geometry']}}, label)
        record = dict(id=item['id'], partition='CAL', session=item['session'], supported=False,
            fixed_native_compatible=bool(alignment['compatible']), fixed_native_phase=alignment['phase'],
            fixed_native_geometry_gap_m=alignment['geometry_gap_m'], S_or_U_local_point_queries=0)
        try:
            image = cv2.imread(str(C.ROOT/item['image']), cv2.IMREAD_COLOR)
            if image is None:
                raise Abstain('CAL_RGB_DECODE_FAILED')
            with np.load(C.ROOT/item['mask_npz'], allow_pickle=False) as arrays:
                mask = arrays['pallet_mask_clean'].copy()
            patch, valid, geometry_audit = rectify_global_top(image, mask, item['geometry'], label['target']['truth'])
            controls = describe_controls(patch, valid, metric_step_m=geometry_audit['metric_step_m'])
            record.update(supported=True, geometry_audit=geometry_audit, controls=controls)
            # Neutral grey outside supported global foreground; no local KP mark or GT wire.
            display = np.where(valid, np.clip(np.rint(patch*255), 0, 255), 127).astype(np.uint8)
            for name, array in (('REFERENCE_XZ', display), ('GRID_AXIS_EXCHANGE', display.T)):
                path = output/'patches'/f'{index:02d}_{name}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() or not cv2.imwrite(str(path), array):
                    raise FileExistsError('Preserve earlier diagnostic patch')
                patch_bindings[str(path.relative_to(output))] = C.sha(path)
        except Abstain as exc:
            record['reason'] = str(exc)
        records.append(record)
    if sum(r['fixed_native_compatible'] for r in records) != 44:
        raise ValueError('Original compatible44 must be reported without filtering incompatible2')
    full = fit_template(records)
    folds = []
    for session in sorted({r['session'] for r in records}):
        model = fit_template(records, heldout_session=session)
        held = [r for r in records if r['session'] == session]
        if set(model['training_ids']) & {r['id'] for r in held}:
            raise ValueError('Held-session template leakage')
        comparisons = [compare_template(r, model) for r in held]
        folds.append(dict(heldout_session=session, model=model,
            heldout_ids=[r['id'] for r in held], comparisons=comparisons, counts=_count_verdicts(comparisons)))
    logo = [r for fold in folds for r in fold['comparisons']]
    by_session = {f['heldout_session']:f['counts'] for f in folds}
    report = dict(schema='CAL_GLOBAL_TEXTURE_AXIS_FEASIBILITY_REPORT_V4_27', recipe=RECIPE,
        count=46, supported=sum(r['supported'] for r in records), fixed_native_compatible=44,
        records=records, full_CAL_template=full,
        full_CAL_in_sample_comparisons=[compare_template(r, full) for r in records],
        LOGO_folds=folds, LOGO_counts=_count_verdicts(logo), session_counts=by_session,
        session_balanced_reference_control_win_fraction_including_abstentions=float(np.mean([
            f['counts']['REFERENCE']/len(f['heldout_ids']) for f in folds])),
        interpretation='artificial transpose discrimination conditional on inherited reference physical axes; not independent phase labels or pose improvement',
        all46_retained=True, model_or_likelihood_adopted=False, new_pose_NN_or_local_KP_queries=0)
    _write_new(output/'PROBE_REPORT.json', report)
    for relative, digest in protocol['input_sha256'].items():
        if C.sha(C.ROOT/relative) != digest:
            raise ValueError('Probe input changed during execution')
    _verify_plan(output)
    _write_new(output/'PROBE_SEAL.json', dict(rows=46, report_sha256=C.sha(output/'PROBE_REPORT.json'),
        preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'), source_sha256=protocol['source_sha256'],
        input_sha256=protocol['input_sha256'], patch_sha256=patch_bindings,
        CAL_only=True, default_or_likelihood_adoption=False, pose_or_NN_calls=0))
    return dict(output=str(output), supported=report['supported'], LOGO_counts=report['LOGO_counts'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'prepare', 'probe'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--root-go', action='store_true', help='Only after explicit root authorization')
    args = parser.parse_args()
    result = write_plan(args.output) if args.command == 'plan' else (
        prepare if args.command == 'prepare' else probe)(args.output, root_go=args.root_go)
    print(json.dumps(C.finite(result), ensure_ascii=False))
