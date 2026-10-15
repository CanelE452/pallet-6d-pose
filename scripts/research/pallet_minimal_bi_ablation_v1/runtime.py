"""Minimal current-image B/I correction and the unchanged BI-only pose solver.

The scientific source is new_gain10_paired_BI_v2.observe_branch/common_BI_pose.
Only unused OT-region construction and U-display model/regions are removed.
The frozen state RGB/ResNet features, Tiny/MobileSAM masks, B/I measurement,
boundary recovery, covariance, solver, guards and fallback remain unchanged.
No N3, annotation, scoring, HTML, C cost or historical pose-bank API is used.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_ot_pnp as JOINT
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_masks_v2 as MASK
from scripts.research.pallet_visibility_curriculum_20261013_v1 import visible_bi_boundary_recovery_v4_3b as RECOVERY
from scripts.research.pallet_visibility_curriculum_20261013_v1.joint_observation import JointObservationModel
from scripts.research.pallet_visibility_curriculum_20261013_v1.state_visibility_v2 import StateVisibilityModel, SCHEMA as STATE_SCHEMA
from scripts.research.pallet_visibility_curriculum_20261013_v1.state_visibility_v4 import geometry_state_gates_v4, infer_state_visibility_v4
from scripts.research.pallet_visibility_curriculum_20261013_v1.visible_bi_correction_v2 import correct_visible_bi


SCHEMA = 'MINIMAL_BI_RUNTIME_V1'
SOLVER_CONFIG = {**JOINT._DEFAULTS, 'analytic_jacobian': True}
# Locked before this module's first actual observation/pose parity run.
PARITY_POLICY = dict(numeric_atol=1e-9, numeric_rtol=0.,
    exact_fields=['states', 'native IDs and order', 'mask arrays',
                  'accepted/fallback branch', 'solver configuration', 'source tags'],
    observation_omitted_keys=['ot_domains', 'u_region_display'],
    tolerance_may_be_relaxed_after_results=False)


@dataclass
class Components:
    state_model: StateVisibilityModel
    mask_provider: Callable
    device: str = 'cpu'
    bindings: dict = field(default_factory=dict)
    # Explicit injection is for pure contract tests, never production loading.
    state_producer: Callable | None = None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _bound(path, expected, bindings, role):
    path = Path(path).resolve()
    actual = sha256(path)
    if actual != expected:
        raise ValueError('Frozen ' + role + ' binding changed')
    bindings[role] = dict(sha256=actual)
    return path


def load_components(*, asset_root, state_descriptor, mask_descriptor,
                    boundary_config, device='cpu'):
    """Load only state/ResNet, Tiny and MobileSAM assets, with frozen hashes.

    Paths are explicit local configuration. No experiment config, N3 output,
    N3 weights, U model or OT model is opened, even when R1 runs by itself.
    This does not move or rewrite dependency-module globals.
    """
    root = Path(asset_root).resolve()
    paths = {key: Path(value).resolve() for key, value in dict(
        state_descriptor=state_descriptor, mask_descriptor=mask_descriptor,
        boundary_config=boundary_config).items()}
    bindings = {key: dict(sha256=sha256(path)) for key, path in paths.items()}
    descriptor = json.loads(paths['state_descriptor'].read_text())
    if descriptor.get('schema') != STATE_SCHEMA:
        raise ValueError('Unsupported actual-visibility descriptor')
    array_path = _bound(paths['state_descriptor'].parent / descriptor['arrays_file'],
        descriptor['arrays_sha256'], bindings, 'state_arrays')
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for array in arrays.values():
        array.setflags(write=False)
    network = None
    if descriptor.get('feature_family') != 'NON_S_RGB26' or descriptor.get('backbone'):
        backbone = descriptor['backbone']
        weights = _bound(root / backbone['weights_repo_relative'],
            backbone['weights_sha256'], bindings, 'state_backbone')
        network = JointObservationModel({'backbone': backbone}, {}, weights, device)
    model = StateVisibilityModel(descriptor, arrays, network)
    masks = json.loads(paths['mask_descriptor'].read_text())
    config = json.loads(paths['boundary_config'].read_text())
    tiny = masks['pallet_segmenter']
    tiny_path = _bound(root / tiny['weights_repo_relative'],
        tiny['weights_sha256'], bindings, 'tiny_segmenter')
    checkpoint = _bound(paths['boundary_config'].parent / config['mobile_sam_checkpoint'],
        config['model_sha256']['mobile_sam_checkpoint'], bindings, 'mobile_sam')
    provider = MASK.StatewiseMaskProvider(tiny_weights=tiny_path,
        tiny_sha256=tiny['weights_sha256'],
        sam_vendor=paths['boundary_config'].parent / config['mobile_sam_vendor'],
        sam_checkpoint=checkpoint, sam_sha256=config['model_sha256']['mobile_sam_checkpoint'],
        device=device)
    return Components(model, provider, device, bindings)


def _json(value):
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _array(value, shape, name):
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{name} must have finite shape {shape}')
    return array


def _rotation(value, name):
    R = _array(value, (3, 3), name)
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7, rtol=0) or not np.isclose(np.linalg.det(R), 1, atol=1e-7):
        raise ValueError(f'{name} must be a proper rotation')
    return R


def registered_geometry(image_hw, K, native_xyz, coarse_pose):
    """Exact arithmetic extracted from frozen StatewiseJointV2 geometry."""
    K = _array(K, (3, 3), 'K')
    xyz = _array(native_xyz, (8, 3), 'native_xyz')
    extents = _array(coarse_pose['cf_extents'], (3,), 'coarse cf_extents')
    if np.any(extents <= 0) or not np.allclose(xyz, C.cuboid(*extents), atol=1e-10, rtol=0):
        raise ValueError('native_xyz must preserve the registered original cuboid basis, dimensions and order')
    R = _rotation(coarse_pose.get('R_cf', coarse_pose.get('R')), 'coarse R_cf')
    t = _array(coarse_pose.get('centroid', coarse_pose.get('t')), (3,), 'coarse centroid')
    physical = _rotation(coarse_pose.get('R_physical', R), 'coarse R_physical')
    Q = R.T @ physical
    projected, depth = JOINT.project_native(xyz, R, t, K)
    if np.any(depth <= 0) or not np.isfinite(projected).all():
        raise ValueError('Coarse registered cuboid must project with positive camera depth')
    camera_in_object = -R.T @ t
    planes = ConvexHull(xyz).equations
    scale = max(float(np.linalg.norm(extents)), 1e-9)
    hidden = []
    for point in xyz:
        incident = np.abs(planes[:, :3] @ point + planes[:, 3]) <= 1e-8 * scale
        ray = camera_in_object - point
        cosines = planes[incident, :3] @ ray / np.linalg.norm(ray)
        hidden.append(bool(np.max(cosines) < -np.sin(np.deg2rad(2))))
    hull = cv2.convexHull(projected.astype(np.float32)).reshape((-1, 2))
    contour = hull.astype(np.float32).reshape((-1, 1, 2))
    roles = ['B' if abs(cv2.pointPolygonTest(contour, tuple(map(float, point)), True)) < 1e-3
             else 'I' for point in projected]
    return dict(native_xyz=xyz, K=K, R_cf=R, centroid=t, R_physical=physical,
                physical_registration_Q=Q, cf_extents=extents,
                initial_projection_xy=projected, initial_self_hidden=np.array(hidden, dtype=bool),
                initial_roles=roles, image_hw=list(image_hw),
                S_source='fixed_registered_coarse_cuboid_outward_face_visibility',
                S_is_conditional_on_coarse_pose=True,
                B_I_role_is_geometry_not_actual_visibility=True,
                raw_points_used_for_geometry=False)


def _mask(array, image_hw, name):
    mask = np.asarray(array, dtype=float)
    if mask.shape != image_hw or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError(f'{name} must be finite canvas probabilities in [0,1]')
    return mask.copy()


def _prompt(raw, hidden, image_hw):
    height, width = image_hw
    ids = [i for i in range(8) if not hidden[i] and np.isfinite(raw[i]).all()
           and not np.array_equal(raw[i], [-1., -1.])
           and 0 <= raw[i, 0] < width and 0 <= raw[i, 1] < height]
    points = raw[ids].copy()
    box = np.r_[points.min(axis=0), points.max(axis=0)].tolist() if ids else None
    return ids, points, box


def _masks(image, raw, geometry, eligible_ids, components):
    ids, points, box = _prompt(raw, geometry['initial_self_hidden'], image.shape[:2])
    audit = dict(prompt_native_ids=ids, prompt_points=points.tolist(), prompt_box_xyxy=box,
        S_excluded_before_prompt=True, ninth_center_used=False,
        initial_projection_used_as_prompt=False, provider_invoked=False)
    if len(eligible_ids):
        if not isinstance(components.state_model, StateVisibilityModel) and components.state_producer is None:
            raise ValueError('Explicitly loaded production state model required')
        if components.mask_provider is None:
            raise ValueError('Current-RGB mask provider required')
        bundle = components.mask_provider(image, prompt_points=points.copy(), prompt_native_ids=ids.copy(),
            prompt_box_xyxy=deepcopy(box), initial_projection_xy=geometry['initial_projection_xy'].copy(),
            K=geometry['K'].copy())
        correction_mask = _mask(bundle['pallet_mask'], image.shape[:2], 'provider pallet_mask')
        state_mask = _mask(bundle.get('state_pallet_mask', bundle['pallet_mask']),
            image.shape[:2], 'provider state_pallet_mask')
        audit['provider_provenance'] = deepcopy(bundle.get('provenance', {}))
        audit['shared_mask'] = 'state_pallet_mask' not in bundle
        audit.update(source='current_RGB_mask_provider', provider_invoked=True)
    else:
        correction_mask = state_mask = np.zeros(image.shape[:2], dtype=float)
        audit.update(source='no_gated_RGB_queries_no_mask_forward', shared_mask=True)
        bundle = dict(pallet_mask=np.zeros(image.shape[:2], np.float32),
            state_pallet_mask=np.zeros(image.shape[:2], np.float32),
            provenance={'reason': 'no_gated_RGB_queries_no_provider_invocation'})
    return correction_mask, state_mask, audit, bundle


def observe(image_bgr, native_points, K, coarse_pose, *, components,
            prediction_source='RAW_YOLO', record_id='current'):
    """Observe this branch independently, returning measured XY, not pose XY.

    U remains an abstention. Its approximate display regions are deliberately
    absent. Only B/I can receive local correction/recovery; S is removed before
    mask prompts, state RGB queries and B/I pixel queries. Initial S is still
    conditional on this branch's own input F pose.
    """
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError('Actual nonempty uint8 BGR image required')
    raw = np.array(native_points, dtype=float, copy=True)
    if raw.shape not in ((8, 2), (9, 2)):
        raise ValueError('Native 8 or 9 points required')
    original_pose = deepcopy(coarse_pose)
    geometry = registered_geometry(image.shape[:2], K, C.cuboid(*coarse_pose['cf_extents']), coarse_pose)
    hidden = geometry['initial_self_hidden']
    gated = raw.copy()
    gated[:8][hidden] = [-1., -1.]
    _, eligible = geometry_state_gates_v4(image, gated, hidden,
        geometry['initial_roles'], geometry['initial_projection_xy'], independent_truncation=None)
    correction_mask, state_mask, mask_audit, masks = _masks(image, gated, geometry, eligible, components)
    observed = (components.state_producer or infer_state_visibility_v4)(image, gated.copy(), hidden.copy(),
        list(geometry['initial_roles']), geometry['initial_projection_xy'].copy(), components.state_model,
        pallet_mask=state_mask, independent_truncation=None)
    states = list(observed['states'])
    records = observed.get('records', [])
    if (len(states) != 8 or any(s not in {'S','B','I','O','T','U'} for s in states)
            or len(records) != 8 or any(int(r['native_id']) != k or r['state'] != states[k]
                                      for k, r in enumerate(records))):
        raise ValueError('Native state records must preserve order')
    if any(k not in eligible for k in observed.get('rgb_queried_ids', [])):
        raise ValueError('Excluded geometric gate queried')
    if any(records[k].get('rgb_evaluated') for k in np.flatnonzero(hidden)):
        raise ValueError('S received a local RGB query')
    if any((s == 'S') != bool(hidden[k]) for k, s in enumerate(states)):
        raise ValueError('Initial S gate must remain fixed')
    if any(s in {'B','I'} and s != geometry['initial_roles'][k] for k, s in enumerate(states)):
        raise ValueError('Visible role changed')
    normalized_pose = dict(original_pose, R_cf=geometry['R_cf'].tolist(),
        centroid=geometry['centroid'].tolist(), cf_extents=geometry['cf_extents'].tolist())
    correction = correct_visible_bi(image, gated.copy(), geometry['K'], normalized_pose, states,
        pallet_mask=correction_mask, config=None, native_xyz=geometry['native_xyz'])
    observation = _json(dict(schema='STATEWISE_RGB_OBSERVATIONS_V4', states=states,
        initial_geometry=geometry, state_observation=observed, mask_audit=mask_audit,
        visible_correction=correction, frozen_visible_measurements=deepcopy(correction['visible_measurements']),
        original_raw_points=raw.tolist(), original_coarse_pose=original_pose,
        provenance=dict(GT_inputs=False, S_image_queries=0, S_mask_prompt_queries=0,
            S_coordinates_removed_before_image_pipeline=True, current_RGB_state_inference=True,
            B_I_measured_before_joint_OT=True, initial_pose_coordinates_are_not_observations=True,
            conditional_T_cannot_override_available_nonS_image_query=True,
            original_native_model_and_registration_preserved=True,
            state_producer_injected_for_tests=components.state_producer is not None)))
    # Match the canonical JSON boundary before recovery, not merely its inputs.
    base = deepcopy(observation['visible_correction'])
    recovered = RECOVERY.recover_visible_boundary(image, base['raw_points'], K, coarse_pose,
        observation['states'], pallet_mask=masks['pallet_mask'], base_correction=base,
        native_xyz=observation['initial_geometry']['native_xyz'])
    observation['visible_correction'] = recovered
    observation['frozen_visible_measurements'] = deepcopy(recovered['visible_measurements'])
    assert all(not r.get('pixel_queried') for k, r in enumerate(recovered['records'])
               if hidden[k] or states[k] in ('S','O','T','U'))
    assert all(not r.get('rgb_evaluated') for k, r in enumerate(records) if hidden[k])
    return dict(observation=_json(observation), base_visible_correction=base,
        masks={key: np.array(masks[key], copy=True) for key in ('pallet_mask', 'state_pallet_mask')},
        masks_provenance=deepcopy(masks.get('provenance', {})),
        operations=dict(schema=SCHEMA, prediction_source=str(prediction_source), record_id=str(record_id),
            state_and_S_recomputed_for_own_input=True, raw_state_calibration_coverage_not_validated=True,
            state_ResNet_features_retained=True, U_display_model_loaded=False,
            OT_domains_built=False, C_terms=False, old_pose_bank_inputs=False,
            new_N3_or_YOLO_forwards=0, measurement_coordinates_are_not_final_pose_projection=True))


def _bi_only(result, frame, config):
    """Frozen experiment _bi_only body, without importing its report stack."""
    states = ['U' if s in ('O','T') else s for s in result['states']]
    coarse = frame['initial_pose']
    solved = JOINT.solve_joint_ot_pnp(C.cuboid(*coarse['cf_extents']), frame['K'], states,
        deepcopy(result['frozen_visible_measurements']), [], frame['raw_hw'],
        initializations=[dict(R=coarse['R_cf'], t=coarse['centroid'])], config=config)
    Q = np.asarray(coarse['R_cf']).T @ np.asarray(coarse['R_physical'])
    if solved['accepted']:
        pose = dict(available=True, R_cf=solved['R'], R_physical=(np.asarray(solved['R']) @ Q).tolist(),
            centroid=solved['t'], cf_extents=coarse['cf_extents'], source='CORRECTED_BI_ONLY_PNP')
    else:
        seed = solved.get('corrected_BI_seed') or {}
        if seed.get('R') is not None and seed.get('t') is not None and seed.get('visible_support_guards_pass'):
            pose = dict(available=True, R_cf=seed['R'], R_physical=(np.asarray(seed['R']) @ Q).tolist(),
                centroid=seed['t'], cf_extents=coarse['cf_extents'], source='CORRECTED_BI_SEED_FALLBACK')
        else:
            pose = dict(coarse, source='ORIGINAL_PNP_FALLBACK')
    return dict(actual_pose=pose, solution=solved, states=states,
        frozen_visible_measurements=deepcopy(result['frozen_visible_measurements']))


def solve(observation, coarse_pose, K, image_hw):
    """Canonical fresh BI-only PnP, all guards and rejected/fallback rows kept."""
    source = dict(states=list(observation['states']),
        frozen_visible_measurements=deepcopy(observation['frozen_visible_measurements']))
    xyz = np.asarray(observation['initial_geometry']['native_xyz'], float)
    expected = C.cuboid(*coarse_pose['cf_extents'])
    if not np.array_equal(xyz, expected):
        raise ValueError('Own registered native W/D model changed')
    ids = [m['native_corner'] for m in source['frozen_visible_measurements']]
    if len(ids) != len(set(ids)) or any(source['states'][k] not in ('B','I') for k in ids):
        raise ValueError('Only unique B/I image observations may enter PnP')
    result = _bi_only(source, dict(initial_pose=coarse_pose, K=K, raw_hw=image_hw), deepcopy(SOLVER_CONFIG))
    result.update(method='BI_ONLY_FRESH_PNP', input_native_ids=ids,
        observed_states_preserved=list(observation['states']), solver_configuration=deepcopy(SOLVER_CONFIG),
        O_T_U_are_not_PnP_points=True, S_image_terms=0, OT_U_region_pose_constraints=0,
        E_or_cached_pose_bank_starts=0, own_input_F_coarse_start=True,
        full_current_V33_method=False, measurement_coordinates_are_not_final_pose_projection=True)
    result['accepted'] = bool(result['solution']['accepted'])
    result['fallback'] = not result['accepted']
    pose = result['actual_pose']
    result['model_projection_xy'], _ = JOINT.project_native(expected, np.asarray(pose['R_cf']),
        np.asarray(pose['centroid']), np.asarray(K))
    return C.finite(result)
