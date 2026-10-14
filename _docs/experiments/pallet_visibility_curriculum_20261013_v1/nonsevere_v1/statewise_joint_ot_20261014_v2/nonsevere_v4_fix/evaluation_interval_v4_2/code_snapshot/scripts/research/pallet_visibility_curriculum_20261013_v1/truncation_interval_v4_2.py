"""CAL maximum raw localization disks for conditional T admission.

This is an empirical interval, not a physical or future coverage certificate.
The fitted quantity uses raw image predictions and immutable non-S scene scale.
No initial-S coordinate is indexed in either fitting or the runtime gate.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'TRUNCATION_LOCALIZATION_INTERVAL_CAL_V4_2'
STATES = frozenset(('S', 'B', 'I', 'O', 'T', 'U'))
HUMAN_ELIGIBLE = frozenset(('DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED'))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _model_digest(model):
    return _digest({k: v for k, v in model.items() if k != 'model_sha256'})


def load_truncation_interval_model(value):
    if value is None:
        raise ValueError('An explicitly fitted CAL-only truncation model is required')
    model = deepcopy(value) if isinstance(value, Mapping) else json.loads(Path(value).read_text())
    if model.get('schema') != SCHEMA or model.get('fit_partition') != 'CAL':
        raise ValueError('A fitted CAL-only truncation localization interval is required')
    ratio = float(model['radius_ratio'])
    if not np.isfinite(ratio) or ratio < 0:
        raise ValueError('radius_ratio must be finite and nonnegative')
    if model.get('recipe') != 'ALL_VALID_INITIAL_NON_S_DIRECT_AND_EXTERNAL_UNTRIMMED_MAX':
        raise ValueError('Unexpected truncation interval calibration recipe')
    if model.get('model_sha256') != _model_digest(model):
        raise ValueError('Truncation interval model binding changed')
    return model


def _CAL_frames(rows):
    rows = list(rows)
    # Reject the entire batch before coordinate access, including mixed splits.
    if any(str(row.get('partition', '')).upper() != 'CAL' for row in rows):
        raise ValueError('TEST or unspecified partition is forbidden in interval fit/coverage')
    ids = [str(row['id']) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Calibration frame IDs must be unique')
    return rows


def _eligible_points(row):
    if not bool(row['physical_native_compatible']):
        return []
    hidden = list(row['initial_self_hidden'])
    human, valid = list(row['human_states_native']), list(row['valid_native'])
    if len(hidden) != 8 or len(human) != 8 or len(valid) != 8:
        raise ValueError('Eight native S/state/valid records required')
    scale = float(row['nonself_prediction_scale_px'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Frozen non-S prediction scale must be positive finite')
    result = []
    for k in range(8):
        # Gate before indexing either raw or annotation coordinates.
        if bool(hidden[k]) or human[k] not in HUMAN_ELIGIBLE or not bool(valid[k]):
            continue
        raw = np.asarray(row['raw_points'][k], dtype=float)
        target = np.asarray(row['gt_native'][k], dtype=float)
        if raw.shape != (2,) or target.shape != (2,):
            raise ValueError('Eligible raw/annotation coordinates must be xy')
        if not np.isfinite(raw).all() or np.all(raw == -1):
            raise ValueError('Eligible CAL raw prediction is missing; maximum recipe cannot silently trim it')
        if not np.isfinite(target).all():
            raise ValueError('Valid eligible CAL annotation must be finite')
        error = float(np.linalg.norm(raw-target))
        result.append({'id': str(row['id']), 'session': str(row['session']),
            'native_corner': k, 'human_state': human[k], 'raw_xy': raw.tolist(),
            'target_xy': target.tolist(), 'raw_error_px': error,
            'nonself_prediction_scale_px': scale, 'normalized_error': error/scale})
    return result


def fit_truncation_interval_model(cal_frames, *, heldout_session=None, calibration_bindings=None):
    """Fit fullCAL or session-excluded maximum; no trimming or grid selection.

    Frame keys: id, partition, session, physical_native_compatible,
    initial_self_hidden, human_states_native, valid_native, raw_points,
    gt_native, nonself_prediction_scale_px. Native GT mapping is caller-fixed;
    this function never chooses/rephases it from prediction error.
    """
    frames = _CAL_frames(cal_frames)
    train = [r for r in frames if heldout_session is None or str(r['session']) != str(heldout_session)]
    if not train:
        raise ValueError('No training CAL frames')
    points = [point for row in train for point in _eligible_points(row)]
    if not points:
        raise ValueError('No eligible CAL points for an untrimmed maximum')
    # Stable source ordering resolves ties; no coordinate-dependent rephasing.
    maximum = max(points, key=lambda p: p['normalized_error'])
    model = {'schema': SCHEMA, 'fit_partition': 'CAL',
        'recipe': 'ALL_VALID_INITIAL_NON_S_DIRECT_AND_EXTERNAL_UNTRIMMED_MAX',
        'radius_ratio': maximum['normalized_error'], 'fit_ids': [str(r['id']) for r in train],
        'compatible_ids': [str(r['id']) for r in train if r['physical_native_compatible']],
        'excluded_native_incompatible_ids': [str(r['id']) for r in train if not r['physical_native_compatible']],
        'heldout_session': None if heldout_session is None else str(heldout_session),
        'eligible_point_count': len(points),
        'max_error_point': {k:v for k,v in maximum.items() if k not in ('raw_xy','target_xy')},
        'initial_S_coordinates_accessed': False, 'native_GT_phase_fixed_by_caller': True,
        'scale_recomputed': False, 'trimming_or_quantile_or_hyperparameter_selection': False,
        'calibration_bindings': deepcopy(calibration_bindings or {}),
        'empirical_training_coverage_only': True,
        'physical_truncation_certificate': False, 'future_coverage_guarantee': False,
        'runtime_annotation_or_GT_compatibility_inputs': False}
    model['model_sha256'] = _model_digest(model)
    return model


def evaluate_interval_coverage(model, cal_frames):
    """Describe CAL-only full/heldout localization coverage without refitting."""
    model = load_truncation_interval_model(model)
    frames = _CAL_frames(cal_frames)
    points = [p for row in frames for p in _eligible_points(row)]
    records = [p | {'covered': p['normalized_error'] <= model['radius_ratio'] + 1e-12} for p in points]
    return {'fit_model_sha256': model['model_sha256'], 'evaluation_partition': 'CAL',
        'frame_ids': [str(r['id']) for r in frames], 'eligible_points': len(records),
        'covered_points': sum(p['covered'] for p in records), 'records': records,
        'initial_S_coordinates_accessed': False, 'physical_or_future_coverage_claim': False}


def disk_canvas_relation(center_xy, radius_px, image_hw):
    """Exact Euclidean disk/closed [0,W]x[0,H] intersection, including corners."""
    center = np.asarray(center_xy, dtype=float)
    hw = np.asarray(image_hw, dtype=float)
    radius = float(radius_px)
    if (center.shape != (2,) or not np.isfinite(center).all() or hw.shape != (2,)
            or not np.isfinite(hw).all() or np.any(hw <= 0) or np.any(hw != np.floor(hw))
            or not np.isfinite(radius) or radius < 0):
        raise ValueError('Finite xy/nonnegative radius and actual positive integer canvas required')
    h, w = hw
    closest = np.clip(center, [0., 0.], [w, h])
    distance = float(np.linalg.norm(center-closest))
    # Touching the closed canvas is an intersection and demotes conditional T.
    separated = bool(distance > radius)
    return {'center_xy': center.tolist(), 'radius_px': radius,
        'closest_canvas_xy': closest.tolist(), 'distance_to_closed_canvas_px': distance,
        'whole_disk_outside': separated, 'intersects_closed_canvas': not separated,
        'canvas_closure': '[0,W] x [0,H]', 'uses_Euclidean_corner_distance': True}


def apply_truncation_interval(states, raw_points, image_hw, nonself_scale_px, model,
                              state_records, initial_self_hidden):
    """Demote unsupported conditional T to U; no B/I/O/S changes or RGB reads."""
    model = load_truncation_interval_model(model)
    result = list(states); records = deepcopy(list(state_records)); hidden = list(initial_self_hidden)
    if (len(result) != 8 or len(records) != 8 or len(hidden) != 8
            or any(s not in STATES for s in result)):
        raise ValueError('Eight native states/records/initial S flags required')
    scale = float(nonself_scale_px)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('The unchanged non-S prediction scale must be finite positive')
    radius = model['radius_ratio'] * scale
    audits = []
    for k in range(8):
        if records[k]['native_id'] != k or records[k]['state'] != result[k]:
            raise ValueError('State records must match native ordered states')
        if bool(hidden[k]) != (result[k] == 'S'):
            raise ValueError('Initial S takes precedence and must already be fixed')
        if result[k] != 'T':
            continue
        certified = bool(records[k].get('physical_truncation_certified', False))
        before = deepcopy(records[k])
        if certified:
            item = {'native_corner': k, 'original_state': 'T', 'final_state': 'T',
                'independently_certified_T_bypass': True, 'raw_coordinate_accessed': False,
                'reason': 'independent_physical_truncation_certificate_bypasses_interval_gate'}
        else:
            raw = np.asarray(raw_points[k], dtype=float)
            if raw.shape != (2,):
                raise ValueError('Conditional T raw coordinate must have shape2')
            missing = not np.isfinite(raw).all() or np.all(raw == -1)
            relation = None if missing else disk_canvas_relation(raw, radius, image_hw)
            retain = bool(relation is not None and relation['whole_disk_outside'])
            result[k] = 'T' if retain else 'U'
            item = {'native_corner': k, 'original_state': 'T', 'final_state': result[k],
                'independently_certified_T_bypass': False, 'raw_coordinate_accessed': True,
                'localization_disk': relation, 'raw_prediction_missing': missing,
                'reason': 'whole_CAL_localization_disk_separated_from_canvas' if retain
                    else 'missing_raw_prediction_cannot_support_hard_T' if missing
                    else 'CAL_localization_disk_intersects_canvas'}
        records[k].update(state=result[k], pre_interval_state='T',
            original_T_hypothesis=before, interval_gate_applied=not certified,
            interval_gate_retained_T=result[k]=='T', interval_gate_reason=item['reason'])
        if not certified:
            records[k].update(reason=item['reason'], abstained=result[k]=='U',
                              original_conditional_T_hypothesis=True)
        audits.append(item)
    return {'states': result, 'records': records,
        'audit': {'schema': 'TRUNCATION_INTERVAL_GATE_AUDIT_V4_2',
            'model_sha256': model['model_sha256'], 'radius_ratio': model['radius_ratio'],
            'unchanged_nonself_prediction_scale_px': scale, 'radius_px': radius,
            'original_T_ids': [k for k,s in enumerate(states) if s=='T'],
            'retained_T_ids': [k for k,s in enumerate(result) if s=='T'],
            'demoted_T_to_U_ids': [a['native_corner'] for a in audits if a['final_state']=='U'],
            'records': audits, 'S_raw_coordinates_accessed': False, 'RGB_reads': 0,
            'B_I_O_S_roles_unchanged': True, 'retained_conditional_T_is_not_independent_certificate': True,
            'empirical_CAL_max_is_not_future_coverage_guarantee': True}}
