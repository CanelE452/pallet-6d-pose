"""Explicit numeric allowlist shared by COMMON and BI observation scorers.

IDs, paths, states, image bindings and diagnostic/reference quantities cannot
enter this extractor.  Each scalar has a separate validity bit.  The model
normalizer is fit on valid TRAIN values and appends these validity bits.
"""
from __future__ import annotations

import numpy as np

from .observer import CONFIGURATION, TRACE_FIELDS
from scripts.research.pallet_visibility_curriculum_20261013_v1.interior_observation import NEIGHBORS

BASE_NAMES = (
    'candidate.is_keep', 'candidate.is_intersection',
    'candidate.delta_x_sigma', 'candidate.delta_y_sigma', 'candidate.delta_norm_sigma',
    'candidate.delta_x_canvas', 'candidate.delta_y_canvas', 'candidate.delta_norm_object',
    'candidate.applied_x_canvas', 'candidate.applied_y_canvas', 'anchor.x_canvas', 'anchor.y_canvas',
    'candidate.raw_intersection_delta_x_sigma', 'candidate.raw_intersection_delta_y_sigma',
    'candidate.blend_alpha', 'candidate.covariance_xx_sigma2', 'candidate.covariance_xy_sigma2',
    'candidate.covariance_yy_sigma2', 'candidate.conditional_covariance_xx_sigma2',
    'candidate.conditional_covariance_xy_sigma2', 'candidate.conditional_covariance_yy_sigma2',
    'candidate.intersection_sine', 'candidate.boundary_source_count', 'candidate.interior_source_count',
    'anchor.projection_delta_x_sigma', 'anchor.projection_delta_y_sigma',
    'anchor.native_x_extent', 'anchor.native_y_extent', 'anchor.native_z_extent',
    'geometry.extent_x_diagonal', 'geometry.extent_y_diagonal', 'geometry.extent_z_diagonal',
    'geometry.projected_width_canvas', 'geometry.projected_height_canvas')
NEIGHBOR_NAMES = ('eligible', 'guide_direction_x', 'guide_direction_y', 'guide_length_object', 'guide_length_canvas')
LINE_NAMES = ('supported', 'selected_by_candidate', 'transition_sample_fraction', 'sample_fraction',
              'normal_x', 'normal_y', 'offset_at_anchor_sigma', 'actual_support_length_object',
              'inlier_count_fraction', 'inlier_fraction', 'inlier_span_fraction', 'inlier_rms_sigma', 'angle_change_limit')
TRACE_NAMES = ('valid_grid_fraction',) + TRACE_FIELDS
FEATURE_NAMES = tuple(BASE_NAMES) + tuple(
    f'neighbor{slot}.{name}' for slot in range(3) for name in NEIGHBOR_NAMES) + tuple(
    f'neighbor{slot}.cue{cue}.{family}.{name}' for slot in range(3) for cue in ('B', 'I')
    for family, names in (('line', LINE_NAMES), ('trace', TRACE_NAMES)) for name in names)
FEATURE_DIMENSION = len(FEATURE_NAMES)
MODEL_INPUT_DIMENSION = 2 * FEATURE_DIMENSION


def candidate_features(bank, corner, candidate):
    """Return (finite values[F], binary validity[F]) with no state argument."""
    values = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    validity = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    features = {}
    def add(name, value, valid=True):
        number = float(value) if valid else 0.
        if valid and not np.isfinite(number):
            raise ValueError('Nonfinite valid feature: ' + name)
        features[name] = (number, bool(valid))
    h, w = bank['image_hw']; sigma = CONFIGURATION['detector_sigma_px']
    projection = np.asarray(bank['initial_projection_xy'], float)
    extent = np.asarray(bank['cf_extents'], float)
    native = np.asarray(corner['native_xyz'], float)
    object_scale = max(float(np.linalg.norm(np.ptp(projection, axis=0))), 1e-12)
    canvas_diagonal = float(np.hypot(w, h)); extent_diagonal = float(np.linalg.norm(extent))
    point = np.asarray(candidate['applied_xy'], float); q0 = np.asarray(candidate['q0'], float); delta = point - q0
    add('candidate.is_keep', candidate['kind'] == 'keep'); add('candidate.is_intersection', candidate['kind'] == 'intersection')
    for name, value in (('delta_x_sigma', delta[0] / sigma), ('delta_y_sigma', delta[1] / sigma),
                        ('delta_norm_sigma', np.linalg.norm(delta) / sigma), ('delta_x_canvas', delta[0] / w),
                        ('delta_y_canvas', delta[1] / h), ('delta_norm_object', np.linalg.norm(delta) / object_scale),
                        ('applied_x_canvas', point[0] / w), ('applied_y_canvas', point[1] / h)):
        add('candidate.' + name, value)
    add('anchor.x_canvas', q0[0] / w); add('anchor.y_canvas', q0[1] / h)
    raw = candidate['raw_intersection_xy']
    for axis, label in enumerate(('x', 'y')):
        add(f'candidate.raw_intersection_delta_{label}_sigma', (raw[axis] - q0[axis]) / sigma if raw is not None else 0., raw is not None)
    add('candidate.blend_alpha', candidate['blend_alpha'])
    for prefix, covariance in (('covariance', candidate['covariance_px2']),
                               ('conditional_covariance', (candidate.get('conditional_covariance') or {}).get('covariance_px2'))):
        for label, i, j in (('xx', 0, 0), ('xy', 0, 1), ('yy', 1, 1)):
            add(f'candidate.{prefix}_{label}_sigma2', covariance[i][j] / sigma ** 2 if covariance is not None else 0., covariance is not None)
    add('candidate.intersection_sine', candidate.get('intersection_sine') or 0., candidate.get('intersection_sine') is not None)
    for cue, name in (('B', 'boundary'), ('I', 'interior')):
        add(f'candidate.{name}_source_count', candidate['source_cues'].count(cue))
    projection_delta = np.asarray(corner['initial_projection_xy']) - q0
    for axis, name in enumerate(('x', 'y')):
        add(f'anchor.projection_delta_{name}_sigma', projection_delta[axis] / sigma)
    for axis, name in enumerate(('x', 'y', 'z')):
        add(f'anchor.native_{name}_extent', native[axis] / extent[axis])
        add(f'geometry.extent_{name}_diagonal', extent[axis] / extent_diagonal)
    add('geometry.projected_width_canvas', np.ptp(projection[:, 0]) / w)
    add('geometry.projected_height_canvas', np.ptp(projection[:, 1]) / h)
    samplings = {r['native_neighbor']: r for r in corner['samplings']}
    rays = {(r['native_neighbor'], r['cue']): r for r in corner['rays']}
    for slot, mate in enumerate(NEIGHBORS[corner['native_corner']]):
        prefix = f'neighbor{slot}.'; sampling = samplings.get(mate)
        add(prefix + 'eligible', sampling is not None)
        guide = projection[mate] - np.asarray(corner['initial_projection_xy'])
        length = float(np.linalg.norm(guide)); direction = guide / length if length > 0 else np.zeros(2)
        for name, value in (('guide_direction_x', direction[0]), ('guide_direction_y', direction[1]),
                            ('guide_length_object', length / object_scale), ('guide_length_canvas', length / canvas_diagonal)):
            add(prefix + name, value, sampling is not None and length > 0)
        for cue in ('B', 'I'):
            ray = rays.get((mate, cue)); p = prefix + f'cue{cue}.'
            if ray is None:
                continue
            supported = bool(ray.get('supported')); summary = ray.get('cue_summary') or {}
            add(p + 'line.supported', supported)
            add(p + 'line.selected_by_candidate', ray['source_line_id'] in candidate['source_line_ids'])
            for name, value, valid in (
                ('transition_sample_fraction', ray.get('transition_sample_count', 0) / CONFIGURATION['along_samples'], 'transition_sample_count' in ray),
                ('sample_fraction', ray.get('samples', 0) / CONFIGURATION['along_samples'], 'samples' in ray),
                ('normal_x', ray.get('normal', [0., 0.])[0], supported),
                ('normal_y', ray.get('normal', [0., 0.])[1], supported),
                ('offset_at_anchor_sigma', (float(np.asarray(ray['normal']) @ q0) + ray['offset']) / sigma if supported else 0., supported),
                ('actual_support_length_object', float(np.ptp(ray.get('actual_inlier_tangent_interval_px', [0., 0.]))) / object_scale, supported),
                ('inlier_count_fraction', ray.get('inlier_count', 0) / CONFIGURATION['along_samples'], 'inlier_count' in ray),
                ('inlier_fraction', ray.get('inlier_fraction', 0.), 'inlier_fraction' in ray),
                ('inlier_span_fraction', ray.get('inlier_span_fraction', 0.), 'inlier_span_fraction' in ray),
                ('inlier_rms_sigma', (ray.get('inlier_rms_px') or 0.) / sigma, ray.get('inlier_rms_px') is not None),
                ('angle_change_limit', ray.get('angle_change_deg', 0.) / CONFIGURATION['maximum_direction_change_deg'], 'angle_change_deg' in ray)):
                add(p + 'line.' + name, value, valid)
            add(p + 'trace.valid_grid_fraction', summary.get('valid_grid_fraction', 0.), 'valid_grid_fraction' in summary)
            for name in TRACE_FIELDS:
                entry = summary.get('values', {}).get(name, {})
                scale = CONFIGURATION['normal_search_radius_px'] if name == 'normal_offset_px' else 1.
                add(p + 'trace.' + name, entry.get('value', 0.) / scale, entry.get('valid', False))
    if set(features) - set(FEATURE_NAMES):
        raise AssertionError('Feature allowlist changed without updating FEATURE_NAMES')
    for index, name in enumerate(FEATURE_NAMES):
        value, valid = features.get(name, (0., False)); values[index] = value; validity[index] = valid
    return values, validity


def condition(state, arm):
    """Only these two values differ between paired model forwards."""
    if state not in ('B', 'I'):
        raise ValueError('Only frozen visible B/I anchors have scorer conditions')
    if arm == 'COMMON':
        return np.array([.5, .5], dtype=np.float64)
    if arm == 'BI':
        return np.array([1., 0.] if state == 'B' else [0., 1.], dtype=np.float64)
    raise ValueError('Expected COMMON or BI arm')


def feature_rows(bank):
    rows = []
    for corner in bank['corners']:
        candidates = corner['candidates']
        if not candidates:
            continue
        matrices = [candidate_features(bank, corner, candidate) for candidate in candidates]
        rows.append(dict(native_corner=corner['native_corner'], state=bank['states'][corner['native_corner']],
            candidate_ids=[c['candidate_id'] for c in candidates], applied_xy=[c['applied_xy'] for c in candidates],
            x_values=[values.tolist() for values, _ in matrices], x_validity=[valid.tolist() for _, valid in matrices],
            partial_replaces_full_likelihood=corner['partial_line'] is not None))
    return rows


def feature_contract():
    return dict(schema='COMMON_LOCAL_NUMERIC_FEATURE_ALLOWLIST_V1', feature_names=list(FEATURE_NAMES),
        scalar_count=FEATURE_DIMENSION, normalized_value_and_validity_dimension=MODEL_INPUT_DIMENSION,
        condition_dimension=2, mlp_input_dimension=MODEL_INPUT_DIMENSION + 2,
        units=dict(sigma='frozen v2 detector_sigma_px=2 for all states', canvas='x/width,y/height,length/canvas_diagonal',
                   object='projected initial registered cuboid diagonal', extent='registered axis extent or 3D diagonal',
                   mask='occupancy or signed occupancy difference', rgb='original LAB gradient and color difference',
                   normal_offset='offset divided by v2 normal_search_radius_px', line_angle='angle / maximum_direction_change_deg'),
        source_mapping=dict(candidate='own common-bank candidate record including applied XY and covariance',
            line='original v2 _line / boundary_refine._fit_line diagnostics',
            trace='original v2 along/offset grid, both cue gates, mask and RGB summaries at each cue selected samples',
            geometry='same frozen N0 native xyz, projected native pose and dimensions for COMMON and BI'),
        missing='value 0 plus validity 0; observed numeric zero retains validity 1',
        normalization='train valid values only; input [normalized values * validity, validity]',
        forbidden=['frame/corner/candidate IDs as features','paths','reference coordinates','reference error',
                   'GT state','GT choice','historical pose success','real difficulty','automatic B/I state outside condition'],
        COMMON_condition=[.5, .5], BI_conditions=dict(B=[1., 0.], I=[0., 1.]))
