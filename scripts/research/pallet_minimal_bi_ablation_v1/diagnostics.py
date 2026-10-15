"""Descriptive saved-state/measurement/solver diagnostics, not causal routing."""
from __future__ import annotations

from collections import Counter
import numpy as np

STATES = ('S', 'B', 'I', 'O', 'T', 'U')
MOVEMENT_EPS_PX = 1e-8
COORDINATE_DIAGNOSTIC_ARMS = (
    'R1_PRECORRECTION_XY_FIXED_OBSERVATION',
    'N1_PRECORRECTION_XY_FIXED_OBSERVATION',
)


def observation_summary(observation, initial_points, pose_result=None):
    """Read coordinates only after the saved B/I state gate.

    A retained backbone point is a fallback, not a new intersection. Exact or
    near-zero motion does not become an observed movement because a flag is true.
    This diagnostic performs no RGB, mask, model projection, or pose calculation.
    """
    states = observation['states']
    if len(states) != 8 or any(s not in STATES for s in states):
        raise ValueError('Eight explicit saved corner states required')
    correction = observation['visible_correction']
    records = {r['native_corner']:r for r in correction['records']}
    measurements = observation['frozen_visible_measurements']
    if len({m['native_corner'] for m in measurements}) != len(measurements):
        raise ValueError('Duplicate measurement IDs')
    measured = []
    for m in measurements:
        k = m['native_corner']
        if type(k) is not int or not 0 <= k < 8 or states[k] not in ('B', 'I'):
            raise ValueError('Only gated B/I observations may enter this diagnostic')
        q, xy = np.asarray(initial_points[k], float), np.asarray(m['xy'], float)
        cov = np.asarray(m['covariance_px2'], float)
        if (q.shape != (2,) or xy.shape != (2,) or not np.isfinite(q).all() or not np.isfinite(xy).all()
                or cov.shape != (2,2) or not np.isfinite(cov).all()
                or not np.allclose(cov, cov.T, atol=1e-10, rtol=0)
                or np.linalg.eigvalsh(cov).min() <= 0):
            raise ValueError('Finite measured coordinates and positive covariance required')
        record = records[k]
        shift = float(np.linalg.norm(xy-q))
        fallback = m['source'] == 'VISIBLE_BACKBONE_FALLBACK'
        measured.append(dict(native_corner=k, state=states[k], source=m['source'],
            original_prediction_retained=fallback, new_image_intersection=not fallback,
            displacement_px=shift, moved=shift > MOVEMENT_EPS_PX,
            corrected_flag=bool(record.get('corrected',False)), reason=record.get('reason'),
            covariance_eigenvalues=np.linalg.eigvalsh(cov).tolist(),
            line_evidence=[dict(native_edge=r.get('endpoints'), supported=r.get('supported'),
                reason=r.get('reason'), normal=r.get('normal'), offset=r.get('offset'),
                inlier_count=r.get('inlier_count'), inlier_fraction=r.get('inlier_fraction'),
                inlier_span_fraction=r.get('inlier_span_fraction'), line_rms_px=r.get('line_rms_px'),
                support_tangent_interval=r.get('actual_inlier_tangent_interval_px'))
                for r in record.get('rays',[])]))
    excluded = [r for r in correction['records'] if states[r['native_corner']] not in ('B','I')]
    if any(r.get('pixel_queried',False) for r in excluded):
        raise ValueError('Excluded S/O/T/U correction query found')
    result = dict(state_counts=dict(Counter(states)), states=list(states),
        measured_BI_n=len(measured), new_intersection_n=sum(m['new_image_intersection'] for m in measured),
        retained_backbone_n=sum(m['original_prediction_retained'] for m in measured),
        moved_n=sum(m['moved'] for m in measured),
        movement_threshold='displacement_px >1e-8; distinct from corrected_flag',
        reason_counts=dict(Counter(m['reason'] for m in measured)), measurements=measured,
        excluded_point_queries=0,
        state_and_measurement_are_conditioned_on_own_initial_pose=True,
        human_state_accuracy='NA unless separately bound existing postseal annotations are supplied',
        mask_absence_is_not_external_occlusion_ground_truth=True)
    if pose_result is not None:
        solution = pose_result['solution']
        result['solver'] = dict(accepted=pose_result['accepted'], fallback=pose_result['fallback'],
            reason=solution.get('reason'), input_native_ids=pose_result['input_native_ids'],
            final_hypothesis=pose_result.get('hypothesis'),
            initialization=solution.get('initialization'),
            constraint_acceptance_is_not_reference_accuracy=True)
    return result


def changed_case_groups(scored_rows):
    """Fixed-ID first example per existing category, never best-error selection."""
    groups = {name:[] for name in ('R1_RECOVERY','R1_DAMAGE','ONLY_N1_RECOVERS','BOTH_NEW_FAIL')}
    for row in sorted(scored_rows, key=lambda r:r['id']):
        if row['partition'] == 'CAL':
            continue
        v = {a:row['metrics'][a]['success_le5'] for a in ('R0','R1','N0','N1')}
        if any(x is None for x in v.values()):
            continue
        if v['R1'] and not v['R0']: groups['R1_RECOVERY'].append(row['id'])
        if v['R0'] and not v['R1']: groups['R1_DAMAGE'].append(row['id'])
        if not v['R1'] and v['N1'] and not v['N0']: groups['ONLY_N1_RECOVERS'].append(row['id'])
        if not v['R1'] and not v['N1']: groups['BOTH_NEW_FAIL'].append(row['id'])
    return dict(selection='lexicographic fixed ID order within each prespecified category; no error ranking',
                all_ids=groups, first_id={k:v[0] if v else None for k,v in groups.items()},
                absent_categories=[k for k,v in groups.items() if not v],
                overlapping_categories_allowed=True, performance_routing=False)


def phase10_comparison(main_scored, replaced_scored):
    """Coordinate-only substitution is conditional, never a fifth main method."""
    index = {r['id']:r for r in replaced_scored}
    if len(index) != len(replaced_scored):
        raise ValueError('Duplicate coordinate-diagnostic ID')
    rows = []
    for r in main_scored:
        if r['partition'] == 'CAL': continue
        replacement = index.get(r['id'])
        rows.append(dict(id=r['id'], partition=r['partition'],
            branches={a:dict(main_metric=r['metrics'][a],
                coordinates_replaced_metric=None if replacement is None else replacement['metrics'][a])
                for a in ('R1','N1')}))
    return dict(rows=rows, diagnostic_names=list(COORDINATE_DIAGNOSTIC_ARMS),
                expected_frame_rows=192, expected_max_pose_solves=384,
                kept_states_support_covariance_and_solver=True,
                natural_independent_no_correction_baseline=False,
                causal_mechanism_identified=False, main_Holm_family_member=False,
                diagnostic_not_computed_n=sum(r['branches']['R1']['coordinates_replaced_metric'] is None for r in rows))
