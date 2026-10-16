"""Offline V4.2 interval runtime audit. No fit, RGB execution or parameter choice."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
sys.path.insert(0, str(ROOT))
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import itertools
import json
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_ot_pnp as V

BASE = ROOT / '_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1'
N = BASE / 'statewise_joint_ot_20261014_v2/nonsevere_v4_fix'
OWN = N / 'independent_review'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(p):
    with gzip.open(p, 'rt') as f:
        return [json.loads(line) for line in f]


def project(xyz, R, t, K):
    camera = np.asarray(xyz) @ np.asarray(R).T + np.asarray(t)
    h = camera @ np.asarray(K).T
    safe = np.where(np.abs(h[:, 2]) < 1e-10, np.where(h[:, 2] < 0, -1e-10, 1e-10), h[:, 2])
    return h[:, :2] / safe[:, None], camera[:, 2]


def metric(pose, truth):
    # Independent scalar implementation, fixed physical C2 and same reference.
    R, G = np.asarray(pose['R_physical']), np.asarray(truth['R'])
    t, g = np.asarray(pose['centroid']), np.asarray(truth['t'])
    xyz = np.asarray(list(itertools.product((-.5, .5), repeat=3))) * truth['xyz']
    angles, adds = [], []
    if truth['order'] != 2:
        raise ValueError('Fixed physical C2 reference required')
    for k in range(2):
        c, s = np.cos(k * np.pi), np.sin(k * np.pi)
        Q = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        target = G @ Q
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace(target.T @ R) - 1) / 2, -1, 1)))))
        adds.append(float(np.linalg.norm(xyz @ R.T + t - (xyz @ target.T + g), axis=1).mean() * 100))
    translation, rotation = float(np.linalg.norm(t - g) * 100), min(angles)
    return {'translation_cm': translation, 'rotation_deg': rotation, 'ADDsym_cm': min(adds),
            'success_5cm_5deg': translation < 5 and rotation < 5}


def region_slacks(domain, point, hw, cfg):
    values = []
    shape = domain.get('shape', 'ellipse')
    if shape == 'ellipse':
        angle = np.deg2rad(domain.get('angle_deg', domain.get('angle', 0)))
        basis = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        center = np.asarray(domain.get('center_xy', domain.get('ellipse_center')))
        axes = np.asarray(domain.get('half_axes_px', domain.get('half_axes')))
        values.append(1 - float(np.sum((basis.T @ (point - center) / axes) ** 2)))
    elif shape == 'box':
        bounds = np.asarray(domain.get('bounds_xyxy', domain.get('box_xyxy')))
        span = bounds[2:] - bounds[:2]
        values.extend(((point - bounds[:2]) / span).tolist())
        values.extend(((bounds[2:] - point) / span).tolist())
    elif shape != 'halfplane':
        raise ValueError('Unknown region shape')
    halfplanes = list(domain.get('halfplanes', []))
    if 'normal' in domain:
        halfplanes.append([*domain['normal'], domain['offset']])
    for a, b, c in halfplanes:
        values.append(float((c - a * point[0] - b * point[1]) / np.hypot(a, b)))
    if domain['state'] == 'T':
        h, w = hw
        margin = domain.get('outside_margin_px', cfg['outside_canvas_margin_px'])
        sides = {'left': -margin - point[0], 'right': point[0] - w - margin,
                 'top': -margin - point[1], 'bottom': point[1] - h - margin}
        allowed = domain.get('outside_canvas_side')
        if allowed is None or allowed in ('unknown', 'any'):
            allowed = list(sides)
        elif isinstance(allowed, str):
            allowed = [allowed]
        # Membership in the explicitly enumerated union; no partial T omission.
        values.append(float(max(sides[k] for k in allowed)))
    return values


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_digest(value):
    a = np.ascontiguousarray(value, dtype='<f8')
    h = hashlib.sha256(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder, output = (N / args.evaluation).resolve(), (OWN / args.output).resolve()
    if folder.parent != N.resolve() or output.parent != OWN.resolve() or output.exists():
        raise ValueError('Existing evaluation/new owned audit folder required')
    P, Z = read(folder / 'PROTOCOL.json'), read(folder / 'PREDICTION_SEAL.json')
    counts, failures = Counter(), []
    def check(name, condition, context=''):
        counts[name] += 1
        if not bool(condition):
            failures.append({'check': name, 'context': context})
    for name, key in [('PREDICTIONS.jsonl.gz', 'predictions_sha256'), ('PROTOCOL.json', 'protocol_sha256'), ('CONTRACT_CHECKS.json', 'contract_checks_sha256')]:
        check('prediction_seal_hash', sha(folder / name) == Z[key], name)
    check('complete98_3arm_untainted_seal', Z['rows'] == 98 and Z['all3arms_complete'] and Z['before_scoring'] and not Z['annotation_targets_accessed'])
    for rel, h in P['source_sha256'].items():
        check('source_current_hash', sha(ROOT / rel) == h, rel)
        check('source_executed_snapshot_hash', sha(folder / 'code_snapshot' / rel) == h, rel)
    for rel, h in P['input_and_model_sha256'].items():
        check('input_model_hash', sha(ROOT / rel) == h, rel)
    for rel, h in P['protected_user_sha256'].items():
        check('protected_user6_hash', sha(ROOT / rel) == h, rel)
    data = rows(folder / 'PREDICTIONS.jsonl.gz')
    check('complete_ordered98', len(data) == 98 and [r['id'] for r in data] == P['ids'] and len(set(P['ids'])) == 98)
    check('same_primary95_population', Counter(r['partition'] for r in data if r['cohort'] == 'PRIMARY_NONSEVERE') == {'CAL': 46, 'TEST': 49})
    cache = {r['id']: r for r in rows(N / 'fresh_observations_v2/OBSERVATIONS.jsonl.gz')}
    model = read(ROOT / P['calibration_model']['path'])
    interval_binding = P['interval_model_bindings']['interval_model']
    check('interval_model_file_binding', sha(ROOT / interval_binding['path']) == interval_binding['sha256'])
    interval = read(ROOT / interval_binding['path'])
    check('full_CAL_interval_model', interval['fit_partition'] == 'CAL' and interval['heldout_session'] is None and interval['eligible_point_count'] == 264 and not interval['future_coverage_guarantee'])
    check('interval_model_self_digest', digest({k:v for k,v in interval.items() if k != 'model_sha256'}) == interval['model_sha256'])
    oldbase_config = read(N / 'evaluation_v1/PROTOCOL.json')['solver_configuration']
    alias_sources = {'SELECTED_V4_NO_OT': {r['id']:r for r in rows(N / 'evaluation_bias_v4_1/PREDICTIONS.jsonl.gz')},
        'SELECTED_V4_NO_PRIOR': {r['id']:r for r in rows(N / 'evaluation_v1/PREDICTIONS.jsonl.gz')}}
    interval_counts = Counter()
    summary, arm_counts, extrema, changes = defaultdict(Counter), Counter(), defaultdict(float), []
    for row in data:
        rt, context = row['runtime'], row['id']
        g, orig = rt['initial_geometry'], cache[context]['runtime']
        xyz, K, hw = np.asarray(g['native_xyz']), np.asarray(g['K']), g['image_hw']
        normals = V.registered_incident_normals(xyz)
        measurements, fit = rt['frozen_visible_measurements'], rt['calibrated_visible_measurements']
        check('fresh_record_and_S_audit', row['fresh_current_RGB_full_API'] and row['fresh_cache_parity']['passed'] and row['S_contract_audit']['passed'] and not row['annotation_targets_read'], context)
        check('state_query_roles_only_conditional_T_abstention', all(a == b if a != 'T' else b in ('T','U') for a,b in zip(orig['states'],rt['states'])) and not rt['provenance']['state_producer_injected_for_tests'], context)
        gate = rt['truncation_interval_guard']
        check('interval_model_runtime_binding', rt['truncation_model_binding']['model_sha256'] == interval['model_sha256'] and gate['model_sha256'] == interval['model_sha256'], context)
        check('interval_SCALE_BI_Q_metadata_guards', gate['O_numeric_geometry_unchanged'] and gate['retained_T_numeric_geometry_and_crop_halfplanes_unchanged'] and gate['all_retained_OT_domains_rebuilt_and_bindings_validated'] and gate['unchanged_nonself_prediction_scale_px'] == orig['ot_domains']['nonself_prediction_scale_px'], context)
        check('initial_S_RGB_and_prompt_traces_cache_equal', rt['state_observation']['rgb_queried_ids'] == orig['state_observation']['rgb_queried_ids'] and rt['mask_audit'] == orig['mask_audit'] and rt['visible_correction'] == orig['visible_correction'], context)
        for k, before in enumerate(orig['states']):
            if before != 'T': continue
            interval_counts[row['partition'] + ':original_T'] += 1
            record = orig['state_observation']['records'][k]
            certified = record.get('physical_truncation_certified', False)
            raw = np.asarray(orig['original_raw_points'][k], float)
            missing = not np.isfinite(raw).all() or np.all(raw == -1)
            h,w = hw; radius = interval['radius_ratio'] * orig['ot_domains']['nonself_prediction_scale_px']
            distance = None if missing else float(np.linalg.norm(raw - np.clip(raw,[0,0],[w,h])))
            retained = certified or (not missing and distance > radius)
            check('independent_closed_disk_canvas_T_gate', rt['states'][k] == ('T' if retained else 'U'), context + '/KP' + str(k))
            current_record = rt['state_observation']['records'][k]
            check('T_history_and_abstention_metadata', current_record['original_T_hypothesis'] == record and current_record['pre_interval_state'] == 'T' and (retained or current_record.get('abstained') is True), context + '/KP' + str(k))
            interval_counts[row['partition'] + (':retained_T' if retained else ':demoted_T_to_U')] += 1
        old_domains = {d['native_corner']:d for d in orig['ot_domains']['domains']}
        keys = ('shape','center_xy','half_axes_px','angle_deg','bounds_xyxy','halfplanes','outside_canvas_side','outside_margin_px','canvas_hw','source_crop_validity_intersection')
        ids = [k for k,state in enumerate(rt['states']) if state in ('O','T')]
        bundle = rt['ot_domains']; binding = bundle['binding']
        effective = ['V' if state in ('B','I') else state for state in rt['states']]
        hidden = g['initial_self_hidden']; raw = np.asarray(rt['original_raw_points'],float).copy(); raw[:8][np.asarray(hidden,bool)] = [-1,-1]
        check('all_domains_new_content_digest', bundle['domains_sha256'] == digest(bundle['domains']), context)
        check('query_canvas_K_native_state_binding', binding['native_ids'] == ids and binding['states'] == effective and binding['image_hw'] == hw and binding['K_sha256'] == array_digest(g['K']) and binding['native_xyz_sha256'] == array_digest(xyz[ids]) and binding['original_queries_sha256'] == array_digest(raw[ids]) and binding['image_sha256'] == rt['visible_correction']['image_pixel_sha256'], context)
        for d in bundle['domains']:
            check('retained_O_T_all_numeric_geometry_frozen', all(d.get(k) == old_domains[d['native_corner']].get(k) for k in keys), context + '/KP' + str(d['native_corner']))
            check('per_domain_current_rebinding', d['binding'] == binding and d['state'] == rt['states'][d['native_corner']], context)

        check('raw_K_native_coarse_Q_cache_identity', rt['original_raw_points'] == orig['original_raw_points'] and g == orig['initial_geometry'], context)
        check('fresh_BI_observation_cache_parity', len(measurements) == len(orig['frozen_visible_measurements']) and all(a['native_corner'] == b['native_corner'] and a['source'] == b['source'] and np.allclose(a['xy'], b['xy'], atol=1e-6, rtol=0) and np.allclose(a['covariance_px2'], b['covariance_px2'], atol=1e-6, rtol=0) for a, b in zip(measurements, orig['frozen_visible_measurements'])), context)
        scale = rt['ot_domains']['nonself_prediction_scale_px']
        ratio = np.clip(scale / model['reference_scale_px'], .5, 2.)
        for a, b in zip(measurements, fit):
            group = model['groups'][a['state'] + (':fallback' if a['source'] == 'VISIBLE_BACKBONE_FALLBACK' else ':corrected')]
            expected = model['covariance_scale'] * np.asarray(a['covariance_px2']) + group['floor_px2'] * ratio ** 2 * np.eye(2)
            check('BI_observed_xy_Sigma_frozen', a['xy'] == b['xy'] and a['covariance_px2'] == b['covariance_px2'], context)
            check('fit_Sigma_reliability_scale_recomputed', np.allclose(expected, b['fit_covariance_px2'], atol=1e-12, rtol=0) and b['fit_reliability_weight'] == group['reliability_weight'], context)
            summary[row['partition']]['actual_supported_RGB_BI' if a['source'] != 'VISIBLE_BACKBONE_FALLBACK' else 'raw_BI_fallback'] += 1
        for arm, audit in row['control_input_audit'].items():
            states = ['U' if state in ('O','T') else state for state in rt['states']] if arm == 'SELECTED_V4_NO_OT' else rt['states']
            cfg = dict(P['solver_configuration']) if arm == 'SELECTED_V4_NO_OT' else dict(oldbase_config)
            if arm == 'SELECTED_V4_NO_PRIOR': cfg['prior_weight'] = 0.
            spec = dict(xyz=g['native_xyz'], K=g['K'], states=states, visible=fit,
                domains=[] if arm == 'SELECTED_V4_NO_OT' else rt['ot_domains']['domains'], image_hw=hw,
                coarse_pose=dict(R=g['R_cf'],t=g['centroid'],physical_registration_Q=g['physical_registration_Q']),
                initializations=None, config=cfg)
            check('control_complete_mathematical_spec_hash', digest(spec) == audit['mathematical_input_sha256'], context + '/' + arm)
            if audit['cached_identical_input_result']:
                old = alias_sources[arm][context]
                check('exact_sealed_control_alias_payload', row['controls'][arm] == old['controls'][arm] and audit['mathematical_input_sha256'] == old['control_input_audit'][arm]['mathematical_input_sha256'], context + '/' + arm)
        current_pose, no_ot_pose = rt['actual_pose'], row['controls']['SELECTED_V4_NO_OT']['actual_pose']
        if len(measurements) >= 3:
            changes.append({'id': context, 'partition': row['partition'], 'cohort': row['cohort'], 'OT_count': len(rt['ot_domains']['domains']),
                'R_equal': current_pose['R_cf'] == no_ot_pose['R_cf'], 't_equal': current_pose['centroid'] == no_ot_pose['centroid']})
        for arm in P['arms']:
            if arm == 'SELECTED_V4_ALL_OT':
                actual, states, sol, readout, domains = rt['actual_pose'], rt['states'], rt['joint_pose'], rt['keypoint_readout'], rt['ot_domains']['domains']
            else:
                c = row['controls'][arm]
                actual, states, sol, readout = c['actual_pose'], c['states'], c['solution'], c['keypoint_readout']
                domains = [] if arm == 'SELECTED_V4_NO_OT' else rt['ot_domains']['domains']
            cc = context + '/' + arm
            cfg = dict(P['solver_configuration'])
            if arm == 'SELECTED_V4_NO_PRIOR':
                cfg['prior_weight'] = 0.
            arm_counts[arm + (':accepted' if sol['accepted'] else ':fallback')] += 1
            check('individual_OT_and_BI_native_ids_cover_states', {d['native_corner'] for d in domains} == {i for i, s in enumerate(states) if s in ('O', 'T')} and {m['native_corner'] for m in fit} == {i for i, s in enumerate(states) if s in ('B', 'I')}, cc)
            R, t = np.asarray(actual['R_cf']), np.asarray(actual['centroid'])
            q, depth = project(xyz, R, t, K)
            check('native_Q_physical_pose', np.allclose(R @ g['physical_registration_Q'], actual['R_physical'], atol=1e-12, rtol=0) and actual['physical_registration_Q'] == g['physical_registration_Q'] and actual['cf_extents'] == g['cf_extents'], cc)
            check('rigid_model_readout_projection', np.allclose(q, readout['rigid_model_corners'], atol=1e-8, rtol=0), cc)
            for m in measurements:
                check('actual_BI_readout_preserved', readout['reported_points'][m['native_corner']] == m['xy'], cc)
            for i, state in enumerate(states):
                if state == 'S':
                    check('S_readout_projection_only', np.allclose(q[i], readout['reported_points'][i], atol=1e-8, rtol=0) and readout['point_sources'][i] == 'FINAL_PNP_S_PROJECTION', cc)
                if state == 'U' or (state in ('O', 'T') and not sol['accepted']):
                    check('unresolved_corners_not_fabricated', readout['reported_points'][i] is None, cc)
            if len(rt['original_raw_points']) == 9:
                check('ninth_center_metadata_frozen', readout['reported_points'][8] == rt['original_raw_points'][8], cc)
            check('S_terms0_no_hard_BI_budget_or_clamp', sol['provenance']['S_image_terms'] == 0 and sol['provenance']['S_region_terms'] == 0 and not sol['provenance']['B_I_absolute_covariance_residual_constraints'] and not sol['provenance']['seed_projection_protection_enabled'], cc)
            ambiguous = bool((sol.get('ambiguity') or {}).get('detected'))
            check('OT_flags_and_no_global_uniqueness', readout['OT_solved'] == (sol['accepted'] and bool(domains) and not ambiguous) and not readout['OT_unique_solution_claimed'], cc)
            if len(fit) == 2 and sol['accepted']:
                check('BI2_forced_family_and_OT_unresolved_as_unique', ambiguous and sol['ambiguity']['pose_family_from_two_BI'] and not readout['OT_solved'], cc)
            if not sol['accepted']:
                seed = sol.get('BI_prior_seed') or {}
                chosen = seed if seed.get('visible_support_guards_pass') and not seed.get('contains_underconstrained_pose_family') else {'R': g['R_cf'], 't': g['centroid']}
                check('explicit_allowed_fallback_Rt', actual['R_cf'] == chosen['R'] and actual['centroid'] == chosen['t'] and not sol['OT_constraints_satisfied'] and actual['fallback'], cc)
                continue
            check('accepted_solver_pose_and_support_flags', np.allclose(R, sol['R'], atol=0, rtol=0) and np.allclose(t, sol['t'], atol=0, rtol=0) and actual['joint_feasible'] and not actual['fallback'], cc)
            all_poses = [('selected', R, t)] + [('sample' + str(i), np.asarray(s['R']), np.asarray(s['t'])) for i, s in enumerate(sol['feasible_pose_samples'])]
            for label, RR, tt in all_poses:
                qq, zz = project(xyz, RR, tt, K)
                check('selected_and_samples_positive_depth', np.min(zz) >= cfg['positive_depth_m'] - 1e-10 and np.isfinite(qq).all(), cc + '/' + label)
                cam, h, w = -RR.T @ tt, *hw
                for m in fit:
                    i = m['native_corner']
                    direction = cam - xyz[i]
                    cosine = float(np.max(normals[i] @ direction / np.linalg.norm(direction)))
                    canvas = min(qq[i, 0] / w, qq[i, 1] / h, (w - qq[i, 0]) / w, (h - qq[i, 1]) / h)
                    check('selected_and_samples_BI_model_and_canvas', cosine + np.sin(np.deg2rad(cfg['geometry_tolerance_deg'])) >= -cfg['feasibility_tolerance'] and canvas >= -cfg['feasibility_tolerance'], cc + '/' + label)
                for d in domains:
                    slack = min(region_slacks(d, qq[d['native_corner']], hw, cfg))
                    check('every_selected_sample_OT_region_scalar', slack >= -cfg['feasibility_tolerance'], cc + '/' + label + '/KP' + str(d['native_corner']))
            data_loss = 0.
            for m in fit:
                err = q[m['native_corner']] - m['xy']
                white = np.linalg.solve(np.linalg.cholesky(m['fit_covariance_px2']), err)
                mah = float(np.linalg.norm(white))
                extrema['soft_fit_mahalanobis_max'] = max(extrema['soft_fit_mahalanobis_max'], mah)
                delta = cfg['huber_scale']
                data_loss += m['fit_reliability_weight'] * 2 * delta ** 2 * (np.sqrt(1 + mah ** 2 / delta ** 2) - 1)
            center, coarse_center = R @ xyz.mean(0) + t, np.asarray(g['R_cf']) @ xyz.mean(0) + g['centroid']
            residual = np.r_[Rotation.from_matrix(R @ np.asarray(g['R_cf']).T).as_rotvec(), center[:2] / center[2] - coarse_center[:2] / coarse_center[2], np.log(center[2] / coarse_center[2])]
            residual -= np.asarray(cfg.get('prior_bias6', [0.] * 6)) * cfg.get('bias_strength', 0.)
            white = np.linalg.solve(np.linalg.cholesky(cfg['prior_covariance6']), residual)
            prior_loss = cfg['prior_weight'] * float(white @ white)
            difference = abs(sol['objective'] - data_loss - prior_loss)
            extrema['objective_difference_max'] = max(extrema['objective_difference_max'], difference)
            check('soft_BI_prior_objective_recomputed', difference < 1e-7 and abs(sol['objective_components']['soft_BI_robust_loss'] - data_loss) < 1e-7 and abs(sol['objective_components']['coarse_pose_prior_loss'] - prior_loss) < 1e-7, cc)
    output.mkdir()
    result = {'schema': 'INDEPENDENT_SEALED_INTERVAL_RUNTIME_AUDIT_V4_2', 'evaluation': args.evaluation,
        'passed': not failures, 'checks_by_group': dict(counts), 'total_checks': sum(counts.values()), 'failures': failures,
        'rows': len(data), 'arm_counts': dict(arm_counts), 'BI_source_status_counts': {k: dict(v) for k, v in summary.items()},
        'extrema': dict(extrema), 'BI_ge3_same_pose_NO_OT_comparisons': changes,
        'source_hashes_live_and_snapshot_checked': len(P['source_sha256']), 'improvement_completion_claim': False,
        'interval_T_gate_counts': dict(interval_counts), 'audit_source_sha256': sha(Path(__file__)),
        'no_new_posefit_RGB_GPU_or_parameter_choice': True, 'coordinate_GT_reads_before_POSTSEAL': 0,
        'evidence_sha256': {n: sha(folder / n) for n in ('PROTOCOL.json', 'PREDICTION_SEAL.json', 'PREDICTIONS.jsonl.gz')}}
    post_path = folder / 'POSTSEAL_SCORING.json'
    if post_path.is_file():
        # The only reference access: after completed prediction AND existing score.
        post = read(post_path)
        labels = {}
        for part, b in post['label_bindings'].items():
            if sha(ROOT / b['path']) != b['sha256']:
                raise ValueError('Postseal reference binding mismatch')
            labels[part] = read(ROOT / b['path'])['rows']
        check('isolated_reference_population46_49', len(labels['CAL']) == 46 and len(labels['TEST']) == 49)
        scored = {r['id']: r for r in post['rows']}
        check('all98_postseal_metric_rows', set(scored) == {r['id'] for r in data})
        old_dir = BASE / 'posefix_sub_comparison_v1'
        old = defaultdict(dict)
        for r in rows(old_dir / 'PREDICTIONS.jsonl.gz'):
            names = {'JOINT_SINGLE_FAMILY': 'OLD_RGB_SINGLE_FAMILY', 'FUSED_ACTIVE': 'BOUNDARY_PROGRESSIVE'}
            if r['method'] in names:
                old[names[r['method']]][r['id']] = r['actual_pose']
        v3 = {r['id']: r['runtime']['actual_pose'] for r in rows(N.parent / 'evaluation_v3/PREDICTIONS.jsonl.gz') if r['cohort'] == 'PRIMARY_NONSEVERE'}
        old['FROZEN_STATEWISE_V3'] = v3
        accumulated, maxima = defaultdict(lambda: defaultdict(list)), defaultdict(float)
        for r in data:
            truth = labels[r['partition']][r['source_record_id']]['target']['truth']
            poses = {'ORIGINAL_N3_PNP': r['initial_pose'], 'SELECTED_V4_ALL_OT': r['runtime']['actual_pose'],
                     **{a: c['actual_pose'] for a, c in r['controls'].items()}}
            if r['cohort'] == 'PRIMARY_NONSEVERE':
                poses.update({a: items[r['id']] for a, items in old.items()})
            for arm, pose in poses.items():
                value, expected = metric(pose, truth), scored[r['id']]['metrics'][arm]
                for key in value:
                    diff = abs(float(value[key]) - float(expected[key]))
                    maxima[key] = max(maxima[key], diff)
                    check('every_physical_metric_row_recomputed', diff < 1e-10, r['id'] + '/' + arm + '/' + key)
                if r['cohort'] == 'PRIMARY_NONSEVERE':
                    accumulated[r['partition']][arm].append(value)
        physical_summary = {}
        for part, arms in accumulated.items():
            physical_summary[part] = {}
            for arm, values in arms.items():
                physical_summary[part][arm] = {'frames': len(values), 'success_count': sum(v['success_5cm_5deg'] for v in values),
                    'mean_translation_cm': float(np.mean([v['translation_cm'] for v in values])),
                    'mean_rotation_deg': float(np.mean([v['rotation_deg'] for v in values])),
                    'mean_ADDsym_cm': float(np.mean([v['ADDsym_cm'] for v in values]))}
                published = post['summary'][part]['pose'][arm]
                check('all_primary_means_success_population_recomputed', len(values) == published['frames'] and physical_summary[part][arm]['success_count'] == published['success_count'] and all(abs(np.mean([v[k] for v in values]) - published['metrics'][k]['mean']) < 1e-10 for k in value), part + '/' + arm)
        test, target = physical_summary['TEST']['SELECTED_V4_ALL_OT'], post['required_target']
        achieved = {'translation': test['mean_translation_cm'] <= target['TEST_translation_cm_mean_at_most'],
            'rotation': test['mean_rotation_deg'] <= target['TEST_rotation_deg_mean_at_most'],
            'success': test['success_count'] >= target['TEST_success_5cm_5deg_at_least']}
        check('predeclared_target_flags_recomputed', achieved == post['target_checks'] and all(achieved.values()) == post['target_achieved'])
        result.update(postseal_reference_access=True, physical_metric_max_difference=dict(maxima), physical_summary=physical_summary,
            target_checks=achieved, numeric_target_achieved=all(achieved.values()), goal_completion_if_targets_fail='must remain active',
            postseal_scoring_sha256=sha(post_path), historical_reference_not_independent_metrology_or_unseen_TEST=True)
    else:
        result.update(postseal_reference_access=False, metrics_not_read_or_computed=True)
    result.update(passed=not failures, checks_by_group=dict(counts), total_checks=sum(counts.values()), failures=failures)
    (output / 'AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'checks': result['total_checks'], 'failures': failures[:10],
        'evaluation': args.evaluation, 'target_checks': result.get('target_checks'), 'metric_max_difference': result.get('physical_metric_max_difference'),
        'arm_counts': dict(arm_counts), 'extrema': dict(extrema), 'output': str(output)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
