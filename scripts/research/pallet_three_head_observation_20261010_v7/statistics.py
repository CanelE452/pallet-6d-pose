"""Summarize new sealed scored rows; no inference, scoring or pose fitting."""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from . import common as C

METRICS = ('translation_cm', 'rotation_deg', 'ADDsym_cm')
UNITS = dict(translation_cm='cm', rotation_deg='degree', ADDsym_cm='cm')
SCOPES = ('common_operational', 'candidate_new_pose', 'both_new_pose')
BOOTSTRAP = C.REPO / '_docs/experiments/pallet_kp_difficulty_20261010_v1/BOOTSTRAP_SESSION_DRAWS.json.gz'


def compact_saved_witnesses(value):
    """Drop unused candidate lists only in RAM; original hashed rows stay intact."""
    if isinstance(value, dict):
        return {key: compact_saved_witnesses(item) for key, item in value.items()
                if key not in ('all_candidate_solutions', 'alternatives')}
    if isinstance(value, list):
        return [compact_saved_witnesses(item) for item in value]
    return value


def compact_rows(path):
    for row in C.rows(path):
        yield compact_saved_witnesses(row)


def moments(values, unit):
    a = np.asarray(values, np.float64)
    C.require(a.ndim == 1 and np.isfinite(a).all(), 'nonfinite recorded metric')
    n = len(a)
    return dict(n=n, mean=float(a.mean()) if n else None,
                sample_variance=float(a.var(ddof=1)) if n > 1 else None,
                sample_std=float(a.std(ddof=1)) if n > 1 else None,
                median=float(np.quantile(a, .5)) if n else None,
                P90=float(np.quantile(a, .9)) if n else None,
                max=float(a.max()) if n else None, unit=unit)


def metric(row, key):
    return float(row['pose']['ADDsym_m']) * 100 if key == 'ADDsym_cm' else float(row['pose'][key])


def summaries(rows):
    available = [r for r in rows if r['pose']['available']]
    new = [r for r in rows if r['new_pose_estimated'] and r['pose']['available']]
    fallback = [r for r in rows if r['fallback_used'] and r['pose']['available']]
    C.require(all(bool(r['pose_available']) == bool(r['pose']['available']) for r in rows),
              'recorded operational availability differs')
    C.require(all(not (r['new_pose_estimated'] and r['fallback_used']) for r in rows),
              'new/fallback statuses overlap')
    return dict(denominator=len(rows), operational=len(available), new_pose=len(new),
                fallback=len(fallback), no_pose=len(rows)-len(available),
                fixed_control=sum(r['output_status'] == 'FRESH_FIXED_CONTROL' for r in rows),
                statuses=dict(Counter(r['output_status'] for r in rows)),
                metrics={scope: {key: moments([metric(r, key) for r in selected], UNITS[key])
                                 for key in METRICS}
                         for scope, selected in [('operational', available), ('new_pose', new),
                                                 ('fallback', fallback)]})


def paired(candidate, comparator, ids, session_index, draws, scope):
    selected = [fid for fid in ids if candidate[fid]['pose']['available'] and
                comparator[fid]['pose']['available'] and
                (scope == 'common_operational' or candidate[fid]['new_pose_estimated']) and
                (scope != 'both_new_pose' or comparator[fid]['new_pose_estimated'])]
    inv = np.asarray([session_index[candidate[fid]['session']] for fid in selected], np.int64)
    counts = np.bincount(inv, minlength=13)
    denominator = draws @ counts
    nonempty = denominator > 0
    values = {}
    for key in METRICS:
        delta = [metric(candidate[fid], key) - metric(comparator[fid], key) for fid in selected]
        session_sum = np.array([math.fsum(v for j, v in enumerate(delta) if inv[j] == s)
                                for s in range(13)], float)
        sampled = (draws @ session_sum)[nonempty] / denominator[nonempty]
        values[key] = dict(unit=UNITS[key], n=len(delta),
                           mean_delta=float(math.fsum(delta)/len(delta)) if delta else None,
                           CI95=np.quantile(sampled, [.025, .975]).tolist() if len(sampled) else None,
                           bootstrap_nonempty_resamples=int(nonempty.sum()))
    return dict(scope=scope, denominator=len(ids), common_frames=len(selected), pair_ids=selected,
                candidate_new_pose_in_pairs=sum(candidate[fid]['new_pose_estimated'] for fid in selected),
                comparator_new_pose_in_pairs=sum(comparator[fid]['new_pose_estimated'] for fid in selected),
                bootstrap_nonempty_resamples=int(nonempty.sum()),
                bootstrap_empty_resamples=int((~nonempty).sum()), metrics=values)


def valid_point(points, k):
    return bool(np.isfinite(points[k]).all() and not np.all(points[k] == -1))


def posthoc(row, n3, observation):
    ref = row['evaluation_reference']
    C.require(ref['phase_control'] == 'N3_SUBPIX', 'posthoc phase is not fixed N3')
    gt = np.asarray(ref['native_points_px'], float)
    prior = np.asarray(n3['native_points'], float)
    out = np.asarray(row['native_points'], float)
    inputs = np.asarray(row.get('input_points', prior), float)
    C.require(gt.shape == (8, 2) and prior.shape == out.shape == inputs.shape == (9, 2),
              'posthoc native coordinate shapes differ')
    C.require(np.array_equal(prior, np.asarray(ref['initial_N3_native_points'], float), equal_nan=True),
              'recorded N3 phase coordinates differ')
    C.require(type(ref['matched']) is bool, 'matched-reference guard absent')
    ids = set(ref['valid_native_ids']) if ref['matched'] else set()
    states = ref['human_states_native']
    C.require(len(states) == 8 and ids <= set(range(8)), 'posthoc reference ID schema differs')
    solver = row.get('solver') or {}
    raw_final = list(solver.get('final_inliers', []))
    final = raw_final if row['new_pose_estimated'] else []
    used = list(solver.get('used', []))
    fit = list(solver.get('fit_input_ids', [])) if row['new_pose_estimated'] else []
    eligible = list(solver.get('eligible', range(8)))
    reproj = list(row.get('reprojected_ids', []))
    C.require(set(reproj) == (set(row['hidden_initial']) if row['hidden_reprojected'] else set()),
              'actual hidden reprojection IDs differ')
    C.require(not set(fit) & set(row['hidden_initial']), 'hidden input entered accepted final fit')
    corners = []
    for k in range(8):
        known = k in ids and np.isfinite(gt[k]).all()
        errors = {name: float(np.linalg.norm(points[k]-gt[k])) if known and valid_point(points, k) else None
                  for name, points in [('N3', prior), ('input', inputs), ('output', out)]}
        quality = ('UNKNOWN_REFERENCE_OR_INPUT' if errors['input'] is None else
                   'CORRECT_WITHIN8PX' if errors['input'] <= 8 else 'INCORRECT_OVER8PX')
        corners.append(dict(id=k, human_state=states[k], reference_known=known,
                            N3_error_px=errors['N3'], input_error_px=errors['input'],
                            output_error_px=errors['output'], input_quality=quality,
                            solver_eligible=k in eligible, solver_used=k in used,
                            accepted_new_pose_fit=k in fit, accepted_new_pose_final_inlier=k in final,
                            raw_candidate_final_inlier=k in raw_final,
                            actually_reprojected=k in reproj,
                            output_source=row.get('output_coordinate_sources', [None]*9)[k],
                            directly_visible_damage_5_to_10=bool(states[k] == 'DIRECT_VISIBLE' and
                                errors['N3'] is not None and errors['output'] is not None and
                                errors['N3'] <= 5 and errors['output'] > 10)))
    delta = {key: metric(row, key)-metric(n3, key) for key in METRICS} if (
        row['pose']['available'] and n3['pose']['available']) else None
    outcome = ('POSE_UNAVAILABLE' if delta is None else
               'BOTH_BETTER' if delta['translation_cm'] < 0 and delta['rotation_deg'] < 0 else
               'BOTH_WORSE' if delta['translation_cm'] > 0 and delta['rotation_deg'] > 0 else
               'MIXED_OR_UNCHANGED')
    audit = row['mask_audit']
    mask = 'MASK_NOT_APPLIED' if not audit['mask_applied'] else 'NO_KNOWN_VISIBILITY' if not audit['known_ids'] else (
        'WRONG_ON_KNOWN' if audit['mask_wrong_on_known'] else 'MATCHES_ON_KNOWN')
    contract = row.get('observation_contract') or {}
    spec = C.METHOD_SPECS.get(row['method'])
    arm, supply = (spec['arm'], spec['supply']) if spec else (None, 'NATIVE_N3')
    C.require(row['head_arm'] == arm and row['observation_supply'] == supply,
              'recorded head/supply differs from fixed method')
    observed = {int(c['id']): c for c in observation.get('corners', [])} if observation else {}
    admission = {int(c['id']): c for c in contract.get('corner_admission', [])}
    records = {int(c['id']): c for c in contract.get('cornerwise_records', [])}
    selected = set(row.get('selected_corner_ids', []))
    C.require(set(admission) == set(observed), 'computed admission/corner observations differ')
    C.require(len(admission) == len(contract.get('corner_admission', [])) and
              len(observed) == (len(observation.get('corners', [])) if observation else 0),
              'duplicate computed boundary IDs')
    if supply == 'BOUNDARY_ONLY':
        C.require(not records and contract['boundary_only_LOO_used_for_admission'] is False and
                  contract['native_N3_used_for_numeric_final_fit'] is False,
                  'sparse-only path used hybrid or native observation gate')
        C.require(selected == {k for k, v in admission.items() if v['accepted']},
                  'sparse pool differs from fixed admission')
    elif supply == 'CORNERWISE_HYBRID':
        C.require(selected == {k for k, v in records.items() if v['accepted']},
                  'hybrid pool differs from saved heldout decisions')
    else:
        C.require(not observed and not admission and not records and not selected,
                  'native control unexpectedly contains boundary proposals')
    boundary = []
    for k, gate in sorted(admission.items()):
        item = observed[k]
        candidate = np.asarray(item['xy'], float)
        C.require(candidate.shape == (2,) and np.isfinite(candidate).all(), 'bad decoded candidate')
        record = records.get(k)
        known = k in ids and np.isfinite(gt[k]).all()
        candidate_error = float(np.linalg.norm(candidate-gt[k])) if known else None
        n3_error = corners[k]['N3_error_px']
        accepted = k in selected and k not in set(row['hidden_initial'])
        reason = (gate['reason'] if not gate['accepted'] else
                  'INITIAL_SELF_HIDDEN' if k in row['hidden_initial'] else
                  'SPARSE_BOUNDARY_ADMITTED_WITHOUT_NATIVE_GATE' if supply == 'BOUNDARY_ONLY' else
                  record['reason'])
        boundary.append(dict(id=k, head_arm=arm, observation_supply=supply,
                             candidate_xy=candidate.tolist(), edges=list(item['edges']), radius_px=item['radius_px'],
                             admission_accepted=gate['accepted'], supply_selected=k in selected,
                             accepted=accepted, reason=reason, self_hidden_excluded=k in row['hidden_initial'],
                             displayed_in_final_output=corners[k]['output_source'] == 'VALIDATED_BOUNDARY_INTERSECTION',
                             accepted_new_pose_fit=accepted and k in fit,
                             accepted_new_pose_final_inlier=accepted and k in final,
                             candidate_error_px=candidate_error, N3_error_px=n3_error,
                             candidate_proxy_quality='UNKNOWN_REFERENCE' if candidate_error is None else
                             'CORRECT_WITHIN8PX' if candidate_error <= 8 else 'INCORRECT_OVER8PX',
                             delta_vs_N3_px=candidate_error-n3_error if known and n3_error is not None else None,
                             candidate_LOO_residual_px=record['candidate_LOO_residual_px'] if record else None,
                             native_N3_LOO_residual_px=record['native_N3_LOO_residual_px'] if record else None,
                             heldout_pose_calls=record['heldout_pose_calls'] if record else 0,
                             validation_initial_prior_is_shared=bool(record['initial_prior_includes_heldout_influence']) if record else False,
                             selection_validation='CORNERWISE_NATIVE_LOO' if record else 'NO_LOO_ADMISSION',
                             physical_boundary_ownership_independently_verified=False))
    pools = {}
    for name, pool_ids in [('eligible', eligible), ('solver_used', used), ('accepted_fit', fit),
                           ('accepted_final_inliers', final)]:
        C.require(set(pool_ids) <= set(range(8)), 'noncorner pool ID')
        counts = Counter(corners[k]['input_quality'] for k in pool_ids)
        pools[name] = dict(ids=pool_ids, quality_counts=dict(counts),
                           human_direct_visible_ids=[k for k in pool_ids if states[k] == 'DIRECT_VISIBLE'])
    return dict(id=row['id'], session=row['session'], method=row['method'], head_arm=arm,
                observation_supply=supply, selected_corner_ids=sorted(selected),
                stored_solver_geometry=solver.get('geometry'), output_status=row['output_status'],
                new_pose_estimated=row['new_pose_estimated'], fallback_used=row['fallback_used'],
                reference_phase='fixed N3_SUBPIX; existing GEOMETRIC_PROXY',
                final_inlier_scope='accepted NEW_POSE only; fallback solver candidate IDs separately recorded',
                corners=corners, pools=pools, boundary_candidates=boundary,
                mask_state=mask, mask_audit=audit, pose_outcome_vs_N3=outcome, delta_vs_N3=delta,
                hidden_initial=row['hidden_initial'], actually_reprojected_ids=reproj,
                hidden_set_changed=row['hidden_set_changed'])


def diagnostic_summary(rows):
    def errors(selected):
        return {name: moments([c[name+'_error_px'] for c in selected if c[name+'_error_px'] is not None], 'px')
                for name in ('N3', 'input', 'output')}
    corners = [c for r in rows for c in r['corners']]
    direct = [c for c in corners if c['human_state'] == 'DIRECT_VISIBLE']
    selfhidden = [c for c in corners if c['human_state'] == 'SELF_OCCLUDED']
    actual = [c for c in corners if c['actually_reprojected']]
    boundary = [c for r in rows for c in r['boundary_candidates']]
    adopted = [c for c in boundary if c['accepted']]
    displayed = [c for c in boundary if c['displayed_in_final_output']]
    def quality(data):
        valid = [c for c in data if c['delta_vs_N3_px'] is not None]
        return dict(candidates=len(data), known_N3_comparisons=len(valid),
                    candidate_proxy_quality_counts=dict(Counter(c['candidate_proxy_quality'] for c in data)),
                    candidate_error_px=moments([c['candidate_error_px'] for c in data
                                               if c['candidate_error_px'] is not None], 'px'),
                    improved=sum(c['delta_vs_N3_px'] < 0 for c in valid),
                    worsened=sum(c['delta_vs_N3_px'] > 0 for c in valid),
                    unchanged=sum(c['delta_vs_N3_px'] == 0 for c in valid),
                    delta_error_px=moments([c['delta_vs_N3_px'] for c in valid], 'px'),
                    reasons=dict(Counter(c['reason'] for c in data)))
    groups = {}
    for mask in ('WRONG_ON_KNOWN', 'MATCHES_ON_KNOWN', 'MASK_NOT_APPLIED', 'NO_KNOWN_VISIBILITY'):
        for outcome in ('BOTH_BETTER', 'BOTH_WORSE', 'MIXED_OR_UNCHANGED', 'POSE_UNAVAILABLE'):
            selected = [r for r in rows if r['mask_state'] == mask and r['pose_outcome_vs_N3'] == outcome]
            groups[mask+'__'+outcome] = dict(n=len(selected), ids=[r['id'] for r in selected],
                new_pose=sum(r['new_pose_estimated'] for r in selected),
                fallback=sum(r['fallback_used'] for r in selected))
    return dict(frames=len(rows), mask_and_pose_groups_vs_fixed_N3=groups,
                boundary_supply_counts=dict(computed=len(boundary),
                    admission_accepted=sum(c['admission_accepted'] for c in boundary),
                    selected_including_H=sum(c['supply_selected'] for c in boundary),
                    selected_excluded_H=sum(c['supply_selected'] and c['self_hidden_excluded'] for c in boundary),
                    selected_nonhidden=sum(c['accepted'] for c in boundary),
                    accepted_NEW_fit=sum(c['accepted_new_pose_fit'] for c in boundary),
                    accepted_NEW_final_inlier=sum(c['accepted_new_pose_final_inlier'] for c in boundary),
                    actually_displayed=sum(c['displayed_in_final_output'] for c in boundary)),
                DIRECT_VISIBLE=dict(corners=len(direct), errors=errors(direct),
                                    damage_5px_to_over10px=sum(c['directly_visible_damage_5_to_10'] for c in direct)),
                SELF_OCCLUDED=dict(corners=len(selfhidden), errors=errors(selfhidden)),
                ACTUALLY_REPROJECTED=dict(corners=len(actual), errors=errors(actual),
                                         unknown_output_error=sum(c['output_error_px'] is None for c in actual)),
                all_candidate_boundary_quality=quality(boundary), adopted_boundary_quality=quality(adopted),
                displayed_boundary_quality=quality(displayed),
                hidden_set_changed_frames=sum(r['hidden_set_changed'] for r in rows),
                causal_boundary_ownership_error_fraction_identified=False)


def compute(new, fixed, observations, cohort, bootstrap):
    ids = list(cohort['ids'])
    C.require(len(ids) == 245 and len(set(ids)) == 245, 'same245 cohort required')
    methods = ('BASE', 'N3_SUBPIX') + tuple(C.METHODS)
    groups = {m: {} for m in methods}
    for row in fixed + new:
        C.require(row['method'] in groups and row['id'] not in groups[row['method']], 'method/ID duplicate')
        groups[row['method']][row['id']] = row
    C.require(all(set(group) == set(ids) for group in groups.values()), 'method cohort differs')
    sessions = bootstrap['sessions']
    draws = np.asarray(bootstrap['counts'], np.int64)
    C.require(draws.shape == (10000, 13) and len(set(sessions)) == 13 and
              (draws >= 0).all() and (draws.sum(1) == 13).all(), 'frozen session draws differ')
    session_index = {s: i for i, s in enumerate(sessions)}
    labels = {r['id']: r['label'] for r in cohort['frames']}
    contrasts = tuple(C.CONTRASTS)
    C.require(len(contrasts) == len(set(contrasts)) == 16, 'fixed sixteen contrasts required')
    C.require(len(new) == 1960 and len(fixed) == 490, 'eight-arm245 scored population differs')
    obs = {(r['head_arm'], r['id']): r for r in observations}
    C.require(len(obs) == len(observations) == 735 and
              all({fid for arm, fid in obs if arm == a} == set(ids) for a in C.ARMS) and
              all(r['GT_input'] is False for r in observations), 'three-head observation population differs')
    strata = {}
    diagnostic_rows = [posthoc(row, groups['N3_SUBPIX'][row['id']],
                              obs[(row['head_arm'], row['id'])] if row['head_arm'] else None)
                       for row in new]
    for name, selected in [('combined', ids), ('easy', [fid for fid in ids if labels[fid] == 'clean']),
                           ('medium', [fid for fid in ids if labels[fid] == 'moderate'])]:
        selected_set = set(selected)
        strata[name] = dict(frames=len(selected), ids=selected,
            methods={m: summaries([groups[m][fid] for fid in selected]) for m in methods},
            contrasts={a+'_minus_'+b: {scope: paired(groups[a], groups[b], selected, session_index, draws, scope)
                                      for scope in SCOPES} for a, b in contrasts},
            diagnostics={m: diagnostic_summary([r for r in diagnostic_rows
                                               if r['method'] == m and r['id'] in selected_set])
                         for m in C.METHODS})
    primary = strata['combined']['contrasts'][C.PRIMARY+'_minus_N3_SUBPIX']['common_operational']
    complete = primary['common_frames'] == 245
    better = complete and all(primary['metrics'][k]['mean_delta'] < 0
                              for k in ('translation_cm', 'rotation_deg'))
    return dict(schema='fixed_three_head_two_supply_saved_row_statistics_v7', complete=True,
                population=dict(frames=245, clean=153, moderate=92, severe_excluded=74,
                                all_frames_retained=True, sessions=13),
                primary_method=C.PRIMARY, comparator='N3_SUBPIX',
                methods=strata['combined']['methods'], contrasts=strata['combined']['contrasts'], strata=strata,
                reference='GEOMETRIC_PROXY DEV; independent physical pose truth not validated',
                verdict=dict(primary_full_operational_T_and_R_improved=bool(better),
                             all245_paired_outputs_available=complete,
                             independent_generalization_established=False),
                bootstrap=dict(resamples=10000, sessions=sessions, new_draws_generated=0,
                               serialized_raw_sha256=bootstrap['serialized_raw_sha256']),
                actual_arithmetic=dict(saved_scored_rows=len(new)+len(fixed), posthoc_rows=len(diagnostic_rows),
                                      source_observation_rows=len(observations), strata=3,
                                      contrasts_per_stratum=16, paired_scopes=3, metric_CIs_requested=432),
                new_detector_head_PnP_optimizer_ray_training_RGB_scoring_calls=0), diagnostic_rows


def run(args):
    C.verify_protocol(args)
    C.protect(args)
    folder = Path(args.input)
    seal = C.read(folder / 'GEOMETRY_SEAL.json')
    receipt = C.read(folder / 'SCORING_RECEIPT.json')
    C.require(seal['complete'] and receipt['complete'] and receipt['scored_method_rows'] == 1960 and
              receipt['fixed_control_rows'] == 490 and receipt['geometry_seal'] == C.binding(folder/'GEOMETRY_SEAL.json'),
              'complete same-seal scored population required')
    C.bound(folder/'V4_CONTROL_PARITY.json', receipt['old_v4_control_parity'], 'unchanged old-v4 control parity')
    C.bound(folder/'INFERENCE_RECEIPT.json', receipt['inference_receipt'], 'completed inference receipt')
    inference = C.read(folder/'INFERENCE_RECEIPT.json')
    C.require(inference['complete'] is True and inference['cleanup_error'] is None and
              inference['actual_complete_frames'] == 245 and inference['method_rows'] == 1960 and
              inference['fixed_rows'] == 490, 'inference completion/cleanup did not pass')
    C.require(C.read(folder/'V4_CONTROL_PARITY.json')['passed'] is True and
              receipt['old_v4_control_parity_before_reference_reads'] is True,
              'old-v4 control pre-reference parity failed')
    for key, filename in [('predictions', 'PREDICTIONS.jsonl.gz'), ('fixed_predictions', 'FIXED_PREDICTIONS.jsonl.gz')]:
        C.bound(folder/filename, receipt[key], 'scored '+key)
    C.bound(folder/'OBSERVATIONS.jsonl.gz', seal['observations'], 'frozen three-head observations')
    observations = list(compact_rows(folder/'OBSERVATIONS.jsonl.gz'))
    new = list(compact_rows(folder/'PREDICTIONS.jsonl.gz'))
    fixed = list(compact_rows(folder/'FIXED_PREDICTIONS.jsonl.gz'))
    cohort = C.read(args.cohort)
    with gzip.open(BOOTSTRAP, 'rt') as stream:
        bootstrap = json.load(stream)
    result, posthoc_rows = compute(new, fixed, observations, cohort, bootstrap)
    result['bindings'] = {name: C.binding(path) for name, path in dict(
        observations=folder/'OBSERVATIONS.jsonl.gz',
        new_scored_rows=folder/'PREDICTIONS.jsonl.gz', fixed_scored_rows=folder/'FIXED_PREDICTIONS.jsonl.gz',
        scoring_receipt=folder/'SCORING_RECEIPT.json', geometry_seal=folder/'GEOMETRY_SEAL.json',
        inference_receipt=folder/'INFERENCE_RECEIPT.json',
        old_v4_control_parity=folder/'V4_CONTROL_PARITY.json',
        cohort=args.cohort, bootstrap=BOOTSTRAP, statistics_code=Path(__file__), protocol=args.protocol).items()}
    result['method_sources'] = {m: dict(path=result['bindings']['fixed_scored_rows' if m in ('BASE','N3_SUBPIX')
                                                        else 'new_scored_rows']['path'], raw_method=m)
                                for m in ('BASE','N3_SUBPIX')+tuple(C.METHODS)}
    result['saved_row_memory_policy'] = dict(original_full_serialized_rows_preserved=True,
        complete_raw_SHA_checked_before_loading=True,
        unused_keys_recursively_discarded_in_RAM=['all_candidate_solutions', 'alternatives'],
        required_coordinate_pool_and_LOO_ID_witnesses_retained=True)
    for name in ('METRICS.json', 'POSTHOC_ROWS.jsonl.gz', 'METRICS.csv'):
        C.output_path(args, name)
    C.save_rows(C.output_path(args, 'POSTHOC_ROWS.jsonl.gz'), posthoc_rows)
    result['posthoc_rows'] = C.binding(Path(args.output)/'POSTHOC_ROWS.jsonl.gz')
    C.write_new(C.output_path(args, 'METRICS.json'), result)
    with C.output_path(args, 'METRICS.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('stratum','method','scope','metric','n','mean','sample_variance','sample_std','median','P90','max','unit'))
        for stratum, data in result['strata'].items():
            for method, summary in data['methods'].items():
                for scope, values in summary['metrics'].items():
                    for key, value in values.items():
                        writer.writerow([stratum,method,scope,key]+[value[k] for k in
                            ('n','mean','sample_variance','sample_std','median','P90','max','unit')])
    C.verify_protocol(args)
    C.protect(args)
    print(json.dumps(dict(primary=result['verdict'], methods=result['methods'])), flush=True)


def main():
    parser = C.parser(__doc__)
    parser.add_argument('--input', default=str(C.PRIVATE/'accuracy'))
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
