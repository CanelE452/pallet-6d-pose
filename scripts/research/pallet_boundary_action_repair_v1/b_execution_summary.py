"""Postseal B execution/source/geometry ledger, never pose accuracy labels."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def same(a, b, scale):
    if a is None or b is None or not a.get('available', True) or not b.get('available', True):
        return False
    return bool(np.max(np.abs(np.asarray(a['R_physical'])-b['R_physical'])) <= 1e-7 and
                np.max(np.abs(np.asarray(a['centroid'])-b['centroid'])) <= 1e-7*scale)


def summarize(inputs, batch, output):
    started = time.monotonic(); batch = Path(batch)
    seal = json.loads((batch/'BATCH_SEAL.json').read_text())
    if not seal['complete'] or seal['predictions_sha256'] != sha(batch/'PREDICTIONS.jsonl.gz'):
        raise ValueError('Complete immutable238 batch required')
    frames = list(rows(inputs)); predictions = list(rows(batch/'PREDICTIONS.jsonl.gz'))
    if len(predictions) != 238 or {x['id'] for x in predictions} != {x['id'] for x in frames}:
        raise ValueError('Exact238 output coverage required')
    counts = defaultdict(Counter); details = []; max_move = defaultdict(lambda: defaultdict(float))
    for i, frame in enumerate(frames):
        folder = batch/'frames'/f'{i:04d}'; fseal = json.loads((folder/'FRAME_SEAL.json').read_text())
        if sha(folder/'RESULT.jsonl.gz') != fseal['result_sha256']:
            raise ValueError('Frame result changed')
        result = next(rows(folder/'RESULT.jsonl.gz'))
        scale = max(float(np.linalg.norm(np.ptp(np.asarray(f['native_xyz']), axis=0))) for f in frame['families'].values())
        e0 = next((c for c in frame['candidates'] if c.get('source') == 'E0'), None)
        cohorts = ['ALL238', 'CAL46' if frame['partition']=='CAL' else 'PRIMARY192', frame['partition']]
        cohorts = list(dict.fromkeys(cohorts))
        for cohort in cohorts:
            counts[cohort]['frames'] += 1
            counts[cohort]['raw_E0_source_role_missing'] += e0 is None
            counts[cohort]['fixed_profiles'] += result['fixed_universe']['profile_count']
            counts[cohort]['fixed_groups'] += result['fixed_universe']['group_count']
            counts[cohort]['B_original_input_supported_frames'] += bool(result['fixed_universe']['profile_count'])
        for arm, pred in result['predictions'].items():
            scores = result['finite_scores'][arm]
            flat = [s for row in scores for s in row.values()]
            alias_n0 = same(pred['actual_pose'], frame['fallback'], scale)
            alias_e0 = same(pred['actual_pose'], e0, scale)
            if pred['fallback']:
                if flat and not any(s['finite_support'] for s in flat):
                    cause = 'NO_FINITE_SUPPORT_FALLBACK'
                elif flat and not any(s['original_feasible'] for s in flat):
                    cause = 'NO_ORIGINAL_FEASIBLE_CANDIDATE_FALLBACK'
                elif not flat:
                    cause = 'NO_RETAINED_VALID_ENDPOINT_FALLBACK'
                else:
                    cause = 'NO_JOINT_SAME_POSE_FINITE_AND_ORIGINAL_FEASIBILITY_FALLBACK'
                source = 'N0_FALLBACK'
            elif alias_n0:
                cause = None; source = 'N0_EXACT_NUMERIC_ALIAS'
            elif alias_e0:
                cause = None; source = 'E0_EXACT_NUMERIC_ALIAS'
            elif pred['selected_candidate_id'].startswith('NEW_FINITE/'):
                cause = None; source = 'NEW_FINITE_ENDPOINT'
            else:
                cause = None; source = 'OTHER_ORIGINAL_CARRIED_CANDIDATE'
            for cohort in cohorts:
                c = counts[cohort]
                c[arm+':frames'] += 1
                c[arm+':fallback'] += bool(pred['fallback'])
                c[arm+':B_applied'] += bool(pred.get('boundary_applied'))
                c[arm+':new_endpoint_with_B_applied'] += source == 'NEW_FINITE_ENDPOINT' and bool(pred.get('boundary_applied'))
                c[arm+':new_endpoint_without_B_input'] += source == 'NEW_FINITE_ENDPOINT' and not bool(pred.get('boundary_applied'))
                c[arm+':source:'+source] += 1
                c[arm+':pose_alias_N0'] += alias_n0
                c[arm+':pose_alias_E0'] += alias_e0
                c[arm+':pose_alias_both_N0_E0'] += alias_n0 and alias_e0
                c[arm+':candidate_family_pairs'] += len(flat)
                c[arm+':joint_feasible_pairs'] += sum(s['feasible'] for s in flat)
                c[arm+':finite_rejected_pairs'] += sum(not s['finite_support'] for s in flat)
                c[arm+':original_rejected_pairs'] += sum(not s['original_feasible'] for s in flat)
                if cause:
                    c[arm+':fallback_reason:'+cause] += 1
            details.append(dict(id=frame['id'], partition=frame['partition'], arm=arm, source=source,
                fallback=pred['fallback'], canonical_fallback_reason=cause, raw_prediction_status=pred['prediction_status'],
                selected_candidate_id=pred['selected_candidate_id'], selected_family=pred.get('selected_family'),
                numerical_pose_alias_N0=alias_n0, numerical_pose_alias_E0=alias_e0,
                B_applied=pred.get('boundary_applied', False)))
        for arm, bank in result['banks'].items():
            valid_generated = [a for a in result['attempts'] if a['arm'] == arm and a['solve'].get('feasible')]
            direct_ids = {c['candidate_id'] for c in bank}
            dedup_n = 0; dedup_to_ineligible_n = 0; unrepresented_n = 0
            for candidate in valid_generated:
                if candidate['candidate_id'] in direct_ids:
                    continue
                representatives = [j for j,c in enumerate(bank) if same(candidate, c, scale)]
                if representatives:
                    dedup_n += 1
                    if not any(s['feasible'] for j in representatives for s in result['finite_scores'][arm][j].values()):
                        dedup_to_ineligible_n += 1
                else:
                    unrepresented_n += 1
            for cohort in cohorts:
                c = counts[cohort]
                c[arm+':retained_bank_poses'] += len(bank)
                c[arm+':feasible_generated_endpoints'] += len(valid_generated)
                c[arm+':generated_endpoints_represented_by_numeric_alias'] += dedup_n
                c[arm+':feasible_endpoint_deduplicated_to_ineligible_representative'] += dedup_to_ineligible_n
                c[arm+':feasible_generated_unrepresented'] += unrepresented_n
        for attempt in result['attempts']:
            sol = attempt['solve']; arm = attempt['arm']
            for cohort in cohorts:
                c = counts[cohort]
                c[arm+':attempts'] += 1
                c[arm+':actual_SLSQP_returned'] += 'optimizer_status' in sol
                c[arm+':technical_exception'] += 'exception' in sol
                c[arm+':optimizer_success'] += bool(sol.get('optimizer_success'))
                c[arm+':feasible_endpoint'] += bool(sol.get('feasible'))
                c[arm+':finite_supported_endpoint'] += bool(sol.get('finite_support'))
                c[arm+':restoration_iterations'] += sol.get('restoration_iterations', 0)
                c[arm+':SLSQP_iterations'] += sol.get('iterations', 0)
                c[arm+':recorded_in_solve_objective_calls'] += sol.get('calls', {}).get('objective', 0)
                c[arm+':recorded_in_solve_constraint_calls'] += sol.get('calls', {}).get('constraint', 0)
                c[arm+':restoration_backtracking_trials'] += sum(r['backtracking_trials'] for r in sol.get('restoration', []))
            for key, value in sol.get('cumulative_SE3', {}).items():
                max_move[arm][key] = max(max_move[arm][key], value)
    result = dict(schema='B_EXECUTION_SOURCE_LEDGER_V1', complete=True, geometry_only_no_accuracy_labels=True,
        inputs_sha256=sha(inputs), batch_seal_sha256=sha(batch/'BATCH_SEAL.json'),
        source_sha256=sha(__file__), counts={k: dict(v) for k, v in counts.items()},
        cumulative_SE3_maxima={k: dict(v) for k,v in max_move.items()}, rows=details,
        elapsed_seconds=time.monotonic()-started, target_or_reference_reads=0, new_optimization=0,
        actual_material_owner='UNVERIFIED', no_fallback_counted_as_new_B_correction=True,
        alias_rule='original strict numerical duplicate tolerance: max R-entry1e-7 and max center-axis1e-7*registered diagonal; C2 and 90 degree alternatives not collapsed; alias counts overlap, disjoint source category gives N0 precedence')
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True);p.add_argument('--batch-dir',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=summarize(a.inputs,a.batch_dir,a.output)
    print(json.dumps({'complete':r['complete'],'counts':r['counts'],'elapsed_seconds':r['elapsed_seconds']},indent=2))
