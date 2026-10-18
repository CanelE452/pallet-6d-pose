"""Join sealed prediction-only outputs. This program never reads references.

Historical six-arm and v1 journals contain already executed outputs. This join
copies their actual poses and statuses, not their evaluation or oracle labels.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import read, rows, save_rows, sha, write
from .registry import ALL_ARMS, B_ARMS, L_ARMS, BASELINES, HISTORICAL


def bound(path, seal, keys):
    value = read(seal); digest = sha(path)
    possibilities = [value.get(k) for k in keys]
    artifacts = value.get('artifacts', {})
    item = artifacts.get(Path(path).name)
    possibilities.append(item.get('sha256') if isinstance(item, dict) else item)
    if digest not in possibilities:
        raise ValueError('Unbound prediction bytes: ' + str(path))
    return value


def indexed(path):
    values = list(rows(path)); result = {r['id']: r for r in values}
    if len(result) != len(values):
        raise ValueError('Duplicate frame ID: ' + str(path))
    return result


def run(run_dir, old_run, original_score, fair_run):
    start = time.time(); run_dir = Path(run_dir); old_run = Path(old_run)
    plan = read(run_dir/'PLAN_LOCK.json')
    if plan.get('new_primary_scoring_allowed') is not False:
        raise ValueError('Plan must have been frozen before reference scoring')
    files = dict(metadata=old_run/'inputs_all_available_v1/PREDICTION_INPUTS238.jsonl.gz',
        B=run_dir/'b_finite_v1/PREDICTIONS.jsonl.gz',
        L=run_dir/'action_predictions_v1/PREDICTIONS.jsonl.gz',
        real=run_dir/'d1_real_candidates_v1/HEAD_FEATURES.jsonl.gz',
        old=old_run/'PREDICTIONS.jsonl.gz',
        baseline=Path(original_score)/'BASELINE_PREDICTIONS.jsonl.gz',
        posefix=Path(fair_run)/'posefix192_v1/PREDICTIONS.jsonl.gz')
    bseal = bound(files['B'], files['B'].parent/'BATCH_SEAL.json', ['predictions_sha256'])
    lseal = bound(files['L'], files['L'].parent/'PREDICTION_SEAL.json', ['predictions_sha256'])
    real_seal = bound(files['real'], files['real'].parent/'FEATURE_SEAL.json', ['head_features_sha256'])
    bound(files['old'], old_run/'PREDICTION_SEAL.json', ['predictions_sha256'])
    bound(files['baseline'], Path(original_score)/'SCORE_SEAL.json', [])
    bound(files['posefix'], files['posefix'].parent/'PREDICTION_SEAL.json', ['predictions_sha256', 'predictions_file_sha256'])
    if lseal['features_sha256'] != sha(files['real']) or not lseal['all_2x2_same_common_real_features']:
        raise ValueError('All models must consume the same new real bank')
    if not bseal['complete'] or not lseal['complete'] or not real_seal['complete']:
        raise ValueError('Incomplete prediction branch')
    data = {k: indexed(v) for k, v in files.items() if k != 'posefix'}
    pf = {}
    for r in rows(files['posefix']):
        if r['method'] in pf.setdefault(r['id'], {}): raise ValueError('Duplicate PoseFix output')
        pf[r['id']][r['method']] = r
    ids = list(data['metadata'])
    if len(ids) != 238 or Counter(r['partition'] for r in data['metadata'].values()) != Counter(TEST=49, REMAINDER=143, CAL=46):
        raise ValueError('Exact original 238 identities required')
    for name, values in data.items():
        if set(values) != set(ids): raise ValueError('Population mismatch: ' + name)
    if set(pf) != set(ids): raise ValueError('PoseFix population mismatch')
    result = []
    for fid in ids:
        meta = data['metadata'][fid]
        identity = {k: meta[k] for k in ('id', 'partition', 'session', 'cluster_id')}
        preds = {}
        for key, expected in [('B', B_ARMS), ('L', L_ARMS)]:
            row = data[key][fid]
            if any(row[k] != identity[k] for k in identity):
                raise ValueError('Canonical identity mismatch: ' + key + '/' + fid)
            if set(row['predictions']) != set(expected): raise ValueError('Unfinished or changed method inventory')
            preds.update(row['predictions'])
        for arm in BASELINES:
            source = pf[fid][arm] if arm.startswith('POSEFIX_SUBPIX_') else data['baseline'][fid]['predictions'][arm]
            preds[arm] = source if 'actual_pose' in source else dict(source, actual_pose=source['pose'])
        for name, source in HISTORICAL.items(): preds[name] = data['old'][fid]['predictions'][source]
        feature = data['real'][fid]
        preds['E0_REGENERATED'] = dict(actual_pose=feature['e0_regenerated'],
            selected_candidate_id='E0_REGENERATED',action_role='E0',
            fallback=feature['e0_regenerated'].get('fallback'),accepted=feature['e0_regenerated'].get('joint_feasible'),
            joint_feasible=feature['e0_regenerated'].get('joint_feasible'),
            prediction_status='COMMON_FIXED_AUTOMATIC_E_OUTPUT_ACTUAL_OR_EXACT_STAGE_REUSED')
        if set(preds) != set(ALL_ARMS): raise ValueError('Final arm inventory changed')
        scale=max(float(np.linalg.norm(np.ptp(np.asarray(f['native_xyz']),axis=0)))for f in meta['families'].values())
        result.append(dict(**identity,object_scale_m=scale,predictions={a: preds[a] for a in ALL_ARMS}, reference_inputs=False))
    save_rows(run_dir/'PREDICTIONS.jsonl.gz', result)
    write(run_dir/'PREDICTION_SEAL.json', dict(schema='BOUNDARY_ACTION_ALL_PREDICTIONS_V1', complete=True,
        rows=238, arms=ALL_ARMS, ordered_ids=ids, predictions_sha256=sha(run_dir/'PREDICTIONS.jsonl.gz'),
        prediction_filename='PREDICTIONS.jsonl.gz', plan_lock_sha256=sha(run_dir/'PLAN_LOCK.json'),
        evaluation_protocol_sha256=sha(run_dir/'EVALUATION_PROTOCOL.json'),
        before_scoring=True, reference_inputs=False, new_reference_reads=0,
        source_sha256=sha(__file__), input_bindings={k:sha(v) for k,v in files.items()},
        input_seals=dict(B=sha(files['B'].parent/'BATCH_SEAL.json'), L=sha(files['L'].parent/'PREDICTION_SEAL.json'),
            common_real=sha(files['real'].parent/'FEATURE_SEAL.json')),
        same_common_real_bank_all_L=True, elapsed_seconds=time.time()-start, seal_epoch=time.time()))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--old-run',type=Path,required=True)
    p.add_argument('--original-score',type=Path,required=True);p.add_argument('--fair-run',type=Path,required=True)
    run(**vars(p.parse_args()))


if __name__=='__main__': main()
