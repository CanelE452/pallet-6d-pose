"""Postseal virtual-corner and actual saved-pose evaluation, never prediction."""
from __future__ import annotations
import argparse
from collections import Counter
import math
from pathlib import Path
import time
import numpy as np
from scripts.research.pallet_accuracy_closeout_20261017_v1.baseline_audit import distribution
from scripts.research.pallet_accuracy_closeout_20261017_v1.evaluation import aggregate, paired
from scripts.research.pallet_statewise_end_to_end_recovery_v1.evaluator import _safe_metric, _status
from scripts.research.pallet_risk_information_v1.common import read, rows, save_rows, sha, write, csv_write
from .registry import ARMS, LEARNED, SEEDS, POPULATIONS, EXPECTED, contrasts


def project_reference(frame, reference):
    """One whole-object C2 correspondence; this is not physical edge truth."""
    native = np.asarray(frame['native_xyz'], float)
    pose = frame['initial_pose']; Q = np.asarray(pose['R_cf']).T @ np.asarray(pose['R_physical'])
    body = native @ Q; raw = np.asarray(frame['initial_points'])[:8]
    K = np.asarray(frame['K']); R = np.asarray(reference['R']); t = np.asarray(reference['t'])
    if reference['order'] != 2: raise ValueError('Original real C2 required')
    result = []
    for S in (np.eye(3), np.diag([-1., 1., -1.])):
        camera = body @ (R @ S).T + t; p = camera @ K.T
        valid = (camera[:, 2] > 0) & np.isfinite(p).all(1)
        xy = p[:, :2] / p[:, 2:]
        queries = valid & np.isfinite(raw).all(1) & ~np.all(raw == [-1., -1.], axis=1)
        cost = float(np.sum((xy[queries] - raw[queries]) ** 2)) if queries.any() else None
        result.append((xy, valid, cost))
    eligible = [i for i, v in enumerate(result) if v[2] is not None]
    if not eligible: return None, None
    phase = min(eligible, key=lambda i: (result[i][2], i))
    return result[phase], phase


def continuous(subset, before, after):
    out = {}
    for axis, threshold in (('translation_cm', 50.), ('translation_cm', 100.), ('rotation_deg', 45.)):
        key = ('T_gt' + str(int(threshold)) + 'cm') if axis == 'translation_cm' else 'R_gt45deg'
        present = [r for r in subset if r['metrics'][before][axis] is not None and r['metrics'][after][axis] is not None]
        out[key] = dict(common_available_n=len(present),
            births=[r['id'] for r in present if r['metrics'][before][axis] <= threshold < r['metrics'][after][axis]],
            resolutions=[r['id'] for r in present if r['metrics'][after][axis] <= threshold < r['metrics'][before][axis]])
    failures = [r for r in subset if not r['metrics'][before]['success'] and not r['metrics'][after]['success']]
    out['both_failure_continuous'] = {axis: distribution([
        r['metrics'][after][axis] - r['metrics'][before][axis] if r['metrics'][after][axis] is not None and
        r['metrics'][before][axis] is not None else None for r in failures], len(failures))
        for axis in ('translation_cm', 'rotation_deg')}
    return out


def _observation_summary(records):
    out = []
    for pop, partitions in POPULATIONS.items():
        for arm in LEARNED:
            for state in ('B', 'I'):
                rr = [r for r in records if r['partition'] in partitions and r['arm'] == arm and r['automatic_state'] == state]
                full = [r for r in rr if r['full_point_reference_error_px'] is not None]
                out.append(dict(population=pop, arm=arm, automatic_state=state, target_corners=len(rr),
                    reference_kind='VIRTUAL_PROJECTED_CUBOID_NOT_MATERIAL_EDGE',
                    reference_available=sum(r['q0_reference_error_px'] is not None for r in rr),
                    full_point_n=len(full), partial_n=sum(r['partial'] for r in rr),
                    changed_n=sum(r['changed'] for r in rr), q0_keep_n=sum(not r['changed'] and not r['partial'] for r in rr),
                    valid_alternative_corners=sum(r['alternatives'] > 0 for r in rr),
                    full_candidate_count=sum(r['candidate_count'] for r in rr),
                    before=distribution([r['q0_reference_error_px'] for r in full], len(rr)),
                    after=distribution([r['full_point_reference_error_px'] for r in full], len(rr)),
                    recovered_corners=sum(r['full_point_reference_error_px'] < r['q0_reference_error_px'] for r in full),
                    damaged_corners=sum(r['full_point_reference_error_px'] > r['q0_reference_error_px'] for r in full),
                    same_corners=sum(r['full_point_reference_error_px'] == r['q0_reference_error_px'] for r in full),
                    accepted_pose_subset_n=sum(r['pose_accepted'] for r in full),
                    accepted_pose_subset_after=distribution([r['full_point_reference_error_px'] for r in full if r['pose_accepted']],
                        sum(r['pose_accepted'] for r in full))))
    return out


def run(run_dir, reference_dir, upstream, bank_root, output):
    started = time.monotonic(); run_dir, output = Path(run_dir), Path(output)
    manifest = read(run_dir / 'PREDICTION_MANIFEST.json'); path = run_dir / 'PREDICTIONS.jsonl.gz'
    if manifest.get('real_reference_inputs') is not False or manifest.get('before_scoring') is not True:
        raise ValueError('Complete blind prediction seal is required before references')
    if manifest['experiment_plan_sha256'] != sha(run_dir / 'EXPERIMENT_PLAN.json'):
        raise ValueError('Fixed plan changed')
    if manifest['artifacts'][path.name] != sha(path): raise ValueError('Predictions changed')
    ledger = list(rows(path)); frames = {r['id']: r for r in rows(upstream)}
    if len(ledger) != 238 or len(set(r['id'] for r in ledger)) != 238 or set(frames) != {r['id'] for r in ledger}:
        raise ValueError('Full238 join required')
    if any(set(r['predictions']) != set(ARMS) for r in ledger): raise ValueError('Missing fixed arm')
    for binding in manifest.get('model_and_input_bindings', []):
        if sha(binding['path']) != binding['sha256']: raise ValueError('Bound source/model/input changed')
    reference_dir = Path(reference_dir); refpath = reference_dir / 'FIXED_REFERENCES.json'
    if sha(refpath) != read(reference_dir / 'SCORE_SEAL.json')['artifacts'][refpath.name]: raise ValueError('Reference changed')
    output.mkdir(parents=True, exist_ok=False)
    # First current real numeric-reference read in this campaign.
    references = read(refpath)['references']
    write(output / 'REFERENCE_OPEN_RECEIPT.json', dict(epoch=time.time(), prediction_manifest_sha256=sha(run_dir / 'PREDICTION_MANIFEST.json'),
        reference_sha256=sha(refpath), evaluator_source_sha256=sha(__file__)))
    scored = []; observation_records = []; metrics_csv = []
    for ordinal, row in enumerate(ledger):
        frame = frames[row['id']]; ref = references[row['partition']][row['id']]
        record = {k: row[k] for k in ('id', 'partition', 'session', 'cluster_id')}
        record.update(poses={}, metrics={}, status={})
        for arm in ARMS:
            entry = row['predictions'][arm]; pose = entry['actual_pose']; m = _safe_metric(pose, ref)
            record['poses'][arm] = pose; record['metrics'][arm] = m; record['status'][arm] = _status(entry)
            metrics_csv.append(dict(**{k: record[k] for k in ('id', 'partition', 'session', 'cluster_id')}, arm=arm, **m,
                fallback=record['status'][arm]['fallback'], raw_status=record['status'][arm]['status']))
        scored.append(record)
        from .bank import read as read_bank
        bank = read_bank(Path(bank_root) / 'frames' / 'real' / f'{ordinal:04d}' / 'BANK.json.gz')
        if bank['frame_id'] != row['id']: raise ValueError('Actual bank/frame order changed')
        aligned, phase = project_reference(frame, ref)
        for arm in LEARNED:
            entry = row['predictions'][arm]
            choices = {r['native_corner']: r for r in entry['choices']}
            for corner in bank['corners']:
                k = corner['native_corner']; state = bank['states'][k]
                if state not in ('B', 'I'): continue
                choice = choices[k]
                candidate = next(c for c in corner['candidates'] if c['candidate_id'] == choice['selected_candidate_id'])
                partial = corner.get('partial_line') is not None
                available = aligned is not None and bool(aligned[1][k])
                q0 = np.asarray(corner['q0']); q = np.asarray(candidate['applied_xy'])
                observation_records.append(dict(id=row['id'], partition=row['partition'], arm=arm, native_corner=k,
                    automatic_state=state, reference_kind='VIRTUAL_PROJECTED_CUBOID_NOT_MATERIAL_EDGE', reference_phase=phase,
                    candidate_count=len(corner['candidates']), alternatives=len(corner['candidates']) - 1,
                    selected_candidate_id=candidate['candidate_id'], selected_kind=candidate['kind'],
                    q0_x=float(q0[0]), q0_y=float(q0[1]), applied_x=float(q[0]) if not partial else None, applied_y=float(q[1]) if not partial else None,
                    move_px=float(np.linalg.norm(q-q0)) if not partial else None, changed=not partial and not np.array_equal(q, q0), partial=partial,
                    partial_normal_count=len(corner['partial_line']['lines']) if partial else 0,
                    covariance_xx=candidate['covariance_px2'][0][0] if not partial else None,
                    covariance_xy=candidate['covariance_px2'][0][1] if not partial else None,
                    covariance_yy=candidate['covariance_px2'][1][1] if not partial else None,
                    source_line_ids='|'.join(candidate['source_line_ids']), physical_source_ids='|'.join(candidate.get('physical_source_ids', [])),
                    q0_reference_error_px=float(np.linalg.norm(q0-aligned[0][k])) if available else None,
                    full_point_reference_error_px=float(np.linalg.norm(q-aligned[0][k])) if available and not partial else None,
                    pose_accepted=record['status'][arm]['accepted'] is True, pose_fallback=record['status'][arm]['fallback'],
                    pose_translation_cm=record['metrics'][arm]['translation_cm'], pose_rotation_deg=record['metrics'][arm]['rotation_deg']))
    subsets = {p: [r for r in scored if r['partition'] in parts] for p, parts in POPULATIONS.items()}
    if {p: len(rr) for p, rr in subsets.items()} != EXPECTED: raise ValueError('Reporting denominator mismatch')
    summary = {p: aggregate(rr, ARMS) for p, rr in subsets.items()}
    pp = {p: {after + '_minus_' + before: {**paired(rr, before, after), 'tail_transitions': continuous(rr, before, after)}
        for before, after in contrasts()} for p, rr in subsets.items()}
    obs_summary = _observation_summary(observation_records)
    save_rows(output / 'SCORED_ROWS.jsonl.gz', scored); csv_write(output / 'ROW_METRICS.csv.gz', metrics_csv)
    csv_write(output / 'OBSERVATION_ROWS.csv.gz', observation_records)
    write(output / 'OBSERVATION_SUMMARY.json', obs_summary); write(output / 'SUMMARY.json', summary); write(output / 'PAIRED.json', pp)
    flat = []
    for pop, ss in summary.items():
        for arm, v in ss.items():
            item = dict(population=pop, arm=arm, denominator=v['denominator'], successes=v['successes'], success_rate=v['success_rate'],
                no_pose=v['no_pose'], fallback=v['status_counts']['fallback']['true_n'], T_gt50cm=v['T_gt50cm'], T_gt100cm=v['T_gt100cm'], R_gt45deg=v['R_gt45deg'])
            for axis in ('translation_cm', 'rotation_deg'):
                item.update({axis + '_' + name: v[axis][name] for name in ('n', 'mean', 'sample_std', 'sample_variance', 'median', 'P90', 'P95', 'max')})
            flat.append(item)
    csv_write(output / 'COMPARISON_ALL_ARMS.csv', flat)
    artifacts = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    result = dict(schema='BI_LOCAL_OBSERVATION_POSTSEAL_EVALUATION_V1', status='COMPLETE', arms=list(ARMS), populations=EXPECTED,
        prediction_manifest_sha256=sha(run_dir / 'PREDICTION_MANIFEST.json'), real_reference_sha256=sha(refpath),
        source_sha256=sha(__file__), artifacts=artifacts, wall_seconds=time.monotonic()-started,
        reference_semantics='pose/cuboid projection only; no real physical edge or independent BI annotation certification', tail_trimming=0)
    write(output / 'EVALUATION_SEAL.json', result)
    print({p: {a: summary[p][a]['successes'] for a in ARMS} for p in summary})
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True); p.add_argument('--bank-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); run(a.run_dir,a.reference_dir,a.upstream,a.bank_root,a.output)


if __name__ == '__main__': main()
