"""Post-seal scoring with the unchanged current physical-C2 references."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import itertools
import json
import math
from pathlib import Path
import statistics
import time

BACKBONES = ('yolo', 'dope', 'resnet18')
SEEDS = (20261018, 20261019, 20261020)
ARMS = ('N0', 'E0') + tuple('D1_H1_MLP_s' + str(s) for s in SEEDS)
PARTITIONS = {'PRIMARY192': ('TEST', 'REMAINDER'), 'TEST49': ('TEST',),
              'REMAINDER143': ('REMAINDER',), 'CAL46': ('CAL',)}


def write_csv(path, data):
    data = list(data)
    if not data:
        raise ValueError('Do not write an empty result table')
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)


def quantile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = (len(values)-1)*p
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi]-values[lo])*(index-lo)


def distribution(values, full_n):
    v = [x for x in values if x is not None]
    return dict(n=len(v), missing=full_n-len(v), mean=statistics.fmean(v) if v else None,
        sample_variance=statistics.variance(v) if len(v)>1 else None,
        sample_std=statistics.stdev(v) if len(v)>1 else None,
        median=statistics.median(v) if v else None, P90=quantile(v, .9), P95=quantile(v, .95), max=max(v) if v else None)


def summarize(rows):
    n = len(rows)
    d = dict(denominator=n, successes=sum(r['success'] for r in rows),
        success_rate_percent=100*sum(r['success'] for r in rows)/n,
        no_pose=sum(not r['pose_available'] for r in rows))
    for prefix, field in [('T_cm', 'translation_cm'), ('R_deg', 'rotation_deg')]:
        d.update({prefix+'_'+k: v for k, v in distribution([r[field] for r in rows], n).items()})
    d.update(T_gt50=sum(r['translation_cm'] is not None and r['translation_cm']>50 for r in rows),
        T_gt100=sum(r['translation_cm'] is not None and r['translation_cm']>100 for r in rows),
        R_gt45=sum(r['rotation_deg'] is not None and r['rotation_deg']>45 for r in rows))
    assert d['T_cm_n']+d['no_pose']==n and d['R_deg_n']+d['no_pose']==n
    return d


def transition(before, after):
    if set(before) != set(after):
        raise ValueError('Pairing must retain every identical frame ID')
    groups = {k: [] for k in ('BOTH_SUCCESS', 'RECOVERY', 'DAMAGE', 'BOTH_FAILURE')}
    for ident in before:
        a, b = before[ident]['success'], after[ident]['success']
        key = 'BOTH_SUCCESS' if a and b else 'RECOVERY' if b else 'DAMAGE' if a else 'BOTH_FAILURE'
        groups[key].append(ident)
    counts = {k: len(v) for k, v in groups.items()}
    delta = sum(r['success'] for r in after.values())-sum(r['success'] for r in before.values())
    assert sum(counts.values()) == len(before) and delta == counts['RECOVERY']-counts['DAMAGE']
    result = dict(denominator=len(before), before_successes=sum(r['success'] for r in before.values()),
        after_successes=sum(r['success'] for r in after.values()), **counts, success_difference_count=delta,
        success_difference_pp=100*delta/len(before))
    for name in ('translation_cm', 'rotation_deg'):
        common = [(before[i][name], after[i][name]) for i in before
                  if before[i][name] is not None and after[i][name] is not None]
        result[name+'_common_n'] = len(common)
        result[name+'_mean_difference_after_minus_before'] = statistics.fmean(b-a for a,b in common) if common else None
    return result, groups


def run(private, original_d1, original_score, output):
    from scripts.research.pallet_boundary_action_repair_v1.action_data import rows, read, sha, write
    from scripts.research.pallet_accuracy_closeout_20261017_v1.baseline_audit import load_bound_original, metric
    private, original_d1, original_score, output = map(Path, (private, original_d1, original_score, output))
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    # No reference file is opened before every backbone's predictions are sealed.
    ps = {}
    for backbone in BACKBONES:
        path = private/backbone/'predictions'
        seal = read(path/'PREDICTION_SEAL.json')
        if not seal['complete'] or seal['rows'] != 238 or sha(path/'PREDICTIONS.jsonl.gz') != seal['predictions_sha256']:
            raise ValueError('All three complete prediction seals must precede scoring')
        ps[backbone] = dict(seal_sha256=sha(path/'PREDICTION_SEAL.json'), predictions_sha256=seal['predictions_sha256'],
            features_sha256=seal['features_sha256'], feature_seal_sha256=seal['feature_seal_sha256'], checkpoints=seal['checkpoints'])
    if any(ps[b]['checkpoints'] != ps['yolo']['checkpoints'] for b in BACKBONES[1:]):
        raise ValueError('Every backbone must use exactly the same original H1 checkpoints and normalizers')
    original, references, reference_bindings = load_bound_original(original_score)
    ids = [r['id'] for r in original]
    meta = {r['id']: {k:r[k] for k in ('partition','session','cluster_id')} for r in original}
    ledger, coverage, feature_bindings = [], [], {}
    original_prediction_dir = original_d1/'action_predictions_v1'
    original_prediction_seal = read(original_prediction_dir/'PREDICTION_SEAL.json')
    if sha(original_prediction_dir/'PREDICTIONS.jsonl.gz') != original_prediction_seal['predictions_sha256']:
        raise ValueError('The original YOLO parity journal must match its original seal')
    original_predictions = {r['id']:r for r in rows(original_prediction_dir/'PREDICTIONS.jsonl.gz')}
    exact_yolo_poses_checked = 0
    for backbone in BACKBONES:
        base = private/backbone
        features_path = base/'candidates/HEAD_FEATURES.jsonl.gz'
        features_seal = read(base/'candidates/FEATURE_SEAL.json')
        if sha(features_path) != features_seal['head_features_sha256']:
            raise ValueError('Feature bank changed after prediction')
        if ps[backbone]['features_sha256'] != sha(features_path) or ps[backbone]['feature_seal_sha256'] != sha(base/'candidates/FEATURE_SEAL.json'):
            raise ValueError('Prediction ancestry does not match the actual scored feature bank')
        feature_bindings[backbone] = dict(features_sha256=sha(features_path), feature_seal_sha256=sha(base/'candidates/FEATURE_SEAL.json'))
        features = {r['id']:r for r in rows(features_path)}
        predictions = list(rows(base/'predictions/PREDICTIONS.jsonl.gz'))
        if [r['id'] for r in predictions] != ids or set(features) != set(ids):
            raise ValueError('The exact current238 IDs/order are required for every backbone')
        for row in predictions:
            ident = row['id']; f = features[ident]
            if any(row[k] != meta[ident][k] for k in meta[ident]):
                raise ValueError('Frozen population metadata changed')
            poses = dict(N0=f.get('fallback', {'available':False}), E0=f.get('e0_regenerated', {'available':False}))
            poses.update({a:row['predictions'][a]['actual_pose'] for a in ARMS if a.startswith('D1_')})
            if backbone == 'yolo':
                for arm in ARMS[2:]:
                    if poses[arm] != original_predictions[ident]['predictions'][arm]['actual_pose']:
                        raise ValueError('YOLO original D1 exact returned-pose parity failed')
                    exact_yolo_poses_checked += 1
            ref = references[meta[ident]['partition']][ident]
            for arm, pose in poses.items():
                ledger.append(dict(backbone=backbone, arm=arm, id=ident, **meta[ident], **metric(pose,ref)))
            eligible = [c for c in f['candidates'] if c['eligible'] and c['actual_pose'].get('available')]
            coverage.append(dict(backbone=backbone, id=ident, **meta[ident],
                initial_pose_available=bool(poses['N0'].get('available')), candidates=len(f['candidates']), eligible_candidates=len(eligible),
                eligible_success_exists=any(metric(c['actual_pose'],ref)['success'] for c in eligible)))
    summaries = []
    for population, partitions in PARTITIONS.items():
        for backbone, arm in itertools.product(BACKBONES, ARMS):
            group = [r for r in ledger if r['partition'] in partitions and r['backbone']==backbone and r['arm']==arm]
            expected = {'PRIMARY192':192, 'TEST49':49, 'REMAINDER143':143, 'CAL46':46}[population]
            if len(group) != expected:
                raise ValueError('Incorrect fixed population denominator')
            summaries.append(dict(population=population, backbone=backbone, arm=arm, **summarize(group)))
    pairs, pair_ids = [], []
    for population, partitions in PARTITIONS.items():
        for backbone in BACKBONES:
            for before_arm, after_arm in itertools.product(('N0','E0'), ARMS[2:]):
                before = {r['id']:r for r in ledger if r['partition'] in partitions and r['backbone']==backbone and r['arm']==before_arm}
                after = {r['id']:r for r in ledger if r['partition'] in partitions and r['backbone']==backbone and r['arm']==after_arm}
                effect, groups = transition(before, after)
                key = dict(population=population, before_backbone=backbone, after_backbone=backbone, before_arm=before_arm, after_arm=after_arm)
                pairs.append(dict(**key, **effect)); pair_ids.append(dict(**key, groups=groups))
        for other, arm in itertools.product(BACKBONES[1:], ARMS[2:]):
            before = {r['id']:r for r in ledger if r['partition'] in partitions and r['backbone']=='yolo' and r['arm']==arm}
            after = {r['id']:r for r in ledger if r['partition'] in partitions and r['backbone']==other and r['arm']==arm}
            effect, groups = transition(before, after)
            key = dict(population=population, before_backbone='yolo', after_backbone=other, before_arm=arm, after_arm=arm)
            pairs.append(dict(**key, **effect)); pair_ids.append(dict(**key, groups=groups))
    candidate_summary = []
    for population, partitions in PARTITIONS.items():
        for backbone in BACKBONES:
            subset = [r for r in coverage if r['partition'] in partitions and r['backbone']==backbone]
            for arm in ARMS[2:]:
                final = {r['id']:r for r in ledger if r['partition'] in partitions and r['backbone']==backbone and r['arm']==arm}
                covered = sum(r['eligible_success_exists'] for r in subset)
                selected = sum(r['success'] for r in final.values())
                if selected > covered:
                    raise ValueError('Selected success without eligible successful bank pose')
                candidate_summary.append(dict(population=population, backbone=backbone, arm=arm, denominator=len(subset),
                    no_initial_pose=sum(not r['initial_pose_available'] for r in subset), eligible_success_exists=covered,
                    no_eligible_success=len(subset)-covered, selected_successes=selected, selection_losses=covered-selected))
    write_csv(output/'ROW_METRICS.csv', ledger)
    write_csv(output/'ALL_POPULATION_METRICS.csv', summaries)
    write_csv(output/'PAIRED_COUNTS.csv', pairs)
    write_csv(output/'CANDIDATE_ROWS.csv', coverage)
    write_csv(output/'CANDIDATE_DIAGNOSTICS.csv', candidate_summary)
    write(output/'PAIRED_IDS.json', dict(contrasts=pair_ids, new_bootstrap=0, intervals='Not computed for new contrasts; old YOLO/PF intervals do not apply'))
    seal = dict(schema='FROZEN_D1_THREE_BACKBONE_CURRENT_C2_SCORE_SEAL_V1', complete=True,
        predictions_bound_before_reference_decode=ps, current_reference_bindings=reference_bindings,
        feature_bindings=feature_bindings, exact_original_YOLO_D1_returned_pose_checks=exact_yolo_poses_checked,
        original_YOLO_parity_prediction_seal_sha256=sha(original_prediction_dir/'PREDICTION_SEAL.json'),
        source_sha256=sha(__file__), source_metric='pallet_accuracy_closeout_20261017_v1.baseline_audit.metric',
        source_metric_sha256=sha(Path(__file__).parents[1]/'pallet_accuracy_closeout_20261017_v1/baseline_audit.py'),
        row_metrics=len(ledger), summaries=len(summaries), paired_contrasts=len(pairs),
        source_populations=dict(PRIMARY192=192, TEST49=49, REMAINDER143=143, CAL46=46),
        current_reference_reconstructed=False, reference_modified=False, new_training=0, new_seeds=0, new_bootstrap=0,
        success='same returned pose T_cm<=5 AND physical C2 R_deg<=5', continuous='available only, untrimmed, sample variance ddof1, linear quantiles',
        all_failures_retained=True, artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        wall_seconds=time.monotonic()-start)
    write(output/'SCORE_SEAL.json', seal)
    return seal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('private','original-d1','original-score','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.private,args.original_d1,args.original_score,args.output)), flush=True)
