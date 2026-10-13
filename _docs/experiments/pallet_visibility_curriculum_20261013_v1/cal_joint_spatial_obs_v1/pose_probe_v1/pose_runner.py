"""CAL-only sealed OOF joint-observation pose probe; no training labels in infer."""
import argparse
import builtins
import collections
import contextlib
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from code_snapshot import common as C
from code_snapshot.curriculum import solve_curriculum
from code_snapshot.run import no_annotations
from code_snapshot.scoring import pose_metric

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(C.finite(value), indent=2, allow_nan=False) + '\n')


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


@contextlib.contextmanager
def inference_guard():
    """Also forbid local supervised audit caches during independent inference."""
    with no_annotations():
        originals = builtins.open, io.open, Path.open
        forbidden = ('OOF_OBSERVATIONS.json', 'LABEL_AUDIT.json', 'SCORED_ROWS', 'RESULTS.json')
        def wrap(function):
            def guarded(path, *args, **kwargs):
                if any(name in str(path) for name in forbidden):
                    raise AssertionError('Supervised scoring artifact read in inference: ' + str(path))
                return function(path, *args, **kwargs)
            return guarded
        builtins.open, io.open, Path.open = map(wrap, originals)
        try:
            yield
        finally:
            builtins.open, io.open, Path.open = originals


def canary():
    checked = []
    for path in (HERE.parent/'OOF_OBSERVATIONS.json', HERE.parent/'LABEL_AUDIT.json',
                 HERE.parent.parent/'ORACLE_CAL_INPUTS.json'):
        try:
            read(path)
        except AssertionError:
            checked.append(path.name)
        else:
            raise AssertionError('Read guard failed for ' + str(path))
    return checked


def infer():
    p = read(HERE/'PROTOCOL.json')
    assert not (HERE/'PREDICTIONS.jsonl.gz').exists()
    for key in ('inputs', 'baseline_predictions', 'prediction_observations'):
        assert sha(p[key]) == p[key + '_sha256']
    assert sha(HERE.parent/'OOF_OBSERVATION_SEAL.json') == p['observation_seal_sha256']
    for name, digest in p['source_sha256'].items():
        assert sha(HERE/'code_snapshot'/name) == digest
    write(HERE/'INFERENCE_CODE_BINDING.json', dict(runner_sha256=sha(__file__),
        protocol_sha256=sha(HERE/'PROTOCOL.json'), before_pose_inference=True))
    result = []
    with inference_guard():
        canaries = canary()
        frames = read(p['inputs'])['frames']
        assert len(frames) == 76 and all(f['partition'] == 'CAL' for f in frames)
        base = {(r['id'], r['method']): r for r in rows(p['baseline_predictions'])}
        records = read(p['prediction_observations'])
        allowed = {'id', 'corner', 'session', 'family', 'OOF_usable_probability',
                   'cutoff', 'RGB_delta_px', 'RGB_regressor_available'}
        assert all(set(record) == allowed for record in records)
        heads = {(r['id'], r['corner'], r['family']): r for r in records}
        assert len(heads) == len(records)
        for frame in frames:
            ident = frame['id']
            initial = frame['initial_poses']['N3_SUBPIX']
            old = base[ident, 'SEGMENTED_RGB_CURRICULUM']
            q = np.asarray(old['original_points'], float)
            raw = np.asarray(frame['points']['N3_SUBPIX'], float)
            assert np.array_equal(q, raw[:8])
            H = np.asarray(old['self_hidden'], bool)
            xyz, K = C.cuboid(*initial['cf_extents']), np.asarray(frame['K'])
            R, t = np.asarray(initial['R_cf']), np.asarray(initial['centroid'])
            Q = R.T @ np.asarray(initial['R_physical'])
            assert np.array_equal(H, C.self_hidden(xyz, R, t))
            h, w = frame['raw_hw']
            for arm in p['arms']:
                app = deepcopy(old['appearance'])
                threshold = arm['threshold']
                admitted, scores, corrections, rejections = [], {}, [], {}
                fitted = q.copy()
                for k in range(8):
                    record = heads.get((ident, k, arm['family']))
                    if record is None:
                        assert H[k] or app['states'][k] == 'OUT_OF_FRAME' or not np.isfinite(q[k]).all() or (q[k] == -1).all()
                        rejections[str(k)] = 'fixed_H_OOB_or_missing'
                        continue
                    assert not H[k] and app['states'][k] not in ('SELF_OCCLUDED', 'OUT_OF_FRAME')
                    assert record['session'] == frame['session'] and record['cutoff'] == threshold
                    probability = float(record['OOF_usable_probability'])
                    assert 0 <= probability <= 1
                    app['visibility_score'][k] = probability
                    scores[str(k)] = probability
                    if probability < threshold:
                        app['states'][k] = 'UNKNOWN'
                        rejections[str(k)] = 'below_joint_usable_cutoff'
                        continue
                    app['states'][k] = 'VISIBLE_BOUNDARY' if app['boundary_score'][k] >= .35 else 'VISIBLE_INTERIOR'
                    app['local_accuracy_score'][k] = max(app['local_accuracy_score'][k], .12)
                    admitted.append(k)
                    if not arm['RGB_coordinate_correction']:
                        continue
                    if not record['RGB_regressor_available']:
                        rejections[str(k)] = 'RGB_regressor_unavailable'
                        continue
                    delta = np.asarray(record['RGB_delta_px'], float)
                    assert delta.shape == (2,) and np.isfinite(delta).all() and np.linalg.norm(delta) <= 16 + 1e-8
                    candidate = q[k] + delta
                    if not (0 <= candidate[0] < w and 0 <= candidate[1] < h):
                        rejections[str(k)] = 'RGB_corrected_point_outside_image'
                        continue
                    fitted[k] = candidate
                    corrections.append(dict(corner=k, original=q[k].tolist(), corrected=candidate.tolist(),
                        RGB_delta_px=delta.tolist(), probability=probability, source='heldout_session_RGB_spatial_ridge'))
                config = dict(p['curriculum_base'], visible_threshold=threshold, easy_visibility=max(.45, threshold))
                solution = solve_curriculum(fitted, xyz, K, R, t, H, app, config)
                assert not any(H[k] for stage in solution['stages'] for k in stage['used_ids'])
                assert np.array_equal(q, raw[:8]) and np.array_equal(H, np.asarray(old['self_hidden']))
                if not solution['accepted']:
                    assert np.array_equal(np.asarray(solution['R']), R) and np.array_equal(np.asarray(solution['t']), t)
                predicted = dict(R_cf=solution['R'], R_physical=(np.asarray(solution['R']) @ Q).tolist(),
                    centroid=solution['t'], cf_extents=initial['cf_extents'])
                result.append(dict(id=ident, session=frame['session'], arm=arm['name'], family=arm['family'],
                    threshold=threshold, initial_self_hidden=H.tolist(), original_points=q.tolist(),
                    original_center=raw[8].tolist() if len(raw) > 8 else None,
                    fitted_measurements=fitted.tolist(), head_probabilities=scores, head_admitted_ids=admitted,
                    coordinate_corrections=corrections, coordinate_rejections=rejections,
                    solution=solution, actual_pose=predicted,
                    provenance=dict(H_unchanged=True, native_phase_unchanged=True, center_unchanged=True,
                        initial_pose_unchanged=True, boundary_accuracy_source='frozen_SEGMENTED_RGB_CURRICULUM',
                        joint_head='LOGO_CAL_selected_usable_le8px', RGB_coordinate_source='LOGO_spatial_frozen_ImageNet_ridge',
                        RGB_regressor_used_geometric_residual=False, projected_points_used_for_fit=False,
                        GT_input=False, TEST_accessed=False, FINAL_accessed=False)))
    assert len(result) == 760
    with gzip.open(HERE/'PREDICTIONS.jsonl.gz', 'wt') as stream:
        for row in result:
            stream.write(json.dumps(C.finite(row), separators=(',', ':'), allow_nan=False) + '\n')
    write(HERE/'PREDICTION_SEAL.json', dict(complete=True, before_pose_scoring=True, GT_read_canary='PASS',
        supervised_artifact_canaries=canaries, rows=len(result), sha256=sha(HERE/'PREDICTIONS.jsonl.gz'),
        protocol_sha256=sha(HERE/'PROTOCOL.json'), runner_sha256=sha(__file__), TEST_accessed=False, FINAL_accessed=False))
    print('JOINT_POSE_SEALED', len(result), flush=True)


def distribution(values):
    x = np.asarray(values, float)
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
        median=float(np.median(x)) if len(x) else None, P90=float(np.quantile(x, .9)) if len(x) else None)


def coordinate_summary(rows_for_arm, oracle):
    direct, eligible_direct, changed_direct, changed_other, sessions = [], [], [], [], collections.defaultdict(list)
    for row in rows_for_arm:
        f = oracle[row['id']]
        meta = f['scoring_metadata']
        gt = np.asarray(f['target']['gt'], float)[:8][meta['native_to_reference_corner_ids']]
        q, corrected = np.asarray(row['original_points']), np.asarray(row['fitted_measurements'])
        changed = {x['corner'] for x in row['coordinate_corrections']}
        for k, state in enumerate(meta['human_states_native']):
            if not np.isfinite(q[k]).all() or (q[k] == -1).all() or not np.isfinite(gt[k]).all() or (gt[k] == -1).all():
                continue
            item = dict(id=row['id'], session=row['session'], corner=k, human_state=state,
                before=float(np.linalg.norm(q[k] - gt[k])), after=float(np.linalg.norm(corrected[k] - gt[k])),
                changed=k in changed, admitted=k in row['head_admitted_ids'])
            if state == 'DIRECT_VISIBLE':
                direct.append(item)
                sessions[row['session']].append(item)
                if str(k) in row['head_probabilities']:
                    eligible_direct.append(item)
                if k in changed:
                    changed_direct.append(item)
            elif k in changed:
                changed_other.append(item)
    def report(items):
        before, after = [x['before'] for x in items], [x['after'] for x in items]
        return dict(before=distribution(before), after=distribution(after),
            mean_delta=float(np.mean(np.asarray(after)-before)) if items else None,
            improved=sum(x['after'] < x['before'] for x in items), worsened=sum(x['after'] > x['before'] for x in items),
            before_within8=sum(x['before'] <= 8 for x in items), after_within8=sum(x['after'] <= 8 for x in items))
    return dict(all_DIRECT_visible=report(direct), predicted_eligible_DIRECT=report(eligible_direct),
        changed_DIRECT=report(changed_direct), changed_DIRECT_initial_le16=report([x for x in changed_direct if x['before'] <= 16]),
        changed_DIRECT_initial_gt16=report([x for x in changed_direct if x['before'] > 16]),
        admitted_DIRECT=report([x for x in direct if x['admitted']]),
        changed_nonDIRECT_count=len(changed_other), changed_nonDIRECT_states=dict(collections.Counter(x['human_state'] for x in changed_other)),
        per_session={s:report(items) for s,items in sessions.items()},
        changed_records=changed_direct + changed_other)


def score():
    p, seal = read(HERE/'PROTOCOL.json'), read(HERE/'PREDICTION_SEAL.json')
    assert seal['before_pose_scoring'] and seal['GT_read_canary'] == 'PASS'
    assert seal['sha256'] == sha(HERE/'PREDICTIONS.jsonl.gz') and seal['protocol_sha256'] == sha(HERE/'PROTOCOL.json')
    frames = read(HERE.parent.parent/'ORACLE_CAL_INPUTS.json')['frames']
    assert len(frames) == 76 and all(f['partition'] == 'CAL' for f in frames)
    oracle = {f['id']:f for f in frames}
    base = {(r['id'],r['method']):r for r in rows(p['baseline_predictions'])}
    scored = rows(HERE/'PREDICTIONS.jsonl.gz')
    groups = collections.defaultdict(list)
    for row in scored:
        truth = oracle[row['id']]['target']['truth']
        row['metric'] = pose_metric(row['actual_pose'],truth)
        row['baseline_metric'] = pose_metric(base[row['id'],'SEGMENTED_RGB_CURRICULUM']['actual_pose'],truth)
        row['frozen_initial_metric'] = pose_metric(base[row['id'],'INITIAL']['actual_pose'],truth)
        groups[row['arm']].append(row)
    rng = np.random.default_rng(20261013)
    sessions = sorted({r['session'] for r in scored})
    draws = rng.integers(0,len(sessions),(10000,len(sessions)))
    summary = {}
    for arm,value in groups.items():
        report = dict(frames=len(value), accepted=sum(r['solution']['accepted'] for r in value),
            corrected_frames=sum(bool(r['coordinate_corrections']) for r in value),
            corrected_corners=sum(len(r['coordinate_corrections']) for r in value),
            metrics={key:distribution([r['metric'][key] for r in value]) for key in ('translation_cm','rotation_deg','ADDsym_cm')},
            success_count=sum(r['metric']['success_5cm_5deg'] for r in value),
            success_5cm_5deg=float(np.mean([r['metric']['success_5cm_5deg'] for r in value])),
            reasons=dict(collections.Counter(r['solution']['reason'] for r in value)),
            paired_vs_SEGMENTED={}, coordinate=coordinate_summary(value,oracle))
        for key in ('translation_cm','rotation_deg','ADDsym_cm','success_5cm_5deg'):
            delta = np.asarray([float(r['metric'][key])-float(r['baseline_metric'][key]) for r in value])
            sums = np.asarray([sum(d for d,r in zip(delta,value) if r['session']==s) for s in sessions])
            counts = np.asarray([sum(r['session']==s for r in value) for s in sessions])
            boot = sums[draws].sum(1)/counts[draws].sum(1)
            report['paired_vs_SEGMENTED'][key] = dict(mean_difference=float(delta.mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),
                improved=int((delta>0).sum()) if key.startswith('success') else int((delta<0).sum()),
                worsened=int((delta<0).sum()) if key.startswith('success') else int((delta>0).sum()))
        report['per_session_pose'] = {s:dict(frames=sum(r['session']==s for r in value),
            translation_cm=distribution([r['metric']['translation_cm'] for r in value if r['session']==s]),
            rotation_deg=distribution([r['metric']['rotation_deg'] for r in value if r['session']==s]),
            success_count=sum(r['metric']['success_5cm_5deg'] for r in value if r['session']==s)) for s in sessions}
        summary[arm] = report
        print(arm,'accepted',report['accepted'],'corrected',report['corrected_corners'],
            'Tmean',report['metrics']['translation_cm']['mean'],'Rmean',report['metrics']['rotation_deg']['mean'],
            'success',report['success_count'],'DIRECTchanged',report['coordinate']['changed_DIRECT'],flush=True)
    with gzip.open(HERE/'SCORED_ROWS.jsonl.gz','wt') as stream:
        for row in scored:
            stream.write(json.dumps(C.finite(row),separators=(',',':'))+'\n')
    write(HERE/'RESULTS.json',dict(schema=p['schema'],CAL_only=True,TEST_accessed=False,FINAL_accessed=False,
        frames=76,summary=summary,baseline_SEGMENTED=dict(translation_cm=116.29147334738165,rotation_deg=19.290009317834865,success_count=37),
        predictions_sha256=seal['sha256'],limitations=p['limits']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('infer','score'))
    args = parser.parse_args()
    infer() if args.command == 'infer' else score()
