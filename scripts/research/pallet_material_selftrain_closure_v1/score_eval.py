"""Legacy Wood scoring only after all model coordinates and D9 poses are locked."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from . import common as C
from .infer_eval import ARMS, validate_metadata, verify_final_lock, image_order_hash
from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
from scripts.research.pallet_recording_disjoint_transfer_v1.evaluate import summary, transitions
from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D


def groups(rows):
    out = {'ALL': [r['id'] for r in rows]}
    severity_names = {'CLEAN': 'CLEAN', 'MODERATE_OCCLUSION': 'MODERATE', 'SEVERE_OCCLUSION': 'SEVERE'}
    for severity in sorted({r['severity'] for r in rows}):
        out[severity_names[severity]] = [r['id'] for r in rows if r['severity'] == severity]
    for key in sorted({r['recording'] for r in rows}):
        out[key] = [r['id'] for r in rows if r['recording'] == key]
    for key in sorted({r['id'].split(':')[0] for r in rows}):
        out['SESSION_' + key] = [r['id'] for r in rows if r['id'].split(':')[0] == key]
    assert all(out.values()), 'Do not manufacture empty severity metrics'
    return out


def score_frame(fid, prediction, truth, fixed=False):
    candidate = C.P.selected(prediction)
    matched = candidate is not None and V.E.C.H.E.O.iou(candidate['box_xyxy'], truth['box']) >= .5
    q = np.full((9, 2), np.nan) if candidate is None else candidate['keypoints_xy']
    perms = [list(range(9))] if fixed else truth['permutations']
    return dict(id=fid, **V.E.P.M.measure(q, truth['gt'], truth['valid'], perms,
                                        truth['hw'], matched, candidate is not None))


def pose_job(row):
    fid, pose, truth = row
    return fid, D.metric(fid, pose, truth)


def sixd(rows):
    value = D.aggregate(rows)
    value['pose_coverage'] = value['available'] / value['frames']
    return value


def paired(left, right, left_pose, right_pose, ids):
    a, b = [left[i] for i in ids], [right[i] for i in ids]
    assert [r['id'] for r in a] == [r['id'] for r in b]
    eligible = [(x, y) for x, y in zip(a, b) if x['evaluable'] and y['evaluable']]
    counts = Counter()
    for x, y in eligible:
        delta = np.mean(y['errors']) - np.mean(x['errors'])
        counts['improved' if delta < -1e-8 else 'worsened' if delta > 1e-8 else 'unchanged'] += 1
    sa, sb = summary(a), summary(b)
    assert sa['corners'] == sb['corners']
    return dict(frames=len(ids), corners=sa['corners'], frame_mean_error=dict(counts),
                PCK10_delta_pp=100 * (sb['PCK']['10'] - sa['PCK']['10']),
                correct10_delta=sb['correct']['10'] - sa['correct']['10'],
                ADDsym_AUC_delta=sixd([right_pose[i] for i in ids])['ADDsym_AUC'] - sixd([left_pose[i] for i in ids])['ADDsym_AUC'],
                transitions=transitions([x for x, _ in eligible], [y for _, y in eligible]),
                canonical_GT_identity_aligned=True)


def main():
    prediction_lock = verify_final_lock()
    result_path = C.DOC / 'WOOD_RESULTS.json'
    if result_path.exists():
        result = C.read(result_path)
        assert result['prediction_lock'] == C.bind(C.DOC / 'WOOD_PREDICTIONS_LOCK.json')
        for b in result['artifact_sources']:
            C.verify(b)
        print('WOOD_RESULTS_ALREADY_FROZEN', flush=True)
        return
    rows = C.read(C.RAW / 'EVAL_METADATA.json')
    ids = validate_metadata(rows)
    assert prediction_lock['image_ids'] == ids
    assert prediction_lock['image_order_sha256'] == image_order_hash(rows)
    g = groups(rows)
    scoring_start = C.DOC / 'WOOD_SCORING_START.json'
    if not scoring_start.exists():
        C.save(scoring_start, dict(utc=C.now(), prediction_lock=C.bind(C.DOC / 'WOOD_PREDICTIONS_LOCK.json')), True)
    assert C.read(scoring_start)['prediction_lock'] == C.bind(C.DOC / 'WOOD_PREDICTIONS_LOCK.json')
    assert prediction_lock['created_at'] <= C.read(scoring_start)['utc']
    predictions = C.read(C.RAW / 'PREDICTIONS.json')
    poses = C.read(C.RAW / 'POSE_PREDICTIONS.json')
    # First reference-coordinate access by this scoring workflow, strictly after verification above.
    truth = C.read(C.P.TRUTH)
    _, pose_truth = D.Pose.metadata('REAL_DEV')
    assert set(ids) <= set(truth) and set(ids) <= set(pose_truth)
    frames, fixed, metrics = {}, {}, {}
    for arm in ARMS:
        assert list(predictions[arm]) == ids and list(poses[arm]) == ids
        frames[arm], fixed[arm] = {}, {}
        for r in rows:
            fid = r['id']
            assert list(truth[fid]['hw']) == r['hw'], 'Native coordinate resolution mismatch'
            frames[arm][fid] = score_frame(fid, predictions[arm][fid], truth[fid])
            fixed[arm][fid] = score_frame(fid, predictions[arm][fid], truth[fid], fixed=True)
        with ProcessPoolExecutor(max_workers=4) as pool:
            metrics[arm] = dict(pool.map(pose_job, [(i, poses[arm][i], pose_truth[i]) for i in ids], chunksize=8))
        print('WOOD_SCORED', arm, len(ids), flush=True)
    results = {group: {arm: dict(twoD=summary([frames[arm][i] for i in ii]),
                                          fixed_ID=summary([fixed[arm][i] for i in ii]),
                                          sixD=sixd([metrics[arm][i] for i in ii]))
                       for arm in ARMS} for group, ii in g.items()}
    for arm in ARMS:
        all_count = results['ALL'][arm]['twoD']['corners']
        assert all_count == sum(results[x][arm]['twoD']['corners'] for x in g if x in ('CLEAN', 'MODERATE', 'SEVERE'))
        assert all_count == results['ALL']['R0']['twoD']['corners']
    contrast_arms = [('WOOD_RAW_LR5', 'WOOD_REF_LR5'), ('R0', 'WOOD_REF_LR5'),
                     ('SYN_LR5', 'WOOD_REF_LR5'), ('R0', 'WOOD_RAW_LR5')]
    comparisons = {group: {f'{b}-minus-{a}': paired(frames[a], frames[b], metrics[a], metrics[b], ii)
                           for a, b in contrast_arms} for group, ii in g.items()}
    recordings = sorted({r['recording'] for r in rows})
    loro = {rec: {f'{b}-minus-{a}': paired(frames[a], frames[b], metrics[a], metrics[b],
                                       [r['id'] for r in rows if r['recording'] != rec])
                  for a, b in contrast_arms} for rec in recordings if any(r['recording'] != rec for r in rows)}
    for name, value in [('FRAME_METRICS.json', frames), ('FIXED_ID_METRICS.json', fixed), ('POSE_METRICS.json', metrics)]:
        C.save(C.RAW / name, value, True)
    paired_path = C.DOC / 'WOOD_PAIRED_ANALYSIS.json'
    C.save(paired_path, dict(groups=comparisons, leave_one_recording_out=loro, recording_count=len(recordings),
                            primary='WOOD_REF_LR5-minus-WOOD_RAW_LR5', independent_corner_test=False,
                            uncertainty='Only two reused recording groups; recording summaries and LORO are descriptive, not independent replication.'), True)
    source_files = [C.RAW / 'FRAME_METRICS.json', C.RAW / 'FIXED_ID_METRICS.json',
                    C.RAW / 'POSE_METRICS.json', paired_path, C.P.TRUTH,
                    D.Pose.E.C.POSE / 'GEOMETRY_RESOLVED_POSE_GT.json',
                    D.Pose.E.C.POSE / 'AXIS_REVIEW_MANIFEST.json', Path(__file__), Path(D.__file__)]
    C.save(result_path, dict(
        groups=results, frames=len(ids), recordings=len(recordings), group_ids=g,
        prediction_lock=C.bind(C.DOC / 'WOOD_PREDICTIONS_LOCK.json'),
        artifact_sources=[C.bind(p) for p in source_files],
        main_arms=list(C.ARMS), legacy_descriptive_only_arms=['TEACHER'],
        Q1_WOOD='UNRESOLVED', teacher_legacy_scores_are_not_verified_pseudo_quality=True,
        reference='Existing legacy Wood annotations with unknown point provenance; geometry-derived 6D, not independent physical pose.',
        twoD_contract='Unchanged whole-object allowed symmetry; selected detection IoU>=0.5 match; corners0..7; missing/mismatch native diagonal penalty; full denominator retained.',
        pose_contract='Same original deployable D9; corner0..7 solve and9point selector residual; no oracle or model-specific selector.',
        severity_available={s: s in g for s in ('CLEAN', 'MODERATE', 'SEVERE')},
        raw_and_pose_frozen_before_reference_scoring=True,
        independent_confirmation=False, externally_known_material_routing=True), True)
    for arm in ARMS:
        value = results['ALL'][arm]
        print(arm, 'PCK10', value['twoD']['PCK']['10'], 'AUC', value['sixD']['ADDsym_AUC'], flush=True)


if __name__ == '__main__':
    main()
