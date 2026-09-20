"""CPU-only viewer export of all 72 frozen non-green development predictions.

No inference, training, checkpoint choice, label edit, or prediction edit is
performed. Annotation coordinates are used only to check the saved evaluation
and to draw a clearly marked legacy/proxy reference layer.
"""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/pallet_posefix_other_audit_v1'
REPLAY = ROOT / 'data/pallet/results/pallet_posefix_replay_v1'
PREVIOUS = ROOT / 'data/pallet/results/pallet_posefix_large_error_v1'
SPLIT = ROOT / '_docs/experiments/pallet_large_error_refiner_v1/SPLIT.json'
CONTRACT = ROOT / '_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json'
MODEL_KEYS = ('R0', 'A_N2', 'PRIOR', 'REAL_ONLY', 'REPLAY')


def read(path):
    return json.loads(path.read_text())


def binding(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def verified_bytes(item):
    raw = (ROOT / item['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == item['sha256'], item['path']
    if 'bytes' in item:
        assert len(raw) == item['bytes'], item['path']
    return raw


def finite_points(points):
    return [[float(x) if np.isfinite(x) else None for x in p] for p in points]


def selected(prediction):
    index = prediction['selected_index']
    assert index is not None, 'Viewer expects all DEV72 saved detections present'
    return prediction['candidates'][index]


def recompute(points, gt, valid, permutations, metric, hw):
    """Independent complete-object symmetry evaluation; never move the output."""
    assert metric['matched'] and metric['detected'] and metric['evaluable']
    points, gt = np.asarray(points, float), np.asarray(gt, float)
    valid = np.asarray(valid, bool)
    assert points.shape == gt.shape == (9, 2) and valid.shape == (8,)
    diag = float(np.hypot(*hw))
    errors_by_branch = []
    for permutation in permutations:
        canonical = [None] * 8
        for native, target in enumerate(permutation[:8]):
            if not valid[target]:
                continue
            assert np.isfinite(gt[target]).all() and not (gt[target] == -1).all()
            p = points[native]
            error = float(np.linalg.norm(p-gt[target])) if np.isfinite(p).all() and not (p == -1).all() else diag
            canonical[target] = error
        errors_by_branch.append(canonical)
    means = [float(np.mean([x for x in e if x is not None])) for e in errors_by_branch]
    branch = int(np.argmin(means))
    assert branch == metric['branch'], (branch, metric['branch'])
    computed = errors_by_branch[branch]
    for actual, saved in zip(computed, metric['canonical_errors']):
        if actual is None:
            assert saved is None
        else:
            assert np.isclose(actual, saved, atol=1e-7, rtol=0), (actual, saved)
    assert np.isclose(means[branch], metric['frame_mean_px'], atol=1e-7, rtol=0)
    return computed


def main():
    paths = [REPLAY/'PREDICTIONS.json', REPLAY/'PER_FRAME_METRICS.json', SPLIT, CONTRACT]
    split, contract = read(SPLIT), read(CONTRACT)
    records = split['evaluation']
    groups = {x['object_type']: x['permutations'] for x in contract['objects']}
    replay_pred = read(paths[0])['DEV72']
    replay_scores = read(paths[1])['DEV72']
    preds, scores = {}, {}
    for label, arm in [('R0','R0'), ('A_N2','A_N2'), ('REPLAY','POSEFIX_RAW')]:
        preds[label] = {x['id']: x['predictions'][arm] for x in replay_pred}
        scores[label] = {x['id']: x for x in replay_scores[arm]}
    for label, prefix in [('PRIOR','EXISTING'), ('REAL_ONLY','FINETUNED')]:
        pred_path, score_path = PREVIOUS/f'{prefix}_PREDICTIONS.json', PREVIOUS/f'{prefix}_PER_FRAME_METRICS.json'
        paths.extend([pred_path, score_path])
        preds[label] = {x['id']: x['predictions']['POSEFIX_RAW'] for x in read(pred_path)['DEV72']}
        scores[label] = {x['id']: x for x in read(score_path)['DEV72']['POSEFIX_RAW']}
    assert len(records) == len(replay_pred) == 72
    assert [r['id'] for r in records] == [r['id'] for r in replay_pred]
    ids = {r['id'] for r in records}
    assert all(set(v) == ids for v in [*preds.values(), *scores.values()])
    rows, errors = [], {key: [] for key in MODEL_KEYS}
    gt_sources, object_types = Counter(), Counter()
    lost = gained = branch_changes = checks = 0
    for record, saved in zip(records, replay_pred):
        ident = record['id']
        assert saved['image'] == record['image'] and saved['annotation'] == record['annotation']
        image_raw = verified_bytes(record['image'])
        annotation = json.loads(verified_bytes(record['annotation']))
        assert image_raw[:8] == b'\x89PNG\r\n\x1a\n'
        keypoints = annotation['objects'][0]['keypoint_annotations']
        assert len(keypoints) == 9
        gt = np.asarray([k['xy'] if k.get('xy') is not None else [np.nan,np.nan] for k in keypoints], float)
        sources = [k.get('source', 'unknown') for k in keypoints]
        gt_sources.update(sources)
        object_types[record['object_type']] += 1
        valid = scores['R0'][ident]['canonical_valid']
        h, w = saved['raw_hw']
        models = {}
        for label in MODEL_KEYS:
            prediction = preds[label][ident]
            metric = scores[label][ident]
            assert metric['canonical_valid'] == valid
            candidate = selected(prediction)
            r0 = selected(preds['R0'][ident])
            assert prediction['selected_index'] == preds['R0'][ident]['selected_index']
            for field in ('box_xyxy', 'score', 'keypoints_conf'):
                assert candidate[field] == r0[field], (ident, label, field)
            assert candidate['keypoints_xy'][8] == r0['keypoints_xy'][8]
            computed = recompute(candidate['keypoints_xy'], gt, valid,
                groups[record['object_type']], metric, saved['raw_hw'])
            checks += 1
            errors[label].extend(x for x in computed if x is not None)
            models[label] = dict(points=finite_points(candidate['keypoints_xy']),
                branch=metric['branch'], permutation=groups[record['object_type']][metric['branch']],
                canonical_errors=computed, frame_mean_px=metric['frame_mean_px'],
                matched=metric['matched'], detected=metric['detected'])
        a, b = models['A_N2']['canonical_errors'], models['REPLAY']['canonical_errors']
        row_lost = [k for k in range(8) if valid[k] and a[k] <= 10 < b[k]]
        row_gained = [k for k in range(8) if valid[k] and b[k] <= 10 < a[k]]
        lost += len(row_lost); gained += len(row_gained)
        branch_changes += models['A_N2']['branch'] != models['REPLAY']['branch']
        focus = max((k for k in range(8) if valid[k]), key=lambda k: abs(a[k]-b[k]))
        pool = []
        for xy in [*gt[:8][np.asarray(valid,bool)], *np.asarray(models['R0']['points'], float)[:8],
                   *np.asarray(models['REPLAY']['points'], float)[:8]]:
            if np.isfinite(xy).all() and 0 <= xy[0] < w and 0 <= xy[1] < h:
                pool.append(xy)
        pool = np.asarray(pool)
        assert len(pool)
        xmin, ymin = np.maximum(pool.min(0)-25, [0,0])
        xmax, ymax = np.minimum(pool.max(0)+25, [w,h])
        rows.append(dict(id=ident, image=record['image'], annotation=record['annotation'],
            src='data:image/png;base64,' + base64.b64encode(image_raw).decode('ascii'),
            raw_hw=saved['raw_hw'], object_type=record['object_type'], session=record['session'],
            gt_xy=finite_points(gt), gt_sources=sources, canonical_valid=valid, models=models,
            focus_default=focus, crop_limits_xy=[float(xmin),float(xmax),float(ymin),float(ymax)],
            lost=row_lost, gained=row_gained,
            N2_to_replay_mean_improvement_px=models['A_N2']['frame_mean_px']-models['REPLAY']['frame_mean_px']))
    summaries = {}
    for key, values in errors.items():
        e = np.asarray(values)
        summaries[key] = dict(frames=72, matched_frames=72, detected_frames=72,
            corners=len(values), correct10=int((e<=10).sum()), PCK10=float((e<=10).mean()),
            mean_px=float(e.mean()), median_px=float(np.median(e)), P90_px=float(np.quantile(e,.9)))
    payload = dict(records=rows, edges=contract['edges'], summary=summaries,
        comparison=dict(lost=lost, gained=gained, branch_changes=branch_changes,
            improved_frames=sum(r['N2_to_replay_mean_improvement_px']>1e-7 for r in rows),
            worsened_frames=sum(r['N2_to_replay_mean_improvement_px']<-1e-7 for r in rows)),
        bindings=[binding(p) for p in paths]+[binding(Path(__file__))],
        object_types=dict(object_types), keypoint_source_counts=dict(gt_sources),
        warning='DEV72 is a reused development set. All 648 keypoint sources are unknown; these are legacy/proxy references, not independently verified manual GT. No wood-pallet evaluation image is included.',
        independent_error_checks=checks, independent_branch_checks=checks,
        source_image_and_annotation_SHA_checks=144,
        predictions_unmodified=True, GT_used_only_for_diagnostics=True,
        GT_used_to_generate_model_output=False, no_new_inference=True, no_training=True,
        sample_selection='All 72 DEV72 images, none omitted; optional UI ordering uses evaluation error only for inspection.')
    assert dict(gt_sources) == {'unknown':648}
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT/'DATA.json'
    output.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',',':'))+'\n')
    print(json.dumps(dict(path=str(output), rows=len(rows), byte_size=output.stat().st_size,
        summary=summaries, comparison=payload['comparison'], independently_recomputed=checks,
        object_types=dict(object_types), sources=dict(gt_sources)), ensure_ascii=False, indent=2))
    return payload


if __name__ == '__main__':
    main()
