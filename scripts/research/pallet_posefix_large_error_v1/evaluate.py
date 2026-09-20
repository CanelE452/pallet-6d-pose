"""Paired RGB-only PoseFix screen on immutable DEV72 / GREEN150 inputs.

No GT, camera, dimensions, depth, CAD or evaluation-driven checkpoint choice is
passed to the model.  Stored R0 detections are the *same* inputs for every arm.
The output cap is a deterministic view of one uncapped inference, not a rerun.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_posefix_large_error_v1 import core as C
from scripts.research.pallet_large_error_refiner_v1 import run as OLD
from dev_evaluate import population_metadata, iou
from eval_math import measure, summary
from inference import preservation

ARMS = ('R0', 'A_N2', 'POSEFIX_RAW', 'POSEFIX_CAP8')
DATASETS = ('DEV72', 'GREEN150_MANUAL', 'GREEN150_ALL_KNOWN_PROXY')
EXAMPLE = 'eval_pallet07:1778652166837872128'
OLD_PREDICTIONS = OLD.RAW / 'PREDICTIONS.json'
OLD_METRICS = OLD.RAW / 'PER_FRAME_METRICS.json'


def top(prediction):
    index = prediction['selected_index']
    return None if index is None else prediction['candidates'][index]


def assert_same(a, b, path='value', atol=1e-8):
    """Nested numerical parity, including metadata and invalid-point masks."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for key in a:
            assert_same(a[key], b[key], f'{path}.{key}', atol)
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for index, (left, right) in enumerate(zip(a, b)):
            assert_same(left, right, f'{path}[{index}]', atol)
    elif isinstance(a, float):
        assert np.isclose(a, b, atol=atol, rtol=0, equal_nan=False), (path, a, b)
    else:
        assert a == b, (path, a, b)


def recovery_damage(base, new, matched_only=False):
    """Compare errors of identical canonical GT identities, not sorted errors."""
    assert [row['id'] for row in base] == [row['id'] for row in new]
    initial, final = [], []
    frames = 0
    for before, after in zip(base, new):
        assert before['matched'] == after['matched']
        assert before['detected'] == after['detected']
        if not before['evaluable']:
            assert not after['evaluable']
            continue
        assert after['evaluable']
        assert before['canonical_valid'] == after['canonical_valid']
        if matched_only and not before['matched']:
            continue
        frames += 1
        for index, valid in enumerate(before['canonical_valid']):
            if valid:
                initial.append(before['canonical_errors'][index])
                final.append(after['canonical_errors'][index])
    a, b = np.asarray(initial, dtype=float), np.asarray(final, dtype=float)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    hard, good = a > 20, a < 5
    nhard, ngood = int(hard.sum()), int(good.sum())
    recovered = int((hard & (b <= 10)).sum())
    damaged = int((good & (b > 10)).sum())
    return dict(frames=frames, corners=len(initial), hard=nhard,
                recovered=recovered,
                recovery_rate=recovered / nhard if nhard else None,
                good=ngood, damaged=damaged,
                damage_rate=damaged / ngood if ngood else None,
                matched_only=matched_only,
                canonical_GT_identity_aligned=True)


def evaluation_inputs():
    paths = [OLD_PREDICTIONS, OLD_METRICS, OLD.DOC / 'SPLIT.json',
             OLD.GREEN, C.E.SYM_DOC / 'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json']
    bindings = [C.E.bound(path) for path in paths]
    for binding in bindings:
        C.F.verify(binding)
    saved = C.E.read(OLD_PREDICTIONS)
    previous = C.E.read(OLD_METRICS)
    split = C.E.read(OLD.DOC / 'SPLIT.json')
    green = C.E.read(OLD.GREEN)
    records = {'DEV72': split['evaluation'], 'GREEN150': green['records']}
    for dataset, count in [('DEV72', 72), ('GREEN150', 150)]:
        assert len(saved[dataset]) == len(records[dataset]) == count
        assert [row['id'] for row in saved[dataset]] == [row['id'] for row in records[dataset]]
        assert len({row['id'] for row in records[dataset]}) == count
    groups = {row['object_type']: row for row in C.E.read(paths[-1])['objects']}
    pe, population = population_metadata()
    targets = {item.frame_id: (item, meta) for item, meta in population}
    return saved, previous, records, groups, pe, targets, bindings


def read_targets(dataset, record, hw, pe, targets):
    """Evaluation-only target access; call after all inference for this frame."""
    C.F.verify(record['annotation'])
    if dataset == 'DEV72':
        item, _ = targets[record['id']]
        assert str(item.label) == record['annotation']['path']
        target = pe.E._legacy_forbidden_target(item)
        return (np.asarray(target.keypoints_xy), np.asarray(target.box_xyxy),
                [('DEV72', np.asarray(target.keypoint_supervision_mask))],
                record['object_type'])
    annotation = C.E.read(C.ROOT / record['annotation']['path'])
    gt, valid = OLD.annotation_arrays(annotation)
    _, manual = OLD.annotation_arrays(annotation, True)
    h, w = hw
    inside = valid & (gt[:, 0] >= 0) & (gt[:, 0] < w) & (gt[:, 1] >= 0) & (gt[:, 1] < h)
    assert inside.any(), record['id']
    box = np.r_[gt[inside].min(0), gt[inside].max(0)]
    return (gt, box, [('GREEN150_MANUAL', manual),
                     ('GREEN150_ALL_KNOWN_PROXY', valid)],
            'plastic_standard_110x110x15')


def score_prediction(prediction, gt, box, valid, perms, hw, record):
    candidate = top(prediction)
    matched = candidate is not None and iou(candidate['box_xyxy'], box) >= .5
    points = np.full((9, 2), np.nan) if candidate is None else candidate['keypoints_xy']
    score = measure(points, gt, valid, perms, hw, matched, candidate is not None)
    return dict(id=record['id'], session=record['session'], **score)


def baseline_audit():
    """CPU-only proof that the target population/metric reproduces saved scores."""
    saved, previous, records, groups, pe, targets, _ = evaluation_inputs()
    checked = 0
    for dataset in ('DEV72', 'GREEN150'):
        for row, record in zip(saved[dataset], records[dataset]):
            gt, box, modes, obj = read_targets(dataset, record, row['raw_hw'], pe, targets)
            for mode, valid in modes:
                for arm in ('R0', 'A_N2'):
                    metric = score_prediction(row['ungated'][arm], gt, box, valid,
                                              groups[obj]['permutations'], row['raw_hw'], record)
                    old = next(r for r in previous[mode]['ungated'][arm] if r['id'] == row['id'])
                    assert_same(old, metric, f'{mode}/{arm}/{row["id"]}')
                    checked += 1
    return dict(passed=True, metric_parity_checks=checked, images=222,
                GT_used_only_for_evaluation=True, GPU_used=False)


def movement(base, new):
    before, after = top(base), top(new)
    if before is None:
        assert after is None
        return []
    a, b = np.asarray(before['keypoints_xy'])[:8], np.asarray(after['keypoints_xy'])[:8]
    valid = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    return np.linalg.norm(b[valid] - a[valid], axis=-1).tolist()


@torch.no_grad()
def evaluate(model_name='existing'):
    prefix = model_name.upper()
    result_path = C.DOC / f'{prefix}_RESULTS.json'
    if result_path.exists():
        result = C.E.read(result_path)
        assert result['complete'] and result['model'] == model_name
        C.verify_bindings(result['source_bindings'])
        for binding in result['data_bindings'] + result['artifacts']:
            C.F.verify(binding)
        print(json.dumps(dict(stage='ALREADY_COMPLETE', model=model_name,
                              path=str(result_path)), ensure_ascii=False), flush=True)
        return result

    source = C.source_bindings()
    C.verify_bindings(source)
    protocol = C.E.bound(C.DOC / 'PROTOCOL.json')
    saved, previous, records, groups, pe, targets, bindings = evaluation_inputs()
    bindings.append(protocol)
    if model_name == 'finetuned':
        fit_path = C.DOC / 'FIT.json'
        fit = C.E.read(fit_path)
        assert fit['complete'] and fit['step'] == 300 and fit['trainability_pass']
        C.F.verify(fit['checkpoint'])
        bindings.extend([C.E.bound(fit_path), fit['checkpoint']])
    start = time.monotonic()
    C.E.gpu()
    model = C.load_model() if model_name == 'existing' else C.load_finetuned()
    model.eval()
    model.requires_grad_(False)
    metrics = {dataset: {arm: [] for arm in ARMS} for dataset in DATASETS}
    predictions = {'DEV72': [], 'GREEN150': []}
    previous_by_id = {dataset: {arm: {r['id']: r for r in previous[dataset]['ungated'][arm]}
                               for arm in ('R0', 'A_N2')} for dataset in DATASETS}
    preserves, parity_checks, done = 0, 0, 0
    all_movements = {dataset: {arm: [] for arm in ARMS} for dataset in ('DEV72', 'GREEN150')}
    for dataset in ('DEV72', 'GREEN150'):
        for oldrow, record in zip(saved[dataset], records[dataset]):
            C.F.verify(record['image'])
            assert oldrow['image'] == record['image']
            bgr = cv2.imread(str(C.ROOT / record['image']['path']))
            assert bgr is not None and list(bgr.shape[:2]) == oldrow['raw_hw']
            base = copy.deepcopy(oldrow['ungated']['R0'])
            input_copy = copy.deepcopy(base)
            raw = C.predict(model, bgr, input_copy, cap_fraction=None)
            assert_same(base, input_copy, 'model mutated saved R0 input', atol=0)
            raw_copy = copy.deepcopy(raw)
            capped = C.cap_prediction(base, raw, .01, bgr.shape[:2])
            assert_same(raw_copy, raw, 'cap mutated raw prediction', atol=0)
            outputs = dict(R0=base, A_N2=copy.deepcopy(oldrow['ungated']['A_N2']),
                           POSEFIX_RAW=raw, POSEFIX_CAP8=capped)
            for arm, prediction in outputs.items():
                assert prediction['selected_index'] == base['selected_index']
                preservation(base['candidates'], prediction['candidates'], base['selected_index'])
                preserves += 1
                values = movement(base, prediction)
                all_movements[dataset][arm].extend(values)
                if arm == 'POSEFIX_CAP8' and values:
                    assert max(values) <= .01 * np.hypot(*bgr.shape[:2]) + 1e-4
            predictions[dataset].append(dict(id=record['id'], image=record['image'],
                annotation=record['annotation'], session=record['session'],
                raw_hw=list(bgr.shape[:2]), predictions=outputs))

            # No target or target-conditioned choice is used until outputs are fixed.
            gt, box, modes, obj = read_targets(dataset, record, bgr.shape[:2], pe, targets)
            for mode, valid in modes:
                for arm, prediction in outputs.items():
                    metric = score_prediction(prediction, gt, box, valid,
                        groups[obj]['permutations'], bgr.shape[:2], record)
                    metrics[mode][arm].append(metric)
                    if arm in ('R0', 'A_N2'):
                        assert_same(previous_by_id[mode][arm][record['id']], metric,
                                    f'{mode}/{arm}/{record["id"]}')
                        parity_checks += 1
            done += 1
            if done % 20 == 0 or done == 222:
                print(json.dumps(dict(stage='EVALUATING', model=model_name,
                    done=done, total=222, elapsed_seconds=time.monotonic() - start,
                    gpu=C.E.gpu()), ensure_ascii=False), flush=True)

    results = {}
    for dataset, arm_metrics in metrics.items():
        results[dataset] = {arm: dict(**summary(rows),
            recovery_damage=recovery_damage(arm_metrics['R0'], rows),
            matched_recovery_damage=recovery_damage(arm_metrics['R0'], rows, True))
            for arm, rows in arm_metrics.items()}
    move_summary = {dataset: {arm: dict(corners=len(values),
        mean_px=float(np.mean(values)) if values else None,
        median_px=float(np.median(values)) if values else None,
        P90_px=float(np.quantile(values, .9)) if values else None,
        max_px=float(np.max(values)) if values else None)
        for arm, values in arms.items()} for dataset, arms in all_movements.items()}
    C.verify_bindings(source)
    for binding in bindings:
        C.F.verify(binding)
    pred_path = C.RAW / f'{prefix}_PREDICTIONS.json'
    metric_path = C.RAW / f'{prefix}_PER_FRAME_METRICS.json'
    C.freeze(pred_path, predictions)
    C.freeze(metric_path, metrics)
    result = dict(complete=True, model=model_name, source_bindings=source,
        data_bindings=bindings, artifacts=[C.E.bound(pred_path), C.E.bound(metric_path)],
        results=results, movements=move_summary, evaluated_images=done,
        preservation_checks=preserves, baseline_metric_parity_checks=parity_checks,
        original_R0_and_N2_metric_parity=True, independent_test=False,
        evaluation_reused=True, auto_promoted=False, GT_input=False,
        PnP_input=False, depth_input=False, CAD_input=False, dimensions_input=False,
        capped_prediction='same raw output; per-corner radial cap 0.01 image diagonal',
        image_drop=False, GT_per_corner_reassignment=False,
        checkpoint_selection='fixed checkpoint before evaluation; no evaluation selection',
        negative_FP='Not rerun; all detector boxes/scores/candidates unchanged',
        elapsed_seconds=time.monotonic() - start)
    C.freeze(result_path, result)
    print(json.dumps(dict(stage='COMPLETE', model=model_name, results=str(result_path)),
                     ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('existing', 'finetuned'), default='existing')
    parser.add_argument('--baseline-audit', action='store_true',
                        help='CPU-only metric parity audit; no model loading or inference')
    args = parser.parse_args()
    if args.baseline_audit:
        print(json.dumps(baseline_audit(), ensure_ascii=False))
    else:
        evaluate(args.model)


if __name__ == '__main__':
    main()
