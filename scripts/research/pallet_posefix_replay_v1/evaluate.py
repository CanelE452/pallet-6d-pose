"""Frozen 222-image paired evaluation of one RGB-only PoseFix replay checkpoint.

This is a reused development screen, not independent confirmation. No target,
dimension, camera, depth, CAD, or GT-based correspondence enters inference.
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
from scripts.research.pallet_posefix_replay_v1 import core as N
from scripts.research.pallet_posefix_large_error_v1 import evaluate as O

C = N.C
ARMS, DATASETS, EXAMPLE = O.ARMS, O.DATASETS, O.EXAMPLE
top = O.top


def verify_result(result):
    assert result['complete']
    C.verify_bindings(result['source_bindings'])
    for binding in result['data_bindings'] + result['artifacts']:
        N.F.verify(binding)


def history():
    """Read immutable previous arms; never select a checkpoint using their scores."""
    previous, bindings = {}, []
    for label, prefix in (('EXISTING_SYNTHETIC', 'EXISTING'),
                          ('REAL_ONLY', 'FINETUNED')):
        path = C.DOC / f'{prefix}_RESULTS.json'
        result = N.E.read(path)
        verify_result(result)
        assert result['evaluated_images'] == 222
        previous[label] = result
        bindings.append(N.E.bound(path))
    return previous, bindings


def delta_summary(current, previous):
    """Report all prespecified output modes, never the better mode alone."""
    result = {}
    for label, old in previous.items():
        result[label] = {}
        for dataset in DATASETS:
            result[label][dataset] = {}
            for arm in ARMS:
                after, before = current[dataset][arm], old['results'][dataset][arm]
                assert after['corners'] == before['corners']
                assert after['evaluable_frames'] == before['evaluable_frames']
                for key in ('detected', 'matched', 'missing'):
                    assert after[key] == before[key]
                if arm in ('R0', 'A_N2'):
                    O.assert_same(before, after, f'historical baseline/{label}/{dataset}/{arm}')
                result[label][dataset][arm] = dict(
                    PCK10_delta_pp=100 * (after['PCK']['10'] - before['PCK']['10']),
                    matched_median_delta_px=after['matched_pooled_corner8_median_px'] - before['matched_pooled_corner8_median_px'],
                    matched_P90_delta_px=after['matched_pooled_corner8_P90_px'] - before['matched_pooled_corner8_P90_px'],
                    recovered_delta=after['recovery_damage']['recovered'] - before['recovery_damage']['recovered'],
                    damaged_delta=after['recovery_damage']['damaged'] - before['recovery_damage']['damaged'])
    return result


def assert_preserved(base, prediction):
    assert prediction['selected_index'] == base['selected_index']
    O.preservation(base['candidates'], prediction['candidates'], base['selected_index'])
    before, after = top(base), top(prediction)
    if before is None:
        assert after is None
        return
    p, q = np.asarray(before['keypoints_xy']), np.asarray(after['keypoints_xy'])
    assert p.shape == q.shape == (9, 2)
    assert np.isfinite(q[np.isfinite(p)]).all(), 'Non-finite new coordinate'
    O.assert_same(before['keypoints_xy'][8], after['keypoints_xy'][8], 'centroid', atol=0)
    invalid = ~np.isfinite(p).all(-1) | (p == -1).all(-1)
    assert np.array_equal(p[invalid], q[invalid], equal_nan=True), 'Invalid-point mask changed'


@torch.no_grad()
def evaluate():
    N.setup()
    N.verify()
    result_path = N.DOC / 'REAL_RESULTS.json'
    if result_path.exists():
        result = N.E.read(result_path)
        assert result['model'] == 'replay'
        verify_result(result)
        print(json.dumps(dict(stage='ALREADY_COMPLETE', path=str(result_path))), flush=True)
        return result
    fit_path = N.DOC / 'FIT.json'
    fit = N.E.read(fit_path)
    assert fit['complete'] and fit['step'] == 300
    N.F.verify(fit['checkpoint'])
    source = {str(p.relative_to(ROOT)): N.E.bound(p) for p in (
        Path(__file__), Path(O.__file__), Path(C.__file__), Path(N.__file__))}
    C.verify_bindings(source)
    saved, previous, records, groups, pe, targets, bindings = O.evaluation_inputs()
    older, old_bindings = history()
    bindings.extend(old_bindings + [N.E.bound(N.DOC / 'PROTOCOL.json'),
                                    N.E.bound(fit_path), fit['checkpoint']])
    start = time.monotonic()
    N.E.gpu()
    model = N.load_model().eval().requires_grad_(False)
    metrics = {dataset: {arm: [] for arm in ARMS} for dataset in DATASETS}
    predictions = {'DEV72': [], 'GREEN150': []}
    previous_by_id = {dataset: {arm: {r['id']: r for r in previous[dataset]['ungated'][arm]}
                               for arm in ('R0', 'A_N2')} for dataset in DATASETS}
    preserves, parity_checks, done = 0, 0, 0
    movements = {dataset: {arm: [] for arm in ARMS} for dataset in predictions}
    for dataset in predictions:
        for oldrow, record in zip(saved[dataset], records[dataset]):
            N.F.verify(record['image'])
            assert oldrow['image'] == record['image']
            bgr = cv2.imread(str(ROOT / record['image']['path']))
            assert bgr is not None and list(bgr.shape[:2]) == oldrow['raw_hw']
            base = copy.deepcopy(oldrow['ungated']['R0'])
            inference_input = copy.deepcopy(base)
            raw = C.predict(model, bgr, inference_input, cap_fraction=None)
            O.assert_same(base, inference_input, 'model mutated R0 input', atol=0)
            raw_copy = copy.deepcopy(raw)
            capped = C.cap_prediction(base, raw, .01, bgr.shape[:2])
            O.assert_same(raw_copy, raw, 'cap mutated raw prediction', atol=0)
            outputs = dict(R0=base, A_N2=copy.deepcopy(oldrow['ungated']['A_N2']),
                           POSEFIX_RAW=raw, POSEFIX_CAP8=capped)
            for arm, prediction in outputs.items():
                assert_preserved(base, prediction)
                preserves += 1
                values = O.movement(base, prediction)
                movements[dataset][arm].extend(values)
                if arm == 'POSEFIX_CAP8' and values:
                    assert max(values) <= .01 * np.hypot(*bgr.shape[:2]) + 1e-4
            predictions[dataset].append(dict(id=record['id'], image=record['image'],
                annotation=record['annotation'], session=record['session'],
                raw_hw=list(bgr.shape[:2]), predictions=outputs))

            # Outputs are fixed before any annotation is read for this frame.
            gt, box, modes, obj = O.read_targets(dataset, record, bgr.shape[:2], pe, targets)
            for mode, valid in modes:
                for arm, prediction in outputs.items():
                    metric = O.score_prediction(prediction, gt, box, valid,
                        groups[obj]['permutations'], bgr.shape[:2], record)
                    metrics[mode][arm].append(metric)
                    if arm in ('R0', 'A_N2'):
                        O.assert_same(previous_by_id[mode][arm][record['id']], metric,
                                      f'{mode}/{arm}/{record["id"]}')
                        parity_checks += 1
            done += 1
            if done % 20 == 0 or done == 222:
                print(json.dumps(dict(stage='REPLAY_EVALUATING', done=done,
                    total=222, elapsed_seconds=time.monotonic() - start,
                    gpu=N.E.gpu()), ensure_ascii=False), flush=True)
    assert (done, preserves, parity_checks) == (222, 888, 744)
    results = {dataset: {arm: dict(**O.summary(rows),
        recovery_damage=O.recovery_damage(arm_metrics['R0'], rows),
        matched_recovery_damage=O.recovery_damage(arm_metrics['R0'], rows, True))
        for arm, rows in arm_metrics.items()} for dataset, arm_metrics in metrics.items()}
    move_summary = {dataset: {arm: dict(corners=len(values),
        mean_px=float(np.mean(values)) if values else None,
        median_px=float(np.median(values)) if values else None,
        P90_px=float(np.quantile(values, .9)) if values else None,
        max_px=float(np.max(values)) if values else None)
        for arm, values in arms.items()} for dataset, arms in movements.items()}
    deltas = delta_summary(results, older)
    N.verify()
    C.verify_bindings(source)
    for binding in bindings:
        N.F.verify(binding)
    pred_path, metric_path = N.RAW / 'PREDICTIONS.json', N.RAW / 'PER_FRAME_METRICS.json'
    N.freeze(pred_path, predictions)
    N.freeze(metric_path, metrics)
    result = dict(complete=True, model='replay', source_bindings=source,
        data_bindings=bindings, artifacts=[N.E.bound(pred_path), N.E.bound(metric_path)],
        results=results, movements=move_summary, deltas_vs_history=deltas,
        historical_results={label: value['results'] for label, value in older.items()},
        evaluated_images=done, preservation_checks=preserves,
        baseline_metric_parity_checks=parity_checks,
        original_R0_and_N2_metric_parity=True, independent_test=False,
        evaluation_reused=True, auto_promoted=False, GT_input=False,
        PnP_input=False, depth_input=False, CAD_input=False, dimensions_input=False,
        capped_prediction='same raw output; per-corner radial cap 0.01 image diagonal',
        image_drop=False, GT_per_corner_reassignment=False,
        checkpoint_selection='fixed last300 before evaluation; no evaluation selection',
        negative_FP='Not rerun; all detector boxes/scores/candidates unchanged',
        elapsed_seconds=time.monotonic() - start)
    N.freeze(result_path, result)
    print(json.dumps(dict(stage='REPLAY_REAL_COMPLETE', results=str(result_path))), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-audit', action='store_true',
                        help='CPU-only legacy metric parity check; no model inference')
    args = parser.parse_args()
    if args.baseline_audit:
        print(json.dumps(O.baseline_audit(), ensure_ascii=False))
    else:
        evaluate()


if __name__ == '__main__':
    main()
