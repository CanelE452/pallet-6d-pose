"""Original-ID evaluation of frozen global-layout predictions; no model calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.research.pallet_dht_decoder_probe_v1 import evaluate as METRIC

ROOT = Path(__file__).resolve().parents[3]
ARMS = ('baseline', 'independent', 'global', 'global_shared_only')


def sha(path):
    return METRIC.sha(path)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    METRIC.write(path, value)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def iou(first, second):
    left, top = max(float(first[0]), float(second[0])), max(float(first[1]), float(second[1]))
    right, bottom = min(float(first[2]), float(second[2])), min(float(first[3]), float(second[3]))
    intersection = max(0., right-left)*max(0., bottom-top)
    a = max(0., float(first[2]-first[0]))*max(0., float(first[3]-first[1]))
    b = max(0., float(second[2]-second[0]))*max(0., float(second[3]-second[1]))
    return intersection/(a+b-intersection) if a+b-intersection > 0 else 0.


def candidate_diagnostic(record, xy, observed, inputs):
    """Post-selection GT nearest-candidate bounds; never feeds selection."""
    ev = record['evidence']
    path = inputs.bind(ev['candidate_evidence_path'], ev['candidate_evidence_sha256'])
    with np.load(path, allow_pickle=False) as z:
        raw = z['line_fusion__candidates_xy'][:, 1:].copy()
        raw_valid = z['line_fusion__candidate_valid'][:, 1:].copy()
    pruned = np.asarray(ev['candidate_xy'], float)
    pruned_valid = np.asarray(ev['candidate_valid'], bool)
    base, base_valid = METRIC.finite_points(record['baseline']['points'], record['baseline']['point_valid'])
    require(pruned.shape == (8, 16, 2), 'Candidate diagnostic requires the registered16 pool')
    points = []
    for i in range(8):
        candidates = np.concatenate((base[:8], raw[i]), axis=0)
        valid = np.r_[base_valid[:8], raw_valid[i]] & np.isfinite(candidates).all(1)
        distances = np.where(valid, np.linalg.norm(candidates-xy[i], axis=1), np.inf)
        reduced_distances = np.where(pruned_valid[i], np.linalg.norm(pruned[i]-xy[i], axis=1), np.inf)
        nearest_original, nearest_pruned = int(np.argmin(distances)), int(np.argmin(reduced_distances))
        points.append(dict(point_id=i, observed=bool(observed[i]),
            nearest_original56_distance_px=float(distances[nearest_original]) if np.isfinite(distances[nearest_original]) else None,
            nearest_pruned16_distance_px=float(reduced_distances[nearest_pruned]) if np.isfinite(reduced_distances[nearest_pruned]) else None,
            nearest_original56_index=nearest_original, nearest_pruned16_index=nearest_pruned,
            original56_index_definition='0..7 baseline predicted point IDs;8..55 original48 line intersections.',
            selected_global_distance_px=float(np.linalg.norm(np.asarray(record['arms']['global']['points'])[i]-xy[i])) if observed[i] else None))
    return dict(uses_GT=True, used_for_selection=False, operational_prediction=False, points=points,
        limitation='Per-corner nearest locations may not form a geometrically admissible joint layout. Diagnostic only; no oracle performance claim.')


def layout_diagnostic(record, xy, observed):
    out = {}
    for arm in ARMS[1:]:
        payload = record['arms'][arm]
        items = [(f'top{j+1}', h) for j, h in enumerate(payload['top_hypotheses'])]
        items += [(f'explicit:{name}', h) for name, h in payload.get('explicit_hypotheses', {}).items()]
        stats = []
        for name, h in items:
            pp = np.asarray(h['points_xy'], float)
            errors = np.linalg.norm(pp-xy, axis=1)
            valid_errors = errors[observed]
            stats.append(dict(name=name, hypothesis_index=h.get('index'), kind=h.get('kind'), total=h.get('total'),
                raw_terms=h.get('raw_terms'), weighted_terms=h.get('weighted_terms'),
                same_ID_errors_px=[float(e) if ok else None for e, ok in zip(errors, observed)],
                observed_mean_px=float(valid_errors.mean()) if len(valid_errors) else None,
                observed_median_px=float(np.median(valid_errors)) if len(valid_errors) else None,
                uses_GT_for_diagnostic=True, used_for_selection=False))
        out[arm] = stats
    return out


def continuation(summaries, difficulty, paired, protocol):
    ss = {s['arm']: s for s in summaries if s['population'] == 'real_dev'}
    dd = {(d['arm'], d['difficulty']): d for d in difficulty if d['population'] == 'real_dev'}
    pp = {p['right']: p for p in paired if p['population'] == 'real_dev' and p['left'] == 'global'}
    gate = protocol['continuation_gates']
    good, reference_good = dd['global', 'easy_le10'], dd['independent', 'easy_le10']
    hard = dd['global', 'hard_gt20']
    fraction = (hard['baseline_mean_px']-hard['after_mean_px'])/hard['baseline_mean_px'] if hard['n_points'] and hard['baseline_mean_px'] > 0 else None
    checks = dict(median_and_p90_not_worse_than_both_controls=all(ss['global'][metric] <= ss[arm][metric]
        for metric in ('median_px', 'p90_px') for arm in ('baseline', 'independent')),
        baseline_hard_mean_reduction_at_least_registered_fraction=fraction is not None and fraction >= gate['baseline_hard_mean_reduction_min_fraction'],
        baseline_good_crossing_at_most_registered_fraction=good['easy_to_over10_rate'] is not None and good['easy_to_over10_rate'] <= gate['baseline_good_crossing_max_fraction'],
        baseline_good_crossing_not_more_than_independent=good['easy_to_over10_rate'] is not None and reference_good['easy_to_over10_rate'] is not None and good['easy_to_over10_rate'] <= reference_good['easy_to_over10_rate'],
        both_bootstrap_ci_upper_below_zero=all(pp[a]['ci95'][1] is not None and pp[a]['ci95'][1] < 0 for a in ('baseline', 'independent')))
    return dict(continuation_signal=all(checks.values()), checks=checks, registered_gates=gate,
        baseline_hard_mean_reduction_fraction=fraction, baseline_good_crossing_fraction=good['easy_to_over10_rate'],
        independent_good_crossing_fraction=reference_good['easy_to_over10_rate'], stable_gain_claim=False,
        meaning='Registered exploratory continuation criteria; frozen single backbone and reused DEV319, no independent FINAL.')


def evaluate(run_dir):
    run_dir = Path(run_dir).resolve()
    inputs = METRIC.Inputs()
    protocol = inputs.read(run_dir/'PROTOCOL.json')
    require(protocol['evaluation']['bootstrap_seed'] == METRIC.BOOTSTRAP_SEED, 'Bootstrap seed differs from reused metric implementation')
    source = ROOT/protocol['input_run']
    selection = inputs.read(run_dir/'CALIBRATION_SELECTION.json')
    require(selection['complete'] and selection['PASS'] and selection['no_real_GT_used'], 'Frozen synthetic-only calibration required')
    require(selection['protocol_sha256'] == sha(run_dir/'PROTOCOL.json'), 'Calibration protocol changed')
    predictions = inputs.read(run_dir/'PREDICTIONS.json')
    require(predictions['complete'] and predictions['PASS'] and predictions['selection_uses_GT'] is False, 'Completed GT-free predictions required')
    require(predictions['calibration_selection_sha256'] == sha(run_dir/'CALIBRATION_SELECTION.json'), 'Prediction/calibration binding differs')
    old_completion = inputs.read(source/'EVALUATION_COMPLETION.json')
    old_metrics = inputs.read(source/'FRAME_METRICS.json', old_completion['output_sha256']['FRAME_METRICS.json'])
    old_results = inputs.read(source/'RESULTS.json', old_completion['output_sha256']['RESULTS.json'])
    old = {r['id']: r for r in old_metrics['records']}
    manifest = inputs.read(source/'MANIFEST.json')
    source_records = {r['id']: r for r in manifest['records']}
    real_manifest = inputs.read(ROOT/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json')
    items = {r['frame_id']: r for r in real_manifest['items']}
    require(real_manifest['role'] == 'DEV' and len(items) == 319, 'Reused DEV319 required')
    records = predictions['records']
    require(len(records) == 831 and len({r['id'] for r in records}) == 831, '512+319 unique evaluation frames required')
    require({r['id'] for r in records if r['population'] == 'real_dev'} == set(items), 'Real denominator differs')
    rows, annotation_lookup = [], {}
    max_baseline_error_delta = 0.
    for record in records:
        fid = record['id']
        require(record['population'] in ('synth_val', 'real_dev'), 'Calibration/train cannot enter reported metrics')
        previous = old[fid]
        require(previous['population'] == record['population'], 'Population changed')
        if record['population'] == 'real_dev':
            item = items[fid]
            gt_path = (ROOT/item['gt_v2_path']).resolve()
            obj = inputs.read(gt_path, old_completion['input_sha256'][str(gt_path)])['objects'][0]
            annotation_lookup[fid] = obj['keypoint_annotations']
            xy = np.asarray([p['xy'] for p in obj['keypoint_annotations']], float)
            valid_gt = np.asarray([p['visibility'] > 0 for p in obj['keypoint_annotations']], bool)
            target_box = np.r_[xy[:8].min(0), xy[:8].max(0)]
            candidate_box = record['baseline']['box_xyxy']
            matched = candidate_box is not None and iou(np.asarray(candidate_box), target_box) >= .5
            require(Path(record['image']).resolve() == (ROOT/item['image_path']).resolve(), 'Image identity differs')
        else:
            sr = source_records[fid]['source_record']
            target = np.asarray(sr['targets'][0]['keypoints_normalized'], float)
            h, w = sr['prepared_shape_hw']
            xy = target[:, :2]*[w, h]-sr['reflect_pad_px']
            valid_gt = target[:, 2] > 0
            # Preserve the existing synthetic score-only selected-instance loss match.
            matched = bool(record['synthetic_loss_matched'])
        require(np.array_equal(xy, previous['gt_points']) and np.array_equal(valid_gt, previous['gt_supervised']), 'Frozen GT coordinate/mask differs')
        require(matched == previous['baseline_match_iou50'], 'Original matching changed')
        base, base_valid = METRIC.finite_points(record['baseline']['points'], record['baseline']['point_valid'])
        require(np.array_equal(base, np.asarray(previous['arms']['baseline']['points'])), 'Frozen baseline points changed')
        arm_rows = {}
        for arm in ARMS:
            payload = record['baseline'] if arm == 'baseline' else record['arms'][arm]
            point, valid = METRIC.finite_points(payload['points'], payload['point_valid'])
            require(np.array_equal(valid, base_valid), 'Candidate selection changed point availability')
            require(np.array_equal(point[8], base[8]), 'Centroid8 changed')
            observed = valid_gt & valid & matched
            distances = np.linalg.norm(point-xy, axis=1)
            movement = np.linalg.norm(point-base, axis=1)
            errors = [float(e) if ok else None for e, ok in zip(distances, observed)]
            if arm == 'baseline':
                for actual, saved in zip(errors, previous['arms']['baseline']['errors_px']):
                    require((actual is None) == (saved is None), 'Baseline observation changed')
                    if actual is not None:
                        max_baseline_error_delta = max(max_baseline_error_delta, abs(actual-saved))
            arm_rows[arm] = dict(errors_px=errors, move_px=[float(m) if ok else None for m, ok in zip(movement, valid)], points=payload['points'])
        rows.append(dict(id=fid, population=record['population'], session_id=record['session_id'], gt_points=xy.tolist(),
            gt_supervised=valid_gt.tolist(), baseline_match_iou50=matched, arms=arm_rows,
            candidate_diagnostic=candidate_diagnostic(record, xy, valid_gt & base_valid & matched, inputs),
            layout_diagnostic=layout_diagnostic(record, xy, valid_gt & base_valid & matched)))
    require(max_baseline_error_delta == 0., 'Baseline numerical replay differs')
    summaries, corners, difficulty, pairs = [], [], [], []
    for pop in ('synth_val', 'real_dev'):
        selected = [r for r in rows if r['population'] == pop]
        for arm in ARMS:
            summaries.append(dict(population=pop, arm=arm, **METRIC.summarize(selected, arm)))
            corners.append(dict(population=pop, arm=arm, **METRIC.summarize(selected, arm, tuple(range(8)))))
            difficulty.extend(dict(population=pop, **d) for d in METRIC.point_difficulty(selected, arm))
        if pop == 'real_dev':
            require(len(selected) == 319 and len({r['session_id'] for r in selected}) == 13, 'Real frame/session count changed')
            require(METRIC.summarize(selected, 'baseline')['n_observed_points'] == 2738, 'Real observed-point count changed')
            for right in ('baseline', 'independent'):
                pairs.append(dict(population=pop, **METRIC.paired(selected, 'global', right,
                    resamples=protocol['evaluation']['bootstrap_resamples'])))
    real = [r for r in rows if r['population'] == 'real_dev']
    cohorts = {}

    def cohort(name, keys, description):
        keys = set(keys)
        masked = []
        for row in real:
            if not any((row['id'], i) in keys for i in range(9)):
                continue
            rr = {**row, 'gt_supervised': [v and (row['id'], i) in keys for i, v in enumerate(row['gt_supervised'])], 'arms': {}}
            for arm in ARMS:
                ar = row['arms'][arm]
                rr['arms'][arm] = {**ar, 'errors_px': [v if (row['id'], i) in keys else None for i, v in enumerate(ar['errors_px'])],
                    'move_px': [v if (row['id'], i) in keys else None for i, v in enumerate(ar['move_px'])]}
            masked.append(rr)
        cohorts[name] = dict(description=description, pixel_accuracy_certified=False,
            point_keys=[list(k) for k in sorted(keys)], summaries=[dict(arm=a, **METRIC.summarize(masked, a)) for a in ARMS])

    cohort('recorded_visible', [(fid, i) for fid, aa in annotation_lookup.items() for i, a in enumerate(aa) if a['visibility'] == 2], 'Recorded visibility=2; metadata does not certify accuracy.')
    cohort('explicit_manual_click', [(fid, i) for fid, aa in annotation_lookup.items() for i, a in enumerate(aa) if a.get('source') == 'manual_click'], 'Explicit click-source metadata; not independent measured click accuracy.')
    review_dir = ROOT/'data/pallet/results/pallet_dht_gt_audit_v1'
    root_review = inputs.read(review_dir/'ROOT_GT_VISUAL_REVIEW.json')
    specialist = inputs.read(review_dir/'specialist_gt_review.json')
    for label in sorted({r['selected_point_assessment'] for r in root_review['records']}):
        cohort('root_'+label, [(r['id'], int(r['point'])) for r in root_review['records'] if r['selected_point_assessment'] == label],
            'Previously selected GT-only qualitative point review; small, error-enriched, not independent or pixel-certified.')
    specialist_classes = sorted({p['review_class'] for f in specialist['reviewed_frames'] for p in f['points']})
    for label in specialist_classes:
        cohort('specialist_'+label, [(f['frame_id'], int(p['index'])) for f in specialist['reviewed_frames'] for p in f['points'] if p['review_class'] == label],
            'Pre-existing specialist qualitative point class; fixed before current global selection; not numeric GT correction.')
    reference = next(s for s in old_results['summaries'] if s['population'] == 'real_dev' and s['arm'] == 'baseline')
    baseline_summary = next(s for s in summaries if s['population'] == 'real_dev' and s['arm'] == 'baseline')
    require(all(baseline_summary[k] == reference[k] for k in ('median_px', 'p90_px', 'mean_px', 'n_observed_points')), 'Baseline summary changed')
    decision = continuation(summaries, difficulty, pairs, protocol)
    by_session = {s: [dict(arm=a, **METRIC.summarize([r for r in real if r['session_id'] == s], a)) for a in ARMS]
        for s in sorted({r['session_id'] for r in real})}
    record_lookup = {r['id']: r for r in records}
    resolutions = {fid: f"{record_lookup[fid]['width']}x{record_lookup[fid]['height']}" for fid in items}
    by_resolution = {s: [dict(arm=a, **METRIC.summarize([r for r in real if resolutions[r['id']] == s], a)) for a in ARMS]
        for s in sorted(set(resolutions.values()))}
    selection_behavior = {}
    real_predictions = [r for r in records if r['population'] == 'real_dev']
    for arm in ARMS[1:]:
        selection_behavior[arm] = dict(n_frames=len(real_predictions),
            n_unchanged_frames=sum(np.array_equal(r['arms'][arm]['points'], r['baseline']['points']) for r in real_predictions),
            n_input_fallback_frames=sum(r['arms'][arm]['fallback_reason'] is not None for r in real_predictions),
            n_baseline_invalid_geometry_exception_available=sum(r['arms'][arm].get('baseline_geometry_exception', False) for r in real_predictions),
            n_baseline_invalid_geometry_exception_selected=sum(r['arms'][arm].get('baseline_exception_selected', False) for r in real_predictions))
    results = dict(schema='pallet_dht_global_layout_results_v1', complete=True, PASS=True,
        summaries=summaries, corner_only_summaries=corners, difficulty=difficulty, paired=pairs, cohorts=cohorts,
        continuation=decision, by_session=by_session, by_image_resolution=by_resolution,
        selection_behavior=selection_behavior, n_real_frames=319, n_real_sessions=13, n_synthetic_validation_frames=512,
        no_new_CNN_forwards=True, no_new_training=True, no_GT_modifications=True, independent_final=False,
        selected_weights=selection['selected'], metric_scope='Same original semantic IDs; nine points including unchanged centroid; original visibility and match masks.',
        baseline_error_replay_max_abs_delta_px=max_baseline_error_delta,
        limitations=['Approximate bounded beam search, not guaranteed global optimum.',
            'Frozen hough_joint seed1 EMA and previously seen synthetic pools; 256-frame synthetic-only calibration.',
            'Reused real DEV319 and qualitative review subsets; no independent FINAL or trained-seed replication.',
            'Each subset changes the denominator and is not corrected whole-population performance.',
            'Session paired bootstrap is descriptive conditional on these 13 capture sessions.'])
    inputs.bind(__file__)
    inputs.bind(METRIC.__file__)
    inputs.verify()
    write(run_dir/'FRAME_METRICS.json', dict(schema='pallet_dht_global_layout_frame_metrics_v1', complete=True, records=rows))
    write(run_dir/'RESULTS.json', results)
    write(run_dir/'EVALUATION_COMPLETION.json', dict(schema='pallet_dht_global_layout_evaluation_complete_v1', complete=True, PASS=True,
        created_at_utc=datetime.now(timezone.utc).isoformat(), input_sha256=inputs.hashes,
        output_sha256={n: sha(run_dir/n) for n in ('FRAME_METRICS.json', 'RESULTS.json')},
        calibration_selection_sha256=sha(run_dir/'CALIBRATION_SELECTION.json'),
        predictions_sha256=sha(run_dir/'PREDICTIONS.json'), protocol_sha256=sha(run_dir/'PROTOCOL.json'),
        n_real_frames=319, n_observed_real_points=2738, no_new_model_forward=True,
        original_GT_and_baseline_preserved=True, baseline_error_replay_max_abs_delta_px=max_baseline_error_delta))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    evaluate(parser.parse_args().run_dir)
