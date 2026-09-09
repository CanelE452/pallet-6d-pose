"""Independently audit actual completed outputs; never train, infer or select.

Per-frame metrics use direct NumPy quantiles and literal 1001-threshold AUC.
Representative primary CIs physically repeat observations by resampled session;
no aggregate/statistic helper is called. Missing outputs cannot produce PASS.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ('point_only', 'hough_features', 'hough_joint')
SEEDS = (1, 2, 3)
KEYPOINT = ('keypoint_location_median_px', 'keypoint_location_p90_px')
POSE_FIELDS = {'rotation_median_deg': 'rotation_error_deg',
               'translation_median_cm': 'translation_error_cm',
               'iou3d_median': 'iou3d', 'add_sym_auc': 'add_sym_m'}
METRICS = KEYPOINT + tuple(POSE_FIELDS)
THRESHOLDS = (.001, .25, .5, .85)


def require(condition, text):
    if not condition:
        raise ValueError(text)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        if str(path) not in self.hashes:
            self.hashes[str(path)] = sha(path)
        digest = self.hashes[str(path)]
        require(expected is None or digest == expected, f'Bound input changed: {path}')
        return path

    def read(self, path):
        return json.loads(self.bind(path).read_text())

    def marker(self, path, allow_recorded_runtime_failure=False):
        path = Path(path).resolve()
        data = self.read(path)
        recorded_failure = (allow_recorded_runtime_failure and path.name == 'RUNTIME.json'
            and data.get('PASS') is False and data.get('parity_PASS') is False
            and data.get('timing_collection_complete') is True
            and data.get('status') == 'COMPLETE_WITH_STRICT_PARITY_FAILURE')
        require(data.get('complete') is True and (data.get('PASS', True) is True or recorded_failure),
                f'Actual complete artifact required: {path}')
        for name in ('input_sha256', 'source_sha256', 'output_sha256'):
            for file, digest in data.get(name, {}).items():
                file = Path(file)
                self.bind(file if file.is_absolute() else path.parent / file, digest)
        return data


def close(actual, expected, name, atol=1e-9):
    if actual is None or expected is None:
        require(actual is expected, f'Missing-value mismatch: {name}')
        return 0.
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), f'Nonfinite: {name}')
    delta = abs(float(actual) - float(expected))
    require(delta <= atol, f'{name}: independent={actual}, saved={expected}, delta={delta}')
    return delta


def audit_runtime(inputs, root):
    """Check collected timing and retain, rather than waive, strict failures."""
    runtime = inputs.marker(root / 'RUNTIME.json', allow_recorded_runtime_failure=True)
    require(runtime.get('timing_collection_complete') is True, 'Incomplete timing collection')
    policy = runtime['parity_policy']
    require(policy['atol'] == 1e-4 and policy['rtol'] == 0 and policy['criterion_changed'] is False,
            'Original runtime parity criterion must remain unchanged')
    expected_runs = {(arm, seed) for arm in ARMS for seed in SEEDS}
    require(len(runtime['runs']) == 9 and {(r['arm'], r['seed']) for r in runtime['runs']} == expected_runs,
            'Runtime must retain all nine models')
    frame_ids = runtime['frames']
    require(len(frame_ids) == len(set(frame_ids)) == 26 and runtime['repeats'] == 3,
            'Fixed timing denominator differs')
    expected_observations = {(frame, repeat) for frame in frame_ids for repeat in range(3)}
    fields = ('score', 'box_xyxy', 'keypoints_xy', 'keypoints_conf')
    observed_failures = set()
    all_maxima = {key: 0. for key in fields}
    for row in runtime['runs']:
        observations = row['observations']
        require(row['n'] == len(observations) == 78
                and {(o['frame_id'], o['repeat']) for o in observations} == expected_observations,
                'Runtime omitted or duplicated a fixed observation')
        times = [o['milliseconds'] for o in observations]
        require(all(math.isfinite(t) and t > 0 for t in times), 'Invalid observed timing')
        close(float(np.median(times)), row['median_ms'], 'runtime median')
        close(float(np.percentile(times, 90)), row['p90_ms'], 'runtime P90')
        close(float(np.mean(times)), row['mean_ms'], 'runtime mean')
        row_maximum = 0.
        for observation in observations:
            delta = observation['max_abs_delta_by_field']
            require(set(delta) == set(fields) and all(math.isfinite(v) and v >= 0 for v in delta.values()),
                    'Malformed runtime numerical comparison')
            expected_pass = all(v <= 1e-4 for v in delta.values())
            require(observation['strict_parity_pass'] is expected_pass, 'Runtime strict failure relabeled')
            if not expected_pass:
                observed_failures.add((row['arm'], row['seed'], observation['frame_id'], observation['repeat']))
            row_maximum = max(row_maximum, *delta.values())
            for field in fields:
                all_maxima[field] = max(all_maxima[field], delta[field])
        close(row_maximum, row['max_coordinate_or_score_delta'], 'runtime row maximum')
    failures = runtime['strict_failures']
    recorded_ids = {(f['arm'], f['seed'], f['frame_id'], f['repeat']) for f in failures}
    require(len(failures) == len(recorded_ids) and recorded_ids == observed_failures,
            'Original parity failures missing or duplicated')
    references = {}
    for failure in failures:
        label = f"{failure['arm']}_seed{failure['seed']}"
        if label not in references:
            references[label] = inputs.read(root / 'evaluation' / label / 'PREDICTIONS.json')['frames']
        actual = failure['actual_candidates']; reference = references[label][failure['key']]
        require(len(actual) == len(reference), 'Runtime candidate structure changed')
        delta = {field: 0. for field in fields}
        failed = False
        for left, right in zip(actual, reference):
            for field in fields:
                if left[field] is None or right[field] is None:
                    require(left[field] is right[field], 'Runtime field availability changed')
                    continue
                a, b = np.asarray(left[field]), np.asarray(right[field])
                require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
                        'Runtime finite shape contract differs')
                delta[field] = max(delta[field], float(np.max(np.abs(a - b))))
                failed |= not bool(np.allclose(a, b, atol=1e-4, rtol=0))
        require(failed, 'Saved strict parity failure does not reproduce')
        for field in fields:
            close(delta[field], failure['max_abs_delta_by_field'][field], 'runtime failure/' + field)
    strict_pass = not failures
    require(runtime['PASS'] is strict_pass and runtime['parity_PASS'] is strict_pass,
            'Failed strict runtime check was converted to PASS')
    require(runtime['status'] == ('COMPLETE' if strict_pass else 'COMPLETE_WITH_STRICT_PARITY_FAILURE'),
            'Runtime status hides validation failure')
    return dict(timing_collection_complete=True, observed_timing_count=702,
        strict_parity_PASS=strict_pass, strict_failure_count=len(failures),
        original_atol=1e-4, original_rtol=0, criterion_changed=False,
        maximum_abs_delta_by_field=all_maxima,
        interpretation='Timing arithmetic and failure records verified; failed prediction parity remains failed.')


def load_rows(inputs, directory, label):
    with inputs.bind(directory / 'PAPER_2D_per_frame.csv').open() as stream:
        csv_rows = [r for r in csv.DictReader(stream) if r['kind'] == 'POSITIVE']
    points = {r['frame_id']: dict(session=r['session_id'],
                errors=np.array([float(v) for v in r['top_keypoint_supervised_errors_px'].split(';') if v]))
              for r in csv_rows}
    require(len(csv_rows) == len(points) == 319, f'{label}: must retain all positive CSV rows')
    require(len({r['session'] for r in points.values()}) == 13, f'{label}: session denominator')
    pose_rows = inputs.read(directory / 'POSE_PER_FRAME_BY_ARM.json')['per_frame'][label]
    poses = {r['frame_id']: r for r in pose_rows}
    # Canonical 2D IDs use session:stamp; canonical pose IDs use session__stamp.
    # Join through the frozen image mapping rather than changing either ID set.
    axis = inputs.read(ROOT / 'data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json')
    pose_ids_by_image = {str((ROOT / r['image']).resolve()): r['frame_id'] for r in axis['frames_list']}
    allowed_pose_ids = {pose_ids_by_image[str((ROOT / r['image']).resolve())] for r in csv_rows}
    require(len(allowed_pose_ids) == 319 and len(poses) == len(pose_rows)
            and set(poses) <= allowed_pose_ids, f'{label}: pose frame identity')
    errors = [r['errors'] for r in points.values() if len(r['errors'])]
    flat = np.concatenate(errors) if errors else np.empty(0)
    require(np.isfinite(flat).all(), f'{label}: nonfinite point errors')
    metric = dict(zip(KEYPOINT, (float(np.median(flat)), float(np.quantile(flat, .9)))
                      if len(flat) else (None, None)))
    for name, field in POSE_FIELDS.items():
        if not pose_rows:
            metric[name] = None
            continue
        values = np.array([r[field] for r in pose_rows], dtype=float)
        require(np.isfinite(values).all(), f'{label}: nonfinite pose values')
        if name == 'add_sym_auc':
            diameter = float(np.median([r['diameter_m'] for r in pose_rows]))
            require(diameter > 0, f'{label}: positive diameter required')
            thresholds = np.linspace(0, diameter * .1, 1001)
            recalls = (values[:, None] <= thresholds[None]).mean(axis=0)
            metric[name] = float(np.trapz(recalls, thresholds) / thresholds[-1])
        else:
            metric[name] = float(np.median(values))
    return points, poses, metric, len(errors), len(flat)


def direct_primary_session_ci(stores, metric):
    """10k literal repeated-data medians, using the registered random draws."""
    keypoint = metric in KEYPOINT
    sets = [set(store) for store in stores]
    keys = sorted(set.intersection(*sets))
    if keypoint:
        keys = [k for k in keys if all(len(s[k]['errors']) for s in stores)]
    if not keys:
        return dict(status='NO_COMMON_OBSERVATIONS', paired_frames=0,
                    paired_sessions=0, difference=None, low=None, high=None)
    session_for = lambda row: row['session'] if keypoint else row['session_id']
    for key in keys:
        require(len({session_for(s[key]) for s in stores}) == 1, 'Paired session disagreement')
    sessions = sorted({session_for(stores[0][k]) for k in keys})
    series = []
    for store in stores:
        values, group = [], []
        for key in keys:
            value = store[key]['errors'] if keypoint else [store[key][POSE_FIELDS[metric]]]
            values.extend(value)
            group.extend([sessions.index(session_for(store[key]))] * len(value))
        series.append((np.array(values), np.array(group, dtype=int)))
    observed = sum(float(np.median(v)) for v, _ in series[:3]) / 3 - sum(float(np.median(v)) for v, _ in series[3:]) / 3
    rng = np.random.default_rng(20260902 if keypoint else 20260903)
    # The canonical artifact uses one RNG for frame draws, then session draws.
    # Consume the frame draws without invoking its weighted-quantile algorithm.
    for start in range(0, 10000, 128):
        rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=min(128, 10000 - start))
    draws = []
    for start in range(0, 10000, 128):
        count_batch = rng.multinomial(len(sessions), np.full(len(sessions), 1 / len(sessions)),
                                      size=min(128, 10000 - start))
        for counts in count_batch:
            # A session contributes all of its frames/corners on every repeat.
            statistics = [float(np.median(np.repeat(values, counts[group]))) for values, group in series]
            draws.append(sum(statistics[:3]) / 3 - sum(statistics[3:]) / 3)
    low, high = np.quantile(draws, [.025, .975])
    return dict(status='COMPLETE', paired_frames=len(keys), paired_sessions=len(sessions),
                difference=observed, low=float(low), high=float(high), resamples=10000,
                bootstrap_fraction_better=float(np.mean(np.asarray(draws) < 0)),
                method='Direct np.repeat of complete sessions and np.median; no aggregate helper.')


def ema_hash(checkpoint):
    # Import only to resolve stored model classes; no model forward is called.
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
    import integration  # noqa: F401
    import torch
    payload = torch.load(checkpoint, map_location='cpu')
    require(payload.get('complete') is True and payload.get('stage') == 'main', 'Not a completed main checkpoint')
    require(payload['epoch'] == 1 and payload['optimizer_steps'] == payload['expected_optimizer_steps'] == 6998,
            'Checkpoint actual epoch/update budget differs')
    require(payload.get('ema') is not None, 'Final epoch EMA missing')
    digest = hashlib.sha256()
    for name, tensor in sorted(payload['ema'].state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        require(bool(torch.isfinite(tensor).all()), f'Nonfinite EMA tensor: {name}')
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def audit(run_dir):
    root = Path(run_dir).resolve(); inputs = Inputs()
    protocol = inputs.marker(root / 'TRAIN_PROTOCOL.json')
    require(protocol['arms'] == list(ARMS) and protocol['training']['seeds'] == list(SEEDS), 'Registered arms/seeds differ')
    require(protocol['training']['batch'] == 16 and protocol['training']['epochs'] == 2
            and protocol['expected_optimizer_steps_per_cell'] == 6998, 'Registered main budget differs')
    summary = inputs.marker(root / 'SUMMARY.json'); verdict = inputs.marker(root / 'VERDICT.json')
    inputs.marker(root / 'AGGREGATE_COMPLETE.json')
    training_audit = inputs.marker(root / 'TRAINING_AUDIT.json')
    runtime_audit = audit_runtime(inputs, root)
    render = inputs.marker(root / 'REPORT_RENDER.json')
    require(summary['primary_arm'] == verdict['primary_arm'] == 'hough_joint'
            and summary['primary_reference'] == verdict['primary_reference'] == 'point_only', 'Primary comparison changed')
    document = inputs.bind(root / 'index.html', render['html_sha256']).read_text()
    embedded = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', document, flags=re.S)
    require(embedded is not None, 'Missing native report data')
    gallery = json.loads(embedded.group(1)); frames = {f['id']: f for f in gallery['frames']}
    require(len(frames) == len(gallery['frames']) == 319, 'HTML gallery denominator differs')
    expected_labels = {f'{arm}_seed{seed}' for arm in ARMS for seed in SEEDS}
    require(all(set(frame['runs']) == expected_labels for frame in frames.values()), 'HTML must show exactly the nine actual cells')
    expected_edges = [[i, (i + 1) % 4] for i in range(4)] + [[i, 4 + (i - 3) % 4] for i in range(4, 8)] + [[i, i + 4] for i in range(4)]
    require(gallery['edges'] == expected_edges, 'HTML cuboid role graph differs')
    trained_rows = {(r['arm'], r['seed']): r for r in training_audit['runs']}
    summary_rows = {(r['arm'], r['seed']): r for r in summary['runs']}
    require(len(trained_rows) == len(summary_rows) == 9, 'Must have nine actual cells')
    results, point_stores, pose_stores, checks, traces, ema_values = {}, {}, {}, [], {}, []
    comparable_budgets = []
    for arm in ARMS:
        point_stores[arm], pose_stores[arm] = [], []
        for seed in SEEDS:
            label = f'{arm}_seed{seed}'; directory = root / 'evaluation' / label
            print(f'Independent saved-output audit: {label}', flush=True)
            result = inputs.marker(directory / 'RESULTS.json')
            done = inputs.marker(directory / 'COMPLETION.json')
            trained = inputs.marker(root / 'runs' / label / 'COMPLETION.json')
            config = inputs.read(root / 'runs' / label / 'CELL_CONFIG.json')
            require(trained['stage'] == 'main' and trained['epochs_completed'] == 2
                    and trained['optimizer_steps'] == 6998 and trained['train_frames'] == 55980
                    and trained['val_frames'] == 4020, f'{label}: actual training denominator')
            require(result['arm'] == done['arm'] == trained['arm'] == arm
                    and result['seed'] == done['seed'] == trained['seed'] == seed, 'Cell identity mismatch')
            for receipt in (result, done):
                require(receipt['actual_positive_forwards'] == 319 and receipt['actual_negative_forwards'] == 2689
                        and receipt['baseline_candidate_copying'] is False, f'{label}: actual inference required')
            cp = inputs.bind(trained['checkpoint'], trained['checkpoint_sha256'])
            require(done['checkpoint_sha256'] == result['checkpoint_sha256'] == trained['checkpoint_sha256'], 'Wrong checkpoint')
            values_hash = ema_hash(cp)
            require(values_hash == trained_rows[(arm, seed)]['ema_tensor_sha256'], 'Independent EMA tensor hash differs')
            ema_values.append(values_hash)
            trace = inputs.bind(root / 'runs' / label / 'BATCH_TRACE.jsonl', trained['batch_trace_sha256'])
            traces.setdefault(seed, []).append(inputs.hashes[str(trace)])
            comparable_budgets.append({k: config[k] for k in ('epochs', 'batch', 'lr', 'optimizer', 'amp', 'train_frames', 'val_frames', 'training_recipe')})
            payload = inputs.read(directory / 'PREDICTIONS.json')
            require(payload.get('complete') is True and payload['actual_positive_forwards'] == 319
                    and payload['actual_negative_forwards'] == 2689 and payload['baseline_candidate_copying'] is False
                    and payload['weights_sha256'] == trained['checkpoint_sha256'], 'Prediction receipt/checkpoint differs')
            forward = payload['observations']; metadata = payload['frame_metadata']; predictions = payload['frames']
            require(len(forward) == len(metadata) == len(predictions) == 3008, f'{label}: missing or duplicate prediction rows')
            require(len({r['image'] for r in forward}) == 3008 and all(r['actual_forward'] is True for r in forward), 'Missing actual-forward observations')
            require(Counter(r['kind'] for r in forward) == Counter(positive=319, negative=2689), 'Actual forward population count')
            require(Counter(r['kind'] for r in metadata.values()) == Counter(positive=319, negative=2689), 'Metadata population count')
            require(set(predictions) == set(metadata) == {r['image'] for r in forward}, 'Forward/prediction image keys differ')
            for observation in forward:
                require(observation['candidate_count'] == len(predictions[observation['image']]), 'Observation candidate count differs')
                meta = metadata[observation['image']]
                require(observation['kind'] == meta['kind'] and observation['frame_id'] == meta['frame_id'], 'Forward metadata differs')
            negatives, html_compared = [], 0
            for image_key, meta in metadata.items():
                inputs.bind(meta['image_path'], meta['image_sha256'])
                candidates = predictions[image_key]
                if meta['kind'] == 'negative':
                    negatives.append(candidates)
                    continue
                shown = frames[meta['frame_id']]['runs'][label]
                top = max(candidates, key=lambda c: c['score']) if candidates else {}
                require(shown['points'] == top.get('keypoints_xy') and shown['box'] == top.get('box_xyxy')
                        and shown['score'] == top.get('score') and shown['n_candidates'] == len(candidates),
                        f'HTML differs from saved top-score candidate: {label}/{meta["frame_id"]}')
                html_compared += 1
            require(html_compared == 319, 'Missing HTML prediction row')
            neg = inputs.marker(directory / 'NEGATIVE_OUTCOMES.json')
            require(neg['n_frames'] == neg['actual_forward_count'] == len(negatives) == 2689, 'Negative denominator')
            require(neg['candidate_count'] == sum(map(len, negatives)), 'Negative candidate count')
            neg_check = {}
            for threshold in THRESHOLDS:
                count = sum(any(float(c['score']) >= threshold for c in cs) for cs in negatives)
                candidate_count = sum(float(c['score']) >= threshold for cs in negatives for c in cs)
                expected = neg['by_threshold'][str(threshold)]
                require(expected['n_frames_with_detection'] == count and expected['n_candidates'] == candidate_count, 'Negative threshold counts differ')
                close(count / 2689, expected['fraction_frames_with_detection'], 'negative rate', 1e-12)
                require(result['negative']['by_threshold'][str(threshold)] == expected, 'RESULTS negative outcome differs')
                neg_check[str(threshold)] = dict(frames=count, candidates=candidate_count)
            points, poses, recomputed, matched, observed_points = load_rows(inputs, directory, label)
            require(matched == result['two_d']['keypoint_matched_frame_count_iou50'], '2D matched-frame coverage differs')
            require(len(poses) == result['main_6d'].get('n', 0), 'MAIN6D observation count differs')
            close(len(poses) / 319, result['main_6d_coverage'], 'MAIN6D coverage', 1e-12)
            if 'keypoint_all_labeled' in result['two_d']:
                require(result['two_d']['keypoint_all_labeled']['count'] == observed_points, 'Observed point-error denominator differs')
            discrepancies = {}
            for metric in METRICS:
                section = 'two_d' if metric in KEYPOINT else 'main_6d'
                reported = result[section].get(metric)
                discrepancies[metric] = close(recomputed[metric], reported, label + '/' + metric,
                                               5.1e-7 if metric in KEYPOINT else 1e-9)
                close(summary_rows[(arm, seed)][section].get(metric), reported, 'SUMMARY run/' + metric)
            point_stores[arm].append(points); pose_stores[arm].append(poses)
            results[(arm, seed)] = result
            checks.append(dict(arm=arm, seed=seed, actual_updates=6998, actual_forwards=3008,
                observed_point_errors=observed_points, matched_frames=matched, MAIN_pose_frames=len(poses),
                metric_absolute_differences=discrepancies, negative_recomputed=neg_check,
                HTML_saved_candidate_rows_exact=html_compared, EMA_tensor_sha256=values_hash))
    require(len(set(ema_values)) == 9, 'EMA parameter values duplicated across trained cells')
    require(all(len(set(v)) == 1 for v in traces.values()), 'Paired training tensors differ across arms')
    require(len({v[0] for v in traces.values()}) == 3, 'Seeds have identical training schedules')
    require(all(v == comparable_budgets[0] for v in comparable_budgets), 'Training budgets/recipes differ')
    families = {}
    for arm in ARMS:
        families[arm] = {}
        for metric in METRICS:
            section = 'two_d' if metric in KEYPOINT else 'main_6d'
            values = [results[(arm, seed)][section].get(metric) for seed in SEEDS]
            if any(v is None for v in values):
                mean, sd = None, None
            else:
                mean = sum(values) / 3
                sd = math.sqrt(sum((v - mean) ** 2 for v in values) / 2)
            expected = summary['arms'][arm]['metrics'][metric]
            require(expected['seeds'] == values, f'{arm}/{metric}: per-seed values reordered')
            close(mean, expected['mean'], arm + '/mean/' + metric)
            close(sd, expected['std'], arm + '/ddof1/' + metric)
            families[arm][metric] = dict(mean=mean, sample_std_ddof1=sd)
    representative_ci = {}
    for metric in ('keypoint_location_median_px', 'translation_median_cm'):
        stores = point_stores if metric in KEYPOINT else pose_stores
        calculated = direct_primary_session_ci(stores['hough_joint'] + stores['point_only'], metric)
        expected = summary['comparisons']['hough_joint_vs_point_only']['metrics'][metric]
        require(calculated['paired_frames'] == expected['paired_frames']
                and calculated['paired_sessions'] == expected['paired_sessions'], 'Paired CI population differs')
        close(calculated['difference'], expected['difference'], 'paired difference/' + metric)
        for key in ('low', 'high'):
            close(calculated[key], expected['session_cluster'][key], 'direct bootstrap/' + metric)
        representative_ci[metric] = calculated
    metric_confirmed = {}
    for metric in METRICS:
        value = families['hough_joint'][metric]['mean']
        reference = families['point_only'][metric]['mean']
        lower = metric in KEYPOINT or metric in ('rotation_median_deg', 'translation_median_cm')
        interval = summary['comparisons']['hough_joint_vs_point_only']['metrics'][metric]['session_cluster']
        direction = value is not None and reference is not None and (value < reference if lower else value > reference)
        ci = interval['high'] is not None and interval['low'] is not None and (interval['high'] < 0 if lower else interval['low'] > 0)
        metric_confirmed[metric] = bool(direction and ci)
    kp_coverage = all(results[('hough_joint', seed)]['two_d']['keypoint_matched_frame_count_iou50'] >=
                      results[('point_only', seed)]['two_d']['keypoint_matched_frame_count_iou50'] for seed in SEEDS)
    pose_coverage = all(results[('hough_joint', seed)]['main_6d_coverage'] >=
                        results[('point_only', seed)]['main_6d_coverage'] for seed in SEEDS)
    decisions = dict(keypoint_gain_confirmed=kp_coverage and all(metric_confirmed[m] for m in KEYPOINT),
                     pose_gain_confirmed=pose_coverage and all(metric_confirmed[m] for m in POSE_FIELDS))
    decisions['overall_accuracy_improved'] = decisions['keypoint_gain_confirmed'] and decisions['pose_gain_confirmed']
    require(all(verdict[key] is value for key, value in decisions.items()), 'Verdict differs from registered numerical criteria')
    require(all(sha(path) == digest for path, digest in inputs.hashes.items()), 'Inputs changed during independent audit')
    result = dict(schema='pallet_dht_joint_independent_final_audit_v1', complete=True, PASS=True,
        scope='Actual saved-output integrity and independent arithmetic; PASS is not an accuracy-improvement claim.',
        device='cpu', model_forwards=0, primary_arm='hough_joint', primary_reference='point_only',
        cells=checks, recomputed_seed_statistics=families, representative_independent_session_bootstrap=representative_ci,
        independently_checked_decision=decisions, runtime_audit=runtime_audit,
        total_actual_positive_forwards=9 * 319, total_actual_negative_forwards=9 * 2689,
        HTML_saved_prediction_rows_exact=9 * 319, same_seed_augmented_trace_exact=True,
        nine_distinct_actual_EMA_parameter_states=True,
        numeric_tolerances=dict(keypoint_CSV_six_decimal_reconstruction_absolute=5.1e-7,
                                other_metric_and_CI_absolute=1e-9, HTML_candidate_values='exact JSON values'),
        limitations=['Canonical 6D solver and GT generation are bound inputs, not rerun here.',
                     'AP is not independently recomputed; four negative FP thresholds are recomputed from every candidate.',
                     'Two representative primary session CIs are recomputed; other bootstrap intervals retain their bound original computation.',
                     'HTML coordinate/role graph integrity is checked; browser interaction and perceptual visual QA are separate.',
                     'Reported keypoint supervision count is the entire GT population; pooled errors and matched-frame coverage use observed predictions.'],
        source_sha256={str(Path(__file__).resolve()): sha(__file__)},
        input_sha256=inputs.hashes, output_sha256={})
    target = root / 'INDEPENDENT_FINAL_AUDIT.json'
    temporary = target.with_suffix('.pending.json')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)
    print(json.dumps(dict(PASS=True, output=str(target), sha256=sha(target))))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    audit(parser.parse_args().run_dir)
