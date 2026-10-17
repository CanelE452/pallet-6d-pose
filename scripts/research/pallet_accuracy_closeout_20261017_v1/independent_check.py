"""Independent scalar C2 metrics, distribution, paired and cap arithmetic.

No imports from the new replay/evaluator or original pose solvers. References
are opened only after the stored prediction seals have been checked.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import time

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(pose, reference):
    if not pose.get('available'):
        return dict(translation_cm=None, rotation_deg=None, success=False, pose_available=False)
    assert reference['order'] == 2
    translation = 100*math.dist(pose['centroid'], reference['t'])
    P, G = pose['R_physical'], reference['R']
    traces = [math.fsum(P[i][j]*G[i][j]*sign[j] for i in range(3) for j in range(3))
              for sign in ((1, 1, 1), (-1, 1, -1))]
    rotation = min(math.degrees(math.acos(max(-1., min(1., (tr-1)/2)))) for tr in traces)
    return dict(translation_cm=translation, rotation_deg=rotation,
                success=translation <= 5 and rotation <= 5, pose_available=True)


def quantile(values, probability):
    values = sorted(values)
    position = probability*(len(values)-1)
    left = int(position)
    return values[left]+(position-left)*(values[min(left+1, len(values)-1)]-values[left])


def distribution(values):
    values = [v for v in values if v is not None]
    if not values:
        return dict(n=0, mean=None, sample_std=None, sample_variance=None,
                    median=None, P90=None, P95=None, max=None)
    return dict(n=len(values), mean=statistics.mean(values),
        sample_std=statistics.stdev(values) if len(values)>1 else None,
        sample_variance=statistics.variance(values) if len(values)>1 else None,
        median=statistics.median(values), P90=quantile(values, .90),
        P95=quantile(values, .95), max=max(values))


def cluster_ci(values, frames, key, seed, draws):
    """Independent grouped scalar sums and explicit resampling arithmetic."""
    groups = sorted({frame[key] for frame in frames})
    if len(groups) < 2:
        return None
    sums = [math.fsum(v for f, v in zip(frames, values) if f[key] == group and v is not None) for group in groups]
    counts = [sum(f[key] == group and v is not None for f, v in zip(frames, values)) for group in groups]
    schedule = np.random.default_rng(seed).multinomial(len(groups), [1/len(groups)]*len(groups), draws)
    estimates = []
    for multiplicity in schedule:
        n = sum(int(w)*count for w, count in zip(multiplicity, counts))
        if n:
            estimates.append(math.fsum(int(w)*value for w, value in zip(multiplicity, sums))/n)
    return [quantile(estimates, .025), quantile(estimates, .975)] if estimates else None


def run(task_private, baseline_folder, output):
    start, cpu = time.monotonic(), time.process_time()
    private, baseline = map(Path, (task_private, baseline_folder))
    seal = read(private/'posefix192_v1/PREDICTION_SEAL.json')
    for name, digest in seal['artifacts'].items():
        assert sha(private/'posefix192_v1'/name) == digest
    old_seal = read(baseline/'SCORE_SEAL.json')
    for name in ('BASELINE_PREDICTIONS.jsonl.gz', 'FIXED_REFERENCES.json'):
        assert sha(baseline/name) == old_seal['artifacts'][name]
    base_rows = rows(baseline/'BASELINE_PREDICTIONS.jsonl.gz')
    base_ids = [r['id'] for r in base_rows]
    assert len(base_ids) == len(set(base_ids)) == 238
    newer = rows(private/'posefix192_v1/PREDICTIONS.jsonl.gz')
    assert len(newer) == 714
    indexed = {(r['id'], r['method']): r for r in newer}
    assert len(indexed) == 714
    expected_arms = ('R0', 'N0', 'E0', 'POSEFIX_RAW', 'POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR')
    # All files/seals above are checked before this reference decode.
    refs = read(baseline/'FIXED_REFERENCES.json')['references']
    measured = {}
    largest = 0.
    def compare(actual, expected, label):
        nonlocal largest
        if isinstance(actual, bool) or actual is None or isinstance(actual, str) or isinstance(actual, int):
            assert actual == expected, (label, actual, expected)
        else:
            difference = abs(float(actual)-float(expected))
            largest = max(largest, difference)
            assert difference <= 1e-9, (label, actual, expected, difference)
    for frame in base_rows:
        fid = frame['id']
        for arm in expected_arms:
            entry = frame['predictions'][arm] if arm in frame['predictions'] else indexed[(fid, arm)]
            pose = entry['actual_pose']
            measured[fid, arm] = measure(pose, refs[frame['partition']][fid])
        compare(measure(indexed[(fid, 'POSEFIX_RAW')]['actual_pose'], refs[frame['partition']][fid])['translation_cm'],
                measured[fid, 'POSEFIX_RAW']['translation_cm'], 'RAW independent new/cached F')
        assert indexed[(fid, 'POSEFIX_RAW')]['actual_pose'] == frame['predictions']['POSEFIX_RAW']['actual_pose']

    evaluation = private/'fair_evaluation_v1'
    results = read(evaluation/'FAIR_RESULTS.json')
    with (evaluation/'FAIR_ROW_METRICS.csv').open() as stream:
        flat = list(csv.DictReader(stream))
    assert len(flat) == 1428
    for row in flat:
        value = measured[row['id'], row['arm']]
        for key in ('translation_cm', 'rotation_deg'):
            compare(value[key], None if not row[key] else float(row[key]), key)
        compare(value['success'], row['success'] == 'True', 'joint same-pose success')
        compare(value['pose_available'], row['pose_available'] == 'True', 'availability')

    protocol = read(private/'EVALUATOR_PROTOCOL_LOCK_V2_TECHNICAL_ID_JOIN.json')
    populations = protocol['populations']
    contrasts, cis = 0, 0
    for name, partitions in populations.items():
        frames = [r for r in base_rows if r['partition'] in partitions]
        for arm in expected_arms:
            own = [measured[r['id'], arm] for r in frames]
            saved = results['populations'][name][arm]
            compare(len(own), saved['denominator'], name+' denominator')
            compare(sum(v['success'] for v in own), saved['successes'], name+' successes')
            compare(sum(not v['pose_available'] for v in own), saved['no_pose'], 'no pose stays denominator')
            for key in ('translation_cm', 'rotation_deg'):
                for statistic, value in distribution([v[key] for v in own]).items():
                    compare(value, saved[key][statistic], name+arm+statistic)
            for field, key, bound in [('T_gt50cm', 'translation_cm', 50), ('T_gt100cm', 'translation_cm', 100), ('R_gt45deg', 'rotation_deg', 45)]:
                compare(sum(v[key] is not None and v[key]>bound for v in own), saved[field], field)
        for contrast in results['paired'][name].values():
            a, b = contrast['before'], contrast['after']
            pairs = [(measured[r['id'], a], measured[r['id'], b]) for r in frames]
            labels = ['BOTH_SUCCESS' if x['success'] and y['success'] else 'RECOVERY' if y['success'] else 'DAMAGE' if x['success'] else 'BOTH_FAILURE' for x, y in pairs]
            for label in ('BOTH_SUCCESS', 'RECOVERY', 'DAMAGE', 'BOTH_FAILURE'):
                compare(labels.count(label), contrast['counts'][label], label)
                assert [r['id'] for r, event in zip(frames, labels) if event == label] == contrast['ids'][label]
            contrasts += 1
            for key in ('success_rate', 'translation_cm', 'rotation_deg'):
                values = [int(y['success'])-int(x['success']) if key == 'success_rate'
                    else None if x[key] is None or y[key] is None else y[key]-x[key] for x, y in pairs]
                present = [v for v in values if v is not None]
                saved = contrast['deltas'][key]
                compare(statistics.mean(present), saved['mean'], 'paired delta mean')
                for groupkey, field in [('cluster_id', 'recording_cluster_CI'), ('session', 'named_session_cluster_sensitivity_CI')]:
                    interval = cluster_ci(values, frames, groupkey, protocol['bootstrap']['seed'], protocol['bootstrap']['draws'])
                    expected = saved[field]['CI95']
                    if interval is None:
                        assert expected is None
                    else:
                        for actual, target in zip(interval, expected):
                            compare(actual, target, 'independent clustered CI')
                    cis += 1

    inputs = rows(private/'posefix_inputs_v1/POSEFIX_INPUTS238.jsonl.gz')
    maximum_cap_difference, cap_checks = 0., 0
    for frame in inputs:
        fid = frame['id']
        pf = np.asarray(frame['PoseFix_points'], float)
        yolo = np.asarray(frame['YOLO_points'], float)
        support = np.asarray(frame['support'], bool)
        maximum = .01*math.hypot(*frame['raw_hw'])
        for method, anchor in [('POSEFIX_SUBPIX_PF_ANCHOR', pf), ('POSEFIX_SUBPIX_YOLO_ANCHOR', yolo)]:
            record = indexed[fid, method]
            sub = np.asarray(record['native_SubPix_points'], float)
            expected = anchor.copy()
            for k in range(8):
                if support[k] and np.isfinite(anchor[k]).all() and np.isfinite(sub[k]).all():
                    delta = sub[k]-anchor[k]
                    length = math.hypot(*delta)
                    expected[k] = anchor[k]+delta*min(1., maximum/max(length, 1e-12))
            expected[8] = yolo[8]
            expected[~support] = yolo[~support]
            actual = np.asarray(record['native_points'], float)
            assert np.allclose(actual, expected, rtol=0, atol=1e-9, equal_nan=True), (fid, method, 'cap anchor mixing')
            maximum_cap_difference = max(maximum_cap_difference, float(np.nanmax(np.abs(actual-expected))))
            cap_checks += 1

    receipt = dict(schema='FAIR_POSEFIX_E_INDEPENDENT_ARITHMETIC_V1', passed=True,
        frame_count=238, PRIMARY192=192, CAL46=46, arm_rows=1428,
        independently_checked_population_arm_summaries=len(populations)*len(expected_arms),
        paired_contrasts=contrasts, clustered_CI_arrays=cis,
        independently_verified_anchor_point_arrays=cap_checks,
        maximum_metric_summary_CI_absolute_difference=largest, maximum_cap_absolute_difference_px=maximum_cap_difference,
        independent_C2_scalar_trace_and_stdlib_statistics=True, new_scientific_module_imports=0,
        references_read_after_prediction_seal_checks=True, new_NN=0, new_PnP_or_optimization=0,
        success_counts_PRIMARY192={arm: sum(measured[r['id'], arm]['success'] for r in base_rows if r['partition'] != 'CAL') for arm in expected_arms},
        prediction_seal_sha256=sha(private/'posefix192_v1/PREDICTION_SEAL.json'), fixed_reference_sha256=sha(baseline/'FIXED_REFERENCES.json'),
        evaluator_seal_sha256=sha(evaluation/'EVALUATION_SEAL.json'), checker_source_sha256=sha(__file__),
        elapsed_seconds=time.monotonic()-start, process_cpu_seconds=time.process_time()-cpu,
        maximum_resident_set_size_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with Path(output).open('x') as stream:
        json.dump(receipt, stream, indent=2)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('task-private', 'baseline-folder', 'output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.task_private, args.baseline_folder, args.output), indent=2))
