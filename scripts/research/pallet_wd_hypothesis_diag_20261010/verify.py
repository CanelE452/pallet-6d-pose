"""Independent saved Stage1 evidence audit; no producer, model or PnP imports."""
import argparse
import bisect
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(os.environ.get('PALLET_WD_OUTPUT', ROOT / '_docs/experiments/pallet_wd_hypothesis_diag_20261010'))
NUMERIC = {'T_cm': 'translation_cm', 'R_deg': 'rotation_deg', 'ADDsym_m': 'ADDsym_m', 'IoU3D': 'IoU3D'}
RATES = ('confusion_rate', 'success_rate')
METHODS = ('BASE', 'N3_DIM_SYM', 'SUBPIX', 'N3_THEN_SUBPIX')
POPULATIONS = {'REAL_DEV': 'REAL', 'SYNTH_HELDOUT': 'SYNTH', 'AUX': 'AUX'}
EDGES = {'elevation_deg': (5, 10, 20, 30), 'reference_margin_px': (2, 4, 8, 16), 'distance_m': (2, 3, 4, 6)}
LABELS = {'elevation_deg': ('<5', '5–10', '10–20', '20–30', '>=30'), 'reference_margin_px': ('<2', '2–4', '4–8', '8–16', '>=16'), 'distance_m': ('<2', '2–3', '3–4', '4–6', '>=6')}

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]

def quantile(values, probability):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

def describe(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return dict(n=0, mean=None, variance=None, std=None, median=None, P90=None, max=None)
    mean = math.fsum(finite) / len(finite)
    variance = math.fsum((value - mean) ** 2 for value in finite) / (len(finite) - 1) if len(finite) > 1 else None
    return dict(n=len(finite), mean=mean, variance=variance, std=math.sqrt(variance) if variance is not None else None,
                median=quantile(finite, .5), P90=quantile(finite, .9), max=max(finite))

def vectors(records, arm):
    poses = [row['pose'][arm] for row in records]
    result = {name: [float(pose[key]) if pose.get('available') and pose.get(key) is not None else float('nan') for pose in poses] for name, key in NUMERIC.items()}
    result['available'] = [bool(pose.get('available')) for pose in poses]
    result['confusion_rate'] = [float(pose.get('available') and pose['rotation_deg'] > 45 and abs(pose['yaw_deg']) >= 60) for pose in poses]
    result['success_rate'] = [float(pose.get('available') and pose['translation_cm'] < 5 and pose['rotation_deg'] < 5) for pose in poses]
    return result

def average(by_seed):
    result = {}
    for name in (*NUMERIC, *RATES):
        result[name] = []
        for values in zip(*(packet[name] for packet in by_seed)):
            finite = [value for value in values if math.isfinite(value)]
            result[name].append(math.fsum(finite) / len(finite) if finite else float('nan'))
    result['available'] = [all(values) for values in zip(*(packet['available'] for packet in by_seed))]
    return result

class Check:
    def __init__(self):
        self.comparisons = 0
        self.max_absolute = 0.
        self.max_relative = 0.

    def close(self, observed, expected, label='', absolute=1e-7):
        self.comparisons += 1
        if expected is None:
            assert observed is None, (label, observed, expected)
        elif isinstance(expected, (list, tuple)):
            assert len(observed) == len(expected), label
            for a, b in zip(observed, expected):
                self.close(a, b, label, absolute)
        else:
            assert observed is not None, label
            difference = abs(float(observed) - float(expected))
            self.max_absolute = max(self.max_absolute, difference)
            self.max_relative = max(self.max_relative, difference / max(1., abs(float(expected))))
            assert difference <= absolute + 1e-10 * abs(float(expected)), (label, observed, expected, difference)

class IndependentDraws:
    CACHE = {}

    def __init__(self, labels):
        names = sorted(set(labels))
        positions = {name: index for index, name in enumerate(names)}
        self.group = [positions[label] for label in labels]
        self.n = len(names)
        if self.n not in self.CACHE:
            rng = np.random.default_rng(20260917)
            draws = np.vstack([rng.multinomial(self.n, np.ones(self.n) / self.n, size=100) for _ in range(100)])
            digest = hashlib.sha256(draws.astype('<u2').tobytes()).hexdigest()
            self.CACHE[self.n] = (draws.astype(float), digest)
        self.weights, self.digest = self.CACHE[self.n]
        self.cache = {}

    def interval(self, values, selected=None):
        selected = list(range(len(self.group))) if selected is None else list(selected)
        grouped = defaultdict(list)
        for position, value in zip(selected, values):
            if math.isfinite(float(value)):
                grouped[self.group[position]].append(float(value))
        if not grouped:
            return None
        active = sorted(grouped)
        totals = [math.fsum(grouped[index]) for index in active]
        counts = [len(grouped[index]) for index in active]
        cache_key = (tuple(active), tuple(totals), tuple(counts))
        if cache_key in self.cache:
            return self.cache[cache_key]
        weights = self.weights[:, active]
        denominator = weights @ np.array(counts, float)
        numerator = weights @ np.array(totals, float)
        samples = (numerator[denominator > 0] / denominator[denominator > 0]).tolist()
        interval = [quantile(samples, .025), quantile(samples, .975)] if samples else None
        self.cache[cache_key] = interval
        return interval

def verify_summary(check, packet, vectors, draws, selected):
    size = len(selected)
    assert packet['frames'] == size
    assert packet['available_all_seeds'] == sum(vectors['available'])
    for key in NUMERIC:
        expected = describe(vectors[key])
        assert packet['numerical_available_frames'][key] == expected['n']
        for statistic, value in expected.items():
            check.close(packet['metrics'][key][statistic], value, key + '.' + statistic)
        check.close(packet['metrics'][key]['CI95'], draws.interval(vectors[key], selected), key + '.CI95')
    for key in RATES:
        expected = math.fsum(vectors[key]) / size
        check.close(packet['rates'][key]['rate'], expected, key)
        check.close(packet['rates'][key]['CI95'], draws.interval(vectors[key], selected), key + '.CI95')
        assert packet['rates'][key]['denominator'] == size and packet['rates'][key]['binary_before_seed_mean']

def auc(scores, labels):
    positive = [score for score, label in zip(scores, labels) if math.isfinite(score) and label]
    negative = sorted(score for score, label in zip(scores, labels) if math.isfinite(score) and not label)
    wins = [bisect.bisect_left(negative, score) + .5 * (bisect.bisect_right(negative, score) - bisect.bisect_left(negative, score)) for score in positive]
    return dict(AUC=math.fsum(wins) / (len(positive) * len(negative)) if positive and negative else None,
                positive=len(positive), negative=len(negative), n=len(positive) + len(negative))

def verify_statistics(check, records, published, population):
    assert published['population'] == population and published['phase'] == 'STAGE1_DIAGNOSTIC_ONLY'
    assert published['bin_edges'] == {key: list(edges) for key, edges in EDGES.items()}
    groups = defaultdict(list)
    for row in records:
        groups[row.get('backbone', 'YOLO') + '::' + row['method']].append(row)
    assert set(groups) == set(published['groups'])
    for name, group in sorted(groups.items()):
        ids = sorted({row['id'] for row in group})
        seeds = sorted({row['seed'] for row in group})
        assert seeds == [1, 2, 3]
        index = {(row['seed'], row['id']): row for row in group}
        assert len(index) == len(group) == len(ids) * 3
        ordered = [[index[(seed, fid)] for fid in ids] for seed in seeds]
        base = ordered[0]
        draws = IndependentDraws(ids if population == 'SYNTH_HELDOUT' else [row['session'] for row in base])
        packet = published['groups'][name]
        assert packet['frames'] == len(ids) and packet['rows'] == len(group) and packet['seeds'] == seeds
        metadata = packet['bootstrap']
        assert metadata['units'] == draws.n and metadata['master_frames'] == metadata['frames'] == len(ids)
        assert metadata['resamples'] == 10000 and metadata['seed'] == 20260917 and metadata['draws_sha256_uint16_le'] == draws.digest
        assert metadata['level'] == ('frame' if population == 'SYNTH_HELDOUT' else 'cluster')
        if population != 'SYNTH_HELDOUT':
            assert draws.n == 13 and draws.digest == '63e288a51d7b0612616beefac28fcc625e76e8c5d7b5a0ecc5e8b85c73048fa5'
        arms = {arm: [vectors(seedrows, arm) for seedrows in ordered] for arm in ('S0', 'ORACLE')}
        means = {arm: average(values) for arm, values in arms.items()}
        full = list(range(len(ids)))
        for arm in arms:
            verify_summary(check, packet['seed_mean'][arm], means[arm], draws, full)
            for position, seed in enumerate(seeds):
                verify_summary(check, packet['per_seed'][str(seed)][arm], arms[arm][position], draws, full)
        differences = [a - b for a, b in zip(means['S0']['R_deg'], means['ORACLE']['R_deg'])]
        for statistic, value in describe(differences).items():
            check.close(packet['R_mean_difference_F_minus_oracle'][statistic], value, 'oracle difference')
        check.close(packet['R_mean_difference_F_minus_oracle']['CI95'], draws.interval(differences), 'oracle difference CI')
        assert set(packet['bins']) == {*EDGES, 'material', 'grade'}
        for field, bins in packet['bins'].items():
            if field in EDGES:
                categories = ['UNKNOWN' if row.get(field) is None or not math.isfinite(row[field]) else LABELS[field][bisect.bisect_right(EDGES[field], row[field])] for row in base]
                assert set(bins) == {*LABELS[field], 'UNKNOWN'}
            else:
                categories = [str(row.get(field) or 'UNKNOWN') for row in base]
                assert set(bins) == set(categories)
            for category, result in bins.items():
                selected = [index for index, label in enumerate(categories) if label == category]
                assert result['frames'] == len(selected)
                if not selected:
                    assert result['seed_mean'] is None
                    continue
                for arm in arms:
                    subset = {key: [values[index] for index in selected] for key, values in means[arm].items()}
                    verify_summary(check, result['seed_mean'][arm], subset, draws, selected)
        scores, labels, auc_values = [], [], []
        for position, seedrows in enumerate(ordered):
            seed_scores = [-row['formal_score_gap_px'] if row.get('formal_score_gap_px') is not None else float('nan') for row in seedrows]
            seed_labels = [bool(value) for value in arms['S0'][position]['confusion_rate']]
            result = auc(seed_scores, seed_labels)
            target = packet['score_gap_auc']['per_seed'][position]
            assert target['seed'] == seeds[position]
            for key, value in result.items():
                check.close(target[key], value, 'seed AUC')
            if result['AUC'] is not None:
                auc_values.append(result['AUC'])
            scores.extend(seed_scores)
            labels.extend(seed_labels)
        for key, value in auc(scores, labels).items():
            check.close(packet['score_gap_auc']['pooled_descriptive'][key], value, 'pooled AUC')
        check.close(packet['score_gap_auc']['macro_seed_mean'], math.fsum(auc_values) / len(auc_values) if auc_values else None, 'macro AUC')
        for threshold in (1, 2, 3, 5):
            warning = [math.isfinite(score) and -score < threshold for score in scores]
            caught = sum(a and b for a, b in zip(warning, labels))
            expected = dict(threshold_px=threshold, warning_rate=sum(warning) / len(warning),
                            caught_confusion_rate=caught / sum(labels) if any(labels) else None,
                            warning_precision=caught / sum(warning) if any(warning) else None,
                            valid_score_predictions=sum(math.isfinite(score) for score in scores), predictions=len(scores), warnings=sum(warning), caught=caught)
            target = packet['warnings'][str(threshold)]
            assert target['operator'] == '<'
            for key, value in expected.items():
                check.close(target[key], value, 'warning.' + key)
        print('INDEPENDENT_STAGE1_GROUP', population, name, 'PASS', flush=True)

def cuboid(xyz):
    width, height, depth = xyz
    return np.array([[-width,-height,-depth],[width,-height,-depth],[width,height,-depth],[-width,height,-depth],
                     [-width,-height,depth],[width,-height,depth],[width,height,depth],[-width,height,depth]], float) / 2

def rotations(order):
    return [np.array([[math.cos(index * 2 * math.pi / order), 0, math.sin(index * 2 * math.pi / order)], [0,1,0],
                      [-math.sin(index * 2 * math.pi / order),0,math.cos(index * 2 * math.pi / order)]]) for index in range(order)]

def verify_geometry(check, record, truth):
    if truth is None:
        return 0
    G, target_t, xyz, oracle = truth
    assert record['hypOracle'] == oracle and record['oracle_diagnostic_only'] and record['inference_reference_inputs'] is False
    count = 0
    for arm in ('S0', 'ORACLE'):
        pose = record['pose'][arm]
        actual = record['actual_pose'][arm]
        assert pose['available'] == actual['available']
        if not actual['available']:
            continue
        count += 1
        R = np.array(actual['R_physical']);t = np.array(actual['centroid'])
        check.close(pose['translation_cm'], math.sqrt(math.fsum(float(value) ** 2 for value in t - target_t)) * 100, 'geometry T')
        angles, yaws, adds = [], [], []
        X = cuboid(xyz)
        for symmetry in rotations(record['pose_symmetry_order']):
            target = G @ symmetry
            relative = target.T @ R
            cosine = (math.fsum(float(relative[index,index]) for index in range(3)) - 1) / 2
            angles.append(math.degrees(math.acos(min(1., max(-1., cosine)))))
            yaws.append(abs((math.degrees(math.atan2(relative[0,2], relative[2,2])) + 180) % 360 - 180))
            difference = X @ R.T + t - (X @ target.T + target_t)
            adds.append(math.fsum(math.sqrt(math.fsum(float(value) ** 2 for value in point)) for point in difference) / 8)
        check.close(pose['rotation_deg'], min(angles), 'geometry R', absolute=2e-5)
        check.close(pose['yaw_deg'], min(yaws), 'geometry yaw')
        check.close(pose['ADDsym_m'], min(adds), 'geometry ADD')
        check.close(pose['ADDsym_normalized'], min(adds) / math.sqrt(math.fsum(float(value) ** 2 for value in xyz)), 'geometry normalized ADD')
    return count

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doc', type=Path, default=DOC)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args();doc = args.doc;output = args.output or doc / 'STAGE1_VERIFICATION.json'
    assert not output.exists(), 'Use a fresh verification output'
    required = ['STAGE1_METRICS.json', 'STAGE1_EXECUTION.json', 'STAGE1_SELECTION_SEAL.json'] + [f'STAGE1_ROWS_{population}.jsonl.gz' for population in POPULATIONS.values()]
    assert all((doc / name).is_file() for name in required), 'Stage1 READY evidence is not complete'
    check = Check();audit = read(doc / 'INPUT_AUDIT.json');execution = read(doc / 'STAGE1_EXECUTION.json');allseal = read(doc / 'STAGE1_SELECTION_SEAL.json')
    assert audit['status'] == 'PASS' and audit['reference_pose_values_consumed_for_selection'] is False
    assert execution['status'] == 'COMPLETE_STAGE1_DIAGNOSTIC_ONLY_NO_VERDICT'
    for key in ('stage2_rules_executed','network_inference','training_updates'):
        assert execution[key] == 0
    assert allseal['stage0_parity_sha256'] == sha(doc / 'STAGE0_PARITY.json')
    final_path = doc / execution['source_lock_path'];lock = read(final_path)
    assert sha(final_path) == execution['source_lock_sha256'] == allseal['source_lock_sha256']
    assert sha(doc / 'STAGE1_SELECTION_SEAL.json') == execution['selection_seal_sha256']
    assert lock['solver_calls_before_amendment'] == 0 and sha(doc / lock['previous_lock_path']) == lock['previous_lock_sha256']
    recovery = None
    if execution.get('postseal_amendment_sha256'):
        amendment_path = doc / 'STAGE1_POSTSEAL_AMENDMENT.json'
        assert sha(amendment_path) == execution['postseal_amendment_sha256'] == allseal['postseal_amendment_sha256']
        recovery = read(amendment_path)
        assert recovery['previous_lock_sha256'] == sha(final_path)
        assert recovery['original_frozen_core'] == lock['new_core']
        assert sha(ROOT / recovery['source']['path']) == recovery['source']['sha256']
        assert recovery['repeated_original_F_calls'] == execution['repeated_original_F_calls'] == 0
        for binding in recovery['preserved_selection_bindings'].values():
            assert sha(ROOT / binding['path']) == binding['sha256']
    source_checks = 0
    for binding in lock['new_core'] + lock['original_core']:
        root = ROOT if binding['owner'] == 'published_worktree' else args.source_root or ROOT
        assert sha(root / binding['path']) == binding['sha256'], binding['path']
        source_checks += 1
    input_checks = 0
    for binding in audit['bindings']:
        root = ROOT if binding['owner'] == 'published_worktree' else args.source_root
        if root is not None:
            assert sha(root / binding['path']) == binding['sha256'], binding['path']
            input_checks += 1
    geometry = None
    if args.source_root is not None:
        path = args.source_root / 'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
        binding = next(binding for binding in audit['bindings'] if binding['path'].endswith('GEOMETRY_SIDETABLE.npz'))
        assert sha(path) == binding['sha256']
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in ('stems', 'Xcf', 'R', 't', 'dims')}
        stems = {str(stem): index for index, stem in enumerate(data['stems'])}
        ids = {row['id'] for row in rows(doc / 'STAGE1_SELECTIONS_SYNTH.jsonl.gz')}
        geometry = {}
        for fid in ids:
            index = stems[fid];X = data['Xcf'][index]
            oracle = 'long-face-front' if np.linalg.norm(X[1]-X[0]) > np.linalg.norm(X[4]-X[0]) else 'short-face-front'
            geometry[fid] = (data['R'][index], data['t'][index], data['dims'][index], oracle)
    metrics = read(doc / 'STAGE1_METRICS.json');assert set(metrics) == set(POPULATIONS)
    counts = {};geometry_counts = {};records_by_population = {}
    for population, suffix in POPULATIONS.items():
        saved = rows(doc / f'STAGE1_ROWS_{suffix}.jsonl.gz');selections = rows(doc / f'STAGE1_SELECTIONS_{suffix}.jsonl.gz')
        seal_path = doc / f'STAGE1_SELECTION_SEAL_{suffix}.json';seal = read(seal_path)
        assert seal['rows'] == len(saved) == len(selections) == audit['populations'][suffix]['rows']
        assert seal['selection_sha256'] == sha(doc / seal['selection_path']) == allseal['populations'][suffix]['selection_sha256']
        assert seal['source_lock_sha256'] == sha(final_path) and seal['input_audit_sha256'] == sha(doc / 'INPUT_AUDIT.json')
        assert allseal['populations'][suffix]['seal_sha256'] == sha(seal_path)
        assert seal['method_reference_access'] is False and seal['stage2_rules_executed'] == seal['model_forwards'] == seal['training_updates'] == 0
        assert execution['actual_original_F_calls'][suffix] == len(saved)
        assert execution['outputs'][suffix]['sha256'] == sha(doc / f'STAGE1_ROWS_{suffix}.jsonl.gz')
        index = {(row['backbone'],row['seed'],row['method'],row['id']):row for row in saved}
        assert len(index) == len(saved) and len({row['id'] for row in saved}) == audit['populations'][suffix]['frames']
        count_routes = Counter(f"{row['backbone']}/{row['method']}/seed{row['seed']}" for row in saved)
        assert dict(count_routes) == audit['populations'][suffix]['count_by_route']
        geometric = 0
        for original in selections:
            row = index[(original['backbone'],original['seed'],original['method'],original['id'])]
            for key in original:
                if key != 'selection':
                    assert row[key] == original[key], ('Frozen selected coordinate/metadata changed', key)
            selected = original['selection']
            assert selected['reference_inputs'] is False and selected['rule'] == 'S0' and selected['original_F_calls'] == 1
            assert row['actual_pose']['S0'] == selected['actual_pose'] and row['hypS0'] == selected['hyp']
            assert row['candidates'] == selected['candidates'] and row['formal_score_gap_px'] == selected['formal_score_gap_px']
            assert row['reference_seal_sha256'] == sha(seal_path)
            assert row['actual_pose']['ORACLE'] == selected['candidates'][row['hypOracle']]['actual_pose']
            if selected['hyp'] is not None:
                alternative = next(name for name in selected['candidates'] if name != selected['hyp'])
                score = selected['candidates'][selected['hyp']]['formal_score'];other = selected['candidates'][alternative]['formal_score']
                expected_gap = other - score if score is not None and other is not None else None
                check.close(selected['formal_score_gap_px'], expected_gap, 'formal gap')
            truth = None
            if suffix != 'SYNTH':
                Gcf = np.array(row['reference_R_cf']);cf = np.array(row['reference_cf_extents']);xyz = np.array(row['fixed_metadata']['dimensions_pnp_WH_D_m'])
                target_t = np.array(row['reference_bottom_center']) - Gcf @ np.array([0, cf[1]/2, 0])
                Q = np.eye(3) if abs(cf[0] - xyz[0]) < 1e-6 else rotations(4)[1]
                oracle = 'long-face-front' if row['gt_physical_long_axis'] == 'CF_WIDTH' else 'short-face-front'
                truth = Gcf @ Q, target_t, xyz, oracle
            elif geometry is not None:
                truth = geometry[row['id']]
            geometric += verify_geometry(check, row, truth)
        counts[suffix] = len(saved);geometry_counts[suffix] = geometric;records_by_population[suffix] = saved
        verify_statistics(check, saved, metrics[population], population)
    assert execution['actual_original_F_calls_total'] == sum(counts.values())
    parity = read(doc / 'STAGE0_PARITY.json');assert parity['status'] == 'PASS' and parity['rows'] == 3828 and not parity['failures']
    assert parity['source_lock_sha256'] == sha(final_path) and parity['selection_seal_sha256'] == sha(doc / 'STAGE1_SELECTION_SEAL_REAL.json')
    old = {(row['id'],row['method'],row['seed']):row for row in rows(ROOT / '_docs/experiments/pallet_feature_gradient_joint_20261010/PREDICTIONS.jsonl.gz') if row['method'] in METHODS}
    per_row = {(row['id'],row['method'],row['seed']):row for row in parity['per_row']}
    for row in records_by_population['REAL']:
        key = row['id'],row['method'],row['seed'];previous = old[key];receipt = per_row[key]
        assert row['qFinal'] == previous['qFinal'] and row['hypS0'] == previous['final_hypothesis'] and row['pose']['S0']['available'] == previous['pose']['available']
        assert receipt['PASS'] and receipt['hypothesis_exact'] and receipt['availability_exact']
        for field, name in (('translation_cm','delta_translation_cm'), ('rotation_deg','delta_rotation_deg')):
            difference = abs(row['pose']['S0'][field]-previous['pose'][field]) if row['pose']['S0']['available'] else None
            check.close(receipt[name], difference, 'Stage0 parity')
            assert difference is None or difference <= .01
    for path in [*doc.rglob('*'), *(ROOT / 'scripts/research/pallet_wd_hypothesis_diag_20261010').rglob('*')]:
        if not path.is_file() or path.suffix == '.png':
            continue
        assert path.suffix.lower() not in ('.jpg','.jpeg','.pt','.pth','.npy','.npz','.pyc')
        text = gzip.open(path,'rt').read() if path.suffix == '.gz' else path.read_text()
        assert ('/' + 'home' + '/') not in text
    result = dict(status='PASS', phase='STAGE1_DIAGNOSTIC_ONLY_NO_VERDICT', independent_numeric_comparisons=check.comparisons,
                  max_absolute_difference=check.max_absolute, max_scaled_relative_difference=check.max_relative,
                  population_rows=counts, independent_geometry_pose_count=geometry_counts,
                  SYNTH_geometry_checked=geometry is not None, independent_T_R_yaw_ADD=True,
                  IoU3D_statistics_verified=True, IoU3D_geometry_independently_recomputed=False,
                  original_source_hashes_checked=source_checks, input_binding_hashes_checked=input_checks,
                  frozen_selection_and_coordinates_exact=True, Stage0_original_parity='PASS_3828',
                  bootstrap='independent master multinomial100x100 draws; grouped math.fsum, sorted linear quantiles',
                  AUC='independent positive-vs-negative win/tie count using bisection',
                  additional_solver_calls=0, additional_model_calls=0, new_training_updates=0,
                  scoring_recovery_amendment_checked=recovery is not None,
                  interrupted_reference_margin_call_count='NOT_MEASURED; disclosed and not claimed as zero' if recovery is not None else None,
                  stage2_rules_verified_as_unexecuted=True, verdict='NO_VERDICT_STAGE1', source_sha256=sha(__file__))
    with output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('INDEPENDENT_STAGE1_PASS',check.comparisons,check.max_absolute,flush=True)

if __name__ == '__main__':
    main()
