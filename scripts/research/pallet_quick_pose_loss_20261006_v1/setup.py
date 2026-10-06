"""One read-only cache validation, TRAIN scale and soft-target calibration."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import time
import numpy as np
import torch
from .common import (ROOT, DOC, HARD_DOC, OLD_DOC, BASELINE_ROOT, Data,
                     read, write, sha, hash_value, initialize, numeric_contract)
from .losses import derive_targets, analytic_gradient_checks
from scripts.research.pallet_pose_target_6d_20261006_v1.training import (
    ReadOnlyBanks, inference_support_without_features, teacher_2d)
from scripts.research.pallet_joint_action_handoff_20261006_v1.a_common import finite, local_phase

HARD_COMMIT = 'e8e219ab111bc121235c49078459c679d6dc656e'
CALIBRATION_SIZE = 1024


def artifact(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))


def summary(values):
    values = np.asarray(values, dtype=float)
    return dict(count=len(values), mean=float(values.mean()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                P90=float(np.quantile(values, .9)) if len(values) else None)


def normalized_entropy(probability, action_count):
    probability = np.asarray(probability, dtype=np.float64)
    return -(probability * np.log(np.maximum(probability, 1e-300))).sum(-1) / np.log(action_count)


def solve_temperature(cost, available, action_valid, scale, target_entropy):
    """At most 48 TRAIN-only log-space bisections, with explicit endpoints."""
    count = action_valid.sum(-1)
    assert (count > 1).all()
    def evaluate(tau):
        probability, _, eligible = derive_targets(cost, available, action_valid, scale, tau)
        assert eligible.all()
        return float(normalized_entropy(probability, count).mean())
    low, high = np.log(1e-6), np.log(1e6)
    low_entropy, high_entropy = evaluate(1e-6), evaluate(1e6)
    iterations = 0
    if target_entropy <= low_entropy:
        tau, achieved = 1e-6, low_entropy
        endpoint = 'LOWER_ENDPOINT'
    elif target_entropy >= high_entropy:
        tau, achieved = 1e6, high_entropy
        endpoint = 'UPPER_ENDPOINT'
    else:
        endpoint = None
        for iterations in range(1, 49):
            middle = (low + high) / 2
            tau = float(np.exp(middle)); achieved = evaluate(tau)
            if abs(achieved - target_entropy) <= 1e-4:
                break
            if achieved < target_entropy: low = middle
            else: high = middle
    return dict(tau=tau, target_normalized_entropy=target_entropy,
                achieved_normalized_entropy=achieved, mismatch=achieved - target_entropy,
                endpoint=endpoint, endpoint_entropy=[low_entropy, high_entropy],
                bisection_iterations=iterations, maximum_iterations=48,
                log_interval=[float(np.log(1e-6)), float(np.log(1e6))])


@torch.no_grad()
def soft2d_probability(output, batch):
    """Original local phase, FP32 error, sigma and softmax, with probabilities."""
    gt, valid, branch, _ = local_phase(output['points_raw'], output['point_valid'],
        batch['gt_points'], batch['gt_valid'], batch['permutations'],
        batch['group_valid'], output['box_diagonal'])
    mask = finite(gt, valid)[:, :8] & output['point_support']
    count = mask.sum(-1)
    error = (output['candidate_points'][:, :, :8] - gt[:, None, :8]).square().sum(-1)
    error = torch.where(mask[:, None], error, torch.zeros_like(error)).sum(-1) / count.clamp_min(1)[:, None]
    sigma = output['box_diagonal'] * .08 / 17
    probability = (-error / (2 * sigma[:, None].square())).masked_fill(~output['action_valid'], float('-inf')).softmax(-1)
    index = probability.argmax(-1)
    index = torch.where(count > 0, index, index.new_full(index.shape, -1))
    return probability, index, count, branch


def _validate(source_root, bank_cache, cost_cache):
    """Hash each registered cost/bank once; reuse audited large features honestly."""
    source_root, bank_cache, cost_cache = map(lambda p: Path(p).resolve(), (source_root, bank_cache, cost_cache))
    manifest = read(HARD_DOC / 'POSE_COST_CACHE_MANIFEST.json')
    header = read(cost_cache / 'cache_header.json')
    protocol = read(HARD_DOC / 'PROTOCOL.json')
    assert manifest['status'] == 'PASS' and manifest['completed_rows'] == 55915 and manifest['pending_rows'] == 0
    assert manifest['incomplete_attempted_F'] == manifest['incomplete_attempt_rows'] == 0
    assert manifest['candidate_F_completed'] == 11238915
    assert manifest['binding'] == header['binding'] and manifest['header_sha256'] == sha(cost_cache / 'cache_header.json')
    assert source_root == Path(protocol['source_root']).resolve() == Path(header['source_root']).resolve()
    assert bank_cache == Path(protocol['candidate_bank_cache']).resolve() == Path(header['bank_cache']).resolve()
    assert cost_cache == Path(protocol['pose_cost_cache']).resolve() == Path(manifest['cache_path']).resolve()
    assert header['protocol_sha256'] == sha(HARD_DOC / 'PROTOCOL.json')
    assert header['input_bindings_sha256'] == sha(HARD_DOC / 'INPUT_BINDINGS.json')
    relative_files = ['scripts/research/pallet_pose_target_6d_20261006_v1/' + n for n in
                      ('baseline.py', 'training.py', 'evaluation.py', 'reporting.py', 'cost_cache.py', 'geometry_parity.py')]
    relative_files += ['_docs/experiments/pallet_pose_target_6d_20261006_v1/' + n for n in
                       ('PROTOCOL.json', 'POSE_COST_CACHE_MANIFEST.json', 'INPUT_BINDINGS.json')]
    code_evidence = []
    for relative in relative_files:
        expected = subprocess.check_output(['git', '-C', str(ROOT), 'show', HARD_COMMIT + ':' + relative])
        assert (ROOT / relative).read_bytes() == expected, 'e8 baseline bytes changed: ' + relative
        code_evidence.append(artifact(ROOT / relative))
    assert header['cost_cache_code_sha256'] == sha(ROOT / relative_files[4])
    cost_artifacts = []
    for entry in manifest['files']:
        path = cost_cache / entry['file']
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], path
        cost_artifacts.append(dict(path=str(path), **entry))
    arrays = {name: np.load(cost_cache / entry['file'], mmap_mode='r') for name, entry in header['files'].items()}
    for name, value in arrays.items():
        assert list(value.shape) == header['files'][name]['shape'] and value.dtype == np.dtype(header['files'][name]['dtype'])
    data = Data(source_root)
    rows = np.load(cost_cache / 'train_rows.npy', mmap_mode='r')
    assert len(rows) == 55915 and np.array_equal(rows, data.train_rows)
    assert np.array_equal(rows, np.flatnonzero(arrays['usable_train']))
    ids = [data.source['records'][data.indices[row]]['id'] for row in rows]
    assert len(set(ids)) == 55915
    assert ids == read(HARD_DOC / 'ID_MANIFEST.json')['IDs']['train']
    assert hash_value(ids) == header['train_ids_sha256']
    assert arrays['done'][rows].all() and (arrays['row_state'][rows] == 2).all()
    assert (arrays['action_counts'][rows] == 201).all()
    bank = ReadOnlyBanks(data, bank_cache)
    assert bank.bank_sha256 == header['bank_sha256'] and bank.binding == header['bank_binding']
    assert np.array_equal(bank.counts[rows], arrays['action_counts'][rows])
    order_path = source_root / 'data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/order_seed1.npy'
    order = np.load(order_path, mmap_mode='r')
    old_fit = read(OLD_DOC / 'A_fits/GEO_seed1.json')
    assert order.shape == (6000, 16) and np.isin(order, rows).all() and sha(order_path) == old_fit['order_sha256']
    previous_audit = read(HARD_DOC / 'audit/INDEPENDENT_VERIFICATION.json')
    assert previous_audit['status'] == 'PASS' and previous_audit['full_source_input_SHA_end'] == 'PASS'
    inputs = read(HARD_DOC / 'INPUT_BINDINGS.json')
    feature_states = []
    for entry in inputs['external_inputs']:
        path = Path(entry['path']); stat = path.stat()
        assert stat.st_size == entry['bytes'] and stat.st_mtime_ns == entry['mtime_ns'], 'Audited input state changed: ' + str(path)
        if path.name in ('p3.npy', 'p4.npy'):
            feature_states.append(dict(path=str(path), sha256=entry['sha256'], bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, verification='reused prior completed full-byte SHA plus current unchanged size/mtime; no new content hash'))
    auxiliary = read(HARD_DOC / 'TRAIN_2D_6D_TARGET_COMPARISON.json')
    assert auxiliary['status'] == 'PASS' and auxiliary['full_usable_TRAIN'] == 55915
    auxiliary_artifacts = []
    for entry in auxiliary['array_artifacts']:
        path = cost_cache / Path(entry['path']).name
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
        auxiliary_artifacts.append(dict(entry, path=str(path)))
    aux = {name: np.load(cost_cache / (name + '.npy'), mmap_mode='r') for name in
           ('teacher_2d_index', 'teacher_2d_count', 'teacher_2d_support_count', 'teacher_2d_done')}
    assert aux['teacher_2d_done'][rows].all()
    evidence = dict(status='PASS', scope='one fresh cost/bank/aux/code SHA validation; audited large feature content hashes reused with current unchanged file state',
        cache_binding=header['binding'], bank_sha256=bank.bank_sha256, bank_binding=bank.binding,
        header=artifact(cost_cache / 'cache_header.json'), cost_manifest=artifact(HARD_DOC / 'POSE_COST_CACHE_MANIFEST.json'),
        cost_artifacts=cost_artifacts, auxiliary_artifacts=auxiliary_artifacts, inherited_code=code_evidence,
        features=feature_states, previous_full_input_audit=artifact(HARD_DOC / 'audit/INDEPENDENT_VERIFICATION.json'),
        input_bindings=artifact(HARD_DOC / 'INPUT_BINDINGS.json'), checked_prior_input_states=len(inputs['external_inputs']),
        order=artifact(order_path), source_ROOT=str(source_root), bank_cache=str(bank_cache), cost_cache=str(cost_cache),
        final_F_code=artifact(BASELINE_ROOT / 'scripts/research/pallet_dim_conditioned_p_v1/pose.py'),
        scorer_code=artifact(BASELINE_ROOT / 'scripts/research/pallet_joint_action_handoff_20261006_v1/scorer.py'))
    return data, bank, arrays, aux, rows, ids, protocol['config'], evidence


def prepare(source_root, bank_cache, cost_cache, output_cache):
    started = time.monotonic()
    output_cache = Path(output_cache).resolve()
    assert output_cache != Path(cost_cache).resolve() and output_cache != Path(bank_cache).resolve()
    binding = hash_value(dict(source_root=str(Path(source_root).resolve()), bank_cache=str(Path(bank_cache).resolve()),
        cost_cache=str(Path(cost_cache).resolve()), output_cache=str(output_cache), setup_sha256=sha(__file__),
        losses_sha256=sha(Path(__file__).with_name('losses.py')), hard_manifest_sha256=sha(HARD_DOC / 'POSE_COST_CACHE_MANIFEST.json')))
    out = DOC / 'LOSS_SETUP.json'
    if out.exists():
        prior = read(out)
        assert prior['binding'] == binding and prior['status'] == 'PASS'
        for entry in prior['artifacts']:
            path = Path(entry['path'])
            assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
        return prior
    tests = analytic_gradient_checks()
    data, bank, arrays, aux, rows, ids, config, evidence = _validate(source_root, bank_cache, cost_cache)
    numeric_contract()
    assert torch.cuda.is_available(), 'Pure FP32 CUDA target calibration requires original CUDA arithmetic'
    output_cache.mkdir(parents=True, exist_ok=True)
    guard = output_cache / 'SETUP_BINDING.json'
    if guard.exists(): assert read(guard)['binding'] == binding
    else: write(guard, dict(binding=binding, scope='pure saved-cost/label/support target arithmetic; no NN/F/optimizer'))
    calibration_order = sorted(range(len(rows)), key=lambda i: (hashlib.sha256(('pose_loss_scale_v1:' + ids[i]).encode()).digest(), ids[i]))[:CALIBRATION_SIZE]
    calibration_rows = np.array(rows[calibration_order], dtype=np.int64)
    calibration_ids = [ids[i] for i in calibration_order]
    minimum, spreads, failures, nonop_checks, center_checks = [], [], 0, 0, 0
    for start in range(0, len(rows), 1024):
        selected = rows[start:start + 1024]
        cost = np.array(arrays['cost_ADDsym_m'][selected]); success = np.array(arrays['F_available'][selected])
        assert np.array_equal(success, np.isfinite(cost)) and np.isposinf(cost[~success]).all()
        assert (arrays['candidate_state'][selected] == 2).all()
        count = success.sum(-1); eligible = count > 0
        b = np.where(success, cost, np.inf).min(-1)
        m = np.where(success, cost - np.where(eligible, b, 0.)[:, None], 0.).sum(-1) / np.maximum(count, 1)
        spreads.extend(m[(m > 1e-7) & eligible].tolist()); minimum.extend(b.tolist()); failures += int((~success).sum())
        oracle = np.where(eligible, cost.argmin(-1), -1)
        assert np.array_equal(oracle, arrays['oracle_index'][selected])
        # Source-frame affine/no-op inversion is inherited; no pose is solved.
        for row in selected:
            frame = data.source_frame(int(row))
            assert np.array_equal(bank.points[row, 0], frame['q'], equal_nan=True)
            assert np.array_equal(bank.points[row, :, 8], np.broadcast_to(frame['q'][8], (201, 2)), equal_nan=True)
            nonop_checks += 1; center_checks += 201
    assert spreads, 'BLOCKED_DATA: no identifiable positive TRAIN row cost spread'
    scale = float(np.median(spreads)); assert scale > 0 and np.isfinite(scale)
    head = initialize(1, config, 'cuda')
    probabilities, supports, counts, branches = [], [], [], []
    for start in range(0, len(calibration_rows), 16):
        selected = calibration_rows[start:start + 16]
        batch = {key: torch.from_numpy(np.array(data.arrays[key][selected], copy=True)).cuda()
                 for key in ('points', 'boxes', 'point_valid', 'input_shape', 'gt_points', 'gt_valid')}
        for key in ('permutations', 'group_valid'):
            batch[key] = torch.from_numpy(data.side[key][selected].copy()).cuda()
        candidates, valid = bank.tensor_batch(selected, batch)
        support, diag, pv = inference_support_without_features(head, batch, candidates, valid)
        output = dict(points_raw=batch['points'], point_valid=pv, point_support=support,
                      box_diagonal=diag, candidate_points=candidates, action_valid=valid)
        probability, index, count, branch = soft2d_probability(output, batch)
        original_index, original_count, original_branch = teacher_2d(output, batch)
        assert torch.equal(index, original_index) and torch.equal(count, original_count) and torch.equal(branch, original_branch)
        assert np.array_equal(index.cpu().numpy(), aux['teacher_2d_index'][selected])
        assert np.array_equal(count.cpu().numpy(), aux['teacher_2d_count'][selected])
        assert np.array_equal(support.sum(-1).cpu().numpy(), aux['teacher_2d_support_count'][selected])
        probabilities.append(probability.cpu().numpy()); supports.append(support.cpu().numpy())
        counts.append(count.cpu().numpy()); branches.append(branch.cpu().numpy())
    del head
    probability_2d = np.concatenate(probabilities); support_cal = np.concatenate(supports)
    count_2d = np.concatenate(counts); branch = np.concatenate(branches)
    actions = np.array(arrays['action_counts'][calibration_rows]); use = (actions > 1) & (count_2d > 0)
    assert use.any(), 'No defined multi-action TRAIN calibration target'
    entropy_2d = normalized_entropy(probability_2d[use], actions[use])
    cal_cost = np.array(arrays['cost_ADDsym_m'][calibration_rows]); cal_available = np.array(arrays['F_available'][calibration_rows])
    cal_valid = np.arange(201)[None] < actions[:, None]
    temperature = solve_temperature(cal_cost[use], cal_available[use], cal_valid[use], scale, float(entropy_2d.mean()))
    tau = temperature['tau']
    paths = {name: str(output_cache / (name + '.npy')) for name in
             ('soft_targets', 'r_effective', 'eligible', 'train_rows', 'calibration_rows',
              'calibration_probabilities_2d', 'calibration_support', 'calibration_counts', 'calibration_branches')}
    soft = np.lib.format.open_memmap(paths['soft_targets'], mode='w+', dtype='float32', shape=(len(data.indices), 201)); soft[:] = 0
    effective = np.lib.format.open_memmap(paths['r_effective'], mode='w+', dtype='float32', shape=(len(data.indices), 201)); effective[:] = 0
    eligible_array = np.lib.format.open_memmap(paths['eligible'], mode='w+', dtype='bool', shape=(len(data.indices),)); eligible_array[:] = False
    unsupported = []; target_peaks = []; target_effective = []; eligible_count = 0
    for start in range(0, len(rows), 1024):
        selected = rows[start:start + 1024]
        cost = np.array(arrays['cost_ADDsym_m'][selected]); available = np.array(arrays['F_available'][selected])
        action_valid = np.arange(201)[None] < arrays['action_counts'][selected, None]
        y, r, eligible = derive_targets(cost, available, action_valid, scale, tau)
        assert np.isfinite(y).all() and np.isfinite(r).all()
        assert np.allclose(y[eligible].sum(-1), 1, rtol=0, atol=2e-6)
        inaccessible = eligible & (aux['teacher_2d_support_count'][selected] == 0) & (y[:, 1:].sum(-1) > 0)
        unsupported.extend(selected[inaccessible].tolist())
        soft[selected] = y; effective[selected] = r; eligible_array[selected] = eligible
        target_peaks.extend(y[eligible].max(-1).tolist())
        y64 = y[eligible].astype(np.float64)
        h = -(y64 * np.log(np.maximum(y64, 1e-300))).sum(-1)
        target_effective.extend(np.exp(h).tolist()); eligible_count += int(eligible.sum())
    assert not unsupported, 'BLOCKED_INTEGRITY: positive non-NoOp SOFT6D target mass with support0'
    assert eligible_count == int((arrays['oracle_index'][rows] >= 0).sum()) == 55915
    for value in (soft, effective, eligible_array): value.flush()
    for name, value in [('train_rows', rows), ('calibration_rows', calibration_rows),
                        ('calibration_probabilities_2d', probability_2d), ('calibration_support', support_cal),
                        ('calibration_counts', count_2d), ('calibration_branches', branch)]:
        np.save(paths[name], value)
    result = dict(schema='quick_pose_loss_setup_v1', status='PASS', binding=binding, paths=paths,
        hard_result_commit=HARD_COMMIT, baseline_commit='a22fb14beb5e8df08076385000e0d53503c1ae29',
        scale_m=scale, scale_rule='median mean finite-candidate cost-minus-row-min for TRAIN means >1e-7m',
        scale_positive_rows=len(spreads), scale_row_spreads=summary(spreads), tau=tau, temperature=temperature,
        full_TRAIN_rows=55915, eligible_TRAIN=eligible_count, excluded_all_F_invalid=55915-eligible_count,
        failure_candidates=failures, failure_penalty='max valid relative cost + 1; original cost +inf unchanged',
        cost_target_arithmetic='FP64 saved costs/relative exponentiation; derived training probabilities/effective costs saved FP32',
        unsupported_nonzero_target_mass_rows=unsupported, original_NoOp_rows_checked=nonop_checks,
        original_native_centers_checked=center_checks, target_peak=summary(target_peaks), effective_action_count=summary(target_effective),
        calibration=dict(rows=calibration_rows.tolist(), ids=calibration_ids, SHA_prefix='pose_loss_scale_v1:',
            size=1024, defined_multi_action=int(use.sum()), excluded=int((~use).sum()),
            argmax_exact_original_and_stored=1024, point_support_count_exact=1024,
            target_entropy_mean=float(entropy_2d.mean()), achieved_entropy_mean=temperature['achieved_normalized_entropy'],
            scope='fixed SHA-sorted TRAIN-only calibration; no heldout/real evaluation used'),
        numeric_tests=tests, validation=evidence, artifacts=[artifact(path) for path in paths.values()],
        actual_target_arithmetic_examples=1024, new_refiner_forwards=0, new_backbone_forwards=0,
        new_final_F_calls=0, new_bank_generation=0, optimizer_updates=0, seconds=time.monotonic()-started,
        setup_code_sha256=sha(__file__), loss_code_sha256=sha(Path(__file__).with_name('losses.py')))
    write(output_cache / 'LOSS_SETUP.json', result); write(out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('/home/minjae/Documents/github/pallet-pose'))
    parser.add_argument('--bank-cache', type=Path, default=Path('/tmp/pallet-joint-action-cache'))
    parser.add_argument('--cost-cache', type=Path, default=Path('/tmp/pallet-pose-target-6d-cache'))
    parser.add_argument('--output-cache', type=Path, default=Path('/tmp/pallet-quick-pose-loss-20261006-cache'))
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args(argv)
    if args.test_only:
        print(json.dumps(analytic_gradient_checks(), ensure_ascii=False, indent=2)); return
    result = prepare(args.source_root, args.bank_cache, args.cost_cache, args.output_cache)
    print(json.dumps({k:result[k] for k in ('status','scale_m','tau','eligible_TRAIN','failure_candidates','seconds')},ensure_ascii=False))


if __name__ == '__main__': main()
