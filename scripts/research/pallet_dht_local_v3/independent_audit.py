"""Read-only CPU replay of actual local-v3 training and synthetic evaluation.

No production metric, calibration, sampler, model or forward function is used.
The output verifies execution and arithmetic, not scientific improvement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def compare(actual, expected, label):
    if isinstance(expected, dict):
        for key, value in expected.items():
            compare(actual[key], value, f'{label}.{key}')
    elif isinstance(expected, (float, np.floating)):
        require(np.isclose(actual, expected, atol=1e-12, rtol=0), label)
    else:
        require(actual == expected, label)


def tensor_digest(state):
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode())
        h.update(state[name].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def metric(values, scale):
    raw = values['baseline'].astype(np.float64)
    pred = values['predicted'].astype(np.float64)
    target = values['target'].astype(np.float64)
    mask = values['loss_valid'][:, :8].astype(bool)
    d = values['diagonal'].astype(np.float64)
    corrected = raw+float(scale)*(pred-raw)
    e = np.sqrt(np.square(corrected[:, :8]-target[:, :8]).sum(-1))
    before = np.sqrt(np.square(raw[:, :8]-target[:, :8]).sum(-1))
    good = mask & (before <= 10.)
    crossed = good & (e > 10.)
    x = e[mask]
    require(len(x) and np.isfinite(x).all() and (d > 0).all(), 'Valid metric population')
    return dict(n_frames=len(mask), n_frames_with_loss=int(mask.any(1).sum()),
        n_supervised_corners=int(mask.sum()), mean_px=float(x.mean()),
        median_px=float(np.quantile(x, .5)), p90_px=float(np.quantile(x, .9)),
        mean_diagonal_normalized=float((e/d[:, None])[mask].mean()),
        baseline_good_le10_count=int(good.sum()), good_to_bad_gt10_count=int(crossed.sum()),
        good_to_bad_fraction=float(crossed.sum()/good.sum()) if good.any() else 0.)


def checks(m, baseline, tolerance=1e-12):
    return {**{f'{name}_no_worse': m[name] <= baseline[name]+tolerance
               for name in ('median_px', 'p90_px', 'mean_px')},
            'good_crossing_le_one_percent': m['good_to_bad_fraction'] <= .01+tolerance}


def audit(run):
    run = Path(run).resolve()
    tracked = {}
    def bind(path, expected=None):
        path = Path(path).resolve()
        current = sha(path)
        if expected is not None:
            require(current == expected, f'Binding: {path}')
        tracked[str(path)] = current
        return current
    def bound_json(path, expected=None):
        bind(path, expected)
        return read(path)
    protocol = bound_json(run/'PROTOCOL.json')
    frozen = bound_json(run/'SOURCE_FREEZE.json')
    require(frozen['complete'] and frozen['main_optimizer_steps_at_freeze'] == 0,
            'Pre-main freeze receipt')
    bind(run/'PROTOCOL.json', frozen['protocol_sha256'])
    bind(run/'IMPLEMENTATION_SPEC.json', frozen['implementation_spec_sha256'])
    for path, h in frozen['source_sha256'].items():
        bind(path, h)
    cache = bound_json(run/'CACHE_COMPLETION.json', frozen['cache_completion_sha256'])
    records = bound_json(run/'CACHE_RECORDS.json', cache['cache_records_sha256'])
    arrays = bound_json(run/'CACHE_ARRAYS.json', cache['cache_arrays_sha256'])['arrays']
    populations = records['populations']
    require({k: len(v) for k, v in populations.items()} == {
        k: protocol['split'][k] for k in ('train', 'calibration', 'synth_val', 'real_dev')},
        'Registered population sizes')
    require(sorted(sum(populations.values(), [])) == list(range(2879)), 'Disjoint population partition')
    def array(group, key):
        path = arrays[group][key]['path']
        bind(path, cache['array_sha256'][path])
        return np.load(path, mmap_mode='r', allow_pickle=False)
    baseline = array('inputs', 'baseline_points')
    diagonal = array('inputs', 'diagonal')
    affine = array('inputs', 'raw_to_input_affine')
    available = array('inputs', 'point_valid')
    target = array('targets', 'points')
    mask = array('targets', 'loss_valid')
    gt_valid = array('targets', 'valid')
    matched = array('targets', 'matched')
    require(np.array_equal(mask, gt_valid & available & matched[:, None]), 'Exact supervision mask')
    require(not np.any(mask[populations['real_dev']]) and not np.any(target[populations['real_dev']]),
            'Real targets remain zero/false')
    require(np.array_equal(affine[:, 0, 0], affine[:, 1, 1]) and (affine[:, 0, 0] > 0).all(),
            'Actual input gain is isotropic')
    cfg = protocol['training']
    training, evaluations = [], []
    for seed in cfg['seeds']:
        rng = np.random.default_rng(seed)
        n = cfg['steps']*cfg['batch']
        shuffled = []
        while len(shuffled) < n:
            shuffled += rng.permutation(populations['train']).tolist()
        order = np.array(shuffled[:n], np.int64).reshape(cfg['steps'], cfg['batch'])
        noise_rng = np.random.default_rng(seed+1731)
        noisy = noise_rng.random(n) >= cfg['clean_probability']
        noise = noise_rng.normal(0., cfg['jitter_input_sigma_px'], size=(n, 8, 2))
        noise = np.minimum(cfg['jitter_clip_input_px'], np.maximum(-cfg['jitter_clip_input_px'], noise))
        noise = (noise*noisy[:, None, None]).astype(np.float32).reshape(cfg['steps'], cfg['batch'], 8, 2)
        for arm in protocol['model']['arms']:
            cell = run/'runs'/f'{arm}_seed{seed}'
            done = bound_json(cell/'COMPLETION.json')
            setup = bound_json(cell/'SETUP.json', done['setup_sha256'])
            history = bound_json(cell/'history.json', done['history_sha256'])
            probes = bound_json(cell/'GRADIENT_PROBES.json', done['gradient_probes_sha256'])
            bind(cell/'TRACE.jsonl', done['trace_sha256'])
            trace = [json.loads(row) for row in (cell/'TRACE.jsonl').read_text().splitlines()]
            bind(cell/'INITIAL_STATE.pth')
            initial = torch.load(cell/'INITIAL_STATE.pth', map_location='cpu')
            bind(done['checkpoint'], done['checkpoint_sha256'])
            saved = torch.load(done['checkpoint'], map_location='cpu')
            require(done['complete'] and done['PASS'] and saved['complete'], 'Completed actual training')
            require(done['arm'] == saved['arm'] == arm and done['seed'] == saved['seed'] == seed, 'Arm/seed identity')
            require(done['optimizer_steps'] == saved['optimizer_steps'] == len(trace) == cfg['steps'] == 1000,
                    'Actual final1000 updates')
            require([r['step'] for r in trace] == list(range(1, 1001)), 'One trace per update')
            require(np.array_equal(np.array([r['indices'] for r in trace]), order), 'Actual shuffled indices')
            require(np.array_equal(np.array([r['jitter_input_px'] for r in trace], np.float32), noise), 'Actual input jitter')
            require(set(order.ravel()) <= set(populations['train']), 'Source-only optimizer exposures')
            per_step = mask[order, :8].sum((1, 2))
            require(np.array_equal(per_step, [r['supervised_corners'] for r in trace]), 'Actual denominator each update')
            require(int(per_step.sum()) == done['supervised_corner_exposures'], 'Total supervision exposures')
            initialsha = tensor_digest(initial)
            finalsha = tensor_digest(saved['state_dict'])
            require(initialsha == done['initial_state_tensor_sha256'] == setup['initial_state_tensor_sha256'], 'Initial tensor SHA')
            require(finalsha == done['final_state_tensor_sha256'] == saved['final_state_tensor_sha256'], 'Final tensor SHA')
            require(initialsha != finalsha, 'Learned state changed')
            require(all(torch.isfinite(v).all() for v in saved['state_dict'].values()), 'Finite trained weights')
            require(sum(v.numel() for v in saved['state_dict'].values()) == done['parameters'] == 3243, 'Actual parameter count')
            step_values = [int(v['step'].item()) for v in saved['optimizer']['state'].values()]
            require(step_values and set(step_values) == {1000}, 'Actual AdamW state steps')
            inactive = setup['inactive_parameter_prefixes']
            require(all(torch.equal(v, saved['state_dict'][name]) for name, v in initial.items()
                        if any(name.startswith(prefix) for prefix in inactive)), 'Inactive branch unchanged')
            require(probes['point_loss_only'] and [x['step'] for x in probes['records']] == [1, 2, 4, 16, 100, 500, 1000],
                    'Registered gradient evidence')
            require(any(value is not None and value > 0 for row in probes['records'][1:]
                        for name, value in row['gradient_l2'].items() if name.startswith('cnn.')), 'Actual CNN point-loss gradient')
            require(not done['real_GT_used'] and done['new_backbone_forwards'] == 0, 'No new backbone/real training')
            training.append(dict(arm=arm, seed=seed, optimizer_steps=1000,
                supervised_corner_exposures=int(per_step.sum()), actual_optimizer_parameter_states=len(step_values),
                parameters=3243, initial_state_sha256=initialsha, final_state_sha256=finalsha,
                trace_sha256=done['trace_sha256'], elapsed_seconds=done['elapsed_seconds'],
                first_window_loss=history[0]['mean_loss'], last_window_loss=history[-1]['mean_loss'],
                actual_point_loss_CNN_gradient_nonzero=True,
                final_signed_gates=torch.tanh(saved['state_dict']['gate_logits']).tolist()))
            values = {}
            for population in ('calibration', 'synth_val'):
                directory = run/'evaluation'/f'{arm}_seed{seed}'
                receipt = bound_json(directory/f'{population}_COMPLETE.json')
                require(receipt['complete'] and receipt['PASS'] and not receipt['real_GT_used'], 'Synthetic inference receipt')
                require(receipt['checkpoint_sha256'] == done['checkpoint_sha256'], 'Evaluated checkpoint identity')
                bind(receipt['archive'], receipt['archive_sha256'])
                with np.load(receipt['archive'], allow_pickle=False) as a:
                    v = {key: np.array(a[key]) for key in a.files}
                ix = np.array(populations[population])
                require(np.array_equal(v['indices'], ix), 'Frozen evaluation population IDs/order')
                for key, source in [('baseline', baseline), ('target', target), ('loss_valid', mask), ('diagonal', diagonal)]:
                    require(np.array_equal(v[key], source[ix]), f'Unchanged archive {key}')
                require(np.isfinite(v['predicted']).all() and np.array_equal(v['predicted'][:, 8], v['baseline'][:, 8]),
                        'Finite predictions and exact centroid')
                require(np.array_equal(v['predicted'][~available[ix]], v['baseline'][~available[ix]]), 'Missing corner preservation')
                movement = np.linalg.norm((v['predicted'][:, :8]-v['baseline'][:, :8]).astype(float)
                                          * affine[ix, 0, 0, None, None], axis=-1)
                require(movement.max() <= 4.+1e-4, '4 input-pixel cap allowing only FP32 coordinate rounding')
                values[population] = v
            selection = bound_json(run/f'SELECTION_{arm}_seed{seed}.json')
            result = bound_json(run/'evaluation'/f'{arm}_seed{seed}'/'SYNTHETIC_RESULTS.json')
            require(not selection['real_GT_used'] and not selection['validation_used_for_selection'], 'Calibration-only selection receipt')
            base = metric(values['calibration'], 0.)
            rows = []
            for scale in protocol['calibration']['output_scale_grid']:
                m = metric(values['calibration'], scale)
                rulechecks = checks(m, base, protocol['calibration']['tie_tolerance'])
                rows.append(dict(scale=scale, metrics=m, constraints=rulechecks, eligible=all(rulechecks.values())))
            for actual, expected in zip(selection['calibration']['grid'], rows):
                compare(actual, expected, f'{arm} calibration')
            eligible = [row for row in rows if row['eligible']]
            best = min(row['metrics']['mean_diagonal_normalized'] for row in eligible)
            scale = min(row['scale'] for row in eligible if row['metrics']['mean_diagonal_normalized'] <= best+1e-12)
            require(scale == selection['scale'] == result['scale'], 'Synthetic-only grid/tie replay')
            vm = metric(values['synth_val'], scale)
            vb = metric(values['synth_val'], 0.)
            compare(result['validation'], vm, f'{arm} validation')
            compare(result['baseline'], vb, f'{arm} baseline')
            reduction = 1.-vm['mean_px']/vb['mean_px']
            advancechecks = {**checks(vm, vb), 'nonzero_scale': scale > 0.,
                'mean_reduction_at_least_one_percent': reduction >= protocol['synthetic_advancement']['native_mean_reduction_min_fraction']}
            compare(result['advancement'], dict(advance=all(advancechecks.values()), checks=advancechecks,
                                              mean_reduction_fraction=reduction), f'{arm} advancement')
            evaluations.append(dict(arm=arm, seed=seed, scale=scale, baseline=vb, validation=vm,
                advance=all(advancechecks.values()), mean_reduction_fraction=reduction,
                grid_replayed=True, same_ID_first8_mask_preserved=True))
    for seed in cfg['seeds']:
        group = [x for x in training if x['seed'] == seed]
        require(len({x['trace_sha256'] for x in group}) == len({x['initial_state_sha256'] for x in group}) == 1,
                'Matched arms same initial state, shuffled exposures and jitter')
    pilot = bound_json(run/'PILOT_RESULTS.json')
    require(pilot['complete'] and pilot['PASS'] and not pilot['real_evaluation_done'], 'Synthetic pilot completion')
    for path, expected in tracked.items():
        require(sha(path) == expected, f'Input changed during read-only audit: {path}')
    output = dict(complete=True, PASS=True, schema='pallet_dht_local_v3_independent_audit',
        protocol_sha256=frozen['protocol_sha256'], training=training, synthetic_evaluations=evaluations,
        execution_and_arithmetic_valid=True, scientific_advancement=any(x['advance'] for x in evaluations),
        actual_model_forwards_in_this_audit=0, GPU_used=False, real_GT_read=False,
        numerical_tolerance=dict(metric_absolute=1e-12, metric_relative=0,
            input_shift_cap_rounding_px=1e-4, coordinate_and_mask_comparison='exact'),
        limitations=['Inference is replayed from saved actual outputs; no second model forward.',
                     'Outcome is source-only synthetic validation, not real-domain accuracy.'],
        input_sha256=tracked, source_sha256={str(Path(__file__).resolve()): sha(__file__)})
    destination = run/'provenance/independent_audit/INDEPENDENT_AUDIT.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), 'Do not overwrite an independent audit')
    destination.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True, PASS=True, output=str(destination),
        scientific_advancement=output['scientific_advancement'], synthetic_evaluations=evaluations), indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    audit(arguments.run_dir)
