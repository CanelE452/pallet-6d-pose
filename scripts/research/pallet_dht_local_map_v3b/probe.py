"""Frozen-weight, synthetic-only prior probe; no optimizer or real forward."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from scripts.research.pallet_dht_local_v3.cache import LocalData
from scripts.research.pallet_dht_local_v3.train import verify
from scripts.research.pallet_dht_local_v3.evaluation import choose_scale, scaled_metrics, constraints
from scripts.research.pallet_dht_structured_v2.cache import read, write, sha
from scripts.research.pallet_dht_structured_v2.train import tensor_sha
from . import model as M


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def analytic_prior_check():
    """At constant likelihood, posterior odds must equal the declared prior."""
    weights = torch.zeros((1, 12, 32, 33), dtype=torch.float64)
    length = weights.new_full((1, 12), 32.)
    tangent = weights.new_tensor([1., 0.]).expand(1, 12, 2)
    normal = weights.new_tensor([0., 1.]).expand(1, 12, 2)
    geom = dict(length=length, tangent=tangent, normal=normal,
        midpoint=weights.new_zeros((1, 12, 2)),
        along_coordinates=(torch.arange(32, dtype=weights.dtype)+.5-16.).expand(1, 12, 32),
        edge_valid=torch.ones((1, 12), dtype=torch.bool),
        contrast_present=torch.ones((1, 12), dtype=torch.bool))
    result = M.local_hough(weights, torch.ones_like(weights, dtype=torch.bool), geom)
    p = result['probability'][0, 0]
    rho_odds = float(p[8, 18]/p[8, 16])
    angle_odds = float(p[9, 16]/p[8, 16])
    sigma = math.atan(2.*math.sqrt(2.)/32.)
    require(abs(rho_odds-math.exp(-.5)) < 1e-12, 'rho2 vs rho0 prior odds')
    expected_angle = math.exp(-math.radians(1.)**2/(2.*sigma**2))
    require(abs(angle_odds-expected_angle) < 1e-12, 'alpha1deg vs alpha0 prior odds')
    require(abs(float(result['rho_input_px'][0, 0])) < 1e-12, 'Symmetric prior rho mean')
    return dict(PASS=True, synthetic_analytic_tensors_only=True,
                rho2_to_rho0_probability_ratio=rho_odds,
                alpha1deg_to_alpha0_probability_ratio=angle_odds,
                expected_rho_ratio=math.exp(-.5), expected_angle_ratio=expected_angle)


def run_probe(run):
    run = Path(run).resolve()
    require((run/'PURPOSE.md').exists(), 'Purpose required before forward')
    protocol, freeze = read(run/'PROTOCOL.json'), read(run/'SOURCE_FREEZE.json')
    require(not (run/'RESULTS.json').exists(), 'Do not overwrite a completed probe')
    require(freeze['protocol_sha256'] == sha(run/'PROTOCOL.json'), 'Frozen protocol')
    for path, expected in freeze['source_sha256'].items():
        require(sha(path) == expected, f'Frozen source: {path}')
    base = Path(protocol['source_run'])
    _, bindings = verify(base)
    require(sha(base/'SOURCE_FREEZE.json') == protocol['source_freeze_sha256'], 'Original source freeze')
    require(sha(base/'CACHE_COMPLETION.json') == protocol['source_cache_completion_sha256'], 'Original cache')
    source_done = read(base/'runs/hough_seed1/COMPLETION.json')
    checkpoint = Path(protocol['source_checkpoint'])
    require(sha(checkpoint) == protocol['source_checkpoint_sha256'] == source_done['checkpoint_sha256'], 'Original checkpoint')
    saved = torch.load(checkpoint, map_location='cpu')
    require(saved['complete'] and saved['arm'] == 'hough' and saved['seed'] == 1
            and saved['optimizer_steps'] == 1000 and saved['bindings'] == bindings, 'Completed exact source model')
    require(tensor_sha(saved['state_dict']) == saved['final_state_tensor_sha256'], 'Source tensor hash')
    model = M.LocalHoughPointRefiner(**saved['model_config'])
    model.load_state_dict(saved['state_dict'], strict=True)
    require(all(torch.equal(v, saved['state_dict'][key]) for key, v in model.state_dict().items()), 'No parameter change')
    start_state_sha = tensor_sha(model.state_dict())
    analytic = analytic_prior_check()
    write(run/'ANALYTIC_PREFLIGHT.json', dict(complete=True, PASS=True,
        prior_check=analytic, strict_source_weights_equal=True,
        maximum_parameter_difference=0., parameter_difference_variance=0.,
        checkpoint_sha256=sha(checkpoint), source_sha256=freeze['source_sha256']))
    data = LocalData(base)
    require({p: len(data.populations[p]) for p in protocol['populations']} == protocol['populations'], 'Exact source populations')
    device = protocol['evaluation']['device']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model.to(device).eval()
    outputs = {}
    elapsed = time.perf_counter()
    def predict(population):
        require(population in ('calibration', 'synth_val'), 'No real forward allowed')
        ix = data.populations[population]
        require(not (set(ix) & set(data.populations['real_dev'])), 'No real rows')
        values = {key: [] for key in ('baseline', 'predicted', 'target', 'loss_valid', 'diagonal')}
        for start in range(0, len(ix), protocol['evaluation']['batch_size']):
            rows = ix[start:start+protocol['evaluation']['batch_size']]
            inputs, targets = data.batch(rows, device)
            with torch.inference_mode():
                predicted, diagnostics = model(inputs)
            require(torch.isfinite(predicted).all(), 'Finite prediction')
            require(torch.equal(predicted[:, 8], inputs['baseline_points'][:, 8]), 'Centroid exact')
            require(torch.equal(predicted[~inputs['point_valid']], inputs['baseline_points'][~inputs['point_valid']]), 'Missing points retained')
            values['baseline'].append(inputs['baseline_points'].cpu().numpy())
            values['predicted'].append(predicted.cpu().numpy())
            values['target'].append(targets['points'].cpu().numpy())
            values['loss_valid'].append(targets['loss_valid'].cpu().numpy())
            values['diagonal'].append(inputs['diagonal'].cpu().numpy())
        values = {key: np.concatenate(parts) for key, parts in values.items()}
        values['indices'] = np.array(ix, dtype=np.int64)
        path = run/f'{population}.npz'
        np.savez_compressed(path, **values)
        outputs[str(path)] = sha(path)
        print(f'v3b actual synthetic {population}: {len(ix)}', flush=True)
        return values
    calibration = predict('calibration')
    selected = choose_scale(calibration, protocol)
    choice = dict(complete=True, PASS=True, scale=selected['scale'], calibration=selected,
        protocol_sha256=sha(run/'PROTOCOL.json'), checkpoint_sha256=sha(checkpoint),
        calibration_archive_sha256=sha(run/'calibration.npz'),
        validation_used_for_selection=False, real_GT_used=False,
        source_sha256=freeze['source_sha256'])
    write(run/'SELECTION.json', choice)
    outputs[str(run/'SELECTION.json')] = sha(run/'SELECTION.json')
    validation = predict('synth_val')
    vm = scaled_metrics(validation, selected['scale'])
    vb = scaled_metrics(validation, 0.)
    reduction = (vb['mean_px']-vm['mean_px'])/vb['mean_px']
    check = {**constraints(vm, vb), 'nonzero_scale': selected['scale'] > 0.,
        'mean_reduction_at_least_one_percent': reduction >= protocol['synthetic_advancement']['native_mean_reduction_min_fraction']}
    require(tensor_sha(model.state_dict()) == start_state_sha, 'Parameters unchanged through inference')
    for path, expected in freeze['source_sha256'].items():
        require(sha(path) == expected, 'No source modification during probe')
    require(verify(base)[1] == bindings and sha(checkpoint) == protocol['source_checkpoint_sha256'], 'Old source/checkpoint preserved')
    original_result = read(base/'evaluation/hough_seed1/SYNTHETIC_RESULTS.json')
    inputs = {str(base/'SOURCE_FREEZE.json'): sha(base/'SOURCE_FREEZE.json'),
        str(base/'CACHE_COMPLETION.json'): sha(base/'CACHE_COMPLETION.json'),
        str(checkpoint): sha(checkpoint),
        str(base/'evaluation/hough_seed1/SYNTHETIC_RESULTS.json'): sha(base/'evaluation/hough_seed1/SYNTHETIC_RESULTS.json'),
        str(run/'PROTOCOL.json'): sha(run/'PROTOCOL.json'), str(run/'SOURCE_FREEZE.json'): sha(run/'SOURCE_FREEZE.json')}
    result = dict(complete=True, PASS=True, schema=protocol['schema'],
        execution_validity=True, inference_change_probe_not_learned_final_success=True,
        scale=selected['scale'], calibration=selected, validation=vm, baseline=vb,
        original_v3_hough_validation=original_result['validation'],
        advancement=dict(advance=all(check.values()), checks=check, mean_reduction_fraction=reduction),
        actual_variant_image_forwards=768, optimizer_steps=0, real_image_forwards=0,
        real_GT_used=False, original_weights_bit_exact=True,
        maximum_parameter_difference=0., parameter_difference_variance=0.,
        state_tensor_sha256=start_state_sha, prior_analytic_check=analytic,
        elapsed_seconds=time.perf_counter()-elapsed, input_sha256=inputs,
        source_sha256=freeze['source_sha256'], output_sha256=dict(outputs),
        active_goal_complete=False)
    write(run/'RESULTS.json', result)
    outputs[str(run/'RESULTS.json')] = sha(run/'RESULTS.json')
    outputs[str(run/'ANALYTIC_PREFLIGHT.json')] = sha(run/'ANALYTIC_PREFLIGHT.json')
    write(run/'COMPLETION.json', dict(complete=True, PASS=True,
        protocol_sha256=sha(run/'PROTOCOL.json'), input_sha256=inputs,
        source_sha256=freeze['source_sha256'], output_sha256=outputs,
        optimizer_steps=0, real_image_forwards=0, scientific_advancement=all(check.values())))
    print(json.dumps({k: result[k] for k in ('scale', 'baseline', 'validation', 'advancement')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    run_probe(args.run_dir)
