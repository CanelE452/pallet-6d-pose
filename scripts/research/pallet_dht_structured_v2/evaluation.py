"""Synthetic-only frozen-checkpoint evaluation and margin calibration.

The learned cost chooses proposals without GT. Synthetic GT is used afterwards
to measure each fixed proposal and select a margin on calibration256 only.
Real annotations and populations are explicitly excluded from this module.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .cache import CachedData
from .data_ops import STATES, fixed_states, prepare_batch


SCHEMA = 'pallet_dht_structured_synthetic_evaluation_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def sources():
    folder = Path(__file__).resolve().parent
    return {str(folder/name): sha(folder/name) for name in
            ('evaluation.py', 'train.py', 'model.py', 'data_ops.py', 'proposals.py', 'cache.py')}


def verify_hashes(values):
    for path, digest in values.items():
        if sha(path) != digest:
            raise ValueError(f'Frozen synthetic evaluation input changed: {path}')


def evaluate_population(run, arm, seed, population, device='cuda:0', batch_size=8):
    """Evaluate every valid proposal for four fixed states on cal256 or val512.

    No margin, best model, GT-derived proposal or real-frame evaluation is
    performed here. Costs are FP32 learned outputs; distance arithmetic is
    FP64 on exactly the FP32 coordinates supplied to the learned verifier.
    """
    from .train import load_trained
    run = Path(run).resolve()
    if population not in ('calibration', 'synth_val'):
        raise ValueError('Only calibration and synthetic validation are allowed; real GT is not read')
    protocol = read(run/'PROTOCOL.json')
    if arm not in protocol['architecture']['arms'] or seed not in protocol['training']['seeds']:
        raise ValueError('Unregistered arm or seed')
    if batch_size < 1:
        raise ValueError('Positive inference batch_size required')
    output = run/'evaluation'/f'{arm}_seed{seed}'
    marker = output/f'{population}_COMPLETE.json'
    archive = output/f'{population}.npz'
    checkpoint = run/'runs'/f'{arm}_seed{seed}'/'checkpoint_final.pth'
    binding = {str(run/'PROTOCOL.json'): sha(run/'PROTOCOL.json'),
               str(run/'CACHE_COMPLETION.json'): sha(run/'CACHE_COMPLETION.json'),
               str(checkpoint): sha(checkpoint)}
    own_sources = sources()
    if marker.exists():
        receipt = read(marker)
        if not receipt.get('complete') or not receipt.get('PASS'):
            raise ValueError('Existing population receipt is incomplete')
        if receipt['input_sha256'] != binding or receipt['source_sha256'] != own_sources:
            raise ValueError('Existing population receipt has different source bindings')
        verify_hashes(receipt['output_sha256'])
        return receipt
    cache = CachedData(run)
    indices = np.array(cache.populations[population], np.int64)
    expected = protocol['split']['calibration_count' if population == 'calibration' else 'synthetic_validation_count']
    if len(indices) != expected or any(cache.records[i]['population'] == 'real_dev' for i in indices):
        raise ValueError('Synthetic population mismatch or real row entered evaluation')
    model = load_trained(run, arm, seed, device)
    model.eval()
    n, count_states, h = len(indices), len(STATES), protocol['architecture']['candidate_count']
    costs = np.full((count_states, n, h), np.inf, np.float32)
    errors = np.zeros((count_states, n, h, 8), np.float64)
    valid = np.zeros((count_states, n, h), bool)
    loss_valid = np.array(cache.targets['loss_valid'][indices, :8], bool)
    full_valid = np.array(cache.targets['valid'][indices, :8], bool)
    matched = np.array(cache.targets['matched'][indices], bool)
    diagonal = np.array(cache.inputs['diagonal'][indices], np.float64)
    state_variants = np.zeros((count_states, n, 4), np.float64)
    n_forwards = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for state_index, state in enumerate(STATES):
            for first in range(0, n, batch_size):
                end = min(first+batch_size, n)
                ix = indices[first:end]
                inputs, targets = cache.batch(ix, device=device)
                labels, variants = fixed_states(ix, state, seed=20260909)
                state_variants[state_index, first:end] = variants
                model_inputs, candidate_valid, retained_targets = prepare_batch(inputs, targets, labels, variants)
                if retained_targets is not targets:
                    raise ValueError('Synthetic targets must remain unchanged and separate')
                predicted_cost = model(model_inputs)
                if predicted_cost.shape != (len(ix), h) or not torch.isfinite(predicted_cost).all():
                    raise ValueError('Malformed or nonfinite learned proposal costs')
                mask = candidate_valid.detach().cpu().numpy().astype(bool)
                if mask.shape != (len(ix), h) or not mask[:, 0].all():
                    raise ValueError('Identity proposal must remain valid for every frame')
                q = model_inputs['layouts'].detach().cpu().numpy().astype(np.float64)
                gt = targets['points'].detach().cpu().numpy().astype(np.float64)
                if not np.isfinite(q).all() or not np.isfinite(gt).all():
                    raise ValueError('Synthetic cache coordinates must be finite')
                if not np.array_equal(q[:, 0], model_inputs['baseline_points'].detach().cpu().numpy()):
                    raise ValueError('Identity proposal differs from current prediction state')
                if not np.array_equal(q[:, :, 8], np.repeat(q[:, 0:1, 8], h, axis=1)):
                    raise ValueError('Centroid changed across proposals')
                e = np.linalg.norm(q[:, :, :8]-gt[:, None, :8], axis=-1)
                c = predicted_cost.detach().cpu().numpy().astype(np.float32)
                c[~mask] = np.inf
                costs[state_index, first:end] = c
                errors[state_index, first:end] = e
                valid[state_index, first:end] = mask
                n_forwards += 1
            print(f'Synthetic evaluation {arm} seed{seed} {population}: {state} {n}/{n}', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix('.pending.npz')
    np.savez_compressed(temporary, costs=costs, corner_errors_px=errors,
        candidate_valid=valid, loss_valid=loss_valid, full_gt_valid=full_valid,
        matched=matched, diagonal=diagonal, indices=indices,
        states=np.array(STATES), state_variants=state_variants,
        ids=np.array([cache.records[i]['id'] for i in indices]))
    temporary.replace(archive)
    verify_hashes(binding)
    verify_hashes(own_sources)
    receipt = dict(schema=SCHEMA, complete=True, PASS=True,
        arm=arm, seed=seed, population=population, n_frames=n, n_states=count_states,
        n_cached_head_observations=n*count_states, n_forward_batches=n_forwards,
        n_new_backbone_forwards=0, n_real_rows=0, real_GT_read=False,
        n_loss_supervised_corners=int(loss_valid.sum()),
        n_full_source_supervised_corners=int(full_valid.sum()),
        n_frames_with_loss=int(loss_valid.any(1).sum()), matched_frames=int(matched.sum()),
        gt_role='post-forward candidate errors only; not model input or proposal generation',
        metric_scope='first eight corners; loss_valid requires source v>0, frozen match and predicted availability',
        costs_semantics='Unconstrained learned cost, lower is better; invalid candidates are +inf',
        errors_semantics='FP64 Euclidean pixel distance on cache/model FP32 coordinates; masked only during aggregation',
        source_sha256=own_sources, input_sha256=binding,
        output_sha256={str(archive): sha(archive)}, archive=str(archive),
        elapsed_seconds=time.perf_counter()-started,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    write(marker, receipt)
    return receipt


def select_indices(costs, candidate_valid, margin):
    """Actual GT-free rule: strict cost gap; ties retain identity index0."""
    c = np.asarray(costs, np.float64)
    v = np.asarray(candidate_valid, bool)
    if c.shape != v.shape or c.ndim < 2 or not v[..., 0].all():
        raise ValueError('Costs/masks must agree with identity available')
    if not np.isfinite(c[v]).all():
        raise ValueError('Valid candidate costs must be finite')
    if margin == 'identity_only':
        return np.zeros(c.shape[:-1], np.int64)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError('Margin must be nonnegative or identity_only')
    masked = np.where(v, c, np.inf)
    best = np.argmin(masked, axis=-1)
    best_cost = np.take_along_axis(masked, best[..., None], axis=-1)[..., 0]
    return np.where(c[..., 0]-best_cost > margin, best, 0).astype(np.int64)


def corner_metrics(errors, mask, diagonal, baseline_errors):
    e, base = np.asarray(errors, float), np.asarray(baseline_errors, float)
    mask, diagonal = np.asarray(mask, bool), np.asarray(diagonal, float)
    if e.shape != mask.shape or base.shape != e.shape or e.ndim != 2 or diagonal.shape != (len(e),):
        raise ValueError('Expected frame-by-eight errors and masks')
    if not mask.any() or not np.isfinite(e[mask]).all() or (diagonal <= 0).any():
        raise ValueError('No supervised corners or invalid metric input')
    values = e[mask]
    good = mask & (base <= 10.)
    crossing = good & (e > 10.)
    return dict(n_frames=len(e), n_frames_with_loss=int(mask.any(1).sum()),
        n_supervised_corners=int(mask.sum()), mean_px=float(values.mean()),
        median_px=float(np.median(values)), p90_px=float(np.percentile(values, 90)),
        mean_diagonal_normalized=float((e/diagonal[:, None])[mask].mean()),
        baseline_good_le10_count=int(good.sum()), good_to_bad_gt10_count=int(crossing.sum()),
        good_to_bad_fraction=float(crossing.sum()/good.sum()) if good.any() else 0.,
        good_crossing_denominator='supervised baseline corners with error <=10px')


def rule_metrics(data, margin):
    selected = select_indices(data['costs'], data['candidate_valid'], margin)
    errors = data['corner_errors_px']
    if errors.shape[:3] != data['costs'].shape or errors.shape[-1] != 8:
        raise ValueError('Candidate error dimensions mismatch')
    result = {}
    for si, state in enumerate(STATES):
        picked = errors[si, np.arange(len(selected[si])), selected[si]]
        metric = corner_metrics(picked, data['loss_valid'], data['diagonal'], errors[si, :, 0])
        metric['selected_nonidentity_frames'] = int((selected[si] != 0).sum())
        metric['selected_indices'] = selected[si].tolist()
        result[state] = metric
    return result


def clean_constraints(metrics, baseline, tolerance=1e-12):
    return dict(
        median_no_worse=metrics['median_px'] <= baseline['median_px']+tolerance,
        p90_no_worse=metrics['p90_px'] <= baseline['p90_px']+tolerance,
        mean_no_worse=metrics['mean_px'] <= baseline['mean_px']+tolerance,
        good_crossing_le_one_percent=metrics['good_to_bad_fraction'] <= .01+tolerance)


def choose_margin(data, protocol):
    """Calibration only; invalid/crossing candidates do not enter selection."""
    cfg = protocol['calibration']
    weights = cfg['state_weights']
    if set(weights) != set(STATES) or not np.isclose(sum(weights.values()), 1.):
        raise ValueError('Registered state weights must sum to1')
    baseline = rule_metrics(data, 'identity_only')
    rows = []
    for margin in cfg['margin_grid']:
        metrics = rule_metrics(data, margin)
        constraints = clean_constraints(metrics['clean'], baseline['clean'], cfg['tie_tolerance'])
        score = sum(weights[s]*metrics[s]['mean_diagonal_normalized'] for s in STATES)
        rows.append(dict(margin=margin, objective=float(score), constraints=constraints,
                         eligible=all(constraints.values()), metrics=metrics))
    eligible = [row for row in rows if row['eligible']]
    if not eligible:
        raise ValueError('Identity rule must make calibration constraints feasible')
    best = min(row['objective'] for row in eligible)
    tied = [row for row in eligible if row['objective'] <= best+cfg['tie_tolerance']]
    # Conservative registered order: larger margin, identity-only highest.
    chosen = max(tied, key=lambda row: float('inf') if row['margin'] == 'identity_only' else row['margin'])
    return dict(margin=chosen['margin'], objective=chosen['objective'],
                metrics=chosen['metrics'], constraints=chosen['constraints'],
                baseline=baseline, grid=rows)


def assess_advancement(metrics, baseline, margin, protocol):
    clean = clean_constraints(metrics['clean'], baseline['clean'], protocol['calibration']['tie_tolerance'])
    reductions = {}
    for state in ('point_c4', 'point_and_line_c4'):
        before, after = baseline[state]['mean_px'], metrics[state]['mean_px']
        reductions[state] = (before-after)/before if before > 0 else 0.
    threshold = protocol['synthetic_advancement']['c4_corruption_mean_error_reduction_min_fraction']
    criteria = dict(finite_margin=margin != 'identity_only', **clean,
                    point_c4_reduction_at_least_half=reductions['point_c4'] >= threshold,
                    point_and_line_c4_reduction_at_least_half=reductions['point_and_line_c4'] >= threshold)
    return dict(advance=all(criteria.values()), criteria=criteria,
                C4_state_mean_error_reduction_fraction=reductions,
                requires_both_C4_states_individually=True,
                meaning='Synthetic advancement only, not real accuracy improvement or completion of active user goal')


def load_archive(receipt):
    if not receipt.get('complete') or not receipt.get('PASS'):
        raise ValueError('Completed synthetic evaluation receipt required')
    verify_hashes(receipt['input_sha256'])
    verify_hashes(receipt['source_sha256'])
    verify_hashes(receipt['output_sha256'])
    with np.load(receipt['archive'], allow_pickle=False) as a:
        data = {key: np.array(a[key]) for key in a.files}
    if data['states'].tolist() != list(STATES):
        raise ValueError('State order changed')
    return data


def calibrate_and_evaluate(run, arm, seed, device='cuda:0', batch_size=8):
    """Freeze calibration margin before reading synthetic validation results."""
    run = Path(run).resolve()
    protocol = read(run/'PROTOCOL.json')
    population = evaluate_population(run, arm, seed, 'calibration', device, batch_size)
    cal = load_archive(population)
    selected = choose_margin(cal, protocol)
    selection_path = run/f'SELECTION_{arm}_seed{seed}.json'
    input_sha = {str(run/'PROTOCOL.json'): sha(run/'PROTOCOL.json'),
                 **population['output_sha256'],
                 str(run/'evaluation'/f'{arm}_seed{seed}'/'calibration_COMPLETE.json'):
                 sha(run/'evaluation'/f'{arm}_seed{seed}'/'calibration_COMPLETE.json')}
    selection = dict(schema='pallet_dht_structured_synthetic_selection_v1', complete=True, PASS=True,
        arm=arm, seed=seed, selection_population='synthetic_calibration256',
        no_real_selection=True, synthetic_validation_used_for_selection=False,
        margin=selected['margin'], calibration=selected,
        source_sha256=sources(), input_sha256=input_sha,
        checkpoint_sha256=sha(run/'runs'/f'{arm}_seed{seed}'/'checkpoint_final.pth'))
    if selection_path.exists():
        saved = read(selection_path)
        if saved != selection:
            raise ValueError('Previously frozen selection differs; refusing overwrite')
    else:
        write(selection_path, selection)
    # This is deliberately downstream of committing the calibration selection.
    validation_receipt = evaluate_population(run, arm, seed, 'synth_val', device, batch_size)
    validation = load_archive(validation_receipt)
    metrics = rule_metrics(validation, selection['margin'])
    baseline = rule_metrics(validation, 'identity_only')
    advancement = assess_advancement(metrics, baseline, selection['margin'], protocol)
    result = dict(schema='pallet_dht_structured_synthetic_result_v1', complete=True, PASS=True,
        arm=arm, seed=seed, margin=selection['margin'],
        calibration=selected['metrics'], validation=metrics, validation_baseline=baseline,
        advancement=advancement, real_evaluation_performed=False,
        input_sha256={str(selection_path): sha(selection_path), **validation_receipt['output_sha256'],
            str(run/'PROTOCOL.json'): sha(run/'PROTOCOL.json')}, source_sha256=sources(),
        denominator_policy='First8 source-supervised corners with frozen baseline IoU>=.5 match and prediction availability; counts retained',
        no_model_or_margin_selection_on_validation=True)
    output = run/'evaluation'/f'{arm}_seed{seed}'/'SYNTHETIC_RESULTS.json'
    write(output, result)
    print(json.dumps(dict(arm=arm, seed=seed, margin=result['margin'], advancement=advancement), ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    calibrate_and_evaluate(args.run_dir, args.arm, args.seed, args.device, args.batch_size)


if __name__ == '__main__':
    main()
