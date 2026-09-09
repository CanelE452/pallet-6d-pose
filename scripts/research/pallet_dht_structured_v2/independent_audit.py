"""Independent CPU inspection of immutable training and synthetic results.

Does not instantiate a predictor, update a model, read real GT, or choose a new
margin. Training can be audited incrementally; partial audits never claim that
the full registered experiment completed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


STATES = ('clean', 'point_c4', 'point_and_line_c4', 'local_deformation')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def tensor_digest(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_bindings(values):
    for path, expected in values.items():
        require(digest(path) == expected, f'SHA mismatch: {path}')


def audit_training(run):
    run = Path(run).resolve()
    protocol = read(run/'PROTOCOL.json')
    frozen = read(run/'SOURCE_FREEZE.json')
    implementation = read(run/'IMPLEMENTATION_SPEC.json')
    cache = read(run/'CACHE_COMPLETION.json')
    require(digest(run/'PROTOCOL.json') == frozen['protocol_sha256'] == implementation['protocol_sha256'], 'Protocol binding')
    require(digest(run/'IMPLEMENTATION_SPEC.json') == frozen['implementation_spec_sha256'], 'Implementation binding')
    require(digest(run/'CACHE_COMPLETION.json') == frozen['cache_completion_sha256'], 'Cache binding')
    check_bindings(frozen['source_sha256'])
    metadata = read(run/'CACHE_RECORDS.json')
    descriptions = read(run/'CACHE_ARRAYS.json')['arrays']
    require(digest(run/'CACHE_RECORDS.json') == cache['bindings']['cache_records_sha256'], 'Cache record binding')
    require(digest(run/'CACHE_ARRAYS.json') == cache['bindings']['cache_arrays_sha256'], 'Cache array manifest binding')
    target_files = {v['path']: cache['array_sha256'][v['path']] for v in descriptions['targets'].values()}
    check_bindings(target_files)
    loss_mask = np.load(descriptions['targets']['loss_valid']['path'], mmap_mode='r', allow_pickle=False)[:, :8]
    records, populations = metadata['records'], metadata['populations']
    train = np.array(populations['train'], np.int64)
    require(len(train) == 1792 and len(set(train.tolist())) == 1792, 'Training cohort')
    for other in ('calibration', 'synth_val', 'real_dev'):
        require(not set(train.tolist()).intersection(populations[other]), f'Train/{other} overlap')
    require(all(records[i]['population'] == 'synth_train' for i in train), 'Non-training source row')
    require(not np.any(loss_mask[populations['real_dev']]), 'Real supervision entered cache')
    steps, batch = protocol['training']['steps'], protocol['training']['batch']
    require((steps, batch) == (2000, 16), 'Registered optimizer budget')
    completed, pending, paired = [], [], {}
    inspected = {str(run/name): digest(run/name) for name in
                 ('PROTOCOL.json', 'SOURCE_FREEZE.json', 'IMPLEMENTATION_SPEC.json',
                  'CACHE_COMPLETION.json', 'CACHE_RECORDS.json', 'CACHE_ARRAYS.json')}
    for seed in protocol['training']['seeds']:
        rng = np.random.default_rng(seed)
        epochs = [rng.permutation(train) for _ in range(int(np.ceil(steps*batch/len(train))))]
        order = np.concatenate(epochs)[:steps*batch].reshape(steps, batch)
        rng_state = np.random.default_rng(seed+12345)
        state = rng_state.choice(np.array(STATES), steps*batch, p=[.5, .2, .2, .1]).reshape(steps, batch)
        variants = rng_state.random((steps*batch, 4)).reshape(steps, batch, 4)
        for arm in protocol['architecture']['arms']:
            cell = run/'runs'/f'{arm}_seed{seed}'
            marker = cell/'COMPLETION.json'
            if not marker.exists():
                pending.append(f'{arm}_seed{seed}')
                continue
            done = read(marker)
            require(done['complete'] and done['PASS'] and done['stage'] == 'main', f'Incomplete main cell {arm}')
            require(done['arm'] == arm and done['seed'] == seed, 'Cell identity')
            require(done['optimizer_steps'] == done['expected_optimizer_steps'] == steps, 'Actual optimizer update budget')
            for key in ('real_samples_used', 'calibration_samples_used', 'new_CNN_forwards'):
                require(done[key] == 0, f'Unexpected {key}')
            for filename, key in [('checkpoint_final.pth', 'checkpoint_sha256'), ('TRACE.jsonl', 'trace_sha256'),
                                  ('history.json', 'history_sha256'), ('SETUP.json', 'setup_sha256'),
                                  ('GRADIENT_PROBES.json', 'gradient_probe_sha256'),
                                  ('INITIAL_STATE.pth', 'initial_state_file_sha256')]:
                path = cell/filename
                require(digest(path) == done[key], f'Training artifact changed: {path}')
                inspected[str(path)] = done[key]
            inspected[str(marker)] = digest(marker)
            setup = read(cell/'SETUP.json')
            require(setup['parameters'] == done['parameters'] == 131458, 'Parameter count')
            require(not setup['AMP'] and setup['precision'] == 'FP32', 'Precision contract')
            require(setup['bindings'] == done['bindings'], 'Setup/completion binding')
            require(done['bindings']['source_freeze_sha256'] == digest(run/'SOURCE_FREEZE.json'), 'Source freeze binding')
            trace = [json.loads(line) for line in (cell/'TRACE.jsonl').read_text().splitlines()]
            require(len(trace) == steps, 'Trace row count')
            corners = 0
            for i, row in enumerate(trace):
                require(row['step'] == i+1 and row['indices'] == order[i].tolist(), 'Step/image sampler mismatch')
                require(row['states'] == state[i].tolist(), 'Augmentation-state sampler mismatch')
                require(np.array_equal(np.asarray(row['variants']), variants[i]), 'Augmentation variates mismatch')
                count = int(loss_mask[order[i]].sum())
                require(row['supervised_corners'] == count, 'Wrong loss mask/count in actual trace')
                corners += count
            require(done['supervised_corner_exposures'] == corners, 'Total supervised exposure count')
            require(done['total_batch_exposures'] == steps*batch, 'Total image exposures')
            require(done['state_counts'] == dict(Counter(state.ravel().tolist())), 'Actual state histogram')
            initial = torch.load(cell/'INITIAL_STATE.pth', map_location='cpu')
            checkpoint = torch.load(cell/'checkpoint_final.pth', map_location='cpu')
            initial_hash, final_hash = tensor_digest(initial), tensor_digest(checkpoint['state_dict'])
            require(initial_hash == done['initial_state_tensor_sha256'] == checkpoint['initial_state_tensor_sha256'], 'Initial tensors')
            require(final_hash == done['final_state_tensor_sha256'] == checkpoint['final_state_tensor_sha256'], 'Final tensors')
            require(checkpoint['optimizer_steps'] == checkpoint['expected_optimizer_steps'] == steps and checkpoint['stage'] == 'main', 'Checkpoint update metadata')
            require(checkpoint['bindings'] == done['bindings'], 'Checkpoint source binding')
            actual_steps = Counter(int(v['step'].item()) for v in checkpoint['optimizer']['state'].values() if 'step' in v)
            require(set(actual_steps) == {steps}, 'Serialized optimizer counters')
            require({str(k): v for k, v in actual_steps.items()} == done['optimizer_parameter_step_histogram'], 'Optimizer histogram receipt')
            require(initial_hash != final_hash, 'Model never changed')
            history = read(cell/'history.json')
            require(history[-1]['step'] == history[-1]['optimizer_steps'] == steps, 'Last history step')
            probes = read(cell/'GRADIENT_PROBES.json')
            require(probes['complete'] and probes['backbone_forward_calls'] == 0, 'Gradient probe scope')
            require([r['step'] for r in probes['records']] == [1, 2, 4, 16, 100, 500, 1000, 2000], 'Gradient probe steps')
            if arm != 'point_segment_hough':
                require(all(r['line_cues_are_zero'] for r in probes['records']), 'Hough cues entered control')
            paired.setdefault(seed, []).append((arm, initial_hash, done['trace_sha256']))
            completed.append(dict(arm=arm, seed=seed, PASS=True, optimizer_steps=steps,
                actual_optimizer_parameter_step_histogram=dict(actual_steps),
                initial_state_tensor_sha256=initial_hash, final_state_tensor_sha256=final_hash,
                trace_sha256=done['trace_sha256'], supervised_corner_exposures=corners,
                training_elapsed_seconds_reported=done['elapsed_seconds'],
                training_finished_at_utc=done['finished_at_utc'],
                new_CNN_forwards=0, real_or_calibration_training_rows=0,
                gradient_probes_interpretation='Dependence only, not evidence reliability or accuracy'))
            del checkpoint, initial
    for seed, group in paired.items():
        require(len({r[1] for r in group}) == 1, f'Different initialization across arms seed{seed}')
        require(len({r[2] for r in group}) == 1, f'Different trace/mask exposure across arms seed{seed}')
    complete = not pending
    return dict(schema='pallet_dht_structured_independent_training_audit_v1',
        complete=complete, PASS=True if complete else None,
        completed_cells_PASS=True if completed else None,
        n_completed_cells=len(completed), pending_cells=pending, records=completed,
        same_initialization_and_actual_trace_across_completed_arms=True if completed else None,
        audited_at_utc=datetime.now(timezone.utc).isoformat(),
        scientific_scope='Training execution/provenance, not accuracy improvement',
        no_new_model_forwards=True, real_GT_read=False, input_sha256=inspected,
        source_sha256={str(Path(__file__).resolve()): digest(__file__)})


def audit_synthetic_results(run):
    """Independently replay fixed cost-margin choice and pixel-error metrics.

    Uses saved candidate errors, not the production evaluator's metric helpers.
    It does not verify model forward numerics a second time or read real GT.
    """
    run = Path(run).resolve()
    protocol = read(run/'PROTOCOL.json')
    results, inspected = [], {}
    for seed in protocol['training']['seeds']:
        for arm in protocol['architecture']['arms']:
            selection_path = run/f'SELECTION_{arm}_seed{seed}.json'
            selection = read(selection_path)
            require(selection['complete'] and selection['PASS'] and selection['no_real_selection'], 'Selection scope')
            require(not selection['synthetic_validation_used_for_selection'], 'Validation-based margin choice')
            check_bindings(selection['input_sha256'])
            check_bindings(selection['source_sha256'])
            cell = run/'evaluation'/f'{arm}_seed{seed}'
            result = read(cell/'SYNTHETIC_RESULTS.json')
            require(result['complete'] and result['PASS'] and result['margin'] == selection['margin'], 'Result/margin binding')
            check_bindings(result['input_sha256'])
            check_bindings(result['source_sha256'])
            inspected[str(selection_path)] = digest(selection_path)
            inspected[str(cell/'SYNTHETIC_RESULTS.json')] = digest(cell/'SYNTHETIC_RESULTS.json')
            for population in ('calibration', 'synth_val'):
                receipt = read(cell/f'{population}_COMPLETE.json')
                require(receipt['complete'] and receipt['PASS'] and not receipt['real_GT_read'], 'Population receipt')
                check_bindings(receipt['input_sha256'])
                check_bindings(receipt['source_sha256'])
                check_bindings(receipt['output_sha256'])
                inspected.update(receipt['output_sha256'])
                with np.load(receipt['archive'], allow_pickle=False) as a:
                    costs, errors = np.array(a['costs']), np.array(a['corner_errors_px'])
                    valid, mask = np.array(a['candidate_valid']), np.array(a['loss_valid'])
                    diagonal = np.array(a['diagonal'])
                require(mask.sum() == receipt['n_loss_supervised_corners'], 'Saved loss denominator')
                if population == 'calibration':
                    # Independently replay every registered margin. This is
                    # checking the existing choice, not selecting a new rule.
                    cfg = protocol['calibration']
                    scores = []
                    for grid_index, trial in enumerate(cfg['margin_grid']):
                        objective = 0.
                        clean_flags = None
                        for si, state in enumerate(STATES):
                            candidate_cost = np.where(valid[si], costs[si].astype(float), np.inf)
                            minimum = candidate_cost.argmin(1)
                            picked = np.zeros(len(minimum), int)
                            if trial != 'identity_only':
                                condition = candidate_cost[:, 0]-candidate_cost[np.arange(len(minimum)), minimum] > trial
                                picked[condition] = minimum[condition]
                            distance = errors[si, np.arange(len(picked)), picked]
                            objective += cfg['state_weights'][state]*float((distance/diagonal[:, None])[mask].mean())
                            if state == 'clean':
                                baseline = errors[si, :, 0]
                                p, b = distance[mask], baseline[mask]
                                good = mask & (baseline <= 10.)
                                crossing = float((good & (distance > 10.)).sum()/good.sum()) if good.any() else 0.
                                tol = cfg['tie_tolerance']
                                clean_flags = (np.median(p) <= np.median(b)+tol,
                                    np.quantile(p, .9) <= np.quantile(b, .9)+tol,
                                    p.mean() <= b.mean()+tol, crossing <= .01+tol)
                        saved_grid = selection['calibration']['grid'][grid_index]
                        require(saved_grid['margin'] == trial, 'Calibration grid order')
                        require(abs(saved_grid['objective']-objective) <= 1e-12, 'Calibration objective')
                        require(saved_grid['eligible'] == all(clean_flags), 'Calibration clean constraints')
                        scores.append((trial, objective, all(clean_flags)))
                    feasible = [s for s in scores if s[2]]
                    require(bool(feasible), 'No feasible identity calibration')
                    optimum = min(s[1] for s in feasible)
                    tied = [s for s in feasible if s[1] <= optimum+cfg['tie_tolerance']]
                    expected_margin = sorted(tied, key=lambda s: (s[0] == 'identity_only',
                        0. if s[0] == 'identity_only' else float(s[0])), reverse=True)[0][0]
                    require(expected_margin == selection['margin'], 'Calibration winner or conservative tie break')
                margin = selection['margin']
                for si, state in enumerate(STATES):
                    active_cost = np.where(valid[si], costs[si].astype(float), np.inf)
                    best = active_cost.argmin(1)
                    choice = np.zeros(len(best), int)
                    if margin != 'identity_only':
                        moved = active_cost[:, 0]-active_cost[np.arange(len(best)), best] > margin
                        choice[moved] = best[moved]
                    selected_errors = errors[si, np.arange(len(choice)), choice]
                    pooled = selected_errors[mask]
                    metric = (selection['calibration']['metrics'] if population == 'calibration' else result['validation'])[state]
                    expected = dict(mean_px=np.mean(pooled), median_px=np.median(pooled),
                                    p90_px=np.quantile(pooled, .9),
                                    mean_diagonal_normalized=np.mean((selected_errors/diagonal[:, None])[mask]))
                    for key, value in expected.items():
                        require(abs(float(value)-metric[key]) <= 1e-12, f'{arm}/{population}/{state}/{key} differs')
                    require(metric['selected_indices'] == choice.tolist(), 'GT-free cost choice differs')
                    good = mask & (errors[si, :, 0] <= 10.)
                    count = int((good & (selected_errors > 10.)).sum())
                    require(metric['good_to_bad_gt10_count'] == count and metric['baseline_good_le10_count'] == int(good.sum()), 'Good-corner denominator/crossing')
            results.append(dict(arm=arm, seed=seed, PASS=True, margin=selection['margin'],
                                synthetic_advance=result['advancement']['advance']))
    return dict(schema='pallet_dht_structured_independent_synthetic_audit_v1', complete=True, PASS=True,
        records=results, input_sha256=inspected, real_GT_read=False, new_model_forwards=0,
        scope='Saved learned costs -> frozen margin choices -> first8 supervised metrics; not an accuracy-improvement claim',
        full_calibration_grid_objective_constraints_and_tie_break_replayed=True,
        source_sha256={str(Path(__file__).resolve()): digest(__file__)})


def audit_proposal_headroom(run):
    """Posthoc SAME-ID whole-layout GT oracle for the fixed full arm only.

    One complete existing layout is selected per frame. Corners from different
    layouts are never mixed, and no GT-derived proposal enters inference.
    This oracle minimizes supervised *mean* error, not every individual point.
    """
    run = Path(run).resolve()
    protocol = read(run/'PROTOCOL.json')
    arm = 'point_segment_hough'
    require(arm in protocol['architecture']['arms'], 'Fixed full arm missing')
    output, inspected = [], {str(run/'PROTOCOL.json'): digest(run/'PROTOCOL.json')}
    epsilon = 1e-9  # Numerical comparison tolerance only, never a fitted gate.
    for seed in protocol['training']['seeds']:
        cell = run/'evaluation'/f'{arm}_seed{seed}'
        receipt_path = cell/'synth_val_COMPLETE.json'
        selection_path = run/f'SELECTION_{arm}_seed{seed}.json'
        receipt, selection = read(receipt_path), read(selection_path)
        require(receipt['complete'] and receipt['PASS'] and receipt['population'] == 'synth_val', 'Full-arm validation incomplete')
        require(selection['complete'] and selection['no_real_selection'], 'Frozen synthetic margin required')
        for hashes in (receipt['input_sha256'], receipt['source_sha256'], receipt['output_sha256'],
                       selection['input_sha256'], selection['source_sha256']):
            check_bindings(hashes)
        inspected.update(receipt['output_sha256'])
        inspected[str(receipt_path)] = digest(receipt_path)
        inspected[str(selection_path)] = digest(selection_path)
        with np.load(receipt['archive'], allow_pickle=False) as a:
            e = np.array(a['corner_errors_px'])
            c, valid = np.array(a['costs']), np.array(a['candidate_valid'])
            mask, ids = np.array(a['loss_valid']), np.array(a['ids']).tolist()
            indices = np.array(a['indices'])
        counts = mask.sum(1)
        supervised = counts > 0
        require(len(ids) == 512 and counts.sum() == receipt['n_loss_supervised_corners'], 'Validation denominator')
        state_reports = {}
        for si, state in enumerate(STATES):
            frame_candidate_mean = (e[si]*mask[:, None]).sum(2)/np.maximum(counts[:, None], 1)
            allowed_mean = np.where(valid[si], frame_candidate_mean, np.inf)
            oracle = allowed_mean.argmin(1)
            oracle[~supervised] = 0
            require(np.all(allowed_mean[np.arange(len(ids)), oracle] <= frame_candidate_mean[:, 0]+epsilon), 'Original fallback excluded from oracle')
            active_cost = np.where(valid[si], c[si].astype(float), np.inf)
            raw_argmin = active_cost.argmin(1)
            chosen = np.zeros(len(ids), int)
            if selection['margin'] != 'identity_only':
                change = active_cost[:, 0]-active_cost[np.arange(len(ids)), raw_argmin] > selection['margin']
                chosen[change] = raw_argmin[change]
            q = dict(identity=np.zeros(len(ids), int), actual_selected=chosen,
                     learned_cost_argmin_before_margin=raw_argmin, GT_whole_layout_oracle=oracle)
            metrics = {}
            for method, choice in q.items():
                selected_error = e[si, np.arange(len(ids)), choice]
                pooled = selected_error[mask]
                good = mask & (e[si, :, 0] <= 10.)
                frames = selected_error.sum(1)  # overwritten by masked mean below
                frames = (selected_error*mask).sum(1)/np.maximum(counts, 1)
                metrics[method] = dict(mean_supervised_point_px=float(pooled.mean()),
                    median_supervised_point_px=float(np.median(pooled)),
                    p90_supervised_point_px=float(np.quantile(pooled, .9)),
                    mean_frame_mean_px=float(frames[supervised].mean()),
                    n_good_corners_crossed_above10=int((good & (selected_error > 10.)).sum()),
                    n_original_good_corners=int(good.sum()),
                    GT_used_to_choose_layout=method == 'GT_whole_layout_oracle')
            per_frame, categories = [], Counter()
            for i, frame_id in enumerate(ids):
                before = float(frame_candidate_mean[i, 0]) if supervised[i] else None
                after = float(frame_candidate_mean[i, chosen[i]]) if supervised[i] else None
                best = float(frame_candidate_mean[i, oracle[i]]) if supervised[i] else None
                if not supervised[i]:
                    category = 'no_supervised_corner_in_frozen_loss_mask'
                elif best >= before-epsilon:
                    category = 'no_better_whole_proposal_than_identity'
                elif after >= before-epsilon:
                    category = 'better_proposal_available_but_scoring_or_margin_did_not_improve'
                elif after > best+epsilon:
                    category = 'selected_improved_with_remaining_scoring_or_margin_gap'
                else:
                    category = 'selected_attained_best_available_mean'
                categories[category] += 1
                per_frame.append(dict(id=frame_id, source_index=int(indices[i]),
                    n_supervised_corners=int(counts[i]), n_valid_whole_proposals=int(valid[si, i].sum()),
                    actual_selected_index=int(chosen[i]), raw_cost_argmin_index=int(raw_argmin[i]),
                    GT_oracle_whole_layout_index=int(oracle[i]) if supervised[i] else None,
                    identity_mean_px=before, selected_mean_px=after, GT_oracle_mean_px=best,
                    available_mean_improvement_px=before-best if supervised[i] else None,
                    selected_to_best_available_mean_gap_px=after-best if supervised[i] else None,
                    category=category, uses_GT_for_diagnostic=True,
                    GT_oracle_used_for_actual_selection=False))
            state_reports[state] = dict(n_frames=len(ids), n_supervised_frames=int(supervised.sum()),
                n_supervised_corners=int(mask.sum()), metrics=metrics, frame_categories=dict(categories),
                per_frame=per_frame,
                oracle_best_layout_objective='per-frame mean error of the SAME eight semantic IDs under frozen loss_valid',
                independent_percorner_oracle=False,
                pointwise_non_degradation_guaranteed=False,
                hypothesis_limit='Fixed existing64 slots, only inference-valid complete layouts; no GT points injected')
        output.append(dict(arm=arm, seed=seed, frozen_margin=selection['margin'], states=state_reports))
    return dict(schema='pallet_dht_structured_posthoc_whole_layout_oracle_v1', complete=True, PASS=True,
        records=output, uses_GT=True, used_for_model_training=False, used_for_margin_selection=False,
        used_for_operational_candidate_selection=False, real_GT_read=False, new_model_forwards=0,
        interpretation='Synthetic-only posthoc proposal availability vs learned scoring/margin gap; not deployable correction',
        numerical_comparison_epsilon_px=epsilon, input_sha256=inspected,
        source_sha256={str(Path(__file__).resolve()): digest(__file__)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('training', 'synthetic', 'headroom'), default='training')
    args = parser.parse_args()
    fn = dict(training=audit_training, synthetic=audit_synthetic_results, headroom=audit_proposal_headroom)[args.phase]
    result = fn(args.run_dir)
    out = args.run_dir/'provenance'/'independent_audit'/f'{args.phase.upper()}_AUDIT.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('complete', 'PASS')}, allow_nan=False))


if __name__ == '__main__':
    main()
