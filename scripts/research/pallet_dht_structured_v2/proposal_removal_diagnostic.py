"""Saved-score, no-forward removal of DHT-snap proposal coordinates.

Protocol must exist before execution. Candidate reconstruction and selection
finish before any stored GT/error arrays are used for metric computation.
Frozen verifier weights, learned costs, margin and existing outputs are intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import data_ops as D
from . import proposals as P


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def require(x, why):
    if not x:
        raise AssertionError(why)


def verify(mapping):
    for p, h in mapping.items():
        require(sha(p) == h, f'Changed input: {p}')


def choose(cost, valid, margin):
    require(valid[0] and np.isfinite(cost[valid]).all(), 'Valid original candidate required')
    best = int(np.argmin(np.where(valid, cost, np.inf)))
    return best if cost[0]-cost[best] > margin else 0


def summarize(before, after, mask, full_gt_mask):
    e, b = after[mask], before[mask]
    good = mask & (before <= 10.)
    observed = mask.sum(1)
    frames = observed > 0
    return dict(n_frames=len(mask), n_frames_with_observations=int(frames.sum()),
        n_observed_points=int(mask.sum()), n_gt_points=int(full_gt_mask.sum()),
        n_missing_or_unmatched_points=int(full_gt_mask.sum()-mask.sum()),
        mean_px=float(e.mean()), median_px=float(np.median(e)), p90_px=float(np.quantile(e, .9)),
        frame_mean_px=float(((after*mask).sum(1)/np.maximum(observed, 1))[frames].mean()),
        pck10_observed=float((e <= 10.).mean()), pck20_observed=float((e <= 20.).mean()),
        above20_fraction=float((e > 20.).mean()), above50_fraction=float((e > 50.).mean()),
        above100_fraction=float((e > 100.).mean()),
        original_good_le10_count=int(good.sum()), good_to_bad_gt10_count=int((good & (after > 10.)).sum()),
        good_to_bad_fraction=float((good & (after > 10.)).sum()/good.sum()) if good.any() else 0.,
        improved_observed_points=int((e < b).sum()), worsened_observed_points=int((e > b).sum()))


def difficulty(before, after, mask):
    result = []
    for name, condition in [('easy_le10', before <= 10.),
                             ('moderate_10_20', (before > 10.) & (before <= 20.)),
                             ('hard_gt20', before > 20.)]:
        keep = mask & condition
        b, e = before[keep], after[keep]
        result.append(dict(bin=name, n_observed_points=int(keep.sum()), n_frames=int(keep.any(1).sum()),
            before_mean_px=float(b.mean()) if len(b) else None,
            after_mean_px=float(e.mean()) if len(e) else None,
            mean_delta_px=float((e-b).mean()) if len(e) else None,
            improved_points=int((e < b).sum()), worsened_points=int((e > b).sum()),
            posthoc_GT_bin_not_an_operational_gate=True))
    return result


def run_diagnostic(run):
    run = Path(run).resolve()
    out = run/'provenance'/'proposal_removal_diagnostic'
    proto_path = out/'PROTOCOL.json'
    protocol = read(proto_path)
    require(protocol['registered_before_diagnostic_selection'], 'Operation must be recorded before selection')
    require(protocol['arm'] == 'point_segment' and protocol['seed'] == 1 and protocol['frozen_margin'] == .5, 'Fixed arm/seed/margin')
    require(not (out/'RESULTS.json').exists(), 'Refusing to overwrite completed diagnostic')
    verify(protocol['input_sha256'])
    inputs_sha = {str(proto_path): sha(proto_path), **protocol['input_sha256']}
    sources = {str(Path(p).resolve()): sha(p) for p in (__file__, D.__file__, P.__file__)}
    sources[str(Path(__file__).with_name('model.py').resolve())] = sha(Path(__file__).with_name('model.py'))
    frozen = read(run/'SOURCE_FREEZE.json')
    verify(frozen['source_sha256'])
    records = read(run/'CACHE_RECORDS.json')['records']
    desc = read(run/'CACHE_ARRAYS.json')['arrays']
    complete = read(run/'CACHE_COMPLETION.json')
    arrays = {}
    for key, description in desc['inputs'].items():
        if key == 'p4':
            continue
        path = description['path']
        require(sha(path) == complete['array_sha256'][path], f'Cache input changed: {key}')
        inputs_sha[path] = complete['array_sha256'][path]
        arrays[key] = np.load(path, mmap_mode='r', allow_pickle=False)
    # P4 is never inspected or passed to a model. prepare_batch needs only its
    # device to create geometry tensors; all other input arrays are exact cache.
    def batch(ix):
        return {**{k: torch.from_numpy(np.array(v[ix], copy=True)) for k, v in arrays.items()},
                'p4': torch.empty((len(ix), 0), dtype=torch.float32)}

    saved_real = read(run/'REAL_PREDICTIONS_seed1.json')
    real_by_index = {r['index']: r for r in saved_real['records']}
    tasks = []
    for population in ('calibration', 'synth_val'):
        file = run/'evaluation'/'point_segment_seed1'/f'{population}.npz'
        with np.load(file, allow_pickle=False) as a:
            indices = np.array(a['indices'])
            costs = np.array(a['costs'])
            validity = np.array(a['candidate_valid'])
            variants = np.array(a['state_variants'])
            require(a['states'].tolist() == list(D.STATES), 'Stored synthetic state order')
        for si, state in enumerate(D.STATES):
            tasks.append((population, state, si, indices, costs[si], validity[si], variants[si]))
    ix = np.array([r['index'] for r in saved_real['records']], np.int64)
    cost = np.array([[float(v) if v is not None else np.inf
                     for v in real_by_index[i]['arms']['point_segment']['costs']] for i in ix])
    valid = np.array([real_by_index[i]['arms']['point_segment']['candidate_valid'] for i in ix], bool)
    tasks.append(('real_dev', 'clean', 0, ix, cost, valid, None))
    require({p: len(i) for p, _, _, i, _, _, _ in tasks} == protocol['populations'], 'Population counts')
    selected_records, bundles = [], []
    started = time.perf_counter()
    case = None
    for population, state, si, indices, costs, stored_valid, stored_variants in tasks:
        rows = []
        for first in range(0, len(indices), 8):
            end = min(first+8, len(indices))
            ix = indices[first:end]
            observation = batch(ix)
            labels, variants = D.fixed_states(ix, state)
            if stored_variants is not None:
                require(np.array_equal(variants, stored_variants[first:end]), 'Corruption variates changed')
            original, valid, _ = D.prepare_batch(observation, {}, labels, variants)
            without = dict(observation)
            without['intersection_valid'] = observation['intersection_valid'].clone()
            without['intersection_valid'][:, :, 1:] = False
            removed, removed_valid, _ = D.prepare_batch(without, {}, labels, variants)
            q, bq = original['layouts'].numpy(), removed['layouts'].numpy()
            v, bv = valid.numpy(), removed_valid.numpy()
            require(np.array_equal(v, stored_valid[first:end]), 'Original candidate mask reconstruction changed')
            for j, index in enumerate(ix):
                src = records[index]
                # Compare original FP32 coordinates: this is exactly what the
                # verifier scored, not a new continuous-coordinate proposal.
                exact = (q[j, :, None] == bq[j, None]).all(axis=(-1, -2))
                require(exact[v[j]][:, bv[j]].any(0).all(), 'A no-snap candidate is not an original exact subset')
                retained = v[j] & exact[:, bv[j]].any(1)
                require(np.array_equal(q[j, :4], bq[j, :4]), 'C4 seed coordinates/order changed')
                require(retained[:4].all(), 'Required source C4 seed removed')
                full_index = choose(costs[first+j].astype(float), v[j], .5)
                remove_index = choose(costs[first+j].astype(float), retained, .5)
                final = q[j, remove_index].astype(float)
                baseline = original['baseline_points'][j].numpy().astype(float)
                if remove_index == 0:
                    final = baseline.copy()
                require(np.array_equal(final[8], baseline[8]), 'Centroid changed')
                if population == 'real_dev':
                    saved = real_by_index[int(index)]['arms']['point_segment']
                    require(full_index == saved['selected_index'], 'Original real selection replay differs')
                    require(np.array_equal(q[j, full_index], np.asarray(saved['points'], np.float32)), 'Saved real selected coordinates differ')
                r = dict(id=src['id'], index=int(index), population=population, state=state,
                    original_selected_index=full_index, removed_selected_original_index=remove_index,
                    retained_original_indices=np.flatnonzero(retained).tolist(),
                    original_valid_count=int(v[j].sum()), no_snap_generated_valid_count=int(bv[j].sum()),
                    retained_valid_count=int(retained.sum()),
                    original_order_preserved=True, coordinate_subset_exact=True,
                    identity_cost=float(costs[first+j, 0]), original_selected_cost=float(costs[first+j, full_index]),
                    removed_selected_cost=float(costs[first+j, remove_index]),
                    removed_cost_gap=float(costs[first+j, 0]-costs[first+j, remove_index]),
                    original_selected_points=q[j, full_index].astype(float).tolist(),
                    removed_selected_points=final.tolist(), baseline_points=baseline.tolist(),
                    no_GT_selection=True, frozen_margin=.5)
                selected_records.append(r)
                rows.append(r)
                if population == 'real_dev' and src['id'] == 'plastic_day_01:005838':
                    case = dict(selection=r, baseline=baseline.tolist(), C4_slot3=q[j, 3].astype(float).tolist(),
                        saved_selected=q[j, full_index].astype(float).tolist(), after_removal=final.tolist(),
                        selected_source_intersection_slot=26, selected_source_corner=5,
                        original_kind='snap_corner_c4_3_corner_5_alpha_0.5')
        bundles.append((population, state, si, indices, rows))
        print(f'Saved-score removal {population}/{state}: {len(indices)} exact subset matches', flush=True)
    # Operational diagnostic outputs are committed before GT/error aggregation.
    selections_path = out/'SELECTIONS.json'
    write(selections_path, dict(complete=True, PASS=True, records=selected_records,
        uses_GT=False, new_model_forwards=0, margin_refit=False,
        input_sha256=inputs_sha, source_sha256=sources))

    # GT/error access begins here, after every retained mask and choice exists.
    metrics, metric_bindings = [], {}
    real_metrics_path = run/'REAL_FRAME_METRICS_seed1.json'
    real_gt = {r['id']: r for r in read(real_metrics_path)['records']}
    for population, state, si, indices, rows in bundles:
        if population != 'real_dev':
            archive = run/'evaluation'/'point_segment_seed1'/f'{population}.npz'
            with np.load(archive, allow_pickle=False) as a:
                errors = np.array(a['corner_errors_px'])[si]
                mask, full_mask = np.array(a['loss_valid']), np.array(a['full_gt_valid'])
            n = len(rows)
            baseline_error = errors[:, 0]
            original_error = errors[np.arange(n), [r['original_selected_index'] for r in rows]]
            removal_error = errors[np.arange(n), [r['removed_selected_original_index'] for r in rows]]
        else:
            full_mask = np.array([real_gt[r['id']]['gt_supervised'] for r in rows], bool)
            mask = np.array([full_mask[j] & np.array(real_by_index[r['index']]['baseline']['point_valid'], bool)
                             & real_gt[r['id']]['baseline_match_iou50'] for j, r in enumerate(rows)])
            target = np.array([real_gt[r['id']]['gt_points'] for r in rows], float)
            baseline_error = np.linalg.norm(np.array([r['baseline_points'] for r in rows])-target, axis=-1)
            original_error = np.linalg.norm(np.array([r['original_selected_points'] for r in rows])-target, axis=-1)
            removal_error = np.linalg.norm(np.array([r['removed_selected_points'] for r in rows])-target, axis=-1)
            require(int(mask.sum()) == 2738, 'Official real denominator changed')
            for j, r in enumerate(rows):
                stored = real_gt[r['id']]['arms']['point_segment']['errors_px']
                require(np.array_equal(original_error[j, mask[j]], np.array([x for x in stored if x is not None])), 'Real original metric replay differs')
        methods = dict(baseline=baseline_error, original_full_bank=original_error, no_DHT_snap_subset=removal_error)
        item = dict(population=population, state=state,
            metrics={name: summarize(baseline_error, e, mask, full_mask) for name, e in methods.items()},
            difficulty={name: difficulty(baseline_error, e, mask) for name, e in methods.items()},
            original_selected_nonidentity_ids=[r['id'] for r in rows if r['original_selected_index'] != 0],
            removed_selected_nonidentity_ids=[r['id'] for r in rows if r['removed_selected_original_index'] != 0],
            changed_output_ids=[r['id'] for r in rows if not np.array_equal(r['original_selected_points'], r['removed_selected_points'])],
            retained_count_histogram=dict(Counter(r['retained_valid_count'] for r in rows)))
        metrics.append(item)
    require(case is not None, 'Registered descriptive case missing')
    gt = real_gt[case['selection']['id']]
    target, observed = np.array(gt['gt_points'], float), np.array(gt['gt_supervised'], bool)
    observed &= np.array(real_by_index[case['selection']['index']]['baseline']['point_valid'], bool) & gt['baseline_match_iou50']
    case['posthoc_official_GT_metrics'] = {}
    for name in ('baseline', 'C4_slot3', 'saved_selected', 'after_removal'):
        errors = np.linalg.norm(np.array(case[name])-target, axis=-1)
        case['posthoc_official_GT_metrics'][name] = dict(errors9_px=errors.tolist(),
            mean9_px=float(errors[observed].mean()), median9_px=float(np.median(errors[observed])),
            P90_9_px=float(np.quantile(errors[observed], .9)))
    cm = case['posthoc_official_GT_metrics']
    case['decomposition'] = dict(ID_rotation_mean9_gain_px=cm['baseline']['mean9_px']-cm['C4_slot3']['mean9_px'],
        additional_line_snap_mean9_gain_px=cm['C4_slot3']['mean9_px']-cm['saved_selected']['mean9_px'],
        uses_GT=True, used_for_selection=False, physical_pixel_accuracy_certified=False)
    verify(inputs_sha)
    verify(sources)
    result = dict(schema=protocol['schema'], complete=True, PASS=True,
        metrics=metrics, case=case, scientific_success_claim=False,
        interpretation='Fixed-trained-scorer dependence on DHT-snap candidate availability. No retraining, threshold tuning, or stable accuracy claim.',
        operational_selection_used_GT=False, GT_used_only_after_saved_selection_for_metrics=True,
        source_real_data='Previously examined DEV319, not independent FINAL',
        no_new_model_forwards=True, no_old_outputs_modified=True,
        input_sha256={**inputs_sha, str(selections_path): sha(selections_path)}, source_sha256=sources,
        elapsed_seconds=time.perf_counter()-started)
    write(out/'RESULTS.json', result)
    write(out/'COMPLETION.json', dict(complete=True, PASS=True,
        protocol_sha256=sha(proto_path), source_sha256=sources,
        output_sha256={str(out/name): sha(out/name) for name in ('SELECTIONS.json', 'RESULTS.json')},
        scientific_success_claim=False, new_model_forwards=0))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_diagnostic(args.run_dir)
