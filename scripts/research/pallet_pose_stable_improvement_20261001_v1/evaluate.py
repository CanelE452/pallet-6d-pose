"""Frozen-output GEO/T/R evaluation; no fitting, inference, or model selection.

Run ``freeze`` before ``score`` (or ``all``). Every prediction binding is
verified before references are opened. The original diagnosis owns geometry
and metrics; only recording/seed aggregation and the locked gates are new.
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import time

import numpy as np

from . import common as C

METRICS = ('translation_cm', 'rotation_deg')
FIXED = ('R0', 'PRIOR1', 'FULL125')
BOOT_REPEATS = 2000
BOOT_SEED = 20261001
GUARD_RATIO = 1.05


def model_names():
    return [*FIXED, *(f'{a}_s{s}' for a in C.ARMS for s in C.SEEDS)]


def bindings_in(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            yield value
        else:
            for child in value.values():
                yield from bindings_in(child)
    elif isinstance(value, list):
        for child in value:
            yield from bindings_in(child)


def verify_file(binding):
    actual = C.bind(C.ROOT / binding['path'])
    assert actual['sha256'] == binding['sha256'], binding['path']
    if 'bytes' in binding:
        assert actual['bytes'] == binding['bytes'], binding['path']


def diagnosis_manifest(protocol):
    """Authenticate the historical manifest without opening its references."""
    verify_file(protocol['parent_diagnosis'])
    publication = C.read(C.ROOT / protocol['parent_diagnosis']['path'])
    path = str((C.D.DOC / 'RUN_MANIFEST.json').relative_to(C.ROOT))
    binding = next(b for b in publication['files'] if b['path'] == path)
    verify_file(binding)
    return C.read(C.ROOT / path), binding


def reference_bindings(protocol, rows):
    """Called only after prediction/pose locks; verify old expected hashes."""
    historical, historical_binding = diagnosis_manifest(protocol)
    plastic = [b for b in historical['inputs'] if '/annotations/' in b['path']]
    assert len(plastic) == 128
    audit = C.read(C.DOC / 'DATA_AUDIT.json')
    path = str((C.ROOT / '_docs/experiments/pallet_material_selftrain_closure_v1/EVAL_POPULATION_LOCK.json').relative_to(C.ROOT))
    wood_lock = next(b for b in audit['wood45_cache_audit']['prediction_lock']['checks'] if b['path'] == path)
    verify_file(wood_lock)
    wood = C.read(C.ROOT / path)
    assert wood['n'] == 45
    assert {r['id'] for r in wood['records']} == {r['id'] for r in audit['wood45_cache_audit']['metadata']['records']}
    annotations = plastic + [r['annotation'] for r in wood['records']]
    assert len({b['path'] for b in annotations}) == 173
    metric_path = str((C.ROOT / '_docs/experiments/pallet_pose_objective_followup_v2/METRIC_AND_SELECTION_LOCK.json').relative_to(C.ROOT))
    metric_lock = next(b for b in historical['inputs'] if b['path'] == metric_path)
    verify_file(metric_lock)
    metric_manifest = C.read(C.ROOT / metric_path)
    geometry = [b for b in metric_manifest['inputs'] if Path(b['path']).name in ('AXIS_REVIEW_MANIFEST.json', 'GEOMETRY_RESOLVED_POSE_GT.json')]
    assert len(geometry) == 2
    images = [r['image'] for r in rows]
    for binding in annotations + geometry + images:
        verify_file(binding)
    return dict(verified=True, annotation_count=len(annotations), image_count=len(images),
        annotations=annotations, geometry=geometry, images=images,
        provenance=[historical_binding, wood_lock, metric_lock],
        verification_stage='After complete prediction and pose locks, before Pose.metadata REAL_DEV reference load.')


def locked_inputs():
    """This function is deliberately reference-free."""
    path = C.DOC / 'PREDICTIONS_LOCK.json'
    lock = C.read(path)  # Missing/incomplete output cannot proceed to scoring.
    assert lock['complete'] is True
    for binding in bindings_in(lock):
        verify_file(binding)
    protocol = C.protocol()
    assert C.read(C.ROOT / lock['protocol']['path']) == protocol
    assert set(lock['predictions']) == set(model_names())
    rows = C.read(C.ROOT / lock['metadata']['path'])
    assert isinstance(rows, list) and len(rows) == 173
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids))
    original = C.D.metadata()
    audited = C.read(C.DOC / 'DATA_AUDIT.json')['wood45_cache_audit']['metadata']
    wood = audited['records']
    by_id = {r['id']: r for r in original + wood}
    assert set(by_id) == set(ids)
    for row in rows:
        old = by_id[row['id']]
        for key in ('K', 'xyz', 'hw', 'recording', 'severity', 'image'):
            assert row[key] == old[key], (row['id'], key)
    natural = [r for r in original if r['severity'] != 'CLEAN']
    assert len({r['recording'] for r in natural}) == 6
    assert len({r['recording'] for r in wood}) == 2
    predictions = {}
    for model in model_names():
        value = C.read(C.ROOT / lock['predictions'][model]['path'])
        predictions[model] = value.get('predictions', value)
        assert set(predictions[model]) == set(ids), model
        if model != 'R0':
            for fid in ids:
                C.L.E.assert_preserved(predictions['R0'][fid], predictions[model][fid])
    return lock, rows, predictions, protocol


def groups(rows):
    plastic = {r['id'] for r in C.D.metadata()}
    out = {'FULL128': [r['id'] for r in rows if r['id'] in plastic],
           'NATURAL99': [r['id'] for r in rows if r['id'] in plastic and r['severity'] != 'CLEAN'],
           'CLEAN29': [r['id'] for r in rows if r['id'] in plastic and r['severity'] == 'CLEAN'],
           'WOOD45': [r['id'] for r in rows if r['id'] not in plastic]}
    assert {k: len(v) for k, v in out.items()} == dict(FULL128=128, NATURAL99=99, CLEAN29=29, WOOD45=45)
    out.update({f'WOOD_{severity}': [r['id'] for r in rows if r['id'] not in plastic and r['severity'] == severity]
                for severity in ('CLEAN', 'MODERATE_OCCLUSION')})
    assert len(out['WOOD_CLEAN']) == 38 and len(out['WOOD_MODERATE_OCCLUSION']) == 7
    out.update({f'PLASTIC_{severity}': [r['id'] for r in rows if r['id'] in plastic and r['severity'] == severity]
                for severity in ('MODERATE_OCCLUSION', 'SEVERE_OCCLUSION')})
    assert len(out['PLASTIC_MODERATE_OCCLUSION']) == 21 and len(out['PLASTIC_SEVERE_OCCLUSION']) == 78
    return out


def freeze():
    start = time.monotonic()
    lock, rows, predictions, protocol = locked_inputs()
    destination = C.DOC / 'POSE_PREDICTIONS_LOCK.json'
    if destination.exists():
        existing = C.read(destination)
        assert existing['prediction_lock'] == C.bind(C.DOC / 'PREDICTIONS_LOCK.json')
        for binding in existing['files']:
            verify_file(binding)
        print('POSE_PREDICTIONS_ALREADY_FROZEN', flush=True)
        return
    historical, _ = diagnosis_manifest(protocol)
    selector_path = str((C.ROOT / '_docs/experiments/pallet_selector_recovery_v1/stage2_synth_scorer/SCORER_SELECTION_LOCK.json').relative_to(C.ROOT))
    selector_lock = next(b for b in historical['inputs'] if b['path'] == selector_path)
    verify_file(selector_lock)
    selector = C.read(C.ROOT / selector_path)
    assert selector['winner'] == 'GEO_LINEAR'
    verify_file(selector['checkpoint'])
    # Registry dimensions, symmetry, and positive membership are metadata,
    # not outcome coordinates. Keep their historical expected hashes too.
    legacy_sources_path = C.ROOT / '_docs/experiments/pallet_dim_conditioned_p_v1/SOURCE_BINDINGS.json'
    legacy_sources = list(bindings_in(C.read(legacy_sources_path)))
    required_geometry = {'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json',
        '_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json'}
    geometry_inputs = [b for b in legacy_sources if b['path'] in required_geometry]
    assert {b['path'] for b in geometry_inputs} == required_geometry
    data_audit = C.read(C.DOC / 'DATA_AUDIT.json')
    membership = next(b for b in data_audit['source_bindings'] if b['path'] == 'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json')
    geometry_inputs += [membership]
    for binding in geometry_inputs:
        verify_file(binding)
    ck = C.D.geo_checkpoint()
    output = {}
    for model in model_names():
        output[model] = {r['id']: C.D.candidates(predictions[model][r['id']], r, ck) for r in rows}
        print('FROZEN_GEO', model, len(output[model]), flush=True)
    path = C.RAW / 'POSE_CANDIDATES.json'
    C.save(path, output)
    membership = C.RAW / 'EVAL_GROUPS.json'
    C.save(membership, groups(rows))
    source_codes = [Path(__file__), Path(C.D.__file__), Path(C.D.O.__file__),
                    Path(C.D.O.D.__file__), Path(C.D.O.D.Pose.__file__),
                    Path(C.D.F.__file__), Path(C.D.G.__file__), Path(C.D.M.__file__),
                    C.ROOT / 'scripts/research/pallet_dim_conditioned_p_v1/inference.py',
                    C.ROOT / 'scripts/research/pallet_line_pose_v1/paper_evaluation.py',
                    C.ROOT / 'challenge/evaluation_v2/paper_real_eval.py',
                    C.ROOT / 'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',
                    C.ROOT / 'scripts/paper/pose_metric_closure_v1/symmetry_aware_pose_metrics.py']
    C.save(destination, dict(created_at=C.now(), prediction_lock=C.bind(C.DOC / 'PREDICTIONS_LOCK.json'),
        files=[C.bind(path), C.bind(membership)], codes=[C.bind(p) for p in source_codes],
        models=model_names(), frames=len(rows), references_read=False,
        selector=dict(selection_lock=selector_lock,checkpoint=selector['checkpoint'],winner='GEO_LINEAR'),
        geometry_inputs=geometry_inputs,geometry_provenance=C.bind(legacy_sources_path),
        contract='Unchanged frozen GEO_LINEAR and original corner8 SQPnP/LM per W/D candidate; physical C2 metrics after freeze.',
        protocol=lock['protocol'], wall_seconds=time.monotonic()-start,
        aggregation_contract=dict(bootstrap_repeats=BOOT_REPEATS,bootstrap_seed=BOOT_SEED,
            nonregression_ratio=GUARD_RATIO,paired_seed_order=list(C.SEEDS),
            median='Conditional per-seed population median followed by seed mean; failures separately guarded.',
            failure_guard='Everyseed failure count cannot exceed paired reference; full-population extended-real distributions also retained.'),
        image_forwards=0, fits=0, optimizer_updates=0))


def scalar(value):
    return dict(value=float(value) if np.isfinite(value) else None,
                status='FINITE' if np.isfinite(value) else 'UNDEFINED' if np.isnan(value) else 'POSITIVE_INFINITY' if value > 0 else 'NEGATIVE_INFINITY')


def error_tensor(metrics, names, ids):
    """Unavailable predictions stay in the tensor as +inf, never zero."""
    values = np.array([[[metrics[a][i][k] if metrics[a][i]['available'] else np.inf for k in METRICS]
                       for i in ids] for a in names], dtype=float)
    available = np.array([[metrics[a][i]['available'] for i in ids] for a in names], dtype=bool)
    assert np.isfinite(values[available]).all(), 'Available pose has nonfinite T/R; reject invalid metric artifact'
    assert (values[available] >= 0).all(), 'Available pose has negative error'
    return values


def validate_tensor(values):
    values = np.asarray(values)
    assert values.ndim == 3 and values.shape[-1] == len(METRICS)
    assert not np.isnan(values).any(), 'NaN cannot silently remove a frame from metrics'
    assert not np.isneginf(values).any() and (values >= 0).all()
    assert np.array_equal(np.isposinf(values[:, :, 0]), np.isposinf(values[:, :, 1])), 'Pose failure must apply to both T and R'


def seed_quantiles(values, quantile=.5, conditional=True):
    validate_tensor(values)
    out = []
    for seed in values:
        selected = seed[np.isfinite(seed).all(1)] if conditional else seed
        out.append([C.D.M.extended_quantile(selected[:, k], quantile) if len(selected) else np.nan
                    for k in range(len(METRICS))])
    return np.array(out, dtype=float)


def seed_average(values, quantile=.5, conditional=True):
    return seed_quantiles(values, quantile, conditional).mean(0)


def hierarchical_bootstrap(before, after, recordings, repeats=BOOT_REPEATS, seed=BOOT_SEED):
    """Crossed paired bootstrap: shared recording and seed draws for both arms.

    A recording is sampled with all its frames, retaining cluster-size weight.
    Three seed indices are sampled with replacement, retaining paired SINGLE
    versus DIVERSE correspondence. The same recording draw is used for every
    sampled seed and comparison. No frames are independently resampled.
    """
    assert before.shape == after.shape and before.shape[1] == len(recordings)
    rng = np.random.default_rng(seed)
    names = sorted(set(recordings))
    indices = {r: np.flatnonzero(np.asarray(recordings) == r) for r in names}
    samples = []
    for _ in range(repeats):
        selected_seed = rng.integers(0, before.shape[0], before.shape[0])
        selected_group = rng.integers(0, len(names), len(names))
        selected_frame = np.concatenate([indices[names[n]] for n in selected_group])
        b = seed_average(before[selected_seed][:, selected_frame])
        a = seed_average(after[selected_seed][:, selected_frame])
        samples.append(a-b)
    samples = np.asarray(samples)
    point = seed_average(after)-seed_average(before)
    output = dict(repeats=repeats, seed=seed, recording_count=len(names), training_seed_count=before.shape[0],
        estimand='Difference of mean across-seed conditional population medians; after minus before.',
        paired_recording_and_training_seed=True, same_draws_all_comparisons=True,
        resampling='Crossed cluster/seed bootstrap; whole recordings sampled with replacement; all frames kept; paired seed index sampled with replacement.',
        metrics={})
    for j, key in enumerate(METRICS):
        valid = np.isfinite(samples[:, j])
        finite = samples[valid, j]
        interval = np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None]
        output['metrics'][key] = dict(point_estimate=scalar(point[j]), CI95=interval,
            finite_draws=int(valid.sum()), undefined_draws=int((~valid).sum()),
            status='FINITE_ALL_DRAWS' if valid.all() else 'UNDEFINED_DRAWS_PRESENT',
            upper95_below_zero=bool(valid.all() and interval[1] < 0),
            interval_interpretation='Finite-draw percentile interval; undefined draws explicitly retained in counts and cannot yield gate PASS.')
    return output


def leave_recording_out(before, after, recordings):
    out = {}
    for recording in sorted(set(recordings)):
        keep = np.asarray(recordings) != recording
        delta = seed_average(after[:, keep])-seed_average(before[:, keep])
        out[recording] = dict(frames=int(keep.sum()),
            mean_seed_median_difference={k: scalar(v) for k, v in zip(METRICS, delta)},
            both_negative=bool(np.isfinite(delta).all() and (delta < 0).all()))
    return out


def mean_seed_summary(values):
    out = {}
    for conditional in (True, False):
        mode = 'conditional' if conditional else 'full_population'
        out[mode] = {}
        for label, q in [('median', .5), ('P90', .9)]:
            out[mode][label] = {k: scalar(v) for k, v in zip(METRICS, seed_average(values, q, conditional))}
    out['failure_counts_by_seed'] = np.isinf(values[:, :, 0]).sum(1).tolist()
    out['mean_failure_count'] = float(np.isinf(values[:, :, 0]).sum(1).mean())
    return out


def comparison(before, after, recordings, with_bootstrap=True):
    output = dict(before=mean_seed_summary(before), after=mean_seed_summary(after),
        per_seed_median_difference=[], mean_seed_median_difference={},
        all_seeds_both_medians_smaller=False, LORO=leave_recording_out(before, after, recordings))
    differences = seed_quantiles(after)-seed_quantiles(before)
    output['per_seed_median_difference'] = [{k: scalar(v) for k, v in zip(METRICS, d)} for d in differences]
    output['mean_seed_median_difference'] = {k: scalar(v) for k, v in zip(METRICS, differences.mean(0))}
    output['all_seeds_both_medians_smaller'] = bool(np.isfinite(differences).all() and (differences < 0).all())
    if with_bootstrap:
        output['hierarchical_bootstrap'] = hierarchical_bootstrap(before, after, recordings)
    return output


def guard(before, after, quantiles):
    checks = {}
    for label, q in quantiles:
        b, a = seed_average(before, q), seed_average(after, q)
        for j, key in enumerate(METRICS):
            checks[f'{label}:{key}'] = dict(before=scalar(b[j]), after=scalar(a[j]), ratio_limit=GUARD_RATIO,
                limit=scalar(b[j]*GUARD_RATIO),
                pass_guard=bool(np.isfinite([a[j], b[j]]).all() and a[j] <= GUARD_RATIO*b[j]))
    bf = np.isinf(before[:, :, 0]).sum(1)
    af = np.isinf(after[:, :, 0]).sum(1)
    checks['pose_failures'] = dict(before_by_seed=bf.tolist(), after_by_seed=af.tolist(),
                                  pass_guard=bool((af <= bf).all()), rule='No seed increases full-population failure count.')
    return dict(PASS=all(v['pass_guard'] for v in checks.values()), checks=checks)


def gate_results(metrics, rows, populations, protocol):
    metadata = {r['id']: r for r in rows}
    single, diverse = C.ARMS
    models = {a: [f'{a}_s{s}' for s in C.SEEDS] for a in C.ARMS}
    models.update({a: [a]*len(C.SEEDS) for a in FIXED})
    tensors = {p: {a: error_tensor(metrics, n, ids) for a, n in models.items()} for p, ids in populations.items()}
    hierarchical = {}
    for pop in ('NATURAL99', 'CLEAN29', 'WOOD45'):
        recordings = [metadata[i]['recording'] for i in populations[pop]]
        hierarchical[pop] = {f'{diverse}-minus-{before}': comparison(tensors[pop][before], tensors[pop][diverse], recordings)
                             for before in (single, *FIXED)}
    primary = hierarchical['NATURAL99']
    three = {a: primary[f'{diverse}-minus-{a}']['all_seeds_both_medians_smaller'] for a in (single, *FIXED)}
    uncertainty = {a: all(v['upper95_below_zero'] for v in primary[f'{diverse}-minus-{a}']['hierarchical_bootstrap']['metrics'].values())
                   for a in (single, 'R0')}
    sensitivity = {a: all(v['both_negative'] for v in primary[f'{diverse}-minus-{a}']['LORO'].values()) for a in (single, 'R0')}
    tail = {a: guard(tensors['NATURAL99'][a], tensors['NATURAL99'][diverse], [('P90', .9)]) for a in (single, 'R0')}
    clean = guard(tensors['CLEAN29']['R0'], tensors['CLEAN29'][diverse], [('median', .5), ('P90', .9)])
    gates = dict(all_three_seeds_joint_gain=dict(PASS=all(three.values()), comparisons=three),
        joint_uncertainty=dict(PASS=all(uncertainty.values()), comparisons=uncertainty),
        recording_sensitivity=dict(PASS=all(sensitivity.values()), comparisons=sensitivity),
        natural_tails=dict(PASS=all(v['PASS'] for v in tail.values()), comparisons=tail),
        clean_preservation=clean)
    success = all(v['PASS'] for v in gates.values())
    return dict(PASS=success, verdict='LOCKED_STABILITY_GATES_PASS_ON_REUSED_DEV' if success else 'STABLE_JOINT_IMPROVEMENT_NOT_ESTABLISHED',
        gates=gates, protocol_criteria=protocol['stability_criteria'],
        engineering_tolerance=protocol['stability_criteria']['tolerance_status'],
        strong_generalization=False, goal_complete=False,
        limitations='Repeated DEV; geometry-derived reference; only3trainingseeds/6naturalrecordings. Gate completion is not independent new-recording confirmation.'), hierarchical


def write_csv(path, rows):
    handle = io.StringIO(newline='')
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    C.save(path, handle.getvalue())


def score():
    start = time.monotonic()
    lock, rows, predictions, protocol = locked_inputs()
    pose_lock = C.read(C.DOC / 'POSE_PREDICTIONS_LOCK.json')
    assert pose_lock['prediction_lock'] == C.bind(C.DOC / 'PREDICTIONS_LOCK.json')
    for binding in pose_lock['files'] + pose_lock['codes'] + pose_lock['geometry_inputs'] + [pose_lock['geometry_provenance']] + list(bindings_in(pose_lock['selector'])):
        verify_file(binding)
    results_path = C.DOC / 'RESULTS.json'
    if results_path.exists():
        old = C.read(results_path)
        assert old['prediction_lock'] == C.bind(C.DOC / 'PREDICTIONS_LOCK.json')
        for binding in old['artifacts']:
            verify_file(binding)
        print('RESULTS_ALREADY_FROZEN', flush=True)
        return
    candidates = C.read(C.RAW / 'POSE_CANDIDATES.json')
    populations = C.read(C.RAW / 'EVAL_GROUPS.json')
    scoring = C.DOC / 'SCORING_START.json'
    if not scoring.exists():
        C.save(scoring, dict(created_at=C.now(), prediction_lock=C.bind(C.DOC / 'PREDICTIONS_LOCK.json'),
                            pose_lock=C.bind(C.DOC / 'POSE_PREDICTIONS_LOCK.json'), code=C.bind(Path(__file__))))
    else:
        assert C.read(scoring)['pose_lock'] == C.bind(C.DOC / 'POSE_PREDICTIONS_LOCK.json')
    # Hash reference bytes against historical bindings only after both locks.
    authenticated_references = reference_bindings(protocol, rows)
    reference_path = C.DOC / 'REFERENCE_BINDINGS.json'
    C.save(reference_path, authenticated_references)
    # The first parsed reference opening is below both locks and verification.
    metadata, truth = C.D.O.D.Pose.metadata('REAL_DEV')
    assert set(r['id'] for r in rows) <= set(truth)
    for row in rows:
        k, xyz, source = metadata[row['id']]
        np.testing.assert_allclose(k, row['K'], rtol=0, atol=1e-9)
        np.testing.assert_allclose(xyz, row['xyz'], rtol=0, atol=1e-9)
        assert not source and truth[row['id']]['order'] == 2
    metrics, csv_rows = {}, []
    for model in model_names():
        metrics[model] = {}
        for row in rows:
            fid = row['id']
            record = candidates[model][fid]
            result = C.D.metric(fid, record['GEO_pose'], truth[fid])
            metrics[model][fid] = result
            pred = predictions[model][fid]
            selected = None if pred['selected_index'] is None else pred['candidates'][pred['selected_index']]
            detected = selected is not None
            csv_rows.append(dict(model=model, id=fid, material='PLASTIC' if fid in populations['FULL128'] else 'WOOD',
                recording=row['recording'], severity=row['severity'], detected=detected,
                pose_available=result['available'], pose_status='OK' if result['available'] else 'NO_DETECTION' if not detected else 'POSE_FAILED',
                selected_detection_index=pred['selected_index'], detector_score=selected['score'] if detected else None,
                GEO_name=record['GEO_name'], GEO_fallback=record['GEO_fallback'],
                GEO_changed_from_R0=record['GEO_name'] != candidates['R0'][fid]['GEO_name'],
                translation_cm=result.get('translation_cm'), rotation_deg=result.get('rotation_deg'),
                yaw_deg=result.get('yaw_deg'), axis_correct=result.get('axis_correct'),
                camera_x_signed_cm=result.get('camera_x_signed_cm'), camera_z_signed_cm=result.get('camera_z_signed_cm'),
                ADDsym_normalized=result.get('ADDsym_normalized'), IoU3D=result.get('IoU3D'),
                full_population_error_status='FINITE' if result['available'] else 'POSITIVE_INFINITY'))
        print('SCORED', model, len(metrics[model]), flush=True)
    historical = C.read(C.D.RAW / 'E1_POSE_METRICS.json')
    aliases = dict(R0='identity', PRIOR1='PRIOR1', FULL125='FULL125')
    parity = 0
    for model, old in aliases.items():
        for fid in populations['FULL128']:
            C.D.O.D.close(metrics[model][fid], historical[old][fid])
            parity += 1
    summaries = {p: {a: C.D.summarize(metrics[a][i] for i in ids) for a in model_names()} for p, ids in populations.items()}
    by_recording = {}
    for pop, ids in populations.items():
        by_recording[pop] = {}
        for recording in sorted({r['recording'] for r in rows if r['id'] in ids}):
            selected = [r['id'] for r in rows if r['id'] in ids and r['recording'] == recording]
            by_recording[pop][recording] = dict(frames=len(selected), models={a: C.D.summarize(metrics[a][i] for i in selected) for a in model_names()})
    paired = {}
    for pop, ids in populations.items():
        paired[pop] = {}
        for arm in C.ARMS:
            for seed in C.SEEDS:
                after = f'{arm}_s{seed}'
                before_models = list(FIXED) + ([f'{C.ARMS[0]}_s{seed}'] if arm == C.ARMS[1] else [])
                for before in before_models:
                    paired[pop][f'{after}-minus-{before}'] = C.D.M.paired(metrics[before], metrics[after], ids)
    gates, hierarchical = gate_results(metrics, rows, populations, protocol)
    metrics_path = C.RAW / 'POSE_METRICS.json'
    csv_path = C.RAW / 'FRAME_RESULTS.csv'
    C.save(metrics_path, metrics)
    write_csv(csv_path, csv_rows)
    detail = C.DOC / 'DETAILED_COMPARISONS.json'
    C.save(detail, dict(paired_per_seed=paired, hierarchical=hierarchical, by_recording=by_recording))
    C.save(results_path, dict(complete=True, created_at=C.now(), protocol=lock['protocol'],
        prediction_lock=C.bind(C.DOC / 'PREDICTIONS_LOCK.json'), pose_lock=C.bind(C.DOC / 'POSE_PREDICTIONS_LOCK.json'),
        models=model_names(), populations={k: len(v) for k, v in populations.items()},
        summaries=summaries, stability=gates, hierarchy=hierarchical,
        baseline_metric_parity_checks=parity, full_frame_rows=len(csv_rows),
        row_failure_policy='Failed frame retains row; conditional estimates explicitly conditional, full distribution uses+inf; failure guard checked per seed.',
        schema_note='Each seed is separately reported. Means of seed medians are not pooled-frame medians or medians of paired frame differences.',
        artifacts=[C.bind(metrics_path), C.bind(csv_path), C.bind(detail), C.bind(reference_path)],
        wall_seconds=time.monotonic()-start, image_forwards=0, new_fits=0))
    print('STABILITY', gates['verdict'], {k: v['PASS'] for k, v in gates['gates'].items()}, flush=True)


def self_check():
    """Analytic statistical fixtures, without image data or references."""
    before = np.full((3, 6, 2), [10., 5.])
    after = before - np.array([2., 1.])
    recordings = ['A', 'A', 'B', 'B', 'C', 'C']
    result = hierarchical_bootstrap(before, after, recordings, repeats=40)
    assert result['metrics']['translation_cm']['CI95'] == [-2., -2.]
    assert result['metrics']['rotation_deg']['CI95'] == [-1., -1.]
    assert all(v['both_negative'] for v in leave_recording_out(before, after, recordings).values())
    heterogeneous = before + np.array([0., 100., 1000.])[:, None, None]
    paired = hierarchical_bootstrap(heterogeneous, heterogeneous-1., recordings, repeats=40)
    assert all(v['CI95'] == [-1., -1.] for v in paired['metrics'].values())
    zero = hierarchical_bootstrap(before, before, recordings, repeats=40)
    assert not any(v['upper95_below_zero'] for v in zero['metrics'].values())
    harm = before.copy(); harm[2] += 1
    assert not comparison(before, harm, recordings, with_bootstrap=False)['all_seeds_both_medians_smaller']
    assert guard(before, before*GUARD_RATIO, [('median', .5), ('P90', .9)])['PASS']
    assert not guard(before, before*(GUARD_RATIO+1e-5), [('median', .5)])['PASS']
    bad_nan = after.copy(); bad_nan[:, 0] = np.nan
    try:
        guard(before, bad_nan, [('median', .5)])
    except AssertionError:
        pass
    else:
        raise AssertionError('NaN frame was silently excluded')
    failed = after.copy(); failed[:, 0] = np.inf
    assert not guard(before, failed, [('median', .5)])['PASS']
    assert mean_seed_summary(failed)['failure_counts_by_seed'] == [1, 1, 1]
    assert mean_seed_summary(failed)['full_population']['P90']['translation_cm']['status'] == 'POSITIVE_INFINITY'
    failed[:] = np.inf
    undefined = hierarchical_bootstrap(before, failed, recordings, repeats=40)
    assert all(v['undefined_draws'] == 40 and not v['upper95_below_zero'] for v in undefined['metrics'].values())
    print('EVALUATION_SELF_CHECK_PASS', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'score', 'all', 'self-check'])
    args = parser.parse_args()
    C.D.torch.set_num_threads(1)
    C.D.cv2.setNumThreads(1)
    if args.stage == 'self-check':
        self_check()
    else:
        if args.stage in ('freeze', 'all'):
            freeze()
        if args.stage in ('score', 'all'):
            score()


if __name__ == '__main__':
    main()
