"""Lock all final-checkpoint source VAL routes before reading source labels.

Run ``freeze`` and ``score`` as separate processes.  Neither command fits a
model or reads real evaluation references.  A failed VAL gate authorizes no
real routing.  ``self_check`` uses invented numbers only.
"""
from . import common as C

READS = C.source_guard(allow_source_targets=True)

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


# Importing train's shared prediction functions must not install its fitting
# guard.  This guard remains closed even while verifying the six fit receipts.
_REFERENCE_ACCESS = {'allowed': False}
_SOURCE_REFERENCE_TOKENS = (
    'GEOMETRY_SIDETABLE.npz', '/SOURCE_MANIFEST.json',
    '/DIMENSION_SIDECAR.json', '/SYNTH_RECORDS.json', '/SYNTH_LABELS.npz',
    'SOURCE_VAL_METRICS.npz', 'SOURCE_VAL_GATE.json',
)


def _reference_guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if not _REFERENCE_ACCESS['allowed']:
            assert not any(token in path for token in _SOURCE_REFERENCE_TOKENS), (
                'SOURCE_VAL_REFERENCE_BEFORE_ROUTING_LOCK', path)


sys.addaudithook(_reference_guard)

from . import source_features as F

ARMS = ('R0_ONLY', 'UNION')
SEEDS = (1, 2, 3)
LEARNED = tuple(f'{arm}_s{seed}' for seed in SEEDS for arm in ARMS)
BASELINES = ('R0_GEO',) + tuple(f'DIVERSE251_s{seed}_GEO' for seed in SEEDS)
CHOICES = C.RAW / 'SOURCE_VAL_CHOICES.json'
ROUTING_LOCK = C.DOC / 'SOURCE_VAL_ROUTING_LOCK.json'
GATE_PATH = C.DOC / 'SOURCE_VAL_GATE.json'
TRAIN_PROTOCOL = C.DOC / 'TRAIN_PROTOCOL.json'
VAL_RULE = dict(
    frames=1024, seeds=list(SEEDS), median_strict=True,
    p90_ratio_max=1.05, failure_no_increase=True,
    comparators=['paired_R0_ONLY', 'R0_GEO', 'paired_DIVERSE_GEO'],
    all_seeds_required=True, no_checkpoint_selection=True,
    routing_lock_before_source_labels=True, real_route_requires_gate_pass=True,
    zero_valid_fallback='R0_GEO_if_available_else_failure',
)


def names_for(arm, seed):
    parents = ['R0'] + ([f'DIVERSE251_s{seed}'] if arm == 'UNION' else [])
    assert arm in ARMS and seed in SEEDS
    return [parent + ':' + hypothesis for parent in parents for hypothesis in F.HYP]


def verify_protocol():
    binding = C.read(C.DOC / 'TRAIN_PROTOCOL_SHA.json')
    assert binding['path'] == str(TRAIN_PROTOCOL.relative_to(C.ROOT))
    C.verify(binding)
    protocol = C.read(TRAIN_PROTOCOL)
    code = {binding['path']: binding for binding in protocol['codes']}
    for path in (Path(__file__), Path(C.__file__), C.HERE / 'train.py', Path(F.__file__)):
        assert str(path.resolve().relative_to(C.ROOT)) in code, ('UNBOUND_VAL_OPERATOR', str(path))
    for binding in code.values():
        C.verify(binding)
    for key, expected in VAL_RULE.items():
        assert protocol['source_val'][key] == expected, ('VAL_PROTOCOL', key)
    expected_inputs = dict(
        source_contract=C.DOC / 'SOURCE_CONTRACT.json',
        source_predictions_lock=C.DOC / 'SOURCE_PREDICTIONS_LOCK.json',
        feature_lock=F.FEATURE_LOCK, features=C.RAW / 'SOURCE_FEATURES.npz',
    )
    for key, path in expected_inputs.items():
        assert protocol['inputs'][key] == C.bind(path), ('TRAIN_INPUT_BINDING', key)
    F.verify_all(protocol['inputs'])
    contract = C.read(C.DOC / 'SOURCE_CONTRACT.json')
    assert contract['complete'] and contract['status'] == 'PASS'
    assert contract['fit_eligibility']['eligible_counts']['VAL'] == 1024
    return protocol, contract, C.bind(TRAIN_PROTOCOL)


def verify_training():
    """Require every final fit before selecting any VAL route."""
    protocol, contract, protocol_binding = verify_protocol()
    complete_path = C.DOC / 'TRAINING_COMPLETE.json'
    complete = C.read(complete_path)
    assert complete['complete'] and complete['protocol'] == protocol_binding
    assert complete['fit_count'] == 6 and complete['total_updates'] == 1980
    assert len(complete['fits']) == 6 and 'paired_checks' in complete
    assert complete['all_final_epoch_only'] and complete['source_TRAIN_only']
    assert not complete['VAL_quality_read'] and not complete['real_targets_read']
    assert set(complete['paired_checks']) == {'1', '2', '3'}
    F.verify_all(complete)
    fits = {}
    for binding in complete['fits']:
        receipt = C.read(C.ROOT / binding['path'])
        arm, seed = receipt['arm'], receipt['seed']
        assert arm in ARMS and seed in SEEDS
        name = f'{arm}_s{seed}'
        assert name not in fits
        assert receipt['complete'] and receipt['protocol'] == protocol_binding
        assert receipt['updates'] == 330
        assert receipt['source_TRAIN_only'] and not receipt['VAL_quality_read'] and not receipt['real_targets_read']
        for key in ('initial_state_sha', 'final_state_sha', 'order_sha', 'normalization_sha'):
            assert isinstance(receipt[key], str) and len(receipt[key]) == 64
        expected = C.RAW / 'fits' / name / 'final.pt'
        assert receipt['checkpoint'] == C.bind(expected)
        F.verify_all(receipt)
        start = C.read(C.ROOT / receipt['START']['path'])
        assert start['protocol'] == protocol_binding and start['arm'] == arm and start['seed'] == seed
        assert start['planned_updates'] == 330 and start['rows'] == 2598
        for key in ('initial_state_sha', 'normalization_sha', 'order_sha'):
            assert start[key] == receipt[key] == complete['paired_checks'][str(seed)][key]
        trace = [json.loads(line) for line in (C.ROOT / receipt['trace']['path']).read_text().splitlines()]
        assert len(trace) == 330 and [row['step'] for row in trace] == list(range(1, 331))
        assert trace[-1]['state_sha'] == receipt['final_state_sha']
        assert all(np.isfinite(row['loss']) for row in trace)
        fits[name] = dict(receipt=binding, checkpoint=receipt['checkpoint'])
    assert set(fits) == set(LEARNED)
    return protocol, contract, protocol_binding, complete, fits


def verified_inputs(contract):
    locked = C.read(F.FEATURE_LOCK)
    assert locked['complete'] and not locked['source_targets_read']
    assert not locked['real_targets_read'] and not locked['VAL_quality_scored']
    assert locked['models'] == list(C.MODELS)
    assert locked['metadata'] == C.bind(C.RAW / 'SOURCE_INPUTS.json')
    assert locked['features'] == C.bind(C.RAW / 'SOURCE_FEATURES.npz')
    assert locked['poses'] == C.bind(C.RAW / 'SOURCE_POSES.json')
    assert locked['predictions'] == C.bind(C.DOC / 'SOURCE_PREDICTIONS_LOCK.json')
    prediction_lock = C.read(C.DOC / 'SOURCE_PREDICTIONS_LOCK.json')
    assert prediction_lock['complete'] and prediction_lock['frames'] == 5120
    assert prediction_lock.get('runtime_amendment') == C.protocol().get('runtime_amendment')
    F.verify_all(prediction_lock)
    F.verify_all(locked)
    inputs = C.read(C.RAW / 'SOURCE_INPUTS.json')
    assert len(inputs) == len({row['id'] for row in inputs}) == 5120
    positions = [j for j, row in enumerate(inputs) if row['split'] == 'VAL']
    rows = [inputs[j] for j in positions]
    assert len(rows) == len({row['id'] for row in rows}) == 1024
    assert {row['id'] for row in rows} == set(contract['fit_eligibility']['eligible_ids']['VAL'])
    assert set(contract['fit_eligibility']['eligible_ids']['TRAIN']).isdisjoint(row['id'] for row in rows)
    poses = C.read(C.RAW / 'SOURCE_POSES.json')
    assert poses['ids'] == [row['id'] for row in inputs]
    assert poses['models'] == list(C.MODELS) and not poses['source_targets_read']
    return locked, inputs, positions, rows, poses


def chosen_pose(choice, records, fid):
    if choice['fallback']:
        assert choice['candidate_index'] == -1 and choice['candidate_name'] is None
        return records['R0'][fid]['GEO_pose']
    parent, hypothesis = choice['candidate_name'].split(':')
    assert parent == choice['parent'] and hypothesis == choice['hypothesis']
    found = [h['pose'] for h in records[parent][fid]['hypotheses'] if h['name'] == hypothesis]
    assert len(found) == 1 and found[0]['available']
    return found[0]


def operator_bindings():
    from . import train as T
    return [C.bind(Path(module.__file__)) for module in (sys.modules[__name__], C, F, T, F.G)]


def verify_routing():
    protocol, contract, protocol_binding, complete, fits = verify_training()
    feature_lock, inputs, positions, rows, poses = verified_inputs(contract)
    locked = C.read(ROUTING_LOCK)
    assert locked['complete'] and locked['frames'] == 1024
    assert locked['models'] == list(LEARNED) and locked['baselines'] == list(BASELINES)
    assert locked['protocol'] == protocol_binding
    assert locked['training_complete'] == C.bind(C.DOC / 'TRAINING_COMPLETE.json')
    assert locked['feature_lock'] == C.bind(F.FEATURE_LOCK)
    assert locked['source_predictions_lock'] == C.bind(C.DOC / 'SOURCE_PREDICTIONS_LOCK.json')
    assert locked.get('runtime_amendment') == C.protocol().get('runtime_amendment')
    assert locked['source_contract'] == C.bind(C.DOC / 'SOURCE_CONTRACT.json')
    assert locked['choices'] == C.bind(CHOICES)
    assert locked['fits'] == fits and not locked['source_labels_read']
    assert not locked['real_references_read'] and locked['VAL_quality_scored'] is False
    assert locked['source_VAL_label_values_read'] is False
    assert locked['source_TRAIN_labels_sha_verified'] is True
    assert locked['operators'] == operator_bindings()
    F.verify_all(locked)
    choices = C.read(CHOICES)
    assert choices['ids'] == [row['id'] for row in rows]
    assert choices['models'] == list(LEARNED)
    assert set(choices['records']) == set(LEARNED)
    for model in LEARNED:
        assert set(choices['records'][model]) == set(choices['ids'])
    return protocol, locked, rows, poses, choices


def freeze():
    """Compute final-checkpoint choices; source-reference guard stays closed."""
    start = time.monotonic()
    if ROUTING_LOCK.exists():
        verify_routing()
        print('SOURCE_VAL_ROUTING_ALREADY_LOCKED', flush=True)
        return
    from . import train as T
    protocol, contract, protocol_binding, complete, fits = verify_training()
    feature_lock, inputs, positions, rows, poses = verified_inputs(contract)
    with np.load(C.RAW / 'SOURCE_FEATURES.npz') as stored:
        assert stored['ids'].tolist() == [row['id'] for row in inputs]
        assert stored['split'].tolist() == [row['split'] for row in inputs]
        assert stored['hypothesis_names'].tolist() == list(F.HYP)
        geo = {model: stored[model + '_geo'][positions] for model in C.MODELS}
        valid = {model: stored[model + '_valid'][positions] for model in C.MODELS}
    records, counts = {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            model = f'{arm}_s{seed}'
            checkpoint = torch.load(C.ROOT / fits[model]['checkpoint']['path'], map_location='cpu', weights_only=False)
            names = names_for(arm, seed)
            assert checkpoint['variant'] == 'GEO_LINEAR' and checkpoint['d'] == 94
            assert checkpoint['arm'] == arm and checkpoint['seed'] == seed
            assert checkpoint['protocol'] == protocol_binding
            assert checkpoint['candidate_names'] == names
            receipt = C.read(C.ROOT / fits[model]['receipt']['path'])
            assert T.state_sha(checkpoint['state']) == receipt['final_state_sha']
            assert checkpoint['epochs'] == 30 and checkpoint['updates'] == 330
            assert checkpoint['selection'] == 'final_epoch_only'
            assert checkpoint['normalization_sha'] == receipt['normalization_sha']
            assert T.array_sha(np.stack([checkpoint['mean'], checkpoint['std']])) == receipt['normalization_sha']
            parents = ['R0'] + ([f'DIVERSE251_s{seed}'] if arm == 'UNION' else [])
            features = np.concatenate([geo[parent] for parent in parents], axis=1)
            usable = np.concatenate([valid[parent] for parent in parents], axis=1)
            assert features.shape == (1024, len(names), 94)
            assert usable.shape == (1024, len(names))
            assert np.isfinite(features[usable]).all()
            scores = np.asarray(T.score_candidates(checkpoint, features, usable), float)
            assert scores.shape == usable.shape and np.isfinite(scores[usable]).all()
            selected = np.asarray(T.select_candidates(scores, usable, names), int)
            assert selected.shape == (1024,)
            records[model] = {}
            tally = Counter()
            for j, row in enumerate(rows):
                index = int(selected[j])
                assert -1 <= index < len(names)
                assert index >= 0 or not usable[j].any()
                if index >= 0:
                    assert usable[j, index]
                    name = names[index]
                    parent, hypothesis = name.split(':')
                else:
                    name, parent, hypothesis = None, 'R0', None
                choice = dict(candidate_index=index, candidate_name=name, parent=parent,
                              hypothesis=hypothesis, fallback=index < 0,
                              scores=[float(s) if flag else None for s, flag in zip(scores[j], usable[j])],
                              candidate_valid=usable[j].tolist())
                pose = chosen_pose(choice, poses['records'], row['id'])
                choice['pose_available'] = bool(pose['available'])
                choice['status'] = 'SELECTED' if index >= 0 else 'R0_GEO_FALLBACK' if pose['available'] else 'FAILED'
                records[model][row['id']] = choice
                tally[choice['status']] += 1
                tally['parent_' + parent] += 1
            counts[model] = dict(tally)
    C.save(CHOICES, dict(ids=[row['id'] for row in rows], models=list(LEARNED),
                        baseline_pose_fields={name: dict(parent=name[:-4], field='GEO_pose') for name in BASELINES},
                        records=records, source_labels_read=False, real_references_read=False))
    C.save(ROUTING_LOCK, dict(complete=True, created_at=C.now(), frames=1024,
           models=list(LEARNED), baselines=list(BASELINES), choices=C.bind(CHOICES),
           protocol=protocol_binding, training_complete=C.bind(C.DOC / 'TRAINING_COMPLETE.json'),
           fits=fits, feature_lock=C.bind(F.FEATURE_LOCK), source_contract=C.bind(C.DOC / 'SOURCE_CONTRACT.json'),
           source_predictions_lock=C.bind(C.DOC / 'SOURCE_PREDICTIONS_LOCK.json'),
           runtime_amendment=C.protocol().get('runtime_amendment'),
           metadata=C.bind(C.RAW / 'SOURCE_INPUTS.json'), poses=C.bind(C.RAW / 'SOURCE_POSES.json'),
           operators=operator_bindings(), counts=counts, source_labels_read=False,
           source_VAL_label_values_read=False, source_TRAIN_labels_sha_verified=True,
           source_label_access_disclosure='No source VAL label values read. The existing TRAIN label container is hashed only to verify sealed training inputs; its values are not used for routing.',
           real_references_read=False, VAL_quality_scored=False, fits_executed=0,
           image_forwards=0, read_paths=sorted(set(READS)), wall_seconds=time.monotonic() - start))
    print('SOURCE_VAL_ROUTING_LOCKED', counts, flush=True)


def verify_source_reference_chain():
    """Use immutable contract expectations, never a newly trusted split lock."""
    assert _REFERENCE_ACCESS['allowed']
    contract = C.read(C.DOC / 'SOURCE_CONTRACT.json')
    history_path = C.ROOT / '_docs/experiments/pallet_selector_recovery_v1/stage2_synth_scorer/SYNTHETIC_SPLIT_LOCK.json'
    geometry = C.ROOT / 'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
    historical_rows = C.SOURCE / 'SYNTH_RECORDS.json'
    expected = {}
    for path in (history_path, geometry, historical_rows):
        relative = str(path.relative_to(C.ROOT))
        found = [binding for binding in F.bindings(contract) if binding['path'] == relative]
        assert found, ('UNBOUND_SOURCE_REFERENCE', relative)
        assert all(binding == found[0] for binding in found), ('CONFLICTING_SOURCE_REFERENCE_BINDINGS', relative)
        expected[relative] = found[0]
    history_binding = expected[str(history_path.relative_to(C.ROOT))]
    C.verify(history_binding)
    history = C.read(history_path)
    references = []
    for path in (geometry, historical_rows):
        relative = str(path.relative_to(C.ROOT))
        matches = [binding for binding in F.bindings(history) if binding['path'] == relative]
        assert len(matches) == 1 and matches[0] == expected[relative]
        C.verify(expected[relative])
        references.append(expected[relative])
    return history_binding, references


def score():
    """Only frozen routes enter VAL labels; no model or threshold selection."""
    start = time.monotonic()
    protocol, route_lock, rows, poses, choices = verify_routing()
    # Permit quality/reference files only after all routing bindings are valid.
    _REFERENCE_ACCESS['allowed'] = True
    reference_start = C.now()
    history_binding, references = verify_source_reference_chain()
    if GATE_PATH.exists():
        done = C.read(GATE_PATH)
        assert done['complete'] and done['routing_lock'] == C.bind(ROUTING_LOCK)
        assert done['protocol'] == C.bind(TRAIN_PROTOCOL)
        assert done['source_history_binding'] == history_binding
        assert done['source_reference_bindings'] == references
        F.verify_all(done)
        print('SOURCE_VAL_GATE_ALREADY_COMPLETE', done['status'], flush=True)
        return
    geometry = C.ROOT / 'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
    historical_rows_path = C.SOURCE / 'SYNTH_RECORDS.json'
    ids = [row['id'] for row in rows]
    lookup = {row['id']: row for row in C.read(historical_rows_path) if row['id'] in set(ids)}
    assert set(lookup) == set(ids) and all(row['split'] == 'VAL' for row in lookup.values())
    indices = np.array([lookup[fid]['table_index'] for fid in ids], np.int64)
    with np.load(geometry) as stored:
        assert stored['stems'][indices].tolist() == ids
        rotations, translations = stored['R'][indices], stored['t'][indices]
        cameras, dimensions = stored['K'][indices], stored['dims'][indices]
    for j, row in enumerate(rows):
        fx, fy, cx, cy = cameras[j]
        np.testing.assert_allclose(row['K'], [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], atol=1e-7, rtol=0)
        np.testing.assert_allclose(row['dims'], dimensions[j], atol=1e-7, rtol=0)
    errors = {model: np.full((1024, 2), np.inf) for model in LEARNED + BASELINES}
    for j, fid in enumerate(ids):
        reference_R = rotations[j] @ F.S
        for model in LEARNED:
            choice = choices['records'][model][fid]
            pose = chosen_pose(choice, poses['records'], fid)
            assert bool(pose['available']) == choice['pose_available']
            errors[model][j] = F.error_pair(pose, reference_R, translations[j])
        for model in BASELINES:
            errors[model][j] = F.error_pair(poses['records'][model[:-4]][fid]['GEO_pose'], reference_R, translations[j])
    arrays = dict(ids=np.asarray(ids), models=np.asarray(LEARNED + BASELINES),
                  metric_names=np.asarray(F.KEYS), source_table_index=indices)
    for model, values in errors.items():
        assert values.shape == (1024, 2) and not np.isnan(values).any()
        arrays[model] = values
    metric_path = C.RAW / 'SOURCE_VAL_METRICS.npz'
    F.save_npz(metric_path, **arrays)
    comparisons, failures = {}, []
    for seed in SEEDS:
        model = f'UNION_s{seed}'
        comparisons[model] = {}
        for baseline in (f'R0_ONLY_s{seed}', 'R0_GEO', f'DIVERSE251_s{seed}_GEO'):
            comparison = F.gate(errors[baseline], errors[model], VAL_RULE['p90_ratio_max'])
            assert len(comparison['checks']) == 5
            comparisons[model][baseline] = comparison
            failures.extend(f'{model}/{baseline}/{key}' for key, value in comparison['checks'].items() if not value)
    passed = not failures
    result = dict(complete=True, created_at=C.now(), scope='SOURCE_DECLARED_C2_PROPER_RIGID_VAL_ONLY',
           frames=1024, models=list(LEARNED), baselines=list(BASELINES), seeds=list(SEEDS),
           PASS=passed, status='PASS_SOURCE_VAL_JOINT' if passed else 'SOURCE_RANKING_NO_JOINT_SIGNAL',
           real_routing_authorized=passed, real_reference_accessed=False,
           gate_rule=VAL_RULE, comparisons=comparisons, failed_checks=failures,
           summaries={model: F.summary(values) for model, values in errors.items()},
           routing_lock=C.bind(ROUTING_LOCK), protocol=C.bind(TRAIN_PROTOCOL),
           training_complete=C.bind(C.DOC / 'TRAINING_COMPLETE.json'),
           source_contract=C.bind(C.DOC / 'SOURCE_CONTRACT.json'),
           metrics=C.bind(metric_path), source_reference_bindings=references,
           source_history_binding=history_binding,
           source_reference_read_time=reference_start,
           full_population_failure_policy='All 1024 rows retained; failed poses are +infinity in both T and R. JSON null has POSITIVE_INFINITY or NA_EMPTY status; metrics NPZ preserves infinity.',
           reference_basis='R_reference=renderer.R@diag(1,-1,-1), t=renderer.t, C2 full-rotation geodesic; source_features.error_pair unchanged.',
           label_container_disclosure='Geometry NPZ contains broader source rows; only VAL1024 indices are retained and scored here. No TEST or real references are scored.',
           selection_policy='All six final checkpoints and three paired UNION gates required; no seed, checkpoint, feature, threshold, or baseline selection.',
           failure_action='On any failed comparison, do not generate new real routes or evaluate this method on real references. No fallback winner is selected.',
           fits_executed=0, image_forwards=0, read_paths=sorted(set(READS)),
           wall_seconds=time.monotonic() - start)
    C.save(GATE_PATH, result)
    print('SOURCE_VAL_GATE', result['status'], 'failed_checks', len(failures), flush=True)


def self_check():
    """Meaningful math/failure checks with invented arrays, no artifacts."""
    from . import train as T
    assert T.READS is None, 'Importing shared scorer must not activate fitting guards.'
    good = np.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.]])
    baseline = good * 2
    assert F.gate(baseline, good, 1.05)['PASS']
    assert not F.gate(good, good, 1.05)['PASS']
    failure = good.copy()
    failure[-1] = np.inf
    checked = F.gate(baseline, failure, 1.05)
    assert not checked['PASS'] and not checked['checks']['failure_no_increase']
    assert checked['after']['frames'] == 4 and checked['after']['failed_pose'] == 1
    assert checked['after']['full_population']['translation_cm']['P90_status'] == 'POSITIVE_INFINITY'
    assert F.quantile([1., np.inf], .5) == np.inf
    assert F.summary(np.full((4, 2), np.inf))['failed_pose'] == 4
    pose = dict(available=True, R_physical=np.eye(3).tolist(), centroid=[0., 0., 1.])
    np.testing.assert_allclose(F.error_pair(pose, np.eye(3), np.array([0., 0., 1.])), [0., 0.])
    np.testing.assert_allclose(F.error_pair(pose, np.diag([-1., 1., -1.]), np.array([0., 0., 1.])), [0., 0.])
    np.testing.assert_allclose(F.error_pair(pose, np.eye(3), np.array([.01, 0., 1.])), [1., 0.])
    assert np.isposinf(F.error_pair(dict(available=False), np.eye(3), np.zeros(3))).all()
    assert len(LEARNED) == 6 and len(BASELINES) == 4
    assert len(names_for('R0_ONLY', 1)) == 2 and len(names_for('UNION', 3)) == 4
    # A known linear function tests actual shared scorer/mask/tie semantics,
    # rather than merely comparing one helper to its own output.
    weight = torch.zeros((1, 94), dtype=torch.float32)
    weight[0, 0] = 1.
    checkpoint = dict(variant='GEO_LINEAR', d=94,
                      state={'net.weight': weight, 'net.bias': torch.tensor([7.])},
                      mean=np.zeros(94, np.float32), std=np.ones(94, np.float32))
    features = np.zeros((3, 4, 94), np.float32)
    features[0, :, 0] = [2., 1., -1., -1.]
    mask = np.array([[True] * 4, [False] * 4, [False, True, True, False]])
    names = names_for('UNION', 1)
    scores = T.score_candidates(checkpoint, features, mask)
    np.testing.assert_array_equal(scores[0], [9., 8., 6., 6.])
    assert np.isposinf(scores[~mask]).all()
    np.testing.assert_array_equal(T.select_candidates(scores, mask, names), [2, -1, 1])
    permutation = np.array([3, 2, 1, 0])
    permuted = T.select_candidates(scores[:, permutation], mask[:, permutation], [names[j] for j in permutation])
    np.testing.assert_array_equal(np.where(permuted >= 0, permutation[np.maximum(permuted, 0)], -1), [2, -1, 1])
    fallback_pose = {'available': False}
    fixture = {'R0': {'invented': {'GEO_pose': fallback_pose}}}
    assert chosen_pose(dict(fallback=True, candidate_index=-1, candidate_name=None), fixture, 'invented') is fallback_pose
    assert not _REFERENCE_ACCESS['allowed']
    for path in ('/tmp/val_guard/GEOMETRY_SIDETABLE.npz', '/tmp/val_guard/data/evaluation/fixture.json',
                 '/tmp/val_guard/SOURCE_VAL_METRICS.npz', '/tmp/val_guard/SOURCE_VAL_GATE.json'):
        try:
            with open(path, 'rb'):
                pass
        except AssertionError:
            continue
        raise AssertionError(('REFERENCE_GUARD_NOT_ACTIVE', path))
    print('SOURCE_VAL_SELF_CHECK_PASS_NO_DATA_READ', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'score', 'self_check'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    F.O.D.cv2.setNumThreads(1)
    globals()[args.stage]()


if __name__ == '__main__':
    main()
