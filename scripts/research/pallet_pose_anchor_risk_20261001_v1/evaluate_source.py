"""Freeze four TRAIN-risk-margin source VAL routes before reading source labels.

Run freeze and score in separate processes. No real GT, image inference,
PnP fitting, or optimizer update is performed. self_check uses invented data.
"""
from . import common as C

READS = C.U.source_guard(allow_source_targets=True)

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

_REFERENCE_ACCESS = {'allowed': False}
_SOURCE_REFERENCE_TOKENS = (
    'GEOMETRY_SIDETABLE.npz', '/SOURCE_MANIFEST.json',
    '/DIMENSION_SIDECAR.json', '/SYNTH_RECORDS.json', '/SYNTH_LABELS.npz',
    'SOURCE_VAL_METRICS.npz', 'SOURCE_VAL_GATE.json', 'VAL_ORACLE.json',
)


def _reference_guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if not _REFERENCE_ACCESS['allowed']:
            assert not any(token in path for token in _SOURCE_REFERENCE_TOKENS), (
                'SOURCE_VAL_REFERENCE_BEFORE_ROUTING_LOCK', path)


sys.addaudithook(_reference_guard)

from scripts.research.pallet_pose_union_selection_20261001_v1 import source_features as F

SEEDS = (1, 2, 3)
LEARNED = C.MODEL_NAMES
BASELINES = ('R0_GEO',) + tuple(f'DIVERSE251_s{seed}_GEO' for seed in SEEDS)
CHOICES = C.RAW / 'SOURCE_VAL_CHOICES.json'
ROUTING_LOCK = C.DOC / 'SOURCE_VAL_ROUTING_LOCK.json'
GATE_PATH = C.DOC / 'SOURCE_VAL_GATE.json'
PROTOCOL = C.DOC / 'TRAIN_PROTOCOL.json'
VAL_RULE = dict(
    frames=1024, seeds=list(SEEDS), median_strict=True,
    p90_ratio_max=1.05, failure_no_increase=True,
    comparators=['R0_ONLY', 'R0_GEO', 'paired_DIVERSE_GEO'],
    all_seeds_required=True, no_checkpoint_selection=True,
    routing_lock_before_source_labels=True, real_route_requires_gate_pass=True,
    zero_valid_fallback='R0_GEO_if_available_else_failure',
)


def parents_for(model):
    assert model in LEARNED
    return ['R0'] + ([] if model == 'R0_ONLY' else [f'DIVERSE251_s{int(model[-1])}'])


def names_for(model):
    return [parent + ':' + hypothesis for parent in parents_for(model) for hypothesis in F.HYP]


def operator_bindings():
    from . import convex_train as T
    return [C.bind(Path(module.__file__)) for module in (sys.modules[__name__], C, C.U, F, T)]


def verify_protocol():
    protocol = C.protocol('TRAIN_PROTOCOL')
    bound_codes = {binding['path']: binding for binding in protocol['codes']}
    for binding in operator_bindings():
        assert bound_codes.get(binding['path']) == binding, ('UNBOUND_VAL_OPERATOR', binding['path'])
    for key, expected in VAL_RULE.items():
        assert protocol['source_val'][key] == expected, ('VAL_PROTOCOL', key)
    assert protocol['target_rule'] == 'R0_GEO_PARETO_NONREGRESSION_FIXED_TRAIN_MINMAX'
    assert protocol['lambda_l2'] == 1e-4 and protocol['bias'] == 0
    assert protocol['normalization'] == 'old_float32_then_float64'
    assert protocol['feature_dim'] == 189 and protocol['raw_feature_dim'] == 94
    assert protocol['feature_map'] == 'normalized94_abs_anchor_delta94_identity1'
    assert protocol['loss_rule'] == 'TRAIN_LOG1P_ANCHOR_EXCESS_MARGIN_CE'
    assert protocol['runtime_uses_margin'] is False
    inputs = list(F.bindings(protocol['inputs']))
    for path in (C.PARENT_DOC / 'SOURCE_CONTRACT.json', F.FEATURE_LOCK,
                 C.PARENT_RAW / 'SOURCE_FEATURES.npz', C.PARENT_RAW / 'SOURCE_POSES.json',
                 C.PARENT_DOC / 'SOURCE_PREDICTIONS_LOCK.json'):
        assert C.bind(path) in inputs, ('UNBOUND_FROZEN_SOURCE_INPUT', str(path))
    contract = C.read(C.PARENT_DOC / 'SOURCE_CONTRACT.json')
    assert contract['complete'] and contract['status'] == 'PASS'
    assert contract['fit_eligibility']['eligible_counts'] == {'TRAIN': 2598, 'VAL': 1024}
    return protocol, contract, C.bind(PROTOCOL)


def verify_certificate(value):
    assert value['PASS'] and value['optimizer_success']
    gradient = float(value['gradient_l2'])
    gap = float(value['gradient_l2_squared_over_2lambda'])
    assert np.isfinite([gradient, gap]).all() and gradient >= 0 and gap >= 0
    assert value['max_gap_upper_bound'] == 1e-6 and gap <= 1e-6
    assert value['lambda_l2'] == 1e-4
    assert 0 < value['objective_calls'] <= 2000 and 0 <= value['iterations'] <= 1000
    np.testing.assert_allclose(gap, gradient * gradient / (2 * 1e-4), rtol=1e-12, atol=1e-18)


def verify_training():
    """All four certified final fits must exist before any VAL choice."""
    from . import convex_train as T
    protocol, contract, protocol_binding = verify_protocol()
    complete = C.read(C.DOC / 'TRAINING_COMPLETE.json')
    assert complete['complete'] and complete['protocol'] == protocol_binding
    assert complete['models'] == list(LEARNED)
    assert complete['fit_count'] == len(LEARNED) == 4 and complete['all_certified']
    assert complete['source_TRAIN_only'] and not complete['VAL_quality_read'] and not complete['real_targets_read']
    assert set(complete['margin_sha_by_model']) == set(LEARNED)
    assert set(complete['risk_sha_by_model']) == set(LEARNED)
    for key in ('feature_map', 'feature_dim', 'raw_feature_dim', 'loss_rule', 'runtime_uses_margin'):
        assert complete[key] == protocol[key]
    assert len(complete['fits']) == 4
    F.verify_all(complete)
    fits = {}
    for receipt_binding in complete['fits']:
        receipt = C.read(C.ROOT / receipt_binding['path'])
        model = receipt['model']
        assert model in LEARNED and model not in fits
        assert receipt['complete'] and receipt['protocol'] == protocol_binding
        assert receipt['source_TRAIN_only'] and not receipt['VAL_quality_read'] and not receipt['real_targets_read']
        assert receipt['checkpoint'] == C.bind(C.RAW / 'fits' / model / 'final.json')
        assert receipt_binding == C.bind(C.DOC / f'FIT_{model}.json')
        F.verify_all(receipt)
        verify_certificate(receipt['certificate'])
        checkpoint = C.read(C.ROOT / receipt['checkpoint']['path'])
        start = C.read(C.ROOT / receipt['START']['path'])
        assert start['model'] == model and start['protocol'] == protocol_binding
        for key in ('feature_map', 'feature_dim', 'raw_feature_dim', 'loss_rule', 'runtime_uses_margin'):
            assert checkpoint[key] == receipt[key] == start[key] == complete[key] == protocol[key]
        assert checkpoint['anchor_index_sha'] == receipt['anchor_index_sha'] == start['anchor_index_sha'] == complete['anchor_index_sha']
        assert checkpoint['margin_sha'] == receipt['margin_sha'] == start['margin_sha'] == complete['margin_sha_by_model'][model]
        assert checkpoint['risk_sha'] == receipt['risk_sha'] == start['risk_sha'] == complete['risk_sha_by_model'][model]
        assert checkpoint['schema'] == 'pallet_pose_anchor_risk_linear189_v1'
        assert checkpoint['target_rule'] == protocol['target_rule']
        assert checkpoint['feature_map'] == protocol['feature_map']
        assert checkpoint['feature_dim'] == 189 and checkpoint['raw_feature_dim'] == 94
        assert checkpoint['model'] == model and checkpoint['names'] == names_for(model)
        assert checkpoint['protocol'] == protocol_binding
        assert checkpoint['bias'] == 0 and checkpoint['lambda_l2'] == 1e-4
        assert checkpoint['certificate'] == receipt['certificate']
        assert checkpoint['normalization_sha'] == receipt['normalization_sha']
        assert checkpoint['normalization'] == 'old_float32_then_float64'
        weight = np.asarray(checkpoint['weight'], np.float64)
        mean, std = np.asarray(checkpoint['mean'], np.float32), np.asarray(checkpoint['std'], np.float32)
        assert weight.shape == (189,) and mean.shape == std.shape == (94,)
        assert np.isfinite(weight).all() and np.isfinite(mean).all() and np.isfinite(std).all()
        assert (std >= np.float32(1e-6)).all()
        assert T.OLD.array_sha(np.stack([mean, std])) == receipt['normalization_sha']
        assert T.OLD.array_sha(weight) == receipt['final_weight_sha']
        trace = [json.loads(line) for line in (C.ROOT / receipt['trace']['path']).read_text().splitlines()]
        calls = [row for row in trace if row['event'] == 'objective']
        assert [row['call'] for row in calls] == list(range(1, receipt['objective_calls'] + 1))
        assert receipt['objective_calls'] == checkpoint['certificate']['objective_calls']
        assert receipt['iterations'] == checkpoint['certificate']['iterations']
        assert calls[-1]['weight_sha'] == receipt['final_weight_sha']
        assert calls[-1]['objective'] == checkpoint['certificate']['objective_value']
        for row in trace:
            assert row['feature_map'] == protocol['feature_map']
            assert row['anchor_index_sha'] == complete['anchor_index_sha']
            assert row['loss_rule'] == protocol['loss_rule'] and row['margin_sha'] == checkpoint['margin_sha']
        fits[model] = dict(receipt=receipt_binding, checkpoint=receipt['checkpoint'])
    assert set(fits) == set(LEARNED)
    receipts = [C.read(C.ROOT / fits[model]['receipt']['path']) for model in LEARNED]
    assert len({receipt['normalization_sha'] for receipt in receipts}) == 1
    assert complete['total_objective_calls'] == sum(receipt['objective_calls'] for receipt in receipts)
    return protocol, contract, protocol_binding, complete, fits


def verified_inputs(contract):
    locked = C.read(F.FEATURE_LOCK)
    assert locked['complete'] and not locked['source_targets_read']
    assert not locked['real_targets_read'] and not locked['VAL_quality_scored']
    assert locked['models'] == list(C.U.MODELS)
    assert locked['metadata'] == C.bind(C.PARENT_RAW / 'SOURCE_INPUTS.json')
    assert locked['features'] == C.bind(C.PARENT_RAW / 'SOURCE_FEATURES.npz')
    assert locked['poses'] == C.bind(C.PARENT_RAW / 'SOURCE_POSES.json')
    assert locked['predictions'] == C.bind(C.PARENT_DOC / 'SOURCE_PREDICTIONS_LOCK.json')
    prediction_lock = C.read(C.PARENT_DOC / 'SOURCE_PREDICTIONS_LOCK.json')
    assert prediction_lock['complete'] and prediction_lock['frames'] == 5120
    assert prediction_lock.get('runtime_amendment') == C.U.protocol().get('runtime_amendment')
    F.verify_all(prediction_lock)
    F.verify_all(locked)
    inputs = C.read(C.PARENT_RAW / 'SOURCE_INPUTS.json')
    assert len(inputs) == len({row['id'] for row in inputs}) == 5120
    assert Counter(row['split'] for row in inputs) == {'TRAIN': 4096, 'VAL': 1024}
    positions = [j for j, row in enumerate(inputs) if row['split'] == 'VAL']
    rows = [inputs[j] for j in positions]
    assert len(rows) == len({row['id'] for row in rows}) == 1024
    assert {row['id'] for row in rows} == set(contract['fit_eligibility']['eligible_ids']['VAL'])
    assert set(contract['fit_eligibility']['eligible_ids']['TRAIN']).isdisjoint(row['id'] for row in rows)
    poses = C.read(C.PARENT_RAW / 'SOURCE_POSES.json')
    assert poses['ids'] == [row['id'] for row in inputs]
    assert poses['models'] == list(C.U.MODELS) and not poses['source_targets_read']
    assert not poses['real_targets_read'] and poses['hypothesis_names'] == list(F.HYP)
    return locked, inputs, positions, rows, poses


def verify_anchor_indices(index, rows, poses, r0_valid):
    """Cross-check the input-only R0 GEO identity against its frozen pose."""
    index = np.asarray(index)
    assert index.dtype == np.int64 and index.shape == (len(rows),)
    assert np.isin(index, [-1, 0, 1]).all() and r0_valid.shape == (len(rows), 2)
    for j, row in enumerate(rows):
        record = poses['records']['R0'][row['id']]
        assert bool(record['GEO_pose']['available']) == (index[j] >= 0)
        if index[j] < 0:
            assert not r0_valid[j].any()
            continue
        assert r0_valid[j, index[j]]
        assert record['GEO_name'] == F.HYP[index[j]]
        hypothesis = [h for h in record['hypotheses'] if h['name'] == record['GEO_name']]
        assert len(hypothesis) == 1 and hypothesis[0]['pose'] == record['GEO_pose']
    return index


def chosen_pose(choice, records, fid):
    if choice['fallback']:
        assert choice['candidate_index'] == -1 and choice['candidate_name'] is None
        return records['R0'][fid]['GEO_pose']
    parent, hypothesis = choice['candidate_name'].split(':')
    assert parent == choice['parent'] and hypothesis == choice['hypothesis']
    found = [h['pose'] for h in records[parent][fid]['hypotheses'] if h['name'] == hypothesis]
    assert len(found) == 1 and found[0]['available']
    return found[0]


def verify_routing():
    protocol, contract, protocol_binding, complete, fits = verify_training()
    feature_lock, inputs, positions, rows, poses = verified_inputs(contract)
    locked = C.read(ROUTING_LOCK)
    assert locked['complete'] and locked['frames'] == 1024
    assert locked['models'] == list(LEARNED) and locked['baselines'] == list(BASELINES)
    expected = dict(protocol=protocol_binding, training_complete=C.bind(C.DOC / 'TRAINING_COMPLETE.json'),
                    feature_lock=C.bind(F.FEATURE_LOCK), source_predictions_lock=C.bind(C.PARENT_DOC / 'SOURCE_PREDICTIONS_LOCK.json'),
                    source_contract=C.bind(C.PARENT_DOC / 'SOURCE_CONTRACT.json'), choices=C.bind(CHOICES),
                    metadata=C.bind(C.PARENT_RAW / 'SOURCE_INPUTS.json'), poses=C.bind(C.PARENT_RAW / 'SOURCE_POSES.json'))
    for key, binding in expected.items():
        assert locked[key] == binding, ('ROUTE_BINDING', key)
    assert locked.get('runtime_amendment') == C.U.protocol().get('runtime_amendment')
    assert locked['fits'] == fits and not locked['source_labels_read']
    assert not locked['real_references_read'] and not locked['VAL_quality_scored']
    assert locked['source_VAL_label_values_read'] is False
    assert locked['source_TRAIN_labels_sha_verified'] is True
    assert locked['operators'] == operator_bindings()
    assert locked['feature_map'] == protocol['feature_map']
    assert locked['loss_rule'] == protocol['loss_rule'] and locked['runtime_uses_margin'] is False
    assert locked['raw_feature_dim'] == 94 and locked['feature_dim'] == 189
    F.verify_all(locked)
    choices = C.read(CHOICES)
    assert choices['ids'] == [row['id'] for row in rows]
    assert choices['models'] == list(LEARNED) and set(choices['records']) == set(LEARNED)
    assert not choices['source_labels_read'] and not choices['real_references_read']
    assert choices['feature_map'] == protocol['feature_map']
    assert choices['loss_rule'] == protocol['loss_rule'] and choices['runtime_uses_margin'] is False
    with np.load(C.PARENT_RAW / 'SOURCE_FEATURES.npz') as stored:
        anchor_index = verify_anchor_indices(stored['R0_GEO_index'][positions], rows, poses,
                                             stored['R0_valid'][positions])
    from . import convex_train as T
    assert locked['anchor_index_sha'] == choices['anchor_index_sha'] == T.OLD.array_sha(anchor_index)
    for model in LEARNED:
        assert set(choices['records'][model]) == set(choices['ids'])
        for j, fid in enumerate(choices['ids']):
            record = choices['records'][model][fid]
            assert record['anchor_index'] == int(anchor_index[j])
            assert record['anchor_name'] == (None if anchor_index[j] < 0 else names_for(model)[int(anchor_index[j])])
    return protocol, locked, rows, poses, choices


def freeze():
    """Inference on cached numeric features only; no source reference values."""
    start = time.monotonic()
    assert not _REFERENCE_ACCESS['allowed'], 'Run freeze and score as separate processes.'
    if ROUTING_LOCK.exists():
        verify_routing()
        print('ANCHOR_RISK_SOURCE_VAL_ROUTING_ALREADY_LOCKED', flush=True)
        return
    from . import convex_train as T
    protocol, contract, protocol_binding, complete, fits = verify_training()
    feature_lock, inputs, positions, rows, poses = verified_inputs(contract)
    with np.load(C.PARENT_RAW / 'SOURCE_FEATURES.npz') as stored:
        assert stored['ids'].tolist() == [row['id'] for row in inputs]
        assert stored['split'].tolist() == [row['split'] for row in inputs]
        assert stored['hypothesis_names'].tolist() == list(F.HYP)
        geo = {model: stored[model + '_geo'][positions] for model in C.U.MODELS}
        valid = {model: stored[model + '_valid'][positions] for model in C.U.MODELS}
        anchor_index = stored['R0_GEO_index'][positions]
    anchor_index = verify_anchor_indices(anchor_index, rows, poses, valid['R0'])
    records, counts = {}, {}
    for model in LEARNED:
        checkpoint = C.read(C.ROOT / fits[model]['checkpoint']['path'])
        names = names_for(model)
        features = np.concatenate([geo[parent] for parent in parents_for(model)], axis=1)
        usable = np.concatenate([valid[parent] for parent in parents_for(model)], axis=1)
        assert features.dtype == np.float32 and usable.dtype == bool
        assert features.shape == (1024, len(names), 94) and usable.shape == (1024, len(names))
        assert np.isfinite(features[usable]).all()
        scores = np.asarray(T.score_candidates(checkpoint, features, usable, anchor_index), np.float64)
        assert scores.shape == usable.shape and np.isfinite(scores[usable]).all()
        assert np.isposinf(scores[~usable]).all()
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
            choice = dict(anchor_index=int(anchor_index[j]),
                          anchor_name=None if anchor_index[j] < 0 else names[int(anchor_index[j])],
                          candidate_index=index, candidate_name=name, parent=parent,
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
                        records=records, feature_map=protocol['feature_map'],
                        loss_rule=protocol['loss_rule'], runtime_uses_margin=False,
                        anchor_index_sha=T.OLD.array_sha(anchor_index),
                        source_labels_read=False, real_references_read=False))
    C.save(ROUTING_LOCK, dict(complete=True, created_at=C.now(), frames=1024,
           models=list(LEARNED), baselines=list(BASELINES), choices=C.bind(CHOICES),
           protocol=protocol_binding, training_complete=C.bind(C.DOC / 'TRAINING_COMPLETE.json'),
           fits=fits, feature_lock=C.bind(F.FEATURE_LOCK), source_contract=C.bind(C.PARENT_DOC / 'SOURCE_CONTRACT.json'),
           source_predictions_lock=C.bind(C.PARENT_DOC / 'SOURCE_PREDICTIONS_LOCK.json'),
           runtime_amendment=C.U.protocol().get('runtime_amendment'),
           metadata=C.bind(C.PARENT_RAW / 'SOURCE_INPUTS.json'), poses=C.bind(C.PARENT_RAW / 'SOURCE_POSES.json'),
           operators=operator_bindings(), counts=counts, source_labels_read=False,
           feature_map=protocol['feature_map'], raw_feature_dim=94, feature_dim=189,
           loss_rule=protocol['loss_rule'], runtime_uses_margin=False,
           anchor_rule='Frozen input-only R0 operational GEO candidate; unchanged by learned R0_ONLY.',
           anchor_index_sha=T.OLD.array_sha(anchor_index),
           source_VAL_label_values_read=False, source_TRAIN_labels_sha_verified=True,
           source_label_access_disclosure='No source VAL label values read. Existing TRAIN label containers are hashed only to verify sealed training provenance.',
           real_references_read=False, VAL_quality_scored=False, fits_executed=0,
           image_forwards=0, new_PnP_solves=0, read_paths=sorted(set(READS)),
           wall_seconds=time.monotonic() - start))
    print('ANCHOR_RISK_SOURCE_VAL_ROUTING_LOCKED', counts, flush=True)


def verify_source_reference_chain():
    assert _REFERENCE_ACCESS['allowed']
    contract = C.read(C.PARENT_DOC / 'SOURCE_CONTRACT.json')
    history_path = C.ROOT / '_docs/experiments/pallet_selector_recovery_v1/stage2_synth_scorer/SYNTHETIC_SPLIT_LOCK.json'
    geometry = C.ROOT / 'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
    historical_rows = C.U.SOURCE / 'SYNTH_RECORDS.json'
    expected = {}
    for path in (history_path, geometry, historical_rows):
        relative = str(path.relative_to(C.ROOT))
        found = [binding for binding in F.bindings(contract) if binding['path'] == relative]
        assert found and all(binding == found[0] for binding in found), ('UNBOUND_OR_CONFLICTING_REFERENCE', relative)
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


def save_npz(path, **arrays):
    path = Path(path).resolve()
    assert path.is_relative_to(C.RAW)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)


def comparisons_for(errors):
    comparisons, failures = {}, []
    for seed in SEEDS:
        model = f'UNION_s{seed}'
        comparisons[model] = {}
        for baseline in ('R0_ONLY', 'R0_GEO', f'DIVERSE251_s{seed}_GEO'):
            comparison = F.gate(errors[baseline], errors[model], VAL_RULE['p90_ratio_max'])
            assert len(comparison['checks']) == 5
            comparisons[model][baseline] = comparison
            failures.extend(f'{model}/{baseline}/{key}' for key, value in comparison['checks'].items() if not value)
    assert sum(len(v['checks']) for comp in comparisons.values() for v in comp.values()) == 45
    return comparisons, failures


def score():
    start = time.monotonic()
    protocol, route_lock, rows, poses, choices = verify_routing()
    _REFERENCE_ACCESS['allowed'] = True
    reference_start = C.now()
    history_binding, references = verify_source_reference_chain()
    if GATE_PATH.exists():
        done = C.read(GATE_PATH)
        assert done['complete'] and done['routing_lock'] == C.bind(ROUTING_LOCK)
        assert done['protocol'] == C.bind(PROTOCOL)
        assert done['source_history_binding'] == history_binding and done['source_reference_bindings'] == references
        F.verify_all(done)
        print('ANCHOR_RISK_SOURCE_VAL_GATE_ALREADY_COMPLETE', done['status'], flush=True)
        return
    geometry = C.ROOT / 'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
    ids = [row['id'] for row in rows]
    id_set = set(ids)
    lookup = {row['id']: row for row in C.read(C.U.SOURCE / 'SYNTH_RECORDS.json') if row['id'] in id_set}
    assert set(lookup) == id_set and all(row['split'] == 'VAL' for row in lookup.values())
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
    save_npz(metric_path, **arrays)
    comparisons, failures = comparisons_for(errors)
    passed = not failures
    result = dict(complete=True, created_at=C.now(), scope='SOURCE_DECLARED_C2_PROPER_RIGID_VAL_ONLY',
           frames=1024, models=list(LEARNED), baselines=list(BASELINES), seeds=list(SEEDS),
           PASS=passed, status='PASS_SOURCE_VAL_JOINT' if passed else 'SOURCE_RANKING_NO_JOINT_SIGNAL',
           real_routing_authorized=passed, real_reference_accessed=False,
           gate_rule=VAL_RULE, checks_total=45, checks_passed=45-len(failures),
           comparisons=comparisons, failed_checks=failures,
           summaries={model: F.summary(values) for model, values in errors.items()},
           routing_lock=C.bind(ROUTING_LOCK), protocol=C.bind(PROTOCOL),
           training_complete=C.bind(C.DOC / 'TRAINING_COMPLETE.json'),
           source_contract=C.bind(C.PARENT_DOC / 'SOURCE_CONTRACT.json'),
           metrics=C.bind(metric_path), source_reference_bindings=references, source_history_binding=history_binding,
           source_reference_read_time=reference_start,
           full_population_failure_policy='All1024 rows retained; failed poses +infinity in both T/R. JSON null has POSITIVE_INFINITY or NA_EMPTY status; NPZ preserves infinity.',
           reference_basis='R_reference=renderer.R@diag(1,-1,-1), t=renderer.t, C2 full-rotation geodesic; unchanged source_features.error_pair.',
           label_container_disclosure='Source geometry container includes other source rows; only VAL1024 indices are scored. No TEST or real references scored.',
           selection_policy='Four certified final models; shared R0_ONLY; all3 frozen refiner seeds and all45 comparisons required. No best-seed or threshold selection.',
           failure_action='Any failed comparison prohibits new real routing/evaluation for this method; no fallback winner promotion.',
           fits_executed=0, image_forwards=0, new_PnP_solves=0, read_paths=sorted(set(READS)),
           wall_seconds=time.monotonic() - start)
    C.save(GATE_PATH, result)
    print('ANCHOR_RISK_SOURCE_VAL_GATE', result['status'], 'failed_checks', len(failures), flush=True)


def self_check():
    good = np.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.]])
    errors = {model: good.copy() for model in LEARNED + BASELINES}
    for model in ('R0_ONLY',) + BASELINES:
        errors[model] *= 2
    comparisons, failures = comparisons_for(errors)
    assert not failures and len(comparisons) == 3
    errors['UNION_s3'] = errors['R0_ONLY'].copy()
    _, failures = comparisons_for(errors)
    assert len(failures) == 6 and all('/UNION_s' not in v for v in failures)
    assert all(v.startswith('UNION_s3/') and v.endswith('_median_strict') for v in failures)
    errors['UNION_s3'][-1] = np.inf
    comparisons, failures = comparisons_for(errors)
    assert not comparisons['UNION_s3']['R0_ONLY']['checks']['failure_no_increase']
    assert comparisons['UNION_s3']['R0_ONLY']['after']['frames'] == 4
    assert F.quantile([1., np.inf], .5) == np.inf
    pose = dict(available=True, R_physical=np.eye(3).tolist(), centroid=[0., 0., 1.])
    np.testing.assert_array_equal(F.error_pair(pose, np.eye(3), np.array([0., 0., 1.])), [0., 0.])
    np.testing.assert_array_equal(F.error_pair(pose, np.diag([-1., 1., -1.]), np.array([.01, 0., 1.])), [1., 0.])
    assert np.isposinf(F.error_pair(dict(available=False), np.eye(3), np.zeros(3))).all()
    verify_certificate(dict(PASS=True, optimizer_success=True, gradient_l2=1e-6,
                            gradient_l2_squared_over_2lambda=5e-9, max_gap_upper_bound=1e-6,
                            lambda_l2=1e-4, objective_calls=1, iterations=0))
    assert len(names_for('R0_ONLY')) == 2 and len(names_for('UNION_s3')) == 4
    assert len(LEARNED) == len(BASELINES) == 4
    from . import convex_train as T
    assert T.OLD.READS is None, 'Shared scorer import must not activate fitting-only hooks.'
    weight = np.zeros(189, np.float64)
    weight[0] = 1
    checkpoint = dict(schema='pallet_pose_anchor_risk_linear189_v1', bias=0., lambda_l2=1e-4,
                      weight=weight, mean=np.zeros(94, np.float32), std=np.ones(94, np.float32),
                      feature_map='normalized94_abs_anchor_delta94_identity1', feature_dim=189, raw_feature_dim=94,
                      names=names_for('UNION_s1'), normalization='old_float32_then_float64',
                      loss_rule='TRAIN_LOG1P_ANCHOR_EXCESS_MARGIN_CE', runtime_uses_margin=False)
    features = np.zeros((3, 4, 94), np.float32)
    features[0, :, 0] = [2., 1., -1., -1.]
    mask = np.array([[True] * 4, [False] * 4, [False, True, True, False]])
    names = names_for('UNION_s1')
    anchor_index = np.array([0, -1, 1], np.int64)
    scores = T.score_candidates(checkpoint, features, mask, anchor_index)
    np.testing.assert_array_equal(scores[0], [2., 1., -1., -1.])
    assert scores.dtype == np.float64 and np.isposinf(scores[~mask]).all()
    np.testing.assert_array_equal(T.select_candidates(scores, mask, names), [2, -1, 1])
    permutation = np.array([3, 2, 1, 0])
    permuted = T.select_candidates(scores[:, permutation], mask[:, permutation], [names[j] for j in permutation])
    np.testing.assert_array_equal(np.where(permuted >= 0, permutation[np.maximum(permuted, 0)], -1), [2, -1, 1])
    # Stored float32 std floor must not be re-tested against a stricter FP64 literal.
    checkpoint['std'] = np.full(94, np.float32(1e-6)).tolist()
    np.testing.assert_array_equal(T.score_candidates(checkpoint, np.zeros_like(features), mask, anchor_index)[mask], 0.)
    for path in ('/tmp/convergence_guard/GEOMETRY_SIDETABLE.npz',
                 '/tmp/convergence_guard/data/evaluation/fixture.json',
                 '/tmp/convergence_guard/SOURCE_VAL_METRICS.npz',
                 '/tmp/convergence_guard/SOURCE_VAL_GATE.json'):
        try:
            with open(path, 'rb'):
                pass
        except AssertionError:
            continue
        raise AssertionError(('REFERENCE_GUARD_NOT_ACTIVE', path))
    assert not _REFERENCE_ACCESS['allowed']
    print('ANCHOR_RISK_SOURCE_VAL_SELF_CHECK_PASS_NO_DATA_READ', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'score', 'self_check'])
    args = parser.parse_args()
    F.O.D.cv2.setNumThreads(1)
    globals()[args.stage]()


if __name__ == '__main__':
    main()
