"""Freeze four signed-axis-regression source VAL routes before reading source labels.

Run freeze and score in separate processes. No real GT, image inference,
PnP fitting, or optimizer update is performed. self_check uses invented data.
"""
from . import common as C
from . import direction_features as D

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
MODEL_HASH_KEYS = ('signed_target_sha', 'input_difference_sha', 'base_context_sha',
                   'errors_sha', 'scaled_excess_sha', 'original_valid_sha',
                   'direction_raw_sha', 'direction_difference_sha', 'extended_input_sha')
METHOD_FIELDS = dict(
    solver_rule='BLOCK_GENERALIZED_NEWTON_ARMIJO',
    sign_rule='NONZERO_SIGNED_TARGET_AXES', sign_coefficient=1.,
    target_rule='SIGNED_LOG1P_NORMALIZED_TR_ANCHOR_EXCESS',
    input_rule='RBF253_DIFFERENCE_PLUS_NORMALIZED_DIRECTION18_DIFFERENCE',
    loss_rule='FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_HUBER_PLUS_SIGN_LOGISTIC',
    prediction_rule='MAX_TWO_SIGNED_LOG1P_AXES',
    feature_map='normalized94_abs_anchor_delta94_identity1_fixed_rbf64_signed_direction18',
    feature_dim=271, base_feature_dim=253, direction_dim=18,
    direction_rule=D.RULE, direction_normalization=D.NORMALIZATION, raw_feature_dim=94, context_dim=189, rbf_dim=64, output_dim=2, huber_delta=1.,
    runtime_uses_margin=False, runtime_uses_reference_errors=False, runtime_safe_mask=False,
)
SOLVER_RULE = dict(method='block_generalized_newton_armijo', maxiter=1000, maxfun=2000,
    gradient_linf_tolerance=1e-8, initial_alpha=1., backtrack_factor=.5, armijo_c1=1e-4,
    initialization='zeros', damping=0., huber_kink_curvature=0.)
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
    return [C.bind(Path(module.__file__)) for module in (sys.modules[__name__], C, C.U, F, T, T.B, D, D.O)]


def verify_protocol():
    from . import convex_train as T
    protocol = C.protocol('TRAIN_PROTOCOL')
    bound_codes = {binding['path']: binding for binding in protocol['codes']}
    for binding in operator_bindings():
        assert bound_codes.get(binding['path']) == binding, ('UNBOUND_VAL_OPERATOR', binding['path'])
    for key, expected in VAL_RULE.items():
        assert protocol['source_val'][key] == expected, ('VAL_PROTOCOL', key)
    for key, expected in METHOD_FIELDS.items():
        assert protocol[key] == expected, ('SIGNED_AXES_DIRECTION_PROTOCOL',key)
    assert protocol['solver'] == T.solver_config() == SOLVER_RULE
    assert protocol['certificate'] == dict(optimizer_success=True, gap_upper_bound_max=1e-6, gradient_linf_max=1e-8)
    assert protocol['lambda_l2'] == 1e-4 and protocol['bias'] == 0
    assert protocol['normalization'] == 'old_float32_then_float64'
    assert protocol['feature_dim'] == 271 and protocol['raw_feature_dim'] == 94
    assert protocol['feature_map'] == 'normalized94_abs_anchor_delta94_identity1_fixed_rbf64_signed_direction18'
    assert protocol['inputs']['rbf_basis'] == C.bind(C.RBF_DOC / 'RBF_BASIS.json')
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
    assert value['solver_rule'] == METHOD_FIELDS['solver_rule']
    assert value['sign_rule'] == METHOD_FIELDS['sign_rule'] and value['sign_coefficient'] == 1.
    gradient_linf = float(value['gradient_linf'])
    assert np.isfinite(gradient_linf) and 0 <= gradient_linf <= 1e-8
    assert value['gradient_linf_max'] == 1e-8
    assert 1 <= value['final_accepted_call'] <= value['objective_calls']
    gradient = float(value['gradient_l2'])
    gap = float(value['gradient_l2_squared_over_2lambda'])
    assert np.isfinite([gradient, gap]).all() and gradient >= 0 and gap >= 0
    assert value['max_gap_upper_bound'] == 1e-6 and gap <= 1e-6
    assert value['lambda_l2'] == 1e-4
    assert 0 < value['objective_calls'] <= 2000 and 0 <= value['iterations'] <= 1000
    np.testing.assert_allclose(gap, gradient * gradient / (2 * 1e-4), rtol=1e-12, atol=1e-18)


def verify_newton_trace(trace, start, receipt, checkpoint):
    """Check accepted-state provenance without reading TRAIN targets or inputs."""
    calls = []
    iterations = []
    accepted = None
    pending_iteration = False
    trial_alpha = 1.
    for row in trace:
        assert np.isfinite([row['objective'], row['Huber'], row['Sign_logistic'], row['gradient_l2'],
                            row['gradient_linf']]).all()
        assert min(row['objective'], row['Huber'], row['Sign_logistic'], row['gradient_l2'], row['gradient_linf']) >= 0
        np.testing.assert_allclose(row['gradient_gap_upper_bound'], row['gradient_l2']**2/(2e-4), rtol=1e-12, atol=1e-18)
        if row['event'] == 'objective':
            assert not pending_iteration, 'Accepted trial must be recorded before another objective call.'
            assert row['call'] == len(calls) + 1 and row['call'] <= 2000
            assert row['evaluated'] is True
            assert np.isfinite(row['L2_penalty']) and row['L2_penalty'] >= 0
            assert abs(row['objective'] - row['Huber'] - row['Sign_logistic'] - row['L2_penalty']) <= 1e-12
            if not calls:
                assert row['phase'] == 'initial' and row['call'] == 1
                assert row['iteration'] == 0
                assert row['weight_sha'] == start['initial_weight_sha']
                assert row['armijo_accepted'] is True
                assert row['alpha'] == 0. and row['accepted_point_call_before'] is None
                for key in ('parent_weight_sha', 'parent_objective', 'directional_derivative', 'armijo_bound'):
                    assert row[key] is None
                accepted = row
            else:
                assert row['phase'] == 'trial'
                assert row['iteration'] == len(iterations) + 1
                assert row['accepted_point_call_before'] == accepted['call']
                assert row['parent_weight_sha'] == accepted['weight_sha']
                assert row['parent_objective'] == accepted['objective']
                assert row['alpha'] == trial_alpha
                assert np.isfinite(row['armijo_bound'])
                assert np.isfinite(row['directional_derivative']) and row['directional_derivative'] < 0.
                assert row['armijo_bound'] == accepted['objective'] + 1e-4 * row['alpha'] * row['directional_derivative']
                # The mathematically strict decrease may round to equality at
                # a nearly converged point; use the exact declared FP64 rule.
                assert row['armijo_bound'] <= accepted['objective']
                assert row['armijo_accepted'] == (row['objective'] <= row['armijo_bound'])
                if row['armijo_accepted']:
                    accepted = row
                    pending_iteration = True
                    trial_alpha = 1.
                else:
                    trial_alpha *= .5
                    assert trial_alpha > 0., 'No accepted fit may contain line-search underflow.'
            calls.append(row)
        else:
            assert row['event'] == 'iteration' and pending_iteration
            assert row['iteration'] == len(iterations) + 1 and row['iteration'] <= 1000
            assert row['objective_calls'] == len(calls) == accepted['call']
            assert row['accepted_call'] == accepted['call'] and row['armijo_accepted'] is True
            for key in ('weight_sha', 'objective', 'Huber', 'Sign_logistic', 'gradient_l2', 'gradient_linf',
                        'alpha', 'armijo_bound', 'parent_weight_sha', 'parent_objective',
                        'directional_derivative', 'L2_penalty', 'gradient_gap_upper_bound'):
                assert row[key] == accepted[key]
            iterations.append(row)
            pending_iteration = False
    assert calls and not pending_iteration and accepted is calls[-1]
    assert len(calls) == receipt['objective_calls'] == checkpoint['certificate']['objective_calls']
    assert len(iterations) == receipt['iterations'] == checkpoint['certificate']['iterations']
    assert receipt['fits_executed'] == checkpoint['fits_executed'] == 1
    assert receipt['optimizer_steps'] == checkpoint['optimizer_steps'] == len(iterations)
    for value in (receipt, checkpoint, checkpoint['certificate'], checkpoint['solver']):
        assert value['final_accepted_call'] == accepted['call']
    assert accepted['weight_sha'] == receipt['final_weight_sha']
    assert accepted['objective'] == checkpoint['certificate']['objective_value']
    assert accepted['Huber'] == checkpoint['certificate']['Huber']
    assert accepted['Sign_logistic'] == checkpoint['certificate']['Sign_logistic']
    assert accepted['gradient_l2'] == checkpoint['certificate']['gradient_l2']
    assert accepted['gradient_linf'] == checkpoint['certificate']['gradient_linf']
    return calls, iterations


def verify_training():
    """All four certified final fits must exist before any VAL choice."""
    from . import convex_train as T
    protocol, contract, protocol_binding = verify_protocol()
    complete = C.read(C.DOC / 'TRAINING_COMPLETE.json')
    assert complete['complete'] and complete['protocol'] == protocol_binding
    assert complete['models'] == list(LEARNED)
    assert complete['fit_count'] == len(LEARNED) == 4 and complete['all_certified']
    assert complete['max_objective_calls_per_fit'] == 2000
    assert {p.name for p in (C.RAW / 'fits').iterdir()} == set(LEARNED)
    for model in LEARNED:
        assert not (C.RAW / 'fits' / model / 'FAILED.json').exists()
        assert not (C.DOC / f'REJECTED_{model}.json').exists()
    assert complete['source_TRAIN_only'] and not complete['VAL_quality_read'] and not complete['real_targets_read']
    for key in MODEL_HASH_KEYS:
        assert set(complete[key+'_by_model']) == set(LEARNED)
    assert set(complete['final_accepted_call_by_model']) == set(LEARNED)
    for key in METHOD_FIELDS:
        assert complete[key] == protocol[key]
    assert len(complete['fits']) == 4
    assert complete['basis_SHA_bind'] == protocol['inputs']['rbf_basis']
    basis_artifact = C.read(C.ROOT / complete['basis_SHA_bind']['path'])
    assert basis_artifact['complete'] and basis_artifact['PASS']
    prefit = C.read(C.ROOT/protocol['inputs']['prefit_review']['path'])
    assert prefit['complete'] and prefit['PASS']
    F.verify_all(complete)
    D.verify_training_audit(protocol_binding)
    norm_binding, norm_sha, _, _ = D.normalization_receipt(protocol)
    assert complete['direction_receipt_binding'] == norm_binding and complete['direction_normalization_sha'] == norm_sha
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
        direction_metadata = D.verify_checkpoint(checkpoint, protocol)
        for value in (receipt, start, complete):
            for key, expected in direction_metadata.items(): assert value[key] == expected
        for key in METHOD_FIELDS:
            assert checkpoint[key] == receipt[key] == start[key] == complete[key] == protocol[key]
        assert checkpoint['anchor_index_sha'] == receipt['anchor_index_sha'] == start['anchor_index_sha'] == complete['anchor_index_sha']
        for key in MODEL_HASH_KEYS:
            assert checkpoint[key] == receipt[key] == start[key] == complete[key+'_by_model'][model] == prefit['models'][model][key]
        assert checkpoint['schema'] == 'pallet_pose_signed_axes_direction_linear271x2_v1'
        assert checkpoint['rbf_basis'] == basis_artifact['basis']
        assert checkpoint['rbf_basis_binding'] == checkpoint['basis_SHA_bind'] == receipt['basis_SHA_bind'] == start['basis_SHA_bind'] == complete['basis_SHA_bind']
        T.B.validate_basis(checkpoint['rbf_basis'], checkpoint['mean'], checkpoint['std'])
        assert checkpoint['target_rule'] == protocol['target_rule']
        assert checkpoint['feature_map'] == protocol['feature_map']
        assert checkpoint['feature_dim'] == 271 and checkpoint['raw_feature_dim'] == 94
        assert checkpoint['model'] == model and checkpoint['names'] == names_for(model)
        assert checkpoint['protocol'] == protocol_binding
        assert checkpoint['bias'] == 0 and checkpoint['lambda_l2'] == 1e-4
        assert checkpoint['solver']['success'] is True
        assert checkpoint['solver']['status'] == 'CONVERGED'
        assert checkpoint['solver']['configuration'] == protocol['solver']
        assert checkpoint['solver']['last_evaluated_call'] == receipt['objective_calls']
        assert checkpoint['final_accepted_call'] == complete['final_accepted_call_by_model'][model]
        assert start['solver'] == protocol['solver'] and start['certificate_rule'] == protocol['certificate']
        assert start['initialization'] == 'all_zero_float64'
        assert start['initial_weight_sha'] == T.OLD.array_sha(np.zeros((271, 2), np.float64))
        assert checkpoint['certificate'] == receipt['certificate']
        assert checkpoint['certificate']['loss_rule'] == METHOD_FIELDS['loss_rule']
        assert checkpoint['certificate']['huber_delta'] == 1.
        assert checkpoint['certificate']['output_dim'] == 2
        values=[float(checkpoint['certificate'][key]) for key in ('objective_value','Huber','Sign_logistic','L2_penalty')]
        assert np.isfinite(values).all() and min(values)>=0 and abs(values[0]-sum(values[1:]))<=1e-12
        assert checkpoint['normalization_sha'] == receipt['normalization_sha']
        assert checkpoint['normalization'] == 'old_float32_then_float64'
        weight = np.asarray(checkpoint['weight'], np.float64)
        mean, std = np.asarray(checkpoint['mean'], np.float32), np.asarray(checkpoint['std'], np.float32)
        assert weight.shape == (271, 2) and mean.shape == std.shape == (94,)
        assert np.isfinite(weight).all() and np.isfinite(mean).all() and np.isfinite(std).all()
        assert (std >= np.float32(1e-6)).all()
        assert T.OLD.array_sha(np.stack([mean, std])) == receipt['normalization_sha']
        assert T.OLD.array_sha(weight) == receipt['final_weight_sha']
        trace = [json.loads(line) for line in (C.ROOT / receipt['trace']['path']).read_text().splitlines()]
        calls, iterations = verify_newton_trace(trace, start, receipt, checkpoint)
        for row in trace:
            for key, expected in direction_metadata.items(): assert row[key] == expected
            for key in METHOD_FIELDS:
                assert row[key] == protocol[key]
            assert row['feature_map'] == protocol['feature_map']
            assert row['basis_SHA_bind'] == complete['basis_SHA_bind']
            assert row['anchor_index_sha'] == complete['anchor_index_sha']
            assert row['loss_rule'] == protocol['loss_rule']
            for key in MODEL_HASH_KEYS:
                assert row[key] == checkpoint[key]
        fits[model] = dict(receipt=receipt_binding, checkpoint=receipt['checkpoint'])
    assert set(fits) == set(LEARNED)
    receipts = [C.read(C.ROOT / fits[model]['receipt']['path']) for model in LEARNED]
    assert len({receipt['normalization_sha'] for receipt in receipts}) == 1
    assert complete['total_objective_calls'] == sum(receipt['objective_calls'] for receipt in receipts)
    assert complete['total_iterations'] == sum(receipt['iterations'] for receipt in receipts)
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


def direction_input_bindings(protocol):
    return dict(features=C.bind(C.PARENT_RAW/'SOURCE_FEATURES.npz'),
        feature_lock=C.bind(F.FEATURE_LOCK), poses=C.bind(C.PARENT_RAW/'SOURCE_POSES.json'),
        metadata=C.bind(C.PARENT_RAW/'SOURCE_INPUTS.json'),
        predictions_lock=C.bind(C.PARENT_DOC/'SOURCE_PREDICTIONS_LOCK.json'),
        training_complete=C.bind(C.DOC/'TRAINING_COMPLETE.json'),
        training_verification=C.bind(C.DOC/'TRAIN_CONVERGENCE.json'),
        direction_receipt=protocol['inputs']['direction_receipt'])


def freeze_directions(protocol, positions, rows, poses, features, valid, anchor, feature_lock):
    """Read only the fixed VAL prediction files after independent TRAIN PASS."""
    prediction_lock=C.read(C.PARENT_DOC/'SOURCE_PREDICTIONS_LOCK.json')
    assert prediction_lock['complete'] and prediction_lock['frames']==5120
    source_protocol=C.read(C.ROOT/prediction_lock['protocol']['path'])
    C.verify(prediction_lock['protocol'])
    receipts={}
    for model in C.U.MODELS:
        binding=prediction_lock['receipts'][model]
        C.verify(binding)
        receipt=C.read(C.ROOT/binding['path'])
        assert receipt['complete'] and receipt['model']==model and receipt['frames']==5120
        assert receipt['protocol']==prediction_lock['protocol']
        assert receipt['checkpoint']==source_protocol['checkpoints'][model]
        assert len(receipt['files'])==5120
        receipts[model]=receipt
    def prediction_for(model, i, row):
        index=positions[i]
        binding=receipts[model]['files'][index]
        expected=C.PARENT_RAW/'source_predictions'/model/f'{index:05d}.json'
        assert binding['path']==str(expected.relative_to(C.ROOT))
        C.verify(binding)
        saved=C.read(expected)
        assert saved['id']==row['id'] and saved['model']==model
        assert saved['protocol_sha']==prediction_lock['protocol']['sha256']
        assert saved['checkpoint_sha']==source_protocol['checkpoints'][model]['sha256']
        assert row['split']=='VAL'
        return saved['prediction'],binding
    return D.freeze_inputs('SOURCE_VAL',rows,positions,anchor,features,valid,poses['records'],
        feature_lock['feature_names'],prediction_for,C.bind(PROTOCOL),protocol,direction_input_bindings(protocol))


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
    for key in METHOD_FIELDS:
        assert locked[key] == protocol[key], ('LOCKED_SIGNED_AXES_DIRECTION_METHOD',key)
    assert locked['basis_SHA_bind'] == protocol['inputs']['rbf_basis']
    assert locked['raw_feature_dim'] == 94 and locked['feature_dim'] == 271
    F.verify_all(locked)
    choices = C.read(CHOICES)
    assert choices['ids'] == [row['id'] for row in rows]
    assert choices['models'] == list(LEARNED) and set(choices['records']) == set(LEARNED)
    assert not choices['source_labels_read'] and not choices['real_references_read']
    assert choices['feature_map'] == protocol['feature_map']
    assert choices['loss_rule'] == protocol['loss_rule'] and choices['runtime_uses_margin'] is False
    for key in METHOD_FIELDS:
        assert choices[key] == protocol[key]
    assert choices['basis_SHA_bind'] == protocol['inputs']['rbf_basis']
    with np.load(C.PARENT_RAW / 'SOURCE_FEATURES.npz') as stored:
        anchor_index = verify_anchor_indices(stored['R0_GEO_index'][positions], rows, poses,
                                             stored['R0_valid'][positions])
    from . import convex_train as T
    assert locked['anchor_index_sha'] == choices['anchor_index_sha'] == T.OLD.array_sha(anchor_index)
    with np.load(C.PARENT_RAW/'SOURCE_FEATURES.npz') as stored:
        masks = {m:stored[m+'_valid'][positions] for m in C.U.MODELS}
    _, direction_bindings = D.load_frozen('SOURCE_VAL', rows, positions, anchor_index, masks, protocol_binding, direction_input_bindings(protocol))
    for key, value in direction_bindings.items(): assert locked[key] == choices[key] == value
    for model in LEARNED:
        assert set(choices['records'][model]) == set(choices['ids'])
        for j, fid in enumerate(choices['ids']):
            record = choices['records'][model][fid]
            assert record['anchor_index'] == int(anchor_index[j])
            assert record['anchor_name'] == (None if anchor_index[j] < 0 else names_for(model)[int(anchor_index[j])])
            for usable, axes, score in zip(record['candidate_valid'],record['predicted_signed_axes'],record['scores']):
                if usable:
                    assert len(axes)==2 and np.isfinite(axes).all() and score==max(axes)
                else:
                    assert axes is None and score is None
            if anchor_index[j]>=0:
                assert record['predicted_signed_axes'][int(anchor_index[j])] == [0.,0.]
    return protocol, locked, rows, poses, choices


def freeze():
    """Inference on cached numeric features only; no source reference values."""
    start = time.monotonic()
    assert not _REFERENCE_ACCESS['allowed'], 'Run freeze and score as separate processes.'
    if ROUTING_LOCK.exists():
        verify_routing()
        print('SIGNED_AXES_DIRECTION_SOURCE_VAL_ROUTING_ALREADY_LOCKED', flush=True)
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
    directions, direction_bindings = freeze_directions(protocol, positions, rows, poses, geo, valid, anchor_index, feature_lock)
    records, counts = {}, {}
    for model in LEARNED:
        checkpoint = C.read(C.ROOT / fits[model]['checkpoint']['path'])
        names = names_for(model)
        features = np.concatenate([geo[parent] for parent in parents_for(model)], axis=1)
        usable = np.concatenate([valid[parent] for parent in parents_for(model)], axis=1)
        assert features.dtype == np.float32 and usable.dtype == bool
        assert features.shape == (1024, len(names), 94) and usable.shape == (1024, len(names))
        assert np.isfinite(features[usable]).all()
        direction18 = np.concatenate([directions[parent] for parent in parents_for(model)], axis=1)
        scores = np.asarray(T.score_candidates(checkpoint, features, usable, anchor_index, direction18), np.float64)
        predicted_axes = np.asarray(T.predict_axes(checkpoint, features, usable, anchor_index, direction18), np.float64)
        assert predicted_axes.shape == (*usable.shape, 2) and np.isfinite(predicted_axes).all()
        assert not predicted_axes[~usable].any()
        np.testing.assert_array_equal(scores, np.where(usable, predicted_axes.max(2), np.inf))
        active_rows=np.flatnonzero(usable.any(1))
        np.testing.assert_array_equal(predicted_axes[active_rows,anchor_index[active_rows]],0.)
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
                          predicted_signed_axes=[axes.tolist() if flag else None for axes,flag in zip(predicted_axes[j],usable[j])],
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
                        records=records, **direction_bindings, **{key:protocol[key] for key in METHOD_FIELDS},
                        basis_SHA_bind=protocol['inputs']['rbf_basis'],
                        anchor_index_sha=T.OLD.array_sha(anchor_index),
                        source_labels_read=False, real_references_read=False))
    C.save(ROUTING_LOCK, dict(complete=True, created_at=C.now(), frames=1024,
           models=list(LEARNED), baselines=list(BASELINES), choices=C.bind(CHOICES), **direction_bindings,
           protocol=protocol_binding, training_complete=C.bind(C.DOC / 'TRAINING_COMPLETE.json'),
           fits=fits, feature_lock=C.bind(F.FEATURE_LOCK), source_contract=C.bind(C.PARENT_DOC / 'SOURCE_CONTRACT.json'),
           source_predictions_lock=C.bind(C.PARENT_DOC / 'SOURCE_PREDICTIONS_LOCK.json'),
           runtime_amendment=C.U.protocol().get('runtime_amendment'),
           metadata=C.bind(C.PARENT_RAW / 'SOURCE_INPUTS.json'), poses=C.bind(C.PARENT_RAW / 'SOURCE_POSES.json'),
           operators=operator_bindings(), counts=counts, source_labels_read=False,
           **{key:protocol[key] for key in METHOD_FIELDS}, basis_SHA_bind=protocol['inputs']['rbf_basis'],
           anchor_rule='Frozen input-only R0 operational GEO candidate; unchanged by learned R0_ONLY.',
           anchor_index_sha=T.OLD.array_sha(anchor_index),
           source_VAL_label_values_read=False, source_TRAIN_labels_sha_verified=True,
           source_label_access_disclosure='No source VAL label values read. Existing TRAIN label containers are hashed only to verify sealed training provenance.',
           real_references_read=False, VAL_quality_scored=False, fits_executed=0,
           image_forwards=0, new_PnP_solves=0, read_paths=sorted(set(READS)),
           wall_seconds=time.monotonic() - start))
    print('SIGNED_AXES_DIRECTION_SOURCE_VAL_ROUTING_LOCKED', counts, flush=True)


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
        print('SIGNED_AXES_DIRECTION_SOURCE_VAL_GATE_ALREADY_COMPLETE', done['status'], flush=True)
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
           **{key:protocol[key] for key in METHOD_FIELDS}, basis_SHA_bind=protocol['inputs']['rbf_basis'],
           frames=1024, models=list(LEARNED), baselines=list(BASELINES), seeds=list(SEEDS),
           PASS=passed, status='PASS_SOURCE_VAL_JOINT' if passed else 'SOURCE_RANKING_NO_JOINT_SIGNAL',
           real_routing_authorized=passed, real_reference_accessed=False,
           gate_rule=VAL_RULE, checks_total=45, checks_passed=45-len(failures),
           comparisons=comparisons, failed_checks=failures,
           summaries={model: F.summary(values) for model, values in errors.items()},
           routing_lock=C.bind(ROUTING_LOCK), protocol=C.bind(PROTOCOL),
           **{key:route_lock[key] for key in ('direction_inputs','directions','direction_normalization_sha','direction_receipt_binding')},
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
    print('SIGNED_AXES_DIRECTION_SOURCE_VAL_GATE', result['status'], 'failed_checks', len(failures), flush=True)


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
    verify_certificate(dict(PASS=True, optimizer_success=True, gradient_l2=1e-9,
                            gradient_l2_squared_over_2lambda=5e-15, max_gap_upper_bound=1e-6,
                            lambda_l2=1e-4, objective_calls=1, iterations=0,
                            solver_rule=METHOD_FIELDS['solver_rule'], gradient_linf=1e-9,
                            sign_rule=METHOD_FIELDS['sign_rule'], sign_coefficient=1.,
                            gradient_linf_max=1e-8, final_accepted_call=1))
    # Invented log: one rejected trial, one accepted half step. No solver runs.
    first=dict(event='objective', call=1, phase='initial', iteration=0, evaluated=True,
        weight_sha='zero', objective=1., Huber=.8, Sign_logistic=.2, L2_penalty=0., gradient_l2=1.,
        gradient_linf=.5, gradient_gap_upper_bound=5000., alpha=0.,
        accepted_point_call_before=None, parent_weight_sha=None, parent_objective=None,
        directional_derivative=None, armijo_bound=None, armijo_accepted=True)
    rejected=dict(first, call=2, phase='trial', iteration=1, weight_sha='rejected',
        objective=1.05, Huber=.85, alpha=1., accepted_point_call_before=1,
        parent_weight_sha='zero', parent_objective=1., directional_derivative=-.4,
        armijo_bound=1.-1e-4*.4, armijo_accepted=False)
    accepted=dict(rejected, call=3, weight_sha='accepted', objective=.8, Huber=.6,
        alpha=.5, armijo_bound=1.-1e-4*.5*.4, armijo_accepted=True,
        gradient_l2=1e-9, gradient_linf=1e-9, gradient_gap_upper_bound=5e-15)
    iteration={key:accepted[key] for key in ('weight_sha','objective','Huber','Sign_logistic','L2_penalty',
        'gradient_l2','gradient_linf','gradient_gap_upper_bound','alpha','armijo_bound',
        'armijo_accepted','parent_weight_sha','parent_objective','directional_derivative')}
    iteration.update(event='iteration', iteration=1, objective_calls=3, accepted_call=3)
    toy_trace=[first,rejected,accepted,iteration]
    toy_receipt=dict(objective_calls=3, iterations=1, fits_executed=1, optimizer_steps=1,
        final_accepted_call=3, final_weight_sha='accepted')
    toy_ck=dict(fits_executed=1, optimizer_steps=1, final_accepted_call=3,
        solver=dict(final_accepted_call=3), certificate=dict(objective_calls=3,iterations=1,
        final_accepted_call=3,objective_value=.8,Huber=.6,Sign_logistic=.2,gradient_l2=1e-9,gradient_linf=1e-9))
    verify_newton_trace(toy_trace, {'initial_weight_sha':'zero'}, toy_receipt, toy_ck)
    import copy
    for index,key,value in ((2,'accepted_point_call_before',2),(2,'alpha',.25),
                            (1,'armijo_accepted',True),(3,'accepted_call',2)):
        broken=copy.deepcopy(toy_trace);broken[index][key]=value
        try:verify_newton_trace(broken, {'initial_weight_sha':'zero'}, toy_receipt, toy_ck)
        except AssertionError:pass
        else:raise AssertionError(('BAD_NEWTON_TRACE_ACCEPTED',index,key))
    assert len(names_for('R0_ONLY')) == 2 and len(names_for('UNION_s3')) == 4
    assert len(LEARNED) == len(BASELINES) == 4
    from . import convex_train as T
    assert T.OLD.READS is None, 'Shared scorer import must not activate fitting-only hooks.'
    weight = np.zeros((271,2), np.float64)
    weight[0] = [1.,1.]
    checkpoint = dict(schema='pallet_pose_signed_axes_direction_linear271x2_v1', bias=0., lambda_l2=1e-4,
                      weight=weight, mean=np.zeros(94, np.float32), std=np.ones(94, np.float32),
                      **METHOD_FIELDS,
                      names=names_for('UNION_s1'), normalization='old_float32_then_float64',
                      )
    def toy_basis():
        centers = np.zeros((64, 189), np.float64)
        centers[np.arange(64), np.arange(64)] = 1.
        return dict(schema='pallet_pose_anchor_rbf_runtime_v1', context_dim=189, rbf_dim=64,
                    centers=centers.tolist(), bandwidth_squared=2.,
                    normalization_sha=T.OLD.array_sha(np.stack([
                        np.asarray(checkpoint['mean'], np.float32),
                        np.asarray(checkpoint['std'], np.float32)])))
    checkpoint['rbf_basis'] = toy_basis()
    checkpoint['rbf_basis_binding'] = checkpoint['basis_SHA_bind'] = dict(
        path='invented_fixture_only.json', sha256='0' * 64, bytes=0)
    checkpoint.update(direction_mean=np.zeros(18,np.float32), direction_std=np.ones(18,np.float32),
        direction_normalization_sha=T.OLD.array_sha(np.stack([np.zeros(18,np.float32),np.ones(18,np.float32)])),
        direction_receipt_binding={'invented':True})
    direction18 = np.zeros((3,4,18),np.float32)
    features = np.zeros((3, 4, 94), np.float32)
    features[0, :, 0] = [2., 1., -1., -1.]
    mask = np.array([[True] * 4, [False] * 4, [False, True, True, False]])
    names = names_for('UNION_s1')
    anchor_index = np.array([0, -1, 1], np.int64)
    scores = T.score_candidates(checkpoint, features, mask, anchor_index, direction18)
    from scripts.research.pallet_pose_signed_axes_sign_20261001_v1 import convex_train as PREVIOUS
    previous_checkpoint={**checkpoint,**PREVIOUS.metadata(),'schema':PREVIOUS.CHECKPOINT_SCHEMA,'weight':weight[:253]}
    np.testing.assert_array_equal(scores,PREVIOUS.score_candidates(previous_checkpoint,features,mask,anchor_index))
    np.testing.assert_array_equal(scores[0], [0., -1., -3., -3.])
    axes = T.predict_axes(checkpoint,features,mask,anchor_index,direction18)
    assert axes.shape==(3,4,2) and not axes[~mask].any()
    np.testing.assert_array_equal(axes[0],[[0.,0.],[-1.,-1.],[-3.,-3.],[-3.,-3.]])
    assert scores.dtype == np.float64 and np.isposinf(scores[~mask]).all()
    np.testing.assert_array_equal(T.select_candidates(scores, mask, names), [2, -1, 1])
    permutation = np.array([3, 2, 1, 0])
    permuted = T.select_candidates(scores[:, permutation], mask[:, permutation], [names[j] for j in permutation])
    np.testing.assert_array_equal(np.where(permuted >= 0, permutation[np.maximum(permuted, 0)], -1), [2, -1, 1])
    # The new score is the worse of both predicted axes, never their mean.
    opposed=weight.copy(); opposed[0]=[1.,-1.]
    two_axis_ck={**checkpoint,'weight':opposed}
    two_axis_scores=T.score_candidates(two_axis_ck,features,mask,anchor_index, direction18)
    np.testing.assert_array_equal(two_axis_scores[0],[0.,1.,3.,3.])
    assert T.select_candidates(two_axis_scores,mask,names)[0]==0
    # Nonzero appended inputs participate, with original253 coefficients zero.
    direction_fixture=direction18.copy();direction_fixture[0,:,0]=[1.,2.,-1.,-1.]
    direction_weight=np.zeros_like(weight);direction_weight[253]=[1.,2.]
    direction_scores=T.score_candidates({**checkpoint,'weight':direction_weight},features,mask,anchor_index,direction_fixture)
    np.testing.assert_array_equal(direction_scores[0],[0.,2.,-2.,-2.])
    assert T.select_candidates(direction_scores,mask,names)[0]==2
    # Original tie order remains R0 -> hypothesis -> expert, even when the
    # operational anchor is the other R0 hypothesis; no new anchor priority.
    zero_scores=T.score_candidates({**checkpoint,'weight':np.zeros((271,2))},features,mask,anchor_index, direction18)
    assert T.select_candidates(zero_scores,mask,names)[0]==0
    short_anchor=np.array([1,-1,1],np.int64)
    tied=T.score_candidates({**checkpoint,'weight':np.zeros((271,2))},features,mask,short_anchor, direction18)
    assert T.select_candidates(tied,mask,names)[0]==0 and short_anchor[0]==1
    target_builder=T.signed_targets
    def forbidden(*args,**kwargs):
        raise AssertionError('Runtime called the TRAIN reference-derived target builder')
    T.signed_targets=forbidden
    try:
        np.testing.assert_array_equal(T.score_candidates(checkpoint,features,mask,anchor_index, direction18),scores)
    finally:
        T.signed_targets=target_builder
    # Stored float32 std floor must not be re-tested against a stricter FP64 literal.
    checkpoint['std'] = np.full(94, np.float32(1e-6)).tolist()
    checkpoint['rbf_basis'] = toy_basis()
    np.testing.assert_array_equal(T.score_candidates(checkpoint, np.zeros_like(features), mask, anchor_index, direction18)[mask], 0.)
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
    print('SIGNED_AXES_DIRECTION_SOURCE_VAL_SELF_CHECK_PASS_NO_DATA_READ', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'score', 'self_check'])
    args = parser.parse_args()
    F.O.D.cv2.setNumThreads(1)
    globals()[args.stage]()


if __name__ == '__main__':
    main()
