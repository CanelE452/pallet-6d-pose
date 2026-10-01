"""Lock whole-pose real routes from signed T/R predictions; no real targets enter routing.

No image model, PnP solve, optimization, or new target is used. ``self_check``
uses invented arrays only. Root seals REAL_PROTOCOL before ``freeze``; raw
real references and old real quality are inaccessible until route verification.
"""
from . import common as C

import argparse
from collections import Counter
import csv
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

_REFERENCES = {'allowed': False}
READS = set()
DENIED = ('/data/evaluation/', '/real_gt_v2/annotations/',
          'GEOMETRY_RESOLVED_POSE_GT', 'TRUTH_FOR_DISPLAY', 'AXIS_REVIEW_MANIFEST',
          '/POSE_METRICS.json', '/E1_POSE_METRICS.json', '/FRAME_RESULTS.csv',
          '/RESULTS.json', '/REAL_RESULTS.json', '/REAL_FRAME_RESULTS.csv')


def reference_guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if not _REFERENCES['allowed']:
            assert Path(path).suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp'), ('RAW_IMAGE_BEFORE_ROUTE_LOCK', path)
            assert not any(token in path for token in DENIED), ('REAL_REFERENCE_BEFORE_ROUTE_LOCK', path)
        if path.startswith(str(C.ROOT)):
            READS.add(path)


sys.addaudithook(reference_guard)

LEARNED = C.MODEL_NAMES
BASELINES = ('R0', 'PRIOR1', 'FULL125', 'DIVERSE251_s1', 'DIVERSE251_s2', 'DIVERSE251_s3',
             'SINGLE251_s1', 'SINGLE251_s2', 'SINGLE251_s3')
MODELS = LEARNED + BASELINES
HYP = ('long-face-front', 'short-face-front')
RAW_FEATURE_DIM = 94
FEATURE_DIM = 253
BASE_CONTEXT_DIM = 189
RBF_FEATURE_DIM = 64
FEATURE_MAP = 'normalized94_abs_anchor_delta94_identity1_fixed_rbf64'
CHECKPOINT_SCHEMA = 'pallet_pose_signed_axes_linear253x2_v1'
LOSS_RULE = 'FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_HUBER'
INPUT_RULE = 'RBF253_CANDIDATE_MINUS_R0_ANCHOR'
PREDICTION_RULE = 'MAX_TWO_SIGNED_LOG1P_AXES'
OUTPUT_DIM = 2
HUBER_DELTA = 1.
SIGNED_METHOD = dict(input_rule=INPUT_RULE, prediction_rule=PREDICTION_RULE,
                     output_dim=OUTPUT_DIM, huber_delta=HUBER_DELTA,
                     runtime_uses_reference_errors=False)
REAL_STABILITY_CONTRACT = 'matched_intervention_AND_original_SINGLE251_stability'
ANCHOR_RULE = 'Frozen input-only R0 operational GEO; never learned R0_ONLY or a reference-selected pose.'
MISSING_ANCHOR_RULE = 'All-invalid rows require anchor=-1 and zero context; any valid candidate without a valid R0 GEO anchor is a contract error.'
STABLE_DOC = C.U.PREV
STABLE_RAW = C.U.PREV_RAW
JOINT_DOC = C.ROOT / '_docs/experiments/pallet_pose_joint_recovery_20261001_v1'
JOINT_RAW = C.ROOT / 'data/pallet/results/pallet_pose_joint_recovery_20261001_v1'
REAL_PROTOCOL = C.DOC / 'REAL_PROTOCOL.json'
ROUTE = C.RAW / 'REAL_CHOICES.json'
ROUTE_LOCK = C.DOC / 'REAL_ROUTING_LOCK.json'
RESULTS = C.DOC / 'REAL_RESULTS.json'


def bindings(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            yield value
        else:
            for child in value.values():
                yield from bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from bindings(child)


def verify_all(value):
    seen = {}
    for binding in bindings(value):
        old = seen.setdefault(binding['path'], binding)
        assert old['sha256'] == binding['sha256'], ('CONFLICTING_BINDING', binding['path'])
    for binding in seen.values():
        C.verify(binding)


def old_evaluation():
    # This import is pure; do not import source_features/source_val, whose
    # permanent source-process guard intentionally prevents real references.
    from scripts.research.pallet_pose_stable_improvement_20261001_v1 import evaluate as E
    return E


def input_paths():
    return dict(features=JOINT_RAW / 'REAL_FEATURE_ATTRIBUTION.npz',
                attribution=JOINT_DOC / 'SELECTOR_ATTRIBUTION.json',
                prediction_lock=STABLE_DOC / 'PREDICTIONS_LOCK.json',
                pose_lock=STABLE_DOC / 'POSE_PREDICTIONS_LOCK.json',
                poses=STABLE_RAW / 'POSE_CANDIDATES.json', groups=STABLE_RAW / 'EVAL_GROUPS.json',
                metadata=STABLE_RAW / 'EVAL_METADATA.json',
                reference_bindings=STABLE_DOC / 'REFERENCE_BINDINGS.json',
                stable_protocol=STABLE_DOC / 'PROTOCOL_EFFECTIVE.json',
                stable_data_audit=STABLE_DOC / 'DATA_AUDIT.json',
                stable_publication=STABLE_DOC / 'PUBLICATION_MANIFEST.json')


def operator_paths():
    from . import convex_train as T
    from scripts.research.pallet_pose_anchor_rbf_20261001_v1 import basis as B
    from scripts.research.pallet_pose_anchor_risk_20261001_v1 import convex_train as RISK
    E = old_evaluation()
    return list(dict.fromkeys([Path(__file__), Path(C.__file__), Path(C.U.__file__),
        Path(B.__file__), Path(RISK.__file__),
        Path(T.__file__), Path(T.OLD.__file__), Path(T.RBF.__file__),
        Path(T.RBF.PREV.__file__), Path(T.RBF.PREV.C.__file__),
        Path(E.__file__), Path(E.C.__file__),
        Path(E.C.D.__file__), Path(E.C.D.M.__file__), Path(E.C.D.O.__file__),
        Path(E.C.D.O.D.__file__), Path(E.C.D.O.D.Pose.__file__)]))


def real_protocol_spec():
    """Root may call this input-only builder before sealing REAL_PROTOCOL."""
    parent, _, source_gate = verify_training_and_source()
    return dict(schema='pallet_pose_signed_axes_learned_real_v1', complete=True,
                parent_protocol=C.bind(C.DOC / 'TRAIN_PROTOCOL.json'),
                source_val_gate=source_gate,
                models=list(LEARNED), baselines=list(BASELINES), frames=173,
                real_evaluation=parent['real_evaluation'],
                original_goal_criteria=parent['original_goal_criteria'],
                real_stability_contract=REAL_STABILITY_CONTRACT,
                original_goal_contract_correction=parent['original_goal_contract_correction'],
                inputs={key: C.bind(path) for key, path in input_paths().items()},
                codes=[C.bind(path) for path in operator_paths()],
                whole_pose_selection=True, image_forwards=0, new_PnP_solves=0,
                source_PASS_required=True, reference_values_before_routing=False,
                target_rule=parent['target_rule'], runtime_safe_mask=False,
                loss_rule=LOSS_RULE, runtime_uses_margin=False,
                runtime_uses_fixed_rbf=True, basis_SHA_bind=parent['inputs']['rbf_basis'],
                runtime_basis_scope='Embedded64 centers and fixed bandwidth from frozen TRAIN input features only; no fitting or basis update during real routing.',
                training_target_scope='Signed log1p physical T/R anchor-excess targets are TRAIN-only supervision. Only SHA strings are checked here; no real error targets are loaded or created for routing.',
                runtime_selection='Maximum of two predicted signed-log1p normalized changes; argmin over every original valid whole-pose candidate. Exact ties retain R0 then hypothesis priority, which need not favor the operational anchor over another R0 hypothesis.',
                raw_feature_dim=RAW_FEATURE_DIM, feature_dim=FEATURE_DIM, feature_map=FEATURE_MAP,
                normalization='old_float32_then_float64',
                feature_order='Preserve frozen raw94 normalization, context189 and fixed RBF64. Subtract operational R0 anchor253 from each valid candidate before predicting two axes. Invalid rows remain zero. No basis updates or extra normalization.',
                **SIGNED_METHOD,
                anchor_rule=ANCHOR_RULE, missing_anchor_rule=MISSING_ANCHOR_RULE,
                fallback='R0_GEO_if_available_else_failure',
                feature_validity='Original frame-valid broadcast to2, intersect cached per-hypothesis pose availability.',
                seed_meaning='Three frozen refiner seeds; R0_ONLY fitted once and repeated as a shared control, not three independent fits.')


def verify_training_and_source():
    from . import convex_train as T
    parent = C.protocol('TRAIN_PROTOCOL')
    assert T.FEATURE_DIM == FEATURE_DIM and T.FEATURE_MAP == FEATURE_MAP and T.CHECKPOINT_SCHEMA == CHECKPOINT_SCHEMA
    assert T.LOSS_RULE == LOSS_RULE
    assert parent['raw_feature_dim'] == RAW_FEATURE_DIM and parent['feature_dim'] == FEATURE_DIM
    assert parent['feature_map'] == FEATURE_MAP and parent['normalization'] == 'old_float32_then_float64'
    assert parent['loss_rule'] == LOSS_RULE and parent['runtime_uses_margin'] is False
    for key, value in dict(input_rule=INPUT_RULE, prediction_rule=PREDICTION_RULE,
                           output_dim=OUTPUT_DIM, huber_delta=HUBER_DELTA,
                           runtime_uses_reference_errors=False, runtime_safe_mask=False).items():
        assert parent[key] == value
    assert parent['real_stability_contract'] == REAL_STABILITY_CONTRACT
    pbind = C.bind(C.DOC / 'TRAIN_PROTOCOL.json')
    complete_path = C.DOC / 'TRAINING_COMPLETE.json'
    complete = C.read(complete_path)
    assert complete['complete'] and complete['protocol'] == pbind
    assert complete['models'] == list(LEARNED) and complete['fit_count'] == 4
    assert complete['all_certified'] and complete['source_TRAIN_only']
    assert complete['loss_rule'] == LOSS_RULE and complete['runtime_uses_margin'] is False
    assert complete['basis_SHA_bind'] == parent['inputs']['rbf_basis']
    C.verify(complete['basis_SHA_bind'])
    basis_artifact = C.read(C.ROOT / complete['basis_SHA_bind']['path'])
    assert basis_artifact['complete'] and basis_artifact['PASS']
    assert set(complete['signed_target_sha_by_model']) == set(LEARNED)
    assert set(complete['input_difference_sha_by_model']) == set(LEARNED)
    for key, expected in [('raw_feature_dim', RAW_FEATURE_DIM), ('feature_dim', FEATURE_DIM), ('feature_map', FEATURE_MAP)]:
        assert complete[key] == parent[key] == expected
    assert complete['target_rule'] == parent['target_rule'] == T.TARGET_RULE
    assert not complete['VAL_quality_read'] and not complete['real_targets_read']
    assert len(complete['fits']) == 4
    verify_all(complete)
    fits = {}
    for binding in complete['fits']:
        receipt = C.read(C.ROOT / binding['path'])
        model = receipt['model']
        assert model in LEARNED and model not in fits
        assert binding == C.bind(C.DOC / f'FIT_{model}.json')
        assert receipt['complete'] and receipt['protocol'] == pbind
        assert receipt['loss_rule'] == LOSS_RULE and receipt['runtime_uses_margin'] is False
        for key in ('raw_feature_dim', 'feature_dim', 'feature_map'):
            assert receipt[key] == parent[key]
        assert receipt['source_TRAIN_only'] and not receipt['VAL_quality_read'] and not receipt['real_targets_read']
        assert receipt['checkpoint'] == C.bind(C.RAW / 'fits' / model / 'final.json')
        verify_all(receipt)
        ck = C.read(C.ROOT / receipt['checkpoint']['path'])
        assert ck['schema'] == CHECKPOINT_SCHEMA and ck['model'] == model
        assert ck['loss_rule'] == LOSS_RULE and ck['runtime_uses_margin'] is False
        assert ck['rbf_basis'] == basis_artifact['basis']
        assert ck['rbf_basis_binding'] == ck['basis_SHA_bind'] == receipt['basis_SHA_bind'] == complete['basis_SHA_bind']
        assert ck['rbf_basis']['schema'] == 'pallet_pose_anchor_rbf_runtime_v1'
        T.B.validate_basis(ck['rbf_basis'], ck['mean'], ck['std'])
        assert ck['raw_feature_dim'] == RAW_FEATURE_DIM and ck['feature_dim'] == FEATURE_DIM
        assert ck['feature_map'] == FEATURE_MAP and ck['normalization'] == 'old_float32_then_float64'
        assert ck['names'] == T.candidate_names(model) and ck['protocol'] == pbind
        assert ck['certificate'] == receipt['certificate'] and ck['bias'] == 0. and ck['lambda_l2'] == 1e-4
        assert ck['target_rule'] == receipt['target_rule'] == parent['target_rule'] == T.TARGET_RULE
        assert ck['runtime_safe_mask'] is False and ck['loss_uses_original_valid_mask'] is True
        cert = ck['certificate']
        assert cert['loss_rule'] == LOSS_RULE
        assert cert['PASS'] and cert['optimizer_success']
        assert cert['lambda_l2'] == 1e-4 and np.isfinite(cert['gradient_l2']) and cert['gradient_l2'] >= 0
        gap = cert['gradient_l2_squared_over_2lambda']
        assert cert['max_gap_upper_bound'] == 1e-6 and 0 <= gap <= 1e-6
        np.testing.assert_allclose(gap, cert['gradient_l2'] ** 2 / (2e-4), rtol=1e-12, atol=1e-18)
        assert 0 < cert['objective_calls'] == receipt['objective_calls'] <= 2000
        assert 0 <= cert['iterations'] == receipt['iterations'] <= 1000
        assert receipt['fits_executed'] == 1
        mean, std = np.asarray(ck['mean'], np.float32), np.asarray(ck['std'], np.float32)
        weight = np.asarray(ck['weight'], np.float64)
        assert mean.shape == std.shape == (RAW_FEATURE_DIM,) and weight.shape == (FEATURE_DIM, OUTPUT_DIM)
        assert np.isfinite(mean).all() and np.isfinite(std).all() and np.isfinite(weight).all()
        assert (std >= np.float32(1e-6)).all()
        assert T.OLD.array_sha(np.stack([mean, std])) == ck['normalization_sha'] == receipt['normalization_sha']
        assert T.OLD.array_sha(weight) == receipt['final_weight_sha']
        start = C.read(C.ROOT / receipt['START']['path'])
        assert start['basis_SHA_bind'] == complete['basis_SHA_bind']
        for key in ('raw_feature_dim', 'feature_dim', 'feature_map'):
            assert start[key] == ck[key] == parent[key]
        assert start['anchor_index_sha'] == receipt['anchor_index_sha'] == ck['anchor_index_sha'] == complete['anchor_index_sha']
        assert start['loss_rule'] == LOSS_RULE and start['runtime_uses_margin'] is False
        for key in ('signed_target_sha', 'input_difference_sha', 'base_context_sha', 'errors_sha', 'scaled_excess_sha', 'original_valid_sha'):
            digest = start[key]
            assert isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
            assert digest == receipt[key] == ck[key]
        assert ck['signed_target_sha'] == complete['signed_target_sha_by_model'][model]
        assert ck['input_difference_sha'] == complete['input_difference_sha_by_model'][model]
        assert start['model'] == model and start['protocol'] == pbind and start['target_rule'] == T.TARGET_RULE
        assert start['signed_target_sha'] == receipt['signed_target_sha']
        for key, value in dict(input_rule=INPUT_RULE, prediction_rule=PREDICTION_RULE,
                               output_dim=OUTPUT_DIM, huber_delta=HUBER_DELTA,
                               runtime_uses_reference_errors=False, runtime_safe_mask=False).items():
            assert start[key] == receipt[key] == ck[key] == complete[key] == parent[key] == value
        for key in ('signed_target_sha', 'input_difference_sha', 'base_context_sha',
                    'errors_sha', 'scaled_excess_sha', 'original_valid_sha'):
            assert ck[key] == complete[key + '_by_model'][model]
        trace = [json.loads(line) for line in (C.ROOT / receipt['trace']['path']).read_text().splitlines()]
        calls = [line for line in trace if line['event'] == 'objective']
        iterations = [line for line in trace if line['event'] == 'iteration']
        assert [line['call'] for line in calls] == list(range(1, receipt['objective_calls'] + 1))
        assert [line['iteration'] for line in iterations] == list(range(1, receipt['iterations'] + 1))
        assert calls[-1]['weight_sha'] == receipt['final_weight_sha']
        assert calls[-1]['objective'] == cert['objective_value']
        for row in trace:
            assert row['loss_rule'] == LOSS_RULE and row['signed_target_sha'] == ck['signed_target_sha']
            assert row['basis_SHA_bind'] == complete['basis_SHA_bind']
            for key in ('signed_target_sha', 'input_difference_sha', 'base_context_sha',
                        'errors_sha', 'scaled_excess_sha', 'original_valid_sha',
                        'input_rule', 'prediction_rule', 'target_rule', 'output_dim', 'huber_delta',
                        'runtime_uses_reference_errors', 'runtime_safe_mask'):
                assert row[key] == ck[key]
        fits[model] = dict(receipt=binding, checkpoint=receipt['checkpoint'])
    assert set(fits) == set(LEARNED)
    receipts = [C.read(C.ROOT / fits[model]['receipt']['path']) for model in LEARNED]
    assert len({r['normalization_sha'] for r in receipts}) == 1
    assert sum(r['objective_calls'] for r in receipts) == complete['total_objective_calls']
    gate_path = C.DOC / 'SOURCE_VAL_GATE.json'
    gate = C.read(gate_path)
    assert gate['complete'] and gate['PASS'] and gate['real_routing_authorized'], 'SOURCE_VAL_MUST_PASS_BEFORE_REAL_ROUTING'
    assert gate['frames'] == 1024 and gate['models'] == list(LEARNED)
    assert gate['checks_total'] == gate['checks_passed'] == 45 and gate['failed_checks'] == []
    assert gate['protocol'] == pbind and gate['training_complete'] == C.bind(complete_path)
    assert not gate['real_reference_accessed'] and gate['gate_rule'] == parent['source_val']
    verify_all(gate)
    source_route = C.read(C.ROOT / gate['routing_lock']['path'])
    assert source_route['complete'] and source_route['models'] == list(LEARNED)
    assert source_route['protocol'] == pbind and source_route['fits'] == fits
    assert source_route['basis_SHA_bind'] == complete['basis_SHA_bind']
    assert not source_route['real_references_read'] and not source_route['source_VAL_label_values_read']
    verify_all(source_route)
    return parent, fits, C.bind(gate_path)


def verified_inputs():
    parent, fits, source_gate = verify_training_and_source()
    expected = C.read(C.DOC / 'REAL_PROTOCOL_SHA.json')
    assert expected['path'] == str(REAL_PROTOCOL.relative_to(C.ROOT))
    C.verify(expected)
    protocol = C.read(REAL_PROTOCOL)
    assert protocol == real_protocol_spec(), 'REAL_PROTOCOL_OR_OPERATOR_DRIFT'
    verify_all(protocol)
    inp = protocol['inputs']
    assert inp['stable_protocol'] == parent['evidence']['original_goal_protocol']
    old_protocol = C.read(C.ROOT / inp['stable_protocol']['path'])
    assert protocol['original_goal_criteria'] == parent['original_goal_criteria'] == old_protocol['stability_criteria']
    prediction = C.read(C.ROOT / inp['prediction_lock']['path'])
    poses_lock = C.read(C.ROOT / inp['pose_lock']['path'])
    attr = C.read(C.ROOT / inp['attribution']['path'])
    assert prediction['complete'] and not prediction['inference_reference_coordinates_read']
    assert prediction['metadata'] == inp['metadata']
    assert poses_lock['prediction_lock'] == inp['prediction_lock'] and not poses_lock['references_read']
    assert inp['poses'] in poses_lock['files'] and inp['groups'] in poses_lock['files']
    assert inp['prediction_lock'] in attr['inputs'] and inp['pose_lock'] in attr['inputs']
    assert inp['features'] in attr['artifacts'] and attr['real_reference_reads'] == 0
    for value in (prediction, poses_lock):
        verify_all(value)
    verify_all(attr['inputs'])
    verify_all(attr['codes'])
    rows = C.read(C.ROOT / inp['metadata']['path'])
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)) == 173
    groups = C.read(C.ROOT / inp['groups']['path'])
    assert {k: len(groups[k]) for k in ('NATURAL99', 'CLEAN29', 'WOOD45')} == parent['real_evaluation']['populations']
    assert len(groups['FULL128']) == 128
    assert set(groups['NATURAL99']).isdisjoint(groups['CLEAN29'])
    assert set(groups['NATURAL99']) | set(groups['CLEAN29']) == set(groups['FULL128'])
    assert set(groups['FULL128']).isdisjoint(groups['WOOD45'])
    assert set(groups['FULL128']) | set(groups['WOOD45']) == set(ids)
    assert all(len(v) == len(set(v)) and set(v) <= set(ids) for v in groups.values())
    metadata = {r['id']: r for r in rows}
    assert len({metadata[i]['recording'] for i in groups['NATURAL99']}) == 6
    assert len({metadata[i]['recording'] for i in groups['WOOD45']}) == 2
    poses = C.read(C.ROOT / inp['poses']['path'])
    for model in BASELINES:
        assert set(poses[model]) == set(ids)
        assert all(not poses[model][i]['reference_coordinates_read'] for i in ids)
    return protocol, fits, source_gate, rows, groups, poses


def feature_arrays(path, rows, poses):
    arrays, masks = {}, {}
    with np.load(path, allow_pickle=False) as stored:
        assert stored['ids'].tolist() == [r['id'] for r in rows]
        assert stored['feature_names'].shape == (94,)
        for parent in C.U.MODELS:
            raw, frame_valid = stored[parent + '_geo'], stored[parent + '_valid']
            assert raw.dtype == np.float32 and raw.shape == (173, 2, 94)
            assert frame_valid.dtype == bool and frame_valid.shape == (173,)
            mask = np.broadcast_to(frame_valid[:, None], (173, 2)).copy()
            for j, row in enumerate(rows):
                hs = {h['name']: h['pose'] for h in poses[parent][row['id']]['hypotheses']}
                assert set(hs) <= set(HYP)
                for k, name in enumerate(HYP):
                    mask[j, k] &= name in hs and bool(hs[name]['available'])
            assert np.isfinite(raw[mask]).all()
            arrays[parent], masks[parent] = raw, mask
    return arrays, masks


def anchor_indices(rows, poses, masks):
    """Derive context identity from frozen input-only R0 GEO, never pose error."""
    assert set(masks) == set(C.U.MODELS)
    n = len(rows)
    assert all(mask.dtype == bool and mask.shape == (n, 2) for mask in masks.values())
    present = np.logical_or.reduce([mask.any(1) for mask in masks.values()])
    indices = np.full(n, -1, np.int64)
    for j in np.flatnonzero(present):
        record = poses['R0'][rows[j]['id']]
        name = record['GEO_name']
        assert name in HYP, ('ANCHOR_CANDIDATE_SUPPORT_CONTRACT', rows[j]['id'])
        index = HYP.index(name)
        assert masks['R0'][j, index], ('ANCHOR_CANDIDATE_SUPPORT_CONTRACT', rows[j]['id'])
        hypotheses = [h['pose'] for h in record['hypotheses'] if h['name'] == name]
        assert len(hypotheses) == 1 and hypotheses[0]['available']
        assert record['GEO_pose'] == hypotheses[0], ('R0_GEO_IDENTITY_MISMATCH', rows[j]['id'])
        indices[j] = index
    assert np.array_equal(indices >= 0, present)
    return indices


def pose_for(choice, poses, fid):
    if choice['fallback']:
        assert choice['candidate_index'] == -1 and choice['candidate_name'] is None
        return poses['R0'][fid]['GEO_pose']
    parent, name = choice['candidate_name'].split(':')
    assert parent == choice['parent'] and name == choice['hypothesis']
    found = [h['pose'] for h in poses[parent][fid]['hypotheses'] if h['name'] == name]
    assert len(found) == 1 and found[0]['available']
    return found[0]


def validate_choices(choices, rows, poses, anchor_index):
    ids = [r['id'] for r in rows]
    assert choices['ids'] == ids and choices['models'] == list(LEARNED)
    assert set(choices['records']) == set(LEARNED) and not choices['real_reference_values_read']
    assert choices['feature_map'] == FEATURE_MAP and choices['raw_feature_dim'] == RAW_FEATURE_DIM
    assert choices['feature_dim'] == FEATURE_DIM and choices['anchor_rule'] == ANCHOR_RULE
    assert choices['loss_rule'] == LOSS_RULE and choices['runtime_uses_margin'] is False
    assert choices['runtime_uses_fixed_rbf'] is True
    assert all(choices[key] == value for key, value in SIGNED_METHOD.items())
    C.verify(choices['basis_SHA_bind'])
    assert anchor_index.shape == (len(rows),) and anchor_index.dtype == np.int64
    from . import convex_train as T
    for model in LEARNED:
        assert set(choices['records'][model]) == set(ids)
        names = T.candidate_names(model)
        for j, fid in enumerate(ids):
            chosen = choices['records'][model][fid]
            ai = int(anchor_index[j])
            assert chosen['anchor_index'] == ai
            assert chosen['anchor_name'] == (names[ai] if ai >= 0 else None)
            index = chosen['candidate_index']
            assert -1 <= index < len(names)
            assert chosen['candidate_name'] == (None if index == -1 else names[index])
            assert chosen['fallback'] == (index == -1)
            assert len(chosen['scores']) == len(chosen['valid']) == len(names)
            valid = np.asarray(chosen['valid'], bool)
            assert (ai >= 0) == bool(valid.any()), 'ANCHOR_CANDIDATE_SUPPORT_CONTRACT'
            if ai >= 0:
                assert names[ai] == 'R0:' + poses['R0'][fid]['GEO_name'] and valid[ai]
            scores = np.asarray([np.inf if v is None else v for v in chosen['scores']], float)
            assert np.isfinite(scores[valid]).all() and np.isposinf(scores[~valid]).all()
            assert len(chosen['predicted_signed_axes']) == len(names)
            for flag, axes, score in zip(valid, chosen['predicted_signed_axes'], chosen['scores']):
                if flag:
                    assert len(axes) == 2 and np.isfinite(axes).all() and score == max(axes)
                else:
                    assert axes is None and score is None
            if ai >= 0:
                assert chosen['predicted_signed_axes'][ai] == [0., 0.]
            assert int(T.select_candidates(scores[None], valid[None], names)[0]) == index
            pose = pose_for(chosen, poses, fid)
            assert chosen['pose'] == pose and chosen['pose_available'] == bool(pose['available'])
            if pose['available']:
                assert np.asarray(pose['R_physical']).shape == (3, 3)
                assert np.asarray(pose['centroid']).shape == (3,)
                assert np.isfinite(pose['R_physical']).all() and np.isfinite(pose['centroid']).all()


def verify_routing():
    protocol, fits, source_gate, rows, groups, poses = verified_inputs()
    lock = C.read(ROUTE_LOCK)
    assert lock['complete'] and lock['frames'] == 173 and lock['models'] == list(LEARNED)
    assert lock['protocol'] == C.bind(REAL_PROTOCOL) and lock['fits'] == fits
    assert lock['source_val_gate'] == source_gate
    assert lock['training_complete'] == C.bind(C.DOC / 'TRAINING_COMPLETE.json')
    assert lock['choices'] == C.bind(ROUTE) and lock['inputs'] == protocol['inputs']
    assert lock['operators'] == protocol['codes'] and not lock['real_reference_values_read']
    assert lock['whole_pose_selection'] and lock['new_PnP_solves'] == lock['image_forwards'] == 0
    assert lock['feature_map'] == FEATURE_MAP and lock['raw_feature_dim'] == RAW_FEATURE_DIM and lock['feature_dim'] == FEATURE_DIM
    assert lock['anchor_rule'] == ANCHOR_RULE and lock['missing_anchor_rule'] == MISSING_ANCHOR_RULE
    assert lock['loss_rule'] == LOSS_RULE and lock['runtime_uses_margin'] is False
    assert lock['runtime_uses_fixed_rbf'] is True
    assert all(lock[key] == protocol[key] == value for key, value in SIGNED_METHOD.items())
    assert lock['basis_SHA_bind'] == protocol['basis_SHA_bind']
    assert lock['real_margin_values_loaded'] is False and lock['real_margin_values_computed'] is False
    verify_all(lock)
    from . import convex_train as T
    _, masks = feature_arrays(C.ROOT / protocol['inputs']['features']['path'], rows, poses)
    index = anchor_indices(rows, poses, masks)
    assert lock['anchor_index_sha'] == T.OLD.array_sha(index)
    choices = C.read(ROUTE)
    assert choices['basis_SHA_bind'] == protocol['basis_SHA_bind']
    validate_choices(choices, rows, poses, index)
    return protocol, rows, groups, poses, choices


def freeze():
    started = time.monotonic()
    if ROUTE_LOCK.exists():
        verify_routing()
        print('REAL_ROUTES_ALREADY_LOCKED', flush=True)
        return
    from . import convex_train as T
    protocol, fits, source_gate, rows, groups, poses = verified_inputs()
    arrays, masks = feature_arrays(C.ROOT / protocol['inputs']['features']['path'], rows, poses)
    anchor_index = anchor_indices(rows, poses, masks)
    records, counts = {}, {}
    for model in LEARNED:
        ck = C.read(C.ROOT / fits[model]['checkpoint']['path'])
        names = T.candidate_names(model)
        parents = ['R0'] + ([] if model == 'R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
        x = np.concatenate([arrays[p] for p in parents], axis=1)
        valid = np.concatenate([masks[p] for p in parents], axis=1)
        scores = T.score_candidates(ck, x, valid, anchor_index)
        axes = T.predict_axes(ck, x, valid, anchor_index)
        assert axes.shape == (*valid.shape, 2) and np.isfinite(axes).all()
        assert not axes[~valid].any()
        np.testing.assert_array_equal(scores, np.where(valid, axes.max(2), np.inf))
        selected = T.select_candidates(scores, valid, names)
        records[model] = {}
        for j, row in enumerate(rows):
            index = int(selected[j])
            name = names[index] if index >= 0 else None
            parent, hypothesis = name.split(':') if name else ('R0', None)
            choice = dict(candidate_index=index, candidate_name=name, parent=parent,
                          hypothesis=hypothesis, fallback=index < 0, valid=valid[j].tolist(),
                          anchor_index=int(anchor_index[j]),
                          anchor_name=names[int(anchor_index[j])] if anchor_index[j] >= 0 else None,
                          predicted_signed_axes=[a.tolist() if flag else None for a, flag in zip(axes[j], valid[j])],
                          scores=[float(s) if np.isfinite(s) else None for s in scores[j]])
            pose = pose_for(choice, poses, row['id'])
            records[model][row['id']] = dict(choice, pose=pose, pose_available=bool(pose['available']))
        counts[model] = dict(Counter(r['candidate_name'] or 'R0_GEO_FALLBACK' for r in records[model].values()))
    choices = dict(ids=[r['id'] for r in rows], models=list(LEARNED), records=records,
                   real_reference_values_read=False, whole_pose_selection=True,
                   feature_map=FEATURE_MAP, raw_feature_dim=RAW_FEATURE_DIM, feature_dim=FEATURE_DIM,
                   anchor_rule=ANCHOR_RULE,
                   **SIGNED_METHOD,
                   loss_rule=LOSS_RULE, runtime_uses_margin=False,
                   runtime_uses_fixed_rbf=True, basis_SHA_bind=protocol['basis_SHA_bind'],
                   scores_nonfinite_policy='Invalid candidate score stored as null and valid=false, equivalent to +infinity.')
    validate_choices(choices, rows, poses, anchor_index)
    C.save(ROUTE, choices)
    C.save(ROUTE_LOCK, dict(complete=True, created_at=C.now(), frames=173, models=list(LEARNED),
        protocol=C.bind(REAL_PROTOCOL), training_complete=C.bind(C.DOC / 'TRAINING_COMPLETE.json'),
        source_val_gate=source_gate, fits=fits, inputs=protocol['inputs'], operators=protocol['codes'],
        choices=C.bind(ROUTE), selected_counts=counts, whole_pose_selection=True,
        real_reference_values_read=False, runtime_safe_mask=False, target_rule=T.TARGET_RULE,
        feature_map=FEATURE_MAP, raw_feature_dim=RAW_FEATURE_DIM, feature_dim=FEATURE_DIM,
        anchor_rule=ANCHOR_RULE, missing_anchor_rule=MISSING_ANCHOR_RULE,
        loss_rule=LOSS_RULE, runtime_uses_margin=False, real_margin_values_loaded=False,
        **SIGNED_METHOD,
        runtime_uses_fixed_rbf=True, basis_SHA_bind=protocol['basis_SHA_bind'],
        real_margin_values_computed=False,
        anchor_index_sha=T.OLD.array_sha(anchor_index),
        image_forwards=0, new_PnP_solves=0, fits_executed=0,
        wall_seconds=time.monotonic()-started))
    print('REAL_ROUTES_LOCKED_BEFORE_REFERENCE_VALUES', flush=True)


def gates_for(metrics, rows, groups, rules):
    E = old_evaluation()
    names = dict(UNION=[f'UNION_s{s}' for s in (1, 2, 3)], R0_ONLY=['R0_ONLY'] * 3,
                 ORIGINAL_DIVERSE=[f'DIVERSE251_s{s}' for s in (1, 2, 3)])
    names.update({m: [m] * 3 for m in ('R0', 'PRIOR1', 'FULL125')})
    tensors = {p: {m: E.error_tensor(metrics, seq, ids) for m, seq in names.items()} for p, ids in groups.items()}
    byid = {r['id']: r for r in rows}
    compare = ('R0_ONLY', 'ORIGINAL_DIVERSE', 'R0', 'PRIOR1', 'FULL125')
    hierarchy = {}
    for pop in ('NATURAL99', 'CLEAN29', 'WOOD45'):
        recordings = [byid[i]['recording'] for i in groups[pop]]
        hierarchy[pop] = {before: E.comparison(tensors[pop][before], tensors[pop]['UNION'], recordings,
                                             with_bootstrap=before in ('R0_ONLY', 'R0')) for before in compare}
    primary = hierarchy['NATURAL99']
    three = {m: primary[m]['all_seeds_both_medians_smaller'] for m in compare}
    uncertainty = {m: all(v['upper95_below_zero'] for v in primary[m]['hierarchical_bootstrap']['metrics'].values()) for m in ('R0_ONLY', 'R0')}
    sensitivity = {m: all(v['both_negative'] for v in primary[m]['LORO'].values()) for m in ('R0_ONLY', 'R0')}
    tail = {m: E.guard(tensors['NATURAL99'][m], tensors['NATURAL99']['UNION'], [('P90', .9)]) for m in ('R0_ONLY', 'R0')}
    clean = E.guard(tensors['CLEAN29']['R0'], tensors['CLEAN29']['UNION'], [('median', .5), ('P90', .9)])
    gates = dict(all_three_seeds_joint_gain=dict(PASS=all(three.values()), comparisons=three),
                 joint_uncertainty=dict(PASS=all(uncertainty.values()), comparisons=uncertainty),
                 recording_sensitivity=dict(PASS=all(sensitivity.values()), comparisons=sensitivity),
                 natural_tails=dict(PASS=all(v['PASS'] for v in tail.values()), comparisons=tail),
                 clean_preservation=clean)
    passed = all(v['PASS'] for v in gates.values())
    return dict(PASS=passed, gates=gates, protocol_criteria=rules,
                verdict='LOCKED_STABILITY_GATES_PASS_ON_REUSED_DEV' if passed else 'STABLE_JOINT_IMPROVEMENT_NOT_ESTABLISHED',
                goal_complete=False, strong_generalization=False,
                seed_meaning='Three frozen refiner seeds; deterministic R0_ONLY is shared, not3 independent control fits.',
                limitations='Reused DEV; geometry-derived references; only6 natural recording clusters. No independent test claim.'), hierarchy


def combined_gates(metrics, rows, groups, rules, old_protocol):
    """Keep both prespecified comparator families; aliases never mutate data."""
    matched, matched_hierarchy = gates_for(metrics, rows, groups, rules)
    E = old_evaluation()
    aliases = {f'DIVERSE251_s{s}': f'UNION_s{s}' for s in (1, 2, 3)}
    local_metrics = dict(metrics)
    for slot, model in aliases.items():
        local_metrics[slot] = metrics[model]
    original, original_hierarchy = E.gate_results(local_metrics, rows, groups, old_protocol)
    original = dict(original, evaluation_slot_aliases=aliases,
                    alias_scope='Evaluation-only local dictionary: original DIVERSE slots hold learned UNION predictions for the unchanged original SINGLE251 criteria. Public DIVERSE metrics remain original.',
                    public_baseline_metrics_preserved=True)
    assert set(matched['gates']) == set(original['gates']) and len(matched['gates']) == 5
    gates = {key: dict(PASS=bool(matched['gates'][key]['PASS'] and original['gates'][key]['PASS']),
                       matched_intervention=matched['gates'][key], original_goal=original['gates'][key])
             for key in matched['gates']}
    passed = all(gate['PASS'] for gate in gates.values())
    assert passed == bool(matched['PASS'] and original['PASS'])
    combined = dict(matched, PASS=passed, gates=gates,
        verdict='LOCKED_STABILITY_GATES_PASS_ON_REUSED_DEV' if passed else 'STABLE_JOINT_IMPROVEMENT_NOT_ESTABLISHED',
        real_stability_contract=REAL_STABILITY_CONTRACT,
        matched_intervention_PASS=matched['PASS'], original_goal_PASS=original['PASS'],
        protocol_criteria=dict(matched_intervention=rules, original_goal=old_protocol['stability_criteria']))
    return combined, matched, matched_hierarchy, original, original_hierarchy


def score():
    started = time.monotonic()
    protocol, rows, groups, poses, choices = verify_routing()
    # A complete arbitrary-choice metric cache does not exist. The original
    # reference/metric contract is therefore used after full route/hash checks;
    # compressed T-best/R-best oracle rows cannot score dominated choices.
    _REFERENCES['allowed'] = True
    if RESULTS.exists():
        result = C.read(RESULTS)
        assert result['complete'] and result['routing_lock'] == C.bind(ROUTE_LOCK)
        verify_all(result['artifacts'])
        print('REAL_RESULTS_ALREADY_FROZEN', flush=True)
        return
    E = old_evaluation()
    old_protocol = C.read(C.ROOT / protocol['inputs']['stable_protocol']['path'])
    refs = E.reference_bindings(old_protocol, rows)
    assert refs == C.read(C.ROOT / protocol['inputs']['reference_bindings']['path'])
    C.save(C.DOC / 'REAL_REFERENCE_BINDINGS.json', refs)
    metadata, truth = E.C.D.O.D.Pose.metadata('REAL_DEV')
    assert set(r['id'] for r in rows) <= set(truth)
    for row in rows:
        k, xyz, source = metadata[row['id']]
        np.testing.assert_allclose(k, row['K'], rtol=0, atol=1e-9)
        np.testing.assert_allclose(xyz, row['xyz'], rtol=0, atol=1e-9)
        assert not source and truth[row['id']]['order'] == 2
    metrics, csv_rows = {}, []
    for model in MODELS:
        metrics[model] = {}
        for row in rows:
            fid = row['id']
            choice = choices['records'][model][fid] if model in LEARNED else None
            pose = choice['pose'] if choice else poses[model][fid]['GEO_pose']
            metric = E.C.D.metric(fid, pose, truth[fid])
            metrics[model][fid] = metric
            csv_rows.append(dict(model=model, id=fid, recording=row['recording'], severity=row['severity'],
                material='PLASTIC' if fid in groups['FULL128'] else 'WOOD',
                parent=choice['parent'] if choice else model,
                hypothesis=choice['hypothesis'] if choice else poses[model][fid]['GEO_name'],
                fallback=choice['fallback'] if choice else poses[model][fid]['GEO_fallback'],
                pose_available=metric['available'], translation_cm=metric.get('translation_cm'),
                rotation_deg=metric.get('rotation_deg'),
                full_population_error_status='FINITE' if metric['available'] else 'POSITIVE_INFINITY'))
    # Authenticate historical baseline results through a pre-route bound public
    # manifest, then verify every metric against the old 173-frame calculation.
    publication = C.read(C.ROOT / protocol['inputs']['stable_publication']['path'])
    path = str((STABLE_DOC / 'RESULTS.json').relative_to(C.ROOT))
    rb = next(b for b in publication['files'] if b['path'] == path)
    C.verify(rb)
    previous = C.read(C.ROOT / rb['path'])
    path = str((STABLE_RAW / 'POSE_METRICS.json').relative_to(C.ROOT))
    mb = next(b for b in previous['artifacts'] if b['path'] == path)
    C.verify(mb)
    previous_metrics = C.read(C.ROOT / mb['path'])
    for model in BASELINES:
        for row in rows:
            E.C.D.O.D.close(metrics[model][row['id']], previous_metrics[model][row['id']])
    summaries = {pop: {m: E.C.D.summarize(metrics[m][i] for i in ids) for m in MODELS} for pop, ids in groups.items()}
    stability, matched_stability, hierarchy, original_stability, original_hierarchy = combined_gates(
        metrics, rows, groups, protocol['real_evaluation'], old_protocol)
    by_recording = {}
    for pop, ids in groups.items():
        by_recording[pop] = {}
        for rec in sorted({r['recording'] for r in rows if r['id'] in ids}):
            selected = [r['id'] for r in rows if r['id'] in ids and r['recording'] == rec]
            by_recording[pop][rec] = dict(frames=len(selected), models={m: E.C.D.summarize(metrics[m][i] for i in selected) for m in MODELS})
    metric_path = C.RAW / 'POSE_METRICS.json'
    csv_path = C.DOC / 'REAL_FRAME_RESULTS.csv'
    detail_path = C.DOC / 'REAL_DETAILED_COMPARISONS.json'
    C.save(metric_path, metrics)
    handle = io.StringIO(newline='')
    writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(csv_rows)
    C.save(csv_path, handle.getvalue())
    C.save(detail_path, dict(hierarchical=hierarchy, original_goal_hierarchy=original_hierarchy,
                            real_stability_contract=REAL_STABILITY_CONTRACT, by_recording=by_recording))
    C.save(RESULTS, dict(complete=True, created_at=C.now(), models=list(MODELS),
        protocol=C.bind(REAL_PROTOCOL), routing_lock=C.bind(ROUTE_LOCK),
        populations={p: len(ids) for p, ids in groups.items()}, summaries=summaries,
        stability=stability, hierarchy=hierarchy, full_frame_rows=len(csv_rows),
        matched_intervention_stability=matched_stability,
        original_goal_stability=original_stability, original_goal_hierarchy=original_hierarchy,
        real_stability_contract=REAL_STABILITY_CONTRACT,
        baseline_metric_parity_checks=len(BASELINES)*len(rows),
        row_failure_policy='All173 retained; failed poses have+infinity full-population errors, explicitly conditional medians plus per-seed failure guards.',
        artifacts=[C.bind(p) for p in (metric_path, csv_path, detail_path, C.DOC/'REAL_REFERENCE_BINDINGS.json')],
        image_forwards=0, new_PnP_solves=0, new_fits=0, wall_seconds=time.monotonic()-started))
    print('SIGNED_AXES_REAL_STABILITY', stability['verdict'], flush=True)


def not_run():
    """Record an actual failed source gate; never fabricate an unused result."""
    gate_path = C.DOC / 'SOURCE_VAL_GATE.json'
    stage = 'SOURCE_VAL'
    if not gate_path.exists():
        gate_path = C.DOC / 'SOURCE_FEASIBILITY.json'
        stage = 'SOURCE_TRAIN_FEASIBILITY'
    gate = C.read(gate_path)
    assert gate['complete'] and not gate['PASS']
    paths = [REAL_PROTOCOL, C.DOC / 'REAL_PROTOCOL_SHA.json', ROUTE, ROUTE_LOCK, RESULTS,
             C.DOC / 'REAL_FRAME_RESULTS.csv', C.RAW / 'POSE_METRICS.json',
             C.DOC / 'REAL_REFERENCE_BINDINGS.json', C.DOC / 'REAL_DETAILED_COMPARISONS.json']
    absent = {str(path.relative_to(C.ROOT)): not path.exists() for path in paths}
    assert all(absent.values()), ('UNAUTHORIZED_REAL_ARTIFACT_EXISTS', absent)
    training_path = C.DOC / 'TRAINING_COMPLETE.json'
    training = C.read(training_path) if training_path.exists() else None
    if training is not None:
        assert training['complete'] and training['fit_count'] == 4
    out = dict(complete=True, created_at=C.now(), status='NOT_RUN_SOURCE_GATE_FAILED',
        failed_stage=stage, source_gate=C.bind(gate_path), checks_total=gate['checks_total'],
        checks_passed=gate['checks_passed'], failed_checks=gate['failed_checks'],
        absence_checks=absent, learned_real_routes=0, real_metric_calls=0, raw_real_reference_reads=0,
        image_forwards=0, new_fits_by_real_evaluator=0,
        method_training_fits_completed=training['fit_count'] if training else 0,
        method_training_complete=C.bind(training_path) if training else None,
        method_success=False, goal_complete=False,
        target_rule='SIGNED_LOG1P_NORMALIZED_TR_ANCHOR_EXCESS',
        raw_feature_dim=RAW_FEATURE_DIM, feature_dim=FEATURE_DIM, feature_map=FEATURE_MAP,
        anchor_rule=ANCHOR_RULE, runtime_safe_mask=False,
        loss_rule=LOSS_RULE, runtime_uses_margin=False, earlier_feasibility_recomputed=False,
        real_stability_contract=REAL_STABILITY_CONTRACT,
        real_margin_values_loaded=False, real_margin_values_computed=False,
        reason='A failed predeclared source gate forbids learned real routing or performance evaluation. The earlier feasibility oracle is a separate diagnostic and has not been re-evaluated for this method.',
        unused_evaluator=C.bind(Path(__file__)), invented_selfcheck_scope='Analytic fixtures only; no learned real result is implied.')
    C.save(C.DOC / 'REAL_EVALUATION_NOT_RUN.json', out)
    C.save(C.DOC / 'REAL_EVALUATION_NOT_RUN_KO.md', '\n'.join([
        '# 두 축 변화 회귀 모델의 실사 평가는 실행하지 않음', '',
        f"사전 {stage} 판정이 {gate['checks_passed']}/{gate['checks_total']}로 실패하여 실사 learned routing과 성능 평가를 실행하지 않았다.", '',
        '[미실행 기록](REAL_EVALUATION_NOT_RUN.json)은 원본 실패 gate와 금지된 실사 산출물의 부재를 연결한다. REAL_PROTOCOL·후보 선택·routing lock·실사 점수·평가 CSV는 생성되지 않았다.', '',
        '이전 실사 anchor oracle은 GT 기반 가능성 진단이며 이번 방법에서 재평가하지 않았다. 새 모델의 예측 성능이나 목표 달성을 대신하지 않는다. 새 실사 GT annotation, 이미지 forward, PnP, 실사 metric 계산은0회다. TRAIN의 signed T/R 변화 정답은 학습에만 사용하며 실사 선택에는 읽거나 계산하지 않는다.', '']))
    print('REAL_EVALUATION_NOT_RUN_SOURCE_GATE_FAILED', stage, flush=True)


def self_check():
    from . import convex_train as T
    E = old_evaluation()
    # Whole-pose selection cannot take R from one expert and t from another.
    p0 = dict(available=True, R_physical=np.eye(3).tolist(), centroid=[0., 0., 1.])
    p1 = dict(available=True, R_physical=np.diag([-1., 1., -1.]).tolist(), centroid=[3., 4., 5.])
    poses = {'R0': {'fixture': dict(GEO_pose=p0, hypotheses=[dict(name=HYP[0], pose=p0)])},
             'DIVERSE251_s1': {'fixture': dict(hypotheses=[dict(name=HYP[1], pose=p1)])}}
    chosen = dict(fallback=False, candidate_name='DIVERSE251_s1:'+HYP[1], parent='DIVERSE251_s1', hypothesis=HYP[1])
    assert pose_for(chosen, poses, 'fixture') == p1
    assert pose_for(dict(fallback=True, candidate_index=-1, candidate_name=None), poses, 'fixture') == p0
    names = T.candidate_names('UNION_s1')
    assert T.select_candidates(np.zeros((1, 4)), np.ones((1, 4), bool), names).tolist() == [0]
    assert T.select_candidates(np.full((1, 4), np.inf), np.zeros((1, 4), bool), names).tolist() == [-1]
    # The same pure map used by training handles input-only R0 identity, an
    # entirely failed row, and raw94 normalization before any context feature.
    ids_context = ['anchor0', 'anchor1', 'failed']
    rows_context = [dict(id=fid) for fid in ids_context]
    failed_pose = dict(available=False)
    anchors = {'R0': {fid: dict(GEO_name=HYP[j] if j < 2 else None,
                                GEO_pose=(p0, p1, failed_pose)[j],
                                hypotheses=[dict(name=HYP[0], pose=p0), dict(name=HYP[1], pose=p1)] if j < 2 else [])
                       for j, fid in enumerate(ids_context)}}
    masks_context = {model: np.array([[True, True], [True, True], [False, False]]) for model in C.U.MODELS}
    index_context = anchor_indices(rows_context, anchors, masks_context)
    np.testing.assert_array_equal(index_context, [0, 1, -1])
    mask = np.concatenate([masks_context['R0'], masks_context['DIVERSE251_s1']], axis=1)
    raw = np.zeros((3, 4, RAW_FEATURE_DIM), np.float32)
    raw[:2, :, 0] = [0., 2., 0., 2.]
    raw[2] = np.nan  # Ignored invalid placeholders must not enter normalization.
    mean, std = np.zeros(RAW_FEATURE_DIM, np.float32), np.ones(RAW_FEATURE_DIM, np.float32)
    from scripts.research.pallet_pose_anchor_risk_20261001_v1 import convex_train as RISK
    base = RISK.context_inputs(raw, mask, index_context, mean, std)
    assert base.dtype == np.float64 and base.shape == (3, 4, BASE_CONTEXT_DIM)
    centers = np.zeros((RBF_FEATURE_DIM, BASE_CONTEXT_DIM), np.float64)
    centers[:, 0] = np.arange(RBF_FEATURE_DIM) / 10.
    squared_distances = [float((centers[i, 0]-centers[j, 0])**2)
                         for i in range(RBF_FEATURE_DIM) for j in range(i+1, RBF_FEATURE_DIM)]
    width = float(np.median(squared_distances))
    basis = dict(schema='pallet_pose_anchor_rbf_runtime_v1', context_dim=BASE_CONTEXT_DIM,
                 rbf_dim=RBF_FEATURE_DIM, centers=centers.tolist(), bandwidth_squared=width,
                 normalization_sha=T.OLD.array_sha(np.stack([mean, std])))
    transformed = np.zeros((3, 4, FEATURE_DIM), np.float64)
    transformed[:, :, :BASE_CONTEXT_DIM] = base
    for i, j in zip(*np.where(mask)):
        for k in range(RBF_FEATURE_DIM):
            distance = sum(float(base[i, j, d] - centers[k, d]) ** 2 for d in range(BASE_CONTEXT_DIM))
            transformed[i, j, BASE_CONTEXT_DIM + k] = np.exp(-distance / (2*width))
    np.testing.assert_array_equal(transformed[2], 0.)
    np.testing.assert_array_equal(transformed[:2, :, BASE_CONTEXT_DIM-1], [[1., 0., 0., 0.], [0., 1., 0., 0.]])
    np.testing.assert_array_equal(transformed[:2, :, RAW_FEATURE_DIM], [[0., 2., 0., 2.], [2., 0., 2., 0.]])
    weight = np.zeros((FEATURE_DIM, OUTPUT_DIM), np.float64)
    weight[RAW_FEATURE_DIM, :] = [1., 2.]
    ck = dict(schema=CHECKPOINT_SCHEMA, feature_map=FEATURE_MAP, feature_dim=FEATURE_DIM,
              raw_feature_dim=RAW_FEATURE_DIM, normalization='old_float32_then_float64',
              context_dim=BASE_CONTEXT_DIM, rbf_dim=RBF_FEATURE_DIM,
              loss_rule=LOSS_RULE, runtime_uses_margin=False,
              runtime_uses_reference_errors=False, runtime_safe_mask=False,
              input_rule=INPUT_RULE, target_rule=T.TARGET_RULE, prediction_rule=PREDICTION_RULE,
              output_dim=OUTPUT_DIM, huber_delta=HUBER_DELTA,
              names=names, bias=0., lambda_l2=1e-4, mean=mean, std=std, weight=weight,
              rbf_basis=basis, rbf_basis_binding={'invented': True}, basis_SHA_bind={'invented': True})
    difference = np.zeros_like(transformed)
    for i, j in zip(*np.nonzero(mask)):
        difference[i, j] = transformed[i, j] - transformed[i, index_context[i]]
    scores_context = T.score_candidates(ck, raw, mask, index_context)
    expected = np.where(mask, np.max(difference @ weight, axis=-1), np.inf)
    np.testing.assert_array_equal(scores_context, expected)
    assert np.all(scores_context[np.arange(2), index_context[:2]] == 0.)
    nonlinear_weight = weight.copy()
    nonlinear_weight[BASE_CONTEXT_DIM:, 0] = np.arange(RBF_FEATURE_DIM) / 100.
    nonlinear_weight[BASE_CONTEXT_DIM:, 1] = -np.arange(RBF_FEATURE_DIM) / 200.
    expected_scores = np.where(mask, np.max(difference @ nonlinear_weight, axis=-1), np.inf)
    actual_scores = T.score_candidates(dict(ck, weight=nonlinear_weight), raw, mask, index_context)
    np.testing.assert_allclose(actual_scores, expected_scores, rtol=1e-12, atol=1e-12)
    assert not np.array_equal(actual_scores[mask], scores_context[mask])
    try:
        T.score_candidates(dict(ck, rbf_basis=dict(basis, normalization_sha='0'*64)), raw, mask, index_context)
    except AssertionError:
        pass
    else:
        raise AssertionError('RBF_NORMALIZATION_DRIFT_MUST_STOP')
    original_targets = T.signed_targets
    def forbidden_targets(*args, **kwargs):
        raise AssertionError('RUNTIME_MUST_NOT_COMPUTE_TRAIN_TARGETS')
    try:
        T.signed_targets = forbidden_targets
        np.testing.assert_array_equal(scores_context, T.score_candidates(ck, raw, mask, index_context))
    finally:
        T.signed_targets = original_targets
    try:
        T.score_candidates(dict(ck, runtime_uses_margin=True), raw, mask, index_context)
    except AssertionError:
        pass
    else:
        raise AssertionError('RUNTIME_MARGIN_CHECKPOINT_FLAG_NOT_REJECTED')
    np.testing.assert_array_equal(T.select_candidates(scores_context, mask, names), [0, 1, -1])
    assert np.isposinf(scores_context[2]).all()
    # Raw numeric equality with another expert must not grant anchor identity.
    assert transformed[0, 2, BASE_CONTEXT_DIM-1] == 0. and transformed[1, 3, BASE_CONTEXT_DIM-1] == 0.
    unsupported = {model: value.copy() for model, value in masks_context.items()}
    unsupported['R0'][0] = False
    try:
        anchor_indices(rows_context, anchors, unsupported)
    except AssertionError:
        pass
    else:
        raise AssertionError('MISSING_R0_ANCHOR_MUST_STOP')
    try:
        T.score_candidates(ck, raw, mask, np.array([-1, 1, -1]))
    except AssertionError:
        pass
    else:
        raise AssertionError('PRESENT_ROW_WITH_MINUS_ONE_ANCHOR_MUST_STOP')
    before = np.full((3, 6, 2), [10., 5.])
    after = before - np.array([2., 1.])
    result = E.hierarchical_bootstrap(before, after, ['A','A','B','B','C','C'], repeats=40)
    assert result['metrics']['translation_cm']['CI95'] == [-2., -2.]
    assert result['metrics']['rotation_deg']['CI95'] == [-1., -1.]
    assert E.guard(before, before * 1.05, [('P90', .9)])['PASS']
    failed = after.copy(); failed[2, 0] = np.inf
    assert not E.guard(before, failed, [('P90', .9)])['PASS']
    # Exercise the actual five-gate assembly and every required comparator.
    ids = [f'fixture_{j}' for j in range(6)]
    rows = [dict(id=fid, recording=('A','A','B','B','C','C')[j]) for j, fid in enumerate(ids)]
    groups = {p: ids for p in ('NATURAL99', 'CLEAN29', 'WOOD45')}
    fixture = {m: {fid: dict(available=True, translation_cm=8. if m.startswith('UNION') else 10.,
                             rotation_deg=4. if m.startswith('UNION') else 5.) for fid in ids} for m in MODELS}
    original_bootstrap = E.hierarchical_bootstrap
    try:
        # Only reduce invented-fixture repetitions; production remains2000.
        E.hierarchical_bootstrap = lambda b, a, r: original_bootstrap(b, a, r, repeats=40)
        result, _ = gates_for(fixture, rows, groups, {})
        assert result['PASS'] and len(result['gates']) == 5
        assert set(result['gates']['all_three_seeds_joint_gain']['comparisons']) == {'R0_ONLY','ORIGINAL_DIVERSE','R0','PRIOR1','FULL125'}
        for fid in ids:
            fixture['PRIOR1'][fid]['translation_cm'] = 1.
        result, _ = gates_for(fixture, rows, groups, {})
        assert not result['PASS'] and not result['gates']['all_three_seeds_joint_gain']['comparisons']['PRIOR1']
        assert all(v['PASS'] for k, v in result['gates'].items() if k != 'all_three_seeds_joint_gain')
        for fid in ids:
            fixture['PRIOR1'][fid]['translation_cm'] = 10.
        old_protocol_fixture = dict(stability_criteria=dict(tolerance_status='invented fixture only'))
        both, matched, _, original, _ = combined_gates(fixture, rows, groups, {}, old_protocol_fixture)
        assert both['PASS'] and matched['PASS'] and original['PASS']
        assert len(both['gates']) == 5 and all(g['PASS'] for g in both['gates'].values())
        for fid in ids:
            for seed in (1, 2, 3):
                fixture[f'SINGLE251_s{seed}'][fid]['translation_cm'] = 1.
        both, matched, _, original, _ = combined_gates(fixture, rows, groups, {}, old_protocol_fixture)
        assert matched['PASS'] and not original['PASS'] and not both['PASS']
        assert not both['gates']['all_three_seeds_joint_gain']['PASS']
        assert not both['gates']['all_three_seeds_joint_gain']['original_goal']['PASS']
        assert both['gates']['all_three_seeds_joint_gain']['matched_intervention']['PASS']
        for fid in ids:
            assert all(fixture[f'DIVERSE251_s{s}'][fid]['translation_cm'] == 10. for s in (1, 2, 3))
            for seed in (1, 2, 3):
                fixture[f'SINGLE251_s{seed}'][fid]['translation_cm'] = 10.
            fixture['R0_ONLY'][fid]['translation_cm'] = 1.
        both, matched, _, original, _ = combined_gates(fixture, rows, groups, {}, old_protocol_fixture)
        assert not matched['PASS'] and original['PASS'] and not both['PASS']
        assert not both['gates']['all_three_seeds_joint_gain']['PASS']
    finally:
        E.hierarchical_bootstrap = original_bootstrap
    for path in ('/tmp/real_guard/data/evaluation/fixture.json', '/tmp/real_guard/POSE_METRICS.json', '/tmp/real_guard/GEOMETRY_RESOLVED_POSE_GT.json'):
        try:
            with open(path, 'rb'):
                pass
        except AssertionError:
            continue
        raise AssertionError(('REFERENCE_GUARD_NOT_ACTIVE', path))
    assert not _REFERENCES['allowed']
    assert all(Path(path).suffix in ('.py', '.pyc') for path in READS), ('SELFCHECK_ARTIFACT_READ', sorted(READS))
    print('REAL_EVALUATOR_SELF_CHECK_PASS_INVENTED_DATA_ONLY', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'score', 'self_check', 'not_run'])
    args = parser.parse_args()
    globals()[args.stage]()


if __name__ == '__main__':
    main()
