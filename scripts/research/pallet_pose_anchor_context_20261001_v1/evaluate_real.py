"""Lock anchor-context whole-pose real routes after source PASS, then score.

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
BASELINES = ('R0', 'PRIOR1', 'FULL125', 'DIVERSE251_s1', 'DIVERSE251_s2', 'DIVERSE251_s3')
MODELS = LEARNED + BASELINES
HYP = ('long-face-front', 'short-face-front')
RAW_FEATURE_DIM = 94
FEATURE_DIM = 189
FEATURE_MAP = 'normalized94_abs_anchor_delta94_identity1'
CHECKPOINT_SCHEMA = 'pallet_pose_anchor_context_linear189_v1'
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
    E = old_evaluation()
    return list(dict.fromkeys([Path(__file__), Path(C.__file__), Path(C.U.__file__),
        Path(T.__file__), Path(T.OLD.__file__), Path(T.PREV.__file__), Path(T.PREV.C.__file__),
        Path(E.__file__), Path(E.C.__file__),
        Path(E.C.D.__file__), Path(E.C.D.M.__file__), Path(E.C.D.O.__file__),
        Path(E.C.D.O.D.__file__), Path(E.C.D.O.D.Pose.__file__)]))


def real_protocol_spec():
    """Root may call this input-only builder before sealing REAL_PROTOCOL."""
    parent, _, source_gate = verify_training_and_source()
    return dict(schema='pallet_pose_anchor_context_learned_real_v1', complete=True,
                parent_protocol=C.bind(C.DOC / 'TRAIN_PROTOCOL.json'),
                source_val_gate=source_gate,
                models=list(LEARNED), baselines=list(BASELINES), frames=173,
                real_evaluation=parent['real_evaluation'],
                inputs={key: C.bind(path) for key, path in input_paths().items()},
                codes=[C.bind(path) for path in operator_paths()],
                whole_pose_selection=True, image_forwards=0, new_PnP_solves=0,
                source_PASS_required=True, reference_values_before_routing=False,
                target_rule=parent['target_rule'], runtime_safe_mask=False,
                raw_feature_dim=RAW_FEATURE_DIM, feature_dim=FEATURE_DIM, feature_map=FEATURE_MAP,
                normalization='old_float32_then_float64',
                feature_order='Normalize raw94 in float32; promote to float64; concatenate z94, abs(z-z_R0_GEO)94, candidate-identity1; no extra normalization.',
                anchor_rule=ANCHOR_RULE, missing_anchor_rule=MISSING_ANCHOR_RULE,
                fallback='R0_GEO_if_available_else_failure',
                feature_validity='Original frame-valid broadcast to2, intersect cached per-hypothesis pose availability.',
                seed_meaning='Three frozen refiner seeds; R0_ONLY fitted once and repeated as a shared control, not three independent fits.')


def verify_training_and_source():
    from . import convex_train as T
    parent = C.protocol('TRAIN_PROTOCOL')
    assert T.FEATURE_DIM == FEATURE_DIM and T.FEATURE_MAP == FEATURE_MAP and T.CHECKPOINT_SCHEMA == CHECKPOINT_SCHEMA
    assert parent['raw_feature_dim'] == RAW_FEATURE_DIM and parent['feature_dim'] == FEATURE_DIM
    assert parent['feature_map'] == FEATURE_MAP and parent['normalization'] == 'old_float32_then_float64'
    pbind = C.bind(C.DOC / 'TRAIN_PROTOCOL.json')
    complete_path = C.DOC / 'TRAINING_COMPLETE.json'
    complete = C.read(complete_path)
    assert complete['complete'] and complete['protocol'] == pbind
    assert complete['models'] == list(LEARNED) and complete['fit_count'] == 4
    assert complete['all_certified'] and complete['source_TRAIN_only']
    for key, expected in [('raw_feature_dim', RAW_FEATURE_DIM), ('feature_dim', FEATURE_DIM), ('feature_map', FEATURE_MAP)]:
        assert complete[key] == parent[key] == expected
    assert complete['target_rule'] == parent['target_rule'] == T.TARGET_RULE
    assert complete['source_feasibility'] == parent['inputs']['source_feasibility']
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
        for key in ('raw_feature_dim', 'feature_dim', 'feature_map'):
            assert receipt[key] == parent[key]
        assert receipt['source_TRAIN_only'] and not receipt['VAL_quality_read'] and not receipt['real_targets_read']
        assert receipt['checkpoint'] == C.bind(C.RAW / 'fits' / model / 'final.json')
        verify_all(receipt)
        ck = C.read(C.ROOT / receipt['checkpoint']['path'])
        assert ck['schema'] == CHECKPOINT_SCHEMA and ck['model'] == model
        assert ck['raw_feature_dim'] == RAW_FEATURE_DIM and ck['feature_dim'] == FEATURE_DIM
        assert ck['feature_map'] == FEATURE_MAP and ck['normalization'] == 'old_float32_then_float64'
        assert ck['names'] == T.candidate_names(model) and ck['protocol'] == pbind
        assert ck['certificate'] == receipt['certificate'] and ck['bias'] == 0. and ck['lambda_l2'] == 1e-4
        assert ck['target_rule'] == receipt['target_rule'] == parent['target_rule'] == T.TARGET_RULE
        assert ck['runtime_safe_mask'] is False and ck['loss_uses_original_valid_mask'] is True
        assert receipt['source_feasibility'] == parent['inputs']['source_feasibility']
        cert = ck['certificate']
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
        assert mean.shape == std.shape == (RAW_FEATURE_DIM,) and weight.shape == (FEATURE_DIM,)
        assert np.isfinite(mean).all() and np.isfinite(std).all() and np.isfinite(weight).all()
        assert (std >= np.float32(1e-6)).all()
        assert T.OLD.array_sha(np.stack([mean, std])) == ck['normalization_sha'] == receipt['normalization_sha']
        assert T.OLD.array_sha(weight) == receipt['final_weight_sha']
        start = C.read(C.ROOT / receipt['START']['path'])
        for key in ('raw_feature_dim', 'feature_dim', 'feature_map'):
            assert start[key] == ck[key] == parent[key]
        assert start['anchor_index_sha'] == receipt['anchor_index_sha'] == ck['anchor_index_sha'] == complete['anchor_index_sha']
        assert start['model'] == model and start['protocol'] == pbind and start['target_rule'] == T.TARGET_RULE
        assert start['target_sha'] == receipt['target_sha'] and start['safe_mask_sha'] == receipt['safe_mask_sha']
        assert start['source_feasibility'] == receipt['source_feasibility']
        trace = [json.loads(line) for line in (C.ROOT / receipt['trace']['path']).read_text().splitlines()]
        calls = [line for line in trace if line['event'] == 'objective']
        iterations = [line for line in trace if line['event'] == 'iteration']
        assert [line['call'] for line in calls] == list(range(1, receipt['objective_calls'] + 1))
        assert [line['iteration'] for line in iterations] == list(range(1, receipt['iterations'] + 1))
        assert calls[-1]['weight_sha'] == receipt['final_weight_sha']
        assert calls[-1]['objective'] == cert['objective_value']
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
    verify_all(lock)
    from . import convex_train as T
    _, masks = feature_arrays(C.ROOT / protocol['inputs']['features']['path'], rows, poses)
    index = anchor_indices(rows, poses, masks)
    assert lock['anchor_index_sha'] == T.OLD.array_sha(index)
    choices = C.read(ROUTE)
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
                          scores=[float(s) if np.isfinite(s) else None for s in scores[j]])
            pose = pose_for(choice, poses, row['id'])
            records[model][row['id']] = dict(choice, pose=pose, pose_available=bool(pose['available']))
        counts[model] = dict(Counter(r['candidate_name'] or 'R0_GEO_FALLBACK' for r in records[model].values()))
    choices = dict(ids=[r['id'] for r in rows], models=list(LEARNED), records=records,
                   real_reference_values_read=False, whole_pose_selection=True,
                   feature_map=FEATURE_MAP, raw_feature_dim=RAW_FEATURE_DIM, feature_dim=FEATURE_DIM,
                   anchor_rule=ANCHOR_RULE,
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
    stability, hierarchy = gates_for(metrics, rows, groups, protocol['real_evaluation'])
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
    C.save(detail_path, dict(hierarchical=hierarchy, by_recording=by_recording))
    C.save(RESULTS, dict(complete=True, created_at=C.now(), models=list(MODELS),
        protocol=C.bind(REAL_PROTOCOL), routing_lock=C.bind(ROUTE_LOCK),
        populations={p: len(ids) for p, ids in groups.items()}, summaries=summaries,
        stability=stability, hierarchy=hierarchy, full_frame_rows=len(csv_rows),
        baseline_metric_parity_checks=len(BASELINES)*len(rows),
        row_failure_policy='All173 retained; failed poses have+infinity full-population errors, explicitly conditional medians plus per-seed failure guards.',
        artifacts=[C.bind(p) for p in (metric_path, csv_path, detail_path, C.DOC/'REAL_REFERENCE_BINDINGS.json')],
        image_forwards=0, new_PnP_solves=0, new_fits=0, wall_seconds=time.monotonic()-started))
    print('ANCHOR_CONTEXT_REAL_STABILITY', stability['verdict'], flush=True)


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
        target_rule='R0_GEO_PARETO_NONREGRESSION_FIXED_TRAIN_MINMAX',
        raw_feature_dim=RAW_FEATURE_DIM, feature_dim=FEATURE_DIM, feature_map=FEATURE_MAP,
        anchor_rule=ANCHOR_RULE, runtime_safe_mask=False,
        reason='A failed predeclared source gate forbids learned real routing or performance evaluation. The earlier real feasibility oracle is a separate diagnostic, not this method result.',
        unused_evaluator=C.bind(Path(__file__)), invented_selfcheck_scope='Analytic fixtures only; no learned real result is implied.')
    C.save(C.DOC / 'REAL_EVALUATION_NOT_RUN.json', out)
    C.save(C.DOC / 'REAL_EVALUATION_NOT_RUN_KO.md', '\n'.join([
        '# 새 189차원 anchor context 모델의 실사 평가는 실행하지 않음', '',
        f"사전 {stage} 판정이 {gate['checks_passed']}/{gate['checks_total']}로 실패하여 실사 learned routing과 성능 평가를 실행하지 않았다.", '',
        '[미실행 기록](REAL_EVALUATION_NOT_RUN.json)은 원본 실패 gate와 금지된 실사 산출물의 부재를 연결한다. REAL_PROTOCOL·후보 선택·routing lock·실사 점수·평가 CSV는 생성되지 않았다.', '',
        '이전 실사 anchor oracle은 GT 기반 가능성 진단이다. 새 모델의 예측 성능이나 목표 달성을 대신하지 않는다. 새 실사 GT annotation, 이미지 forward, PnP, 실사 metric 계산은0회다.', '']))
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
    transformed = T.context_inputs(raw, mask, index_context, mean, std)
    assert transformed.dtype == np.float64 and transformed.shape == (3, 4, FEATURE_DIM)
    np.testing.assert_array_equal(transformed[2], 0.)
    np.testing.assert_array_equal(transformed[:2, :, -1], [[1., 0., 0., 0.], [0., 1., 0., 0.]])
    np.testing.assert_array_equal(transformed[:2, :, RAW_FEATURE_DIM], [[0., 2., 0., 2.], [2., 0., 2., 0.]])
    weight = np.zeros(FEATURE_DIM, np.float64)
    weight[RAW_FEATURE_DIM] = 1.
    ck = dict(schema=CHECKPOINT_SCHEMA, feature_map=FEATURE_MAP, feature_dim=FEATURE_DIM,
              raw_feature_dim=RAW_FEATURE_DIM, normalization='old_float32_then_float64',
              names=names, bias=0., lambda_l2=1e-4, mean=mean, std=std, weight=weight)
    scores_context = T.score_candidates(ck, raw, mask, index_context)
    np.testing.assert_array_equal(T.select_candidates(scores_context, mask, names), [0, 1, -1])
    assert np.isposinf(scores_context[2]).all()
    # Raw numeric equality with another expert must not grant anchor identity.
    assert transformed[0, 2, -1] == 0. and transformed[1, 3, -1] == 0.
    unsupported = {model: value.copy() for model, value in masks_context.items()}
    unsupported['R0'][0] = False
    try:
        anchor_indices(rows_context, anchors, unsupported)
    except AssertionError:
        pass
    else:
        raise AssertionError('MISSING_R0_ANCHOR_MUST_STOP')
    try:
        T.context_inputs(raw, mask, np.array([-1, 1, -1]), mean, std)
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
