"""Independent fixed-RBF input and mathematical review; no fitting or quality probes.

The basis is reconstructed before TRAIN labels are permitted. Its reference-free
inputs are the sealed feature cache and metadata eligibility contract. Actual
old-weight calculations check representation preservation only, not performance.
"""
import argparse
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from . import common as C
from scripts.research.pallet_pose_anchor_risk_20261001_v1 import prefit_review as P

MODELS = C.MODEL_NAMES
STATE = {'labels_allowed': False, 'reads': set()}
LOSS_RULE = 'TRAIN_LOG1P_ANCHOR_EXCESS_MARGIN_CE'
SCORE_TOLERANCE = 1e-10


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def guard():
    writable = {C.DOC/'PREFIT_REVIEW.json', C.DOC/'PREFIT_REVIEW_KO.md'}
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode = args[1]
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = isinstance(mode, str) and any(c in mode for c in 'wax+')
        writing |= bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC))
        if not path.is_relative_to(C.ROOT):
            return
        if writing:
            assert path in writable, ('PREFIT_WRITE_SCOPE', str(path))
            return
        text = str(path)
        assert not any(s in text for s in ('/annotations/', 'GEOMETRY_RESOLVED_POSE_GT',
            'AXIS_REVIEW_MANIFEST', 'GEOMETRY_SIDETABLE', '/SOURCE_VAL_', '/REAL_RESULTS',
            '/REAL_CHOICES', '/POSE_METRICS.json', '/REAL_FRAME_RESULTS', '/TRUTH_FOR_DISPLAY')), (
            'PREFIT_FORBIDDEN_REFERENCE_OR_QUALITY', text)
        if not STATE['labels_allowed']:
            assert 'SOURCE_TRAIN_LABELS' not in text, ('BASIS_REVIEW_MUST_BE_LABEL_FREE', text)
        assert path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.pt', '.pth', '.onnx'), text
        STATE['reads'].add(str(path.relative_to(C.ROOT)))
    sys.addaudithook(audit)


def identity_text(identity):
    assert len(identity) == 3 and all(isinstance(x, str) for x in identity)
    return json.dumps(list(identity), ensure_ascii=False, separators=(',', ':'))


def independent_centers(identities, contexts):
    """Independent SHA ordering and byte-distinct selection of exactly 64 rows."""
    assert contexts.dtype == np.float64 and contexts.shape == (len(identities), 189)
    assert np.isfinite(contexts).all()
    unique = {}
    for identity, vector in zip(identities, contexts):
        text = identity_text(identity)
        if text in unique:
            assert unique[text].tobytes() == np.ascontiguousarray(vector).tobytes()
        else:
            unique[text] = np.ascontiguousarray(vector)
    ordered = sorted(unique, key=lambda text: (hashlib.sha256(text.encode('utf-8')).hexdigest(), text))
    seen, chosen, center_ids = set(), [], []
    for text in ordered:
        vector = unique[text]
        if vector.tobytes() in seen:
            continue
        seen.add(vector.tobytes())
        chosen.append(vector)
        center_ids.append(json.loads(text))
        if len(chosen) == 64:
            break
    assert len(chosen) == 64, 'FEWER_THAN_64_DISTINCT_CONTEXTS_STOP'
    centers = np.stack(chosen)
    distances = []
    for i in range(64):
        for j in range(i + 1, 64):
            value = float(np.sum((centers[i] - centers[j]) ** 2, dtype=np.float64))
            assert np.isfinite(value) and value >= 0
            if value > 0:
                distances.append(value)
    assert distances, 'NO_POSITIVE_CENTER_DISTANCE_STOP'
    distances = sorted(distances)
    mid = len(distances) // 2
    width = float(distances[mid] if len(distances) % 2 else
                  (distances[mid - 1] + distances[mid]) / 2)
    assert np.isfinite(width) and width > 0
    return centers, center_ids, width, dict(unique_identity_count=len(unique),
        positive_pair_count=len(distances), center_identity_sha256=[
            hashlib.sha256(identity_text(x).encode('utf-8')).hexdigest() for x in center_ids])


def independent_map(context, valid, centers, width):
    assert context.shape == (*valid.shape, 189) and context.dtype == np.float64
    assert centers.shape == (64, 189) and centers.dtype == np.float64
    assert np.isfinite(context).all() and np.isfinite(centers).all() and 0 < width < np.inf
    out = np.zeros((*valid.shape, 253), np.float64)
    out[valid, :189] = context[valid]
    for i, j in zip(*np.nonzero(valid)):
        for k, center in enumerate(centers):
            distance = float(np.sum((context[i, j] - center) ** 2, dtype=np.float64))
            out[i, j, 189 + k] = np.exp(-distance / (2 * width))
    assert np.isfinite(out).all() and not out[~valid].any()
    assert (out[..., 189:] >= 0).all() and (out[..., 189:] <= 1).all()
    return out


def payload(centers, width, mean, std):
    return dict(schema='pallet_pose_anchor_rbf_runtime_v1', context_dim=189, rbf_dim=64,
        centers=centers.tolist(), bandwidth_squared=width,
        normalization_sha=array_sha(np.stack([mean, std])))


def expect_rejected(callback):
    try:
        callback()
    except (AssertionError, ValueError):
        return
    raise AssertionError('Invalid contract accepted')


def synthetic_review(T, B, previous):
    rng = np.random.default_rng(2026100121)
    # Duplicate identities must have equal context. Equal context under a new
    # identity is skipped; signed-zero byte distinctions remain intentional.
    ids = [[f'f{i:03d}', 'R0' if i % 2 else 'DIVERSE251_s1', T.OLD.HYP[i % 2]] for i in range(80)]
    vectors = rng.normal(size=(80, 189))
    ids += [ids[0], ['other', 'R0', T.OLD.HYP[0]]]
    vectors = np.concatenate([vectors, vectors[:1], vectors[:1]])
    centers, chosen, width, detail = independent_centers(ids, vectors)
    actual, records = B.select_centers(ids[:80] + [ids[-1]], np.r_[vectors[:80], vectors[-1:]])
    np.testing.assert_array_equal(actual, centers)
    assert [r['identity'] for r in records] == chosen
    assert B.bandwidth_squared(centers) == width
    expect_rejected(lambda: B.select_centers(ids, vectors))
    for identity in ids:
        assert B.identity_json(*identity) == identity_text(identity)
    valid = np.array([[1, 1, 1, 1], [0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 1]], bool)
    index = np.array([0, 1, 0, -1, 1], np.int64)
    raw = rng.normal(size=(5, 4, 94)).astype(np.float32)
    raw[~valid] = np.nan
    mean = rng.normal(size=94).astype(np.float32)
    std = (rng.random(94) + .5).astype(np.float32)
    basis = payload(centers, width, mean, std)
    B.validate_basis(basis, mean, std)
    x189 = P.scalar_context(raw, valid, index, mean, std)
    x = independent_map(x189, valid, centers, width)
    np.testing.assert_array_equal(T.context_inputs(raw, valid, index, mean, std), x189)
    np.testing.assert_array_equal(B.map_context(x189, valid, basis), x)
    np.testing.assert_array_equal(T.rbf_inputs(raw, valid, index, mean, std, basis), x)
    math_gap = 0.
    for i, j in zip(*np.nonzero(valid)):
        for k, center in enumerate(centers):
            d = sum(float(v) * float(v) for v in x189[i, j] - center)
            math_gap = max(math_gap, abs(math.exp(-d / (2 * width)) - x[i, j, 189 + k]))
    assert math_gap < 1e-14
    errors = np.array([[[2,3],[6,0],[1,2],[4,12]], [[np.inf,np.inf],[4,3],[8,0],[np.inf,np.inf]],
        [[2,2],[np.inf,np.inf],[np.inf,np.inf],[np.inf,np.inf]], [[np.inf,np.inf]] * 4,
        [[5,4],[6,6],[np.inf,np.inf],[1,8]]], np.float64)
    anchor = np.full((5, 2), np.inf)
    present = np.flatnonzero(valid.any(1))
    anchor[present] = errors[present, index[present]]
    scale = np.array([2., 3.])
    names = T.candidate_names('UNION_s1')
    target, safe = P.scalar_targets(errors, valid, anchor, index, scale, names)
    risk, margin = P.scalar_margin(errors, valid, anchor, index, scale, target)
    with np.errstate(all='raise'):
        np.testing.assert_array_equal(T.margin_arrays(errors, valid, anchor, index, scale, target), (risk, margin))
    old_weight = rng.normal(size=189) * .03
    embedded = np.r_[old_weight, np.zeros(64)]
    old = previous.objective(old_weight, x189, valid, target, margin)
    same = T.objective(embedded, x, valid, target, margin)
    for key in ('value', 'CE', 'unadjusted_CE', 'penalty', 'probability'):
        np.testing.assert_allclose(same[key], old[key], rtol=0, atol=1e-12)
    np.testing.assert_allclose(same['gradient'][:189], old['gradient'], rtol=0, atol=1e-12)
    weight = rng.normal(size=253) * .03
    actual = T.objective(weight, x, valid, target, margin)
    independent = P.scalar_objective(weight, x, valid, target, margin)
    for key in ('value', 'CE', 'penalty', 'gradient', 'probability'):
        np.testing.assert_allclose(actual[key], independent[key], rtol=1e-12, atol=1e-12)
    H = T.hessian(x, actual['probability'])
    separate_H = P.scalar_hessian(x, valid, independent['probability'])
    np.testing.assert_allclose(H, separate_H, rtol=1e-11, atol=1e-12)
    minimum = float(np.linalg.eigvalsh(separate_H)[0])
    assert minimum >= 1e-4 - 1e-10
    dg, dH = [], []
    for k in range(253):
        delta = np.zeros(253); delta[k] = 1e-5
        plus = P.scalar_objective(weight + delta, x, valid, target, margin)
        minus = P.scalar_objective(weight - delta, x, valid, target, margin)
        dg.append(abs((plus['value'] - minus['value']) / 2e-5 - actual['gradient'][k]))
        dH.append(float(np.abs((plus['gradient'] - minus['gradient']) / 2e-5 - H[:, k]).max()))
    assert max(dg) < 1e-8 and max(dH) < 1e-8
    empty = T.objective(weight, np.zeros((2, 4, 253)), np.zeros((2, 4), bool),
        np.full(2, -1, np.int64), np.zeros((2, 4)))
    assert empty['CE'] == 0.
    np.testing.assert_array_equal(empty['gradient'], 1e-4 * weight)
    np.testing.assert_array_equal(T.hessian(np.zeros((2, 4, 253)), empty['probability']), 1e-4*np.eye(253))
    unchanged = ('margin_arrays', 'objective', 'hessian', 'certificate', 'anchored_targets')
    for name in unchanged:
        a = ast.dump(ast.parse(inspect.getsource(getattr(T, name))))
        b = ast.dump(ast.parse(inspect.getsource(getattr(previous, name))))
        assert a == b, ('UNCHANGED_MATH_AST', name)
    assert T.LAMBDA == 1e-4 and T.MAX_ITER == 1000 and T.MAX_CALLS == 2000 and T.GAP_MAX == 1e-6
    cert = T.certificate(SimpleNamespace(success=True),
        dict(gradient=np.full(253, 1e-7), value=.11, CE=.1, penalty=.01, unadjusted_CE=.09), 2, 1)
    assert cert['PASS'] and cert['gradient_l2_squared_over_2lambda'] <= 1e-6
    ck = dict(schema=T.CHECKPOINT_SCHEMA, loss_rule=LOSS_RULE, runtime_uses_margin=False,
        feature_map=T.FEATURE_MAP, feature_dim=253, raw_feature_dim=94,
        normalization='old_float32_then_float64', names=names, bias=0., lambda_l2=1e-4,
        mean=mean.tolist(), std=std.tolist(), weight=weight.tolist(), rbf_basis=basis,
        rbf_basis_binding={'path':'toy/RBF_BASIS.json','sha256':'0'*64,'bytes':1},
        basis_SHA_bind={'path':'toy/RBF_BASIS.json','sha256':'0'*64,'bytes':1})
    expected_score = np.where(valid, np.sum(x * weight, axis=2), np.inf)
    original_margin = T.margin_arrays
    def forbidden(*args, **kwargs):
        raise AssertionError('RUNTIME_MUST_NOT_BUILD_GT_MARGIN')
    T.margin_arrays = forbidden
    try:
        scores = T.score_candidates(ck, raw, valid, index)
    finally:
        T.margin_arrays = original_margin
    np.testing.assert_allclose(scores, expected_score, rtol=1e-12, atol=1e-12)
    expect_rejected(lambda: T.score_candidates({**ck, 'runtime_uses_margin': True}, raw, valid, index))
    expect_rejected(lambda: B.validate_basis({**basis, 'bandwidth_squared': 0.}, mean, std))
    expect_rejected(lambda: B.validate_basis({**basis, 'normalization_sha': 'invalid'}, mean, std))
    return dict(PASS=True, center_SHA_order_and_byte_distinct_exact=True,
        context189_scalar_exact=True, context253_scalar_exact=True, RBF_math_exp_max_difference=math_gap,
        embedded_old_score_and_objective_equivalent=True, embedding_tolerance=1e-12,
        embedded_gradient_first189_equivalent=True, new64_gradient_not_claimed_zero=True,
        independent_gradient_max_difference=float(np.abs(actual['gradient']-independent['gradient']).max()),
        gradient_finite_difference_max=max(dg), hessian_finite_difference_max=max(dH),
        independent_hessian_max_difference=float(np.abs(H-separate_H).max()), hessian_min_eigenvalue=minimum,
        all_invalid_full_denominator_zero_CE_retained=True, runtime_does_not_build_margin=True,
        invalid_feature_zero=True, unchanged_helper_ASTs=list(unchanged),
        certificate_preserves_strong_convex_bound=True, scope='Invented fixtures only; no quality or fit.')


def feature_only_basis(B, T):
    """Reconstruct a frozen basis without opening target/error arrays."""
    C.verify(C.read(C.DOC/'BASIS_PROTOCOL_SHA.json'))
    protocol = C.read(C.DOC/'BASIS_PROTOCOL.json')
    for binding in [*protocol['inputs'].values(), *protocol['codes']]:
        C.verify(binding)
    artifact = C.read(C.DOC/'RBF_BASIS.json')
    assert artifact['complete'] and artifact['PASS'] and artifact['protocol'] == C.bind(C.DOC/'BASIS_PROTOCOL.json')
    assert artifact['source_TRAIN_only'] and not artifact['labels_read']
    assert not artifact['VAL_quality_read'] and not artifact['real_targets_read']
    assert artifact['input_bindings'] == protocol['inputs']
    basis = artifact['basis']
    assert set(protocol['inputs']) == {'source_contract','source_features','source_feature_lock','parent_train_protocol'}
    feature_binding = protocol['inputs']['source_features']
    contract_binding = protocol['inputs']['source_contract']
    parent_protocol = C.read(C.ROOT/protocol['inputs']['parent_train_protocol']['path'])
    assert protocol['inputs']['parent_train_protocol'] == C.bind(C.RISK_DOC/'TRAIN_PROTOCOL.json')
    for new, old in (('source_contract','source_contract'), ('source_features','features'), ('source_feature_lock','feature_lock')):
        assert protocol['inputs'][new] == parent_protocol['inputs'][old]
    feature_lock = C.read(C.ROOT/protocol['inputs']['source_feature_lock']['path'])
    assert feature_lock['features'] == feature_binding
    assert not feature_lock['source_targets_read'] and not feature_lock['real_targets_read']
    contract = C.read(C.ROOT/contract_binding['path'])
    assert contract['complete'] and contract['status'] == 'PASS'
    eligible = set(contract['fit_eligibility']['eligible_ids']['TRAIN'])
    assert len(eligible) == 2598
    with np.load(C.ROOT/feature_binding['path'], allow_pickle=False) as z:
        index = np.array([i for i, fid in enumerate(z['ids']) if str(fid) in eligible], np.int64)
        ids = z['ids'][index]
        assert len(ids) == 2598 and set(ids) == eligible and (z['split'][index] == 'TRAIN').all()
        anchor = z['R0_GEO_index'][index]
        features = {m: z[m+'_geo'][index] for m in ('R0','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3')}
        valid = {m: z[m+'_valid'][index] for m in features}
    raw = features['R0'][valid['R0']]
    mean, std = raw.mean(0), np.maximum(raw.std(0), np.float32(1e-6))
    assert mean.dtype == std.dtype == np.float32 and mean.shape == std.shape == (94,)
    contexts = {}
    identities, vectors = [], []
    for model in MODELS:
        parents = ['R0'] + ([] if model == 'R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
        xx = np.concatenate([features[m] for m in parents], axis=1)
        vv = np.concatenate([valid[m] for m in parents], axis=1)
        context = P.scalar_context(xx, vv, anchor, mean, std)
        np.testing.assert_array_equal(context, T.context_inputs(xx, vv, anchor, mean, std))
        contexts[model] = context
        for i, j in zip(*np.nonzero(vv)):
            if model != 'R0_ONLY' and j < 2:
                continue
            expert, hyp = T.candidate_names(model)[j].split(':')
            identities.append([str(ids[i]), expert, hyp])
            vectors.append(context[i, j])
    centers, center_ids, width, detail = independent_centers(identities, np.stack(vectors))
    np.testing.assert_array_equal(np.asarray(basis['centers'], np.float64), centers)
    assert basis['bandwidth_squared'] == width
    assert basis['normalization_sha'] == array_sha(np.stack([mean, std]))
    assert artifact['center_identities'] == center_ids
    assert artifact['center_identity_sha256'] == detail['center_identity_sha256']
    assert artifact['centers_sha'] == array_sha(centers)
    assert artifact['source_ids_sha'] == array_sha(ids)
    assert artifact['source_index_sha'] == array_sha(index)
    assert artifact['anchor_index_sha'] == array_sha(anchor)
    assert artifact['pool_candidates'] == len(identities) == detail['unique_identity_count']
    assert artifact['pool_context_sha'] == array_sha(np.stack(vectors))
    assert artifact['pool_identities_sha'] == hashlib.sha256(json.dumps(identities,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    for expert in features:
        context = contexts['R0_ONLY'] if expert == 'R0' else contexts[f'UNION_s{expert[-1]}'][:,2:]
        assert artifact['input_hashes'][expert] == dict(raw_sha=array_sha(features[expert]),
            valid_sha=array_sha(valid[expert]), context189_sha=array_sha(context))
    for record, center_id, center in zip(artifact['center_records'], center_ids, centers):
        assert record['identity'] == center_id
        assert record['canonical_json'] == identity_text(center_id)
        assert record['identity_sha256'] == hashlib.sha256(identity_text(center_id).encode()).hexdigest()
        assert record['context_sha'] == array_sha(center)
        assert identities[record['pool_index']] == center_id
        np.testing.assert_array_equal(vectors[record['pool_index']],center)
    np.testing.assert_array_equal(np.asarray(artifact['mean'],np.float32),mean)
    np.testing.assert_array_equal(np.asarray(artifact['std'],np.float32),std)
    B.validate_basis(basis, mean, std)
    assert not STATE['labels_allowed']
    report = dict(PASS=True, artifact=C.bind(C.DOC/'RBF_BASIS.json'),
        protocol=C.bind(C.DOC/'BASIS_PROTOCOL.json'), source_TRAIN_only=True,
        source_ids_sha=array_sha(ids), source_index_sha=array_sha(index),
        normalization_sha=array_sha(np.stack([mean,std])), anchor_index_sha=array_sha(anchor),
        center_array_sha=array_sha(centers), center_identities=center_ids,
        bandwidth_squared=width, detail=detail, candidate_records_before_identity_dedup=len(identities),
        target_or_error_arrays_read=False, read_paths=sorted(STATE['reads']))
    return report, dict(ids=ids, index=index, anchor=anchor, features=features, valid=valid,
        mean=mean, std=std, context189=contexts, centers=centers, width=width, basis=basis)


def actual_review(T, B, previous):
    basis_report, frozen = feature_only_basis(B, T)
    STATE['labels_allowed'] = True
    T.OLD.install_training_guard()
    parent = T.OLD.load_training_inputs()
    previous_protocol_binding = C.read(C.RISK_DOC/'TRAIN_PROTOCOL_SHA.json')
    C.verify(previous_protocol_binding)
    previous_protocol = C.read(C.RISK_DOC/'TRAIN_PROTOCOL.json')
    for key, binding in parent['protocol']['inputs'].items():
        assert previous_protocol['inputs'][key] == binding
    C.verify(previous_protocol['inputs']['prefit_review'])
    prior = C.read(C.ROOT/previous_protocol['inputs']['prefit_review']['path'])
    assert prior['complete'] and prior['PASS']
    old_inputs = prior['inputs']
    assert old_inputs['source_ids_sha'] == array_sha(parent['ids']) == array_sha(frozen['ids'])
    assert old_inputs['source_index_sha'] == array_sha(parent['source_index']) == array_sha(frozen['index'])
    assert old_inputs['normalization_sha'] == parent['normalization_sha'] == basis_report['normalization_sha']
    np.testing.assert_array_equal(parent['mean'], frozen['mean'])
    np.testing.assert_array_equal(parent['std'], frozen['std'])
    with np.load(C.ROOT/parent['protocol']['inputs']['train_labels']['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], parent['ids'])
        anchor_errors = np.stack([z['R0_GEO_T_cm'], z['R0_GEO_R_deg']], axis=-1)
    index = frozen['anchor']
    assert array_sha(index) == old_inputs['anchor_index_sha']
    assert array_sha(anchor_errors) == old_inputs['anchor_errors_sha']
    assert np.array_equal(index >= 0, np.isfinite(anchor_errors).all(1))
    assert parent['ids'][index < 0].tolist() == ['TEX__shard_04_f0110']
    assert array_sha(parent['scale']) == old_inputs['scale_sha']
    results = {}
    for model in MODELS:
        parents = ['R0'] + ([] if model == 'R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
        names = T.candidate_names(model)
        raw = np.concatenate([parent['features'][m] for m in parents], axis=1)
        valid = np.concatenate([parent['valid'][m] for m in parents], axis=1)
        errors = np.concatenate([parent['errors'][m] for m in parents], axis=1)
        target, safe = P.scalar_targets(errors, valid, anchor_errors, index, parent['scale'], names)
        t2, s2 = T.anchored_targets(errors, valid, anchor_errors, index, parent['scale'], names)
        np.testing.assert_array_equal(target,t2); np.testing.assert_array_equal(safe,s2)
        with np.errstate(all='raise'):
            risk, margin = P.scalar_margin(errors, valid, anchor_errors, index, parent['scale'], target)
            r2, m2 = T.margin_arrays(errors, valid, anchor_errors, index, parent['scale'], target)
        np.testing.assert_array_equal(risk, r2); np.testing.assert_array_equal(margin, m2)
        x189 = P.scalar_context(raw, valid, index, parent['mean'], parent['std'])
        np.testing.assert_array_equal(x189, frozen['context189'][model])
        x = independent_map(x189, valid, frozen['centers'], frozen['width'])
        runtime = T.rbf_inputs(raw, valid, index, parent['mean'], parent['std'], frozen['basis'])
        np.testing.assert_array_equal(x, runtime)
        unchanged = dict(target_sha=array_sha(target), safe_mask_sha=array_sha(safe),
            original_valid_sha=array_sha(valid), unscaled_errors_sha=array_sha(errors),
            raw_features_sha=array_sha(raw), risk_sha=array_sha(risk), margin_sha=array_sha(margin),
            anchor_errors_sha=array_sha(anchor_errors), anchor_index_sha=array_sha(index))
        for key, value in unchanged.items():
            assert value == prior['models'][model][key], (model, key)
        assert array_sha(x189) == prior['models'][model]['context_sha']
        assert not margin[valid.any(1), target[valid.any(1)]].any()
        np.testing.assert_array_equal(risk > 0, valid & ~safe)
        # Old trained weights are used only to verify the old feature path is
        # represented. No label-dependent objective, selection or error metric.
        fit_binding = C.bind(C.RISK_DOC/f'FIT_{model}.json')
        fit = C.read(C.ROOT/fit_binding['path'])
        C.verify(fit['checkpoint'])
        old_ck = C.read(C.ROOT/fit['checkpoint']['path'])
        assert fit['complete'] and fit['protocol'] == previous_protocol_binding
        assert old_ck['schema'] == previous.CHECKPOINT_SCHEMA
        np.testing.assert_array_equal(np.array(old_ck['mean'], np.float32), parent['mean'])
        np.testing.assert_array_equal(np.array(old_ck['std'], np.float32), parent['std'])
        w = np.asarray(old_ck['weight'], np.float64)
        assert w.shape == (189,)
        old_score = np.einsum('nkd,d->nk', x189, w)
        expanded_score = np.einsum('nkd,d->nk', x, np.r_[w, np.zeros(64)])
        maximum = float(np.abs(old_score-expanded_score).max())
        assert maximum <= SCORE_TOLERANCE
        old_runtime = previous.score_candidates(old_ck, raw, valid, index)
        embedded_ck = {**old_ck, 'schema':T.CHECKPOINT_SCHEMA, 'feature_map':T.FEATURE_MAP,
            'feature_dim':253, 'weight':np.r_[w,np.zeros(64)].tolist(), 'rbf_basis':frozen['basis'],
            'rbf_basis_binding':basis_report['artifact'], 'basis_SHA_bind':basis_report['artifact']}
        embedded_runtime = T.score_candidates(embedded_ck, raw, valid, index)
        np.testing.assert_array_equal(old_runtime, np.where(valid, old_score, np.inf))
        np.testing.assert_array_equal(embedded_runtime, np.where(valid, expanded_score, np.inf))
        np.testing.assert_allclose(embedded_runtime,old_runtime,rtol=0,atol=SCORE_TOLERANCE)
        results[model] = dict(**unchanged, context_sha=array_sha(x), base_context_sha=array_sha(x189),
            basis_SHA_bind=basis_report['artifact'], feature_dim=253, raw_feature_dim=94,
            shape=list(valid.shape), frames=2598, available_anchor_rows=2597, failed_rows=1,
            original_valid_candidates=int(valid.sum()), positive_margin_candidates=int((margin>0).sum()),
            scalar_target_context_risk_margin_exact=True, scalar_RBF_exact=True,
            previous_target_context_valid_error_hashes_exact=True,
            old_weight_embedding=dict(checkpoint=fit['checkpoint'], receipt=fit_binding,
                score_count=int(old_score.size), finite_valid_score_count=int(valid.sum()),
                maximum_absolute_difference=maximum, tolerance=SCORE_TOLERANCE,
                runtime_API_paths_verified=True, label_quality_or_selection_computed=False),
            invalid_candidates_zero=True, target_margin_zero=True)
    inputs = dict(parent_train_protocol=parent['binding'], risk_train_protocol=previous_protocol_binding,
        previous_prefit=previous_protocol['inputs']['prefit_review'],
        source_feasibility=previous_protocol['inputs']['source_feasibility'],
        features=parent['protocol']['inputs']['features'], train_labels=parent['protocol']['inputs']['train_labels'],
        source_contract=parent['protocol']['inputs']['source_contract'],
        source_ids_sha=array_sha(parent['ids']), source_index_sha=array_sha(parent['source_index']),
        normalization_sha=parent['normalization_sha'], scale=parent['scale'].tolist(),
        scale_sha=array_sha(parent['scale']), anchor_index_sha=array_sha(index),
        anchor_errors_sha=array_sha(anchor_errors), failed_ids=parent['ids'][index<0].tolist(),
        basis=basis_report['artifact'], basis_protocol=basis_report['protocol'])
    return results, inputs, basis_report


def run(write):
    sys.dont_write_bytecode = True
    guard()
    from . import convex_train as T
    from . import basis as B
    from scripts.research.pallet_pose_anchor_risk_20261001_v1 import convex_train as previous
    assert T.OLD.READS is None
    codes = [C.bind(T.__file__), C.bind(B.__file__)]
    with threadpool_limits(limits=1):
        synthetic = synthetic_review(T, B, previous)
        if not write:
            print(json.dumps(synthetic, indent=2)); return
        models, inputs, basis_report = actual_review(T, B, previous)
    assert codes == [C.bind(T.__file__), C.bind(B.__file__)], 'CODE_CHANGED_DURING_REVIEW'
    result = dict(complete=True, PASS=True, created_at=C.now(), source_TRAIN_only=True,
        frames=2598, available_anchor_rows=2597, failed_rows_retained=1, feature_dim=253, raw_feature_dim=94,
        feature_map=T.FEATURE_MAP, loss_rule=LOSS_RULE, runtime_uses_margin=False,
        models=models, inputs=inputs, basis_review=basis_report, basis_SHA_bind=basis_report['artifact'], synthetic=synthetic,
        trainer=codes[0], basis_operator=codes[1], reviewer=C.bind(__file__),
        source_TRAIN_cached_label_values_read=True, raw_source_reference_reads=0,
        VAL_quality_read=False, real_targets_read=False, new_fits=0, optimizer_steps=0,
        actual_data_objective_trials=0, actual_data_policy_probes=0, actual_data_old_weight_embedding_score_checks=True,
        new_reference_metric_calculations=0, image_forwards=0, new_PnP_calls=0,
        read_paths=sorted(STATE['reads']), training_guard_read_paths=sorted(set(T.OLD.READS)),
        scope='Fixed input/basis/supervision verification and invented math; no new method quality calculation.',
        method_success=False, goal_complete=False)
    md = f'''# 고정 RBF64 특징의 학습 전 독립 검산

**입력·수학 검산 PASS. 새 학습이나 성능 평가는 실행하지 않았다.** 변경은 기존189 context 뒤에 고정 RBF64를 추가한253 특징 하나다. 기존94 FP32 정규화→FP64 context, target·risk·margin·valid·2,598행 분모·λ=1e−4와 solver 예산은 유지한다.

센터64개는 참조와 TRAIN label을 열기 전에 metadata로 고정된 eligible TRAIN 및 기존 feature cache만으로 독립 재현했다. `[frameID,expert,hypothesis]`의 UTF-8 canonical JSON을 SHA256 순서로 정렬하고, 동일 identity를 중복 제거한 다음 float64 context189의 bytes가 다른 최초64개를 선택했다. 양의 센터 쌍별 제곱거리 median을 bandwidth²로 고정했다. 실제 센터·identity·normalization·bandwidth는 잠금 artifact와 정확히 같았다. 해당 단계에서 source 오류·target·VAL 품질·실사 참조를 읽지 않았으며, 단계별 접근 경계와 해시를 JSON에 기록했다.

후보와 센터를 하나씩 순회한 별도 RBF 구현이 실제253 map과 정확히 일치했다. all-invalid 실패1행은 anchor=-1과 zero253 특징을 유지한다. 유효 후보가 있는데 anchor가 없으면 중단하며, 원래 invalid mask를 바꾸지 않는다. TRAIN label 접근을 허용한 이후에는 기존 scalar target·risk·margin으로 직전 risk PREFIT의 모든 supervision hash가 그대로임을 대조했다.

실제 네 이전189 checkpoint에0을64개 붙여 전체 TRAIN의 동일 입력 score가 유지되는지 확인했다. 허용 절대차는 {SCORE_TOLERANCE:g}이며 모델별 최대차는 JSON에 기록했다. 이 계산은 표현 경로 보존 확인뿐이며 실제 label-dependent objective·선택률·T/R 성능을 계산하지 않았다. 253 reduction 순서 차이 때문에 수학적 동등성과 부동소수 허용오차를 구분한다.

목적식·margin·target·Hessian·certificate의 AST를 직전 구현과 대조했다. 합성 fixture에서 zero64 embedding의 목적값과 앞189 gradient가 유지됨을 확인했으며, 새64 gradient가0이라고 주장하지 않는다. 독립 scalar objective/covariance Hessian 및253방향 중앙차분 최대 오차는 gradient {synthetic['gradient_finite_difference_max']:.3g}, Hessian {synthetic['hessian_finite_difference_max']:.3g}였다. 최소 Hessian 고유값은 {synthetic['hessian_min_eigenvalue']:.9g}로 λ-strong convexity 계약을 지킨다. runtime에서 margin 생성함수를 오류로 바꿔도 score가 동작하며, runtime에 GT 가산항이나 안전 mask를 넣지 않는다.

이 검산은 새로운 비선형 특징이 T/R를 개선한다는 증거가 아니다. 후속 학습과 고정 source45 및 실사 두 비교 묶음의5개 AND 조건은 별도 실행·판정이 필요하다. 새 fit·optimizer step·VAL/실사 품질 조회·forward·PnP·threshold sweep은0회다.

[독립 검산 JSON](PREFIT_REVIEW.json) · [고정 basis](RBF_BASIS.json)
'''
    C.save(C.DOC/'PREFIT_REVIEW_KO.md', md)
    result['note'] = C.bind(C.DOC/'PREFIT_REVIEW_KO.md')
    C.save(C.DOC/'PREFIT_REVIEW.json', result)
    print('RBF_PREFIT_REVIEW_PASS', C.bind(C.DOC/'PREFIT_REVIEW.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('selfcheck', 'review'))
    run(parser.parse_args().stage == 'review')
