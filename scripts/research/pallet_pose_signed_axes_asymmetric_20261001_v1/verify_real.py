"""Independent asymmetric signed-direction271 real routes, physical C2 errors, and both stability families.

Run only after root confirms REAL_RESULTS complete. No evaluator/pose helper is
imported; no images, PnP, fitting, or neural forwards. Raw reference access is
blocked until every input-only route is independently reproduced. This audit
checks geometry-derived reused-DEV references, not independent physical truth.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
from . import verify_source as S
from . import convex_train as T

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments' / HERE.name
RAW = ROOT / 'data/pallet/results' / HERE.name
STABLE_DOC = ROOT / '_docs/experiments/pallet_pose_stable_improvement_20261001_v1'
STABLE_RAW = ROOT / 'data/pallet/results/pallet_pose_stable_improvement_20261001_v1'
HYP = ('long-face-front', 'short-face-front')
LEARNED = ('R0_ONLY', 'UNION_s1', 'UNION_s2', 'UNION_s3')
BASELINES = ('R0', 'PRIOR1', 'FULL125', 'DIVERSE251_s1', 'DIVERSE251_s2', 'DIVERSE251_s3',
             'SINGLE251_s1', 'SINGLE251_s2', 'SINGLE251_s3')
MODELS = LEARNED + BASELINES
METRICS = ('translation_cm', 'rotation_deg')
STATE = {'references_allowed': False, 'reads': set(), 'binding_hash_only': False}

def read(path):
    return json.loads(Path(path).read_text())

def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)

def verify(binding):
    actual = bind(ROOT / binding['path'])
    assert actual['sha256'] == binding['sha256'], binding['path']
    assert 'bytes' not in binding or actual['bytes'] == binding['bytes']

def bindings(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            yield value
        else:
            for child in value.values():
                yield from bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from bindings(child)

def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()

def model_parts(model):
    return ['R0'] + ([] if model == 'R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])

def names_for(model):
    return [parent + ':' + hypothesis for parent in model_parts(model) for hypothesis in HYP]

def tie_key(name):
    parent, hypothesis = name.split(':')
    return parent != 'R0', hypothesis, parent

def selected_pose(choice, records, fid):
    if choice['fallback']:
        assert choice['candidate_index'] == -1 and choice['candidate_name'] is None
        return records['R0'][fid]['GEO_pose']
    parent, hypothesis = choice['candidate_name'].split(':')
    assert parent == choice['parent'] and hypothesis == choice['hypothesis']
    pool = [h['pose'] for h in records[parent][fid]['hypotheses'] if h['name'] == hypothesis]
    assert len(pool) == 1 and pool[0]['available']
    return pool[0]

# Pure independent scalar map; importing this verifier never installs a source hook.
independent_context=S.independent_context
independent_scores=S.independent_scores
independent_axes=S.independent_axes

def verify_bound(binding):
    """Hashing historical image/weight bytes is allowed; decoding/loading is not."""
    previous = STATE['binding_hash_only']
    STATE['binding_hash_only'] = True
    try:
        verify(binding)
    finally:
        STATE['binding_hash_only'] = previous


def verify_all(value):
    for binding in bindings(value):
        verify_bound(binding)


def install_guard():
    writable = {DOC/'REAL_VERIFICATION.json', DOC/'REAL_VERIFICATION_KO.md'}
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(ROOT):
            return
        text = str(path)
        mode = args[1]
        if isinstance(mode, str) and any(c in mode for c in 'wax+'):
            assert path in writable, ('WRITE_SCOPE', text)
            return
        if not STATE['references_allowed']:
            assert not any(x in text for x in ('/annotations/', 'GEOMETRY_RESOLVED_POSE_GT',
                'AXIS_REVIEW_MANIFEST', '/POSE_METRICS.json', '/REAL_RESULTS.json',
                '/REAL_FRAME_RESULTS.csv', '/REAL_DETAILED_COMPARISONS.json',
                '/TRUTH_FOR_DISPLAY')), ('BEFORE_ROUTE_VERIFICATION', text)
        if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.pt', '.pth', '.onnx'):
            assert STATE['binding_hash_only'], ('DECODE_OR_MODEL_LOAD_FORBIDDEN', text)
        STATE['reads'].add(str(path.relative_to(ROOT)))
    sys.addaudithook(hook)




def verify_fit(model,receipt_binding,receipt,protocol,train_protocol,lock,complete,prefit,basis_artifact):
    verify_all(receipt)
    assert receipt['complete'] and receipt['model']==model
    assert receipt['protocol']==protocol['parent_protocol']
    assert lock['fits'][model]==dict(receipt=receipt_binding,checkpoint=receipt['checkpoint'])
    ck=read(ROOT/receipt['checkpoint']['path']);start=read(ROOT/receipt['START']['path'])
    assert ck['schema']=='pallet_pose_signed_axes_asymmetric_linear271x2_v1'
    assert ck['model']==start['model']==model and ck['names']==start['names']==names_for(model)
    assert ck['protocol']==start['protocol']==protocol['parent_protocol']
    assert ck['certificate']==receipt['certificate']
    for artifact in (ck,start,receipt):
        for key,value in S.METHOD_FIELDS.items():assert artifact[key]==value
        assert artifact['basis_SHA_bind']==protocol['basis_SHA_bind']
        assert artifact['direction_receipt_binding']==protocol['direction_receipt_binding']
        assert artifact['direction_normalization_sha']==protocol['direction_normalization_sha']
        assert artifact['anchor_index_sha']==complete['anchor_index_sha']
        for key in S.HASH_KEYS:
            assert artifact[key]==prefit['models'][model][key]==complete[key+'_by_model'][model]
    assert ck['rbf_basis']==basis_artifact['basis']
    assert ck['rbf_basis_binding']==ck['basis_SHA_bind']==protocol['basis_SHA_bind']
    S.independent_basis(ck)
    mean,std=np.asarray(ck['mean'],np.float32),np.asarray(ck['std'],np.float32)
    assert array_sha(np.stack([mean,std]))==ck['normalization_sha']==receipt['normalization_sha']
    old=read(ROOT/protocol['direction_receipt_binding']['path'])
    assert old['complete'] and old['PASS'] and old['normalization']['array_sha']==protocol['direction_normalization_sha']
    for key in ('mean','std'):
        np.testing.assert_array_equal(np.asarray(ck['direction_'+key],np.float32),np.asarray(old['normalization'][key+'18'],np.float32))
    norm18=np.stack([np.asarray(ck['direction_mean'],np.float32),np.asarray(ck['direction_std'],np.float32)])
    assert array_sha(norm18)==ck['direction_normalization_sha']
    weight=np.asarray(ck['weight'],np.float64)
    assert weight.shape==(271,2) and np.isfinite(weight).all()
    assert array_sha(weight)==receipt['final_weight_sha']
    assert start['initial_weight_sha']==array_sha(np.zeros((271,2),np.float64))
    assert ck['lambda_l2']==1e-4 and ck['bias']==0. and ck['loss_uses_original_valid_mask']
    assert start['solver']==train_protocol['solver']==T.solver_config()
    cert=ck['certificate'];assert cert['PASS'] and cert['optimizer_success']
    for key in ('loss_rule','sign_rule','sign_coefficient','solver_rule','underprediction_coefficient','underprediction_cost','overprediction_cost','underprediction_zero_curvature'):
        assert cert[key]==S.METHOD_FIELDS[key]
    assert cert['gradient_linf']<=cert['gradient_linf_max']==1e-8
    assert cert['gradient_l2_squared_over_2lambda']<=cert['max_gap_upper_bound']==1e-6
    np.testing.assert_allclose(cert['gradient_l2_squared_over_2lambda'],cert['gradient_l2']**2/(2e-4),rtol=1e-12,atol=1e-18)
    assert complete['final_accepted_call_by_model'][model]==cert['final_accepted_call']
    assert complete['final_objective_components_by_model'][model]=={k:cert[k] for k in ('objective_value','Huber','Huber_symmetric','Huber_underprediction','Sign_logistic','L2_penalty')}
    trace=[json.loads(line) for line in (ROOT/receipt['trace']['path']).read_text().splitlines()]
    S.independent_trace(trace,start,receipt,ck)
    for row in trace:
        for key,value in S.METHOD_FIELDS.items():assert row[key]==value
        for key in S.HASH_KEYS:assert row[key]==ck[key]
        for key in ('basis_SHA_bind','direction_receipt_binding','direction_normalization_sha','anchor_index_sha'):assert row[key]==ck[key]
    return ck


def verify_direction_inputs(protocol,train_protocol,lock,rows,poses,features,valid,anchor,feature_names):
    receipt_binding=lock['direction_inputs'];verify_bound(receipt_binding)
    assert receipt_binding==bind(DOC/'REAL_DIRECTION_INPUTS.json')
    receipt=read(ROOT/receipt_binding['path'])
    assert receipt['complete'] and receipt['PASS'] and receipt['scope']=='REAL'
    assert receipt['frames']==173 and receipt['dimension']==18
    assert receipt['protocol']==lock['protocol'] and receipt['rule']==S.METHOD_FIELDS['direction_rule']
    assert receipt['parity']==dict(atol_px=1e-4,rtol=1e-6)
    assert receipt['new_PnP_solves']==receipt['image_forwards']==receipt['fits_executed']==0
    assert receipt['source_label_values_read'] is receipt['real_reference_values_read'] is False
    assert receipt['normalization_recomputed'] is False and receipt['additional_padding']==0
    assert receipt['training_verification']==bind(DOC/'TRAIN_CONVERGENCE.json')
    inp=protocol['inputs'];parents=('R0','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3')
    expected={**{key:inp[key] for key in ('features','attribution','prediction_lock','pose_lock','poses','metadata')},
        **{'prediction_'+model:inp['prediction_'+model] for model in parents},
        'training_complete':bind(DOC/'TRAINING_COMPLETE.json'),'training_verification':bind(DOC/'TRAIN_CONVERGENCE.json'),
        'direction_receipt':protocol['direction_receipt_binding']}
    assert receipt['inputs']==expected
    verify_all(receipt)
    assert all(item in protocol['codes'] for item in receipt['operators'])
    assert receipt['directions']==lock['directions']==bind(RAW/'REAL_DIRECTIONS.npz')
    for key in ('direction_receipt_binding','direction_normalization_sha'):
        assert receipt[key]==lock[key]==protocol[key]
    assert protocol['direction_receipt_binding']==train_protocol['inputs']['direction_receipt']
    assert protocol['direction_normalization_sha']==train_protocol['direction_normalization_sha']
    ids=[row['id'] for row in rows]
    with np.load(ROOT/lock['directions']['path'],allow_pickle=False) as z:
        assert set(z.files)=={'ids','source_index','anchor_index'}|{m+s for m in parents for s in ('_direction18','_valid')}
        assert z['ids'].tolist()==ids
        np.testing.assert_array_equal(z['source_index'],np.arange(173))
        np.testing.assert_array_equal(z['anchor_index'],anchor)
        directions={m:z[m+'_direction18'].copy() for m in parents}
        for m in parents:np.testing.assert_array_equal(z[m+'_valid'],valid[m])
    px=[feature_names.index(f'residual{k}_px') for k in range(9)]
    normalized=[feature_names.index(f'residual{k}_bboxnorm') for k in range(9)]
    prediction_lock=read(ROOT/inp['prediction_lock']['path'])
    audit={}
    for model in parents:
        saved=directions[model]
        assert saved.dtype==np.float32 and saved.shape==(173,2,18)
        assert np.isfinite(saved).all() and not saved[~valid[model]].any()
        binding=inp['prediction_'+model];assert prediction_lock['predictions'][model]==binding
        verify_bound(binding);predictions=read(ROOT/binding['path']);predictions=predictions.get('predictions',predictions)
        assert set(predictions)==set(ids)
        maximum=0.;points=0
        for i,row in enumerate(rows):
            prediction=predictions[row['id']]
            if not valid[model][i].any():continue
            chosen=prediction['selected_index'];assert chosen is not None
            candidate=prediction['candidates'][chosen];q=np.asarray(candidate['keypoints_xy'],np.float64)
            assert q.shape==(9,2) and np.isfinite(q).all() and not (q==-1).all(1).any()
            hypotheses={h['name']:h for h in poses[model][row['id']]['hypotheses']}
            for j,hyp in enumerate(HYP):
                if not valid[model][i,j]:continue
                h=hypotheses[hyp]
                expected,norm,diagonal=S.scalar_direction(h['pose'],q,candidate['box_xyxy'],row['K'])
                difference=np.abs(saved[i,j].astype(np.float64)-expected.astype(np.float64))
                assert (difference<=1e-4/diagonal+1e-6*np.abs(expected)).all(),(model,row['id'],hyp)
                for now,before,scale in ((norm,features[model][i,j,px],1.),(norm/diagonal,features[model][i,j,normalized],diagonal),
                                        (norm[:8],h['inference_cues']['corner8_residual_px'],1.)):
                    before=np.asarray(before,np.float64)
                    assert (np.abs(now-before)<=1e-4/scale+1e-6*np.abs(before)).all()
                maximum=max(maximum,float(difference.max()));points+=9
        stored=receipt['models'][model]
        assert stored['prediction_bindings']==[binding]*173
        assert stored['direction_sha']==array_sha(saved) and stored['valid_sha']==array_sha(valid[model])
        assert stored['raw94_sha']==array_sha(features[model]) and stored['valid_candidates']==int(valid[model].sum())
        audit[model]=dict(projected_points=points,independent_max_direction_difference=maximum,direction_sha=array_sha(saved))
    return directions,dict(PASS=True,receipt=receipt_binding,arrays=lock['directions'],models=audit,
        raw94_residual_norm_parity=True,no_extra_padding=True,no_PnP=True,no_reference_access=True)


def verify_routes():
    # No performance/reference helper is imported here or in the scalar scorer.
    protocol_binding = read(DOC/'REAL_PROTOCOL_SHA.json')
    verify_bound(protocol_binding)
    protocol = read(DOC/'REAL_PROTOCOL.json')
    assert protocol['schema'] == 'pallet_pose_signed_axes_asymmetric_learned_real_v1'
    assert protocol['complete'] and protocol['frames'] == 173
    assert protocol['models'] == list(LEARNED) and protocol['baselines'] == list(BASELINES)
    assert protocol['runtime_uses_margin'] is False and not protocol['runtime_safe_mask']
    assert protocol['runtime_uses_fixed_rbf'] is True
    assert protocol['loss_rule'] == 'FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_ASYMMETRIC_HUBER_PLUS_SIGN_LOGISTIC'
    assert protocol['feature_map'] == 'normalized94_abs_anchor_delta94_identity1_fixed_rbf64_signed_direction18'
    assert protocol['feature_dim'] == 271 and protocol['raw_feature_dim'] == 94
    assert protocol['reference_values_before_routing'] is False
    assert protocol['image_forwards'] == protocol['new_PnP_solves'] == 0
    verify_all(protocol['inputs']); verify_all(protocol['codes'])
    verify_bound(protocol['parent_protocol'])
    train_protocol = read(ROOT/protocol['parent_protocol']['path'])
    assert protocol['basis_SHA_bind'] == train_protocol['inputs']['rbf_basis']
    verify_bound(protocol['basis_SHA_bind'])
    basis_artifact = read(ROOT/protocol['basis_SHA_bind']['path'])
    assert basis_artifact['complete'] and basis_artifact['PASS']
    assert basis_artifact['schema'] == 'pallet_pose_anchor_rbf_basis_v1'
    verify_bound(train_protocol['inputs']['prefit_review'])
    prefit = read(ROOT/train_protocol['inputs']['prefit_review']['path'])
    assert prefit['complete'] and prefit['PASS']
    assert protocol['real_evaluation'] == train_protocol['real_evaluation']
    assert protocol['original_goal_criteria'] == train_protocol['original_goal_criteria']
    assert train_protocol['evidence']['original_goal_protocol'] == protocol['inputs']['stable_protocol']
    stable_protocol = read(ROOT/protocol['inputs']['stable_protocol']['path'])
    assert protocol['original_goal_criteria'] == stable_protocol['stability_criteria']
    assert protocol['real_stability_contract'] == 'matched_intervention_AND_original_SINGLE251_stability'
    verify_bound(protocol['source_val_gate'])
    source = read(ROOT/protocol['source_val_gate']['path'])
    assert source['complete'] and source['PASS'] and source['real_routing_authorized']
    assert source['checks_total'] == source['checks_passed'] == 45 and not source['failed_checks']
    assert not source['real_reference_accessed']
    independently_verified_source = read(DOC/'SOURCE_VAL_VERIFICATION.json')
    assert independently_verified_source['complete'] and independently_verified_source['PASS']
    assert independently_verified_source['source_gate']==protocol['source_val_gate']
    assert independently_verified_source['checks_total']==independently_verified_source['checks_passed']==45
    assert independently_verified_source['source_gate_PASS'] is True
    complete = read(DOC/'TRAINING_COMPLETE.json')
    assert complete['complete'] and complete['all_certified'] and complete['fit_count'] == 4
    assert complete['models'] == list(LEARNED) and complete['protocol'] == protocol['parent_protocol']
    assert complete['basis_SHA_bind'] == protocol['basis_SHA_bind']
    assert T.metadata()==S.METHOD_FIELDS
    verify_all(train_protocol['codes'])
    assert bind(T.__file__) in train_protocol['codes']
    for key,value in S.METHOD_FIELDS.items():
        assert train_protocol[key]==prefit[key]==complete[key]==value
    for key in ('loss_rule','input_rule','prediction_rule','sign_rule','sign_coefficient','solver_rule','underprediction_coefficient','underprediction_cost','overprediction_cost','underprediction_zero_curvature','base_feature_dim','direction_dim','direction_rule','direction_normalization'):
        assert protocol[key]==S.METHOD_FIELDS[key]
    train_verified=read(DOC/'TRAIN_CONVERGENCE.json')
    assert train_verified['complete'] and train_verified['PASS'] and train_verified['protocol']==protocol['parent_protocol']
    for key in S.HASH_KEYS:assert set(complete[key+'_by_model'])==set(LEARNED)
    assert complete['source_TRAIN_only'] and not complete['VAL_quality_read'] and not complete['real_targets_read']
    assert source['training_complete'] == bind(DOC/'TRAINING_COMPLETE.json')
    lock = read(DOC/'REAL_ROUTING_LOCK.json')
    assert lock['complete'] and lock['frames'] == 173 and lock['models'] == list(LEARNED)
    assert lock['protocol'] == bind(DOC/'REAL_PROTOCOL.json')
    assert lock['training_complete'] == bind(DOC/'TRAINING_COMPLETE.json')
    assert lock['source_val_gate'] == protocol['source_val_gate']
    assert lock['inputs'] == protocol['inputs'] and lock['operators'] == protocol['codes']
    assert not lock['real_reference_values_read'] and not lock['runtime_safe_mask']
    assert lock['whole_pose_selection'] and lock['image_forwards'] == lock['new_PnP_solves'] == lock['fits_executed'] == 0
    assert not lock['runtime_uses_margin'] and not lock['real_margin_values_loaded'] and not lock['real_margin_values_computed']
    assert lock['runtime_uses_fixed_rbf'] is True and lock['basis_SHA_bind'] == protocol['basis_SHA_bind']
    verify_all(lock)
    inp = protocol['inputs']
    assert len(inp) == 15
    assert {'prediction_'+m for m in ('R0','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3')} <= set(inp)
    prediction = read(ROOT/inp['prediction_lock']['path'])
    pose_lock = read(ROOT/inp['pose_lock']['path'])
    attribution = read(ROOT/inp['attribution']['path'])
    assert prediction['complete'] and not prediction['inference_reference_coordinates_read']
    assert prediction['metadata'] == inp['metadata']
    assert pose_lock['prediction_lock'] == inp['prediction_lock'] and not pose_lock['references_read']
    assert inp['poses'] in pose_lock['files'] and inp['groups'] in pose_lock['files']
    assert inp['prediction_lock'] in attribution['inputs'] and inp['pose_lock'] in attribution['inputs']
    assert inp['features'] in attribution['artifacts'] and attribution['real_reference_reads'] == 0
    # Original image/weight bytes may be hashed; no image/model decoder is used.
    verify_all(prediction); verify_all(pose_lock)
    verify_all(attribution['inputs']); verify_all(attribution['codes'])
    rows = read(ROOT/inp['metadata']['path']); ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)) == 173
    groups = read(ROOT/inp['groups']['path'])
    assert {p:len(groups[p]) for p in ('NATURAL99','CLEAN29','WOOD45')} == {'NATURAL99':99,'CLEAN29':29,'WOOD45':45}
    assert set(groups['NATURAL99']).isdisjoint(groups['CLEAN29'])
    assert set(groups['NATURAL99']) | set(groups['CLEAN29']) == set(groups['FULL128'])
    assert len(groups['FULL128']) == 128 and set(groups['FULL128']).isdisjoint(groups['WOOD45'])
    assert set(groups['FULL128']) | set(groups['WOOD45']) == set(ids)
    assert all(len(v) == len(set(v)) and set(v) <= set(ids) for v in groups.values())
    byid = {r['id']:r for r in rows}
    assert len({byid[i]['recording'] for i in groups['NATURAL99']}) == 6
    poses = read(ROOT/inp['poses']['path'])
    assert set(poses) == set(BASELINES)
    assert all(set(poses[m]) == set(ids) for m in BASELINES)
    assert all(not poses[m][fid]['reference_coordinates_read'] for m in BASELINES for fid in ids)
    arrays, masks = {}, {}
    parents = ('R0','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3')
    with np.load(ROOT/inp['features']['path'], allow_pickle=False) as data:
        assert data['ids'].tolist() == ids and data['feature_names'].shape == (94,)
        feature_names=data['feature_names'].tolist()
        for parent in parents:
            raw, frame = data[parent+'_geo'], data[parent+'_valid']
            assert raw.dtype == np.float32 and raw.shape == (173,2,94)
            assert frame.dtype == bool and frame.shape == (173,)
            mask = np.zeros((173,2),bool)
            for j,fid in enumerate(ids):
                hypotheses = {h['name']:h['pose'] for h in poses[parent][fid]['hypotheses']}
                for k,name in enumerate(HYP):
                    mask[j,k] = bool(frame[j] and name in hypotheses and hypotheses[name]['available'])
            assert np.isfinite(raw[mask]).all()
            arrays[parent], masks[parent] = raw,mask
    anchor = np.full(173,-1,np.int64)
    for j,fid in enumerate(ids):
        if any(mask[j].any() for mask in masks.values()):
            record = poses['R0'][fid]
            name = record['GEO_name']; assert name in HYP
            k = HYP.index(name); assert masks['R0'][j,k]
            assert next(h['pose'] for h in record['hypotheses'] if h['name']==name) == record['GEO_pose']
            anchor[j] = k
    assert array_sha(anchor) == lock['anchor_index_sha']
    choices = read(ROOT/lock['choices']['path'])
    assert choices['ids'] == ids and choices['models'] == list(LEARNED)
    assert not choices['real_reference_values_read'] and choices['whole_pose_selection']
    assert not choices['runtime_uses_margin'] and choices['loss_rule'] == protocol['loss_rule']
    assert choices['runtime_uses_fixed_rbf'] is True and choices['basis_SHA_bind'] == protocol['basis_SHA_bind']
    for key in ('loss_rule','input_rule','prediction_rule','sign_rule','sign_coefficient','solver_rule','output_dim','huber_delta','underprediction_coefficient','underprediction_cost','overprediction_cost','underprediction_zero_curvature','base_feature_dim','direction_dim','direction_rule','direction_normalization','runtime_uses_reference_errors'):
        assert lock[key]==choices[key]==protocol[key]==S.METHOD_FIELDS[key]
    directions,direction_checks=verify_direction_inputs(protocol,train_protocol,lock,rows,poses,arrays,masks,anchor,feature_names)
    for key in ('direction_inputs','directions','direction_receipt_binding','direction_normalization_sha'):
        assert choices[key]==lock[key]
    maximum_gap = 0.; maximum_axes_gap=0.; route_count = 0; model_checks = {}
    receipts = {}
    for b in complete['fits']:
        verify_bound(b); receipt = read(ROOT/b['path']); receipts[receipt['model']] = (b,receipt)
    assert set(receipts) == set(LEARNED)
    for model in LEARNED:
        receipt_binding,receipt = receipts[model]
        ck=verify_fit(model,receipt_binding,receipt,protocol,train_protocol,lock,complete,prefit,basis_artifact)
        names=names_for(model)
        raw = np.concatenate([arrays[p] for p in model_parts(model)],axis=1)
        valid = np.concatenate([masks[p] for p in model_parts(model)],axis=1)
        direction18=np.concatenate([directions[p] for p in model_parts(model)],axis=1)
        score = independent_scores(ck,raw,valid,anchor,direction18)
        axes = independent_axes(ck,raw,valid,anchor,direction18)
        np.testing.assert_allclose(score,T.score_candidates(ck,raw,valid,anchor,direction18),rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(axes,T.predict_axes(ck,raw,valid,anchor,direction18),rtol=1e-12,atol=1e-10)
        assert not axes[~valid].any()
        active=np.flatnonzero(valid.any(1));assert not axes[active,anchor[active]].any()
        assert set(choices['records'][model]) == set(ids)
        selected=[]
        for j,fid in enumerate(ids):
            choice=choices['records'][model][fid]
            available=np.flatnonzero(valid[j]).tolist()
            winner=min(available,key=lambda k:(float(score[j,k]),tie_key(names[k]))) if available else -1
            assert choice['candidate_index'] == winner,(model,fid,'ROUTE_MISMATCH')
            assert choice['candidate_name'] == (names[winner] if winner>=0 else None)
            assert choice['fallback'] == (winner<0)
            assert choice['valid'] == valid[j].tolist()
            assert choice['anchor_index'] == int(anchor[j])
            assert choice['anchor_name'] == (names[int(anchor[j])] if anchor[j]>=0 else None)
            saved_axes=np.array([v if flag else [0.,0.] for v,flag in zip(choice['predicted_signed_axes'],valid[j])],np.float64)
            assert all(v is None for v,flag in zip(choice['predicted_signed_axes'],valid[j]) if not flag)
            np.testing.assert_allclose(axes[j],saved_axes,rtol=1e-12,atol=1e-10)
            maximum_axes_gap=max(maximum_axes_gap,float(np.max(np.abs(axes[j]-saved_axes))))
            saved=np.array([np.inf if v is None else v for v in choice['scores']])
            np.testing.assert_array_equal(saved,np.where(valid[j],saved_axes.max(1),np.inf))
            np.testing.assert_allclose(score[j],saved,rtol=1e-12,atol=1e-10)
            if available:
                maximum_gap=max(maximum_gap,float(np.max(np.abs(score[j,available]-saved[available]))))
            p=selected_pose(choice,poses,fid)
            assert choice['pose'] == p and choice['pose_available'] == p['available']
            selected.append(winner); route_count+=1
        model_checks[model]=dict(checkpoint=receipt['checkpoint'],routes=173,selected_index_sha=array_sha(np.array(selected,np.int64)),
            context_sha=array_sha(independent_context(ck,raw,valid,anchor)),
            difference271_sha=array_sha(S.independent_extended_difference(ck,raw,valid,anchor,direction18)),runtime_uses_margin=False,
            target_or_safety_mask_used=False)
    assert route_count == 692
    return protocol,rows,groups,poses,choices,dict(routes=route_count,models=model_checks,
        maximum_independent_score_difference=maximum_gap,anchor_index_sha=array_sha(anchor),
        whole_pose_exact=True,reference_values_read=False,runtime_margin_loaded_or_computed=False,
        feature_dim=271,base_feature_dim=253,direction_dim=18,output_dim=2,
        direction_inputs=direction_checks,maximum_independent_axes_difference=maximum_axes_gap,
        context_dim=189,rbf_dim=64,runtime_uses_fixed_rbf=True,
        frozen_RBF_basis=protocol['basis_SHA_bind'],independent_basis_selection_review=train_protocol['inputs']['prefit_review'],
        basis_reselected_or_updated=False,
        source45_verified=True)


def yaw_rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def reference_map(protocol,rows):
    """Direct locked geometry reconstruction, without importing Pose.metadata."""
    assert STATE['references_allowed']
    refs=read(ROOT/protocol['inputs']['reference_bindings']['path'])
    assert refs['verified'] and refs['annotation_count'] == refs['image_count'] == 173
    verify_all(refs)
    assert read(DOC/'REAL_REFERENCE_BINDINGS.json') == refs
    expected={Path(b['path']).name:b for b in refs['geometry']}
    assert set(expected)=={'AXIS_REVIEW_MANIFEST.json','GEOMETRY_RESOLVED_POSE_GT.json'}
    axes=read(ROOT/expected['AXIS_REVIEW_MANIFEST.json']['path'])['frames_list']
    geometry=read(ROOT/expected['GEOMETRY_RESOLVED_POSE_GT.json']['path'])['frames']
    pose_lock=read(ROOT/protocol['inputs']['pose_lock']['path'])
    geometry_contract={Path(b['path']).name:b for b in pose_lock['geometry_inputs']}
    registry=read(ROOT/geometry_contract['CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json']['path'])['objects']
    symmetry=read(ROOT/geometry_contract['OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json']['path'])['objects']
    registry={r['object_type']:r['physical_dimensions_m'] for r in registry}
    symmetry={r['object_type']:r['group_order'] for r in symmetry}
    rowmap={(ROOT/r['image']['path']).resolve():r for r in rows}
    annotation_paths={(ROOT/b['path']).resolve() for b in refs['annotations']}
    truth={}; swapped=0
    for item in axes:
        image=(ROOT/item['image']).resolve()
        if image not in rowmap:
            continue
        row=rowmap[image]; fid=row['id']; assert fid not in truth
        annotation=(ROOT/item['annotation']).resolve(); assert annotation in annotation_paths
        intrinsics=read(annotation)['camera_data']['intrinsics']
        K=np.array([[intrinsics['fx'],0,intrinsics['cx']],[0,intrinsics['fy'],intrinsics['cy']],[0,0,1]])
        np.testing.assert_allclose(K,row['K'],rtol=0,atol=1e-9)
        dims=registry[item['object_type']]
        xyz=np.array([dims['x'],dims['y'],dims['z']])
        np.testing.assert_allclose(xyz,row['xyz'],rtol=0,atol=1e-9)
        assert symmetry[item['object_type']] == 2
        g=geometry[item['frame_id']]; d=g['physical_dimensions_m']
        cf=np.array([d['across'],d['height'],d['along']])
        swap=abs(cf[0]-xyz[0])>=1e-6; swapped+=int(swap)
        q=yaw_rotation(np.pi/2) if swap else np.eye(3)
        truth[fid]={'R':np.asarray(g['R_gt_representative'])@q,'t':np.asarray(g['t_gt'])}
        assert np.isfinite(truth[fid]['R']).all() and np.isfinite(truth[fid]['t']).all()
        np.testing.assert_allclose(truth[fid]['R'].T@truth[fid]['R'],np.eye(3),rtol=0,atol=1e-6)
        assert abs(np.linalg.det(truth[fid]['R'])-1)<1e-6
    assert set(truth)=={r['id'] for r in rows}
    return truth,dict(reference_bindings=bind(ROOT/protocol['inputs']['reference_bindings']['path']),
        geometry=refs['geometry'],frames=173,C2_order=2,registry_axes_swapped=swapped,
        reference='R_gt_representative @ fixed registry-axis Q; t_gt; no source Rx(pi) transform.',
        scope='Geometry-derived reference on reused DEV; not independent physical GT.')


def physical_error(pose,truth):
    if not pose['available']:
        return np.array([np.inf,np.inf]),0.
    R=np.asarray(pose['R_physical'],np.float64); G=truth['R']
    assert R.shape==G.shape==(3,3)
    delta=np.asarray(pose['centroid'])-truth['t']
    translation=float(np.linalg.norm(delta)*100)
    by_trace=[]; by_frobenius=[]
    for symmetry in (np.eye(3),yaw_rotation(np.pi)):
        reference=G@symmetry
        # Independently evaluate the SO(3) geodesic via matrix trace.
        relative=reference.T@R
        cosine=float(np.clip((sum(relative[j,j] for j in range(3))-1)/2,-1,1))
        by_trace.append(float(np.degrees(np.arccos(cosine))))
        by_frobenius.append(float(np.degrees(2*np.arcsin(np.clip(np.linalg.norm(R-reference,'fro')/(2*np.sqrt(2)),0,1)))))
    angle=min(by_trace); alternate=min(by_frobenius)
    assert np.isfinite([translation,angle,alternate]).all()
    assert abs(angle-alternate)<1e-4,(angle,alternate)
    return np.array([translation,angle]),abs(angle-alternate)


def scalar(value):
    value=float(value)
    return dict(value=value if np.isfinite(value) else None,
        status='FINITE' if np.isfinite(value) else 'UNDEFINED' if np.isnan(value) else 'POSITIVE_INFINITY' if value>0 else 'NEGATIVE_INFINITY')


def quantile(values,q):
    ordered=np.sort(np.asarray(values,float))
    if not len(ordered): return np.nan
    position=(len(ordered)-1)*q; low=int(math.floor(position)); high=int(math.ceil(position))
    if low==high: return float(ordered[low])
    if np.isposinf(ordered[high]): return np.inf
    return float(ordered[low]+(position-low)*(ordered[high]-ordered[low]))


def quantiles(tensor,q=.5,conditional=True):
    assert tensor.shape[0]==3 and tensor.shape[-1]==2
    assert not np.isnan(tensor).any() and (tensor>=0).all()
    assert np.array_equal(np.isfinite(tensor[:,:,0]),np.isfinite(tensor[:,:,1]))
    result=[]
    for seed in tensor:
        selected=seed[np.isfinite(seed).all(1)] if conditional else seed
        result.append([quantile(selected[:,j],q) for j in range(2)])
    return np.asarray(result)


def averages(tensor,q=.5,conditional=True):
    return quantiles(tensor,q,conditional).mean(0)


def mean_summary(tensor):
    result={}
    for conditional,mode in ((True,'conditional'),(False,'full_population')):
        result[mode]={label:{key:scalar(v) for key,v in zip(METRICS,averages(tensor,q,conditional))}
                      for label,q in (('median',.5),('P90',.9))}
    failures=np.isinf(tensor[:,:,0]).sum(1)
    result.update(failure_counts_by_seed=failures.tolist(),mean_failure_count=float(failures.mean()))
    return result


def bootstrap(before,after,recordings):
    groups=sorted(set(recordings)); indices=[np.flatnonzero(np.asarray(recordings)==g) for g in groups]
    generator=np.random.default_rng(20261001); samples=[]
    for _ in range(2000):
        seeds=generator.integers(0,3,size=3)
        selected_groups=generator.integers(0,len(groups),size=len(groups))
        frames=np.concatenate([indices[g] for g in selected_groups])
        samples.append(averages(after[seeds][:,frames])-averages(before[seeds][:,frames]))
    samples=np.asarray(samples); point=averages(after)-averages(before)
    result=dict(repeats=2000,seed=20261001,recording_count=len(groups),training_seed_count=3,
        paired_recording_and_training_seed=True,same_draws_all_comparisons=True,metrics={})
    for j,key in enumerate(METRICS):
        finite=np.isfinite(samples[:,j]); values=samples[finite,j]
        interval=[quantile(values,.025),quantile(values,.975)] if len(values) else [None,None]
        result['metrics'][key]=dict(point_estimate=scalar(point[j]),CI95=interval,
            finite_draws=int(finite.sum()),undefined_draws=int((~finite).sum()),
            status='FINITE_ALL_DRAWS' if finite.all() else 'UNDEFINED_DRAWS_PRESENT',
            upper95_below_zero=bool(finite.all() and interval[1]<0))
    return result


def compare(before,after,recordings,with_bootstrap):
    differences=quantiles(after)-quantiles(before)
    result=dict(before=mean_summary(before),after=mean_summary(after),
        per_seed_median_difference=[{k:scalar(v) for k,v in zip(METRICS,d)} for d in differences],
        mean_seed_median_difference={k:scalar(v) for k,v in zip(METRICS,differences.mean(0))},
        all_seeds_both_medians_smaller=bool(np.isfinite(differences).all() and (differences<0).all()),LORO={})
    for recording in sorted(set(recordings)):
        keep=np.asarray(recordings)!=recording
        difference=averages(after[:,keep])-averages(before[:,keep])
        result['LORO'][recording]=dict(frames=int(keep.sum()),
            mean_seed_median_difference={k:scalar(v) for k,v in zip(METRICS,difference)},
            both_negative=bool(np.isfinite(difference).all() and (difference<0).all()))
    if with_bootstrap: result['hierarchical_bootstrap']=bootstrap(before,after,recordings)
    return result


def guard(before,after,levels):
    result={}
    for label,q in levels:
        b,a=averages(before,q),averages(after,q)
        for j,key in enumerate(METRICS):
            result[f'{label}:{key}']=dict(before=scalar(b[j]),after=scalar(a[j]),ratio_limit=1.05,
                limit=scalar(b[j]*1.05),pass_guard=bool(np.isfinite([a[j],b[j]]).all() and a[j]<=b[j]*1.05))
    b=np.isinf(before[:,:,0]).sum(1); a=np.isinf(after[:,:,0]).sum(1)
    result['pose_failures']=dict(before_by_seed=b.tolist(),after_by_seed=a.tolist(),pass_guard=bool((a<=b).all()))
    return dict(PASS=all(v['pass_guard'] for v in result.values()),checks=result)


def family(metrics,rows,populations,original=False):
    sequences={m:[m]*3 for m in ('R0','PRIOR1','FULL125')}
    sequences['UNION']=[f'UNION_s{s}' for s in (1,2,3)]
    if original:
        sequences['SINGLE251']=[f'SINGLE251_s{s}' for s in (1,2,3)]
        comparators=('SINGLE251','R0','PRIOR1','FULL125'); paired=('SINGLE251','R0')
    else:
        sequences.update(R0_ONLY=['R0_ONLY']*3,ORIGINAL_DIVERSE=[f'DIVERSE251_s{s}' for s in (1,2,3)])
        comparators=('R0_ONLY','ORIGINAL_DIVERSE','R0','PRIOR1','FULL125'); paired=('R0_ONLY','R0')
    def tensor(pop,seq):
        return np.array([[metrics[m][fid] for fid in populations[pop]] for m in seq])
    arrays={p:{m:tensor(p,seq) for m,seq in sequences.items()} for p in ('NATURAL99','CLEAN29','WOOD45')}
    recordings={r['id']:r['recording'] for r in rows}; hierarchy={}
    def key(name): return f'DIVERSE251-minus-{name}' if original else name
    for pop,values in arrays.items():
        rec=[recordings[fid] for fid in populations[pop]]
        hierarchy[pop]={key(m):compare(values[m],values['UNION'],rec,original or m in paired) for m in comparators}
    primary=hierarchy['NATURAL99']
    three={m:primary[key(m)]['all_seeds_both_medians_smaller'] for m in comparators}
    uncertainty={m:all(v['upper95_below_zero'] for v in primary[key(m)]['hierarchical_bootstrap']['metrics'].values()) for m in paired}
    sensitivity={m:all(v['both_negative'] for v in primary[key(m)]['LORO'].values()) for m in paired}
    tails={m:guard(arrays['NATURAL99'][m],arrays['NATURAL99']['UNION'],[('P90',.9)]) for m in paired}
    gates=dict(all_three_seeds_joint_gain=dict(PASS=all(three.values()),comparisons=three),
        joint_uncertainty=dict(PASS=all(uncertainty.values()),comparisons=uncertainty),
        recording_sensitivity=dict(PASS=all(sensitivity.values()),comparisons=sensitivity),
        natural_tails=dict(PASS=all(v['PASS'] for v in tails.values()),comparisons=tails),
        clean_preservation=guard(arrays['CLEAN29']['R0'],arrays['CLEAN29']['UNION'],[('median',.5),('P90',.9)]))
    return dict(PASS=all(v['PASS'] for v in gates.values()),gates=gates),hierarchy


def compare_subset(expected,actual,path='root',counts=None):
    """Compare independent numerical/status fields, ignoring explanatory prose."""
    if counts is None: counts=Counter()
    if isinstance(expected,dict):
        assert isinstance(actual,dict),path
        for k,v in expected.items():
            assert k in actual,(path,k)
            compare_subset(v,actual[k],path+'.'+k,counts)
    elif isinstance(expected,list):
        assert isinstance(actual,list) and len(expected)==len(actual),path
        for i,(x,y) in enumerate(zip(expected,actual)):
            compare_subset(x,y,f'{path}[{i}]',counts)
    elif isinstance(expected,(float,np.floating)):
        assert actual is not None and math.isclose(expected,actual,rel_tol=2e-12,abs_tol=1e-9),(path,expected,actual)
        counts['numeric']+=1
    else:
        assert expected==actual,(path,expected,actual)
        counts['status_or_discrete']+=1
    return counts


def verify_metrics(protocol,rows,groups,poses,choices):
    truth,reference_checks=reference_map(protocol,rows)
    result=read(DOC/'REAL_RESULTS.json')
    assert result['complete'] and result['models']==list(MODELS)
    assert result['protocol']==bind(DOC/'REAL_PROTOCOL.json') and result['routing_lock']==bind(DOC/'REAL_ROUTING_LOCK.json')
    assert result['full_frame_rows']==2249 and result['baseline_metric_parity_checks']==1557
    assert result['image_forwards']==result['new_PnP_solves']==result['new_fits']==0
    assert result['populations']=={p:len(ids) for p,ids in groups.items()}
    verify_all(result['artifacts'])
    saved=read(RAW/'POSE_METRICS.json'); assert set(saved)==set(MODELS)
    ids=[r['id'] for r in rows]; independent={}; largest=np.zeros(2); frobenius_gap=0.
    for model in MODELS:
        assert set(saved[model])==set(ids); independent[model]={}
        for fid in ids:
            pose=choices['records'][model][fid]['pose'] if model in LEARNED else poses[model][fid]['GEO_pose']
            errors,gap=physical_error(pose,truth[fid]); independent[model][fid]=errors
            actual=saved[model][fid]; assert actual['id']==fid and actual['available']==pose['available']
            if pose['available']:
                stored=np.array([actual[k] for k in METRICS])
                np.testing.assert_allclose(errors,stored,rtol=1e-12,atol=1e-9,err_msg=f'{model}/{fid}')
                largest=np.maximum(largest,np.abs(errors-stored)); frobenius_gap=max(frobenius_gap,gap)
            else:
                assert all(k not in actual for k in METRICS)
    manifest=read(ROOT/protocol['inputs']['stable_publication']['path'])
    binding=next(b for b in manifest['files'] if b['path']==str((STABLE_DOC/'RESULTS.json').relative_to(ROOT)))
    verify_bound(binding); previous=read(ROOT/binding['path'])
    metrics_binding=next(b for b in previous['artifacts'] if b['path']==str((STABLE_RAW/'POSE_METRICS.json').relative_to(ROOT)))
    verify_bound(metrics_binding); old=read(ROOT/metrics_binding['path'])
    exact_baseline=0
    for model in BASELINES:
        for fid in ids:
            assert saved[model][fid]==old[model][fid],(model,fid,'HISTORICAL_BASELINE_DIFFERENT')
            exact_baseline+=1
    table=list(csv.DictReader((DOC/'REAL_FRAME_RESULTS.csv').open(newline='')))
    assert len(table)==2249 and len({(r['model'],r['id']) for r in table})==2249
    for row in table:
        model,fid=row['model'],row['id']; metric=saved[model][fid]
        assert row['pose_available']==str(metric['available'])
        for key in METRICS:
            assert row[key]==str(metric[key]) if metric['available'] else row[key]==''
    summary_checks=0
    for pop,subset in groups.items():
        for model in MODELS:
            summary=result['summaries'][pop][model]
            values=np.array([independent[model][fid] for fid in subset])
            finite=np.isfinite(values).all(1)
            for mode,selection in (('conditional',values[finite]),('full_population',values)):
                for j,key in enumerate(METRICS):
                    for label,q in (('median',.5),('P90',.9)):
                        expected=quantile(selection[:,j],q)
                        stored=summary[mode][key][label]
                        if np.isfinite(expected):
                            assert math.isclose(expected,stored,rel_tol=2e-12,abs_tol=1e-9),(pop,model,mode,key,label)
                        else:
                            assert stored is None
                        summary_checks+=1
    return independent,result,dict(frames=173,models=13,pose_records=2249,physical_T_R_values=4498,
        maximum_error_difference=largest.tolist(),rotation_trace_vs_Frobenius_max_deg=frobenius_gap,
        historical_baseline_metric_dicts_exact=exact_baseline,public_CSV_unique_rows=2249,
        summary_quantile_checks=summary_checks,references=reference_checks,
        original_metric_binding=metrics_binding,
        failures={m:int(sum(not saved[m][fid]['available'] for fid in ids)) for m in MODELS})


def selfcheck():
    S.selfcheck()
    pose=dict(available=True,R_physical=np.eye(3).tolist(),centroid=[0,0,1])
    errors,gap=physical_error(pose,dict(R=yaw_rotation(np.pi),t=np.array([0,0,1])))
    np.testing.assert_allclose(errors,[0,0],atol=1e-6)
    assert physical_error(dict(available=False),{})[0].tolist()==[np.inf,np.inf]
    assert quantile([0,10],.9)==9 and np.isposinf(quantile([0,np.inf],.9))
    before=np.full((3,6,2),10.);after=before-1
    recordings=['a','a','b','b','c','c']
    c=compare(before,after,recordings,True)
    assert c['all_seeds_both_medians_smaller']
    assert all(x['upper95_below_zero'] for x in c['hierarchical_bootstrap']['metrics'].values())
    assert all(x['both_negative'] for x in c['LORO'].values())
    assert guard(before,after,[('median',.5),('P90',.9)])['PASS']
    damaged=after.copy();damaged[0,0,:]=np.inf
    assert not guard(before,damaged,[('P90',.9)])['PASS']
    # Shared-control draws still use three paired refiner seed indices.
    assert c['hierarchical_bootstrap']['training_seed_count']==3
    print('INDEPENDENT_REAL_SELFCHECK_PASS artifact_reads=0',flush=True)


def run():
    assert (DOC/'REAL_RESULTS.json').is_file(),'Wait for root score-complete authorization.'
    install_guard()
    protocol,rows,groups,poses,choices,routing=verify_routes()
    before_reads=sorted(STATE['reads'])
    STATE['references_allowed']=True
    metrics,result,physical=verify_metrics(protocol,rows,groups,poses,choices)
    matched,matched_hierarchy=family(metrics,rows,groups,False)
    original,original_hierarchy=family(metrics,rows,groups,True)
    counts=Counter()
    for expected,actual,label in ((matched,result['matched_intervention_stability'],'matched'),
        (original,result['original_goal_stability'],'original'),
        (matched_hierarchy,result['hierarchy'],'matched_hierarchy'),
        (original_hierarchy,result['original_goal_hierarchy'],'original_hierarchy')):
        compare_subset(expected,actual,label,counts)
    detail=read(DOC/'REAL_DETAILED_COMPARISONS.json')
    assert detail['hierarchical']==result['hierarchy'] and detail['original_goal_hierarchy']==result['original_goal_hierarchy']
    combined={k:bool(matched['gates'][k]['PASS'] and original['gates'][k]['PASS']) for k in matched['gates']}
    assert set(combined)==set(result['stability']['gates']) and len(combined)==5
    for key,passed in combined.items():
        gate=result['stability']['gates'][key]
        assert gate['PASS']==passed
        compare_subset(matched['gates'][key],gate['matched_intervention'],key+'.matched',counts)
        compare_subset(original['gates'][key],gate['original_goal'],key+'.original',counts)
    assert result['stability']['PASS']==all(combined.values())==bool(matched['PASS'] and original['PASS'])
    assert result['stability']['goal_complete'] is False and result['stability']['strong_generalization'] is False
    output=dict(complete=True,PASS=True,scope='Independent numerical/provenance verification, not an assertion of stable improvement.',
        code=bind(Path(__file__)),independent_scalar_helper=bind(S.__file__),protocol=bind(DOC/'REAL_PROTOCOL.json'),
        result=bind(DOC/'REAL_RESULTS.json'),routing_lock=bind(DOC/'REAL_ROUTING_LOCK.json'),
        independent_source_verification=bind(DOC/'SOURCE_VAL_VERIFICATION.json'),
        routing=routing,physical_metrics=physical,
        stability=dict(matched=matched,original_goal=original,combined_categories=combined,
            combined_PASS=all(combined.values()),numeric_and_status_checks=dict(counts),
            bootstrap_repeats=2000,bootstrap_seed=20261001,
            bootstrap_comparison_populations=18,quantile_rule='Conditional per-seed population quantile then mean across3 paired refiner seeds; full-population infinity retained separately.',
            original_SINGLE251_comparator_restored=True,gate_family_combination='category-wise AND'),
        read_boundary=dict(real_reference_values_before_independent_routes=False,
            pre_reference_read_paths=before_reads,all_read_paths=sorted(STATE['reads']),
            reference_containers_may_contain_other_rows_but_only_fixed173_scored=True),
        new_fits=0,image_forwards=0,new_PnP_solves=0,threshold_probes=0,
        limitations=['Reused DEV with6 natural recording clusters; no new recording confirmation.',
                    'Real references are reconstructed from annotations and dimensions; not independently measured physical truth.',
                    'PASS here certifies faithful computation; see combined_PASS for method stability.'])
    for path,value in ((DOC/'REAL_VERIFICATION.json',output),(DOC/'REAL_VERIFICATION_KO.md',
        '# 실사 선택·T/R·두 판정 묶음 독립 검산\n\n'
        '**독립 계산 검산: PASS.** 이 표시는 방법의 개선 성공을 뜻하지 않습니다. '
        f'실제 combined stability는 **{"PASS" if all(combined.values()) else "FAIL"}**입니다.\n\n'
        '- 고정 4개 checkpoint와 입력 특징만으로 692개 선택을 독립 재현했습니다. 189열의 정규화·anchor 차이·identity에 고정 TRAIN center 거리의 Gaussian64를 더한253열과 동결한 방향18열을 직접 계산하고 각각 anchor 차분하여271열에서 두 축 내적 및 max 점수를 재현했습니다. basis 바인딩·normalization SHA·쌍 거리 median bandwidth를 검증했고 실사에서 center·폭을 재선정하거나 margin/안전 GT mask를 사용하지 않았습니다.\n'
        '- 선택 재현과 잠금 SHA 검증 후에만 원본 참조를 읽었습니다. 고정 13모델 × 173행의 T/R 4,498값과 CSV 2,249행을 검산했고, 기존 9모델의 1,557 metric 사전은 이전 결과와 완전히 같습니다.\n'
        '- geometry의 R_cf에 registry 축 Q를 적용한 C2 참조를 직접 구성했습니다. translation norm, rotation matrix trace 및 별도 Frobenius 계산을 대조했습니다. 새 PnP나 이미지 inference는 없습니다.\n'
        '- 새 R0_ONLY/paired DIVERSE 비교와 원래 SINGLE251 비교의 각 5개 판정을 별도로 재계산했습니다. bootstrap 2,000회·seed 20261001·동일 recording/seed 추출, LORO, 5% 꼬리, clean median/P90 및 seed별 실패 수를 그대로 검산했습니다. 최종 5항목은 두 묶음의 항목별 AND입니다.\n\n'
        '| 항목 | matched | 원래 SINGLE251 | 결합 |\n|---|---|---|---|\n'+''.join(
            f'| {k} | {matched["gates"][k]["PASS"]} | {original["gates"][k]["PASS"]} | {combined[k]} |\n' for k in combined)+
        '\n실사는 반복 사용한 DEV이며 geometry-derived reference입니다. 독립 물리 GT나 새 recording 일반화 검증으로 해석할 수 없습니다. 검산에서 새 fit·forward·PnP·문턱 탐색은 모두 0입니다.\n\n'
        '[기계 검산 영수증](REAL_VERIFICATION.json) · [실제 결과](REAL_RESULTS.json) · [검산 코드](../../../scripts/research/pallet_pose_signed_axes_asymmetric_20261001_v1/verify_real.py)\n')):
        with path.open('x') as handle:
            handle.write(value if isinstance(value,str) else json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('REAL_INDEPENDENT_VERIFICATION_PASS',json.dumps(dict(routing=692,physical_values=4498,
        baseline_records=1557,combined=combined)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('selfcheck','verify'))
    args=parser.parse_args()
    if args.action=='selfcheck':selfcheck()
    else:run()
