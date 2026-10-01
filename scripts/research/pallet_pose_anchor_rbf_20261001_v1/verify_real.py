"""Independent fixed-RBF real routes, physical C2 errors, and both stability families.

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

def independent_base_context(checkpoint, features, valid, anchor_index):
    """Independent scalar candidate loop for the fixed 189-column map."""
    mean, std = np.asarray(checkpoint['mean'], np.float32), np.asarray(checkpoint['std'], np.float32)
    assert features.dtype == np.float32 and valid.dtype == bool
    out = np.zeros((*valid.shape, 189), np.float64)
    assert features.shape == (*valid.shape, 94) and anchor_index.shape == (len(valid),)
    for i in range(len(valid)):
        candidates = np.flatnonzero(valid[i]).tolist()
        anchor = int(anchor_index[i])
        if not candidates:
            assert anchor == -1
            continue
        assert anchor in (0, 1) and anchor in candidates
        reference = ((features[i, anchor] - mean) / std).astype(np.float64)
        for j in candidates:
            z = ((features[i, j] - mean) / std).astype(np.float64)
            out[i, j, :94] = z
            out[i, j, 94:188] = np.array([abs(float(z[q]) - float(reference[q])) for q in range(94)])
            out[i, j, 188] = float(j == anchor)
    assert np.isfinite(out).all() and not out[~valid].any()
    return out

def independent_basis(checkpoint):
    """Verify the embedded immutable basis without using its runtime helper."""
    basis = checkpoint['rbf_basis']
    assert basis['schema'] == 'pallet_pose_anchor_rbf_runtime_v1'
    assert basis['context_dim'] == 189 and basis['rbf_dim'] == 64
    centers = np.asarray(basis['centers'], np.float64)
    assert centers.shape == (64, 189) and np.isfinite(centers).all()
    assert len({row.tobytes() for row in centers}) == 64
    distances = np.array([np.sum((centers[i]-centers[j])**2)
                          for i in range(64) for j in range(i+1,64)])
    assert np.isfinite(distances).all() and (distances >= 0).all()
    width = float(basis['bandwidth_squared'])
    assert np.isfinite(width) and width > 0 and width == float(np.median(distances[distances>0]))
    mean, std = np.asarray(checkpoint['mean'], np.float32), np.asarray(checkpoint['std'], np.float32)
    assert mean.shape == std.shape == (94,) and (std >= np.float32(1e-6)).all()
    assert basis['normalization_sha'] == array_sha(np.stack([mean,std]))
    assert checkpoint['rbf_basis_binding'] == checkpoint['basis_SHA_bind']
    return centers, width


def independent_context(checkpoint, features, valid, anchor_index):
    """Independent candidate loop: existing189 plus64 fixed Gaussian values."""
    centers, width = independent_basis(checkpoint)
    base = independent_base_context(checkpoint, features, valid, anchor_index)
    result = np.zeros((*valid.shape,253),np.float64)
    result[:,:,:189] = base
    for i,j in zip(*np.where(valid)):
        distance = np.sum((base[i,j][None,:]-centers)**2,axis=1)
        assert np.isfinite(distance).all() and (distance>=0).all()
        result[i,j,189:] = np.exp(-distance/(2*width))
    assert np.isfinite(result).all() and not result[~valid].any()
    return result


def independent_scores(checkpoint, features, valid, anchor_index):
    context = independent_context(checkpoint, features, valid, anchor_index)
    values = np.sum(context * np.asarray(checkpoint['weight'], np.float64), axis=2)
    return np.where(valid, values, np.inf)

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


def verify_routes():
    # No performance/reference helper is imported here or in the scalar scorer.
    protocol_binding = read(DOC/'REAL_PROTOCOL_SHA.json')
    verify_bound(protocol_binding)
    protocol = read(DOC/'REAL_PROTOCOL.json')
    assert protocol['schema'] == 'pallet_pose_anchor_rbf_learned_real_v1'
    assert protocol['complete'] and protocol['frames'] == 173
    assert protocol['models'] == list(LEARNED) and protocol['baselines'] == list(BASELINES)
    assert protocol['runtime_uses_margin'] is False and not protocol['runtime_safe_mask']
    assert protocol['runtime_uses_fixed_rbf'] is True
    assert protocol['loss_rule'] == 'TRAIN_LOG1P_ANCHOR_EXCESS_MARGIN_CE'
    assert protocol['feature_map'] == 'normalized94_abs_anchor_delta94_identity1_fixed_rbf64'
    assert protocol['feature_dim'] == 253 and protocol['raw_feature_dim'] == 94
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
    complete = read(DOC/'TRAINING_COMPLETE.json')
    assert complete['complete'] and complete['all_certified'] and complete['fit_count'] == 4
    assert complete['models'] == list(LEARNED) and complete['protocol'] == protocol['parent_protocol']
    assert complete['basis_SHA_bind'] == protocol['basis_SHA_bind']
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
    assert len(inp) == 11
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
    maximum_gap = 0.; route_count = 0; model_checks = {}
    receipts = {}
    for b in complete['fits']:
        verify_bound(b); receipt = read(ROOT/b['path']); receipts[receipt['model']] = (b,receipt)
    assert set(receipts) == set(LEARNED)
    for model in LEARNED:
        receipt_binding,receipt = receipts[model]
        assert receipt['complete'] and receipt['protocol'] == protocol['parent_protocol']
        assert receipt['basis_SHA_bind'] == protocol['basis_SHA_bind']
        verify_bound(receipt['START']); start = read(ROOT/receipt['START']['path'])
        assert start['basis_SHA_bind'] == protocol['basis_SHA_bind']
        verify_bound(receipt['trace'])
        for trace_row in [json.loads(line) for line in (ROOT/receipt['trace']['path']).read_text().splitlines()]:
            assert trace_row['basis_SHA_bind'] == protocol['basis_SHA_bind']
        assert lock['fits'][model]['receipt'] == receipt_binding
        assert lock['fits'][model]['checkpoint'] == receipt['checkpoint']
        verify_bound(receipt['checkpoint']); ck = read(ROOT/receipt['checkpoint']['path'])
        assert ck['schema'] == 'pallet_pose_anchor_rbf_linear253_v1'
        assert ck['rbf_basis'] == basis_artifact['basis']
        assert ck['rbf_basis_binding'] == ck['basis_SHA_bind'] == protocol['basis_SHA_bind']
        assert ck['model'] == model and ck['protocol'] == protocol['parent_protocol']
        assert ck['loss_rule'] == protocol['loss_rule'] and not ck['runtime_uses_margin']
        assert ck['certificate']['PASS'] and ck['certificate']['gradient_l2_squared_over_2lambda'] <= 1e-6
        assert ck['lambda_l2'] == .0001 and ck['bias'] == 0.
        names = names_for(model); assert ck['names'] == names
        mean,std=np.asarray(ck['mean'],np.float32),np.asarray(ck['std'],np.float32)
        assert mean.shape == std.shape == (94,) and (std >= np.float32(1e-6)).all()
        assert array_sha(np.stack([mean,std])) == ck['normalization_sha']
        assert np.asarray(ck['weight']).shape == (253,)
        raw = np.concatenate([arrays[p] for p in model_parts(model)],axis=1)
        valid = np.concatenate([masks[p] for p in model_parts(model)],axis=1)
        score = independent_scores(ck,raw,valid,anchor)
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
            saved=np.array([np.inf if v is None else v for v in choice['scores']])
            np.testing.assert_allclose(score[j],saved,rtol=1e-12,atol=1e-10)
            if available:
                maximum_gap=max(maximum_gap,float(np.max(np.abs(score[j,available]-saved[available]))))
            p=selected_pose(choice,poses,fid)
            assert choice['pose'] == p and choice['pose_available'] == p['available']
            selected.append(winner); route_count+=1
        model_checks[model]=dict(checkpoint=receipt['checkpoint'],routes=173,selected_index_sha=array_sha(np.array(selected,np.int64)),
            context_sha=array_sha(independent_context(ck,raw,valid,anchor)),runtime_uses_margin=False)
    assert route_count == 692
    return protocol,rows,groups,poses,choices,dict(routes=route_count,models=model_checks,
        maximum_independent_score_difference=maximum_gap,anchor_index_sha=array_sha(anchor),
        whole_pose_exact=True,reference_values_read=False,runtime_margin_loaded_or_computed=False,
        feature_dim=253,context_dim=189,rbf_dim=64,runtime_uses_fixed_rbf=True,
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
    ck=dict(mean=[0.]*94,std=[1.]*94,weight=[.1]*253)
    centers=np.zeros((64,189),np.float64)
    centers[:,0]=np.arange(64)/10
    width=float(np.median([(centers[i,0]-centers[j,0])**2 for i in range(64) for j in range(i+1,64)]))
    ck['rbf_basis']=dict(schema='pallet_pose_anchor_rbf_runtime_v1',context_dim=189,rbf_dim=64,
        centers=centers.tolist(),bandwidth_squared=width,
        normalization_sha=array_sha(np.stack([np.zeros(94,np.float32),np.ones(94,np.float32)])))
    ck['rbf_basis_binding']=ck['basis_SHA_bind']={'invented':True}
    features=np.arange(3*4*94,dtype=np.float32).reshape(3,4,94)/100
    valid=np.array([[True,True,True,True],[True,False,True,False],[False]*4])
    anchor=np.array([1,0,-1],np.int64)
    mapped=independent_context(ck,features,valid,anchor)
    assert mapped.shape==(3,4,253) and not mapped[~valid].any()
    assert mapped[0,1,188]==mapped[1,0,188]==1 and not mapped[0,1,94:188].any()
    scalar=np.zeros((3,4,64))
    for i,j in zip(*np.where(valid)):
        for c in range(64):
            distance=sum(float(mapped[i,j,k]-centers[c,k])**2 for k in range(189))
            scalar[i,j,c]=np.exp(-distance/(2*width))
    np.testing.assert_allclose(mapped[:,:,189:],scalar,rtol=1e-12,atol=1e-12)
    scores=independent_scores(ck,features,valid,anchor)
    np.testing.assert_allclose(scores[valid],np.sum(mapped[valid]*.1,axis=1),rtol=0,atol=1e-12)
    assert np.isposinf(scores[~valid]).all()
    try:
        independent_context(dict(ck,rbf_basis=dict(ck['rbf_basis'],normalization_sha='0'*64)),features,valid,anchor)
    except AssertionError:
        pass
    else:
        raise AssertionError('BASIS_NORMALIZATION_DRIFT_MUST_STOP')
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
        code=bind(Path(__file__)),protocol=bind(DOC/'REAL_PROTOCOL.json'),
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
        '- 고정 4개 checkpoint와 입력 특징만으로 692개 선택을 독립 재현했습니다. 189열의 정규화·anchor 차이·identity에 고정 TRAIN center 거리의 Gaussian64를 더한253열을 직접 계산했습니다. basis 바인딩·normalization SHA·쌍 거리 median bandwidth를 검증했고 실사에서 center·폭을 재선정하거나 margin/안전 GT mask를 사용하지 않았습니다.\n'
        '- 선택 재현과 잠금 SHA 검증 후에만 원본 참조를 읽었습니다. 고정 13모델 × 173행의 T/R 4,498값과 CSV 2,249행을 검산했고, 기존 9모델의 1,557 metric 사전은 이전 결과와 완전히 같습니다.\n'
        '- geometry의 R_cf에 registry 축 Q를 적용한 C2 참조를 직접 구성했습니다. translation norm, rotation matrix trace 및 별도 Frobenius 계산을 대조했습니다. 새 PnP나 이미지 inference는 없습니다.\n'
        '- 새 R0_ONLY/paired DIVERSE 비교와 원래 SINGLE251 비교의 각 5개 판정을 별도로 재계산했습니다. bootstrap 2,000회·seed 20261001·동일 recording/seed 추출, LORO, 5% 꼬리, clean median/P90 및 seed별 실패 수를 그대로 검산했습니다. 최종 5항목은 두 묶음의 항목별 AND입니다.\n\n'
        '| 항목 | matched | 원래 SINGLE251 | 결합 |\n|---|---|---|---|\n'+''.join(
            f'| {k} | {matched["gates"][k]["PASS"]} | {original["gates"][k]["PASS"]} | {combined[k]} |\n' for k in combined)+
        '\n실사는 반복 사용한 DEV이며 geometry-derived reference입니다. 독립 물리 GT나 새 recording 일반화 검증으로 해석할 수 없습니다. 검산에서 새 fit·forward·PnP·문턱 탐색은 모두 0입니다.\n\n'
        '[기계 검산 영수증](REAL_VERIFICATION.json) · [실제 결과](REAL_RESULTS.json) · [검산 코드](../../../scripts/research/pallet_pose_anchor_rbf_20261001_v1/verify_real.py)\n')):
        with path.open('x') as handle:
            handle.write(value if isinstance(value,str) else json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('REAL_INDEPENDENT_VERIFICATION_PASS',json.dumps(dict(routing=692,physical_values=4498,
        baseline_records=1557,combined=combined)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('selfcheck','verify'))
    args=parser.parse_args()
    if args.action=='selfcheck':selfcheck()
    else:run()
