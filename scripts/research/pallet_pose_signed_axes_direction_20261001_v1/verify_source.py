"""Independent signed-direction271 model VAL route, physical metric, and 45-gate audit.

Run verify only after root reports the four-model freeze and score complete.
No evaluation helper is imported. Before all 4096 saved routes are verified,
the audit hook denies source reference and VAL quality containers. Afterwards
only the locked 1024 VAL reference indices are scored; real GT stays denied.
"""
import argparse
from collections import Counter
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
PARENT_DOC = ROOT / '_docs/experiments/pallet_pose_union_selection_20261001_v1'
PARENT_RAW = ROOT / 'data/pallet/results/pallet_pose_union_selection_20261001_v1'
SOURCE = ROOT / 'data/pallet/results/pallet_selector_recovery_v1/stage2_synth_scorer'
HYP = ('long-face-front', 'short-face-front')
LEARNED = ('R0_ONLY', 'UNION_s1', 'UNION_s2', 'UNION_s3')
BASELINES = ('R0_GEO', 'DIVERSE251_s1_GEO', 'DIVERSE251_s2_GEO', 'DIVERSE251_s3_GEO')
METRICS = ('translation_cm', 'rotation_deg')
STATE = {'references_allowed': False, 'reads': set()}
HASH_KEYS = ('signed_target_sha', 'input_difference_sha', 'base_context_sha',
             'errors_sha', 'scaled_excess_sha', 'original_valid_sha',
             'direction_raw_sha','direction_difference_sha','extended_input_sha')
METHOD_FIELDS = dict(target_rule='SIGNED_LOG1P_NORMALIZED_TR_ANCHOR_EXCESS',
    input_rule='RBF253_DIFFERENCE_PLUS_NORMALIZED_DIRECTION18_DIFFERENCE',
    loss_rule='FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_HUBER_PLUS_SIGN_LOGISTIC',
    sign_rule='NONZERO_SIGNED_TARGET_AXES',sign_coefficient=1.,
    prediction_rule='MAX_TWO_SIGNED_LOG1P_AXES',
    feature_map='normalized94_abs_anchor_delta94_identity1_fixed_rbf64_signed_direction18',
    feature_dim=271, raw_feature_dim=94, context_dim=189, rbf_dim=64, output_dim=2, huber_delta=1.,
    runtime_uses_margin=False, runtime_uses_reference_errors=False, runtime_safe_mask=False,
    solver_rule='BLOCK_GENERALIZED_NEWTON_ARMIJO',base_feature_dim=253,direction_dim=18,
    direction_rule='BBOX_DIAGONAL_NORMALIZED_PROJECTED_MINUS_OBSERVED_XY9',
    direction_normalization='R0_TRAIN_valid_float32_mean_std_floor_1e-6_then_float64_candidate_minus_anchor')



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


def install_guard():
    allowed_writes = {DOC / 'SOURCE_VAL_VERIFICATION.json', DOC / 'SOURCE_VAL_VERIFICATION_KO.md'}
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(ROOT):
            return
        text = str(path)
        mode = args[1]
        flags = args[2] if len(args)>2 and isinstance(args[2],int) else 0
        if (isinstance(mode, str) and any(letter in mode for letter in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            assert path in allowed_writes, ('VERIFIER_WRITE_SCOPE', text)
            return
        assert not any(token in text for token in ('/data/evaluation/', '/real_gt_v2/',
            '/annotations/', 'GEOMETRY_RESOLVED_POSE_GT', 'AXIS_REVIEW_MANIFEST', 'TRUTH_FOR_DISPLAY'))
        assert path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.pt', '.pth', '.onnx'), text
        if not STATE['references_allowed']:
            assert not any(token in text for token in ('GEOMETRY_SIDETABLE.npz', '/SYNTH_RECORDS.json',
                '/SOURCE_MANIFEST.json', '/DIMENSION_SIDECAR.json', '/SYNTH_LABELS.npz',
                'SOURCE_VAL_METRICS.npz', 'SOURCE_VAL_GATE.json', 'VAL_ORACLE.json')), ('BEFORE_ROUTE_VERIFICATION', text)
        STATE['reads'].add(str(path.relative_to(ROOT)))
    sys.addaudithook(hook)


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


def independent_difference(checkpoint, features, valid, anchor_index):
    context = independent_context(checkpoint, features, valid, anchor_index)
    difference = np.zeros_like(context)
    for i in range(len(valid)):
        available = np.flatnonzero(valid[i]).tolist()
        if not available:
            assert anchor_index[i] == -1
            continue
        anchor = int(anchor_index[i])
        assert anchor in (0,1) and anchor in available
        for j in available:
            difference[i,j] = context[i,j]-context[i,anchor]
        assert not difference[i,anchor].any()
    assert np.isfinite(difference).all() and not difference[~valid].any()
    return difference


def independent_extended_difference(checkpoint,features,valid,anchor_index,direction18):
    old=independent_difference(checkpoint,features,valid,anchor_index)
    assert direction18.dtype==np.float32 and direction18.shape==(*valid.shape,18)
    mean=np.asarray(checkpoint['direction_mean'],np.float32)
    std=np.asarray(checkpoint['direction_std'],np.float32)
    assert mean.shape==std.shape==(18,) and (std>=np.float32(1e-6)).all()
    assert array_sha(np.stack([mean,std]))==checkpoint['direction_normalization_sha']
    extra=np.zeros((*valid.shape,18),np.float64)
    for i in range(len(valid)):
        active=np.flatnonzero(valid[i]).tolist()
        if not active:
            assert anchor_index[i]==-1;continue
        a=int(anchor_index[i]);reference=((direction18[i,a]-mean)/std).astype(np.float64)
        for j in active:
            extra[i,j]=((direction18[i,j]-mean)/std).astype(np.float64)-reference
        assert not extra[i,a].any()
    return np.concatenate([old,extra],axis=2)


def independent_axes(checkpoint, features, valid, anchor_index,direction18):
    difference = independent_extended_difference(checkpoint, features, valid, anchor_index,direction18)
    weight = np.asarray(checkpoint['weight'], np.float64)
    assert weight.shape == (271,2) and np.isfinite(weight).all()
    axes = np.zeros((*valid.shape,2),np.float64)
    for i,j in zip(*np.where(valid)):
        for axis in range(2):
            axes[i,j,axis] = np.dot(difference[i,j],weight[:,axis])
    assert np.isfinite(axes).all() and not axes[~valid].any()
    return axes


def independent_scores(checkpoint, features, valid, anchor_index,direction18):
    axes = independent_axes(checkpoint, features, valid, anchor_index,direction18)
    return np.where(valid,np.maximum(axes[:,:,0],axes[:,:,1]),np.inf)


def scalar_direction(pose,points,box,K):
    """Independent coordinate-by-coordinate projection, without a runtime helper."""
    ext=np.asarray(pose['cf_extents'],np.float64)
    R=np.asarray(pose['R_cf'],np.float64);t=np.asarray(pose['centroid'],np.float64)
    q=np.asarray(points,np.float64);box=np.asarray(box,np.float64);K=np.asarray(K,np.float64)
    assert pose['available'] and ext.shape==(3,) and R.shape==(3,3) and t.shape==(3,)
    assert q.shape==(9,2) and box.shape==(4,) and K.shape==(3,3)
    assert all(np.isfinite(a).all() for a in (ext,R,t,q,box,K)) and (ext>0).all()
    np.testing.assert_array_equal(K[2],[0.,0.,1.])
    signs=((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
           (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1),(0,0,0))
    uv=np.empty((9,2),np.float64)
    for k,sign in enumerate(signs):
        xyz=[float(sign[a])*float(ext[a])/2 for a in range(3)]
        camera=[sum(float(R[a,b])*xyz[b] for b in range(3))+float(t[a]) for a in range(3)]
        h=[sum(float(K[a,b])*camera[b] for b in range(3)) for a in range(3)]
        assert h[2]!=0.
        uv[k]=[h[0]/h[2],h[1]/h[2]]
    diagonal=math.sqrt(sum(max(float(box[a+2]-box[a]),1e-6)**2 for a in (0,1)))
    delta=uv-q;norm=np.sqrt(np.sum(delta*delta,axis=1))
    return (delta/diagonal).reshape(18).astype(np.float32),norm,diagonal


def verify_direction_inputs(protocol,lock,rows,positions,poses,features,valid,anchor_index):
    """Authenticate all VAL direction inputs before allowing any quality read."""
    binding=lock['direction_inputs'];verify(binding)
    assert binding['path']==str((DOC/'SOURCE_VAL_DIRECTION_INPUTS.json').relative_to(ROOT))
    receipt=read(ROOT/binding['path'])
    assert receipt['complete'] and receipt['PASS']
    assert receipt['scope']=='SOURCE_VAL' and receipt['frames']==1024 and receipt['dimension']==18
    assert receipt['protocol']==lock['protocol']
    assert receipt['rule']==METHOD_FIELDS['direction_rule']
    assert receipt['parity']==dict(atol_px=1e-4,rtol=1e-6)
    assert not receipt['source_label_values_read'] and not receipt['real_reference_values_read']
    assert not receipt['normalization_recomputed'] and receipt['additional_padding']==0
    assert receipt['new_PnP_solves']==receipt['image_forwards']==receipt['fits_executed']==0
    assert all(item in protocol['codes'] for item in receipt['operators'])
    assert receipt['training_verification']==bind(DOC/'TRAIN_CONVERGENCE.json')
    expected_inputs=dict(features=protocol['inputs']['features'],feature_lock=lock['feature_lock'],
        poses=lock['poses'],metadata=lock['metadata'],predictions_lock=protocol['inputs']['source_predictions_lock'],
        training_complete=lock['training_complete'],training_verification=bind(DOC/'TRAIN_CONVERGENCE.json'),
        direction_receipt=protocol['inputs']['direction_receipt'])
    assert receipt['inputs']==expected_inputs
    for item in bindings(receipt):verify(item)
    assert lock['directions']==bind(RAW/'SOURCE_VAL_DIRECTIONS.npz')
    assert receipt['directions']==lock['directions']
    verify(lock['directions'])
    assert lock['direction_receipt_binding']==protocol['inputs']['direction_receipt']
    old=read(ROOT/protocol['inputs']['direction_receipt']['path'])
    assert lock['direction_normalization_sha']==protocol['direction_normalization_sha']==old['normalization']['array_sha']
    assert receipt['direction_receipt_binding']==lock['direction_receipt_binding']
    assert receipt['direction_normalization_sha']==lock['direction_normalization_sha']
    models=('R0','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3')
    with np.load(ROOT/lock['directions']['path'],allow_pickle=False) as z:
        assert set(z.files)=={'ids','source_index','anchor_index'}|{m+s for m in models for s in ('_direction18','_valid')}
        np.testing.assert_array_equal(z['ids'],[r['id'] for r in rows])
        np.testing.assert_array_equal(z['source_index'],positions)
        np.testing.assert_array_equal(z['anchor_index'],anchor_index)
        directions={m:z[m+'_direction18'] for m in models}
        for m in models:np.testing.assert_array_equal(z[m+'_valid'],valid[m])
    feature_lock=read(ROOT/lock['feature_lock']['path'])
    names=feature_lock['feature_names']
    px=[names.index(f'residual{k}_px') for k in range(9)]
    normalized=[names.index(f'residual{k}_bboxnorm') for k in range(9)]
    prediction_lock=read(ROOT/protocol['inputs']['source_predictions_lock']['path'])
    source_protocol=read(ROOT/prediction_lock['protocol']['path']);verify(prediction_lock['protocol'])
    audit={}
    for m in models:
        cached=directions[m]
        assert cached.dtype==np.float32 and cached.shape==(1024,2,18)
        assert np.isfinite(cached).all() and not cached[~valid[m]].any()
        verify(prediction_lock['receipts'][m])
        prediction_receipt=read(ROOT/prediction_lock['receipts'][m]['path'])
        assert prediction_receipt['model']==m and prediction_receipt['complete']
        assert prediction_receipt['protocol']==prediction_lock['protocol']
        maximum=0.;points=0;selected_bindings=[]
        for n,(row,source_index) in enumerate(zip(rows,positions)):
            item=prediction_receipt['files'][int(source_index)]
            assert item['path']==str((PARENT_RAW/'source_predictions'/m/f'{int(source_index):05d}.json').relative_to(ROOT))
            verify(item);selected_bindings.append(item)
            saved=read(ROOT/item['path'])
            assert saved['id']==row['id'] and saved['model']==m
            assert saved['protocol_sha']==prediction_lock['protocol']['sha256']
            assert saved['checkpoint_sha']==source_protocol['checkpoints'][m]['sha256']
            prediction=saved['prediction'];hypotheses={h['name']:h for h in poses['records'][m][row['id']]['hypotheses']}
            if not valid[m][n].any():continue
            selected=prediction['selected_index'];assert selected is not None
            candidate=prediction['candidates'][selected]
            q=np.asarray(candidate['keypoints_xy'],np.float64)
            assert np.isfinite(q).all() and not (q==-1).all(1).any()
            for j,hyp in enumerate(HYP):
                if not valid[m][n,j]:continue
                record=hypotheses[hyp]
                expected,norm,diagonal=scalar_direction(record['pose'],q,candidate['box_xyxy'],row['K'])
                delta=np.abs(cached[n,j].astype(np.float64)-expected.astype(np.float64))
                tolerance=1e-4/diagonal+1e-6*np.abs(expected)
                assert (delta<=tolerance).all(),(m,row['id'],hyp,'direction_parity')
                for current,prior,scale in ((norm,features[m][n,j,px],1.),
                    (norm/diagonal,features[m][n,j,normalized],diagonal),
                    (norm[:8],record['inference_cues']['corner8_residual_px'],1.)):
                    prior=np.asarray(prior,np.float64)
                    assert (np.abs(current-prior)<=1e-4/scale+1e-6*np.abs(prior)).all()
                maximum=max(maximum,float(delta.max()));points+=9
        audit[m]=dict(valid_candidates=int(valid[m].sum()),projected_points=points,
            independent_direction_max_absolute=maximum,direction_sha=array_sha(cached),
            prediction_receipt=prediction_lock['receipts'][m],selected_prediction_count=len(selected_bindings))
        stored=receipt['models'][m]
        assert stored['prediction_bindings']==selected_bindings
        assert stored['direction_sha']==array_sha(cached) and stored['valid_sha']==array_sha(valid[m])
        assert stored['raw94_sha']==array_sha(features[m]) and stored['valid_candidates']==int(valid[m].sum())
    return directions,dict(PASS=True,receipt=binding,arrays=lock['directions'],models=audit,
        source_VAL_input_only=True,quality_read=False,new_PnP=0,new_image_forwards=0,
        normalization='Frozen R0 TRAIN5194 only; no VAL normalization')


def independent_trace(trace, start, receipt, checkpoint):
    """Authenticate Armijo/accepted-point bookkeeping using logs only.

    No training feature, target, Hessian, or objective is computed here.
    Consecutive trials share a parent and use alpha 1,1/2,... until accepted.
    """
    objectives=[];iterations=[];accepted=None;pending=None;expected_alpha=1.
    rejected=0
    for row in trace:
        assert row['event'] in ('objective','iteration')
        values=[row[key] for key in ('objective','Huber','Sign_logistic','L2_penalty','gradient_l2','gradient_linf','gradient_gap_upper_bound')]
        assert np.isfinite(values).all() and min(values)>=0
        assert abs(row['objective']-row['Huber']-row['Sign_logistic']-row['L2_penalty'])<=1e-12
        np.testing.assert_allclose(row['gradient_gap_upper_bound'],row['gradient_l2']**2/(2e-4),rtol=1e-12,atol=1e-18)
        if row['event']=='iteration':
            assert pending is not None
            assert row['iteration']==len(iterations)+1 and row['iteration']<=1000
            assert row['objective_calls']==len(objectives)==row['accepted_call']==pending['call']
            for key in ('weight_sha','objective','Huber','Sign_logistic','L2_penalty','gradient_l2','gradient_linf','gradient_gap_upper_bound',
                        'alpha','armijo_bound','armijo_accepted','parent_weight_sha','parent_objective','directional_derivative'):
                assert row[key]==pending[key]
            iterations.append(row);pending=None
            continue
        assert pending is None
        assert row['evaluated'] is True and row['call']==len(objectives)+1 and row['call']<=2000
        if not objectives:
            assert row['phase']=='initial' and row['iteration']==0 and row['alpha']==0.
            assert row['weight_sha']==start['initial_weight_sha'] and row['armijo_accepted'] is True
            for key in ('accepted_point_call_before','parent_weight_sha','parent_objective','directional_derivative','armijo_bound'):
                assert row[key] is None
            accepted=row
        else:
            assert row['phase']=='trial' and row['iteration']==len(iterations)+1
            assert row['accepted_point_call_before']==accepted['call']
            assert row['parent_weight_sha']==accepted['weight_sha'] and row['parent_objective']==accepted['objective']
            assert row['alpha']==expected_alpha
            slope=float(row['directional_derivative']);assert np.isfinite(slope) and slope<0
            bound=accepted['objective']+1e-4*row['alpha']*slope
            assert row['armijo_bound']==bound and bound<=accepted['objective']
            accepted_trial=row['objective']<=bound
            assert row['armijo_accepted']==accepted_trial
            if accepted_trial:
                accepted=row;pending=row;expected_alpha=1.
            else:
                rejected+=1;expected_alpha*=.5;assert expected_alpha>0
        objectives.append(row)
    assert pending is None and objectives and accepted is objectives[-1]
    assert len(objectives)==receipt['objective_calls']==checkpoint['certificate']['objective_calls']
    assert len(iterations)==receipt['iterations']==checkpoint['certificate']['iterations']
    for value in (receipt,checkpoint,checkpoint['certificate'],checkpoint['solver']):
        assert value['final_accepted_call']==accepted['call']
    assert checkpoint['solver']['last_evaluated_call']==len(objectives)
    assert accepted['weight_sha']==receipt['final_weight_sha']
    for field,key in (('objective','objective_value'),('Huber','Huber'),('Sign_logistic','Sign_logistic'),('L2_penalty','L2_penalty'),
                      ('gradient_l2','gradient_l2'),('gradient_linf','gradient_linf')):
        assert accepted[field]==checkpoint['certificate'][key]
    assert accepted['gradient_linf']<=1e-8 and accepted['gradient_gap_upper_bound']<=1e-6
    return dict(PASS=True,objective_calls=len(objectives),accepted_iterations=len(iterations),
        rejected_trials=rejected,initial_evaluation_calls=1,all_trials_counted=True,
        final_accepted_call=accepted['call'],last_evaluated_call=len(objectives),
        Armijo_and_last_accepted_state_exact=True,TRAIN_objective_recomputed=False)


def verify_routes():
    from . import common as C
    from . import convex_train as T
    assert T.OLD.READS is None and T.READS is None, 'Do not import a fitting or source-score process into this verifier.'
    protocol = C.protocol('TRAIN_PROTOCOL')
    assert protocol['models'] == list(LEARNED)
    for key,expected in METHOD_FIELDS.items():
        assert protocol[key] == expected
    assert T.metadata() == {key:value for key,value in METHOD_FIELDS.items()}
    assert protocol['lambda_l2'] == 1e-4 and protocol['bias'] == 0.
    assert protocol['solver'] == T.solver_config() == dict(method='block_generalized_newton_armijo',
        maxiter=1000,maxfun=2000,gradient_linf_tolerance=1e-8,initial_alpha=1.,
        backtrack_factor=.5,armijo_c1=1e-4,initialization='zeros',damping=0.,huber_kink_curvature=0.)
    assert protocol['certificate'] == dict(optimizer_success=True,gap_upper_bound_max=1e-6,gradient_linf_max=1e-8)
    assert protocol['normalization'] == 'old_float32_then_float64'
    assert protocol['feature_dim'] == 271 and protocol['raw_feature_dim'] == 94
    assert protocol['feature_map'] == 'normalized94_abs_anchor_delta94_identity1_fixed_rbf64_signed_direction18'
    assert protocol['loss_rule'] == METHOD_FIELDS['loss_rule']
    assert protocol['runtime_uses_margin'] is False
    basis_binding = protocol['inputs']['rbf_basis']
    assert basis_binding == bind(C.RBF_DOC/'RBF_BASIS.json')
    verify(basis_binding)
    basis_artifact = read(ROOT / basis_binding['path'])
    assert basis_artifact['complete'] and basis_artifact['PASS']
    assert basis_artifact['schema'] == 'pallet_pose_anchor_rbf_basis_v1'
    verify(protocol['inputs']['prefit_review'])
    prefit = read(ROOT / protocol['inputs']['prefit_review']['path'])
    assert prefit['complete'] and prefit['PASS']
    assert prefit['basis_SHA_bind'] == prefit['inputs']['basis'] == basis_binding
    assert prefit['source_TRAIN_only'] and prefit['source_TRAIN_cached_label_values_read']
    assert not prefit['VAL_quality_read'] and not prefit['real_targets_read']
    assert prefit['frames'] == 2598 and prefit['available_anchor_rows'] == 2597
    assert prefit['failed_rows_retained'] == 1
    for key in ('loss_rule','target_rule','input_rule','prediction_rule','output_dim'):
        assert prefit[key] == METHOD_FIELDS[key]
    assert prefit['trainer'] == bind(T.__file__)
    previous_prefit_binding = prefit['inputs']['previous_RBF_prefit']
    verify(previous_prefit_binding)
    previous_prefit = read(ROOT/previous_prefit_binding['path'])
    assert previous_prefit['complete'] and previous_prefit['PASS']
    assert previous_prefit['basis_SHA_bind'] == basis_binding
    complete = read(DOC / 'TRAINING_COMPLETE.json')
    assert complete['complete'] and complete['all_certified'] and complete['fit_count'] == 4
    assert complete['models'] == list(LEARNED) and complete['protocol'] == bind(DOC / 'TRAIN_PROTOCOL.json')
    assert complete['source_TRAIN_only'] and not complete['VAL_quality_read'] and not complete['real_targets_read']
    assert complete['loss_rule'] == protocol['loss_rule'] and complete['runtime_uses_margin'] is False
    assert complete['basis_SHA_bind'] == basis_binding
    assert set(complete['final_accepted_call_by_model']) == set(LEARNED)
    for key in METHOD_FIELDS:
        assert complete[key] == protocol[key]
    for key in HASH_KEYS:
        assert set(complete[key+'_by_model']) == set(LEARNED)
    lock = read(DOC / 'SOURCE_VAL_ROUTING_LOCK.json')
    assert lock['complete'] and lock['frames'] == 1024 and lock['models'] == list(LEARNED)
    assert lock['baselines'] == list(BASELINES)
    assert lock['protocol'] == bind(DOC / 'TRAIN_PROTOCOL.json')
    assert lock['training_complete'] == bind(DOC / 'TRAINING_COMPLETE.json')
    assert not lock['source_labels_read'] and not lock['source_VAL_label_values_read']
    assert not lock['real_references_read'] and not lock['VAL_quality_scored']
    assert lock['basis_SHA_bind'] == basis_binding
    for key in METHOD_FIELDS:
        assert lock[key] == protocol[key]
    assert lock['source_TRAIN_labels_sha_verified'] is True
    assert all(binding in protocol['codes'] for binding in lock['operators'])
    for binding in bindings(lock):
        verify(binding)
    contract = read(ROOT / lock['source_contract']['path'])
    assert contract['complete'] and contract['status'] == 'PASS'
    assert protocol['inputs']['source_contract'] == lock['source_contract']
    input_rows = read(ROOT / lock['metadata']['path'])
    positions = np.flatnonzero([row['split'] == 'VAL' for row in input_rows])
    rows = [input_rows[j] for j in positions]
    ids = [row['id'] for row in rows]
    assert len(ids) == len(set(ids)) == 1024
    assert set(ids) == set(contract['fit_eligibility']['eligible_ids']['VAL'])
    assert set(ids).isdisjoint(contract['fit_eligibility']['eligible_ids']['TRAIN'])
    feature_lock = read(ROOT / lock['feature_lock']['path'])
    verify(feature_lock['features'])
    assert feature_lock['features'] == protocol['inputs']['features']
    assert feature_lock['poses'] == lock['poses'] and feature_lock['metadata'] == lock['metadata']
    poses = read(ROOT / lock['poses']['path'])
    assert poses['ids'] == [row['id'] for row in input_rows]
    assert not poses['source_targets_read'] and not poses['real_targets_read']
    choices = read(ROOT / lock['choices']['path'])
    assert choices['ids'] == ids and choices['models'] == list(LEARNED)
    assert not choices['source_labels_read'] and not choices['real_references_read']
    assert choices['feature_map'] == protocol['feature_map']
    assert choices['basis_SHA_bind'] == basis_binding
    for key in ('direction_inputs','directions','direction_receipt_binding','direction_normalization_sha'):
        assert choices[key]==lock[key]
    for key in METHOD_FIELDS:
        assert choices[key] == protocol[key]
    with np.load(ROOT / feature_lock['features']['path'], allow_pickle=False) as stored:
        assert stored['ids'].tolist() == poses['ids']
        assert stored['split'].tolist() == [row['split'] for row in input_rows]
        assert stored['hypothesis_names'].tolist() == list(HYP)
        features = {m: stored[m + '_geo'][positions] for m in ('R0',) + tuple(f'DIVERSE251_s{s}' for s in (1, 2, 3))}
        valid = {m: stored[m + '_valid'][positions] for m in features}
        anchor_index = stored['R0_GEO_index'][positions]
    directions,direction_check=verify_direction_inputs(protocol,lock,rows,positions,poses,features,valid,anchor_index)
    assert anchor_index.dtype == np.int64 and anchor_index.shape == (1024,)
    assert np.isin(anchor_index, [-1, 0, 1]).all()
    assert array_sha(anchor_index) == choices['anchor_index_sha'] == lock['anchor_index_sha']
    for j, fid in enumerate(ids):
        record = poses['records']['R0'][fid]
        index = int(anchor_index[j])
        assert bool(record['GEO_pose']['available']) == (index >= 0)
        if index < 0:
            assert not valid['R0'][j].any()
        else:
            assert valid['R0'][j, index] and record['GEO_name'] == HYP[index]
            matching = [hyp for hyp in record['hypotheses'] if hyp['name'] == HYP[index]]
            assert len(matching) == 1 and matching[0]['pose'] == record['GEO_pose']
    anchor_checks = dict(rows=1024, input_only_R0_GEO_index_sha=array_sha(anchor_index),
        counts=dict(Counter(anchor_index.tolist())), GEO_name_and_full_pose_exact=True,
        learned_R0_ONLY_choice_not_used_as_anchor=True)
    anchor_checks['direction_inputs']=direction_check
    model_checks, fit_models, score_gap, axes_gap = {}, set(), 0., 0.
    total_calls,total_iterations=0,0
    for receipt_binding in complete['fits']:
        verify(receipt_binding)
        receipt = read(ROOT / receipt_binding['path'])
        model = receipt['model']
        assert model in LEARNED and model not in fit_models
        fit_models.add(model)
        assert receipt['complete'] and receipt['protocol'] == lock['protocol']
        assert receipt_binding == lock['fits'][model]['receipt']
        assert receipt['checkpoint'] == lock['fits'][model]['checkpoint']
        for key in ('checkpoint', 'START', 'trace'):
            verify(receipt[key])
        start = read(ROOT / receipt['START']['path'])
        certificate = receipt['certificate']
        assert certificate['PASS'] and certificate['optimizer_success']
        assert certificate['solver_rule'] == METHOD_FIELDS['solver_rule']
        assert certificate['gradient_linf_max'] == 1e-8 and 0<=certificate['gradient_linf']<=1e-8
        assert certificate['lambda_l2'] == 1e-4 and certificate['gradient_l2_squared_over_2lambda'] <= 1e-6
        assert 0 < certificate['objective_calls'] <= 2000 and 0 <= certificate['iterations'] <= 1000
        checkpoint = read(ROOT / receipt['checkpoint']['path'])
        assert checkpoint['schema'] == 'pallet_pose_signed_axes_direction_linear271x2_v1'
        assert checkpoint['solver']['success'] is True and checkpoint['solver']['status']=='CONVERGED'
        assert checkpoint['solver']['configuration'] == protocol['solver'] == start['solver']
        assert start['certificate_rule'] == protocol['certificate']
        assert start['initialization']=='all_zero_float64'
        assert start['initial_weight_sha']==array_sha(np.zeros((271,2),np.float64))
        assert checkpoint['final_accepted_call'] == complete['final_accepted_call_by_model'][model]
        assert checkpoint['rbf_basis'] == basis_artifact['basis']
        assert checkpoint['rbf_basis_binding'] == checkpoint['basis_SHA_bind'] == basis_binding
        assert receipt['basis_SHA_bind'] == start['basis_SHA_bind'] == basis_binding
        assert checkpoint['protocol'] == lock['protocol'] and checkpoint['model'] == model
        assert checkpoint['certificate'] == certificate and checkpoint['names'] == names_for(model)
        assert start['model'] == model and start['protocol'] == lock['protocol']
        assert checkpoint['normalization'] == 'old_float32_then_float64'
        assert checkpoint['loss_uses_original_valid_mask'] is True
        assert start['loss_uses_original_valid_mask'] is True
        assert checkpoint['bias'] == 0. and checkpoint['lambda_l2'] == 1e-4
        assert checkpoint['axis_order'] == start['axis_order'] == list(METRICS)
        assert checkpoint['signed_target_scales'] == start['signed_target_scales'] == prefit['inputs']['scale']
        assert receipt['source_TRAIN_only'] and not receipt['VAL_quality_read'] and not receipt['real_targets_read']
        for key in METHOD_FIELDS:
            assert checkpoint[key] == receipt[key] == start[key] == complete[key] == protocol[key]
        for key in HASH_KEYS:
            assert checkpoint[key] == receipt[key] == start[key] == complete[key+'_by_model'][model] == prefit['models'][model][key]
        for artifact in (checkpoint,receipt,start,complete,prefit):
            assert artifact['direction_receipt_binding']==protocol['inputs']['direction_receipt']
            assert artifact['direction_normalization_sha']==protocol['direction_normalization_sha']
        np.testing.assert_array_equal(np.asarray(checkpoint['direction_mean'],np.float32),np.asarray(protocol['direction_mean'],np.float32))
        np.testing.assert_array_equal(np.asarray(checkpoint['direction_std'],np.float32),np.asarray(protocol['direction_std'],np.float32))
        assert certificate['loss_rule'] == METHOD_FIELDS['loss_rule']
        assert certificate['sign_rule'] == METHOD_FIELDS['sign_rule']
        assert certificate['sign_coefficient'] == METHOD_FIELDS['sign_coefficient'] == 1.
        assert certificate['huber_delta'] == 1. and certificate['output_dim'] == 2
        np.testing.assert_allclose(certificate['gradient_l2_squared_over_2lambda'],
            certificate['gradient_l2']**2/(2e-4),rtol=1e-12,atol=1e-18)
        vals = [float(certificate[key]) for key in ('objective_value','Huber','Sign_logistic','L2_penalty')]
        assert np.isfinite(vals).all() and min(vals)>=0 and abs(vals[0]-vals[1]-vals[2]-vals[3])<=1e-12
        trace = [json.loads(line) for line in (ROOT/receipt['trace']['path']).read_text().splitlines()]
        trace_check=independent_trace(trace,start,receipt,checkpoint)
        calls=[row for row in trace if row['event']=='objective']
        assert [row['call'] for row in calls] == list(range(1,receipt['objective_calls']+1))
        assert receipt['objective_calls'] == certificate['objective_calls']
        assert receipt['iterations'] == certificate['iterations'] == receipt['optimizer_steps'] == checkpoint['optimizer_steps']
        assert receipt['fits_executed'] == checkpoint['fits_executed'] == 1
        assert calls[-1]['weight_sha'] == receipt['final_weight_sha']
        assert calls[-1]['objective'] == certificate['objective_value']
        for trace_row in trace:
            assert trace_row['basis_SHA_bind'] == basis_binding
            assert trace_row['anchor_index_sha'] == complete['anchor_index_sha']
            assert trace_row['direction_normalization_sha']==protocol['direction_normalization_sha']
            assert trace_row['direction_receipt_binding']==protocol['inputs']['direction_receipt']
            for key in METHOD_FIELDS:
                assert trace_row[key] == protocol[key]
            for key in HASH_KEYS:
                assert trace_row[key] == checkpoint[key]
        total_calls+=receipt['objective_calls'];total_iterations+=receipt['iterations']
        assert checkpoint['anchor_index_sha'] == receipt['anchor_index_sha'] == start['anchor_index_sha'] == complete['anchor_index_sha']
        mean, std = np.asarray(checkpoint['mean'], np.float32), np.asarray(checkpoint['std'], np.float32)
        assert (std >= np.float32(1e-6)).all()
        assert array_sha(np.stack([mean, std])) == receipt['normalization_sha'] == checkpoint['normalization_sha']
        assert np.asarray(checkpoint['weight'],np.float64).shape == (271,2)
        assert array_sha(np.asarray(checkpoint['weight'], np.float64)) == receipt['final_weight_sha']
        x = np.concatenate([features[m] for m in model_parts(model)], axis=1)
        mask = np.concatenate([valid[m] for m in model_parts(model)], axis=1)
        raw_direction=np.concatenate([directions[m] for m in model_parts(model)],axis=1)
        names = names_for(model)
        assert np.array_equal(mask.any(1), anchor_index >= 0)
        scores = T.score_candidates(checkpoint, x, mask, anchor_index,raw_direction)
        manual_context = independent_context(checkpoint, x, mask, anchor_index)
        np.testing.assert_array_equal(manual_context[:,:,:189], T.context_inputs(x, mask, anchor_index, mean, std))
        np.testing.assert_array_equal(manual_context, T.rbf_inputs(x, mask, anchor_index, mean, std, checkpoint['rbf_basis']))
        difference = independent_extended_difference(checkpoint,x,mask,anchor_index,raw_direction)
        np.testing.assert_array_equal(difference,T.difference_inputs(x,mask,anchor_index,mean,std,checkpoint['rbf_basis'],raw_direction,checkpoint['direction_mean'],checkpoint['direction_std']))
        predicted_axes = T.predict_axes(checkpoint,x,mask,anchor_index,raw_direction)
        manual_axes = independent_axes(checkpoint,x,mask,anchor_index,raw_direction)
        local_axes_gap = float(np.abs(predicted_axes[mask]-manual_axes[mask]).max()) if mask.any() else 0.
        assert local_axes_gap <= 1e-10
        axes_gap = max(axes_gap,local_axes_gap)
        assert not predicted_axes[~mask].any() and not manual_axes[~mask].any()
        active_rows = np.flatnonzero(mask.any(1))
        assert not predicted_axes[active_rows,anchor_index[active_rows]].any()
        assert not manual_axes[active_rows,anchor_index[active_rows]].any()
        np.testing.assert_array_equal(scores,np.where(mask,predicted_axes.max(2),np.inf))
        manual = np.where(mask,np.maximum(manual_axes[:,:,0],manual_axes[:,:,1]),np.inf)
        local_gap = float(np.abs(scores[mask] - manual[mask]).max()) if mask.any() else 0.
        assert local_gap <= 1e-10
        score_gap = max(score_gap, local_gap)
        assert set(choices['records'][model]) == set(ids)
        counts = Counter()
        for j, fid in enumerate(ids):
            record = choices['records'][model][fid]
            assert record['anchor_index'] == int(anchor_index[j])
            assert record['anchor_name'] == (names[int(anchor_index[j])] if anchor_index[j] >= 0 else None)
            assert record['candidate_valid'] == mask[j].tolist(), 'Runtime must use original geometric validity, never a reference-safe mask.'
            available = np.flatnonzero(mask[j]).tolist()
            selected = min(available, key=lambda k: (float(scores[j, k]), tie_key(names[k]))) if available else -1
            manual_selected = min(available, key=lambda k: (float(manual[j, k]), tie_key(names[k]))) if available else -1
            assert selected == manual_selected == record['candidate_index']
            assert record['scores'] == [float(s) if flag else None for s, flag in zip(scores[j], mask[j])]
            assert record['predicted_signed_axes'] == [axes.tolist() if flag else None for axes,flag in zip(predicted_axes[j],mask[j])]
            if selected>=0:
                assert scores[j,selected]<=0. and manual[j,manual_selected]<=0.
            for k in available:
                np.testing.assert_allclose(record['predicted_signed_axes'][k],manual_axes[j,k],rtol=0,atol=1e-10)
            assert record['candidate_name'] == (names[selected] if selected >= 0 else None)
            assert record['fallback'] == (selected < 0)
            pose = selected_pose(record, poses['records'], fid)
            assert record['pose_available'] == bool(pose['available'])
            expected_status = 'SELECTED' if selected >= 0 else 'R0_GEO_FALLBACK' if pose['available'] else 'FAILED'
            assert record['status'] == expected_status
            counts[expected_status] += 1
        model_checks[model] = dict(rows=1024, original_geometric_mask_exact=True,
            selected_indices_and_names_exact=True, shared_score_cells_exact=int(mask.sum()),
            independent_score_max_absolute_error=local_gap, status_counts=dict(counts),
            scalar189_map_bit_exact=True, independent253_map_bit_exact=True,
            direction_projection_within_fixed_tolerance=True,direction_normalization_difference_bit_exact=True,
            extended271_input_bit_exact=True,
            candidate_minus_anchor_bit_exact=True, predicted_signed_axes_saved_exact=True,
            predicted_axis_cells=2*int(mask.sum()), independent_axis_max_absolute_error=local_axes_gap,
            difference_sha=array_sha(difference), runtime_reference_targets_used=False,
            frozen_RBF_basis=basis_binding, context_sha=array_sha(manual_context),
            newton_trace=trace_check,checkpoint=receipt['checkpoint'], fit_receipt=receipt_binding)
    assert fit_models == set(LEARNED)
    assert total_calls == complete['total_objective_calls'] and total_iterations == complete['total_iterations']
    anchor_checks['independent_axis_max_absolute_difference']=axes_gap
    assert not STATE['references_allowed']
    return protocol, lock, contract, rows, poses, choices, model_checks, score_gap, anchor_checks


def physical_errors(pose, renderer_rotation, renderer_translation):
    """Canonical SO(3) trace metric, cross-checked by Frobenius inner product."""
    if not pose['available']:
        return np.array([np.inf, np.inf], np.float64), 0.
    rotation = np.asarray(pose['R_physical'], np.float64)
    translation = np.asarray(pose['centroid'], np.float64)
    reference = np.asarray(renderer_rotation, np.float64) @ np.diag([1., -1., -1.])
    assert rotation.shape == reference.shape == (3, 3) and translation.shape == (3,)
    assert np.isfinite(rotation).all() and np.isfinite(translation).all()
    angles, alternate = [], []
    for symmetry in (np.eye(3), np.diag([-1., 1., -1.])):
        equivalent = reference @ symmetry
        relative = equivalent.T @ rotation
        cosine = np.clip((np.trace(relative) - 1.) / 2., -1., 1.)
        angles.append(float(np.degrees(np.arccos(cosine))))
        inner = float(np.sum(equivalent * rotation))
        alternate.append(float(np.degrees(np.arccos(np.clip((inner - 1.) / 2., -1., 1.)))))
    error = np.array([float(np.linalg.norm(translation - renderer_translation) * 100.), min(angles)])
    assert np.isfinite(error).all() and (error >= 0).all()
    gap = abs(error[1] - min(alternate))
    assert gap <= 1e-6, ('EQUIVALENT_C2_FORMULA_GAP', gap)
    return error, gap


def quantile(values, fraction):
    values = sorted(float(v) for v in values)
    assert not any(math.isnan(v) or v < 0 for v in values)
    if not values:
        return None
    rank = (len(values)-1) * fraction
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return values[lower]
    if math.isinf(values[upper]):
        return float('inf')
    return values[lower] + (rank-lower) * (values[upper]-values[lower])


def summary(errors):
    available = np.isfinite(errors).all(1)
    assert np.array_equal(available, np.isfinite(errors).any(1))
    value = dict(frames=len(errors), valid_pose=int(available.sum()), failed_pose=int((~available).sum()))
    for mode, rows in [('full_population', errors), ('conditional', errors[available])]:
        value[mode] = {}
        for j, metric in enumerate(METRICS):
            cell = {}
            for label, q in [('median', .5), ('P90', .9)]:
                result = quantile(rows[:, j], q)
                cell[label] = result if result is not None and math.isfinite(result) else None
                cell[label + '_status'] = 'NA_EMPTY' if result is None else 'FINITE' if math.isfinite(result) else 'POSITIVE_INFINITY'
            value[mode][metric] = cell
    return value


def checks(before, after):
    output = {'failure_no_increase': int((~np.isfinite(after).all(1)).sum()) <= int((~np.isfinite(before).all(1)).sum())}
    for j, metric in enumerate(METRICS):
        output[metric + '_median_strict'] = quantile(after[:, j], .5) < quantile(before[:, j], .5)
        output[metric + '_P90_guard'] = quantile(after[:, j], .9) <= 1.05 * quantile(before[:, j], .9)
    return output


def reference_chain(contract):
    history = ROOT / '_docs/experiments/pallet_selector_recovery_v1/stage2_synth_scorer/SYNTHETIC_SPLIT_LOCK.json'
    geometry = ROOT / 'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
    record_path = SOURCE / 'SYNTH_RECORDS.json'
    expected = {}
    for path in (history, geometry, record_path):
        rel = str(path.relative_to(ROOT))
        matches = [b for b in bindings(contract) if b['path'] == rel]
        assert matches and all(b == matches[0] for b in matches)
        expected[rel] = matches[0]
    history_binding = expected[str(history.relative_to(ROOT))]
    verify(history_binding)
    historical = read(history)
    for path in (geometry, record_path):
        rel = str(path.relative_to(ROOT))
        matches = [b for b in bindings(historical) if b['path'] == rel]
        assert matches == [expected[rel]]
        verify(expected[rel])
    return history_binding, expected, geometry, record_path


def run():
    sys.dont_write_bytecode = True
    assert (DOC / 'SOURCE_VAL_GATE.json').is_file(), 'Wait for completed fits, frozen routes, and source score before verification.'
    assert not (DOC / 'SOURCE_VAL_VERIFICATION.json').exists()
    assert not (DOC / 'SOURCE_VAL_VERIFICATION_KO.md').exists()
    install_guard()
    protocol, lock, contract, rows, poses, choices, route_checks, score_gap, anchor_checks = verify_routes()
    # The complete GT-free route replay and its original validity masks passed.
    STATE['references_allowed'] = True
    gate = read(DOC / 'SOURCE_VAL_GATE.json')
    assert gate['complete'] and gate['routing_lock'] == bind(DOC / 'SOURCE_VAL_ROUTING_LOCK.json')
    assert gate['protocol'] == bind(DOC / 'TRAIN_PROTOCOL.json')
    assert gate['training_complete'] == bind(DOC / 'TRAINING_COMPLETE.json')
    assert gate['frames'] == 1024 and gate['models'] == list(LEARNED) and gate['baselines'] == list(BASELINES)
    assert not gate['real_reference_accessed']
    assert gate['source_contract'] == lock['source_contract']
    for key in ('direction_inputs','directions','direction_receipt_binding','direction_normalization_sha'):
        assert gate[key]==lock[key]==choices[key]
    for key in METHOD_FIELDS:
        assert gate[key] == protocol[key]
    history_binding, references, geometry, record_path = reference_chain(contract)
    assert gate['source_history_binding'] == history_binding
    assert all(b == references[b['path']] for b in gate['source_reference_bindings'])
    ids = [row['id'] for row in rows]
    wanted = set(ids)
    metadata = {r['id']: r for r in read(record_path) if r['id'] in wanted}
    assert set(metadata) == wanted and all(r['split'] == 'VAL' for r in metadata.values())
    index = np.array([metadata[fid]['table_index'] for fid in ids], np.int64)
    assert len(set(index.tolist())) == 1024
    with np.load(geometry, allow_pickle=False) as table:
        assert table['stems'][index].tolist() == ids
        rotations, translations = table['R'][index], table['t'][index]
        intrinsics, dimensions = table['K'][index], table['dims'][index]
    for j, row in enumerate(rows):
        fx, fy, cx, cy = intrinsics[j]
        np.testing.assert_allclose(row['K'], [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], rtol=0, atol=1e-7)
        np.testing.assert_allclose(row['dims'], dimensions[j], rtol=0, atol=1e-7)
    recomputed = {m: np.full((1024, 2), np.inf, np.float64) for m in LEARNED + BASELINES}
    frobenius_gap = 0.
    for j, fid in enumerate(ids):
        for model in LEARNED + BASELINES:
            pose = selected_pose(choices['records'][model][fid], poses['records'], fid) if model in LEARNED else poses['records'][model[:-4]][fid]['GEO_pose']
            value, difference = physical_errors(pose, rotations[j], translations[j])
            recomputed[model][j] = value
            frobenius_gap = max(frobenius_gap, difference)
    verify(gate['metrics'])
    physical_max_gap = 0.
    with np.load(ROOT / gate['metrics']['path'], allow_pickle=False) as stored:
        assert stored['ids'].tolist() == ids and stored['models'].tolist() == list(LEARNED + BASELINES)
        assert stored['metric_names'].tolist() == list(METRICS)
        np.testing.assert_array_equal(stored['source_table_index'], index)
        saved = {m: stored[m] for m in LEARNED + BASELINES}
    for model, errors in recomputed.items():
        np.testing.assert_array_equal(errors, saved[model])
        finite = np.isfinite(errors)
        physical_max_gap = max(physical_max_gap, float(np.abs(errors[finite] - saved[model][finite]).max()) if finite.any() else 0.)
        assert summary(errors) == gate['summaries'][model]
    old_gate_path = ROOT / '_docs/experiments/pallet_pose_signed_axes_sign_20261001_v1/SOURCE_VAL_GATE.json'
    assert bind(old_gate_path)==protocol['evidence']['previous_source_gate']
    old_gate = read(old_gate_path)
    assert old_gate['complete'] and old_gate['frames'] == 1024
    assert old_gate['source_contract'] == gate['source_contract']
    assert old_gate['source_history_binding'] == history_binding and old_gate['source_reference_bindings'] == gate['source_reference_bindings']
    verify(old_gate['metrics'])
    baseline_parity = {}
    with np.load(ROOT / old_gate['metrics']['path'], allow_pickle=False) as previous:
        assert previous['ids'].tolist() == ids
        for model in BASELINES:
            assert previous[model].dtype == saved[model].dtype == np.float64
            assert previous[model].shape == saved[model].shape == (1024, 2)
            assert previous[model].tobytes() == saved[model].tobytes()
            baseline_parity[model] = dict(error_cells=2048, bit_exact=True, array_sha=array_sha(saved[model]))
    failed, count = [], 0
    for seed in (1, 2, 3):
        model = f'UNION_s{seed}'
        for baseline in ('R0_ONLY', 'R0_GEO', f'DIVERSE251_s{seed}_GEO'):
            original = gate['comparisons'][model][baseline]
            actual = checks(recomputed[baseline], recomputed[model])
            assert actual == original['checks']
            assert original['PASS'] == all(actual.values())
            assert original['before'] == summary(recomputed[baseline]) and original['after'] == summary(recomputed[model])
            count += len(actual)
            failed.extend(f'{model}/{baseline}/{criterion}' for criterion, passed in actual.items() if not passed)
    assert count == gate['checks_total'] == 45 and failed == gate['failed_checks']
    assert 45 - len(failed) == gate['checks_passed'] and gate['PASS'] == (not failed)
    assert gate['real_routing_authorized'] == gate['PASS']
    verdict = 'PASS' if gate['PASS'] else 'FAIL'
    from datetime import datetime, timezone
    result = dict(complete=True, PASS=True, created_at=datetime.now(timezone.utc).isoformat(),
        independent_verification=True, implementation='GT-free replay plus independent scalar189 context, fixed Gaussian64, candidate-minus-operational-anchor253 plus scalar-projected direction18 and scalar two-axis dot products/max/tie selection; direct cached-pose C2 trace and Frobenius checks; independent sorted interpolation and all45 gates. No evaluate_source/source_features metric helper imported.',
        verifier=bind(Path(__file__)), source_gate=bind(DOC / 'SOURCE_VAL_GATE.json'),
        protocol=bind(DOC / 'TRAIN_PROTOCOL.json'), training_complete=bind(DOC / 'TRAINING_COMPLETE.json'),
        routing_lock=bind(DOC / 'SOURCE_VAL_ROUTING_LOCK.json'), choices=bind(ROOT / lock['choices']['path']),
        reference_chain=list(references.values()), VAL_table_index_sha=array_sha(index),
        route_rows_reproduced=4096, route_models=route_checks,
        operational_anchor_identity=anchor_checks, feature_map=protocol['feature_map'],
        feature_dim=271, context_dim=189, rbf_dim=64,
        frozen_RBF_basis=protocol['inputs']['rbf_basis'],
        signed_target_and_input_review=protocol['inputs']['prefit_review'],
        independent_basis_selection_review=read(ROOT/protocol['inputs']['prefit_review']['path'])['inputs']['previous_RBF_prefit'],
        basis_provenance='Existing RBF PREFIT verifies TRAIN-only centers. Signed PREFIT verifies exact reuse and anchor difference. Replay checks the fixed payload; no basis is reselected or updated.',
        solver_rule=protocol['solver_rule'],solver_config=protocol['solver'],
        sign_rule=protocol['sign_rule'],sign_coefficient=protocol['sign_coefficient'],
        loss_rule=protocol['loss_rule'], target_rule=protocol['target_rule'], input_rule=protocol['input_rule'],
        prediction_rule=protocol['prediction_rule'], output_dim=2,
        runtime_uses_margin=False,runtime_uses_reference_errors=False,
        target_provenance='Six TRAIN target/input/error hashes checked across PREFIT, CK, START, FIT, TRACE and COMPLETE. No TRAIN target or reference error values are opened/reconstructed for replay.',
        independent_axis_max_absolute_difference=anchor_checks['independent_axis_max_absolute_difference'],
        original_runtime_valid_masks_preserved=True, GT_safe_mask_used_in_runtime=False,
        independent_score_max_absolute_difference=score_gap, source_VAL_reference_rows_scored=1024,
        source_TRAIN_reference_rows_scored=0, source_TEST_reference_rows_scored=0, real_reference_rows_scored=0,
        recomputed_pose_errors=8192, recomputed_error_cells_exact=16384,
        physical_metric_max_absolute_difference=physical_max_gap,
        alternate_Frobenius_C2_max_difference_deg=frobenius_gap,
        summary_models_exact=8, full_population_frames_per_model=1024,
        fixed_GEO_baseline_bit_parity=baseline_parity, previous_baseline_metrics=old_gate['metrics'],
        previous_baseline_gate=bind(old_gate_path), checks_total=45, checks_passed=gate['checks_passed'],
        failed_checks=failed, source_gate_PASS=gate['PASS'], all_gate_booleans_exact=True,
        label_container_disclosure='The historically bound geometry NPZ contains broader source rows; only the1024 verified VAL indices were selected and scored. No TRAIN/TEST/real reference errors were computed.',
        source_reference_read_only_after_all_routes_verified=True, read_paths=sorted(STATE['reads']),
        new_fits=0, image_forwards=0, new_PnP_calls=0, new_route_artifacts=0, threshold_changes=0,
        real_GT_reads=0, method_success=False, goal_complete=False)
    markdown = f'''# Sign-logistic Source VAL 독립 검산

**검산 PASS. 학습된 방법의 source gate는 {verdict} ({gate['checks_passed']}/45)**이다. 검산 통과와 방법 성공은 별개다.

GT 접근을 막은 상태에서 최종4개 checkpoint의 **4,096개 선택**을 원래 cached feature로 재현했다. 입력-only R0 GEO index를 GEO 이름과 완전한 pose dictionary에 대조했다. float32 정규화→float64 z94, abs(z−z_anchor)94, identity1과 고정 Gaussian64의 phi253을 독립 계산한 뒤 같은 행 anchor의 phi253을 뺐다. 원래189·phi253·anchor 차분 입력 모두 실제 구현과 bit 단위로 같다. 별도 후보·축 반복의 내적으로 두 signed-axis 예측을 계산하고 max 점수와 기존 동률 규칙으로 선택했다. 저장된 모든 두 축 예측값은 공유 scorer와 정확히 같고 독립 계산 최대 차이는 {anchor_checks['independent_axis_max_absolute_difference']:.3g}, score 차이는 {score_gap:.3g}이다. 모든 index·후보 이름·fallback·원래 geometric-valid mask가 일치했다.

anchor 입력과 두 예측은 정확히0이다. 이는 예측상 선택 점수≤0일 뿐 실제 T/R 비악화 보장은 아니다. 추론에서 TRAIN target 생성기를 차단한 합성 검산도 통과했으며 margin·참조 기반 safe mask·새 GT target을 사용하지 않는다. 기존6개 및 추가3개 TRAIN 해시는 PREFIT·checkpoint·START·FIT·TRACE·COMPLETE에서 provenance로만 대조했다.

center 선정 자체는 기존 RBF PREFIT에 근거하며 이번 signed PREFIT는 해당 phi253의 정확한 재사용과 anchor 차분을 확인했다. checkpoint embedded basis와 원본 RBF_BASIS 바인딩, normalization SHA, 고정 bandwidth를 확인했다. VAL 입력이나 품질로 center·폭·정규화를 다시 만들거나 맞추지 않았다. 추가18개는 O에서 동결한 R0 TRAIN 정규화만 사용한다. 모든 VAL 방향을 원래 q9/K/고정 pose로 독립 재투영했고, float32 정규화와 anchor 차분을 별도 계산해 확장271 입력과 비교했다. invalid 후보는271 입력과 두 예측을 모두0으로 유지한다.

Huber+Sign_logistic+L2 합이 objective와 같은지 확인했다. 계수1과 nonzero-target sign 규칙은 메타데이터에서 대조했으며 source 추론에는 sign label/참조오차를 전달하지 않았다. 모든 objective trial을 포함한 Newton trace의 초기점·채택점 연결, Armijo 값/alpha 순서, gradient L∞≤1e-8 및 gap≤1e-6, 최종 채택 call을 추가 확인했다. 거부 trial 뒤 이전 채택점을 보존하는 계약이며 초기평가와 모든 거부 trial도2,000회 예산에 포함한다. 이 검산에서 TRAIN 최적화나 타깃 계산은 하지 않았다.

전체 routing lock과 모든 최종 fit 바인딩 확인 후에만 기존 source 참조의 **VAL1,024개 index**를 읽었다. renderer R에 `diag(1,-1,-1)`를 적용한 물리 기준과 C2 대칭 회전, 중심 이동 cm를 직접 계산했다. 평가 helper의 metric/gate 함수는 import하지 않았다. **8모델×1,024 pose = 8,192개**, T/R **16,384개 오류 값**이 저장 배열과 정확히 일치한다. 동일 회전각의 Frobenius 식도 확인했으며 최대 차이는 {frobenius_gap:.3g}°다.

직전 signed_axes_sign 결과의 R0 GEO 및 DIVERSE3 GEO **8,192개 오류 값**은 현재 fixed baseline과 byte 단위로 같다. 기존 결과를 새 모델 성능처럼 다시 생성하거나 baseline을 바꾸지 않았다.

전체1,024분모·실패/+∞·median/P90을 독립 정렬 보간으로 확인하고, 각 UNION을 R0_ONLY/R0_GEO/paired DIVERSE_GEO와 비교하는 **45개 Boolean**을 모두 재현했다. 임계값·seed·checkpoint를 바꾸지 않았다. 실패 목록은 JSON에 전부 남긴다.

원본 geometry 컨테이너에는 다른 source 행도 들어 있지만 검산 대상은 잠금된 VAL1,024 index뿐이다. TRAIN/TEST/실사 참조 오류 계산, 새 fit, 이미지 forward, PnP 및 새 routing 산출물은0회다. 이 VAL은 반복 사용 개발 자료이며 실제 실사 개선이나 전체 목표 달성의 증거가 아니다.

[검산 JSON](SOURCE_VAL_VERIFICATION.json), [source gate](SOURCE_VAL_GATE.json), [선택 동결](SOURCE_VAL_ROUTING_LOCK.json)
'''
    with (DOC / 'SOURCE_VAL_VERIFICATION_KO.md').open('x') as handle:
        handle.write(markdown)
    result['note'] = bind(DOC / 'SOURCE_VAL_VERIFICATION_KO.md')
    with (DOC / 'SOURCE_VAL_VERIFICATION.json').open('x') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('SOURCE_VAL_INDEPENDENT_VERIFICATION_PASS', dict(source_gate=verdict,
        checks_passed=gate['checks_passed'], route_rows=4096, error_cells=16384,
        baseline_bit_exact_cells=8192, receipt=bind(DOC / 'SOURCE_VAL_VERIFICATION.json')), flush=True)


def selfcheck():
    from . import convex_train as T
    assert T.OLD.READS is None and T.READS is None
    pose = dict(available=True, R_physical=np.eye(3).tolist(), centroid=[0., 0., 1.])
    basis = np.diag([1., -1., -1.])
    np.testing.assert_array_equal(physical_errors(pose, basis, np.array([0., 0., 1.]))[0], [0., 0.])
    np.testing.assert_array_equal(physical_errors(pose, np.diag([-1., -1., 1.]), np.array([.01, 0., 1.]))[0], [1., 0.])
    assert np.isposinf(physical_errors(dict(available=False), basis, np.zeros(3))[0]).all()
    equal = np.array([[1., 2.], [3., 4.]])
    assert not checks(equal, equal)['translation_cm_median_strict']
    assert quantile([1., np.inf], .5) == np.inf
    assert summary(np.array([[1., 2.], [np.inf, np.inf]]))['failed_pose'] == 1
    assert T.metadata()==METHOD_FIELDS
    checkpoint = dict(**METHOD_FIELDS,schema='pallet_pose_signed_axes_direction_linear271x2_v1',bias=0.,lambda_l2=1e-4,
        normalization='old_float32_then_float64',names=names_for('UNION_s1'),
        weight=(np.arange(542,dtype=float).reshape(271,2)/100).tolist(),
        mean=np.zeros(94).tolist(),std=np.ones(94).tolist())
    centers = np.zeros((64,189),np.float64)
    centers[:,0] = np.arange(64)/10
    width = float(np.median([(centers[i,0]-centers[j,0])**2 for i in range(64) for j in range(i+1,64)]))
    checkpoint['rbf_basis'] = dict(schema='pallet_pose_anchor_rbf_runtime_v1',context_dim=189,rbf_dim=64,
        centers=centers.tolist(),bandwidth_squared=width,
        normalization_sha=array_sha(np.stack([np.zeros(94,np.float32),np.ones(94,np.float32)])))
    checkpoint['rbf_basis_binding'] = checkpoint['basis_SHA_bind'] = {'invented':True}
    x = np.arange(2*4*94, dtype=np.float32).reshape(2, 4, 94) / np.float32(100)
    valid = np.array([[True, False, True, True], [False]*4])
    anchor_index = np.array([0, -1], np.int64)
    direction=np.arange(2*4*18,dtype=np.float32).reshape(2,4,18)/np.float32(400)
    direction[~valid]=0.
    dm=np.linspace(-.1,.1,18,dtype=np.float32);ds=np.linspace(.5,2.,18,dtype=np.float32)
    checkpoint.update(direction_mean=dm.tolist(),direction_std=ds.tolist(),
        direction_normalization_sha=array_sha(np.stack([dm,ds])),direction_receipt_binding={'invented':True})
    independent = independent_context(checkpoint, x, valid, anchor_index)
    np.testing.assert_array_equal(independent[:,:,:189], T.context_inputs(x, valid, anchor_index, checkpoint['mean'], checkpoint['std']))
    np.testing.assert_array_equal(independent, T.rbf_inputs(x, valid, anchor_index, checkpoint['mean'], checkpoint['std'],checkpoint['rbf_basis']))
    scalar = np.zeros((2,4,64))
    for i,j in zip(*np.where(valid)):
        for c in range(64):
            distance=sum(float(independent[i,j,k]-centers[c,k])**2 for k in range(189))
            scalar[i,j,c]=np.exp(-distance/(2*width))
    np.testing.assert_allclose(scalar,independent[:,:,189:],rtol=1e-12,atol=1e-12)
    assert not independent[~valid].any()
    np.testing.assert_allclose(independent_scores(checkpoint, x, valid, anchor_index,direction),
                               T.score_candidates(checkpoint, x, valid, anchor_index,direction), rtol=1e-15, atol=1e-10)
    assert not STATE['references_allowed'] and not STATE['reads']
    difference=independent_extended_difference(checkpoint,x,valid,anchor_index,direction)
    np.testing.assert_array_equal(difference,T.difference_inputs(x,valid,anchor_index,checkpoint['mean'],checkpoint['std'],checkpoint['rbf_basis'],direction,dm,ds))
    np.testing.assert_array_equal(difference[0,0],0.)
    np.testing.assert_allclose(independent_axes(checkpoint,x,valid,anchor_index,direction),T.predict_axes(checkpoint,x,valid,anchor_index,direction),rtol=1e-15,atol=1e-10)
    original_targets=T.signed_targets
    try:
        def forbidden_targets(*args,**kwargs):
            raise AssertionError('NO_RUNTIME_REFERENCE_TARGETS')
        T.signed_targets=forbidden_targets
        np.testing.assert_allclose(independent_scores(checkpoint,x,valid,anchor_index,direction),
            T.score_candidates(checkpoint,x,valid,anchor_index,direction),rtol=1e-15,atol=1e-10)
    finally:
        T.signed_targets=original_targets
    # Explicit mixed axes are aggregated by max, not averaging or absolute value.
    assert np.maximum(np.array([-2.,-3.]),np.array([1.,-1.])).tolist()==[1.,-1.]
    scores=independent_scores(checkpoint,x,valid,anchor_index,direction)
    assert scores[0,0]==0. and np.isposinf(scores[1]).all()
    assert T.select_candidates(scores,valid,names_for('UNION_s1'))[1]==-1
    # Exact-zero ties use OLD R0/hypothesis key; anchor has no special tie priority.
    tied=T.select_candidates(np.zeros((1,4)),np.ones((1,4),bool),names_for('UNION_s1'))
    assert tied.tolist()==[0]
    # Invented log exercises rejected trials and immutable accepted-parent links.
    def logged(call,value,gradient,alpha,accepted):
        return dict(event='objective',phase='initial' if call==1 else 'trial',call=call,
            iteration=0 if call==1 else 1,alpha=alpha,evaluated=True,
            objective=value,Huber=value-.3,Sign_logistic=.2,L2_penalty=.1,gradient_l2=gradient,gradient_linf=gradient,
            gradient_gap_upper_bound=gradient**2/(2e-4),weight_sha=str(call)*64,
            accepted_point_call_before=None if call==1 else 1,
            parent_weight_sha=None if call==1 else '1'*64,parent_objective=None if call==1 else 2.,
            directional_derivative=None if call==1 else -1.,
            armijo_bound=None if call==1 else 2.-1e-4*alpha,armijo_accepted=accepted)
    log=[logged(1,2.,1.,0.,True),logged(2,3.,1.,1.,False),
         logged(3,2.5,1.,.5,False),logged(4,1.,1e-10,.25,True)]
    iteration={key:value for key,value in log[-1].items() if key not in ('call','phase','evaluated','accepted_point_call_before')}
    iteration.update(event='iteration',iteration=1,objective_calls=4,accepted_call=4)
    log.append(iteration)
    cert=dict(objective_calls=4,iterations=1,final_accepted_call=4,objective_value=1.,Huber=.7,Sign_logistic=.2,L2_penalty=.1,
              gradient_l2=1e-10,gradient_linf=1e-10)
    fit=dict(objective_calls=4,iterations=1,final_accepted_call=4,final_weight_sha='4'*64)
    ck=dict(final_accepted_call=4,certificate=cert,solver=dict(final_accepted_call=4,last_evaluated_call=4))
    receipt=independent_trace(log,dict(initial_weight_sha='1'*64),fit,ck)
    assert receipt['rejected_trials']==2 and receipt['objective_calls']==4
    import copy
    for position,key,value in ((2,'alpha',.25),(3,'accepted_point_call_before',3),(4,'accepted_call',3)):
        broken=copy.deepcopy(log);broken[position][key]=value
        try:independent_trace(broken,dict(initial_weight_sha='1'*64),fit,ck)
        except AssertionError:pass
        else:raise AssertionError('CORRUPTED_NEWTON_TRACE_ACCEPTED')
    toy_pose=dict(available=True,R_cf=np.eye(3).tolist(),centroid=[.2,-.1,4.],cf_extents=[1.2,.14,.8])
    toy_K=np.array([[600.,2.,320.],[0.,620.,240.],[0.,0.,1.]])
    toy_q=np.full((9,2),200.);toy_box=[10.,20.,610.,460.]
    from . import direction_features as D
    d,norm,diagonal=scalar_direction(toy_pose,toy_q,toy_box,toy_K)
    produced=D.project_direction(toy_pose,toy_q,toy_box,toy_K)
    np.testing.assert_array_equal(d,produced['direction18'])
    np.testing.assert_allclose(norm,produced['norm_px'],rtol=1e-14,atol=1e-12)
    assert diagonal==produced['diagonal']
    print('VERIFY_SOURCE_SELFCHECK_PASS: invented C2/failure/quantile, scalar189+fixedGaussian64, anchor difference253+18, signed-two-axis dot/max and GT-target-free runtime; artifact reads0.',flush=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('selfcheck', 'verify'))
    arguments = parser.parse_args()
    (selfcheck if arguments.stage == 'selfcheck' else run)()
