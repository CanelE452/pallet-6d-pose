"""Independent TRAIN-only audit of frozen reprojection-direction information.

No model fitting, candidate ranking, physical quality evaluation, or new PnP.
The linear algebra below projects INPUT columns only; it never accepts targets.
``selfcheck`` uses invented arrays and performs no artifact reads.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

MODELS=('R0','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3')
HYP=('long-face-front','short-face-front')
HASH_KEYS=('signed_target_sha','input_difference_sha','base_context_sha','errors_sha','scaled_excess_sha','original_valid_sha')
READS=[]
ALLOWED_PREDICTIONS=set()


def install_guard(C):
    """Deny raw references, images, model weights, VAL quality and new policies."""
    sys.dont_write_bytecode=True
    allowed_arrays={C.PARENT_RAW/'SOURCE_FEATURES.npz',C.PARENT_RAW/'SOURCE_TRAIN_LABELS.npz',
                    C.RAW/'TRAIN_DIRECTIONS.npz',C.RAW/'REPRESENTATION_ARRAYS.npz'}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=Path(os.fsdecode(args[0])).resolve();name=str(path)
        mode=args[1];flags=args[2] if len(args)>2 else 0
        write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if write and path.is_relative_to(C.ROOT):
            assert path in {C.DOC/'VERIFICATION.json',C.DOC/'VERIFICATION_KO.md'},('VERIFIER_OUTPUT_ONLY',name)
        if path.suffix in ('.py','.pyc'):
            return
        forbidden=('GEOMETRY_SIDETABLE','GEOMETRY_RESOLVED_POSE_GT','AXIS_REVIEW',
            'SYNTH_RECORDS','SYNTH_LABELS','SOURCE_VAL_','REAL_EVALUATION','REAL_RESULTS',
            'TRAIN_CONVERGENCE','POSE_METRICS','TRUTH_FOR_DISPLAY','/data/evaluation/',
            '/fits/','/model_parameters/','rejected_optimizer_state')
        assert not any(token in name for token in forbidden),('FORBIDDEN_AUDIT_READ',name)
        assert path.suffix.lower() not in ('.jpg','.jpeg','.png','.webp','.pt','.pth'),('NO_IMAGE_OR_CHECKPOINT',name)
        if not path.is_relative_to(C.ROOT):
            return
        if '/source_predictions/' in name:
            assert name in ALLOWED_PREDICTIONS,('ONLY_ELIGIBLE_TRAIN_PREDICTIONS',name)
        if path.suffix=='.npz':
            assert path in allowed_arrays,('ONLY_FROZEN_TRAIN_ARRAYS',name)
        READS.append(name)
    sys.addaudithook(hook)


def verify_bindings(C,value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            C.verify(value)
        else:
            for item in value.values():
                verify_bindings(C,item)
    elif isinstance(value,list):
        for item in value:
            verify_bindings(C,item)


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def scalar_projection(pose, camera, points, box):
    """Explicit component equations, separate from the producer's matmul."""
    camera = np.asarray(camera, np.float64)
    rotation = np.asarray(pose['R_cf'], np.float64)
    translation = np.asarray(pose['centroid'], np.float64)
    extent = np.asarray(pose['cf_extents'], np.float64)
    points, box = np.asarray(points, np.float64), np.asarray(box, np.float64)
    assert camera.shape == rotation.shape == (3, 3)
    assert translation.shape == extent.shape == (3,)
    assert points.shape == (9, 2) and box.shape == (4,)
    assert all(np.isfinite(v).all() for v in (camera, rotation, translation, extent, points, box))
    assert (extent > 0).all()
    signs = ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
             (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1),(0,0,0))
    projection = np.empty((9, 2), np.float64)
    depth = np.empty(9, np.float64)
    for k, sign in enumerate(signs):
        xyz = [float(sign[a]) * float(extent[a]) / 2. for a in range(3)]
        cam = [sum(float(rotation[a,b]) * xyz[b] for b in range(3)) + float(translation[a])
               for a in range(3)]
        homogeneous = [sum(float(camera[a,b]) * cam[b] for b in range(3)) for a in range(3)]
        assert np.isfinite(homogeneous).all() and homogeneous[2] != 0.
        depth[k] = cam[2]
        for a in range(2):
            projection[k,a] = homogeneous[a] / homogeneous[2]
    wh = [max(float(box[a+2])-float(box[a]), 1e-6) for a in range(2)]
    diagonal = math.sqrt(wh[0]**2 + wh[1]**2)
    residual = projection - points
    norm9 = np.asarray([math.sqrt(float(v[0])**2+float(v[1])**2) for v in residual])
    direction = np.asarray([float(residual[k,a]) / diagonal for k in range(9) for a in range(2)], np.float32)
    assert np.isfinite(direction).all() and diagonal > 0
    return dict(projected=projection, depth=depth, residual=residual, norm9=norm9,
                diagonal=diagonal, direction18=direction)


def independent_normalization(direction, valid):
    direction, valid = np.asarray(direction), np.asarray(valid)
    assert direction.dtype == np.float32 and valid.dtype == bool
    assert direction.shape == (*valid.shape, 18) and np.isfinite(direction).all()
    values = direction[valid]
    assert len(values)
    mean = values.mean(axis=0, dtype=np.float32)
    std = np.maximum(values.std(axis=0, dtype=np.float32), np.float32(1e-6))
    return mean, std


def independent_direction_difference(direction, valid, anchor_index, mean, std):
    """Original FP32 normalize first; promote before anchor subtraction."""
    out = np.zeros(direction.shape, np.float64)
    for row in range(len(valid)):
        candidates = np.flatnonzero(valid[row])
        if not len(candidates):
            assert anchor_index[row] == -1
            continue
        anchor = int(anchor_index[row])
        assert anchor in (0, 1) and valid[row,anchor]
        normalized = {}
        for candidate in candidates:
            value = np.divide(np.subtract(direction[row,candidate], mean), std)
            assert value.dtype == np.float32 and np.isfinite(value).all()
            normalized[int(candidate)] = value.astype(np.float64)
        for candidate in candidates:
            out[row,candidate] = normalized[int(candidate)]-normalized[anchor]
    assert np.isfinite(out).all() and not out[~valid].any()
    return out


def canonical_bytes(row):
    row = np.array(row, np.float64, copy=True, order='C')
    row[row == 0.] = 0.
    return row.tobytes()


def independent_collisions(inputs, targets):
    """Exact input bytes after +/-0 canonicalization, numeric target equality."""
    inputs, targets = np.asarray(inputs), np.asarray(targets)
    assert inputs.ndim == targets.ndim == 2 and len(inputs) == len(targets)
    assert inputs.dtype == targets.dtype == np.float64
    assert np.isfinite(inputs).all() and np.isfinite(targets).all()
    groups = {}
    for row in range(len(inputs)):
        groups.setdefault(canonical_bytes(inputs[row]), []).append(row)
    duplicate = [rows for rows in groups.values() if len(rows) > 1]
    conflicts, sign_conflicts = [], []
    for rows in duplicate:
        values = {tuple(float(v) for v in targets[row]) for row in rows}
        signs = {tuple(int(v) for v in np.sign(targets[row])) for row in rows}
        if len(values) > 1:
            conflicts.append(rows)
        if len(signs) > 1:
            sign_conflicts.append(rows)
    return dict(rows=len(inputs), unique_input_rows=len(groups), duplicate_groups=len(duplicate),
                rows_in_duplicate_groups=sum(map(len, duplicate)),
                conflicting_target_groups=len(conflicts), rows_in_conflicting_target_groups=sum(map(len, conflicts)),
                conflicting_sign_groups=len(sign_conflicts), rows_in_conflicting_sign_groups=sum(map(len, sign_conflicts)))


def grouped_rows(matrix):
    mapping={};members=[];digest=[];assignment=[]
    for i,row in enumerate(matrix):
        key=canonical_bytes(row)
        if key not in mapping:
            mapping[key]=len(members);members.append([]);digest.append(hashlib.sha256(key).hexdigest())
        assignment.append(mapping[key]);members[mapping[key]].append(i)
    return np.asarray(assignment,np.int64),members,digest


def describe_targets(target):
    out={}
    for axis,name in enumerate(('T','R')):
        vals=sorted(float(row[axis]) for row in target)
        signs=sorted({-1 if v<0 else 1 if v>0 else 0 for v in vals})
        out[name]=dict(minimum=vals[0],maximum=vals[-1],range=vals[-1]-vals[0],
            distinct_values=len(set(vals)),signs=signs,target_conflict=vals[0]!=vals[-1],
            sign_conflict=len(signs)>1,strict_opposite=vals[0]<0<vals[-1])
    return out


def describe_groups(groups,target,anchors):
    out=dict(groups=len(groups),repeated_groups=0,repeated_candidate_rows=0,maximum_group_size=0,
        anchor_only_groups=0,nonanchor_only_groups=0,anchor_nonanchor_groups=0,
        axes={axis:{kind:dict(groups=0,candidate_rows=0) for kind in ('target_conflict','sign_conflict','strict_opposite')}
              for axis in ('T','R')})
    for members in groups:
        n=len(members);a=sum(bool(anchors[j]) for j in members)
        out['maximum_group_size']=max(out['maximum_group_size'],n)
        if n>1:out['repeated_groups']+=1;out['repeated_candidate_rows']+=n
        role='anchor_only_groups' if a==n else 'nonanchor_only_groups' if a==0 else 'anchor_nonanchor_groups'
        out[role]+=1
        described=describe_targets(target[members])
        for axis in ('T','R'):
            for kind in ('target_conflict','sign_conflict','strict_opposite'):
                if described[axis][kind]:
                    out['axes'][axis][kind]['groups']+=1
                    out['axes'][axis][kind]['candidate_rows']+=n
    return out


def verify_collision_record(record,x,extra,target,valid,index,ids,names,old_saved,new_saved):
    positions=[(i,j) for i in range(len(valid)) for j in range(valid.shape[1]) if valid[i,j]]
    anchors=np.array([j==index[i] for i,j in positions],bool)
    flatx=np.stack([x[i,j] for i,j in positions]);flate=np.stack([extra[i,j] for i,j in positions])
    y=np.stack([target[i,j] for i,j in positions]);extended=np.concatenate([flatx,flate],axis=1)
    og,oldgroups,oh=grouped_rows(flatx);ng,newgroups,nh=grouped_rows(extended)
    oldfull=np.full(valid.shape,-1,np.int64);newfull=oldfull.copy()
    for k,(i,j) in enumerate(positions):oldfull[i,j]=og[k];newfull[i,j]=ng[k]
    np.testing.assert_array_equal(oldfull,old_saved);np.testing.assert_array_equal(newfull,new_saved)
    assert record['old_group_assignment_sha']==array_sha(oldfull)
    assert record['extended_group_assignment_sha']==array_sha(newfull)
    assert record['old_all_valid']==describe_groups(oldgroups,y,anchors)
    assert record['extended_all_valid']==describe_groups(newgroups,y,anchors)
    non=~anchors
    for prefix,matrix in (('old',flatx),('extended',extended)):
        _,groups,_=grouped_rows(matrix[non])
        assert record[prefix+'_nonanchor_only']==describe_groups(groups,y[non],np.zeros(int(non.sum()),bool))
    expected=[]
    for gid,members in enumerate(oldgroups):
        if len(members)<2:continue
        subgroups=[]
        for subid in sorted({int(ng[j]) for j in members}):
            sub=[j for j in members if ng[j]==subid]
            subgroups.append(dict(extended_group_id=subid,vector_sha256=nh[subid],candidate_rows=len(sub),
                anchor_rows=int(anchors[sub].sum()),target=describe_targets(y[sub])))
        a=int(anchors[members].sum())
        examples=[dict(id=str(ids[positions[j][0]]),candidate=names[positions[j][1]],
            frame_index=positions[j][0],candidate_index=positions[j][1]) for j in members[:5]]
        expected.append(dict(old_group_id=gid,vector_sha256=oh[gid],candidate_rows=len(members),anchor_rows=a,
            nonanchor_rows=len(members)-a,mixed_anchor_nonanchor=0<a<len(members),forced_anchor_only=a==len(members),
            target=describe_targets(y[members]),extended_subgroups=len(subgroups),split=len(subgroups)>1,
            subgroups=subgroups,first_five_identity_examples=examples))
    assert record['repeated_old_groups']==expected
    assert record['old_repeated_groups_split']==sum(g['split'] for g in expected)
    for axis in ('T','R'):
        for kind in ('target_conflict','sign_conflict','strict_opposite'):
            affected=[g for g in expected if g['target'][axis][kind]]
            unresolved=sum(any(s['target'][axis][kind] for s in g['subgroups']) for g in affected)
            assert record['conflict_resolution'][axis][kind]==dict(old_groups=len(affected),
                split_groups=sum(g['split'] for g in affected),fully_resolved_groups=len(affected)-unresolved,
                unresolved_groups=unresolved)
    assert record['valid_candidate_rows']==len(positions)
    assert record['forced_anchor_rows']==int(anchors.sum())==2597
    assert record['nonanchor_candidate_rows']==int(non.sum())
    assert record['invalid_candidate_rows']==int((~valid).sum())
    assert record['all_invalid_frame_rows']==int((~valid.any(1)).sum())==1
    return dict(PASS=True,old_groups=len(oldgroups),extended_groups=len(newgroups),
        exact_target_range_and_sign_checks=True,anchor_groups_separated=True,
        all_subgroup_splits_and_conflict_resolution_verified=True,
        old_group_assignment_sha=array_sha(oldfull),extended_group_assignment_sha=array_sha(newfull))


def independent_old_features(raw, valid, anchor_index, mean, std, basis):
    """Reconstruct unchanged normalized94/context189/RBF253 without producer."""
    mean, std = np.asarray(mean,np.float32), np.asarray(std,np.float32)
    centers = np.asarray(basis['centers'],np.float64)
    width = float(basis['bandwidth_squared'])
    assert mean.shape == std.shape == (94,) and centers.shape == (64,189) and width > 0
    phi = np.zeros((*valid.shape,253),np.float64)
    for row in range(len(valid)):
        available = np.flatnonzero(valid[row])
        if not len(available):
            assert anchor_index[row] == -1
            continue
        anchor = int(anchor_index[row]);assert anchor in (0,1) and valid[row,anchor]
        z = {int(j):np.divide(np.subtract(raw[row,j],mean),std).astype(np.float64) for j in available}
        for j in available:
            phi[row,j,:94] = z[int(j)]
            phi[row,j,94:188] = np.abs(z[int(j)]-z[anchor])
            phi[row,j,188] = float(j==anchor)
            for k in range(64):
                squared = np.sum((phi[row,j,:189]-centers[k])**2,dtype=np.float64)
                phi[row,j,189+k] = np.exp(-squared/(2.*width))
    difference = np.zeros_like(phi)
    for row,j in zip(*np.nonzero(valid)):
        difference[row,j] = phi[row,j]-phi[row,anchor_index[row]]
    assert np.isfinite(phi).all() and not phi[~valid].any() and not difference[~valid].any()
    return phi,difference


def independent_targets(errors, valid, anchor, anchor_index, scale):
    scaled,target = np.zeros_like(errors),np.zeros_like(errors)
    for row in range(len(valid)):
        if not valid[row].any():
            assert anchor_index[row] == -1 and np.isposinf(anchor[row]).all()
            continue
        np.testing.assert_array_equal(errors[row,anchor_index[row]],anchor[row])
        for j in np.flatnonzero(valid[row]):
            for axis in range(2):
                scaled[row,j,axis]=(float(errors[row,j,axis])-float(anchor[row,axis]))/float(scale[axis])
                target[row,j,axis]=np.sign(scaled[row,j,axis])*np.log1p(abs(scaled[row,j,axis]))
    assert np.isfinite(target).all() and not target[~valid].any()
    return scaled,target


def input_only_projection(old, added, rcond):
    """Independent pivoted-QR + small-R SVD projection, no target arguments.

    Only old and added feature matrices are arguments. This is a descriptive
    column-space calculation, not a learned pose scorer or target regression.
    A direct least-squares coefficient matrix could amplify roundoff near the
    numerical rank cutoff; projecting with orthonormal columns avoids this.
    """
    old, added = np.asarray(old, np.float64), np.asarray(added, np.float64)
    assert old.ndim == added.ndim == 2 and len(old) == len(added)
    assert np.isfinite(old).all() and np.isfinite(added).all() and rcond > 0
    q,r,_ = linalg.qr(old, mode='economic', pivoting=True)
    ur,singular,_ = linalg.svd(r,full_matrices=False,lapack_driver='gesvd')
    rank=int(np.sum(singular>singular[0]*rcond)) if len(singular) else 0
    column_basis=q@ur[:,:rank]
    residual = added - column_basis@(column_basis.T@added)
    return dict(rank=int(rank), singular_values=singular, residual=residual,
                residual_frobenius=float(np.linalg.norm(residual)),
                added_frobenius=float(np.linalg.norm(added)),
                residual_column_norm=np.linalg.norm(residual, axis=0),
                orthogonality_max_abs=float(np.max(np.abs(old.T @ residual))))


def load_and_verify_directions(C,protocol):
    """Independent input binding/member loader and every frozen projection."""
    feature_protocol=C.protocol('AUDIT_PROTOCOL')
    assert protocol['feature_protocol']==C.bind(C.DOC/'AUDIT_PROTOCOL.json')
    assert protocol['codes']==feature_protocol['codes']
    for key in ('frames','models','scorer_models','representation','parity','normalization','budgets'):
        assert protocol[key]==feature_protocol[key]
    assert protocol['frames']==2598 and protocol['models']==list(MODELS)
    assert protocol['direction_dim']==18
    assert protocol['parity']==dict(atol_px=1e-4,rtol=1e-6)
    assert protocol['representation']['novelty_frobenius_max']==1e-4
    assert protocol['inputs']['direction_receipt']==C.bind(C.DOC/'FEATURE_AUDIT.json')
    receipt=C.read(C.DOC/'FEATURE_AUDIT.json')
    assert receipt['complete'] and receipt['PASS'] and receipt['protocol']==protocol['feature_protocol']
    assert receipt['source_TRAIN_only'] and not receipt['source_labels_read']
    assert not receipt['VAL_quality_read'] and not receipt['real_targets_read']
    for key in ('fits','image_forwards','PnP_calls','new_pose_estimates','argmin_calls','selected_policy_changes'):
        assert receipt[key]==0
    assert receipt['directions']==protocol['inputs']['direction_features']==C.bind(C.RAW/'TRAIN_DIRECTIONS.npz')
    inputs=feature_protocol['inputs']
    assert receipt['bindings']['inputs']==inputs and receipt['bindings']['codes']==feature_protocol['codes']
    for new,old in (('source_contract','source_contract'),('source_features','features'),('source_feature_lock','feature_lock')):
        assert protocol['inputs'][new]==inputs[old]
    lock=C.read(C.ROOT/inputs['feature_lock']['path'])
    assert lock['complete'] and not lock['source_targets_read'] and not lock['real_targets_read']
    for key,new in (('features','features'),('poses','poses'),('metadata','metadata'),('predictions','source_predictions_lock')):
        assert lock[key]==inputs[new]
    verify_bindings(C,lock['operators'])
    contract=C.read(C.ROOT/inputs['source_contract']['path'])
    assert contract['complete'] and contract['status']=='PASS'
    eligible=set(contract['fit_eligibility']['eligible_ids']['TRAIN']);assert len(eligible)==2598
    assert not eligible.intersection(contract['fit_eligibility']['eligible_ids']['VAL'])
    rows=C.read(C.ROOT/inputs['metadata']['path'])
    assert len(rows)==5120 and len({r['id'] for r in rows})==5120
    with np.load(C.ROOT/inputs['features']['path'],allow_pickle=False) as container:
        assert container['ids'].tolist()==[r['id'] for r in rows]
        source_index=np.array([i for i,r in enumerate(rows) if r['id'] in eligible],np.int64)
        ids=container['ids'][source_index].copy()
        assert len(ids)==2598 and (container['split'][source_index]=='TRAIN').all()
        assert container['hypothesis_names'].tolist()==list(HYP)
        raw={m:container[m+'_geo'][source_index].copy() for m in MODELS}
        valid={m:container[m+'_valid'][source_index].copy() for m in MODELS}
        index=container['R0_GEO_index'][source_index].copy()
    rows=[rows[i] for i in source_index]
    assert all(r['split']=='TRAIN' for r in rows)
    with np.load(C.RAW/'TRAIN_DIRECTIONS.npz',allow_pickle=False) as container:
        assert set(container.files)=={'ids','source_index','anchor_index'}|{m+s for m in MODELS for s in ('_direction18','_valid')}
        np.testing.assert_array_equal(container['ids'],ids)
        np.testing.assert_array_equal(container['source_index'],source_index)
        np.testing.assert_array_equal(container['anchor_index'],index)
        direction={m:container[m+'_direction18'].copy() for m in MODELS}
        for m in MODELS:
            np.testing.assert_array_equal(container[m+'_valid'],valid[m])
    for key,value in (('ids_sha',ids),('source_index_sha',source_index),('anchor_index_sha',index)):
        assert receipt[key]==array_sha(value)
    assert ids[index<0].tolist()==['TEX__shard_04_f0110']
    prediction_lock=C.read(C.ROOT/inputs['source_predictions_lock']['path'])
    assert prediction_lock['complete'] and prediction_lock['models']==list(MODELS) and prediction_lock['frames']==5120
    assert prediction_lock['protocol']==lock['protocol'] and prediction_lock['metadata']==inputs['metadata']
    C.verify(prediction_lock['protocol'])
    if prediction_lock.get('runtime_amendment'):
        C.verify(prediction_lock['runtime_amendment'])
    source_protocol=C.read(C.ROOT/prediction_lock['protocol']['path'])
    assert source_protocol['input']==inputs['metadata']
    assert source_protocol['source_contract']==inputs['source_contract']
    poses=C.read(C.ROOT/inputs['poses']['path'])
    assert poses['models']==list(MODELS) and poses['hypothesis_names']==list(HYP)
    assert len(poses['ids'])==5120 and not poses['source_targets_read'] and not poses['real_targets_read']
    pxcols=[lock['feature_names'].index(f'residual{k}_px') for k in range(9)]
    normcols=[lock['feature_names'].index(f'residual{k}_bboxnorm') for k in range(9)]
    verified={}
    for model in MODELS:
        assert raw[model].shape==(2598,2,94) and raw[model].dtype==np.float32
        assert valid[model].shape==(2598,2) and valid[model].dtype==bool
        assert direction[model].shape==(2598,2,18) and direction[model].dtype==np.float32
        assert not direction[model][~valid[model]].any()
        assert int(valid[model].sum())==5194 and (~valid[model].any(1)).sum()==1
        old=receipt['models'][model]
        for key,value in (('direction_sha',direction[model]),('valid_sha',valid[model]),('raw94_sha',raw[model])):
            assert old[key]==array_sha(value)
        b=prediction_lock['receipts'][model];C.verify(b)
        pred_receipt=C.read(C.ROOT/b['path'])
        assert pred_receipt['complete'] and pred_receipt['model']==model and pred_receipt['frames']==5120
        assert pred_receipt['protocol']==prediction_lock['protocol']
        assert pred_receipt['checkpoint']==source_protocol['checkpoints'][model]
        assert len(pred_receipt['files'])==5120
        selected=[pred_receipt['files'][int(i)] for i in source_index]
        for i,b in zip(source_index,selected):
            path=C.PARENT_RAW/'source_predictions'/model/f'{int(i):05d}.json'
            assert b['path']==str(path.relative_to(C.ROOT));ALLOWED_PREDICTIONS.add(str(path.resolve()))
        largest=dict(direction18=0.,raw94_px=0.,raw94_bboxnorm=0.,frozen_corner8_px=0.)
        checked=0
        for n,(row,b) in enumerate(zip(rows,selected)):
            C.verify(b);saved=C.read(C.ROOT/b['path'])
            assert saved['id']==row['id'] and saved['model']==model
            assert saved['protocol_sha']==prediction_lock['protocol']['sha256']
            assert saved['checkpoint_sha']==source_protocol['checkpoints'][model]['sha256']
            prediction=saved['prediction'];rec=poses['records'][model][row['id']]
            hypotheses={h['name']:h for h in rec['hypotheses']}
            assert len(hypotheses)==len(rec['hypotheses']) and set(hypotheses)<=set(HYP)
            if not valid[model][n].any():
                continue
            chosen=prediction['selected_index']
            assert chosen is not None and 0<=chosen<len(prediction['candidates'])
            candidate=prediction['candidates'][chosen]
            for j,name in enumerate(HYP):
                if not valid[model][n,j]:
                    continue
                h=hypotheses[name];assert h['pose']['available']
                out=scalar_projection(h['pose'],row['K'],candidate['keypoints_xy'],candidate['box_xyxy'])
                np.testing.assert_allclose(out['direction18'],direction[model][n,j],rtol=1e-6,atol=1e-7)
                largest['direction18']=max(largest['direction18'],float(np.max(np.abs(out['direction18'].astype(np.float64)-direction[model][n,j]))))
                comparisons=(('raw94_px',out['norm9'],raw[model][n,j,pxcols],1.),
                    ('raw94_bboxnorm',out['norm9']/out['diagonal'],raw[model][n,j,normcols],out['diagonal']),
                    ('frozen_corner8_px',out['norm9'][:8],h['inference_cues']['corner8_residual_px'],1.))
                for key,new,oldvalue,divisor in comparisons:
                    difference=np.abs(np.asarray(new)-np.asarray(oldvalue))
                    assert np.all(difference<=1e-4/divisor+1e-6*np.abs(oldvalue)),(model,n,j,key)
                    largest[key]=max(largest[key],float(difference.max()))
                checked+=1
        assert checked==5194
        verified[model]=dict(PASS=True,frames=2598,valid_candidates=checked,invalid_rows=1,
            projected_points=checked*9,independent_max_abs_difference=largest,
            direction_numeric_tolerance=dict(atol=1e-7,rtol=1e-6))
    mean,std=independent_normalization(direction['R0'],valid['R0'])
    np.testing.assert_array_equal(mean,np.asarray(receipt['normalization']['mean18'],np.float32))
    np.testing.assert_array_equal(std,np.asarray(receipt['normalization']['std18'],np.float32))
    assert receipt['normalization']['array_sha']==array_sha(np.stack([mean,std]))
    return dict(ids=ids,source_index=source_index,anchor_index=index,features=raw,valid=valid,
        direction=direction,mean18=mean,std18=std,feature_receipt=receipt,projection=verified,
        feature_protocol=feature_protocol)


def selfcheck():
    pose = dict(R_cf=np.eye(3).tolist(), centroid=[.15, -.2, 4.], cf_extents=[1.3, .11, 1.1])
    camera = np.array([[600., 2., 320.], [0., 610., 240.], [0., 0., 1.]])
    xyz = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],[0,0,0.]])*np.array(pose['cf_extents'])/2.
    cam = xyz+np.array(pose['centroid'])
    homogeneous = cam @ camera.T
    projected = homogeneous[:,:2]/homogeneous[:,2:]
    known_residual = np.arange(18,dtype=np.float64).reshape(9,2)/8.-1.
    got = scalar_projection(pose, camera, projected-known_residual, [10.,20.,110.,220.])
    np.testing.assert_allclose(got['projected'], projected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(got['residual'], known_residual, atol=1e-12, rtol=0)
    np.testing.assert_allclose(got['direction18'], (known_residual/math.sqrt(50000.)).reshape(18).astype(np.float32), atol=1e-9, rtol=0)
    # Two observations can have the same residual magnitudes and opposite signs.
    other = scalar_projection(pose, camera, projected+known_residual, [10.,20.,110.,220.])
    np.testing.assert_allclose(other['norm9'], got['norm9'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(other['direction18'], -got['direction18'], atol=1e-9, rtol=0)
    rng = np.random.default_rng(20261001)
    raw = rng.normal(size=(4,2,18)).astype(np.float32)
    valid = np.array([[1,1],[1,1],[0,0],[1,1]],bool)
    raw[~valid]=0;anchor=np.array([0,1,-1,0],np.int64)
    mean,std=independent_normalization(raw,valid)
    delta=independent_direction_difference(raw,valid,anchor,mean,std)
    assert not delta[np.flatnonzero(valid.any(1)),anchor[valid.any(1)]].any()
    assert not delta[~valid].any()
    inputs=np.array([[0.,0.],[0.,0.],[1.,2.],[1.,2.],[2.,1.]])
    targets=np.array([[0.,0.],[0.,0.],[1.,-1.],[-1.,1.],[0.,1.]])
    collision=independent_collisions(inputs,targets)
    assert collision==dict(rows=5,unique_input_rows=3,duplicate_groups=2,rows_in_duplicate_groups=4,
        conflicting_target_groups=1,rows_in_conflicting_target_groups=2,
        conflicting_sign_groups=1,rows_in_conflicting_sign_groups=2)
    matrix=rng.normal(size=(40,5));matrix[:,4]=matrix[:,0]+matrix[:,1]
    added=matrix@rng.normal(size=(5,3));added[:,2]+=rng.normal(size=40)
    rcond=np.finfo(np.float64).eps*max(matrix.shape)
    checked=input_only_projection(matrix,added,rcond)
    u,s,_=np.linalg.svd(matrix,full_matrices=False);rank=int((s>s[0]*rcond).sum())
    expected=added-u[:,:rank]@(u[:,:rank].T@added)
    assert checked['rank']==rank==4
    np.testing.assert_allclose(checked['residual'],expected,rtol=1e-10,atol=1e-11)
    assert checked['residual_column_norm'][0]<1e-10 and checked['residual_column_norm'][2]>1.
    return dict(PASS=True, invented_fixtures_only=True, scalar_projection_with_skew=True,
        signed_residual_direction_not_recoverable_from_norm_fixture=True,
        original_FP32_normalization_and_zero_anchor=True, invalid_row_retained=True,
        exact_input_collision_and_numeric_target_conflict=True,
        independent_pivoted_QR_small_SVD_vs_direct_SVD_projection=True, targets_not_projection_arguments=True,
        new_model_fits=0, candidate_selections=0, artifact_reads=0)


def verify_span(record,x,extra):
    assert record['rows']==len(x) and record['labels_used'] is False
    assert not record['matrix_centering'] and not record['intercept_added'] and not record['frame_or_target_weights']
    assert record['threshold']==1e-4
    rcond=max(x.shape)*np.finfo(np.float64).eps
    checked=input_only_projection(x,extra,rcond)
    assert checked['rank']==record['old']['rank']
    ranks={}
    largest_spectrum_difference=0.
    for key,matrix in (('old',x),('extra',extra),('extended',np.concatenate([x,extra],axis=1))):
        singular=linalg.svd(matrix,full_matrices=False,compute_uv=False,lapack_driver='gesvd')
        tolerance=float(max(matrix.shape)*np.finfo(np.float64).eps*singular[0])
        rank=int(np.sum(singular>tolerance));ranks[key]=rank
        saved=record[key]
        assert saved['shape']==list(matrix.shape) and saved['rank']==rank
        np.testing.assert_allclose(saved['tolerance'],tolerance,rtol=1e-9,atol=1e-12)
        allowance=1e-9*max(1.,float(singular[0]))
        np.testing.assert_allclose(saved['singular_values'],singular,rtol=1e-7,atol=allowance)
        largest_spectrum_difference=max(largest_spectrum_difference,float(np.abs(singular-np.asarray(saved['singular_values'])).max()))
    assert record['rank_gain']==ranks['extended']-ranks['old']
    scale=max(1.,float(np.max(np.abs(extra))))
    # Direct-SVD crosscheck tests the actual residual matrix independently of
    # both the producer's implementation and its stored summary fields.
    u,s,_=linalg.svd(x,full_matrices=False,lapack_driver='gesvd')
    direct=extra-u[:,:ranks['old']]@(u[:,:ranks['old']].T@extra)
    residual_gap=float(np.max(np.abs(checked['residual']-direct)))
    assert residual_gap<=1e-7*scale
    norm=float(np.linalg.norm(extra));resnorm=float(np.linalg.norm(checked['residual']))
    colnorm=np.linalg.norm(extra,axis=0);rescol=np.linalg.norm(checked['residual'],axis=0)
    ratio=resnorm/norm if norm else 0.
    ratios=np.divide(rescol,colnorm,out=np.zeros_like(rescol),where=colnorm>0)
    # QR and SVD must agree in scaled absolute and relative input quantities;
    # no tolerance is used to change the predeclared novelty classification.
    for key,value in (('extra_frobenius_norm',norm),('residual_frobenius_norm',resnorm),
                      ('extra_column_norm',colnorm),('residual_column_norm',rescol)):
        np.testing.assert_allclose(record[key],value,rtol=1e-7,atol=1e-7*scale)
    np.testing.assert_allclose(record['frobenius_residual_ratio'],ratio,rtol=1e-7,atol=1e-7)
    np.testing.assert_allclose(record['per_column_residual_ratio'],ratios,rtol=1e-7,atol=1e-7)
    assert bool(record['extra_information_below_fixed_ratio'])==(ratio<=1e-4)
    return dict(PASS=True,rows=len(x),ranks=ranks,frobenius_residual_ratio=ratio,
        extra_information_below_fixed_ratio=ratio<=1e-4,
        independent_method='pivoted_QR_plus_small_R_GESVD_and_direct_GESVD_crosscheck',
        residual_matrix_QR_vs_direct_SVD_max_abs=residual_gap,
        all_spectra_max_abs_difference=largest_spectrum_difference,
        labels_used=False,coefficient_matrix_exported=False,
        numeric_verification_tolerance=dict(rtol=1e-7,scaled_atol=1e-7),
        novelty_threshold_unchanged=1e-4)


def verify_recovery(C,result,protocol):
    """Bind the explicit operational recovery; never claim original-run success."""
    recovery=result['finalization_recovery']
    for key in ('plan','incident','preserved_partial_array'):
        C.verify(recovery[key])
    C.verify(result['execution_code'])
    assert recovery['plan']==C.bind(C.DOC/'RECOVERY_PLAN.json')
    assert recovery['incident']==C.bind(C.DOC/'OUTPUT_BINDING_INCIDENT.json')
    plan=C.read(C.DOC/'RECOVERY_PLAN.json');incident=C.read(C.DOC/'OUTPUT_BINDING_INCIDENT.json')
    assert plan['complete'] and incident['complete']
    verify_bindings(C,plan);verify_bindings(C,incident)
    assert plan['protocol']==incident['protocol']==result['protocol']==C.bind(C.DOC/'REPRESENTATION_PROTOCOL.json')
    assert plan['original_producer']==incident['original_producer']==result['code']
    assert plan['original_producer'] in protocol['codes']
    assert plan['recovery_code']==result['execution_code']==C.bind(C.HERE/'finalize_representation.py')
    assert plan['partial_array']==incident['partial_array']==recovery['preserved_partial_array']==result['arrays']
    assert plan['incident']==recovery['incident']
    assert plan['expected_array_keys']==25 and len(recovery['array_keys_exactly_reconstructed'])==25
    assert incident['exit_code']==1 and incident['per_model_audits_completed']==4
    assert incident['representation_json_absent'] and incident['representation_md_absent']
    C.verify(incident['first_incident'])
    first=C.read(C.ROOT/incident['first_incident']['path'])
    assert first['complete'] and first['exit_code']==1 and first['per_model_audits_completed']==0
    assert first['protocol']==result['protocol'] and first['original_producer']==result['code']
    assert all(first['absence_at_failure'].values())
    assert first['selector_fits']==first['new_policy_argmin']==plan['new_fits']==plan['new_policy_argmin']==0
    assert plan['source_VAL_quality_reads']==plan['real_evaluations']==0
    assert recovery['original_producer_run_completed'] is False
    assert recovery['separate_guard_used'] and recovery['separate_guard_array_write_allowed'] is False
    assert recovery['same_sealed_math_recomputed'] and recovery['diagnostic_attempt']==3
    return dict(PASS=True,plan=recovery['plan'],incident=recovery['incident'],
        first_incident=incident['first_incident'],execution_code=result['execution_code'],
        preserved_partial_array=recovery['preserved_partial_array'],
        original_producer_run_completed=False,diagnostic_attempt=3,
        original_model_loop_counts_by_attempt=[0,4,4],
        separate_guard_array_write_allowed=False,expected_array_keys=25,
        expected_key_hashes=recovery['array_keys_exactly_reconstructed'],array_schema=incident['array_schema'])


def main():
    from . import common as C
    install_guard(C)
    started=time.monotonic()
    assert not (C.DOC/'VERIFICATION.json').exists() and not (C.DOC/'VERIFICATION_KO.md').exists()
    protocol=C.protocol('REPRESENTATION_PROTOCOL')
    result=C.read(C.DOC/'REPRESENTATION_AUDIT.json')
    assert result['complete'] and result['PASS'] and result['diagnostic_only']
    assert result['protocol']==C.bind(C.DOC/'REPRESENTATION_PROTOCOL.json')
    assert result['inputs']==protocol['inputs'] and result['representation']==protocol['representation']
    assert result['source_TRAIN_only'] and not result['VAL_quality_read'] and not result['real_targets_read']
    for key in ('raw_source_reference_reads','new_reference_metric_calculations','optimizer_steps','new_fits',
                'actual_data_weight_probes','actual_data_objective_probes','new_policy_argmin','image_forwards','new_PnP_calls'):
        assert result[key]==0
    assert not result['prior_weights_read'] and not result['method_success'] and not result['goal_complete']
    C.verify(result['code']);assert result['code'] in protocol['codes']
    assert result['arrays']==C.bind(C.RAW/'REPRESENTATION_ARRAYS.npz')
    recovery=verify_recovery(C,result,protocol)
    data=load_and_verify_directions(C,protocol)
    ids,index=data['ids'],data['anchor_index']
    parent=C.read(C.ROOT/protocol['inputs']['parent_train_protocol']['path'])
    assert protocol['inputs']['parent_train_protocol']==C.bind(C.PARENT_DOC/'TRAIN_PROTOCOL.json')
    for new,old in (('source_contract','source_contract'),('source_features','features'),
                    ('source_train_labels','train_labels'),('source_feature_lock','feature_lock')):
        assert protocol['inputs'][new]==parent['inputs'][old]
    prefit=C.read(C.ROOT/protocol['inputs']['sign_prefit']['path'])
    assert protocol['inputs']['sign_prefit']==C.bind(C.SIGN_DOC/'PREFIT_REVIEW.json')
    assert prefit['complete'] and prefit['PASS']
    assert prefit['inputs']['features']==protocol['inputs']['source_features']
    assert prefit['inputs']['train_labels']==protocol['inputs']['source_train_labels']
    basis_artifact=C.read(C.ROOT/protocol['inputs']['rbf_basis']['path'])
    assert basis_artifact['complete'] and basis_artifact['PASS']
    assert protocol['inputs']['rbf_basis']==prefit['basis_SHA_bind']==C.bind(C.RBF_DOC/'RBF_BASIS.json')
    with np.load(C.ROOT/protocol['inputs']['source_train_labels']['path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],ids)
        np.testing.assert_array_equal(z['source_index'],data['source_index'])
        np.testing.assert_array_equal(np.flatnonzero(z['eligible_train_mask']),data['source_index'])
        assert z['hypothesis_names'].tolist()==list(HYP)
        errors={m:np.stack([z[m+'_T_cm'],z[m+'_R_deg']],axis=-1) for m in MODELS}
        anchor=np.stack([z['R0_GEO_T_cm'],z['R0_GEO_R_deg']],axis=-1)
        scale=np.asarray([z['sT_cm'].item(),z['sR_deg'].item()],np.float64)
    assert np.isfinite(scale).all() and (scale>0).all()
    values=data['features']['R0'][data['valid']['R0']]
    mean,std=values.mean(0),np.maximum(values.std(0),np.float32(1e-6))
    assert array_sha(np.stack([mean,std]))==prefit['inputs']['normalization_sha']==basis_artifact['normalization_sha']==result['normalization94_sha']
    for key,value in (('source_ids_sha',ids),('source_index_sha',data['source_index']),
                      ('anchor_index_sha',index),('anchor_errors_sha',anchor),('scale_sha',scale)):
        assert prefit['inputs'][key]==array_sha(value)
    with np.load(C.RAW/'REPRESENTATION_ARRAYS.npz',allow_pickle=False) as z:
        expected={'ids','source_index','anchor_index','mean18','std18'}|{m+'_'+k for m in C.MODEL_NAMES
            for k in ('extra18','axis_target','valid','old_group_ids','extended_group_ids')}
        assert set(z.files)==expected
        saved={k:z[k].copy() for k in z.files}
    assert set(saved)==set(recovery['expected_key_hashes'])==set(recovery['array_schema'])
    for key,value in saved.items():
        assert array_sha(value)==recovery['expected_key_hashes'][key],key
        assert recovery['array_schema'][key]==dict(dtype=str(value.dtype),shape=list(value.shape)),key
    for key,value in (('ids',ids),('source_index',data['source_index']),('anchor_index',index),
                      ('mean18',data['mean18']),('std18',data['std18'])):
        np.testing.assert_array_equal(saved[key],value)
    assert result['source_ids_sha']==array_sha(ids) and result['source_index_sha']==array_sha(data['source_index'])
    assert result['anchor_index_sha']==array_sha(index)
    assert result['frames']==2598 and result['available_anchor_rows']==2597 and result['failed_rows_retained']==1
    assert result['normalization18']['sha']==array_sha(np.stack([data['mean18'],data['std18']]))
    np.testing.assert_array_equal(np.asarray(result['normalization18']['mean'],np.float32),data['mean18'])
    np.testing.assert_array_equal(np.asarray(result['normalization18']['std'],np.float32),data['std18'])
    checked_models={}
    for model in C.MODEL_NAMES:
        experts=['R0']+([] if model=='R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
        names=[f'{m}:{h}' for m in experts for h in HYP]
        raw=np.concatenate([data['features'][m] for m in experts],axis=1)
        valid=np.concatenate([data['valid'][m] for m in experts],axis=1)
        error=np.concatenate([errors[m] for m in experts],axis=1)
        raw18=np.concatenate([data['direction'][m] for m in experts],axis=1)
        assert np.isfinite(error[valid]).all() and (error[valid]>=0).all() and np.isposinf(error[~valid]).all()
        phi,x=independent_old_features(raw,valid,index,mean,std,basis_artifact['basis'])
        with np.errstate(all='raise'):
            scaled,target=independent_targets(error,valid,anchor,index,scale)
        extra=independent_direction_difference(raw18,valid,index,data['mean18'],data['std18'])
        hashes=dict(signed_target_sha=array_sha(target),input_difference_sha=array_sha(x),
            base_context_sha=array_sha(phi),errors_sha=array_sha(error),scaled_excess_sha=array_sha(scaled),
            original_valid_sha=array_sha(valid))
        record=result['models'][model]
        assert record['names']==names and record['forced_anchor_zero'] and record['all_invalid_rows_retained']==1
        assert set(record['prior_six_hashes'])==set(HASH_KEYS)
        for key in HASH_KEYS:
            assert hashes[key]==record['prior_six_hashes'][key]==prefit['models'][model][key],(model,key)
        np.testing.assert_array_equal(saved[model+'_axis_target'],target)
        np.testing.assert_array_equal(saved[model+'_valid'],valid)
        np.testing.assert_array_equal(saved[model+'_extra18'],extra)
        assert record['extra18_sha']==array_sha(extra)
        assert record['extended271_sha']==array_sha(np.concatenate([x,extra],axis=2))
        collisions=verify_collision_record(record['collisions'],x,extra,target,valid,index,ids,names,
            saved[model+'_old_group_ids'],saved[model+'_extended_group_ids'])
        nonanchor=valid.copy();rows=np.flatnonzero(valid.any(1));nonanchor[rows,index[rows]]=False
        span=verify_span(record['input_span'],x[nonanchor],extra[nonanchor])
        checked_models[model]=dict(PASS=True,prior_six_hashes=hashes,
            extra18_sha=array_sha(extra),extended271_sha=record['extended271_sha'],
            collisions=collisions,input_span=span)
        print('INDEPENDENT_DIRECTION_REPRESENTATION_VERIFIED',model,flush=True)
    insufficient=all(v['input_span']['extra_information_below_fixed_ratio'] for v in checked_models.values())
    assert result['novelty']['all_models_ratio_le1e_4']==insufficient
    expected_status='INSUFFICIENT_ADDITIONAL_INPUT_EVIDENCE' if insufficient else 'INPUT_NONREDUNDANCY_ONLY_NOT_PERFORMANCE_EVIDENCE'
    assert result['novelty']['status']==expected_status
    out=dict(complete=True,PASS=True,created_at=C.now(),diagnostic_only=True,
        protocol=C.bind(C.DOC/'REPRESENTATION_PROTOCOL.json'),feature_protocol=protocol['feature_protocol'],
        audited_feature_receipt=C.bind(C.DOC/'FEATURE_AUDIT.json'),
        audited_representation=C.bind(C.DOC/'REPRESENTATION_AUDIT.json'),
        code=C.bind(__file__),projection=data['projection'],models=checked_models,recovery=recovery,
        frames=2598,available_anchor_rows=2597,invalid_rows_retained=1,
        source_TRAIN_only=True,source_feature_containers_include_VAL_inputs=True,
        input_only_projection=True,source_labels_used_only_for_hash_and_collision_checks=True,
        old_six_hashes_per_model_verified=True,all_projection_points=4*5194*9,
        novelty=dict(all_models_ratio_le1e_4=insufficient,status=expected_status,performance_evidence=False),
        raw_reference_reads=0,VAL_quality_reads=0,real_reference_reads=0,new_model_fits=0,
        model_weight_reads=0,new_objective_evaluations=0,new_policy_argmin=0,new_PnP_calls=0,
        image_forwards=0,method_success=False,goal_complete=False,
        numerical_scope='Independent scalar pose projection and prior norms; independent FP32 normalization/253 reconstruction/targets/hashes; exact collision membership, all target ranges and subgroup conflicts; pivoted QR plus small-R GESVD and direct GESVD input-span verification. No label enters either projection helper.',
        bindings=[C.bind(C.DOC/'AUDIT_PROTOCOL.json'),C.bind(C.DOC/'REPRESENTATION_PROTOCOL.json'),
            C.bind(C.DOC/'FEATURE_AUDIT.json'),C.bind(C.DOC/'REPRESENTATION_AUDIT.json'),result['arrays'],
            recovery['plan'],recovery['incident'],recovery['first_incident'],recovery['execution_code'],
            *protocol['inputs'].values(),*protocol['codes'],C.bind(__file__)],
        read_paths=sorted(set(READS)),wall_seconds=time.monotonic()-started)
    lines=['# 잔차 방향 입력 감사 독립 검산','',
        '**검산 PASS. 입력 진단이며 학습·T/R 개선 성과가 아니다.** TRAIN2598행과 실패1행을 유지했다. 새 fit·weight 적용·후보 선택·PnP·VAL 품질·실사 참조 조회는 모두0이다.','',
        '생산자의 원래 실행은 두 번 중단됐다. 첫 시도는 threadpool 초기화에서 모델 루프 전에 멈췄고, 두 번째는4모델 계산과 NPZ 저장 뒤 자신의 출력을 읽어 hash를 만드는 guard에서 멈췄다. 별도 공개 finalizer가 같은 봉인 수학을 재계산하고 기존 NPZ25개 key의 dtype·shape·bytes 및 파일 SHA 불변을 검증한 세 번째 진단 시도에서 완료했다. 원래 guard가 성공했다고 기록하지 않으며, 독립 검산도 이 복구 체인과25개 해시를 확인했다.','',
        '원래 고정 R_cf/t/치수/K를 성분별 투영식으로 재구성하여4개 expert의 유효20,776후보,186,984점을 확인했다. 저장된 signed18과 기존94의 잔차9·corner8 캐시를 사전 허용오차에서 대조했다. 정규화18은 R0 TRAIN 유효5,194후보만 사용하며 기존253과타깃6해시는 모델마다 정확히 일치한다.','',
        '입력/확장 입력의 ±0만 통일한 byte 그룹, 모든 반복 그룹의 T/R 범위·부호 충돌·분할·남은 충돌·anchor 구분을 독립 재구성했다. 열공간 투영은 label 없이 pivoted QR와 작은 R의 GESVD를 사용해 생산자의 직접 SVD와 교차했다.','',
        '| 모델 | 기존 rank | 확장 rank | 독립 잔차 비율 | 1e−4 이하 |','|---|---:|---:|---:|---|']
    for model,v in checked_models.items():
        s=v['input_span'];lines.append(f"| {model} | {s['ranks']['old']} | {s['ranks']['extended']} | {s['frobenius_residual_ratio']:.9g} | {s['extra_information_below_fixed_ratio']} |")
    lines+=['','추가 입력의 비중복성은 정답 정보·성능 향상·도메인 전이의 증거가 아니다. 같은 입력의 타깃 충돌이 없더라도 표현 충분성을 입증하지 않는다. 이번 감사는 후속 학습을 승인하지 않으며 원래 source45/실사5 조건을 변경하지 않는다.','',
        '기존 source 컨테이너에는 VAL 입력도 있지만 투영·진단은 사전 적격TRAIN2598행만 사용했다. target은 동결 TRAIN 전용 캐시에서 해시와 충돌 설명에만 사용했으며 새 물리 오류를 계산하지 않았다.','',
        '[독립 검산 JSON](VERIFICATION.json) · [방향 특징 감사](FEATURE_AUDIT.json) · [표현 감사](REPRESENTATION_AUDIT_KO.md) · [고정 프로토콜](REPRESENTATION_PROTOCOL.json) · [운영 중단·복구](RUNTIME_INITIALIZATION_KO.md) · [복구 계획](RECOVERY_PLAN.json)','']
    C.save(C.DOC/'VERIFICATION_KO.md','\n'.join(lines))
    C.save(C.DOC/'VERIFICATION.json',out)
    return dict(PASS=True,complete=True,models=list(checked_models),frames=2598,
        novelty_status=expected_status,new_fits=0,new_policy_argmin=0,
        wall_seconds=out['wall_seconds'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selfcheck',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(selfcheck() if args.selfcheck else main(),ensure_ascii=False),flush=True)
