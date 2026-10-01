"""Independent locked-route appearance verification, before any quality read.

No producer/evaluator imports, backbone inference, fitting, pose solve or new
candidate selection. Only stored tokens, images, frozen poses and input metadata.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from . import common as C
from scripts.research.pallet_pose_dino_native_inputs_20261001_v1 import verify_inputs as N
from scripts.research.pallet_pose_dino_input_audit_20261001_v1 import verify_train_inputs as I

MODELS=I.MODELS
HYP=I.HYP
LEARNED=('R0_ONLY','UNION_s1','UNION_s2','UNION_s3')
ATOL=RTOL=2e-6
RULE='NATIVE_UNPADDED_SUPPORTED_PROJECTED_CORNERS8_DINO384_MEAN_PLUS_SUPPORT_FRACTION'
SUPPORT_RULE='PREVIOUS_SUPPORT_AND_PAD_LE_U_LT_WIDTH_MINUS_PAD_AND_PAD_LE_V_LT_HEIGHT_MINUS_PAD'
READS=set()
bind,read,array_sha=I.bind,I.read,I.array_sha


def verify(binding):
    """Bindings may retain logical symlink paths and omit the byte-size field.

    Open exactly the bound path and require its content SHA. The read guard
    independently resolves paths for its allowlist; no sampler rule changes.
    """
    path=C.ROOT/binding['path']
    assert I.sha(path)==binding['sha256'],binding['path']
    if 'bytes' in binding:
        assert path.stat().st_size==binding['bytes'],binding['path']


def guard(allowed,scope):
    allowed={Path(p).resolve() for p in allowed}
    outputs={C.DOC/f'{scope}_APPEARANCE_VERIFICATION.json',C.DOC/f'{scope}_APPEARANCE_VERIFICATION_KO.md'}
    denied=('SOURCE_TRAIN_LABELS','SYNTH_LABELS','SYNTH_RECORDS','GEOMETRY_SIDETABLE',
        'GEOMETRY_RESOLVED_POSE_GT','SOURCE_VAL_METRICS','REAL_RESULTS','REAL_FRAME_RESULTS',
        'POSE_METRICS','TRUTH_FOR_DISPLAY','AXIS_REVIEW_MANIFEST','/fits/','/model_parameters/')
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(C.ROOT):
            return
        mode,flags=args[1:3]
        writing=(isinstance(mode,str) and any(k in mode for k in 'wax+')) or (
            isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        assert not any(k in str(path) for k in denied),('APPEARANCE_NO_QUALITY_OR_WEIGHT',str(path))
        assert path.suffix.lower() not in ('.pt','.pth','.onnx'),('APPEARANCE_NO_MODEL',str(path))
        if writing:
            assert path in outputs,('APPEARANCE_WRITE_SCOPE',str(path))
        else:
            assert path in allowed or path in outputs or path.suffix in ('.py','.pyc'),('APPEARANCE_READ_SCOPE',str(path))
            if path.suffix not in ('.py','.pyc'):
                READS.add(str(path.relative_to(C.ROOT)))
    sys.addaudithook(hook)


def padding(scope,row):
    if scope=='SOURCE_VAL':
        assert row['split']=='VAL' and row['pad']==100
        return 100
    assert scope=='REAL' and 'pad' not in row
    assert all(k in row for k in ('id','image','hw','K','xyz','recording','severity'))
    return 0


def normalized_difference(raw,valid,anchor,mean,std):
    """Independent scalar FP32 normalization, then FP64 R0-anchor subtraction."""
    out=np.zeros(raw.shape,np.float64)
    normalized=np.zeros(raw.shape,np.float64)
    for i,j in zip(*np.nonzero(valid)):
        normalized[i,j]=((raw[i,j]-mean)/std).astype(np.float64)
    for i,j in zip(*np.nonzero(valid)):
        assert anchor[i] in (0,1) and valid[i,anchor[i]]
        out[i,j]=normalized[i,j]-normalized[i,anchor[i]]
    assert np.isfinite(out).all() and not out[~valid].any()
    rows=np.flatnonzero(valid.any(1))
    assert not out[rows,anchor[rows]].any()
    return out


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def fixed_normalization(train,producer):
    keys=('visual_receipt','visual_protocol','visual_verification')
    bindings={k:train['inputs'][k] for k in keys}
    receipt=read(C.ROOT/bindings['visual_receipt']['path'])
    protocol=read(C.ROOT/bindings['visual_protocol']['path'])
    checked=read(C.ROOT/bindings['visual_verification']['path'])
    assert receipt['complete'] and receipt['input_construction_pass'] and receipt['source_TRAIN_only']
    assert checked['complete'] and checked['PASS'] and checked['verification_PASS'] and checked['normalization_exact']
    assert checked['source_TRAIN_only'] and checked['public_input_receipt']==bindings['visual_receipt']
    assert receipt['protocol']==checked['protocol']==bindings['visual_protocol']
    assert receipt['descriptors']==checked['descriptors']==train['inputs']['visual_features']
    assert protocol['descriptor_rule']==receipt['descriptor_rule']==RULE
    assert protocol['support_rule']==receipt['support_rule']==SUPPORT_RULE
    assert producer['backbone']==protocol['frozen_parent']['backbone']
    assert producer['precision']==dict(matmul_allow_tf32=False,cudnn_allow_tf32=False,cudnn_benchmark=False,
        inference_dtype='float32',token_dtype='float16')
    assert protocol['frozen_parent']['tokens']['shape']==[384,56,42]
    norm=receipt['normalization']
    assert norm['count']==5194 and norm['dtype']=='float32' and norm['std_floor']==1e-6
    assert norm['valid_zero_support_candidates_included']
    mean,std=np.asarray(norm['mean385'],np.float32),np.asarray(norm['std385'],np.float32)
    assert mean.shape==std.shape==(385,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std>=np.float32(1e-6)).all()
    digest=array_sha(np.stack([mean,std]))
    assert digest==norm['array_sha']==checked['normalization']['array_sha']
    fields=dict(visual_normalization_sha=digest,visual_receipt_binding=bindings['visual_receipt'],
        visual_protocol_binding=bindings['visual_protocol'],visual_verification_binding=bindings['visual_verification'])
    for key,value in fields.items():
        assert train[key]==producer[key]==value
    np.testing.assert_array_equal(np.asarray(train['visual_mean'],np.float32),mean)
    np.testing.assert_array_equal(np.asarray(train['visual_std'],np.float32),std)
    assert not producer['normalization_recomputed']
    return mean,std,fields


def masks_and_anchor(scope,stored,rows,indices,poses):
    masks={}
    if scope=='SOURCE_VAL':
        assert stored['hypothesis_names'].tolist()==list(HYP)
        for m in MODELS:
            masks[m]=stored[m+'_valid'][indices].copy()
        anchor=stored['R0_GEO_index'][indices].copy()
    else:
        assert stored['ids'].tolist()==[r['id'] for r in rows]
        for m in MODELS:
            frame=stored[m+'_valid']
            assert frame.shape==(len(rows),) and frame.dtype==bool
            mask=np.zeros((len(rows),2),bool)
            for i,row in enumerate(rows):
                hs={h['name']:h['pose'] for h in poses[m][row['id']]['hypotheses']}
                for k,name in enumerate(HYP):
                    mask[i,k]=bool(frame[i] and name in hs and hs[name]['available'])
            masks[m]=mask
        anchor=np.full(len(rows),-1,np.int64)
        present=np.logical_or.reduce([m.any(1) for m in masks.values()])
        for i in np.flatnonzero(present):
            name=poses['R0'][rows[i]['id']]['GEO_name'];assert name in HYP
            anchor[i]=HYP.index(name)
    present=np.logical_or.reduce([m.any(1) for m in masks.values()])
    assert np.array_equal(anchor>=0,present) and np.isin(anchor,[-1,0,1]).all()
    for i,row in enumerate(rows):
        r0=poses['R0'][row['id']]
        if anchor[i]<0:
            assert not r0['GEO_pose']['available']
        else:
            name=HYP[int(anchor[i])]
            assert masks['R0'][i,anchor[i]] and r0['GEO_name']==name
            hs=[h['pose'] for h in r0['hypotheses'] if h['name']==name]
            assert len(hs)==1 and hs[0]==r0['GEO_pose'] and hs[0]['available']
    return masks,anchor


def run(scope):
    torch.set_num_threads(1)
    assert scope in ('SOURCE_VAL','REAL') and I.ATOL==I.RTOL==N.ATOL==N.RTOL==ATOL==RTOL
    started=time.monotonic();n=1024 if scope=='SOURCE_VAL' else 173
    lock_path=C.DOC/f'{scope}_ROUTING_LOCK.json'
    protocol_path=C.DOC/('TRAIN_PROTOCOL.json' if scope=='SOURCE_VAL' else 'REAL_PROTOCOL.json')
    train_path=C.DOC/'TRAIN_PROTOCOL.json';receipt_path=C.DOC/f'{scope}_APPEARANCE_INPUTS.json'
    # Explicit bootstrap only; no evaluator import or recursive reference traversal.
    lock,p,train,producer=read(lock_path),read(protocol_path),read(train_path),read(receipt_path)
    assert lock['complete'] and lock['frames']==n and lock['models']==list(LEARNED)
    assert lock['protocol']==bind(protocol_path)==producer['protocol']
    assert lock['visual_inputs']==bind(receipt_path) and lock['visual_features']==producer['descriptors']
    assert producer['complete'] and producer['PASS'] and producer['scope']==scope and producer['frames']==n
    assert producer['rule']==RULE and producer['support_rule']==SUPPORT_RULE
    assert producer['source_label_values_read'] is False and producer['real_reference_values_read'] is False
    assert producer['fits_executed']==producer['new_PnP_solves']==0 and producer['additional_padding']==0
    assert producer['original_candidate_validity_preserved'] and not producer['normalization_recomputed']
    inputs=producer['inputs']
    assert producer['training_verification']==inputs['training_verification']
    assert inputs['training_verification']==bind(C.DOC/'TRAIN_CONVERGENCE.json')
    assert inputs['training_complete']==lock['training_complete']==bind(C.DOC/'TRAINING_COMPLETE.json')
    train_checked=read(C.DOC/'TRAIN_CONVERGENCE.json');complete=read(C.DOC/'TRAINING_COMPLETE.json')
    assert train_checked['complete'] and train_checked['PASS'] and train_checked['protocol']==bind(train_path)
    assert set(train_checked['models'])==set(LEARNED)
    assert complete['complete'] and complete['all_certified'] and complete['fit_count']==4 and complete['protocol']==bind(train_path)
    source=read(C.ROOT/inputs['metadata']['path'])
    if scope=='SOURCE_VAL':
        assert len(source)==5120 and len({r['id'] for r in source})==5120
        indices=np.array([i for i,r in enumerate(source) if r['split']=='VAL'],np.int64)
        rows=[source[i] for i in indices]
        assert lock['source_VAL_label_values_read'] is False and not lock['VAL_quality_scored'] and not lock['real_references_read']
        predlock=read(C.ROOT/inputs['predictions_lock']['path'])
        assert predlock['complete'] and predlock['frames']==5120
        pred_receipt_binding=predlock['receipts']['R0']
        pred_receipt=read(C.ROOT/pred_receipt_binding['path'])
        assert pred_receipt['complete'] and pred_receipt['protocol']==predlock['protocol']
        prediction_bindings=[pred_receipt['files'][int(i)] for i in indices]
        assert lock['source_contract']==train['inputs']['source_contract']
        extra_bindings=[pred_receipt_binding,lock['source_contract']]
        gate_binding=None
    else:
        rows=source;indices=np.arange(len(rows),dtype=np.int64)
        assert p['complete'] and p['frames']==173 and p['parent_protocol']==bind(train_path)
        assert p['source_PASS_required'] and lock['source_val_gate']==p['source_val_gate']
        assert not lock['real_reference_values_read']
        gate_binding=p['source_val_gate'] # Hash provenance only: do not parse source quality values.
        predlock=read(C.ROOT/inputs['prediction_lock']['path'])
        assert predlock['complete'] and not predlock['inference_reference_coordinates_read']
        assert inputs['prediction_R0']==predlock['predictions']['R0']
        saved=read(C.ROOT/inputs['prediction_R0']['path']);real_predictions=saved.get('predictions',saved)
        assert set(real_predictions)=={r['id'] for r in rows}
        prediction_bindings=[inputs['prediction_R0']]*len(rows)
        extra_bindings=[gate_binding]
    assert len(rows)==n and len({r['id'] for r in rows})==n
    assert producer['tokens']['path']==str((C.RAW/f'{scope}_TOKENS.npy').relative_to(C.ROOT))
    assert producer['descriptors']['path']==str((C.RAW/f'{scope}_APPEARANCE.npz').relative_to(C.ROOT))
    native_bindings=[train['inputs'][k] for k in ('visual_receipt','visual_protocol','visual_verification')]
    artifact_bindings=[*inputs.values(),*native_bindings,*extra_bindings,producer['tokens'],producer['descriptors'],
        lock['choices'],*prediction_bindings,*[r['image'] for r in rows],*producer['operators']]
    allowed=[lock_path,protocol_path,train_path,receipt_path,C.DOC/'TRAIN_PROTOCOL_SHA.json',
        C.DOC/'REAL_PROTOCOL_SHA.json',C.DOC/'TRAIN_CONVERGENCE.json',C.DOC/'TRAINING_COMPLETE.json',
        Path(__file__),Path(I.__file__),Path(N.__file__),*[C.ROOT/b['path'] for b in artifact_bindings]]
    guard(allowed,scope)
    seen={}
    for b in artifact_bindings:
        previous=seen.setdefault(b['path'],b)
        assert previous==b
    for b in seen.values():verify(b)
    verify(read(C.DOC/'TRAIN_PROTOCOL_SHA.json'))
    if scope=='REAL':verify(read(C.DOC/'REAL_PROTOCOL_SHA.json'))
    mean,std,fields=fixed_normalization(train,producer)
    for key,value in fields.items():assert lock[key]==value
    raw_poses=read(C.ROOT/inputs['poses']['path'])
    poses=raw_poses['records'] if scope=='SOURCE_VAL' else raw_poses
    if scope=='SOURCE_VAL':
        contract=read(C.ROOT/lock['source_contract']['path'])
        assert contract['complete'] and contract['status']=='PASS'
        assert {r['id'] for r in rows}==set(contract['fit_eligibility']['eligible_ids']['VAL'])
        assert {r['id'] for r in rows}.isdisjoint(contract['fit_eligibility']['eligible_ids']['TRAIN'])
        assert raw_poses['ids']==[r['id'] for r in source]
        assert not raw_poses['source_targets_read'] and not raw_poses['real_targets_read']
        feature_lock=read(C.ROOT/inputs['feature_lock']['path'])
        assert feature_lock['complete'] and not feature_lock['source_targets_read'] and not feature_lock['real_targets_read']
        for key in ('features','poses','metadata'):assert feature_lock[key]==inputs[key]
        assert feature_lock['predictions']==inputs['predictions_lock']
    else:
        pose_lock=read(C.ROOT/inputs['pose_lock']['path'])
        assert not pose_lock['references_read'] and pose_lock['prediction_lock']==inputs['prediction_lock']
        assert inputs['poses'] in pose_lock['files']
        assert predlock['metadata']==inputs['metadata']
        for m in MODELS:
            assert all(not poses[m][row['id']]['reference_coordinates_read'] for row in rows)
    stored=load_npz(C.ROOT/inputs['features']['path'])
    if scope=='SOURCE_VAL':assert stored['ids'].tolist()==[r['id'] for r in source]
    masks,anchor=masks_and_anchor(scope,stored,rows,indices,poses)
    arrays=load_npz(C.ROOT/producer['descriptors']['path'])
    expected={'ids','source_index','anchor_index','crop_matrices','padding_px','native_rectangles_xyxy'}|{
        m+s for m in MODELS for s in ('_appearance385','_valid','_support8')}
    assert set(arrays)==expected and arrays['ids'].tolist()==[r['id'] for r in rows]
    np.testing.assert_array_equal(arrays['source_index'],indices);np.testing.assert_array_equal(arrays['anchor_index'],anchor)
    assert arrays['source_index'].dtype==arrays['anchor_index'].dtype==arrays['padding_px'].dtype==np.int64
    assert lock['anchor_index_sha']==array_sha(anchor)
    matrices=arrays['crop_matrices'];assert matrices.shape==(n,3,3) and matrices.dtype==np.float64 and np.isfinite(matrices).all()
    present=anchor>=0
    assert not matrices[~present].any() and producer['allinvalid_rows']==int((~present).sum())
    choices=read(C.ROOT/lock['choices']['path'])
    assert choices['ids']==[r['id'] for r in rows]
    assert choices['visual_inputs']==lock['visual_inputs'] and choices['visual_features']==lock['visual_features']
    for m in MODELS:
        x,valid,support=(arrays[m+s] for s in ('_appearance385','_valid','_support8'))
        assert x.shape==(n,2,385) and x.dtype==np.float32 and np.isfinite(x).all()
        assert valid.shape==(n,2) and valid.dtype==bool and support.shape==(n,2,8) and support.dtype==bool
        np.testing.assert_array_equal(valid,masks[m]);assert not x[~valid].any() and not support[~valid].any()
    for m in LEARNED:
        combined=masks['R0'] if m=='R0_ONLY' else np.concatenate([masks['R0'],masks[f'DIVERSE251_s{m[-1]}']],axis=1)
        for i,row in enumerate(rows):
            choice=choices['records'][m][row['id']]
            np.testing.assert_array_equal(choice['candidate_valid' if scope=='SOURCE_VAL' else 'valid'],combined[i])
            assert choice['anchor_index']==int(anchor[i])
    tokens=np.load(C.ROOT/producer['tokens']['path'],mmap_mode='r',allow_pickle=False)
    assert isinstance(tokens,np.memmap) and tokens.dtype==np.float16 and tokens.shape==(n,384,56,42)
    assert len(producer['frame_receipts'])==n
    maxima={m:dict(max_absolute=0.,max_tolerance_fraction=0.,checked_candidates=0) for m in MODELS}
    token_hashes=crop_hashes=0
    for i,(row,frame,pb) in enumerate(zip(rows,producer['frame_receipts'],prediction_bindings)):
        assert frame['id']==row['id'] and frame['image']==row['image'] and frame['source_index']==int(indices[i]) and frame['prediction']==pb
        pad=padding(scope,row)
        rectangle,_=N.native_intersection(np.zeros((8,2)),np.zeros(8,bool),row['hw'],pad)
        assert frame['padding_px']==arrays['padding_px'][i]==pad
        np.testing.assert_array_equal(rectangle,arrays['native_rectangles_xyxy'][i])
        np.testing.assert_array_equal(rectangle,frame['native_rectangle_xyxy'])
        image=cv2.imdecode(np.frombuffer((C.ROOT/row['image']['path']).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        assert image is not None and list(image.shape[:2])==row['hw']
        token=tokens[i];assert np.isfinite(token).all()
        assert frame['image_forward']==bool(present[i])
        if scope=='SOURCE_VAL':
            saved=read(C.ROOT/pb['path'])
            assert saved['id']==row['id'] and saved['model']=='R0'
            assert saved['protocol_sha']==predlock['protocol']['sha256'] and saved['checkpoint_sha']==pred_receipt['checkpoint']['sha256']
            prediction=saved['prediction']
        else:prediction=real_predictions[row['id']]
        if not present[i]:
            assert not token.any()
            for m in MODELS:assert not arrays[m+'_appearance385'][i].any() and not arrays[m+'_support8'][i].any()
            continue
        assert array_sha(token)==frame['token_sha'];token_hashes+=1
        selected=prediction['selected_index'];assert selected is not None
        matrix=I.crop_matrix(prediction['candidates'][selected]['box_xyxy'])
        np.testing.assert_array_equal(matrix,matrices[i]);np.testing.assert_array_equal(matrix,frame['matrix'])
        # Independently reconstruct the existing OpenCV image transform; no new padding.
        crop=cv2.warpAffine(image,matrix[:2],(576,768),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        rgb=(crop[:,:,::-1].astype(np.float32)-np.array([123.68,116.78,103.94],np.float32)).transpose(2,0,1)
        assert array_sha(rgb)==frame['crop_rgb_sha'];crop_hashes+=1
        crop_points=[];native_masks=[]
        for m in MODELS:
            hypotheses={h['name']:h['pose'] for h in poses[m][row['id']]['hypotheses']}
            for k,name in enumerate(HYP):
                if not masks[m][i,k]:
                    cp,native=np.zeros((8,2),np.float64),np.zeros(8,bool)
                else:
                    assert name in hypotheses and hypotheses[name]['available']
                    uv,cp,prepared=I.project(hypotheses[name],row['K'],matrix,row['hw'])
                    _,native=N.native_intersection(uv,prepared,row['hw'],pad)
                np.testing.assert_array_equal(native,arrays[m+'_support8'][i,k])
                crop_points.append(cp);native_masks.append(native)
        samples=I.sample_double(torch.from_numpy(token.astype(np.float64))[None],np.concatenate(crop_points),np.concatenate(native_masks))
        for j,m in enumerate(MODELS):
            for k in range(2):
                slot=j*2+k;expected=I.ordered_mean_double(samples[slot*8:(slot+1)*8],native_masks[slot])
                actual=arrays[m+'_appearance385'][i,k];assert actual[-1]==expected[-1]
                error,fraction=I.compare(actual,expected)
                maxima[m]['max_absolute']=max(maxima[m]['max_absolute'],error)
                maxima[m]['max_tolerance_fraction']=max(maxima[m]['max_tolerance_fraction'],fraction)
                maxima[m]['checked_candidates']+=int(masks[m][i,k])
        if (i+1)%100==0 or i==n-1:print('VERIFY_LOCKED_APPEARANCE',scope,i+1,'/',n,flush=True)
    assert token_hashes==crop_hashes==producer['image_forwards']==int(present.sum())
    models={}
    for m in MODELS:
        x,valid,support=(arrays[m+s] for s in ('_appearance385','_valid','_support8'))
        hist={str(k):int((support.sum(2)[valid]==k).sum()) for k in range(9)}
        summary=producer['models'][m]
        assert summary['valid_candidates']==int(valid.sum())==maxima[m]['checked_candidates']
        for key,a in (('descriptor_sha',x),('valid_sha',valid),('support_sha',support)):
            assert summary[key]==array_sha(a)
        assert summary['supported_corner_histogram']==hist
        np.testing.assert_array_equal(x[:,:,-1],support.sum(2).astype(np.float32)/8)
        models[m]=dict(comparison=maxima[m],supported_corner_histogram=hist,valid_zero_support_candidates=int((valid&~support.any(2)).sum()))
    difference_hashes={}
    for model in LEARNED:
        parents=['R0']+([] if model=='R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
        raw=np.concatenate([arrays[m+'_appearance385'] for m in parents],axis=1)
        valid=np.concatenate([masks[m] for m in parents],axis=1)
        difference_hashes[model]=array_sha(normalized_difference(raw,valid,anchor,mean,std))
    result=dict(complete=True,PASS=True,scope=scope,frames=n,protocol=bind(protocol_path),routing_lock=bind(lock_path),
        input_receipt=bind(receipt_path),descriptors=producer['descriptors'],tokens=producer['tokens'],
        code=bind(Path(__file__)),independent_operators=[bind(Path(I.__file__)),bind(Path(N.__file__))],
        training_protocol=bind(train_path),training_verification=inputs['training_verification'],
        training_complete=inputs['training_complete'],source_gate_binding_hash_only=gate_binding,
        **fields,models=models,array_hashes={k:array_sha(v) for k,v in arrays.items()},
        normalized_visual_difference_sha_by_model=difference_hashes,
        comparison=dict(atol=ATOL,rtol=RTOL,fixed_before_actual=True,reference='Scalar projection/native intersection and Torch CPUfloat64 grid_sample ordered mean'),
        normalization_recomputed=False,fixed_TRAIN_normalization_exact=True,
        image_hashes_checked=n,image_dimensions_checked=n,crop_hashes_checked=crop_hashes,
        token_frame_hashes_checked=token_hashes,full_token_file_sha_checked=True,allinvalid_rows=int((~present).sum()),
        source_label_values_read=False,real_reference_values_read=False,source_quality_values_read=False,
        real_quality_values_read=False,new_image_forwards=0,new_fits=0,new_PnP_solves=0,new_policy_selections=0,
        backbone_forward_recomputed=False,backbone_inference_independently_verified=False,
        input_only=True,performance_improvement_measured=False,method_success=False,goal_complete=False,
        limitation='Checks pixels/crop and retained-token-to-descriptor transformation, not an independent backbone forward or T/R quality.',
        read_paths=sorted(READS),elapsed_seconds=time.monotonic()-started)
    C.save(C.DOC/f'{scope}_APPEARANCE_VERIFICATION.json',result)
    C.save(C.DOC/f'{scope}_APPEARANCE_VERIFICATION_KO.md',
        f'# {scope} RGB 입력 독립 검산\n\n입력 검산 PASS. 고정 {n}행의 이미지 SHA·치수·R0 crop·token·후보 지원점·385차원 특징을 확인했습니다. '+
        '정규화는 고정 TRAIN 값을 그대로 사용했습니다. source/실사 정답과 성능 값은 읽지 않았고 새 추론·학습·pose 계산·선택은 0회입니다. '+
        'backbone 자체를 재추론한 검증이나 T/R 개선 판정은 아닙니다.\n\n'+
        f'[전체 영수증]({scope}_APPEARANCE_VERIFICATION.json)\n')
    print('LOCKED_APPEARANCE_VERIFICATION_PASS',scope,bind(C.DOC/f'{scope}_APPEARANCE_VERIFICATION.json'),flush=True)
    return result


def selfcheck():
    torch.set_num_threads(1)
    import hashlib
    import tempfile
    with tempfile.TemporaryDirectory(prefix='visual_binding_fixture_') as folder:
        target=Path(folder)/'invented.dat';target.write_bytes(b'invented image binding fixture')
        logical=Path(folder)/'logical.dat';logical.symlink_to(target)
        binding=dict(path=str(logical),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        verify(binding);verify(dict(binding,bytes=target.stat().st_size))
        for bad in (dict(binding,sha256='0'*64),dict(binding,bytes=target.stat().st_size+1)):
            try:verify(bad)
            except AssertionError:pass
            else:raise AssertionError('Changed synthetic content or size accepted')
    assert I.selfcheck()['PASS']
    assert padding('SOURCE_VAL',dict(split='VAL',pad=100))==100
    assert padding('REAL',dict.fromkeys(('id','image','hw','K','xyz','recording','severity')))==0
    uv=np.array([[100.,100.],[475.999,667.999],[99.99,100.],[476.,100.],
        [200.,668.],[200.,99.99],[0.,0.],[575.,767.]])
    prepared=np.ones(8,bool)
    _,native=N.native_intersection(uv,prepared,[768,576],100)
    np.testing.assert_array_equal(native,[1,1,0,0,0,0,0,0])
    _,full=N.native_intersection(uv,prepared,[768,576],0)
    np.testing.assert_array_equal(full,prepared)
    rng=np.random.default_rng(20261001656385)
    raw=rng.normal(size=(3,4,385)).astype(np.float32);valid=np.array([[1,1,1,1],[0,1,1,0],[0,0,0,0]],bool)
    raw[~valid]=0.;anchor=np.array([0,1,-1]);mean=rng.normal(size=385).astype(np.float32);std=(rng.random(385)+.2).astype(np.float32)
    x=normalized_difference(raw,valid,anchor,mean,std)
    for i,j in zip(*np.nonzero(valid)):
        expected=np.array([float(np.float32((raw[i,j,c]-mean[c])/std[c]))-float(np.float32((raw[i,anchor[i],c]-mean[c])/std[c])) for c in range(385)])
        np.testing.assert_array_equal(x[i,j],expected)
    assert not x[~valid].any() and not x[0,0].any() and not x[1,1].any()
    image=rng.integers(0,256,size=(160,120,3),dtype=np.uint8)
    matrix=I.crop_matrix([10.,20.,100.,140.])
    crop=cv2.warpAffine(image,matrix[:2],(576,768),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    assert crop.shape==(768,576,3)
    return dict(PASS=True,invented_only=True,native_source_pad100=True,real_explicit_pad0=True,
        independent_projection_sampler=True,fixed_TRAIN_normalization_difference_exact=True,
        empty_rows_preserved=True,synthetic_symlink_optional_bytes_checked=True,
        actual_inputs_read=0,actual_images_read=0,model_forwards=0,fits=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['selfcheck','verify'])
    parser.add_argument('--scope',choices=['SOURCE_VAL','REAL'])
    args=parser.parse_args()
    if args.action=='selfcheck':
        assert args.scope is None
        print(json.dumps(selfcheck(),ensure_ascii=False,indent=2))
    else:
        assert args.scope is not None
        run(args.scope)
