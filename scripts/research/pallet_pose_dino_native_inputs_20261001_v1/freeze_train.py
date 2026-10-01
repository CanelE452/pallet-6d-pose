"""Seal and re-pool frozen TRAIN tokens using original-image support only.

No GPU, backbone, RGB loading, target reading, selection or optimization.
Output construction checks are not an independent verification of this code.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from . import visual_features as V

ROOT=Path(__file__).resolve().parents[3]
NAME='pallet_pose_dino_native_inputs_20261001_v1'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
PARENT_NAME='pallet_pose_dino_input_audit_20261001_v1'
PARENT_DOC=ROOT/'_docs/experiments'/PARENT_NAME
PARENT_RAW=ROOT/'data/pallet/results'/PARENT_NAME
MODELS=('R0','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3')
HYP=('long-face-front','short-face-front')
READS=set()
WRITES=set()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)


def verify(binding):
    assert bind(ROOT/binding['path'])==binding,binding['path']


def read(path):
    return json.loads(Path(path).read_text())


def array_sha(value):
    a=np.ascontiguousarray(value)
    h=hashlib.sha256(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def install_guard(allowed):
    allowed={Path(x).resolve() for x in allowed}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(ROOT):
            return
        mode,flags=args[1],args[2]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(
            isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        assert not any(x in str(path) for x in ('SOURCE_TRAIN_LABELS','SYNTH_LABELS','SYNTH_RECORDS',
            'GEOMETRY_RESOLVED_POSE_GT','SOURCE_VAL_','TRAIN_CONVERGENCE','REAL_','POSE_METRICS',
            '/fits/','/model_parameters/')),('FORBIDDEN_INPUT',str(path))
        assert path.suffix.lower() not in ('.jpg','.jpeg','.png','.webp','.pt','.pth'),('NO_IMAGE_OR_WEIGHTS',str(path))
        if writing:
            assert path.is_relative_to(RAW) or path.is_relative_to(DOC),('WRITE_DENIED',str(path))
            WRITES.add(str(path.relative_to(ROOT)))
            return
        if path.suffix in ('.py','.pyc'):
            return
        assert path in allowed or path.is_relative_to(RAW) or path.is_relative_to(DOC),('READ_DENIED',str(path))
        READS.add(str(path.relative_to(ROOT)))
    sys.addaudithook(hook)


def seal():
    previous=read(PARENT_DOC/'INPUT_PROTOCOL.json')
    receipt=read(PARENT_RAW/'TRAIN_APPEARANCE_INPUTS.json')
    checked=read(PARENT_DOC/'INPUT_VERIFICATION.json')
    padding=read(PARENT_DOC/'PADDING_SUPPORT_AUDIT.json')
    assert previous['train_only'] and previous['frames']==2598 and previous['output_dim']==385
    assert receipt['complete'] and receipt['input_construction_pass'] and receipt['source_TRAIN_only']
    assert checked['complete'] and checked['PASS'] and checked['verification_PASS']
    assert checked['protocol']==receipt['protocol']==bind(PARENT_DOC/'INPUT_PROTOCOL.json')
    assert checked['input_receipt']==bind(PARENT_RAW/'TRAIN_APPEARANCE_INPUTS.json')
    assert checked['descriptors']==padding['descriptors']==receipt['descriptors'] and checked['tokens']==receipt['tokens']
    assert padding['complete'] and padding['PASS'] and padding['protocol']==receipt['protocol']
    inputs=dict(previous_protocol=bind(PARENT_DOC/'INPUT_PROTOCOL.json'),
        previous_receipt=bind(PARENT_RAW/'TRAIN_APPEARANCE_INPUTS.json'),
        previous_descriptors=receipt['descriptors'],tokens=receipt['tokens'],
        previous_verification=bind(PARENT_DOC/'INPUT_VERIFICATION.json'),
        padding_audit=bind(PARENT_DOC/'PADDING_SUPPORT_AUDIT.json'),
        **previous['inputs'])
    codes=[bind(Path(__file__)),bind(Path(V.__file__)),bind(Path(V.PREVIOUS.__file__))]
    for b in [*inputs.values(),*codes]:
        verify(b)
    result=dict(schema='pallet_pose_dino_native_inputs_v1',created_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,codes=codes,design=bind(DOC/'DESIGN_KO.md'),frames=2598,models=list(MODELS),
        hypothesis_names=list(HYP),train_only=True,output_dim=385,descriptor_rule=V.RULE,support_rule=V.SUPPORT_RULE,
        native_support=dict(padding_source="existing SOURCE_INPUTS row['pad']; explicit integer, no default",
            rectangle='[pad,width-pad) x [pad,height-pad)',combined_with='all previous geometric support conditions',
            source_coordinate_system='already prepared image coordinates; no translation or new crop',
            zero_support_descriptor='all 385 components zero; preserve original candidate validity'),
        frozen_parent=dict(crop=previous['crop'],tokens=previous['tokens'],backbone=previous['backbone']),
        descriptor='Same raw384 tokens, half-pixel bilinear border samples, per-channel sorted FP32 supported mean, support_count/8',
        normalization=dict(source='R0_original_valid_TRAIN_candidates',count=5194,dtype='float32',std_floor=1e-6,
            valid_zero_support_candidates_included=True),
        original_candidate_validity_preserved=True,original_allinvalid_rows=1,
        existing_tokens_reused=True,old_inputs_immutable=True,full_tokens_duplicated=False,
        no_GPU=True,no_new_forward=True,no_fit=True,no_label_values=True,no_new_pose=True,
        no_VAL_features=True,no_real_features=True,
        caveat='Token/context computation still used the unchanged crop, including reflected padding; restricting sample locations does not remove contextual padding influence.',
        independent_verification_required_before_fit=True,stable_joint_improvement_achieved=False)
    save(DOC/'INPUT_PROTOCOL.json',result)
    print('NATIVE_INPUT_PROTOCOL_SEALED',bind(DOC/'INPUT_PROTOCOL.json'),flush=True)


def freeze():
    started=time.monotonic()
    protocol=read(DOC/'INPUT_PROTOCOL.json')
    assert protocol['train_only'] and protocol['frames']==2598 and protocol['output_dim']==385
    assert protocol['support_rule']==V.SUPPORT_RULE and protocol['descriptor_rule']==V.RULE
    assert [bind(Path(__file__)),bind(Path(V.__file__)),bind(Path(V.PREVIOUS.__file__))]==protocol['codes']
    allowed=[DOC/'INPUT_PROTOCOL.json',*[ROOT/b['path'] for b in protocol['inputs'].values()]]
    install_guard(allowed)
    for b in [*protocol['inputs'].values(),*protocol['codes'],protocol['design']]:
        verify(b)
    parent=read(ROOT/protocol['inputs']['previous_receipt']['path'])
    parent_protocol=read(ROOT/protocol['inputs']['previous_protocol']['path'])
    verification=read(ROOT/protocol['inputs']['previous_verification']['path'])
    padding_audit=read(ROOT/protocol['inputs']['padding_audit']['path'])
    assert parent['complete'] and verification['PASS'] and verification['input_receipt']==protocol['inputs']['previous_receipt']
    assert parent['protocol']==protocol['inputs']['previous_protocol']
    assert parent['tokens']==protocol['inputs']['tokens'] and parent['descriptors']==protocol['inputs']['previous_descriptors']
    for key,binding in parent_protocol['inputs'].items():
        assert protocol['inputs'][key]==binding
    source=read(ROOT/protocol['inputs']['metadata']['path'])
    contract=read(ROOT/protocol['inputs']['source_contract']['path'])
    poses=read(ROOT/protocol['inputs']['poses']['path'])
    eligible=set(contract['fit_eligibility']['eligible_ids']['TRAIN'])
    indices=np.array([i for i,r in enumerate(source) if r['id'] in eligible],np.int64)
    rows=[source[i] for i in indices]
    assert len(rows)==len(eligible)==2598 and all(r['split']=='TRAIN' for r in rows)
    assert poses['ids']==[r['id'] for r in source]
    with np.load(ROOT/protocol['inputs']['previous_descriptors']['path'],allow_pickle=False) as z:
        previous={k:z[k].copy() for k in z.files}
    expected_keys={'ids','source_index','anchor_index','crop_matrices','mean385','std385'}|{
        m+s for m in MODELS for s in ('_appearance385','_valid','_support8')}
    assert set(previous)==expected_keys and previous['ids'].tolist()==[r['id'] for r in rows]
    np.testing.assert_array_equal(previous['source_index'],indices)
    with np.load(ROOT/protocol['inputs']['features']['path'],allow_pickle=False) as z:
        assert z['ids'].tolist()==[r['id'] for r in source]
        assert tuple(z['hypothesis_names'].tolist())==HYP
        np.testing.assert_array_equal(previous['anchor_index'],z['R0_GEO_index'][indices])
        for m in MODELS:
            np.testing.assert_array_equal(previous[m+'_valid'],z[m+'_valid'][indices])
    pads=np.array([r['pad'] for r in rows],np.int64)
    rectangles=np.stack([V.native_rectangle(r['hw'],r['pad']) for r in rows])
    assert all(int(r['pad'])==r['pad'] for r in rows)
    tokens=np.load(ROOT/protocol['inputs']['tokens']['path'],mmap_mode='r',allow_pickle=False)
    assert isinstance(tokens,np.memmap) and tokens.shape==(2598,384,56,42) and tokens.dtype==np.float16
    arrays={k:previous[k].copy() for k in ('ids','source_index','anchor_index','crop_matrices')}
    values={m:np.zeros((2598,2,385),np.float32) for m in MODELS}
    support={m:np.zeros((2598,2,8),bool) for m in MODELS}
    valid={m:previous[m+'_valid'] for m in MODELS}
    for m in MODELS:
        assert valid[m].dtype==bool and valid[m].shape==(2598,2) and int(valid[m].sum())==5194
        np.testing.assert_array_equal(valid[m].any(1),valid['R0'].any(1))
    allinvalid=~valid['R0'].any(1)
    assert int(allinvalid.sum())==1 and not previous['crop_matrices'][allinvalid].any()
    frames=[]
    verified_token_hashes=0
    for i,(row,old_frame) in enumerate(zip(rows,parent['frame_receipts'])):
        assert old_frame['id']==row['id'] and old_frame['source_index']==int(indices[i])
        assert old_frame['image']==row['image']
        token=tokens[i]
        assert np.isfinite(token).all()
        if allinvalid[i]:
            assert not token.any() and not old_frame['image_forward']
        else:
            assert array_sha(token)==old_frame['token_sha']
            verified_token_hashes+=1
            matrix=previous['crop_matrices'][i]
            np.testing.assert_array_equal(matrix,old_frame['matrix'])
            for m in MODELS:
                hypotheses={h['name']:h for h in poses['records'][m][row['id']]['hypotheses']}
                assert set(hypotheses)==set(HYP)
                for k,name in enumerate(HYP):
                    if not valid[m][i,k]:
                        continue
                    pose=hypotheses[name]['pose']
                    assert pose['available']
                    result=V.describe(pose,row['K'],matrix,row['hw'],token,row['pad'])
                    np.testing.assert_array_equal(result['prepared_support8'],previous[m+'_support8'][i,k])
                    values[m][i,k]=result['descriptor']
                    support[m][i,k]=result['support8']
        frame=dict(old_frame)
        frame.update(image_forward=False,token_reused=not bool(allinvalid[i]),
            parent_image_forward=old_frame['image_forward'],padding_px=int(pads[i]),
            native_rectangle_xyxy=rectangles[i].tolist())
        frames.append(frame)
        if (i+1)%200==0 or i==len(rows)-1:
            print('NATIVE_REPOOL_TRAIN',i+1,'/',len(rows),'new_forwards',0,'seconds',round(time.monotonic()-started,1),flush=True)
    assert verified_token_hashes==2597
    mean=values['R0'][valid['R0']].mean(axis=0,dtype=np.float32)
    std=np.maximum(values['R0'][valid['R0']].std(axis=0,dtype=np.float32),np.float32(1e-6))
    arrays.update(mean385=mean,std385=std)
    details=dict(ids=arrays['ids'],source_index=indices,anchor_index=arrays['anchor_index'],
                 padding_px=pads,native_rectangles_xyxy=rectangles)
    models={}
    for m in MODELS:
        assert np.isfinite(values[m]).all() and not values[m][~valid[m]].any() and not support[m][~valid[m]].any()
        np.testing.assert_array_equal(values[m][:,:,-1],support[m].sum(2).astype(np.float32)/8)
        old_support=previous[m+'_support8']
        assert not (support[m]&~old_support).any()
        removed=old_support&~support[m]
        counts=support[m].sum(2)[valid[m]]
        old_audit=padding_audit['models'][m]
        assert int(removed.sum())==old_audit['padding_supported_points']
        assert int(old_support.sum())==old_audit['supported_points']
        assert int(removed.any(2).sum())==old_audit['candidates_with_padding_support']
        arrays.update({m+'_appearance385':values[m],m+'_valid':valid[m],m+'_support8':support[m]})
        details.update({m+'_prepared_support8':old_support,m+'_native_support8':support[m],m+'_removed_support8':removed})
        models[m]=dict(valid_candidates=int(valid[m].sum()),descriptor_sha=array_sha(values[m]),
            valid_sha=array_sha(valid[m]),support_sha=array_sha(support[m]),
            supported_corner_histogram={str(k):int((counts==k).sum()) for k in range(9)},
            previous_supported_points=int(old_support.sum()),native_supported_points=int(support[m].sum()),
            removed_prepared_padding_points=int(removed.sum()),candidates_with_removed_support=int(removed.any(2).sum()),
            frames_with_removed_support=int(removed.any((1,2)).sum()),
            valid_zero_native_support_candidates=int((counts==0).sum()),
            valid_descriptor_changed_candidates=int(np.any(values[m]!=previous[m+'_appearance385'],axis=2)[valid[m]].sum()),
            original_valid_unchanged=True)
    assert set(arrays)==expected_keys
    RAW.mkdir(parents=True,exist_ok=True)
    output_path=RAW/'TRAIN_APPEARANCE.npz'
    details_path=RAW/'NATIVE_SUPPORT_DETAILS.npz'
    for path,data in ((output_path,arrays),(details_path,details)):
        with path.open('xb') as f:
            np.savez_compressed(f,**data)
    result=dict(complete=True,input_construction_pass=True,source_TRAIN_only=True,
        independent_verification_status='PENDING',independent_verification_required_before_fit=True,
        protocol=bind(DOC/'INPUT_PROTOCOL.json'),tokens=protocol['inputs']['tokens'],descriptors=bind(output_path),
        native_support_details=bind(details_path),parent_protocol=protocol['inputs']['previous_protocol'],
        parent_receipt=protocol['inputs']['previous_receipt'],parent_descriptors=protocol['inputs']['previous_descriptors'],
        parent_verification=protocol['inputs']['previous_verification'],padding_audit=protocol['inputs']['padding_audit'],
        frames=2598,image_forwards=0,parent_image_forwards=parent['image_forwards'],original_allinvalid_rows=1,
        original_candidate_validity_preserved=True,fits_executed=0,optimizer_steps=0,policy_selections=0,
        source_label_values_read=False,source_VAL_features_extracted=False,real_features_extracted=False,
        new_PnP_solves=0,new_image_forwards=0,GPU_used=False,image_files_read=0,weights_read=0,
        existing_tokens_reused=True,token_hashes_checked=verified_token_hashes,full_tokens_duplicated=False,
        descriptor_rule=V.RULE,support_rule=V.SUPPORT_RULE,
        native_padding=dict(metadata=protocol['inputs']['metadata'],values=sorted(map(int,np.unique(pads))),
            ids_sha=array_sha(arrays['ids']),source_index_sha=array_sha(indices),padding_sha=array_sha(pads),
            rectangles_sha=array_sha(rectangles),rectangle_definition='[pad,width-pad) x [pad,height-pad)'),
        normalization=dict(source='R0_original_valid_TRAIN_candidates',count=5194,dtype='float32',std_floor=1e-6,
            mean385=mean.tolist(),std385=std.tolist(),array_sha=array_sha(np.stack([mean,std])),
            valid_zero_support_candidates_included=True),
        models=models,frame_receipts=frames,read_paths=sorted(READS),write_paths=sorted(WRITES),
        numpy_version=np.__version__,elapsed_seconds=time.monotonic()-started,
        context_padding_caveat=protocol['caveat'],stable_joint_improvement_achieved=False,
        performance_improvement_measured=False,method_success=False,goal_complete=False)
    save(RAW/'TRAIN_APPEARANCE_INPUTS.json',result)
    save(DOC/'TRAIN_APPEARANCE_INPUTS.json',result)
    lines=['# 원본 이미지 영역으로 제한한 DINO TRAIN 입력','',
        '기존 token 재풀링은 완료했습니다. 독립 검산은 아직 진행하지 않았습니다. 새 학습·추론·T/R 성능 평가는 0회이며, 학습 준비가 검증되었다는 판정이 아닙니다.','',
        '| model | valid candidates | previous support points | native support points | removed padding points | valid zero-support candidates |',
        '|---|---:|---:|---:|---:|---:|']
    for m,s in models.items():
        lines.append(f"| {m} | {s['valid_candidates']} | {s['previous_supported_points']} | {s['native_supported_points']} | {s['removed_prepared_padding_points']} | {s['valid_zero_native_support_candidates']} |")
    lines+=['', '기존 TRAIN 2,598행·원래 실패 1행·전체 유효 후보 20,776개를 보존했습니다. '+
            '이전 crop과 DINO token은 그대로입니다. 지원 조건만 기존 메타데이터의 pad에 따른 원본 이미지 영역으로 제한하고 '+
            'R0 유효 후보 5,194개에서 FP32 mean/std를 다시 계산했습니다. 지원점이 0개라도 원래 후보를 무효화하지 않았습니다.','',
            'token 계산에는 반사 패딩을 포함한 이전 crop을 사용했으므로 내부 위치를 표본화해도 문맥을 통한 패딩 영향은 남을 수 있습니다.','',
            '[영수증](TRAIN_APPEARANCE_INPUTS.json) · [프로토콜](INPUT_PROTOCOL.json) · [설계](DESIGN_KO.md)','']
    text='\n'.join(lines)
    with (DOC/'CONSTRUCTION_KO.md').open('x') as f:
        f.write(text)
    print('NATIVE_INPUT_CONSTRUCTION_COMPLETE',dict(frames=2598,new_forwards=0,fits=0,independent_verification='PENDING',receipt=bind(DOC/'TRAIN_APPEARANCE_INPUTS.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['selfcheck','seal','freeze'])
    action=parser.parse_args().action
    if action=='selfcheck':
        print(json.dumps(V.selfcheck(),ensure_ascii=False,indent=2))
    elif action=='seal':
        seal()
    else:
        freeze()
