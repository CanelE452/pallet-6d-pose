"""Independent native-support re-pooling verification; TRAIN inputs only.

Uses the previous independent scalar projector and Torch float64 sampler, not
either production visual_features module. No model, image, target or policy.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scripts.research.pallet_pose_dino_input_audit_20261001_v1 import verify_train_inputs as A

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_pose_dino_native_inputs_20261001_v1'
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
MODELS, HYP = A.MODELS, A.HYP
ATOL = RTOL = 2e-6
READS = set()
bind, read, verify, array_sha = A.bind, A.read, A.verify, A.array_sha


def guard(allowed):
    allowed = {Path(p).resolve() for p in allowed}
    outputs = {DOC/'INPUT_VERIFICATION.json', DOC/'INPUT_VERIFICATION_KO.md'}
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(ROOT):
            return
        forbidden = ('SOURCE_TRAIN_LABELS', 'SYNTH_LABELS', 'SYNTH_RECORDS', 'GEOMETRY_SIDETABLE',
                     'GEOMETRY_RESOLVED_POSE_GT', 'SOURCE_VAL_', 'TRAIN_CONVERGENCE',
                     'POSE_METRICS', 'REAL_', '/fits/', '/model_parameters/')
        assert not any(x in str(path) for x in forbidden), ('NO_TARGET_OR_QUALITY_READ', str(path))
        assert path.suffix.lower() not in ('.jpg','.jpeg','.png','.webp','.pt','.pth'), ('NO_IMAGE_OR_WEIGHT', str(path))
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if writing:
            assert path in outputs, ('WRITE_SCOPE', str(path))
        else:
            assert path in allowed or path in outputs or path.suffix in ('.py','.pyc'), ('READ_SCOPE', str(path))
            if path.suffix not in ('.py','.pyc'):
                READS.add(str(path.relative_to(ROOT)))
    sys.addaudithook(hook)


def native_intersection(uv, prepared, hw, pad):
    """Separate scalar half-open original-image rectangle and intersection."""
    assert np.ndim(pad) == 0 and np.isfinite(pad) and int(pad) == pad and pad >= 0
    assert len(hw) == 2 and all(int(v) == v and v > 0 for v in hw)
    assert 2*pad < min(hw)
    rectangle = np.array([pad, pad, hw[1]-pad, hw[0]-pad], np.float64)
    uv, prepared = np.asarray(uv, np.float64), np.asarray(prepared)
    assert uv.shape == (8,2) and prepared.shape == (8,) and prepared.dtype == bool
    mask = np.zeros(8, bool)
    for k in range(8):
        u,v = map(float, uv[k])
        mask[k] = bool(prepared[k] and np.isfinite(u) and np.isfinite(v) and
                       pad <= u < hw[1]-pad and pad <= v < hw[0]-pad)
    return rectangle, mask


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def run():
    torch.set_num_threads(1)
    started = time.monotonic()
    assert A.ATOL == A.RTOL == ATOL == RTOL
    protocol_path, receipt_path = DOC/'INPUT_PROTOCOL.json', RAW/'TRAIN_APPEARANCE_INPUTS.json'
    p, r = read(protocol_path), read(receipt_path)
    assert p['frames'] == 2598 and p['train_only'] and p['output_dim'] == 385
    assert r['complete'] and r['input_construction_pass'] and r['source_TRAIN_only']
    assert r['protocol'] == bind(protocol_path)
    artifacts = [*p['inputs'].values(), *p['codes'], p['design'], r['descriptors'], r['native_support_details']]
    allowed = [protocol_path, receipt_path, DOC/'TRAIN_APPEARANCE_INPUTS.json', Path(__file__),
               Path(A.__file__), *[ROOT/b['path'] for b in artifacts]]
    guard(allowed)
    seen = {}
    for b in artifacts:
        if b['path'] in seen:
            assert seen[b['path']] == b
        else:
            verify(b); seen[b['path']] = b
    assert (DOC/'TRAIN_APPEARANCE_INPUTS.json').read_bytes() == receipt_path.read_bytes()
    parent = read(ROOT/p['inputs']['previous_receipt']['path'])
    old_protocol = read(ROOT/p['inputs']['previous_protocol']['path'])
    old_verification = read(ROOT/p['inputs']['previous_verification']['path'])
    padding = read(ROOT/p['inputs']['padding_audit']['path'])
    assert parent['complete'] and old_verification['complete'] and old_verification['PASS']
    assert padding['complete'] and padding['PASS']
    assert parent['protocol'] == old_verification['protocol'] == padding['protocol'] == p['inputs']['previous_protocol']
    assert old_verification['input_receipt'] == p['inputs']['previous_receipt']
    assert old_verification['code'] == bind(Path(A.__file__))
    assert parent['descriptors'] == padding['descriptors'] == old_verification['descriptors'] == p['inputs']['previous_descriptors']
    assert parent['tokens'] == old_verification['tokens'] == p['inputs']['tokens'] == r['tokens']
    assert p['frozen_parent'] == {k:old_protocol[k] for k in ('crop','tokens','backbone')}
    for key,binding in old_protocol['inputs'].items():
        assert p['inputs'][key] == binding
    for key in ('previous_protocol','previous_receipt','previous_descriptors','previous_verification'):
        assert r[key.replace('previous','parent')] == p['inputs'][key]
    assert r['padding_audit'] == p['inputs']['padding_audit']
    assert r['descriptor_rule'] == p['descriptor_rule'] and r['support_rule'] == p['support_rule']
    for key in ('image_forwards','fits_executed','optimizer_steps','policy_selections','new_PnP_solves',
                'new_image_forwards','image_files_read','weights_read'):
        assert r[key] == 0, key
    for key in ('GPU_used','source_label_values_read','source_VAL_features_extracted','real_features_extracted',
                'full_tokens_duplicated','stable_joint_improvement_achieved','performance_improvement_measured',
                'method_success','goal_complete'):
        assert r[key] is False, key
    assert r['existing_tokens_reused'] and r['original_candidate_validity_preserved']
    assert r['frames'] == 2598 and r['parent_image_forwards'] == 2597 and r['original_allinvalid_rows'] == 1
    assert r['independent_verification_status'] == 'PENDING' and r['independent_verification_required_before_fit']

    source = read(ROOT/p['inputs']['metadata']['path'])
    contract = read(ROOT/p['inputs']['source_contract']['path'])
    poses = read(ROOT/p['inputs']['poses']['path'])
    eligible = set(contract['fit_eligibility']['eligible_ids']['TRAIN'])
    indices = np.array([i for i,row in enumerate(source) if row['id'] in eligible], np.int64)
    rows = [source[i] for i in indices]
    assert len(rows) == len(eligible) == 2598 and all(row['split'] == 'TRAIN' for row in rows)
    assert poses['ids'] == [row['id'] for row in source]
    feature_lock = read(ROOT/p['inputs']['feature_lock']['path'])
    assert feature_lock['complete'] and not feature_lock['source_targets_read'] and not feature_lock['real_targets_read']
    for k in ('features','poses','metadata'):
        assert feature_lock[k] == p['inputs'][k]
    assert feature_lock['predictions'] == p['inputs']['source_predictions_lock']
    original, previous, details = (load_npz(ROOT/p['inputs']['previous_descriptors']['path']),
        load_npz(ROOT/r['descriptors']['path']), load_npz(ROOT/r['native_support_details']['path']))
    # Keep explicit names: original=prepared-canvas inputs; previous=new native inputs.
    arrays = previous
    main_keys = {'ids','source_index','anchor_index','crop_matrices','mean385','std385'} | {
        m+s for m in MODELS for s in ('_appearance385','_valid','_support8')}
    detail_keys = {'ids','source_index','anchor_index','padding_px','native_rectangles_xyxy'} | {
        m+s for m in MODELS for s in ('_prepared_support8','_native_support8','_removed_support8')}
    assert len(main_keys) == 18 and set(arrays) == set(original) == main_keys
    assert len(detail_keys) == 17 and set(details) == detail_keys
    assert arrays['ids'].tolist() == [row['id'] for row in rows]
    assert arrays['source_index'].dtype == np.int64
    np.testing.assert_array_equal(arrays['source_index'], indices)
    for key in ('ids','source_index','anchor_index','crop_matrices'):
        assert array_sha(arrays[key]) == array_sha(original[key])
        if key in details:
            assert array_sha(details[key]) == array_sha(arrays[key])
    with np.load(ROOT/p['inputs']['features']['path'], allow_pickle=False) as z:
        assert z['ids'].tolist() == [row['id'] for row in source] and tuple(z['hypothesis_names']) == HYP
        np.testing.assert_array_equal(arrays['anchor_index'], z['R0_GEO_index'][indices])
        for m in MODELS:
            assert array_sha(arrays[m+'_valid']) == array_sha(z[m+'_valid'][indices])
    matrices, anchors = arrays['crop_matrices'], arrays['anchor_index']
    assert matrices.shape == (2598,3,3) and matrices.dtype == np.float64
    assert np.isfinite(matrices).all()
    pads = details['padding_px']
    assert pads.dtype == np.int64 and pads.shape == (2598,)
    assert details['native_rectangles_xyxy'].shape == (2598,4) and details['native_rectangles_xyxy'].dtype == np.float64
    for i,row in enumerate(rows):
        rectangle,_ = native_intersection(np.zeros((8,2)), np.zeros(8,bool), row['hw'], row['pad'])
        assert pads[i] == row['pad']
        np.testing.assert_array_equal(details['native_rectangles_xyxy'][i], rectangle)
    allinvalid = ~arrays['R0_valid'].any(1)
    assert int(allinvalid.sum()) == 1 and np.all(anchors[allinvalid] == -1)
    assert not matrices[allinvalid].any()
    for m in MODELS:
        values,valid,support = (arrays[m+s] for s in ('_appearance385','_valid','_support8'))
        assert values.shape == (2598,2,385) and values.dtype == np.float32 and np.isfinite(values).all()
        assert valid.shape == (2598,2) and valid.dtype == bool and int(valid.sum()) == 5194
        assert support.shape == (2598,2,8) and support.dtype == bool
        assert array_sha(valid) == array_sha(original[m+'_valid'])
        np.testing.assert_array_equal(valid.any(1), ~allinvalid)
        for suffix in ('_prepared_support8','_native_support8','_removed_support8'):
            a = details[m+suffix]; assert a.shape == support.shape and a.dtype == bool
        np.testing.assert_array_equal(details[m+'_prepared_support8'], original[m+'_support8'])
        np.testing.assert_array_equal(details[m+'_native_support8'], support)
        np.testing.assert_array_equal(details[m+'_removed_support8'], original[m+'_support8'] & ~support)
        assert not (support & ~original[m+'_support8']).any()
        assert not values[~valid].any() and not support[~valid].any()
        zero = valid & ~support.any(2)
        assert not values[zero].any(), 'Zero native support must retain a valid zero descriptor'
        np.testing.assert_array_equal(values[:,:,-1], support.sum(2).astype(np.float32)/8)

    tokens = np.load(ROOT/r['tokens']['path'], mmap_mode='r', allow_pickle=False)
    assert isinstance(tokens,np.memmap) and tokens.shape == (2598,384,56,42) and tokens.dtype == np.float16
    assert len(parent['frame_receipts']) == len(r['frame_receipts']) == 2598
    maxima = {m:dict(max_absolute=0.,max_tolerance_fraction=0.,checked_candidates=0) for m in MODELS}
    checked_hashes = 0
    for i,row in enumerate(rows):
        old_frame, frame = parent['frame_receipts'][i], r['frame_receipts'][i]
        assert frame['id'] == old_frame['id'] == row['id'] and frame['source_index'] == int(indices[i])
        assert frame['image'] == old_frame['image'] == row['image']
        expected_frame = dict(old_frame, image_forward=False, token_reused=not bool(allinvalid[i]),
            parent_image_forward=old_frame['image_forward'], padding_px=int(pads[i]),
            native_rectangle_xyxy=details['native_rectangles_xyxy'][i].tolist())
        assert frame == expected_frame
        token = tokens[i]
        assert np.isfinite(token).all()
        if allinvalid[i]:
            assert not token.any() and not frame['token_reused']
            continue
        assert array_sha(token) == old_frame['token_sha']
        checked_hashes += 1
        np.testing.assert_array_equal(matrices[i], old_frame['matrix'])
        assert anchors[i] in (0,1) and arrays['R0_valid'][i,anchors[i]]
        assert poses['records']['R0'][row['id']]['GEO_name'] == HYP[int(anchors[i])]
        crops, native_masks = [], []
        for m in MODELS:
            hypotheses = {h['name']:h for h in poses['records'][m][row['id']]['hypotheses']}
            assert set(hypotheses) == set(HYP)
            for k,name in enumerate(HYP):
                pose = hypotheses[name]['pose']
                assert bool(pose['available']) == bool(arrays[m+'_valid'][i,k])
                uv,crop,prepared = A.project(pose,row['K'],matrices[i],row['hw'])
                np.testing.assert_array_equal(prepared, original[m+'_support8'][i,k])
                rectangle,native = native_intersection(uv,prepared,row['hw'],row['pad'])
                np.testing.assert_array_equal(rectangle,details['native_rectangles_xyxy'][i])
                np.testing.assert_array_equal(native,arrays[m+'_support8'][i,k])
                crops.append(crop); native_masks.append(native)
        sampled = A.sample_double(torch.from_numpy(token.astype(np.float64))[None],
                                  np.concatenate(crops),np.concatenate(native_masks))
        for j,m in enumerate(MODELS):
            for k in range(2):
                slot=j*2+k
                expected=A.ordered_mean_double(sampled[slot*8:(slot+1)*8],native_masks[slot])
                actual=arrays[m+'_appearance385'][i,k]
                assert actual[-1] == expected[-1]
                error,fraction=A.compare(actual,expected)
                maxima[m]['max_absolute']=max(maxima[m]['max_absolute'],error)
                maxima[m]['max_tolerance_fraction']=max(maxima[m]['max_tolerance_fraction'],fraction)
                maxima[m]['checked_candidates']+=int(arrays[m+'_valid'][i,k])
        if (i+1)%200 == 0 or i == len(rows)-1:
            print('VERIFY_NATIVE_INPUT',i+1,'/',len(rows),'seconds',round(time.monotonic()-started,1),flush=True)
    assert checked_hashes == r['token_hashes_checked'] == 2597
    assert sum(v['checked_candidates'] for v in maxima.values()) == 20776
    mean=arrays['R0_appearance385'][arrays['R0_valid']].mean(axis=0,dtype=np.float32)
    std=np.maximum(arrays['R0_appearance385'][arrays['R0_valid']].std(axis=0,dtype=np.float32),np.float32(1e-6))
    for key,wanted in (('mean385',mean),('std385',std)):
        assert arrays[key].shape == (385,) and arrays[key].dtype == np.float32
        np.testing.assert_array_equal(arrays[key],wanted)
        np.testing.assert_array_equal(np.asarray(r['normalization'][key],np.float32),wanted)
    assert r['normalization']['count'] == 5194 and r['normalization']['dtype'] == 'float32'
    assert r['normalization']['std_floor'] == 1e-6 and r['normalization']['valid_zero_support_candidates_included']
    assert r['normalization']['array_sha'] == array_sha(np.stack([mean,std]))
    models={}
    for m in MODELS:
        values,valid,support=(arrays[m+s] for s in ('_appearance385','_valid','_support8'))
        old=original[m+'_support8']; removed=old&~support
        counts=support.sum(2)[valid]
        statistics=dict(valid_candidates=int(valid.sum()),descriptor_sha=array_sha(values),
            valid_sha=array_sha(valid),support_sha=array_sha(support),
            supported_corner_histogram={str(k):int((counts==k).sum()) for k in range(9)},
            previous_supported_points=int(old.sum()),native_supported_points=int(support.sum()),
            removed_prepared_padding_points=int(removed.sum()),candidates_with_removed_support=int(removed.any(2).sum()),
            frames_with_removed_support=int(removed.any((1,2)).sum()),
            valid_zero_native_support_candidates=int((counts==0).sum()),
            valid_descriptor_changed_candidates=int(np.any(values!=original[m+'_appearance385'],axis=2)[valid].sum()),
            original_valid_unchanged=True)
        assert statistics == r['models'][m]
        prior=padding['models'][m]
        assert statistics['previous_supported_points'] == prior['supported_points']
        assert statistics['removed_prepared_padding_points'] == prior['padding_supported_points']
        assert statistics['candidates_with_removed_support'] == prior['candidates_with_padding_support']
        assert statistics['frames_with_removed_support'] == prior['frames_with_padding_support']
        models[m]=dict(**statistics,comparison=maxima[m],input_statistics=A.input_statistics(
            values,valid,anchors,mean,std,arrays['R0_appearance385']))
    native_meta=r['native_padding']
    assert native_meta['metadata'] == p['inputs']['metadata'] and native_meta['values'] == sorted(set(map(int,pads)))
    for name,arr in [('ids_sha',arrays['ids']),('source_index_sha',indices),('padding_sha',pads),
                     ('rectangles_sha',details['native_rectangles_xyxy'])]:
        assert native_meta[name] == array_sha(arr)
    result=dict(complete=True,PASS=True,verification_PASS=True,source_TRAIN_only=True,
        independent_verification_status='PASS',producer_status_retained='PENDING in immutable construction receipt; this separate verifier supplies the completed verdict',
        code=bind(Path(__file__)),independent_reference_code=bind(Path(A.__file__)),protocol=bind(protocol_path),
        input_receipt=bind(receipt_path),public_input_receipt=bind(DOC/'TRAIN_APPEARANCE_INPUTS.json'),
        tokens=r['tokens'],descriptors=r['descriptors'],native_support_details=r['native_support_details'],
        parent_verification=p['inputs']['previous_verification'],padding_audit=p['inputs']['padding_audit'],
        frames=2598,all_invalid_rows=1,valid_candidate_descriptors=20776,token_frame_hashes_checked=2597,
        zero_token_invalid_frames_checked=1,full_token_file_sha_checked=True,token_bytes_unchanged=True,
        main_array_keys=18,support_detail_array_keys=17,main_array_hashes={k:array_sha(v) for k,v in arrays.items()},
        support_detail_array_hashes={k:array_sha(v) for k,v in details.items()},
        comparison=dict(atol=ATOL,rtol=RTOL,tolerance_unchanged_from_parent=True,
            reference='Previous independent scalar rigid projection + new scalar native rectangle intersection + Torch CPU float64 bilinear sampler/sorted mean; no producer helper'),
        models=models,normalization_exact=True,normalization=dict(array_sha=array_sha(np.stack([mean,std])),
            count=5194,dtype='float32',std_floor=1e-6,valid_zero_support_candidates_included=True),
        exact_removed_support_matches_parent_padding_audit=True,original_candidate_validity_unchanged=True,
        original_R0_anchor_identity_unchanged=True,anchor_reference='R0 operational descriptor at original R0_GEO_index for every expert',
        new_fits=0,optimizer_steps=0,policy_selections=0,new_PnP_calls=0,new_image_forwards=0,GPU_used=False,
        image_files_read=0,weight_files_read=0,target_values_read=0,VAL_quality_reads=0,real_reference_reads=0,real_routes=0,
        backbone_forward_recomputed=False,backbone_inference_correctness_independently_verified=False,
        shared_cache_disclosure='Metadata/pose/feature containers contain other splits; only fixed eligible TRAIN indices are numerically verified',
        context_padding_caveat=r['context_padding_caveat'],method_success=False,goal_complete=False,
        stable_joint_improvement_achieved=False,performance_improvement_measured=False,
        read_paths=sorted(READS),elapsed_seconds=time.monotonic()-started)
    text=render(result)
    with (DOC/'INPUT_VERIFICATION.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    with (DOC/'INPUT_VERIFICATION_KO.md').open('x') as f:f.write(text)
    print('NATIVE_INPUT_VERIFICATION_PASS',bind(DOC/'INPUT_VERIFICATION.json'),flush=True)
    return result


def render(r):
    lines=['# 원영상 경계 재풀링의 독립 입력 검산','',
        '**입력 검산 PASS입니다. 새 학습·후보 선택·T/R 성능 평가는 0회이며, 안정적인 공동 개선 목표 달성 판정이 아닙니다.**','',
        'TRAIN 2,598행과 원래 무효 1행, 유효 후보 20,776개를 유지했습니다. 기존 FP16 token 파일 전체 SHA와 2,597개 유효 frame token SHA를 확인했습니다. '
        '별도 scalar 투영·원영상 반열린 경계와 Torch CPU float64 보간·정렬 평균으로 새 descriptor를 전수 검산했습니다. 허용오차 atol=rtol=2e-6은 이전 감사와 동일합니다.','',
        '| Expert | 유효 후보 | 제거된 반사영역 지원점 | 원영상 지원점 | 유효하지만 지원점 0인 후보 | 최대 절대차 |','|---|---:|---:|---:|---:|---:|']
    for m in MODELS:
        s=r['models'][m];lines.append(f"| {m} | {s['valid_candidates']} | {s['removed_prepared_padding_points']} | {s['native_supported_points']} | {s['valid_zero_native_support_candidates']} | {s['comparison']['max_absolute']:.12g} |")
    lines+=['','제거된 지원점·영향 후보·영향 frame 수는 이전 padding 감사와 정확히 일치합니다. 새 지원 mask는 이전 mask의 부분집합이며, 원래 pose 유효성은 바꾸지 않았습니다. '
        '지원점 0인 유효 후보도 zero385로 유지하여 R0 유효 후보 5,194개 전체의 FP32 mean/std 계산에 포함했습니다. main 18개 배열과 support 상세 17개 배열의 dtype·shape·내용을 대조했습니다.','',
        '정규화 후 차분의 anchor는 모든 expert에서 원래 R0 operational GEO 후보입니다. DIVERSE 자신의 동일 W/D 후보를 anchor로 바꾸지 않았습니다. '
        '입력 가설 차이 통계는 정답·모델 점수를 사용하지 않은 입력 대비이며 T/R 개선 여부를 뜻하지 않습니다.','',
        '원본 RGB·모델 가중치·GT 오류·VAL 품질·실사 참조를 읽지 않았고, backbone·GPU·PnP를 실행하지 않았습니다. '
        '기존 crop/token을 그대로 사용했으므로 원영상 내부 token에도 반사 padding의 문맥 영향은 남을 수 있습니다. '
        '이 검산은 저장 token 이후의 입력 구성 검증이며 backbone 추론 자체의 독립 재실행이 아닙니다.','',
        '생산 영수증의 PENDING은 생산 당시의 독립 검산 상태로 보존했습니다. 이 별도 영수증이 이후 완료된 PASS를 기록하며 기존 파일을 수정하지 않습니다.','',
        '[검산 JSON](INPUT_VERIFICATION.json) · [생산 영수증](TRAIN_APPEARANCE_INPUTS.json) · [입력 프로토콜](INPUT_PROTOCOL.json)','']
    return '\n'.join(lines)


def selfcheck():
    torch.set_num_threads(1)
    points=np.array([[100.,100.],[475.999,667.999],[476.,200.],[200.,668.],
                     [99.999,200.],[200.,99.999],[200.,200.],[np.nan,200.]])
    rectangle,mask=native_intersection(points,np.ones(8,bool),[768,576],100)
    np.testing.assert_array_equal(rectangle,[100.,100.,476.,668.])
    np.testing.assert_array_equal(mask,[1,1,0,0,0,0,1,0])
    _,none=native_intersection(points,np.zeros(8,bool),[768,576],100);assert not none.any()
    for bad in (-1,100.25,288):
        try:native_intersection(points,np.ones(8,bool),[768,576],bad)
        except AssertionError:pass
        else:raise AssertionError('Invalid pad accepted')
    pose=dict(available=True,cf_extents=[2.,2.,2.],R_cf=np.eye(3).tolist(),centroid=[0.,0.,5.])
    K=np.array([[100.,0.,100.],[0.,100.,100.],[0.,0.,1.]])
    uv,crop,prepared=A.project(pose,K,np.eye(3),[768,576])
    _,native=native_intersection(uv,prepared,[768,576],100)
    assert prepared.all() and int(native.sum())==2
    yy,xx=np.mgrid[:56,:42];feature=np.broadcast_to(xx+2*yy,(384,56,42)).astype(np.float16)
    sampled=A.sample_double(torch.from_numpy(feature.astype(np.float64))[None],crop,native)
    expected=A.ordered_mean_double(sampled,native)
    grid=(crop[native]+.5)*[42/576,56/768]-.5
    np.testing.assert_allclose(expected[:384],np.mean(grid[:,0]+2*grid[:,1]),atol=1e-12,rtol=0)
    assert expected[-1]==.25
    assert not A.ordered_mean_double(np.zeros((8,384)),np.zeros(8,bool)).any()
    A.compare(expected.astype(np.float32),expected)
    try:A.compare(expected+1e-3,expected)
    except AssertionError:pass
    else:raise AssertionError('Fixed tolerance fault escaped')
    return dict(PASS=True,invented_only=True,half_open_native_boundaries=True,
        independent_linear_grid=True,zero_support=True,fixed_tolerance_fault_rejected=True,
        actual_inputs_read=0,images_read=0,weights_read=0,forwards=0,fits=0,policy_selections=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--selfcheck',action='store_true');group.add_argument('--write',action='store_true')
    args=parser.parse_args()
    if args.selfcheck:print(json.dumps(selfcheck(),ensure_ascii=False))
    else:run()
