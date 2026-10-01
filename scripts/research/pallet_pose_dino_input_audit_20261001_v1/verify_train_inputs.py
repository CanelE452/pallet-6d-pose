"""Independent, TRAIN-only audit of retained DINO tokens and appearance inputs.

The reference uses scalar rigid projection and Torch CPU float64 grid_sample.
It does not import the producer, visual_features, images, labels or weights.
``--selfcheck`` uses only invented arrays; ``--write`` requires finished inputs.
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
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_pose_dino_input_audit_20261001_v1'
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
MODELS = ('R0', 'DIVERSE251_s1', 'DIVERSE251_s2', 'DIVERSE251_s3')
HYP = ('long-face-front', 'short-face-front')
SIGNS = ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
         (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))
ATOL = RTOL = 2e-6
READS = set()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def verify(binding):
    assert bind(ROOT / binding['path']) == binding, binding['path']


def array_sha(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def install_guard(allowed):
    allowed = {Path(p).resolve() for p in allowed}
    outputs = {DOC/'INPUT_VERIFICATION.json', DOC/'INPUT_VERIFICATION_KO.md'}
    forbidden = ('SOURCE_TRAIN_LABELS', 'SYNTH_LABELS', 'SYNTH_RECORDS',
                 'GEOMETRY_RESOLVED_POSE_GT', 'TRAIN_CONVERGENCE', 'SOURCE_VAL_',
                 '/fits/', '/model_parameters/', 'POSE_METRICS', 'REAL_')
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if not p.is_relative_to(ROOT):
            return
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(
            isinstance(flags, int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        assert not any(t in str(p) for t in forbidden), ('FORBIDDEN_AUDIT_INPUT', str(p))
        assert p.suffix.lower() not in ('.jpg','.jpeg','.png','.webp','.pt','.pth'), ('NO_IMAGES_OR_WEIGHTS', str(p))
        if writing:
            assert p in outputs, ('WRITE_DENIED', str(p))
        else:
            assert p in allowed or p in outputs or p.suffix in ('.py','.pyc'), ('READ_DENIED', str(p))
            if p.suffix not in ('.py','.pyc'):
                READS.add(str(p.relative_to(ROOT)))
    sys.addaudithook(hook)


def crop_matrix(box):
    """Independent scalar form of the pre-existing 1.25-expanded wide crop."""
    x0,y0,x1,y1 = map(float, box)
    assert all(math.isfinite(x) for x in (x0,y0,x1,y1)) and x1>x0 and y1>y0
    w,h = x1-x0,y1-y0
    if w > .75*h:
        h = w/.75
    else:
        w = .75*h
    w *= 1.25
    h *= 1.25
    cx,cy = (x0+x1)/2.,(y0+y1)/2.
    return np.array([[288./w,0.,288.*(.5-cx/w)+144.],
                     [0.,384./h,384.*(.5-cy/h)+192.],[0.,0.,1.]], np.float64)


def project(pose, K, matrix, hw):
    """Project exactly eight fixed physical corners, without a pose solver."""
    uv = np.zeros((8,2), np.float64)
    crop = np.zeros((8,2), np.float64)
    support = np.zeros(8, bool)
    if not pose['available']:
        return uv,crop,support
    K,M = np.asarray(K,np.float64),np.asarray(matrix,np.float64)
    ext,R,t = (np.asarray(pose[k],np.float64) for k in ('cf_extents','R_cf','centroid'))
    assert ext.shape==(3,) and R.shape==(3,3) and t.shape==(3,) and (ext>0).all()
    assert K.shape==M.shape==(3,3) and all(np.isfinite(a).all() for a in (ext,R,t,K,M))
    np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-6,rtol=0)
    assert abs(np.linalg.det(R)-1.)<=1e-6
    for i,sign in enumerate(SIGNS):
        corner = [sign[j]*ext[j]/2. for j in range(3)]
        xyz = [sum(R[a,b]*corner[b] for b in range(3))+t[a] for a in range(3)]
        if xyz[2] <= 0.:
            continue
        p = [sum(K[a,b]*xyz[b] for b in range(3)) for a in range(3)]
        uv[i] = p[0]/p[2],p[1]/p[2]
        if not np.isfinite(uv[i]).all():
            continue
        crop[i] = [M[a,0]*uv[i,0]+M[a,1]*uv[i,1]+M[a,2] for a in range(2)]
        support[i] = bool(np.isfinite(crop[i]).all() and
            0.<=uv[i,0]<hw[1] and 0.<=uv[i,1]<hw[0] and
            0.<=crop[i,0]<576. and 0.<=crop[i,1]<768.)
    return uv,crop,support


def sample_double(token_tensor, crop, support):
    """Independent Torch CPU float64 sampler; no producer sampler invocation."""
    assert token_tensor.shape==(1,384,56,42) and token_tensor.dtype==torch.float64
    crop,support = np.asarray(crop,np.float64),np.asarray(support)
    assert crop.shape==(len(support),2) and support.dtype==bool
    out = np.zeros((len(support),384),np.float64)
    if support.any():
        points = crop[support]
        assert np.isfinite(points).all() and (points>=0).all() and (points<[576.,768.]).all()
        grid = torch.from_numpy(2.*(points+.5)/np.array([576.,768.])-1.).reshape(1,-1,1,2)
        out[support] = F.grid_sample(token_tensor,grid,mode='bilinear',padding_mode='border',
                                    align_corners=False)[0,:,:,0].T.numpy()
    return out


def ordered_mean_double(samples, support):
    assert samples.shape==(8,384) and support.shape==(8,)
    out = np.zeros(385,np.float64)
    if support.any():
        out[:384] = np.sort(samples[support],axis=0).mean(axis=0,dtype=np.float64)
    out[-1] = int(support.sum())/8.
    return out


def compare(actual, expected):
    actual,expected = np.asarray(actual,np.float64),np.asarray(expected,np.float64)
    assert actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
    error = np.abs(actual-expected)
    limit = ATOL+RTOL*np.abs(expected)
    assert np.all(error<=limit), ('FIXED_DESCRIPTOR_TOLERANCE_EXCEEDED',float(error.max()),float((error/limit).max()))
    return float(error.max(initial=0.)),float((error/limit).max(initial=0.))


def quantiles(a):
    a = np.asarray(a,np.float64)
    assert a.ndim==1 and len(a) and np.isfinite(a).all()
    return dict(zip(('minimum','P25','median','P75','P90','P99','maximum'),
                    map(float,np.quantile(a,[0,.25,.5,.75,.9,.99,1]))))


def input_statistics(values, valid, anchor, mean, std, r0_values):
    """Input contrast only: no target, prediction weight or candidate selection."""
    both = valid.all(1)
    raw_delta = values[both,0].astype(np.float64)-values[both,1].astype(np.float64)
    normalized = ((values-mean)/std).astype(np.float64)
    normalized_r0 = ((r0_values-mean)/std).astype(np.float64)
    delta = normalized[both,0]-normalized[both,1]
    anchor_delta = np.zeros_like(normalized)
    for i in np.flatnonzero(valid.any(1)):
        assert anchor[i] in (0,1) and valid[i,anchor[i]]
        anchor_delta[i,valid[i]] = normalized[i,valid[i]]-normalized_r0[i,anchor[i]]
        if values is r0_values:
            assert not anchor_delta[i,anchor[i]].any()
    assert not anchor_delta[~valid].any()
    vv = values[valid]
    nonzero = np.count_nonzero(vv,axis=0)
    varying = np.ptp(vv,axis=0)>0.
    return dict(valid_candidates=int(valid.sum()),both_valid_rows=int(both.sum()),
        hypothesis_descriptor_different_rows=int(np.any(raw_delta!=0.,axis=1).sum()),
        hypothesis_raw_L2=quantiles(np.linalg.norm(raw_delta,axis=1)),
        hypothesis_normalized_L2=quantiles(np.linalg.norm(delta,axis=1)),
        valid_candidate_anchor_difference_L2=quantiles(np.linalg.norm(anchor_delta[valid],axis=1)),
        nonzero_channel_count=int((nonzero>0).sum()),varying_channel_count=int(varying.sum()),
        nonzero_count_by_channel=nonzero.tolist(),varying_by_channel=varying.tolist(),
        std_by_channel=np.std(vv.astype(np.float64),axis=0).tolist(),
        normalized_candidate_anchor_difference_sha=array_sha(anchor_delta),
        anchor_reference='R0 operational descriptor at original R0_GEO_index for every expert',
        performance_metric=False,targets_used=False,policy_selections=0)


def run():
    torch.set_num_threads(1)
    started = time.monotonic()
    protocol_path,receipt_path = DOC/'INPUT_PROTOCOL.json',RAW/'TRAIN_APPEARANCE_INPUTS.json'
    # Bootstrap reads are restricted to the sealed protocol, receipt and TRAIN membership metadata.
    p,result = read(protocol_path),read(receipt_path)
    assert p['frames']==2598 and p['train_only'] and p['output_dim']==385 and p['no_fit'] and p['no_label_values']
    assert result['complete'] and result['input_construction_pass'] and result['source_TRAIN_only']
    assert result['protocol']==bind(protocol_path) and result['frames']==2598 and result['image_forwards']==2597
    assert result['fits_executed']==result['new_PnP_solves']==0
    assert not result['source_label_values_read'] and not result['source_VAL_features_extracted'] and not result['real_features_extracted']
    assert result['original_candidate_validity_preserved'] and result['original_allinvalid_rows']==1
    required = {'source_contract','feature_lock','features','poses','metadata','source_predictions_lock'}
    assert set(p['inputs'])==required
    assert result['tokens']['path']==str((RAW/'TRAIN_TOKENS.npy').relative_to(ROOT))
    assert result['descriptors']['path']==str((RAW/'TRAIN_APPEARANCE.npz').relative_to(ROOT))
    source = read(ROOT/p['inputs']['metadata']['path'])
    contract = read(ROOT/p['inputs']['source_contract']['path'])
    eligible = set(contract['fit_eligibility']['eligible_ids']['TRAIN'])
    indices = np.array([i for i,row in enumerate(source) if row['id'] in eligible],np.int64)
    rows = [source[i] for i in indices]
    assert len(rows)==len(eligible)==2598 and all(r['split']=='TRAIN' for r in rows)
    lock = read(ROOT/p['inputs']['source_predictions_lock']['path'])
    r0_receipt_binding = lock['receipts']['R0']
    r0 = read(ROOT/r0_receipt_binding['path'])
    assert lock['complete'] and r0['complete'] and r0['protocol']==lock['protocol']
    prediction_bindings = [r0['files'][i] for i in indices]
    allowed = [protocol_path,receipt_path,Path(__file__),ROOT/p['previous_input_audit']['path'],
               ROOT/r0_receipt_binding['path'],ROOT/result['tokens']['path'],ROOT/result['descriptors']['path'],
               *[ROOT/b['path'] for b in p['inputs'].values()],*[ROOT/b['path'] for b in p['codes']],
               *[ROOT/b['path'] for b in prediction_bindings]]
    install_guard(allowed)
    for b in [p['previous_input_audit'],r0_receipt_binding,result['tokens'],result['descriptors'],*p['inputs'].values(),*p['codes']]:
        verify(b)
    feature_lock = read(ROOT/p['inputs']['feature_lock']['path'])
    assert feature_lock['complete'] and not feature_lock['source_targets_read'] and not feature_lock['real_targets_read']
    for k in ('features','poses','metadata'):
        assert feature_lock[k]==p['inputs'][k]
    assert feature_lock['predictions']==p['inputs']['source_predictions_lock']
    with np.load(ROOT/p['inputs']['features']['path'],allow_pickle=False) as z:
        assert z['ids'].tolist()==[r['id'] for r in source] and tuple(z['hypothesis_names'].tolist())==HYP
        original = {m:z[m+'_valid'][indices].copy() for m in MODELS}
        original_anchor = z['R0_GEO_index'][indices].copy()
    poses = read(ROOT/p['inputs']['poses']['path'])
    assert poses['ids']==[r['id'] for r in source]
    with np.load(ROOT/result['descriptors']['path'],allow_pickle=False) as z:
        arrays = {k:z[k].copy() for k in z.files}
    expected_keys = {'ids','source_index','anchor_index','crop_matrices','mean385','std385'}|{
        m+s for m in MODELS for s in ('_appearance385','_valid','_support8')}
    assert set(arrays)==expected_keys
    assert arrays['ids'].tolist()==[r['id'] for r in rows]
    np.testing.assert_array_equal(arrays['source_index'],indices)
    np.testing.assert_array_equal(arrays['anchor_index'],original_anchor)
    for m in MODELS:
        assert arrays[m+'_appearance385'].shape==(2598,2,385) and arrays[m+'_appearance385'].dtype==np.float32
        assert arrays[m+'_valid'].shape==(2598,2) and arrays[m+'_valid'].dtype==bool
        assert arrays[m+'_support8'].shape==(2598,2,8) and arrays[m+'_support8'].dtype==bool
        np.testing.assert_array_equal(arrays[m+'_valid'],original[m])
        assert int(original[m].sum())==5194
        assert np.array_equal(original[m].any(1),original['R0'].any(1))
    all_invalid = ~original['R0'].any(1)
    assert int(all_invalid.sum())==1 and np.all(original_anchor[all_invalid]==-1)
    matrices = arrays['crop_matrices']
    assert matrices.shape==(2598,3,3) and matrices.dtype==np.float64 and np.isfinite(matrices).all()
    tokens = np.load(ROOT/result['tokens']['path'],mmap_mode='r',allow_pickle=False)
    assert tokens.shape==(2598,384,56,42) and tokens.dtype==np.float16 and isinstance(tokens,np.memmap)
    frames = result['frame_receipts']
    assert len(frames)==2598
    maxima = {m:dict(max_absolute=0.,max_tolerance_fraction=0.,checked_candidates=0) for m in MODELS}
    token_hash_count = 0
    for j,(row,frame,pb) in enumerate(zip(rows,frames,prediction_bindings)):
        assert frame['id']==row['id'] and frame['source_index']==int(indices[j])
        assert frame['image']==row['image'] and frame['prediction']==pb
        verify(pb)
        pred = read(ROOT/pb['path'])
        assert pred['id']==row['id'] and pred['model']=='R0' and pred['protocol_sha']==lock['protocol']['sha256']
        assert pred['checkpoint_sha']==r0['checkpoint']['sha256']
        feature = tokens[j]
        assert np.isfinite(feature).all()
        assert frame['all_original_candidates_invalid']==bool(all_invalid[j])
        if all_invalid[j]:
            assert not frame['image_forward'] and not feature.any() and not matrices[j].any()
            for m in MODELS:
                assert not arrays[m+'_appearance385'][j].any() and not arrays[m+'_support8'][j].any()
            continue
        assert frame['image_forward'] and array_sha(feature)==frame['token_sha']
        token_hash_count += 1
        selected_index = pred['prediction']['selected_index']
        assert selected_index is not None
        matrix = crop_matrix(pred['prediction']['candidates'][selected_index]['box_xyxy'])
        np.testing.assert_array_equal(matrix,matrices[j])
        np.testing.assert_array_equal(matrix,np.asarray(frame['matrix'],np.float64))
        all_crop,all_support = [],[]
        for m in MODELS:
            hypotheses = {h['name']:h for h in poses['records'][m][row['id']]['hypotheses']}
            assert set(hypotheses)==set(HYP)
            for k,name in enumerate(HYP):
                if original[m][j,k]:
                    assert hypotheses[name]['pose']['available']
                    _,crop,support = project(hypotheses[name]['pose'],row['K'],matrix,row['hw'])
                else:
                    crop,support = np.zeros((8,2),np.float64),np.zeros(8,bool)
                np.testing.assert_array_equal(support,arrays[m+'_support8'][j,k])
                all_crop.append(crop)
                all_support.append(support)
        sampled = sample_double(torch.from_numpy(feature.astype(np.float64))[None],np.concatenate(all_crop),np.concatenate(all_support))
        for m_index,m in enumerate(MODELS):
            for k in range(2):
                index = m_index*2+k
                expected = ordered_mean_double(sampled[index*8:(index+1)*8],all_support[index])
                actual = arrays[m+'_appearance385'][j,k]
                assert actual[-1]==expected[-1]
                error,fraction = compare(actual,expected)
                maxima[m]['max_absolute'] = max(maxima[m]['max_absolute'],error)
                maxima[m]['max_tolerance_fraction'] = max(maxima[m]['max_tolerance_fraction'],fraction)
                maxima[m]['checked_candidates'] += int(original[m][j,k])
        if (j+1)%200==0 or j==len(rows)-1:
            print('VERIFY_DINO_TRAIN_INPUT',j+1,'/',len(rows),'seconds',round(time.monotonic()-started,1),flush=True)
    assert token_hash_count==2597 and sum(v['checked_candidates'] for v in maxima.values())==20776
    mean = arrays['R0_appearance385'][original['R0']].mean(axis=0,dtype=np.float32)
    std = np.maximum(arrays['R0_appearance385'][original['R0']].std(axis=0,dtype=np.float32),np.float32(1e-6))
    for key,expected in (('mean385',mean),('std385',std)):
        assert arrays[key].dtype==np.float32 and arrays[key].shape==(385,)
        np.testing.assert_array_equal(arrays[key],expected)
        np.testing.assert_array_equal(np.asarray(result['normalization'][key],np.float32),expected)
    assert result['normalization']['count']==5194 and result['normalization']['dtype']=='float32'
    assert result['normalization']['std_floor']==1e-6
    assert result['normalization']['array_sha']==array_sha(np.stack([mean,std]))
    model_results = {}
    for m in MODELS:
        values,valid,support = (arrays[m+s] for s in ('_appearance385','_valid','_support8'))
        assert not values[~valid].any() and not support[~valid].any()
        np.testing.assert_array_equal(values[:,:,-1],support.sum(2).astype(np.float32)/8)
        summary = result['models'][m]
        assert summary['descriptor_sha']==array_sha(values) and summary['valid_sha']==array_sha(valid)
        assert summary['support_sha']==array_sha(support) and summary['valid_candidates']==5194
        hist = {str(k):int((support.sum(2)[valid]==k).sum()) for k in range(9)}
        assert hist==summary['supported_corner_histogram']
        model_results[m] = dict(comparison=maxima[m],support_histogram=hist,
            descriptor_sha=array_sha(values),valid_sha=array_sha(valid),support_sha=array_sha(support),
            input_statistics=input_statistics(values,valid,original_anchor,mean,std,arrays['R0_appearance385']))
    receipt = dict(complete=True,PASS=True,verification_PASS=True,source_TRAIN_only=True,
        verification_scope='Retained-token to descriptor, frozen projection/crop, original validity, normalization and input contrast only',
        code=bind(Path(__file__)),protocol=bind(protocol_path),input_receipt=bind(receipt_path),
        tokens=result['tokens'],descriptors=result['descriptors'],
        comparison=dict(atol=ATOL,rtol=RTOL,reference='Torch CPU float64 grid_sample, align_corners=False, border, channelwise sorted float64 mean',tolerance_fixed_before_actual=True),
        frames=2598,all_invalid_rows=1,valid_candidate_descriptors=20776,token_frame_hashes_checked=token_hash_count,
        zero_token_invalid_frames_checked=1,full_token_file_sha_checked=True,normalization_exact=True,
        normalization=dict(array_sha=array_sha(np.stack([mean,std])),valid_candidates=5194,dtype='float32',std_floor=1e-6),
        array_hashes={k:array_sha(v) for k,v in arrays.items()},models=model_results,
        memory='Read-only NPY mmap; one frame converted to float64 at a time',
        shared_cache_disclosure='Source feature/pose/metadata files include other splits; only eligible TRAIN rows enter numerical verification',
        new_fits=0,optimizer_steps=0,policy_selections=0,new_PnP_calls=0,new_image_forwards=0,
        image_files_read=0,weight_files_read=0,target_values_read=0,VAL_quality_reads=0,real_reference_reads=0,real_routes=0,
        backbone_forward_recomputed=False,backbone_inference_correctness_independently_verified=False,
        inference_limit='Retained tokens and producer provenance are verified; this audit does not independently rerun the frozen backbone or image preprocessing',
        method_success=False,goal_complete=False,performance_improvement_measured=False,
        read_paths=sorted(READS),elapsed_seconds=time.monotonic()-started)
    text = render(receipt)
    for path,value in ((DOC/'INPUT_VERIFICATION.json',json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n'),
                       (DOC/'INPUT_VERIFICATION_KO.md',text)):
        with path.open('x') as f:
            f.write(value)
    print('INPUT_VERIFICATION_PASS',bind(DOC/'INPUT_VERIFICATION.json'),flush=True)
    return receipt


def render(r):
    lines = ['# DINO TRAIN 입력 독립 검산', '',
        '입력 검산 PASS입니다. 새 학습·T/R 성능 평가·실사 경로 실행은 모두 0회이며, 이 결과는 성능 개선의 증거가 아닙니다.', '',
        'TRAIN 2,598행을 모두 보존하고 원래 실패 1행, 유효 후보 20,776개를 검산했습니다. 저장된 FP16 token을 한 프레임씩 읽어 독립 scalar 투영과 Torch CPU float64 grid_sample로 재계산했습니다. '+
        '픽셀 중심 변환, border 처리, 지원점 mask, 채널별 정렬 평균을 확인했으며 사전 고정 허용오차는 atol=rtol=2e-6입니다.', '',
        '| 모델 | 후보 수 | descriptor 최대 절대차 | 서로 다른 두 가설 행 | raw 차이 L2 중앙값 | 정규화 차이 L2 중앙값 | 변동 채널 수 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m in MODELS:
        v=r['models'][m];s=v['input_statistics'];c=v['comparison']
        lines.append(f"| {m} | {c['checked_candidates']} | {c['max_absolute']:.12g} | {s['hypothesis_descriptor_different_rows']} | {s['hypothesis_raw_L2']['median']:.9g} | {s['hypothesis_normalized_L2']['median']:.9g} | {s['varying_channel_count']} |")
    lines += ['', '가설 간 차이와 채널 변동은 입력의 구별 가능성만 보여 줍니다. 실제 물리 T/R 방향이나 안전한 개선을 구별한다는 뜻은 아닙니다. '+
        '원래 후보 valid mask와 0지원 후보를 유지했으며, 지원점은 가시성 정답이 아닙니다.', '',
        'NPY 파일 전체 SHA, 유효 프레임 2,597개의 token SHA, 실패 프레임의 0 token, 모든 NPZ 배열 hash와 R0 유효 후보 5,194개의 FP32 mean/std를 확인했습니다. '+
        '원본 RGB·모델 가중치·목표 오류는 읽지 않았고 backbone을 재실행하지 않았습니다. 따라서 이 검산은 저장된 token 이후의 입력 구성과 기록된 provenance를 검증하며, backbone 추론 자체를 독립 재현한 검산은 아닙니다.', '',
        '공유 source feature/pose/metadata 캐시는 다른 split도 포함하지만 수치 검산에는 봉인된 eligible TRAIN 행만 사용했습니다.', '',
        '[전체 영수증](INPUT_VERIFICATION.json) · [입력 설계 검토](FEATURE_DESIGN_REVIEW_KO.md) · '+
        '[독립 검산 코드](../../../scripts/research/'+NAME+'/verify_train_inputs.py)', '']
    return '\n'.join(lines)


def selfcheck():
    torch.set_num_threads(1)
    xx,yy=np.meshgrid(np.arange(42),np.arange(56))
    feature=np.broadcast_to((xx+2*yy)[None],(384,56,42)).astype(np.float16).copy()
    token_xy=np.array([[0,0],[41,55],[12.25,20.5],[19,31]],np.float64)
    crop=(token_xy+.5)*[576/42,768/56]-.5
    sampled=sample_double(torch.from_numpy(feature.astype(np.float64))[None],crop,np.ones(4,bool))
    np.testing.assert_allclose(sampled[:,0],token_xy[:,0]+2*token_xy[:,1],atol=1e-12,rtol=0)
    for i in (0,1,3):
        np.testing.assert_allclose(sampled[i],feature[:,int(token_xy[i,1]),int(token_xy[i,0])].astype(np.float64),atol=1e-12,rtol=0)
    boundary=np.array([[0,0],[576-1e-6,768-1e-6],[-1,100],[100,768]],np.float64)
    support=np.array([1,1,0,0],bool)
    edge=sample_double(torch.from_numpy(feature.astype(np.float64))[None],boundary,support)
    assert not edge[2:].any()
    np.testing.assert_array_equal(edge[0],feature[:,0,0])
    np.testing.assert_array_equal(edge[1],feature[:,-1,-1])
    rng=np.random.default_rng(20261001656385)
    samples=rng.normal(size=(8,384));mask=np.array([1,0,1,1,0,1,1,1],bool);samples[~mask]=0.
    reference=ordered_mean_double(samples,mask)
    for _ in range(20):
        permutation=rng.permutation(8)
        np.testing.assert_array_equal(reference,ordered_mean_double(samples[permutation],mask[permutation]))
    assert not ordered_mean_double(np.zeros((8,384)),np.zeros(8,bool)).any()
    pose=dict(available=True,cf_extents=[2.,2.,2.],R_cf=np.eye(3).tolist(),centroid=[0.,0.,5.])
    K=np.array([[100.,0.,288.],[0.,100.,384.],[0.,0.,1.]])
    uv,crop,support=project(pose,K,np.eye(3),[768,576])
    for i,sign in enumerate(SIGNS):
        np.testing.assert_array_equal(uv[i],[288.+100*sign[0]/(5+sign[2]),384.+100*sign[1]/(5+sign[2])])
    assert support.all() and np.array_equal(uv,crop)
    rotated=dict(pose,R_cf=np.diag([-1.,1.,-1.]).tolist())
    uv2,crop2,support2=project(rotated,K,np.eye(3),[768,576])
    permutation=[5,4,7,6,1,0,3,2]
    np.testing.assert_array_equal(uv2,uv[permutation])
    s1=sample_double(torch.from_numpy(feature.astype(np.float64))[None],crop,support)
    s2=sample_double(torch.from_numpy(feature.astype(np.float64))[None],crop2,support2)
    np.testing.assert_array_equal(ordered_mean_double(s1,support),ordered_mean_double(s2,support2))
    for unavailable in (dict(available=False),dict(pose,centroid=[0.,0.,-2.])):
        _,_,v=project(unavailable,K,np.eye(3),[768,576]);assert not v.any()
    np.testing.assert_allclose(crop_matrix([0.,0.,288.,384.]),[[.8,0.,172.8],[0.,.8,230.4],[0.,0.,1.]],atol=1e-12,rtol=0)
    r0_values=np.zeros((2,2,385),np.float32);r0_values[0,1]=2.
    expert=r0_values.copy();expert[0]+=3.
    valid=np.array([[1,1],[0,0]],bool);anchor=np.array([1,-1],np.int64)
    stats=input_statistics(expert,valid,anchor,np.zeros(385,np.float32),np.ones(385,np.float32),r0_values)
    np.testing.assert_allclose(stats['valid_candidate_anchor_difference_L2']['maximum'],3*math.sqrt(385),atol=1e-12,rtol=0)
    assert stats['valid_candidate_anchor_difference_L2']['minimum']>0.
    compare(np.zeros(385),np.zeros(385))
    try:
        compare(np.ones(385)*1e-4,np.zeros(385))
    except AssertionError:
        pass
    else:
        raise AssertionError('Changed descriptor escaped fixed tolerance')
    return dict(PASS=True,invented_only=True,analytic_linear_grid=True,border_no_wrap=True,
        exact_permutation=True,C2=True,unavailable=True,all_experts_use_R0_anchor=True,fixed_tolerance_fault_rejected=True,
        actual_files_read=0,images_read=0,weights_read=0,forwards=0,fits=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--selfcheck',action='store_true')
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    assert args.selfcheck != args.write, 'Choose --selfcheck or --write explicitly'
    if args.selfcheck:
        print(json.dumps(selfcheck(),ensure_ascii=False,indent=2))
    else:
        run()
