"""Publish the completed TRAIN-only DINO input audit; no model or target access.

The producer and independent input verification must finish before --write.
This renderer reads compact frozen input arrays and six existing synthetic RGBs.
It never loads retained patch tokens, model weights, GT labels, or pose metrics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_pose_dino_input_audit_20261001_v1'
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
PREVIOUS = ROOT / '_docs/experiments/pallet_pose_signed_axes_asymmetric_20261001_v1'
MODELS = ('R0', 'DIVERSE251_s1', 'DIVERSE251_s2', 'DIVERSE251_s3')
HYP = ('long-face-front', 'short-face-front')
SIGNS = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], np.float64)
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def verify(binding):
    actual = bind(ROOT / binding['path'])
    assert actual['sha256'] == binding['sha256'], binding['path']
    if 'bytes' in binding:
        assert actual['bytes'] == binding['bytes'], binding['path']


def read(path):
    return json.loads(Path(path).read_text())


def bound_read(binding):
    verify(binding)
    return read(ROOT / binding['path'])


def array_sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def install_guard(allowed):
    allowed = {Path(p).resolve() for p in allowed}
    outputs = {DOC / 'REPORT_KO.md', DOC / 'REPORT_DATA.json'}
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(ROOT):
            return
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing:
            assert path in outputs or path.is_relative_to(DOC / 'figures'), ('REPORT_WRITE_DENIED', str(path))
        elif path.suffix not in ('.py', '.pyc'):
            assert path in allowed or path in outputs or path.is_relative_to(DOC / 'figures'), ('REPORT_READ_DENIED', str(path))
    sys.addaudithook(hook)


def project(pose, K, matrix, hw):
    """Projection of the saved pose only: no solve, target, or candidate choice."""
    uv = np.zeros((8,2), np.float64)
    crop = np.zeros((8,2), np.float64)
    supported = np.zeros(8, bool)
    if not pose['available']:
        return uv, crop, supported
    ext, rotation, translation = (np.asarray(pose[k], np.float64) for k in ('cf_extents','R_cf','centroid'))
    K, matrix = np.asarray(K, np.float64), np.asarray(matrix, np.float64)
    xyz = (SIGNS * (ext / 2)) @ rotation.T + translation
    positive = xyz[:,2] > 0
    p = xyz[positive] @ K.T
    uv[positive] = p[:,:2] / p[:,2,None]
    finite = positive & np.isfinite(uv).all(1)
    crop[finite] = np.column_stack([uv[finite], np.ones(int(finite.sum()))]) @ matrix[:2].T
    supported = finite & np.isfinite(crop).all(1)
    supported &= (uv >= 0).all(1) & (uv < [hw[1],hw[0]]).all(1)
    supported &= (crop >= 0).all(1) & (crop < [576,768]).all(1)
    return uv, crop, supported


def load():
    paths = dict(protocol=DOC/'INPUT_PROTOCOL.json', input_receipt=DOC/'TRAIN_APPEARANCE_INPUTS.json',
                 input_verification=DOC/'INPUT_VERIFICATION.json', cache_contract=DOC/'CACHE_CONTRACT.json',
                 preprocessing_pure_check=DOC/'PREPROCESSING_PURE_CHECK.json',
                 padding_support_audit=DOC/'PADDING_SUPPORT_AUDIT.json',
                 previous_Q_source_gate=PREVIOUS/'SOURCE_VAL_GATE.json')
    initial = {k: read(p) for k,p in paths.items()}
    p, r, v, cache, old = (initial[k] for k in ('protocol','input_receipt','input_verification','cache_contract','previous_Q_source_gate'))
    assert p['train_only'] and p['frames'] == 2598 and p['output_dim'] == 385
    assert p['no_fit'] and p['no_label_values'] and p['no_VAL_features'] and p['no_real_features']
    assert r['complete'] and r['input_construction_pass'] and r['source_TRAIN_only']
    assert v['complete'] and v['PASS'] and v['source_TRAIN_only']
    assert v['protocol'] == r['protocol'] == bind(paths['protocol'])
    assert v['input_receipt']['sha256'] == bind(paths['input_receipt'])['sha256']
    assert v['descriptors'] == r['descriptors'] and v['tokens'] == r['tokens']
    assert v['frames'] == r['frames'] == 2598 and r['image_forwards'] == 2597
    assert r['original_allinvalid_rows'] == v['all_invalid_rows'] == 1
    assert v['valid_candidate_descriptors'] == 20776 and r['normalization']['count'] == 5194
    assert r['fits_executed'] == r['new_PnP_solves'] == v['new_fits'] == v['target_values_read'] == 0
    assert not any(r[k] for k in ('source_label_values_read','source_VAL_features_extracted','real_features_extracted'))
    assert not v['method_success'] and not v['goal_complete'] and not v['performance_improvement_measured']
    assert not v['backbone_forward_recomputed'] and not v['backbone_inference_correctness_independently_verified']
    assert cache['complete'] and cache['PASS']
    assert old['complete'] and old['checks_total'] == 45 and old['checks_passed'] == 43 and not old['PASS']
    assert not old['real_routing_authorized']
    padding = initial['padding_support_audit']
    assert padding['complete'] and padding['PASS'] and padding['frames'] == 2598
    assert padding['protocol'] == r['protocol'] and padding['descriptors'] == r['descriptors']
    assert padding['metadata'] == p['inputs']['metadata'] and padding['poses'] == p['inputs']['poses']
    assert not any(padding[k] for k in ('original_support_mask_changed','descriptor_values_changed','normalization_changed','performance_improvement_measured'))
    inputs = {k: bind(path) for k,path in paths.items()}
    inputs.update(metadata=p['inputs']['metadata'], poses=p['inputs']['poses'], descriptors=r['descriptors'],
                  source_contract=p['inputs']['source_contract'])
    # Bootstrap source metadata is input-only. The whitelist below admits only
    # the six deterministic synthetic TRAIN images subsequently selected.
    metadata = bound_read(inputs['metadata'])
    with np.load(ROOT/r['descriptors']['path'], allow_pickle=False) as z:
        ids = z['ids'].copy(); indices = z['source_index'].copy(); anchors = z['anchor_index'].copy()
    selected = [metadata[int(i)] for i in indices[:6]]
    allowed = [Path(__file__), *paths.values(), ROOT/v['input_receipt']['path'],
               *[ROOT/b['path'] for b in inputs.values()], *[ROOT/row['image']['path'] for row in selected]]
    install_guard(allowed)
    for b in inputs.values():
        verify(b)
    verify(v['input_receipt'])
    contract = bound_read(inputs['source_contract'])
    eligible = set(contract['fit_eligibility']['eligible_ids']['TRAIN'])
    expected = np.array([i for i,row in enumerate(metadata) if row['id'] in eligible],np.int64)
    np.testing.assert_array_equal(indices, expected)
    assert len(ids) == len(indices) == len(eligible) == 2598
    assert ids.tolist() == [metadata[int(i)]['id'] for i in indices]
    assert all(metadata[int(i)]['split'] == 'TRAIN' for i in indices)
    with np.load(ROOT/r['descriptors']['path'], allow_pickle=False) as z:
        # Features and patch tokens themselves are not needed to render this report.
        arrays = {key:z[key].copy() for key in ('ids','source_index','anchor_index','crop_matrices')}
        for m in MODELS:
            arrays[m+'_valid'] = z[m+'_valid'].copy()
            arrays[m+'_support8'] = z[m+'_support8'].copy()
    for key,value in arrays.items():
        assert array_sha(value) == v['array_hashes'][key]
    histograms = {}
    differences = {}
    for m in MODELS:
        valid = arrays[m+'_valid']; support = arrays[m+'_support8']
        assert int(valid.sum()) == 5194 and valid.shape == (2598,2)
        h = {str(k):int((support.sum(2)[valid] == k).sum()) for k in range(9)}
        assert h == r['models'][m]['supported_corner_histogram'] == v['models'][m]['support_histogram']
        histograms[m] = h
        differences[m] = v['models'][m]['input_statistics']
        assert not differences[m]['performance_metric'] and not differences[m]['targets_used']
    core = dict(frames=2598,image_forwards=2597,original_allinvalid_rows=1,
        valid_candidates_total=20776,valid_candidates_per_model=5194,descriptor_dim=385,
        normalization_count=5194,new_fits=0,new_PnP_solves=0,new_T_R_evaluations=0,
        source_VAL_features_extracted=False,real_features_extracted=False,
        stable_joint_improvement_achieved=False)
    coverage = {k:cache['summary'][k] for k in ('all5120','eligibleTRAIN2598','VAL1024')}
    for key,n,c in (('all5120',5120,710),('eligibleTRAIN2598',2598,376),('VAL1024',1024,140)):
        assert coverage[key]['rows'] == n and coverage[key]['wide_image_identity_matches'] == c
        assert coverage[key]['strict_reusable'] == 0
    return dict(initial=initial,inputs=inputs,arrays=arrays,metadata=metadata,
                poses=bound_read(inputs['poses']),core=core,support_histograms=histograms,
                descriptor_differences=differences,cache_coverage=coverage)


def support_figure(data):
    import matplotlib.pyplot as plt
    h = np.array([[data['support_histograms'][m][str(k)] for k in range(9)] for m in MODELS])
    fig,ax = plt.subplots(figsize=(12,4.5),layout='constrained')
    im = ax.imshow(np.log1p(h),cmap='cividis',aspect='auto',vmin=0,vmax=np.log1p(5194))
    ax.set_xticks(range(9),range(9));ax.set_yticks(range(4),['R0','DIVERSE s1','DIVERSE s2','DIVERSE s3'])
    ax.set_xlabel('Supported projected corners out of 8 (geometric support, not visibility)')
    ax.set_title('Frozen synthetic TRAIN inputs | 5,194 original valid candidates per expert\nNo training or T/R performance evaluation')
    for i in range(4):
        for j in range(9):
            ax.text(j,i,f'{h[i,j]:,}',ha='center',va='center',color='white' if h[i,j]<200 else 'black',fontsize=11)
    colorbar=fig.colorbar(im,ax=ax,pad=.02)
    ticks=np.array([0,5,50,500,5194]);colorbar.set_ticks(np.log1p(ticks),labels=[str(x) for x in ticks])
    colorbar.set_label('Candidate count (log1p colour scale)')
    path=DOC/'figures/support_histogram.png';fig.savefig(path,dpi=160);plt.close(fig)
    return bind(path)


def gallery(data):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    arrays,metadata,poses=data['arrays'],data['metadata'],data['poses']
    rows=[]
    for j in range(6):
        index=int(arrays['source_index'][j]);row=metadata[index];fid=str(arrays['ids'][j])
        assert row['id']==fid and row['split']=='TRAIN'
        k=int(arrays['anchor_index'][j]);record=poses['records']['R0'][fid]
        assert k in (0,1) and arrays['R0_valid'][j,k] and record['GEO_name']==HYP[k]
        pose=record['GEO_pose'];hyp={x['name']:x['pose'] for x in record['hypotheses']}
        assert pose==hyp[HYP[k]] and pose['available']
        matrix=arrays['crop_matrices'][j]
        uv,crop,support=project(pose,row['K'],matrix,row['hw'])
        np.testing.assert_array_equal(support,arrays['R0_support8'][j,k])
        pad=int(row['pad']);h,w=row['hw'];bounds=[pad,pad,w-pad,h-pad]
        in_original=(uv[:,0]>=pad)&(uv[:,0]<w-pad)&(uv[:,1]>=pad)&(uv[:,1]<h-pad)
        verify(row['image'])
        with Image.open(ROOT/row['image']['path']) as image:
            assert [image.height,image.width]==row['hw']
        rows.append(dict(id=fid,source_index=index,split='TRAIN',source='synthetic TRAIN RGB',
            image=row['image'],image_hw=row['hw'],dimensions_m=row['dims'],
            dims_cm=(np.asarray(row['dims'],np.float64)*100).tolist(),K=row['K'],
            physical_dimension_order='Width, Height, Depth',projection_extents='saved cf_extents, not physical W/H/D reordered',
            anchor_index=k,anchor_name=HYP[k],anchor_pose=pose,crop_matrix=matrix.tolist(),
            projected8=uv.tolist(),crop_points8=crop.tolist(),support8=support.tolist(),
            corner_signs=SIGNS.tolist(),edges=[list(x) for x in EDGES],additional_padding=0,
            pad=pad,original_image_bounds_xyxy=bounds,padding_supported8=(support&~in_original).tolist(),
            GT_overlay=False,new_pose_estimate=False,physical_pose_errors_shown=False))
    artifacts=[]
    for page in range(3):
        fig,axes=plt.subplots(2,2,figsize=(15,12))
        fig.subplots_adjust(top=.86,bottom=.045,left=.015,right=.985,hspace=.27,wspace=.035)
        for i,row in enumerate(rows[2*page:2*page+2]):
            with Image.open(ROOT/row['image']['path']) as image:
                rgb=np.asarray(image.convert('RGB'))
            dims=row['dims_cm'];title=f"{row['id']} | W {dims[0]:.2f} / H {dims[1]:.2f} / D {dims[2]:.2f} cm"
            for col in range(2):
                ax=axes[i,col];ax.imshow(rgb);ax.set_xlim(-.5,rgb.shape[1]-.5);ax.set_ylim(rgb.shape[0]-.5,-.5);ax.axis('off')
                ax.set_title(title+'\n'+('Prepared synthetic RGB (additional padding: 0)' if col==0 else 'Existing R0 anchor: '+row['anchor_name']),fontsize=10)
                x0,y0,x1,y1=row['original_image_bounds_xyxy']
                ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#e900dc',linestyle='--',linewidth=1.8,label='Original image boundary; reflected padding outside'))
            ax=axes[i,1];uv=np.asarray(row['projected8']);support=np.asarray(row['support8'],bool)
            for a,b in EDGES:
                ax.plot(uv[[a,b],0],uv[[a,b],1],color='#00cdeb',lw=1.4,alpha=.75)
            ax.scatter(uv[support,0],uv[support,1],s=45,color='#ffdf00',edgecolors='black',linewidths=.6,label='Supported projected corner')
            if (~support).any():
                ax.scatter(uv[~support,0],uv[~support,1],s=45,color='#ef476f',marker='x',label='Unsupported projected corner')
            for n,point in enumerate(uv):
                if 0<=point[0]<rgb.shape[1] and 0<=point[1]<rgb.shape[0]:
                    ax.text(point[0]+4,point[1]-4,str(n),color='white',fontsize=9,bbox=dict(facecolor='black',alpha=.6,pad=.6))
            ax.legend(loc='upper left',fontsize=8,framealpha=.9)
        fig.suptitle('SYNTHETIC TRAIN RGB — FIXED INPUT ILLUSTRATIONS\nFirst six eligible TRAIN rows; no performance-based selection\nMagenta dashed box: original image; reflected padding outside\nPrepared-canvas support is not visibility | No GT or new T/R result',fontsize=14,y=.99)
        path=DOC/'figures'/f'synthetic_train_inputs_{page+1}.jpg'
        fig.savefig(path,dpi=130,pil_kwargs={'quality':93});plt.close(fig);artifacts.append(bind(path))
    return artifacts,rows


def render(data,rows):
    r=data['initial']['input_receipt'];v=data['initial']['input_verification']
    lines=['# DINO 외형 입력의 TRAIN 감사','',
        '**새 모델 학습 0회 · 새 T/R 성능 평가 0회 · 실사 실행 0회. 원래의 안정적인 T/R 동시 개선 목표는 아직 달성하지 못했다.**', '',
        '이번 단계는 고정된 합성 TRAIN RGB와 기존 후보 pose에서 외형 특징을 만들고, 저장된 입력을 독립적으로 검산했다. 입력 감사의 PASS는 개선 성능의 PASS가 아니다. 직전 비대칭 손실 모델(Q)의 source 판정은 **43/45 조건 통과, 전체 FAIL(2개 실패)**로 그대로이며, 이번 입력으로 재학습하거나 새 후보 선택 정책을 평가하지 않았다.','',
        f"적격 TRAIN **{r['frames']:,}행**을 전부 유지했다. 기존 후보가 모두 실패한 1행은 0으로 남겼고, 고정 DINO backbone 순전파는 **{r['image_forwards']:,}회**였다. 4개 expert의 유효 후보는 합계 **20,776개**이며, 새 pose 추정·PnP·GT 오류 계산은 없었다.", '',
        '## 기존 캐시 재사용 감사','',
        '기존 diversity 캐시 8,505개는 source TRAIN 8,192개·source held 64개·실사 249개로 구성되어 있다. 아래는 tail 캐시까지 합친 wide-cache 합집합과 현재 source 집합의 대응이다. 실사 캐시값과 GT 배열은 이번 감사에서 읽지 않았다.','',
        '| 현재 집합 | 전체 행 | 이미지 ID·SHA 일치 | 현재 crop 행렬까지 정확히 일치 |',
        '|---|---:|---:|---:|']
    for key,label in [('all5120','고정 source 전체'),('eligibleTRAIN2598','적격 TRAIN'),('VAL1024','source VAL')]:
        c=data['cache_coverage'][key];lines.append(f"| {label} | {c['rows']:,} | {c['wide_image_identity_matches']:,} | {c['strict_reusable']} |")
    lines += ['', '이미지가 같아도 기존 역변환 box로 만든 crop과 현재 고정 R0 box의 crop 행렬이 정확히 같지 않았다. 토큰 격자와 전처리까지 동일해야 한다는 재사용 계약에 따라 기존 토큰은 재사용하지 않았다. 이 차이가 성능을 바꾼다는 결론은 내리지 않는다. VAL 140행은 캐시 식별·입력 메타데이터 대응만 확인했으며, 새 VAL 특징이나 품질 수치는 계산하지 않았다.', '',
        '## 고정 입력 구성과 지원점 분포','',
        '현재 R0의 선택 box로 만든 동일한 wide crop(576×768)을 모든 후보가 공유한다. 고정 DINOv2 ViT-S/14의 384×56×42 FP16 token 격자에서 기존 pose의 8개 모서리 투영 위치를 bilinear 방식으로 읽는다. 양의 깊이·준비된 이미지 경계·crop 경계 안의 점만 지원점이며, 이는 **가시성·가림 여부의 정답이 아니다**. 중심점은 사용하지 않는다.','',
        '지원점의 384개 채널을 채널별 정렬 후 평균하고 지원점 수/8을 붙여 385차원을 만든다. 정렬은 코너 순서에 따른 부동소수점 합산 차이를 막는다. R0의 원래 유효 TRAIN 후보 5,194개로만 FP32 mean/std(최솟값 1e-6)를 고정했다. 기존 candidate valid mask는 바꾸지 않았다.','',
        '| Expert | 유효 후보 | 지원점 0–5개 | 6개 | 7개 | 8개 |',
        '|---|---:|---:|---:|---:|---:|']
    for m in MODELS:
        h=data['support_histograms'][m]
        lines.append(f"| {m} | 5,194 | {sum(h[str(k)] for k in range(6))} | {h['6']} | {h['7']} | {h['8']:,} |")
    lines += ['', '![전체 유효 후보의 지원점 수](figures/support_histogram.png)', '',
        '### 준비된 이미지의 반사 padding: 학습 전 입력 계약 보완 필요','',
        '현재 지원점 검사는 원영상이 아니라 이미 사방 100 px 반사 padding을 포함한 prepared canvas를 기준으로 한다. 지원점 위치가 원래 영상 영역 밖의 반사 띠에 들어가는 경우가 확인되었다. 아래 숫자는 현재 입력을 바꾸지 않고 집계한 값이다.','',
        '| Expert | 지원점 전체 | 반사 padding 내 지원점 | 영향을 받는 후보 / 5,194 |',
        '|---|---:|---:|---:|']
    for m in MODELS:
        s=data['initial']['padding_support_audit']['models'][m]
        lines.append(f"| {m} | {s['supported_points']:,} | {s['padding_supported_points']:,} | {s['candidates_with_padding_support']:,} |")
    lines += ['', '다음 학습은 원본 영상 영역을 구분한 입력 수정·독립 검산 후 진행한다. 기존 pad 메타데이터로 원영상 밖의 샘플 위치를 제외하는 계약을 먼저 고정하고, 보존된 token으로 다시 pooling한다. 이 단계는 입력 계약 보완이며 GT·성능 문턱 탐색이 아니다. 이 보고서의 현재 descriptor·지원 mask·정규화는 수정하지 않았다. 원영상 안에서 읽는 token도 receptive field를 통해 padding 문맥의 영향을 받을 수 있어, 위치 제외만으로 문맥 영향까지 제거된다고 보장할 수 없다.','',
        '## 독립 검산과 두 가설의 입력 차이','',
        '독립 검산은 저장된 token을 한 프레임씩 읽고 별도 scalar 투영과 Torch CPU float64 grid_sample로 20,776개 descriptor를 다시 구성했다. 사전 허용오차는 atol=rtol=2e-6이다. 모든 token 파일·프레임 hash, 유효성, 지원점, 정규화를 확인했다. **backbone 순전파와 RGB 전처리 전체를 독립적으로 다시 실행한 검산은 아니다.**', '',
        '| Expert | 서로 다른 W/D 가설 입력 행 / 2,597 | raw 차이 L2 중앙값 | 정규화 차이 L2 중앙값 | 검산 최대 절대차 |',
        '|---|---:|---:|---:|---:|']
    for m in MODELS:
        s=data['descriptor_differences'][m];c=v['models'][m]['comparison']
        lines.append(f"| {m} | {s['hypothesis_descriptor_different_rows']:,} | {s['hypothesis_raw_L2']['median']:.9g} | {s['hypothesis_normalized_L2']['median']:.9g} | {c['max_absolute']:.9g} |")
    lines += ['', 'W/D는 고정 long-face-front/short-face-front 가설을 뜻한다. 위 값은 각 expert 내부 두 가설의 입력 차이이며, 서로 다른 expert 사이의 성능 비교가 아니다. 입력이 달라진다는 사실만으로 올바른 T/R 방향이나 안전한 개선 후보를 구별할 수 있다고 결론 내릴 수 없다. 정답 오류·새 selector 점수·학습 가중치는 읽지 않았다.', '',
        '## 합성 TRAIN RGB 6장과 물리 치수','',
        '정렬된 성능으로 고르지 않고, 봉인된 적격 TRAIN 배열의 **첫 6행**을 그대로 사용했다. 왼쪽은 원래 준비된 합성 RGB, 오른쪽은 기존 R0 anchor pose의 8개 모서리와 입력 지원점이다. 자홍색 점선 안은 원영상이고 바깥은 기존 반사 padding이다. 추가 padding은 0이며, GT 윤곽·T/R 오류·새 모델 결과는 표시하지 않는다. 치수는 입력 메타데이터의 물리 W/H/D(m)에 100을 곱한 cm이고, 투영에는 별도로 저장된 camera-facing extents를 사용한다.', '',
        '| TRAIN ID | source index | W (cm) | H (cm) | D (cm) |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        d=row['dims_cm'];lines.append(f"| {row['id']} | {row['source_index']} | {d[0]:.2f} | {d[1]:.2f} | {d[2]:.2f} |")
    lines += ['', '![합성 TRAIN 입력 1–2](figures/synthetic_train_inputs_1.jpg)', '',
        '![합성 TRAIN 입력 3–4](figures/synthetic_train_inputs_2.jpg)', '',
        '![합성 TRAIN 입력 5–6](figures/synthetic_train_inputs_3.jpg)', '',
        '## 재현 범위와 증거','',
        '공유 source metadata·pose·feature 컨테이너에는 VAL 행도 들어 있지만, 새 descriptor 생성과 수치 검산에는 적격 TRAIN 행만 사용했다. 기존 캐시 대응 감사의 VAL 입력 메타데이터 확인과 새로운 VAL 특징 추출은 구분한다. 공개 그림은 모두 합성 TRAIN 입력이며 실사 결과가 아니다.', '',
        f"전체 FP16 token 파일은 로컬에 보존했다({r['tokens']['bytes']:,} bytes). 약 4.7 GB token, 원본 RGB, NPZ와 backbone 가중치는 GitHub 공개 번들에 넣지 않는다. 공개 영수증에는 경로·SHA가 남지만, 전체 재현에는 해당 로컬 자료가 필요하다. 이 보고서 생성은 저장 token이나 backbone을 다시 읽거나 실행하지 않는다.", '',
        '[입력 프로토콜](INPUT_PROTOCOL.json) · [캐시 계약 감사](CACHE_CONTRACT.json) · [추출 영수증](TRAIN_APPEARANCE_INPUTS.json) · [독립 입력 검산](INPUT_VERIFICATION_KO.md) · [독립 검산 JSON](INPUT_VERIFICATION.json) · [전처리 순수 검산](PREPROCESSING_PURE_CHECK.json) · [반사 padding 감사](PADDING_SUPPORT_AUDIT.json)', '',
        '[보고서 수치·그림 provenance](REPORT_DATA.json) · [입력 설계 검토](FEATURE_DESIGN_REVIEW_KO.md) · [선행 방법 검토](PRIOR_METHODS_KO.md) · [공개 검토](PUBLIC_REVIEW_KO.md) · [공개 파일 목록](PUBLICATION_MANIFEST.json)', '',
        f'[재현 코드](../../../scripts/research/{NAME}/report.py) · [직전 Q 결과](../pallet_pose_signed_axes_asymmetric_20261001_v1/REPORT_KO.md)', '']
    assert len(lines) <= 130, len(lines)
    return '\n'.join(lines)


def write():
    assert not (DOC/'REPORT_DATA.json').exists() and not (DOC/'REPORT_KO.md').exists(), 'Frozen report already exists'
    data=load();(DOC/'figures').mkdir(parents=True,exist_ok=True)
    support=support_figure(data);photos,rows=gallery(data)
    text=render(data,rows)
    with (DOC/'REPORT_KO.md').open('x') as f:
        f.write(text)
    result=dict(complete=True,status='INPUT_AUDIT_ONLY_NOT_A_PERFORMANCE_RESULT',code=bind(Path(__file__)),
        inputs=data['inputs'],core=data['core'],support_histograms=data['support_histograms'],
        cache_coverage=data['cache_coverage'],descriptor_differences=data['descriptor_differences'],
        padding_support_audit=data['inputs']['padding_support_audit'],
        padding_support_summary=data['initial']['padding_support_audit']['models'],
        current_descriptor_ready_for_training=False,padding_contract_correction_before_fit=True,
        previous_Q_source_gate=dict(checks_total=45,checks_passed=43,PASS=False,unchanged=True),
        gallery=rows,gallery_selection='First six rows of frozen eligible TRAIN ids/source_index; no outcome-based selection',
        figures=[support,*photos],report=bind(DOC/'REPORT_KO.md'),
        source_TRAIN_only=True,synthetic_RGB_images=6,actual_RGB_images=6,new_fits=0,new_PnP_solves=0,
        report_image_forwards=0,new_T_R_evaluations=0,policy_selections=0,
        method_success=False,goal_complete=False,performance_improvement_measured=False,
        original_candidate_validity_preserved=True,source_VAL_features_extracted=False,real_features_extracted=False,
        image_type='synthetic source TRAIN; not real-world evaluation',
        report_reads_retained_tokens=False,report_reads_model_weights=False,report_reads_target_values=False,
        private_reproduction=dict(tokens=data['initial']['input_receipt']['tokens'],
            descriptors=data['initial']['input_receipt']['descriptors'],public_copy=False),
        shared_cache_disclosure=data['initial']['input_verification']['shared_cache_disclosure'],
        independent_verification_limit=data['initial']['input_verification']['inference_limit'])
    with (DOC/'REPORT_DATA.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,default=serial,allow_nan=False);f.write('\n')
    print('DINO_INPUT_REPORT_COMPLETE',json.dumps(dict(report=bind(DOC/'REPORT_KO.md'),data=bind(DOC/'REPORT_DATA.json'),figures=4,synthetic_RGB_images=6,new_fits=0),ensure_ascii=False),flush=True)


def selfcheck():
    pose=dict(available=True,cf_extents=[2.,2.,2.],R_cf=np.eye(3).tolist(),centroid=[0.,0.,5.])
    K=np.array([[100.,0.,288.],[0.,100.,384.],[0.,0.,1.]])
    uv,crop,support=project(pose,K,np.eye(3),[768,576])
    expected=np.array([[288+100*s[0]/(5+s[2]),384+100*s[1]/(5+s[2])] for s in SIGNS])
    np.testing.assert_array_equal(uv,expected);np.testing.assert_array_equal(crop,uv);assert support.all()
    _,_,unsupported=project(dict(available=False),K,np.eye(3),[768,576]);assert not unsupported.any()
    return dict(PASS=True,invented_only=True,actual_input_reads=0,image_reads=0,forwards=0,fits=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write',action='store_true');mode.add_argument('--selfcheck',action='store_true');args=parser.parse_args()
    if args.write:
        write()
    else:
        print(json.dumps(selfcheck()))
