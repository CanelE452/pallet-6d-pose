"""Visualize the six frozen RGB/mask/outline cases without new observations.

The first phase deliberately omits candidate names, success flags, errors, and
costs.  The scenes were known before this audit; this is not independent blind
validation.  RGB contrast copies are for display only and never feed numerics.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x')as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def _graphics():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    font=Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False
    return plt


def contour_points(mask):
    """Display-only contours of an existing binary mask; no image detector."""
    import cv2
    a=np.asarray(mask)
    if not np.isin(a,[0,1,False,True]).all():
        raise ValueError('Do not choose a new threshold for a nonbinary mask')
    found,_=cv2.findContours(a.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1,2)for c in found]


def _canvas(ax,rgb,roi,title):
    h,w=rgb.shape[:2]
    ax.imshow(rgb,extent=(-.5,w-.5,h-.5,-.5),interpolation='nearest')
    x0,y0,x1,y1=roi;ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5)
    ax.set_aspect('equal');ax.set_title(title,fontsize=10);ax.tick_params(labelsize=7)
    ax.set_xlabel('원영상 x [px]',fontsize=8);ax.set_ylabel('원영상 y [px]',fontsize=8)


def _mask_canvas(ax,mask,roi,title):
    h,w=np.asarray(mask).shape
    ax.imshow(mask,cmap='gray',vmin=0,vmax=1,extent=(-.5,w-.5,h-.5,-.5),interpolation='nearest')
    x0,y0,x1,y1=roi;ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5)
    ax.set_aspect('equal');ax.set_title(title,fontsize=10);ax.tick_params(labelsize=7)
    ax.set_xlabel('원영상 x [px]',fontsize=8);ax.set_ylabel('원영상 y [px]',fontsize=8)


def _save(plt,fig,path):
    fig.savefig(path,dpi=120,pil_kwargs={'quality':89,'optimize':True});plt.close(fig)


def _contours(ax,contours,color='#f2bd27',lw=.9):
    for c in contours:
        q=np.vstack((c,c[:1]));ax.plot(q[:,0],q[:,1],c=color,lw=lw)


def display_roi(case,arrays,image_hw):
    """Display-only expansion includes the full stored mask/outline, no scores."""
    h,w=image_hw;original=list(map(int,case['crop_xyxy']))
    points=[np.asarray([[original[0],original[1]],[original[2]-1,original[3]-1]],float),
            np.asarray(case['outline']['polygon_xy'],float)]
    for key in('pallet_mask_raw','pallet_mask_clean','pallet_mask_tight'):
        y,x=np.nonzero(np.asarray(arrays[key]))
        if len(x):points.append(np.array([[x.min(),y.min()],[x.max(),y.max()]],float))
    q=np.concatenate(points);pad=24
    return [max(0,int(np.floor(q[:,0].min()))-pad),max(0,int(np.floor(q[:,1].min()))-pad),
            min(w,int(np.ceil(q[:,0].max()))+pad+1),min(h,int(np.ceil(q[:,1].max()))+pad+1)]


def _render_initial(source_root,score_private,state_private,output,*,runtime_guard):
    from PIL import Image
    from scripts.research.pallet_minimal_bi_ablation_v1.gallery import public_image_inventory
    from .bindings import load_cases,load_mask_arrays
    started,cpu=time.monotonic(),time.process_time()
    root,out=Path(source_root),Path(output)
    if out.exists():raise FileExistsError('Preserve every initial-review rendering attempt')
    cases=load_cases(root,Path(score_private),Path(state_private))
    permission=public_image_inventory(root)
    plt=_graphics();out.mkdir(parents=True);proof=[];articles=[]
    for case in cases:
        number=case['ordinal'];ident=case['id'];source=Path(case['image_path'])
        if sha(source)!=case['image_sha256']:raise ValueError('Bound RGB changed '+ident)
        rgb=np.asarray(Image.open(source).convert('RGB'));h,w=rgb.shape[:2]
        arrays=load_mask_arrays(case)
        for k,a in arrays.items():
            if k in('pallet_mask','pallet_mask_raw','pallet_mask_clean','state_pallet_mask','pallet_mask_tight')and np.asarray(a).shape!=(h,w):
                raise ValueError('Stored mask requires an unverified transform '+k)
        roi=display_roi(case,arrays,(h,w));x0,y0,x1,y1=roi
        if not(0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Original crop out of canvas')
        actual=np.asarray(arrays['pallet_mask']);clean=np.asarray(arrays['pallet_mask_clean'])
        if not np.array_equal(actual,clean):raise ValueError('Recorded consumer-clean alias mismatch')
        raw=np.asarray(arrays['pallet_mask_raw']);raw_contours=contour_points(raw);used_contours=contour_points(actual)
        folder=out/f'case_{number:02d}';folder.mkdir()
        Image.fromarray(rgb).save(folder/'scene_RGB_unmarked.jpg',quality=92,optimize=True)
        Image.fromarray(rgb[y0:y1,x0:x1]).save(folder/'crop_RGB_unmarked.png')
        # One fixed display curve across all cases; never used to read mask,
        # compute outline/costs, classify support, or alter source bytes.
        contrast=np.rint(255*np.sqrt(rgb.astype(float)/255)).astype(np.uint8)
        Image.fromarray(contrast[y0:y1,x0:x1]).save(folder/'crop_RGB_display_gamma_half.jpg',quality=90,optimize=True)
        aspect=(y1-y0)/(x1-x0);height=max(6.,min(12.,2*(5.3*aspect+.8)))
        fig,axes=plt.subplots(2,3,figsize=(16,height),layout='constrained')
        fig.suptitle(f'사례 {number} · 원 RGB와 저장된 마스크 단계 · 오류·후보·성공 표시 없음',fontsize=12)
        _canvas(axes[0,0],rgb,roi,'원 RGB crop · 표시선 없음')
        _mask_canvas(axes[0,1],arrays['state_pallet_mask'],roi,'저장 state soft mask\n외곽 비용의 실제 입력과 분리')
        _mask_canvas(axes[0,2],raw,roi,'저장 SAM 선택 binary raw mask\n새 분할·threshold 없음')
        _mask_canvas(axes[1,0],clean,roi,'저장 clean / 실제 pallet_mask\n원 후처리 결과 · 정확한 alias')
        _mask_canvas(axes[1,1],arrays['pallet_mask_tight'],roi,'저장 tight mask\n실제 비용 입력으로 대체하지 않음')
        _canvas(axes[1,2],rgb,roi,'저장 raw/실제 clean의 contour\n표시용 파생물 · 새 영상 검출 아님')
        _contours(axes[1,2],raw_contours,'#edb721',.8);_contours(axes[1,2],used_contours,'#22d3b5',.85)
        _save(plt,fig,folder/'mask_stages.jpg')
        outline=case['outline'];polygon=np.asarray(outline['polygon_xy'],float);segments=np.asarray(outline['observed_boundary_segments_xy'],float);samples=np.asarray(outline['observed_boundary_samples_xy'],float)
        fig,axes=plt.subplots(1,3,figsize=(17,max(3.5,min(7.,5.7*aspect+.7))),layout='constrained')
        fig.suptitle(f'사례 {number} · 원 RGB 위의 경계 생성 계보 · 모델·참조·비용 미표시',fontsize=12)
        _canvas(axes[0],rgb,roi,'실제 clean mask contour\n팔레트 물리 외곽이라는 판정은 별도')
        _contours(axes[0],used_contours,'#22d3b5',1.)
        _canvas(axes[1],rgb,roi,'저장 convex hull 전체(주황 점선)\n실제 비용의 유한 선분(청록 실선)')
        closed=np.vstack((polygon,polygon[:1]));axes[1].plot(closed[:,0],closed[:,1],c='#ed9220',ls='--',lw=1.4)
        for j,segment in enumerate(segments):
            axes[1].plot(segment[:,0],segment[:,1],c='#20d4bd',lw=1.8)
            center=segment.mean(0);axes[1].annotate('E'+str(j),center,fontsize=7,color='#117d72',clip_on=True)
        _canvas(axes[2],rgb,roi,'같은 저장 observed 표본과 원 인덱스\n청록=실제 유한 관측 · 별도 위치 추정 아님')
        axes[2].scatter(samples[:,0],samples[:,1],s=3,c='#21c5af')
        for j in range(0,len(samples),max(1,len(samples)//12)):
            axes[2].annotate(str(j),samples[j],fontsize=7,color='#14a68c',clip_on=True)
        _save(plt,fig,folder/'boundary_generation.jpg')
        allowed=case['image_sha256']in permission
        proof.append(dict(id=ident,ordinal=number,source_RGB_sha256=case['image_sha256'],
            RGB_publication_inventory_match=allowed,RGB_existing_public_member=permission.get(case['image_sha256']),
            original_image_hw=[h,w],crop_original_xyxy=roi,parent_crop_original_xyxy=case['crop_xyxy'],
            display_crop_definition='Union of parent crop and entire stored binary raw/clean/tight mask and stored polygon, fixed 24px padding, clipped to original canvas; no metric/candidate/reference used.',
            crop_original_RGB_array_sha256=hashlib.sha256(np.ascontiguousarray(rgb[y0:y1,x0:x1]).tobytes()).hexdigest(),
            display_only_contrast=dict(gamma=.5,formula='round(255*sqrt(original_uint8/255))',used_in_numerics=False),
            masks={k:dict(shape=list(np.asarray(a).shape),dtype=str(np.asarray(a).dtype),array_sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())for k,a in arrays.items()},
            actual_outline_mask='pallet_mask = pallet_mask_clean exact',raw_contours=len(raw_contours),actual_contours=len(used_contours),
            stored_polygon_xy=polygon.tolist(),stored_used_segments_xy=segments.tolist(),stored_observed_samples_xy=samples.tolist(),
            no_candidate_error_cost_success_labels=True,reference_projection_not_in_this_phase=True,
            case_path=f'case_{number:02d}',numeric_inputs_changed=False))
        articles.append(f'<article id="case{number}"><h2>사례 {number} · RGB·마스크·관측 경계 우선 검토</h2><p>후보 이름·성공 표시·자세 오차·비용을 숨긴 초기 검토 화면입니다. 과거에 본 장면이므로 독립 블라인드 검증은 아닙니다. 표시선 없는 전체 장면을 먼저 보고 같은 좌표의 확대와 원 마스크 단계를 대조합니다.</p><div class="rgb"><figure><figcaption>원 전체 장면 · 표시선 없음(표시용 JPEG)</figcaption><img src="case_{number:02d}/scene_RGB_unmarked.jpg"></figure><figure><figcaption>원 RGB crop · 원픽셀 PNG</figcaption><img src="case_{number:02d}/crop_RGB_unmarked.png"></figure><figure><figcaption>표시 전용 gamma0.5 · 수치 입력 아님</figcaption><img src="case_{number:02d}/crop_RGB_display_gamma_half.jpg"></figure></div><img src="case_{number:02d}/mask_stages.jpg"><img src="case_{number:02d}/boundary_generation.jpg"><p>주황 raw contour와 청록 clean contour를 구별합니다. convex hull의 연결 사실과 실제 팔레트 외곽 지지의 물리 판정은 별개입니다. 원 화면 경계에서 제외된 부분은 전체 hull 점선과 유한 used 선분 사이 차이로 보이며, 원 clip 계보 장부에서 따로 검산합니다.</p></article>')
    (out/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>원 RGB·관측 우선 검토</title><style>body{font-family:sans-serif;max-width:1800px;margin:20px auto;background:#edf1f5;line-height:1.7;padding:12px}header,article{background:white;padding:20px;margin-bottom:20px;border-radius:8px}img{max-width:100%;height:auto}.rgb{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}figure{margin:0}figcaption{font-size:13px}</style><header><h1>여섯 고정 사례: 원 RGB·실제 저장 마스크·경계 우선 검토</h1><p>이는 기존 오류 사례를 이미 알고 하는 관측 검토입니다. 독립 블라인드 판정이나 새 정답이 아닙니다. 이 초기 페이지에는 후보·오차·성공·비용 표를 넣지 않았습니다. 새 모델·선·마스크·자세는 만들지 않았습니다. 원 RGB와 표시 전용 명암 사본은 분리했습니다.</p></header>'+''.join(articles)+'</html>',encoding='utf-8')
    write(out/'INITIAL_GALLERY_PROOF.json',dict(schema='PHYSICAL_SUPPORT_RGB_MASK_INITIAL_GALLERY_V1',cases=proof))
    receipt=dict(schema='INITIAL_NONINDEPENDENT_RGB_MASK_REVIEW_RENDER_V1',cases=len(proof),
        candidate_error_cost_success_labels=0,not_independent_blind_validation=True,new_pose_NN_mask_or_line_detection_calls=0,
        all_RGB_permission_verified=all(p['RGB_publication_inventory_match']for p in proof),
        runtime_guard=runtime_guard.receipt(),
        output_scope='private initial review; public promotion only after complete permission audit',
        elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
        files={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()})
    write(out/'INITIAL_GALLERY_RECEIPT.json',receipt);return receipt


def render_initial(source_root,score_private,state_private,output):
    from .audit_guard import SupportAuditOnly
    with SupportAuditOnly()as guard:
        return _render_initial(source_root,score_private,state_private,output,runtime_guard=guard)


def _wire(ax,xy,color,label):
    q=np.asarray(xy,float)
    edges=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
    for i,(a,b)in enumerate(edges):ax.plot(q[[a,b],0],q[[a,b],1],c=color,lw=1.35,label=label if i==0 else None)


def _parents(ax,lineage,*,labels=True,highlight=None):
    polygon=np.asarray(lineage['polygon_xy'],float)
    for j,(a,b)in enumerate(zip(polygon,np.roll(polygon,-1,axis=0))):
        q=np.stack((a,b));ax.plot(q[:,0],q[:,1],c='#ed9d27',ls='--',lw=.8,alpha=.7)
        if labels:ax.annotate('H'+str(j),(a+b)/2,fontsize=6,color='#9e5516',clip_on=True)
    for record in lineage['used_segments']:
        q=np.asarray(record['xy']);selected=record['segment_id']==highlight
        ax.plot(q[:,0],q[:,1],c='#e91c78'if selected else'#18bda9',lw=2.7 if selected else 1.)
        if labels:ax.annotate('E'+str(record['segment_id']),q.mean(0),xytext=(0,6),textcoords='offset points',fontsize=6,color='#106e65',clip_on=True)
    for record in lineage['excluded_frame_parts']:
        q=np.asarray(record['xy']);ax.plot(q[:,0],q[:,1],c='#e33232',ls=':',lw=2.1)


def _svg_parents(folder,lineage,roi):
    x0,y0,x1,y1=roi;elements=[f'<image href="crop_RGB_unmarked.png" x="{x0-.5}" y="{y0-.5}" width="{x1-x0}" height="{y1-y0}"/>']
    polygon=np.asarray(lineage['polygon_xy'])
    for j,(a,b)in enumerate(zip(polygon,np.roll(polygon,-1,axis=0))):
        elements.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#ef9b20" stroke-width=".7" stroke-dasharray="3 2"><title>H{j}: original full convex-hull edge</title></line>')
    for r in lineage['used_segments']:
        a,b=r['xy'];mid=(np.asarray(a)+np.asarray(b))/2;sid=r['segment_id']
        elements.append(f'<g><title>E{sid}, parent H{r["original_hull_edge_index"]}, source samples {r["sample_indices"]}</title><line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#12bda6" stroke-width="1.3"/><text x="{mid[0]}" y="{mid[1]-2}" font-size="5" fill="#04a88f">E{sid}</text></g>')
    for r in lineage['excluded_frame_parts']:
        a,b=r['xy'];elements.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#ff2929" stroke-width="2" stroke-dasharray="2 2"><title>Excluded frame clip; parent H{r["original_hull_edge_index"]}</title></line>')
    (folder/'parent_segments.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0-.5} {y0-.5} {x1-x0} {y1-y0}">'+''.join(elements)+'</svg>')


def _zoom_roi(lineage,segment,image_hw):
    h,w=image_hw;polygon=np.asarray(lineage['polygon_xy']);j=segment['original_hull_edge_index'];n=len(polygon)
    q=polygon[[(j-1)%n,j,(j+1)%n,(j+2)%n]];pad=24
    lo=np.floor(q.min(0)).astype(int)-pad;hi=np.ceil(q.max(0)).astype(int)+pad+1
    # Keep enough source context around a very short edge, with no resizing
    # offsets injected into any scientific coordinate.
    center=(lo+hi)/2;span=np.maximum(hi-lo,[100,80]);lo=np.floor(center-span/2).astype(int);hi=np.ceil(center+span/2).astype(int)
    return [max(0,int(lo[0])),max(0,int(lo[1])),min(w,int(hi[0])),min(h,int(hi[1]))]


def _render_complete(source_root,score_private,state_private,analysis_private,initial_folder,initial_review_receipt,output,*,runtime_guard):
    from PIL import Image
    from matplotlib.colors import TwoSlopeNorm
    from .bindings import load_cases,load_mask_arrays,load_dense,permission_bindings
    from .registration_audit import cuboid,project
    start,cpu=time.monotonic(),time.process_time()
    root,analysis,initial,out=map(Path,(source_root,analysis_private,initial_folder,output))
    initial_review_receipt=Path(initial_review_receipt)
    if not initial_review_receipt.is_file():raise FileNotFoundError('First RGB/mask review record must exist before candidate/cost render')
    if out.exists():raise FileExistsError('Preserve every full-gallery attempt')
    first=json.loads((initial/'INITIAL_GALLERY_RECEIPT.json').read_text())
    for name,digest in first['files'].items():
        if sha(initial/name)!=digest:raise ValueError('Initial review display changed '+name)
    cases=load_cases(root,Path(score_private),Path(state_private));permissions=permission_bindings(root,cases)
    plt=_graphics();shutil.copytree(initial,out);(out/'index.html').rename(out/'initial_index.html')
    proofs=[];articles=[];zoom_count=0
    for case in cases:
        ident=case['id'];number=case['ordinal'];folder=out/f'case_{number:02d}'
        lineage_path=analysis/'mask_lineage_v2'/(ident.replace(':','__')+'.json');lineage=json.loads(lineage_path.read_text())
        if not all(lineage['checks'].values()):raise ValueError('Original mask/outline transformation gate failed')
        rgb=np.asarray(Image.open(case['image_path']).convert('RGB'));h,w=rgb.shape[:2]
        arrays=load_mask_arrays(case);roi=display_roi(case,arrays,(h,w));aspect=(roi[3]-roi[1])/(roi[2]-roi[0])
        records=load_dense(case,score_private);panels=case['panels'];observed=[];modeled=[]
        for row in records:
            observed.append(next(v for v in row['per_observation']if v.get('direction')=='observed_to_model'))
            modeled.append(next(v for v in row['per_observation']if v.get('direction')=='model_to_observed'))
        sample=np.asarray(observed[0]['sample_xy'],float)
        if not np.array_equal(sample,observed[1]['sample_xy'])or not np.array_equal(sample,lineage['observed_boundary_samples_xy']):
            raise ValueError('Observed sample identity/order mismatch')
        delta=np.asarray(observed[1]['weighted_sample_contributions'])-np.asarray(observed[0]['weighted_sample_contributions'])
        expected=observed[1]['weighted_contribution']-observed[0]['weighted_contribution']
        if abs(math.fsum(delta)-expected)>1e-9:raise ValueError('Saved observed Delta sum mismatch')
        extent=max(float(np.abs(delta).max()),1e-15);norm=TwoSlopeNorm(vmin=-extent,vcenter=0,vmax=extent)
        # Original mask/hull and both actual fixed candidates, same display ROI.
        fig,axes=plt.subplots(2,2,figsize=(15,max(7.,min(13.,2*(7.5*aspect+.8)))),layout='constrained')
        fig.suptitle(ident+' · 관측/모델을 분리한 동일 원좌표 비교',fontsize=13)
        _canvas(axes[0,0],rgb,roi,'실제 clean contour · raw는 주황\n물리 외곽 판정과 분리')
        _contours(axes[0,0],[np.asarray(x)for x in lineage['raw_mask_contours_xy']],'#e9b827',.8)
        _contours(axes[0,0],[np.asarray(x)for x in lineage['clean_mask_contours_xy']],'#18cdb3',1.)
        _canvas(axes[0,1],rgb,roi,'전체 parent H / 사용 E / 제외 frame 부분\n빨강점선 제외 · 청록 실제 유한 관측')
        _parents(axes[0,1],lineage)
        for ax,panel,label in zip(axes[1],panels,('실제 선택 s','기존 comparator 대조')):
            _canvas(ax,rgb,roi,label+' · '+panel['family']+'\n실선은 저장 모델 투영 · 점은 같은 observed Delta')
            _wire(ax,panel['model_projection_xy'],'#34b68b'if label=='실제 선택 s'else'#e89723','모델')
            points=ax.scatter(sample[:,0],sample[:,1],c=delta,cmap='coolwarm',norm=norm,s=9,zorder=5)
        fig.colorbar(points,ax=list(axes[1]),shrink=.7,label='저장 weighted sample Delta: comparator - s (양수=s 지지)')
        _save(plt,fig,folder/'physical_layers.jpg');_svg_parents(folder,lineage,roi)
        ref=case['reference'];refq=project(cuboid(ref['xyz']),ref['R'],ref['t'],case['K'])
        fig,axes=plt.subplots(1,2,figsize=(14,max(3.5,min(7.,7*aspect+.7))),layout='constrained')
        _canvas(axes[0],rgb,roi,'원 RGB · 후보/참조 표시 없음')
        _canvas(axes[1],rgb,roi,'기존 기하 재구성 참조의 별도 투영\n독립 물리 정답 아님 · 기존 K/치수/R/t 그대로')
        _wire(axes[1],refq,'#b75ce8','기존 기하 참조');_contours(axes[1],[np.asarray(x)for x in lineage['clean_mask_contours_xy']],'#28cdb4',.7)
        _save(plt,fig,folder/'reference_projection.jpg')
        # Model-side samples belong to different projected segments and arc
        # normalizers. Never subtract them by shared integer index.
        fig,axes=plt.subplots(1,2,figsize=(15,max(3.8,min(7.5,7.5*aspect+.8))),layout='constrained')
        modelproof=[]
        model_color_max=max(float(np.max(o['weighted_sample_contributions']))for o in modeled)
        for ax,observation,panel,label in zip(axes,modeled,panels,('s 자체 모델 표본','대조 자체 모델 표본')):
            q=np.asarray(observation['sample_xy']);cost=np.asarray(observation['weighted_sample_contributions'])
            _canvas(ax,rgb,roi,label+f' · N={len(q)} / arc={observation["normalizer"]:.6g}px\n저장 방향 비용={observation["weighted_contribution"]:.9g}')
            _parents(ax,lineage,labels=False);_wire(ax,panel['model_projection_xy'],'#308d75','자체 모델')
            sc=ax.scatter(q[:,0],q[:,1],c=cost,cmap='magma',vmin=0,vmax=max(model_color_max,1e-15),s=13,zorder=7);fig.colorbar(sc,ax=ax,shrink=.65,label='각 후보 자체 weighted 기여 (Delta 아님)')
            if abs(math.fsum(cost)-observation['weighted_contribution'])>1e-9:raise ValueError('Saved model sample sum mismatch')
            modelproof.append(dict(candidate_id=panel['candidate_id'],sample_xy=observation['sample_xy'],
                sample_weighted_contributions=observation['weighted_sample_contributions'],normalizer=observation['normalizer'],
                weighted_total=observation['weighted_contribution'],sample_indices_not_paired_across_candidates=True))
        fig.suptitle('model→observed는 후보별로 따로 표시 · 동일 인덱스 차이 없음',fontsize=12)
        _save(plt,fig,folder/'model_to_observed_separate.jpg')
        lookup={r['segment_id']:r for r in lineage['used_segments']};groups={}
        extremes=[]
        for role in('largest_positive_samples','largest_negative_samples'):
            for r in case['witness'][role]:
                i=r['sample_index']
                if not np.array_equal(sample[i],[r['x_px'],r['y_px']])or float(delta[i])!=float(r['Delta']):raise ValueError('Original extreme witness differs')
                sid=lineage['sample_parent_segment_ids'][i];groups.setdefault(sid,[]).append(dict(sample_index=i,stored_list=role,Delta=float(delta[i])))
                extremes.append(i)
        zoomproof=[];zoomhtml=[]
        contrast=np.rint(255*np.sqrt(rgb.astype(float)/255)).astype(np.uint8)
        for sid,members in sorted(groups.items()):
            parent=lookup[sid];zroi=_zoom_roi(lineage,parent,(h,w));z_aspect=(zroi[3]-zroi[1])/(zroi[2]-zroi[0]);indexes=np.asarray(parent['sample_indices'],int)
            fig,axes=plt.subplots(1,3,figsize=(17,max(3.3,min(8.5,5.7*z_aspect+.8))),layout='constrained')
            fig.suptitle(f'{ident} · 부모 E{sid}/H{parent["original_hull_edge_index"]} 전체와 이웃 · 같은 선의 표본은 독립 증거가 아님',fontsize=11)
            _canvas(axes[0],rgb,zroi,'원 RGB 확대 · 표시선 없음')
            _mask_canvas(axes[1],arrays['pallet_mask_clean'],zroi,'실제 사용 clean mask와 부모/이웃\n분류용 새 mask 아님');_parents(axes[1],lineage,highlight=sid)
            _canvas(axes[2],contrast,zroi,'display-only gamma0.5 · 원좌표 Delta\n색: 양수 s 지지 / 음수 대조 지지')
            _parents(axes[2],lineage,labels=False,highlight=sid)
            sc=axes[2].scatter(sample[indexes,0],sample[indexes,1],c=delta[indexes],cmap='coolwarm',norm=norm,s=15,zorder=5)
            for m in members:axes[2].annotate(str(m['sample_index']),sample[m['sample_index']],fontsize=7,color='yellow',clip_on=True)
            fig.colorbar(sc,ax=axes[2],shrink=.65,label='원 weighted sample Delta')
            name=f'parent_E{sid:03d}_context.jpg';_save(plt,fig,folder/name);zoom_count+=1
            zoomproof.append(dict(segment_id=sid,original_hull_edge_index=parent['original_hull_edge_index'],
                full_parent_xy=parent['full_parent_xy'],used_segment_xy=parent['xy'],display_crop_original_xyxy=zroi,
                all_source_sample_indices=parent['sample_indices'],source_extreme_members=members,
                segment_signed_Delta=math.fsum(delta[indexes]),segment_absolute_Delta=math.fsum(abs(delta[indexes])),image=name))
            zoomhtml.append(f'<details><summary>E{sid}/H{parent["original_hull_edge_index"]} · 기존 극값 목록의 부모 전체와 이웃</summary><p>원표본 {", ".join(str(x["sample_index"])+" ("+str(x["Delta"])+")"for x in members)}. 작은 값 목록에 음수가 없으면 실제 부호를 그대로 유지합니다. 부모 signed Delta={math.fsum(delta[indexes]):.9g}, 절댓값합={math.fsum(abs(delta[indexes])):.9g}.</p><img loading="lazy" src="case_{number:02d}/{name}"></details>')
        if not set(extremes).issubset({m['sample_index']for z in zoomproof for m in z['source_extreme_members']}):raise ValueError('Original +/- extreme witness was dropped')
        per_segment=''.join(f'<tr><td>E{r["segment_id"]}/H{r["original_hull_edge_index"]}</td><td>{len(r["sample_indices"])}</td><td>{math.fsum(delta[r["sample_indices"]]):.9g}</td><td>{math.fsum(abs(delta[r["sample_indices"]])):.9g}</td><td>{html.escape(str(r["xy"]))}</td></tr>'for r in lineage['used_segments'])
        proofs.append(dict(id=ident,ordinal=number,group=case['group'],source_RGB_sha256=case['image_sha256'],source_mask_sha256=case['mask_file_sha256'],
            parent_crop_original_xyxy=case['crop_xyxy'],display_crop_original_xyxy=roi,display_transform_only=True,
            fixed_candidate_panels=panels,reference_projection_xy=refq.tolist(),reference_role='existing geometric reconstruction; not independent physical truth',
            original_observed_sample_xy=sample.tolist(),observed_weighted_contributions=[o['weighted_sample_contributions']for o in observed],
            observed_Delta=delta.tolist(),observed_Delta_total=expected,model_to_observed_candidate_specific=modelproof,
            parent_contexts=zoomproof,parent_segment_count=len(lineage['used_segments']),all_parent_segments_shown=True,
            mask_lineage_path='mask_lineage_v2/'+lineage_path.name,mask_lineage_sha256=sha(lineage_path),
            original_witness_extremes_preserved=True,model_indices_never_paired=True))
        articles.append(f'<article id="case{number}"><h2>{number}. {html.escape(ident)} · 원 여섯 사례 유지</h2><p><a href="initial_index.html#case{number}">후보/비용을 숨긴 초기 원RGB·mask 검토</a> · <a href="case_{number:02d}/parent_segments.svg">전체 부모 ID 확대·hover</a></p><h3>관측 경계와 두 고정 모델</h3><img loading="lazy" src="case_{number:02d}/physical_layers.jpg"><p>H는 원 convex hull edge, E는 clip 후 비용에 실제 들어간 유한 부모 선분, 빨강 점선은 제외된 frame 조각입니다. 같은 observed 표본의 색은 comparator−s 비용차이며, 해당 픽셀이 물리 외곽이라는 판정은 아닙니다.</p><h3>기존 참조 — 독립 물리 정답 아님</h3><img loading="lazy" src="case_{number:02d}/reference_projection.jpg"><h3>후보마다 다른 model→observed 표본</h3><img loading="lazy" src="case_{number:02d}/model_to_observed_separate.jpg"><p>각 후보의 자체 원 표본·arc normalizer·가중 비용입니다. 같은 index끼리 빼지 않았습니다.</p><h3>기존 가장 큰 양/음 표본의 전체 부모와 이웃</h3>'+''.join(zoomhtml)+f'<details><summary>전체 유한 부모 {len(lineage["used_segments"])}개 · 표본 {len(sample)}개 분모</summary><table><tr><th>부모 ID</th><th>원 표본수</th><th>signed Delta</th><th>absolute Delta</th><th>원 좌표</th></tr>{per_segment}</table></details></article>')
    header='<header><h1>외곽선 관측의 물리 지지: 여섯 고정 사례</h1><p>새 자세·학습·mask·선·후보·cost 재평가·성능 측정은 없습니다. 기존 RGB/마스크/경계/저장 후보/기하 참조의 관계를 분리한 사후 감사입니다. 기존 사례를 아는 상태이므로 독립 blind 검증이 아닙니다.</p><p><a href="initial_index.html">1차 RGB·mask·hull 화면(오류·후보·성공·비용 미표시)</a> · <a href="../REPORT_KO.md">판정과 미해결 질문</a> · <a href="MODEL_COMPARISON_PROOF.json">원좌표·표본·모델 출처 장부</a></p><p>모든 원사진·mask는 기존 공개 inventory의 정확한 SHA와 연결했습니다. 원 full RGB 파일/NPZ는 새로 복제하지 않고 허용된 표시용 JPEG·crop·mask 파생 그림만 게시합니다. 표시 gamma0.5 사본은 원본 옆에 별도 표시하며 수치 입력을 바꾸지 않습니다.</p></header>'
    (out/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>외곽선 물리 지지 감사</title><style>body{font-family:sans-serif;max-width:1800px;margin:20px auto;padding:14px;background:#eef2f5;line-height:1.7}header,article{background:white;padding:20px;margin-bottom:24px;border-radius:9px}img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:4px;border-bottom:1px solid #ddd}details{margin:12px 0}</style>'+header+''.join(articles)+'</html>')
    write(out/'MODEL_COMPARISON_PROOF.json',dict(schema='PHYSICAL_SUPPORT_FIXED_MODEL_DISPLAY_PROOF_V1',cases=proofs))
    write(out/'PUBLICATION_PERMISSION.json',permissions)
    receipt=dict(schema='OUTLINE_PHYSICAL_SUPPORT_ACTUAL_GALLERY_V1',cases=6,
        source_initial_gallery_receipt_sha256=sha(initial/'INITIAL_GALLERY_RECEIPT.json'),
        initial_mask_review_receipt_sha256=sha(initial_review_receipt),
        parent_segments_displayed=sum(c['parent_segment_count']for c in proofs),original_observed_samples=sum(len(c['original_observed_sample_xy'])for c in proofs),
        extreme_parent_context_images=zoom_count,reference_panels_separate=6,original_model_panels=12,
        original_RGB_and_mask_permission_verified=True,new_cost_evaluations=0,new_observations=0,new_pose_calls=0,new_NN_forwards=0,
        model_sample_indices_never_paired=True,elapsed_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,
        runtime_guard=runtime_guard.receipt(),assets={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()})
    write(out/'GALLERY_RECEIPT.json',receipt);return receipt


def render_complete(*args,**kwargs):
    from .audit_guard import SupportAuditOnly
    with SupportAuditOnly()as guard:return _render_complete(*args,**kwargs,runtime_guard=guard)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in('source-root','score-private','state-private','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--phase',choices=('initial','complete'),default='initial')
    for key in('analysis-private','initial-folder','initial-review-receipt'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.phase=='initial':result=render_initial(a.source_root,a.score_private,a.state_private,a.output)
    else:
        if not all((a.analysis_private,a.initial_folder,a.initial_review_receipt)):p.error('complete requires analysis-private, initial-folder, initial-review-receipt')
        result=render_complete(a.source_root,a.score_private,a.state_private,a.analysis_private,a.initial_folder,a.initial_review_receipt,a.output)
    print(json.dumps(result,ensure_ascii=False))
