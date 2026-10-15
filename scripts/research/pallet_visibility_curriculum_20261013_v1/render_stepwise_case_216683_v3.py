"""Frozen actual-image pipeline replay for p07:216683; no new pose fits.

Separate measured image coordinates, model projections, masks and reference.
S raw coordinates are gated before indexing for all input visualizations.
Only stored intermediate poses are shown. The last correspondence inspection
uses geometry alone and never alters the frozen predictions or scores.
"""
from __future__ import annotations
import os
for _thread_key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_thread_key]='1'

from copy import deepcopy
import html
import json
from pathlib import Path
import shutil

import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager, patheffects
from matplotlib.patches import Circle, Rectangle
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import common as C
from . import evaluate_u_regions_v4_5 as U
from . import evaluation_interval_v4_2b as OLD
from . import statewise_evaluation_v4 as E
from . import visible_bi_correction_v2 as BI
from . import render_incompatible_pose_compare_v2 as COMP
from .scoring import pose_metric

ID='eval_pallet07:1778652166837872128'
OUTPUT=U.BOUNDARY/'stepwise_case_216683_v3'
FONT=Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
STATE_COLORS={'B':'#00b8ff','I':'#8a7bff','O':'#ffb322','U':'#bbbbbb','S':'#e979f0','T':'#fd5252'}
ORANGE='#ff8a30';GREEN='#82f557';BLUE='#00d4ff'


def _pose(R,t,g):
    R=np.asarray(R)
    return dict(available=True,R_cf=R.tolist(),centroid=list(t),
        R_physical=(R@np.asarray(g['physical_registration_Q'])).tolist(),
        cf_extents=g['cf_extents'])


def _inputs():
    rows,old_protocol,seal=E._verify_prediction_seal(OLD.OUTPUT)
    old=next(r for r in rows if r['id']==ID)
    seal_u=C.read(U.OUTPUT/'TEST_PREDICTION_SEAL.json')
    assert seal_u['before_new_partition_pose_scoring']
    assert C.sha(U.OUTPUT/'TEST_PREDICTIONS.jsonl.gz')==seal_u['predictions_sha256']
    assert C.sha(U.OUTPUT/'PROTOCOL.json')==seal_u['protocol_sha256']
    protocol=C.read(U.OUTPUT/'PROTOCOL.json')
    E._verify_sources(protocol['source_sha256']);E._verify_sources(protocol['input_sha256'])
    current=next(r for r in C.rows(U.OUTPUT/'TEST_PREDICTIONS.jsonl.gz') if r['id']==ID)
    canonical=next(r for r in C.read(E.COHORT/'PREDICTION_INPUTS.json')['frames'] if r['id']==ID)
    assert current['states']==old['runtime']['states']
    assert canonical['initial_poses']['N3_SUBPIX']==old['initial_pose']
    assert canonical['points']['N3_SUBPIX']==old['runtime']['original_raw_points']
    correction_path=U.RECOVERY/'TEST_predictions'/(ID.replace(':','__')+'.json')
    correction=C.read(correction_path)
    assert C.sha(correction_path)==protocol['input_sha256'][str(correction_path.relative_to(C.ROOT))]
    n3_path=C.ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1/predictions/REAL_DEV/N3_DIM_SYM_seed1.json'
    n3_file=C.read(n3_path);assert n3_file['complete'] and n3_file['GT_input'] is False
    n3_record=next(r for r in n3_file['records'] if r['id']==ID)
    n3=n3_record['candidates'][canonical['selected_index']]['keypoints_xy']
    assert n3_record['selected_index']==canonical['selected_index']
    evidence_path=U.BOUNDARY/'image_evidence_v2'/(ID.replace(':','__')+'.json')
    evidence=C.read(evidence_path);mask_path=evidence_path.parent/evidence['mask_npz']
    assert C.sha(mask_path)==evidence['mask_file_sha256']
    image_path=C.ROOT/old['image'];assert C.sha(image_path)==old['image_sha256']==evidence['image_sha256']
    image=cv2.imread(str(image_path));assert BI.image_sha256(image)==correction['image_pixel_sha256']
    with np.load(mask_path,allow_pickle=False) as masks:
        sam=np.array(masks['pallet_mask']);tiny=np.array(masks['state_pallet_mask'])
    assert sam.shape==tiny.shape==tuple(old['raw_hw'])
    pf_dir=E.COHORT/'posefix_sub_comparison_v1';pf_seal=C.read(pf_dir/'PREDICTION_SEAL.json')
    assert C.sha(pf_dir/'PREDICTIONS.jsonl.gz')==pf_seal['predictions_sha256'] and pf_seal['before_scoring']
    pf=next(r for r in C.rows(pf_dir/'PREDICTIONS.jsonl.gz') if r['id']==ID and r['method']=='POSEFIX_RAW')
    # All pose sources are frozen before posthoc reference access.
    report=C.read(U.OUTPUT/'TEST_POSE_SCORE.json')
    target_path=E.COHORT/'test_reused_v1/ISOLATED_TEST_SCORING.json'
    assert C.sha(target_path)==report['target_binding']['sha256']
    label=C.read(target_path)['rows'][ID]
    paths=[Path(__file__),FONT,image_path,n3_path,evidence_path,mask_path,correction_path,target_path,
        E.COHORT/'PREDICTION_INPUTS.json',OLD.OUTPUT/'PREDICTIONS.jsonl.gz',OLD.OUTPUT/'PREDICTION_SEAL.json',
        U.OUTPUT/'TEST_PREDICTIONS.jsonl.gz',U.OUTPUT/'TEST_PREDICTION_SEAL.json',U.OUTPUT/'TEST_POSE_SCORE.json',
        pf_dir/'PREDICTIONS.jsonl.gz',pf_dir/'PREDICTION_SEAL.json',COMP.OUTPUT/'CASES.json']
    return dict(old=old,current=current,canonical=canonical,n3=n3,correction=correction,
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB),sam=sam,tiny=tiny,pf=pf,label=label,
        protocol=protocol,bindings={str(p):C.sha(p) for p in paths})


def _nonS(raw,hidden):
    points={}
    for k in range(8):
        if hidden[k]:continue
        points[k]=list(raw[k])
    return points


def _base(image,title,*,bottom=550):
    fig,ax=plt.subplots(figsize=(10.6,8),dpi=130)
    fig.patch.set_facecolor('white');ax.set_facecolor('#eef1f5');ax.imshow(image)
    ax.set_xlim(-22,662);ax.set_ylim(bottom,-24)
    ax.add_patch(Rectangle((0,0),640,480,fill=False,ec='#92a1b5',lw=1.2,ls='--'))
    if bottom>480:
        ax.text(14,505,'회색 영역 = 실제 영상 밖',fontsize=11,color='#52657b')
    ax.set_title(title,fontsize=17,pad=15);ax.set_xlabel('영상 x (px)');ax.set_ylabel('영상 y (px)')
    fig.subplots_adjust(top=.9,bottom=.1,left=.07,right=.99)
    return fig,ax


def _text(ax,label,q,color,offset=(5,-11),size=11):
    ax.annotate(label,q,xytext=offset,textcoords='offset points',color=color,fontsize=size,
        fontweight='bold',clip_on=True,path_effects=[patheffects.withStroke(linewidth=2.5,foreground='#101720')])


def _points(ax,points,*,states=None,color=BLUE,marker='x',prefix='',suffixes=None):
    for k,q in points.items():
        c=STATE_COLORS[states[k]] if states else color
        ax.scatter(*q,s=75,c=c,marker=marker,linewidths=2.2,zorder=9)
        _text(ax,prefix+str(k)+(suffixes or {}).get(k,''),q,c)


def _wire(ax,q,color=ORANGE,numbering=False,dashed=False):
    for a,b in COMP.EDGES:
        ax.plot(q[[a,b],0],q[[a,b],1],color='#101720',lw=4,alpha=.55,ls='--' if dashed else '-')
        ax.plot(q[[a,b],0],q[[a,b],1],color=color,lw=2,ls='--' if dashed else '-')
    if numbering:
        _points(ax,{k:q[k] for k in range(8)},color=color,marker='s',prefix='P')


def _save(fig,path):
    fig.savefig(path,dpi=130,bbox_inches='tight',facecolor='white');plt.close(fig)


def _svg_number_diagram(model,ref,mapping,source_mapping):
    # A canonical coordinate diagram, never a new fitted camera pose.
    fig,axes=plt.subplots(1,2,figsize=(13,5.6),dpi=130)
    tops=[2,3,7,6,2]
    for ax,which,title in zip(axes,(source_mapping,mapping),('현재 검사: 같은 번호끼리 연결','기하 확인: 모델 좌표에서 같은 위치끼리 연결')):
        a=model[:,[0,2]];b=ref[np.asarray(which)][:,[0,2]]
        ax.plot(a[tops,0],a[tops,1],c=ORANGE,lw=2);ax.plot(b[tops,0],b[tops,1],c=GREEN,lw=2,ls='--')
        for k in (2,3,6,7):
            p,g=a[k],b[k]
            if not np.allclose(p,g,atol=1e-8):
                ax.annotate('',xy=g,xytext=p,arrowprops={'arrowstyle':'->','color':'#d63f39','lw':2.5,'connectionstyle':'arc3,rad=.22'})
            ax.scatter(*p,c=ORANGE,s=50,edgecolors='#253448');ax.scatter(*g,c=GREEN,s=28,edgecolors='#253448')
            ax.text(p[0]+.025,p[1]+.035,f'P{k}',fontsize=12,color='#263b55')
            ax.text(g[0]+.025,g[1]-.075,f'G{which[k]}',fontsize=12,color='#263b55')
        gap=float(np.linalg.norm(model[2]-ref[which[2]]))
        ax.text(.035,.06,f'P2 → G{which[2]} : {gap:.2f}m',transform=ax.transAxes,fontsize=15,fontweight='bold',color='#d63f39' if gap>1e-8 else '#2d7423',bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('물체 X (m)');ax.set_ylabel('물체 Z (m)');ax.margins(.2);ax.set_title(title,fontsize=14)
    fig.suptitle('번호 순서가 다릅니다. 두 저장 모델의 3D 크기·형상은 같습니다.',fontsize=17,y=.99)
    fig.tight_layout(rect=[0,0,1,.92]);return fig


def _build(context):
    old=context['old'];rt=old['runtime'];cur=context['current'];g=rt['initial_geometry'];K=np.asarray(g['K']);xyz=np.asarray(g['native_xyz'])
    hidden=g['initial_self_hidden'];states=rt['states'];image=context['image'];corr=context['correction'];truth=context['label']['target']['truth']
    seed=cur['arms'][U.ARMS[1]]['solution']['BI_prior_seed'];initial=old['initial_pose'];final=cur['arms'][U.ARMS[1]]['actual_pose']
    seedpose=_pose(seed['R'],seed['t'],g);metrics={name:pose_metric(pose,truth) for name,pose in (
        ('initial',initial),('BI_prior',seedpose),('final',final),('PoseFix',context['pf']['actual_pose']))}
    rawsets={'YOLO':_nonS(context['canonical']['points']['BASE'],hidden),
        'N3':_nonS(context['n3'],hidden),'N3_SubPix':_nonS(rt['original_raw_points'],hidden)}
    fit=cur['frozen_BI_fit'];measured={m['native_corner']:m['xy'] for m in fit}
    assert all(m['source']=='VISIBLE_BACKBONE_FALLBACK' for m in fit)
    assert all(measured[k]==rawsets['N3_SubPix'][k] for k in measured)
    assert corr['recovered_native_ids']==[]
    qinit=np.asarray(g['initial_projection_xy']);qfinal=C.project(xyz,final['R_cf'],final['centroid'],K)
    qseed=C.project(xyz,seed['R'],seed['t'],K)
    same_u=cur['arms'][U.ARMS[0]]['actual_pose']['R_cf']==final['R_cf'] and cur['arms'][U.ARMS[0]]['actual_pose']['centroid']==final['centroid']
    assert same_u and states==['B','I','U','B','O','B','B','S']
    delta_cm=(np.asarray(final['centroid'])-np.asarray(truth['t']))*100
    init_delta_cm=(np.asarray(final['centroid'])-np.asarray(initial['centroid']))*100
    Qref=np.asarray(truth['body_R']).T@np.asarray(truth['R']);ref_xyz=C.cuboid(*truth['body_xyz'])@Qref
    model_xyz=xyz@np.asarray(g['physical_registration_Q'])
    cost=np.linalg.norm(model_xyz[:,None,:]-ref_xyz[None,:,:],axis=2)
    rows,cols=linear_sum_assignment(cost);mapping=cols.tolist();shape_gap=float(cost[rows,cols].max())
    assert mapping==[4,0,3,7,5,1,2,6] and shape_gap<1e-8
    steps=[]
    def add(title,caption,kind,draw,*,keyfacts=None):
        i=len(steps)+1;name=f'{i:02d}_{kind}.png';draw(OUTPUT/'assets'/name)
        steps.append(dict(number=i,title=title,caption=caption,kind=kind,image='assets/'+name,keyfacts=keyfacts or [],GT_overlay=kind=='pose_comparison'))
    def pointstage(path,points,title,before=None):
        fig,ax=_base(image,title)
        if before:
            for k,pt in points.items():
                a=np.asarray(before[k]);b=np.asarray(pt)
                ax.scatter(*a,s=33,facecolors='none',edgecolors='#fff4ac',zorder=7)
                ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#fff4ac','lw':1.5})
        _points(ax,points);_save(fig,path)
    add('1. 처음 YOLO 키포인트','저장된 YOLO 좌표입니다. ×는 영상 예측점입니다. 뒤에서 S로 분류되는 7번의 원좌표는 표시에서 제외했습니다. 9번째 중심점은 이번 B/I 자세 입력에 포함하지 않습니다.',
        'yolo',lambda p:pointstage(p,rawsets['YOLO'],'1. 처음 YOLO 키포인트 (비S 좌표)'),keyfacts=['실제 640×480 영상','초기 S7의 원좌표 표시 제외'])
    add('2. N3 모델을 거친 좌표','같은 영상의 저장된 N3_DIM_SYM 출력입니다. 노란 화살표는 YOLO → N3 이동입니다. PoseFix가 이 단계에 들어가는 것은 아닙니다.',
        'n3',lambda p:pointstage(p,rawsets['N3'],'2. N3 모델 결과',rawsets['YOLO']))
    add('3. SubPix와 이동한도를 적용한 초기 좌표','N3 좌표에 SubPix와 YOLO 기준 총이동 제한을 적용한 실제 N3_SUBPIX 캐시입니다. 이후 현재 방법의 상태 판정과 보정은 이 좌표에서 시작합니다.',
        'subpix',lambda p:pointstage(p,rawsets['N3_SubPix'],'3. N3 + SubPix 이후 초기 좌표',rawsets['N3']))
    def coarse(path):
        fig,ax=_base(image,'4. 초기 PnP 모델 투영 (사각형 = 모델 위치)');_wire(ax,qinit,numbering=True)
        _points(ax,rawsets['N3_SubPix'],color=BLUE,marker='x');_save(fig,path)
    add('4. 초기 PnP로 대략적인 자세','주황 박스는 N3+SubPix의 저장된 초기 PnP입니다. 파란 ×는 영상 좌표이고, 주황 사각형은 3D 모델을 투영한 좌표입니다. 이때부터 팔레트 모델·등록축은 현재 보정 과정에서 고정됩니다.',
        'initial_pnp',coarse,keyfacts=[f"사후 초기 오차 {metrics['initial']['translation_cm']:.2f}cm / {metrics['initial']['rotation_deg']:.2f}°",'native 모델 X×Y×Z = 1.1×0.11×1.3m'])
    def roles(path):
        fig,ax=_base(image,'5. 초기 자세에서 S / B / I 역할 정의');_wire(ax,qinit,color='#a9bed0')
        roles=['S' if hidden[k] else g['initial_roles'][k] for k in range(8)]
        _points(ax,{k:qinit[k] for k in range(8)},states=roles,marker='s',suffixes={k:':'+roles[k] for k in range(8)})
        _save(fig,path)
    add('5. 기하로 S와 B/I 역할 정의','S7은 초기 모델의 면 방향으로 정합니다. S7에는 RGB 가림 판정·경계보정을 하지 않습니다. B/I 역할은 아직 실제 가림 여부를 뜻하지 않습니다.',
        'geometric_roles',roles,keyfacts=['S7: 모델 투영만 사용','B: 경계 역할 / I: 내부 역할'])
    def masks(path):
        fig,axes=plt.subplots(1,2,figsize=(13,5.6),dpi=130)
        for ax,mask,title in ((axes[0],context['tiny'],'가림 분류 입력: 저장된 Tiny soft mask'),(axes[1],context['sam'],'B 경계 보정 입력: 저장된 MobileSAM mask')):
            ax.imshow(image);ax.imshow(mask,cmap='magma',vmin=0,vmax=1,alpha=.35);ax.set_title(title,fontsize=14);ax.set_xlim(0,640);ax.set_ylim(480,0);ax.axis('off')
        box=rt['mask_audit']['prompt_box_xyxy'];axes[1].add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,ec=BLUE,lw=1.5,ls='--'))
        _points(axes[1],{k:rawsets['N3_SubPix'][k] for k in rt['mask_audit']['prompt_native_ids']},color=BLUE)
        fig.tight_layout();_save(fig,path)
    add('6. 가림 판정·경계 보정에 사용한 마스크','당시 저장된 실제 마스크입니다. Tiny mask는 가림 분류의 특징, MobileSAM mask는 B 경계 보정에 사용했습니다. SAM의 파란 사각형은 당시 프롬프트입니다. 이 시각화를 위해 모델을 다시 실행하지 않았습니다.',
        'mask_sources',masks,keyfacts=['마스크가 보임 여부의 정답은 아님','S7 프롬프트 없음'])
    def visibility(path):
        fig,ax=_base(image,'7. 실제 RGB 가림 판정 결과')
        suffix={}
        for rec in rt['state_observation']['records']:
            k=rec['native_id'];suffix[k]=':'+states[k]+(f" {rec['p_visible']:.3f}" if rec['p_visible'] is not None else '')
        _points(ax,rawsets['N3_SubPix'],states=states,suffixes=suffix)
        _points(ax,{7:qinit[7]},states=states,marker='s',suffixes={7:':S (PnP)'});_save(fig,path)
    add('7. S를 제외한 점의 가림 판단','B0·I1·B3·B5·B6은 보이는 점으로, 4는 O로 판정했습니다. U2는 y=486.73px로 영상 높이 480 밖이어서 RGB 판정을 하지 않았습니다. T로 확정한 점은 없습니다.',
        'rgb_states',visibility,keyfacts=['최종 상태 B, I, U, B, O, B, B, S','O4 보임 점수 0.386','U2: 영상 밖 / RGB 미조회'])
    def correction(path):
        fig,ax=_base(image,'8. B/I 보정 검사 → 5점 모두 원좌표 유지')
        ax.contour(context['sam'].astype(float),levels=[.5],colors=['#e37cf4'],linewidths=.8,alpha=.65)
        _points(ax,measured,states=states,marker='o',suffixes={k:':'+states[k]+' 유지' for k in measured})
        rec=next(x for x in corr['boundary_recovery_records'] if x['native_corner']==5)
        ray=next(x for x in rec['expanded_rays'] if x['supported'])
        pts=np.asarray(ray['observed_points']);ax.scatter(pts[:,0],pts[:,1],c=GREEN,s=24,zorder=6)
        normal=np.asarray(ray['normal']);line=pts-(pts@normal+ray['offset'])[:,None]*normal
        ax.plot(line[:,0],line[:,1],c=GREEN,lw=2.3)
        for attempt in rec['observed_line_recentering']:
            _points(ax,{5:attempt['observed_anchor_xy']},color='#fff36c',marker='D',suffixes={5:' 탐색 기준'})
        ax.text(16,526,'초록: B5→4의 실제 지지된 선 15/15  |  두 번째 선이 없어 교점 보정 0개',fontsize=10,color='#32475e')
        _save(fig,path)
    add('8. B/I 영상 보정과 추가 B 복구 시도','다섯 점 모두 두 개의 지지된 선을 얻지 못해 원좌표를 유지했습니다. B5→4 선 하나는 15/15 표본으로 지지됐지만 B5→6은 5/15에 그쳤습니다. 관측 선으로 탐색 기준을 옮겨 다시 검사한 선도 방향차 54.52°로 보류됐습니다. 노란 탐색 기준은 확정 KP가 아닙니다.',
        'bi_correction',correction,keyfacts=['B0·I1·B3·B5·B6 좌표 변경 0개','B0/B3/B6: 표본 부족','I1: 방향차 20.57° / 13.25°'])
    def prior(path):
        fig,ax=_base(image,'9. B/I 관측 + CAL prior로 자세 최적화');_wire(ax,qseed,numbering=True)
        _points(ax,measured,states=states,marker='o',suffixes={k:':'+states[k]+' 관측' for k in measured});_save(fig,path)
    add('9. 보이는 B/I 점으로 PnP 자세 최적화','저장된 BI_prior_seed 자세입니다. B/I 관측은 고정된 다섯 점이고, PnP가 모델 자세를 움직였습니다. 보정 좌표가 없더라도 초기 PnP와는 사용하는 점·가중치·prior가 달라 자세는 바뀔 수 있습니다.',
        'bi_prior_pnp',prior,keyfacts=[f"사후 BI/prior 오차 {metrics['BI_prior']['translation_cm']:.2f}cm / {metrics['BI_prior']['rotation_deg']:.2f}°",'S7은 자세에서 투영'])
    def domains(path):
        fig,ax=_base(image,'10. O4 / U2 위치 범위를 제약으로 사용',bottom=840)
        for region in [*rt['ot_domains']['domains'],*cur['u_region_observation']['u_regions']]:
            k=region['native_corner'];center=region['center_xy'];radius=region['half_axes_px'][0];color=STATE_COLORS[region['state']]
            ax.add_patch(Circle(center,radius,ec=color,fc=color,alpha=.15,lw=2));ax.add_patch(Circle(center,radius,ec=color,fill=False,lw=2))
            ax.scatter(*center,s=65,c=color,marker='x',linewidths=2.2,zorder=9)
            _text(ax,f'{k}:{states[k]} r={radius:.2f}px',center,color,offset=(-105,35) if k==4 else (-125,-35))
            ax.scatter(*qfinal[k],s=65,c=ORANGE,marker='s',zorder=9)
            _text(ax,f'P{k} 최종 투영',qfinal[k],ORANGE,offset=(15,-20) if k==4 else (15,14))
        _wire(ax,qfinal,color=ORANGE);_save(fig,path)
    add('10. O/U 영역을 포함한 최종 제약','O4 반경은 49.44px, U2 반경은 311.54px입니다. U 영역은 영상 밖까지 그대로 유지합니다. 두 점의 최종 모델 투영이 이미 원 안에 있으므로 이 사례의 선택 자세는 NO_OT·NO_U와 같습니다.',
        'region_constraints',domains,keyfacts=['T 없음','O4 중심→최종 투영 약 9.67px','U2 중심→최종 투영 약 12.75px','U는 정밀 좌표로 보정하지 않음'])
    def final_view(path):
        fig,ax=_base(image,'11. 최종 PnP 모델과 고정된 B/I 관측');_wire(ax,qfinal,numbering=True)
        _points(ax,measured,states=states,marker='o',suffixes={k:':'+states[k]+' 관측' for k in measured})
        _points(ax,{7:qfinal[7]},states=states,marker='s',suffixes={7:':S PnP투영'});_save(fig,path)
    add('11. 현재 방법의 최종 자세','주황 박스는 하나의 최종 rigid pose입니다. 원형 B/I 표시는 영상 관측이므로 박스의 모델 투영과 정확히 겹치지 않을 수 있습니다. S7은 PnP 투영만 사용하고, U2는 보고 좌표에서 미확정으로 남습니다.',
        'final_pnp',final_view,keyfacts=[f"최종 오차 {metrics['final']['translation_cm']:.2f}cm / {metrics['final']['rotation_deg']:.2f}°",f"초기→최종 깊이 +{init_delta_cm[2]:.2f}cm"])
    existing=COMP.OUTPUT/'assets'/(ID.replace(':','__')+'_details.png')
    context['bindings'][str(existing)]=C.sha(existing)
    add('12. 사후 PoseFix / 현재 / GT 비교','이 단계의 초록 G 박스·축·번호는 기존 정답 모델입니다. 추론에는 사용하지 않았습니다. 현재 중심은 기준 대비 x=-6.04cm, y=+2.04cm, z=+7.85cm 차이이며, 전체 위치 차이는 10.11cm입니다. 카메라 x는 오른쪽, y는 아래쪽, z는 깊이입니다.',
        'pose_comparison',lambda p:shutil.copyfile(existing,p),keyfacts=[f"PoseFix RAW {metrics['PoseFix']['translation_cm']:.2f}cm / {metrics['PoseFix']['rotation_deg']:.2f}°",'실제 영상 위치·크기 차이와 번호 문제를 분리'])
    source_mapping=context['label']['metadata']['native_to_reference_corner_ids']
    add('13. G/P 번호가 다른 이유','현재 모델과 기준 모델의 3D 크기·형상은 같지만 코너 순서가 다릅니다. 기존 identity 대응은 번호가 어긋납니다. 오른쪽 그림은 기하만으로 같은 3D 위치를 연결한 진단입니다. 이 대응을 추론이나 기존 평가에 새로 적용하지 않았으며, 10.11cm 중심 오차도 그대로입니다.',
        'corner_numbering',lambda p:_save(_svg_number_diagram(model_xyz,ref_xyz,mapping,source_mapping),p),
        keyfacts=['같은 3D 위치 대응: P0→G4, P1→G0, P2→G3, P3→G7','P4→G5, P5→G1, P6→G2, P7→G6','번호 문제 ≠ 중심 위치 오차'])
    result=dict(id=ID,image=old['image'],image_sha256=old['image_sha256'],steps=steps,
        state_records=rt['state_observation']['records'],initial_roles=g['initial_roles'],initial_S_ids=[7],
        input_points_nonS=rawsets,initial_pose=initial,stored_BI_prior_pose=seedpose,final_pose=final,
        fixed_BI_measurements=fit,BI_correction_records=corr['records'],B_recovery_records=corr['boundary_recovery_records'],
        measured_BI_changed_count=0,domain_observations={'O_T':rt['ot_domains']['domains'],'U':cur['u_region_observation']['u_regions']},
        O_audit=cur['arms'][U.ARMS[1]]['solution']['joint_constraint_audit'],U_audit=cur['arms'][U.ARMS[1]]['solution']['U_region_constraint_audit'],
        U_changed_selected_pose=False,metrics=metrics,reference_center_error_cm=delta_cm,
        initial_to_final_center_change_cm=init_delta_cm,
        numbering_diagnostic={'source_mapping':source_mapping,'geometry_only_correspondence':mapping,
            'unordered_shape_max_gap_m':shape_gap,'source_identity_max_gap_m':float(cost[np.arange(8),source_mapping].max()),
            'not_applied_to_prediction_or_existing_scoring':True},
        stage_sources_are_stored_actual_data=True,new_backbone_or_mask_forwards=0,new_PnP_fits=0,
        S_raw_coordinate_reads_for_render=0,S_current_RGB_queries=0,
        GT_used_for_runtime=False,GT_posthoc_metrics=True,GT_overlay_only_in_stage12=True)
    return C.finite(result)


def _page(data):
    encoded=json.dumps(data['steps'],ensure_ascii=False).replace('<','\\u003c')
    items=''.join(f'<button class="stage-button" data-step="{s["number"]-1}">{s["number"]}. {html.escape(s["title"].split(". ",1)[1])}</button>' for s in data['steps'])
    state_rows=[]
    for record in data['state_records']:
        score='—' if record['p_visible'] is None else format(record['p_visible'],'.3f')
        queried='실행' if record['rgb_evaluated'] else '미조회'
        state_rows.append(f'<tr><td>{record["native_id"]}</td><td>{record["state"]}</td><td>{score}</td><td>{queried}</td></tr>')
    state_table=''.join(state_rows)
    # Use fixed strings and escaped data only; runtime data are assigned via textContent.
    page='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실제 단계별 과정 · p07 10.11cm / 4.73°</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf2f8;color:#172b43;font:16px/1.55 Arial,'Noto Sans CJK KR','NanumGothic',sans-serif}main{max-width:1550px;margin:0 auto;padding:22px}header{background:white;border:1px solid #d4e0eb;padding:20px 24px;border-radius:12px}h1{margin:0 0 8px;font-size:26px}p{margin:10px 0}.lede{font-size:17px}.summary{display:flex;gap:12px;flex-wrap:wrap;margin:15px 0}.summary span{background:#edf4fc;border-radius:7px;padding:8px 12px}.navigation{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:16px 0}button,a.button{background:white;border:1px solid #a9bfd6;border-radius:7px;padding:8px 12px;color:#193a5b;cursor:pointer;font-size:15px;text-decoration:none}button:disabled{opacity:.35}input{width:300px}.stage-list{display:flex;overflow:auto;gap:8px;padding:12px 0;scrollbar-width:thin}.stage-button{white-space:nowrap;padding:8px 10px;font-size:13px}.stage-button.active{background:#244f76;color:white}.layout{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:16px}.picture{background:white;border:1px solid #d4e0eb;border-radius:12px;padding:14px}#stage-title{font-size:22px;margin:0 0 12px}#stage-image{display:block;width:100%;height:auto}.caption{font-size:16px;line-height:1.7}aside{background:white;border:1px solid #d4e0eb;border-radius:12px;padding:18px;height:max-content}aside h3{margin:0 0 12px;font-size:18px}#facts{padding-left:18px}#facts li{margin:8px 0}.small{font-size:13px;color:#536b84}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:7px;border-bottom:1px solid #dbe5ef}.status{background:#fff6e8;padding:12px;border-radius:7px;margin:16px 0}details{margin-top:16px}a{color:#1b568e}.gallery{display:none;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.gallery.open{display:grid}.gallery article{background:white;padding:14px;border-radius:9px;border:1px solid #d4e0eb}.gallery img{width:100%;height:auto}.gallery h3{font-size:17px}.gallery p{font-size:14px}footer{padding:18px 0}@media(max-width:850px){main{padding:12px}.layout{grid-template-columns:1fr}.gallery{grid-template-columns:1fr}h1{font-size:22px}header{padding:16px}#stage-title{font-size:19px}input{width:160px}}
</style></head><body><main><header><h1>첨부한 사진의 실제 처리 과정</h1><p class="lede">eval_pallet07:1778652166837872128 · 현재 <strong>10.11cm / 4.73°</strong> 사례입니다. 처음 좌표부터 저장된 실제 결과를 순서대로 보여줍니다.</p><div class="summary"><span>초기 9.18cm → 최종 10.11cm</span><span>B/I 좌표 보정 0개</span><span>O/U 제약으로 자세 변경 없음</span></div><p class="small">× = 영상 예측 · ○ = 고정된 B/I 관측 · 주황 박스/□ = PnP 모델 투영. S7은 원예측 좌표를 표시하지 않으며 현재 RGB/보정에서 제외됩니다. 마지막 단계의 GT는 사후 설명용입니다.</p></header>
<div class="navigation"><button id="previous">← 이전</button><strong id="counter"></strong><button id="next">다음 →</button><input id="slider" type="range" min="0" max="12" value="0" aria-label="처리 단계 선택"><button id="overview">전체 단계 펼치기</button><a class="button" id="full-image" target="_blank">큰 이미지 ↗</a></div><nav class="stage-list">'''+items+'''</nav><div class="layout"><section class="picture"><h2 id="stage-title"></h2><img id="stage-image" alt="실제 영상 처리 단계"><p id="caption" class="caption"></p></section><aside><h3>현재 단계에서 확인할 것</h3><ul id="facts"></ul><div class="status"><strong>이 사례에서 실제로 일어난 일</strong><p>가림 판정 뒤 B/I 다섯 점을 선택했지만, 경계 보정은 모두 보류됐습니다. 넓은 O/U 범위는 현재 자세를 줄이지 못했습니다.</p></div><details><summary>점별 실제 상태·보임 점수</summary><table><thead><tr><th>KP</th><th>상태</th><th>보임 점수</th><th>RGB</th></tr></thead><tbody>'''+state_table+'''</tbody></table></details><p class="small">이 페이지는 모델·PnP를 다시 실행하거나 중간 자세를 만들어내지 않았습니다. 저장된 640×480 좌표·카메라를 사용했습니다.</p><a href="PROCESS_DATA.json">단계별 좌표·근거 JSON</a></aside></div><section id="gallery" class="gallery"></section><footer class="small">번호 대응 진단과 중심 위치 오차는 별개의 값입니다. 이 사례는 전체 TEST49 평가에 그대로 포함됐습니다.</footer></main><script>
const STEPS='''+encoded+''';let index=0;const image=document.getElementById('stage-image');function select(i){index=Math.max(0,Math.min(STEPS.length-1,i));const s=STEPS[index];document.getElementById('stage-title').textContent=s.title;image.src=s.image;image.alt=s.title+' 실제 영상';document.getElementById('caption').textContent=s.caption;document.getElementById('counter').textContent=(index+1)+' / '+STEPS.length;document.getElementById('slider').value=index;document.getElementById('previous').disabled=index===0;document.getElementById('next').disabled=index===STEPS.length-1;document.getElementById('full-image').href=s.image;const facts=document.getElementById('facts');facts.replaceChildren();s.keyfacts.forEach(t=>{const li=document.createElement('li');li.textContent=t;facts.appendChild(li)});document.querySelectorAll('.stage-button').forEach((b,k)=>b.classList.toggle('active',k===index));document.body.dataset.stage=String(index+1)}document.getElementById('previous').onclick=()=>select(index-1);document.getElementById('next').onclick=()=>select(index+1);document.getElementById('slider').oninput=e=>select(Number(e.target.value));document.querySelectorAll('.stage-button').forEach(b=>b.onclick=()=>select(Number(b.dataset.step)));document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')select(index+1);if(e.key==='ArrowLeft')select(index-1)});document.getElementById('overview').onclick=()=>{const g=document.getElementById('gallery');if(!g.children.length){STEPS.forEach(s=>{const a=document.createElement('article'),h=document.createElement('h3'),img=document.createElement('img'),p=document.createElement('p');h.textContent=s.title;img.src=s.image;img.loading='lazy';img.alt=s.title;p.textContent=s.caption;a.append(h,img,p);g.appendChild(a)})}g.classList.toggle('open');document.getElementById('overview').textContent=g.classList.contains('open')?'전체 단계 접기':'전체 단계 펼치기'};select(0);
</script></body></html>'''
    return page


def main():
    if OUTPUT.exists():raise FileExistsError('Preserve existing stepwise replay')
    context=_inputs()
    font_manager.fontManager.addfont(str(FONT));plt.rcParams['font.family']='NanumGothic';plt.rcParams['axes.unicode_minus']=False
    OUTPUT.mkdir(parents=True);(OUTPUT/'assets').mkdir()
    data=_build(context);C.write(OUTPUT/'PROCESS_DATA.json',data);(OUTPUT/'index.html').write_text(_page(data),encoding='utf8')
    dest=OUTPUT/'code_snapshot'/Path(__file__).name;dest.parent.mkdir();shutil.copyfile(__file__,dest)
    for path,digest in context['bindings'].items():assert C.sha(path)==digest
    E._verify_sources(context['protocol']['protected_user_sha256']);E._verify_sources(context['protocol']['source_sha256'])
    C.write(OUTPUT/'RENDER_RECEIPT.json',dict(passed=True,case=ID,stages=len(data['steps']),
        input_sha256=context['bindings'],output_sha256={str(p.relative_to(OUTPUT)):C.sha(p) for p in OUTPUT.rglob('*') if p.is_file()},
        new_model_forwards=0,new_PnP_fits=0,S_raw_coordinate_reads_for_render=0,
        GT_runtime_inputs=False,posthoc_GT_overlay_step=12,
        unordered_geometry_mapping_not_used_to_change_predictions_or_scores=True,
        actual_before_after_correction_changes=0,current_U_pose_changes=0))
    print('OUTPUT',OUTPUT,flush=True)


if __name__=='__main__':main()
