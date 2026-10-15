"""Scientific overlay of the sealed first-CAL static smoke; no inference.

Only its exact saved packet/result and explicitly approved original RGB are
read. The plotted starts precede continuous optimization and method selection.
Model wires are predictions. Actual NMS/TLS support remains a finite line
measurement, never a new precise corner or ground-truth annotation.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from . import common as C

B=C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
SMOKE=B/'clean_fresh_BI_actual_runtime_smoke_v4_32'
OUTPUT=B/'clean_fresh_BI_actual_runtime_process_CAL_v4_32_v1/display_v2'
IMAGE=C.ROOT/'data/evaluation/pallet_eval_v1/dev_existing/sessions/eval_cad/rgb/1778653003088339968.png'
CASE='eval_cad:1778653003088339968'
PINS={'SEAL.json':'a8341769d6733debe770bae6954da63ec4ab3b9ea3978d9ee242cc3b700dd193',
      'WORKER_PACKET.json':'6d3f198e47856720dd13186e0a24ed1141a59563d6d319ad24ce9371e159f306',
      'RESULT.json':'a947c44812ecfeba2a028c1ac3bcafa70110b6331ec181cac882777f24b82eb3'}
IMAGE_SHA='42538673f59293fa6f942dde65e4524236d4e8b6c98ec40db3ef9c58e591beb3'
STATES=['I','U','B','U','B','B','S','B']
POLICIES=('ALL_BI_REPLACE','RGB_NONFALLBACK_PRESERVE')
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
PURPLE='#a659e8';CYAN='#00c7c7';GOLD='#ffe04d';WHITE='#ffffff'


def digest(value):
    return hashlib.sha256(json.dumps(C.finite(value),sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def observed_BI(packet):
    states=packet['clean_input']['states'];result=[]
    for row in packet['clean_input']['visible_measurements']:
        k=int(row['native_corner'])
        if states[k]not in('B','I'):continue  # State gate before any observed xy.
        if row['state']!=states[k]:raise ValueError('Archived BI state differs')
        xy=np.asarray(row['xy'],float)
        if xy.shape!=(2,)or not np.isfinite(xy).all():raise ValueError('Finite original BI only')
        result.append(deepcopy(row))
    if [r['native_corner']for r in result]!=[0,2,4,5,7]:raise ValueError('Fixed BI archive changed')
    return result


def saved_finite_segment(group):
    """Use saved TLS coefficients and actual support extent; no TLS refit."""
    n=np.asarray(group['normal_unit_xy'],float);c=float(group['offset_px'])
    points=np.asarray(group['actual_inlier_xy'],float)
    if n.shape!=(2,)or points.ndim!=2 or points.shape[1]!=2 or len(points)<2:
        raise ValueError('Exact finite actual support required')
    if not np.isclose(n@n,1.,atol=1e-10):raise ValueError('Saved TLS unit normal required')
    tangent=np.array([-n[1],n[0]])
    lo,hi=(points@tangent).min(),(points@tangent).max()
    return np.stack((-c*n+lo*tangent,-c*n+hi*tangent))


def projection(family,pose,K):
    # native_t is a native-origin translation, not necessarily model center.
    return C.project(np.asarray(family['native_xyz']),np.asarray(pose['R']),
        np.asarray(pose['native_t']if'native_t'in pose else pose['t']),np.asarray(K))


def first_lowest_native(record):
    starts=[s for s in record['start_packet']['starts']if s['name']!='OWN_INITIAL_COARSE']
    if not starts:raise ValueError('Fixed original native start absent')
    start=starts[0]
    if start['native_objective']!=min(s['native_objective']for s in starts):
        raise ValueError('First archived native-ranked start is not lowest within the retained set')
    matches=[s for s in record['samples']if s['name']==start['name']]
    if len(matches)!=1:raise ValueError('Exact archived static sample required')
    return deepcopy(matches[0])


def _photo(ax,image,title):
    ax.imshow(image,origin='upper',interpolation='nearest')
    h,w=image.shape[:2];ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5)
    ax.set_aspect('equal');ax.set_title(title,fontsize=13,pad=10)
    ax.set_xlabel('original image x [px]');ax.set_ylabel('original image y [px]')


def _wire(ax,xy,*,color,label,style='--',width=1.6):
    for i,(a,b)in enumerate(EDGES):
        ax.plot(xy[[a,b],0],xy[[a,b],1],linestyle=style,color=color,linewidth=width,
                alpha=.9,label=label if i==0 else None)


def _BI(ax,records):
    for row in records:
        k=row['native_corner'];x,y=row['xy'];color=GOLD if row['state']=='B'else'#47bfff'
        ax.scatter([x],[y],s=40,c=color,edgecolors='black',linewidths=.7,zorder=8)
        dx=-35 if x>600 else 7;dy=14 if y<30 else-8
        ax.annotate(row['state']+str(k),(x,y),xytext=(dx,dy),textcoords='offset points',
            color=color,fontsize=11,weight='bold',bbox=dict(facecolor='black',alpha=.6,pad=1.2,edgecolor='none'))


def _U(ax,regions):
    from matplotlib.patches import Ellipse
    for region in regions:
        if region['state']!='U'or region['native_corner']not in(1,3):raise ValueError('Fixed U range only')
        ax.add_patch(Ellipse(region['center_xy'],2*region['half_axes_px'][0],
            2*region['half_axes_px'][1],angle=region['angle_deg'],fill=False,
            linestyle=':',edgecolor='#ff975a',linewidth=1.8,alpha=.85))
    # Deliberately no center marker, center label or pseudo point for U.


def _support(ax,groups):
    for i,group in enumerate(groups):
        points=np.asarray(group['actual_inlier_xy']);segment=saved_finite_segment(group)
        partial=group['proposal_grade']=='PARTIAL_LINE_PROPOSAL'
        ax.scatter(points[:,0],points[:,1],s=11,color=GOLD,alpha=.85,zorder=7,
                   label='saved actual NMS inliers'if i==0 else None)
        ax.plot(segment[:,0],segment[:,1],color=CYAN,linewidth=2.5,
            linestyle='--'if partial else'-',zorder=6,label='finite saved TLS (not a corner)'if i==0 else None)
        middle=segment.mean(0)
        ax.annotate('edge '+str(group['physical_native_edge'])+(' partial'if partial else''),
            middle,xytext=(6,6),textcoords='offset points',color=WHITE,fontsize=9,
            bbox=dict(facecolor='black',alpha=.65,pad=1.5,edgecolor='none'))


def render(*,root_actual_CAL_visualization_GO=False):
    if root_actual_CAL_visualization_GO is not True:raise PermissionError('Explicit fixed-CAL image visualization GO before IO')
    if OUTPUT.exists():raise FileExistsError('Preserve every static process render')
    for name,sha in PINS.items():
        if C.sha(SMOKE/name)!=sha:raise ValueError('Exact sealed smoke '+name+' required')
    if C.sha(IMAGE)!=IMAGE_SHA:raise ValueError('Exact approved first-CAL RGB required')
    source=Path(__file__).resolve();sources={str(source.relative_to(C.ROOT)):C.sha(source),
        str(Path(C.__file__).relative_to(C.ROOT)):C.sha(C.__file__)}
    inputs={str((SMOKE/name).relative_to(C.ROOT)):sha for name,sha in PINS.items()}
    inputs[str(IMAGE.relative_to(C.ROOT))]=IMAGE_SHA
    OUTPUT.mkdir(parents=True)
    C.write(OUTPUT/'SOURCE_PLAN_V1.json',dict(schema='STATIC_FIRST_CAL_ACTUAL_IMAGE_PROCESS_PLAN_V32',
        UTC=datetime.now(timezone.utc).isoformat(),case=CASE,source_sha256=sources,input_sha256=inputs,
        panels=['original full RGB','own independent coarse model prediction + original BI + U ranges',
            'ALL original native active finite actual support','SOURCE original native active finite actual support'],
        static_start_comparison='saved first lowest retained ORIGINAL_WD native start versus own coarse; before continuous/final selection',
        actual_primary_reference_GT_e_inputs=0,new_inference_or_optimization_calls=0,
        U_center_markers=0,S_observed_markers=0,GT_wires=0,external_occlusion_effectiveness_claim=False))
    snap=OUTPUT/'code_snapshot'/source.relative_to(C.ROOT);snap.parent.mkdir(parents=True);shutil.copyfile(source,snap)
    seal=C.read(SMOKE/'SEAL.json');packet=C.read(SMOKE/'WORKER_PACKET.json');result=C.read(SMOKE/'RESULT.json')
    if (seal['case']!=CASE or seal['assets']!={n:PINS[n]for n in('WORKER_PACKET.json','RESULT.json')}
            or packet['identifiers']['id']!=CASE or result['identifiers']['id']!=CASE
            or packet['identifiers']['partition']!='CAL'or packet['clean_input']['states']!=STATES
            or result['ready']is not True or result['continuous_optimizer_calls']!=0):
        raise ValueError('Fixed sealed CAL static-only contract changed')
    records=observed_BI(packet);K=packet['clean_input']['K'];regions=packet['physics']['u_regions']
    own=result['records'][POLICIES[0]]['ORIGINAL_WD']['family']
    coarse=own['coarse_pose'];coarse_xy=projection(own,coarse,K)
    selected={};plans={};sampled={}
    for policy in POLICIES:
        r=result['records'][policy]['ORIGINAL_WD'];plan=r['current_cost_plan']
        if digest(plan)!=r['current_cost_plan_sha256']:raise ValueError('Full archived core plan digest differs')
        if r['family']!=own:raise ValueError('Same ORIGINAL body/coarse required across policies')
        plans[policy]=deepcopy(plan);selected[policy]=first_lowest_native(r)
        sampled[policy]=dict(own_coarse=deepcopy(r['samples'][0]),first_lowest_native=deepcopy(selected[policy]),
            family=deepcopy(r['family']),pose_vector_scale=r['pose_vector_scale'],
            start_packet_sha256=r['start_packet']['start_packet_sha256'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    image=np.asarray(Image.open(IMAGE).convert('RGB'))
    if tuple(image.shape[:2])!=tuple(packet['physics']['image_hw']):raise ValueError('RGB/packet canvas mismatch')
    shutil.copyfile(IMAGE,OUTPUT/'source_rgb.png')
    fig,axs=plt.subplots(2,2,figsize=(16,16),layout='constrained')
    _photo(axs[0,0],image,'1. Actual original RGB — no overlay')
    _photo(axs[0,1],image,'2. Own coarse prediction + original B/I observations')
    _wire(axs[0,1],coarse_xy,color=PURPLE,label='purple dashed = predicted model wire')
    _U(axs[0,1],regions);_BI(axs[0,1],records)
    axs[0,1].text(.02,-.17,'Dots: original B/I only. Dotted arcs: U1/U3 ranges.\nS6: model wire only; no observed point.',
        transform=axs[0,1].transAxes,color='black',fontsize=10,va='top')
    axs[0,1].legend(loc='upper right',fontsize=9)
    for ax,policy,title in((axs[1,0],POLICIES[0],'3. ALL policy: actual finite support used by the native core'),
                            (axs[1,1],POLICIES[1],'4. SOURCE policy: actual finite support retained in the core')):
        groups=plans[policy]['active_groups'];_photo(ax,image,title);_support(ax,groups)
        ax.text(.02,-.17,f"{len(groups)} physical edges; owners {plans[policy]['new_owner_ids']}\n"
            'Cyan: finite saved TLS. Yellow: measured pixel support.\nDashed cyan: partial proposal; no precise new corner.',
            transform=ax.transAxes,color='black',fontsize=9,va='top')
        ax.legend(loc='upper right',fontsize=8)
    fig.suptitle('First fixed CAL: eval_cad:1778653003088339968\nStatic numerical connection only — no O/T here, no occlusion-effectiveness or accuracy claim',fontsize=15)
    fig.savefig(OUTPUT/'actual_process_4panel.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(16,9),layout='constrained')
    for ax,policy,title in zip(axs,POLICIES,('ALL policy','SOURCE policy')):
        r=sampled[policy];seed=r['first_lowest_native'];start_xy=projection(r['family'],seed,K)
        _photo(ax,image,title+' — before continuous optimization')
        _wire(ax,coarse_xy,color=PURPLE,label='own independent coarse: predicted',style='--')
        _wire(ax,start_xy,color=CYAN,label='first lowest retained native start: predicted',style='-',width=1.9)
        ax.legend(loc='upper right',fontsize=9)
        ax.text(.02,-.17,f"Own coarse J={r['own_coarse']['objective']:.6f}, guard pass={r['own_coarse']['evaluation']['accepted']}\n"
            f"{seed['name']}: J={seed['objective']:.6f}, guard pass={seed['evaluation']['accepted']}\n"
            'Native costs differ by source policy. No GT/error or final-method selection.',
            transform=ax.transAxes,color='black',fontsize=10,va='top')
    fig.suptitle('Saved fresh-PnP static starts (ORIGINAL_WD only) — both wires are model predictions',fontsize=15)
    fig.savefig(OUTPUT/'static_native_starts_2panel.png',dpi=150);plt.close(fig)
    C.write(OUTPUT/'PROCESS_DATA.json',dict(schema='STATIC_FIRST_CAL_ACTUAL_PROCESS_DATA_V32',case=CASE,states=STATES,
        source_sha256=sources,input_sha256=inputs,image_hw=packet['physics']['image_hw'],K=K,
        original_BI_archives=records,U_ranges=deepcopy(regions),S_observed_points=[],
        ORIGINAL_WD_only_in_figures=True,own_coarse_projection_xy=coarse_xy.tolist(),
        static_samples=sampled,full_core_plans=plans,full_core_plan_digests={p:digest(v)for p,v in plans.items()},
        finite_support={p:[dict(physical_native_edge=g['physical_native_edge'],component_id=g['component_id'],
            actual_inlier_xy=g['actual_inlier_xy'],TLS_segment_from_saved_coefficients_and_actual_extent=saved_finite_segment(g).tolist(),
            proposal_grade=g['proposal_grade'],activation_owners=g['activation_owners'])for g in plans[p]['active_groups']]for p in POLICIES},
        pose_semantics='native_t is the translation of the native XYZ origin; model center=R_cf@mean(native_xyz)+native_t; Rphysical=R_cf@Q',
        saved_pose_vectors_and_scales_retained=True,all_core_plan_fields_preserved=True,
        U_observed_point_markers=0,S_observed_point_markers=0,GT_wire_or_e_pose=0,
        no_new_RGB_feature_query_NN_TLS_fit_PnP_or_optimizer=True,all_metrics_and_accuracy_claims=False))
    lines=['# 첫 CAL 프레임: 실제 영상과 정지 PnP 연결 확인','',
        '`eval_cad:1778653003088339968` — 상태: `I0, U1, B2, U3, B4, B5, S6, B7`.','',
        '![실제 사진·초기 예측·유한 선 지지 비교](actual_process_4panel.png)','',
        '보라색 점선은 독립 초기 coarse 자세가 예측한 모델이다. B/I 점은 봉인된 영상 관측이며, U1/U3는 넓은 불확실 타원 범위만 표시했다. U 중심점과 S6 관측점은 그리지 않았다. S6에 연결된 모델 wire는 예측이다.','',
        '청록색 선과 노란 표본은 저장된 실제 NMS/TLS 관측의 유한 지지다. 선을 무한 연장하지 않았고 새 교점이나 정밀 KP를 만들지 않았다. 청록 점선은 partial 제안이다. ALL은 4개 물리변·5개 owner, SOURCE는 원 RGB 관측을 보존하면서 fallback owner에 걸린 2개 물리변·2개 owner를 사용한다.','',
        '![연속 최적화 전 정지 시작 자세](static_native_starts_2panel.png)','',
        '| 정책 | own coarse native J / guard | 첫 최저 retained native start J / guard |',
        '|---|---:|---:|']
    for p in POLICIES:
        a=sampled[p]['own_coarse'];z=sampled[p]['first_lowest_native']
        lines.append(f"| {p} | {a['objective']:.8f} / {a['evaluation']['accepted']} | `{z['name']}` {z['objective']:.8f} / {z['evaluation']['accepted']} |")
    lines+=['','두 wire 모두 모델 예측이다. 표시한 fresh PnP 시작 자세는 ORIGINAL_WD의 보존된 native 시작집합 중 첫 최저 비용 항목이며, 연속 최적화나 최종 방법 선택 결과가 아니다. SOURCE의 coarse 비용이 더 낮아도 guard를 통과하지 않는다. 정책마다 목적함수가 다르므로 정책 간 비용으로 우열을 정하지 않는다.','',
        '이 사진에는 외부 O/T 상태가 없다. 실제 수치 연결을 확인하는 CAL 진단이며, 가림처리 효과·정확도 개선·정답 일치를 증명하지 않는다. GT wire, e 자세, 새 추론·특징 질의·선 적합·PnP·최적화는 사용하지 않았다.','',
        '자세의 `native_t`와 모델 중심은 구분한다: `center = R_cf @ mean(native_xyz) + native_t`, `R_physical = R_cf @ Q`. 전체 원본 사진을 유지하며, `PROCESS_DATA.json`에는 원 BI archive, 유한 실제 지지, 전체 core plan, ORIGINAL family, 저장된 pose vector/scale/native_t를 보존했다.','',
        '[원본 RGB](source_rgb.png) · [수치와 provenance](PROCESS_DATA.json) · [입력·소스 계획](SOURCE_PLAN_V1.json) · [렌더 봉인](RENDER_SEAL.json)','']
    (OUTPUT/'REPORT_KO.md').write_text('\n'.join(lines))
    assets={p.name:C.sha(p)for p in sorted(OUTPUT.iterdir())if p.is_file()}
    for rel,sha in inputs.items():
        if C.sha(C.ROOT/rel)!=sha:raise ValueError('Frozen visualization input changed')
    C.write(OUTPUT/'RENDER_SEAL.json',dict(schema='SEALED_STATIC_FIRST_CAL_ACTUAL_PROCESS_V32',
        source_sha256=sources,input_sha256=inputs,assets=assets,case=CASE,
        numerical_runtime_or_reference_changes=0,primary_or_GT_reads=0,
        new_inference_feature_query_or_optimizer=0,visual_QA_pending=True))
    print(str(OUTPUT),C.sha(OUTPUT/'RENDER_SEAL.json'))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-actual-CAL-visualization-GO',action='store_true')
    args=parser.parse_args();render(root_actual_CAL_visualization_GO=args.root_actual_CAL_visualization_GO)
