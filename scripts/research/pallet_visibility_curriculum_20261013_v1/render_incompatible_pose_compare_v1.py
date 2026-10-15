"""Actual seven-image PoseFix/current comparison; render frozen poses only.

Image overlays use each method's actual native cuboid and stored pose. Reference
corners use the separate reference body basis. No model is fitted or reordered
to make it look correct. Physical axes are displayed exactly as stored; metric
rotation alone uses the existing C2 scorer. The canonical numbering diagram
uses only the geometry-phase already allowed by the original compatibility
test, explicitly labeled, and is not a new camera pose.
"""
from __future__ import annotations

import os
for _thread_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_thread_key] = '1'

from copy import deepcopy
import html
import json
from pathlib import Path
import shutil

import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager, patheffects
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.spatial.transform import Rotation

from . import common as C
from . import evaluate_u_regions_v4_5 as U
from . import statewise_evaluation_v4 as E
from .statewise_uncertainty_calibration_v4 import _reference_alignment
from .scoring import pose_metric

OUTPUT = U.BOUNDARY / 'incompatible_pose_compare_v1'
PF = E.COHORT / 'posefix_sub_comparison_v1'
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
COLORS = {'PoseFix': '#00d8ff', '현재': '#ff8733', 'reference': '#70ff54'}
FONT = Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')


def _bound_inputs():
    # Both pose sources must be sealed before reference/annotation access.
    u_seal = C.read(U.OUTPUT / 'TEST_PREDICTION_SEAL.json')
    u_protocol = C.read(U.OUTPUT / 'PROTOCOL.json')
    assert u_seal['before_new_partition_pose_scoring']
    assert C.sha(U.OUTPUT / 'TEST_PREDICTIONS.jsonl.gz') == u_seal['predictions_sha256']
    assert C.sha(U.OUTPUT / 'PROTOCOL.json') == u_seal['protocol_sha256']
    E._verify_sources(u_protocol['source_sha256'])
    E._verify_sources(u_protocol['input_sha256'])
    E._verify_sources(u_protocol['protected_user_sha256'])
    u_score_seal = C.read(U.OUTPUT / 'SCORE_SEAL.json')
    assert C.sha(U.OUTPUT / 'TEST_POSE_SCORE.json') == u_score_seal['TEST_pose_score_sha256']
    p_seal = C.read(PF / 'PREDICTION_SEAL.json')
    assert p_seal['before_scoring'] and p_seal['complete'] and p_seal['severe_frames'] == 0
    assert C.sha(PF / 'PREDICTIONS.jsonl.gz') == p_seal['predictions_sha256']
    assert C.sha(PF / 'PROTOCOL.json') == p_seal['protocol_sha256']
    current = {r['id']: r for r in C.rows(U.OUTPUT / 'TEST_PREDICTIONS.jsonl.gz')}
    pfix = {r['id']: r for r in C.rows(PF / 'PREDICTIONS.jsonl.gz')
            if r['partition'] == 'TEST' and r['method'] == 'POSEFIX_RAW'}
    assert len(current) == len(pfix) == 49 and set(current) == set(pfix)
    score = C.read(U.OUTPUT / 'TEST_POSE_SCORE.json')
    selected = [r for r in score['rows'] if not r['physical_native_compatible']]
    assert len(selected) == 7
    selected.sort(key=lambda r: r['metrics'][U.ARMS[1]]['translation_cm'], reverse=True)
    label_path = E.COHORT / 'test_reused_v1/ISOLATED_TEST_SCORING.json'
    assert C.sha(label_path) == score['target_binding']['sha256']
    labels = C.read(label_path)['rows']
    paths = [U.OUTPUT / n for n in ('TEST_PREDICTIONS.jsonl.gz','TEST_PREDICTION_SEAL.json',
             'TEST_POSE_SCORE.json','SCORE_SEAL.json','PROTOCOL.json')]
    paths += [PF / n for n in ('PREDICTIONS.jsonl.gz','PREDICTION_SEAL.json','PROTOCOL.json')]
    paths += [label_path, Path(__file__), FONT]
    bindings = {str(p): C.sha(p) for p in paths}
    return selected, current, pfix, labels, bindings, u_protocol


def _project(points, pose, K):
    return C.project(points, pose['R_cf'], pose['centroid'], K)


def _physical_axis_projection(R, t, K, length):
    # Full unsigned X/Z line also displays the C2 sign ambiguity correctly.
    points = np.array([[0,0,0], [length,0,0], [-length,0,0], [0,0,length], [0,0,-length]])
    return C.project(points, R, t, K)


def _wire(ax, points, color, *, dashed=False, numbering=None):
    for a, b in EDGES:
        q = points[[a,b]]
        ax.plot(q[:,0], q[:,1], color='black', linewidth=3.9, alpha=.55,
                linestyle='--' if dashed else '-')
        ax.plot(q[:,0], q[:,1], color=color, linewidth=2.1,
                linestyle='--' if dashed else '-')
    if numbering:
        ax.scatter(points[:,0], points[:,1], s=24, facecolors='none' if dashed else color,
                   edgecolors=color, linewidths=1.5)
        for k, q in enumerate(points):
            ax.annotate(f'{numbering}{k}', q, xytext=(4,9 if dashed else -12), textcoords='offset points',
                color=color, fontsize=10, fontweight='bold',
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground='#111')], clip_on=True)


def _axes(ax, xy, color, prefix):
    ax.scatter(*xy[0], s=26, c=color, edgecolors='#111', zorder=6)
    for i, j, name in ((1,2,'X'), (3,4,'Z')):
        ax.plot(xy[[i,j],0], xy[[i,j],1], color=color, linewidth=2.3, alpha=.95)
        ax.annotate(prefix+name, xy[i], xytext=(3,4), textcoords='offset points', color=color,
            fontsize=11, fontweight='bold', clip_on=True,
            path_effects=[patheffects.withStroke(linewidth=2.3, foreground='#111')])


def _roi(qs, image_hw):
    h,w = image_hw
    q=np.vstack(qs); q=q[np.isfinite(q).all(1)]
    low=q.min(0); high=q.max(0);size=high-low
    pad=np.maximum(size*.12, [35,35]);low-=pad;high+=pad
    # Only the photographic display is cropped. No pose/domain is clipped.
    x0=max(0,float(low[0]));x1=min(w,float(high[0]));y0=max(0,float(low[1]));y1=min(h,float(high[1]))
    return [x0,y0,x1,y1]


def _number_diagram(ax, model, reference, colors, phase):
    ids=[k for k in range(8) if model[k,1]>0]
    ax.set_aspect('equal'); ax.set_facecolor('#f4f7fb')
    model2=model[:,[0,2]];ref2=reference[:,[0,2]]
    # Top plane only, four distinct corners; the full eight-vertex gaps are in JSON.
    sequence=[2,3,7,6,2]
    for points, color, line in ((ref2,colors['reference'],'--'),(model2,colors['현재'],'-')):
        ax.plot(points[sequence,0],points[sequence,1],color=color,lw=2,linestyle=line)
    for k in ids:
        p,g=model2[k],ref2[k]
        ax.annotate('',xy=g,xytext=p,arrowprops={'arrowstyle':'->','color':'#8190a5','lw':1.4})
        for point,prefix,color,dy in ((p,'P',colors['현재'],.035),(g,'G',colors['reference'],-.07)):
            ax.scatter(*point,s=42,c=color,edgecolors='#26364a',zorder=4)
            ax.text(point[0]+.025,point[1]+dy,f'{prefix}{k}',color='#17263c',fontsize=11,fontweight='bold',zorder=5)
    ax.axhline(0,color='#bac5d4',lw=.7);ax.axvline(0,color='#bac5d4',lw=.7)
    ax.set_xlabel('물체 X (m)');ax.set_ylabel('물체 Z (m)')
    ax.margins(.20)
    ax.set_title(f'같은 번호끼리 연결한 3D 윗면 / 기존 C2 검사 {phase*180}° 단계',fontsize=12,pad=12)
    ax.grid(alpha=.15)


def _render(case, image, path, details):
    fig=plt.figure(figsize=(18,9.7 if details else 5.7),dpi=130,facecolor='white')
    grid=GridSpec(2 if details else 1,3,figure=fig,height_ratios=[1.04,.8] if details else [1],
                  hspace=.48,wspace=.07)
    titles=['실제 원본 사진',f"PoseFix 원본 · {case['metrics']['PoseFix']['translation_cm']:.2f}cm / {case['metrics']['PoseFix']['rotation_deg']:.2f}°",
            f"현재 방법 · {case['metrics']['현재']['translation_cm']:.2f}cm / {case['metrics']['현재']['rotation_deg']:.2f}°"]
    axlist=[]
    for column in range(3):
        ax=fig.add_subplot(grid[0,column]);ax.imshow(image);ax.axis('off');ax.set_title(titles[column],fontsize=14,pad=13)
        x0,y0,x1,y1=case['display_roi_xyxy'];ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);axlist.append(ax)
    for column,name in ((1,'PoseFix'),(2,'현재')):
        pred=np.asarray(case['projected_corners'][name]);gt=np.asarray(case['projected_corners']['reference_body'])
        _wire(axlist[column],gt,COLORS['reference'],dashed=True)
        _wire(axlist[column],pred,COLORS[name],numbering='P' if details else None)
        if details:
            mapped=np.asarray(case['projected_corners']['reference_mapped_native'])
            # Reference dots have a mapped semantic label; they are not detections.
            for k,q in enumerate(mapped):
                axlist[column].annotate(f'G{k}',q,xytext=(3,8),textcoords='offset points',color=COLORS['reference'],
                    fontsize=9,fontweight='bold',clip_on=True,
                    path_effects=[patheffects.withStroke(linewidth=2.3,foreground='#111')])
            _axes(axlist[column],np.asarray(case['projected_physical_axes'][name]),COLORS[name],'')
            _axes(axlist[column],np.asarray(case['projected_physical_axes']['reference']),COLORS['reference'],'G-')
    if details:
        top=fig.add_subplot(grid[1,0]);_number_diagram(top,np.asarray(case['registered_geometry']['native_physical_best_C2']),
            np.asarray(case['registered_geometry']['mapped_reference_physical']),COLORS,case['registered_geometry']['C2_phase_index'])
        text_ax=fig.add_subplot(grid[1,1:]);text_ax.axis('off')
        dimensions=case['model_details']
        text=(
            '이 번호 비교가 42장 / 7장 구분의 기준입니다.\n\n'
            'P2 ↔ G2처럼 같은 번호로 연결된 두 3D 코너가\n'
            '같은 물체 좌표에 놓이는지 검사했습니다.\n'
            f"이 사례의 최대 코너 차이: {case['registered_geometry']['maximum_gap_m']:.2f}m\n\n"
            f"PoseFix / 현재 native 축 크기 X×Y×Z: {dimensions['PoseFix']['cf_extents']}m\n"
            f"기준 body 축 크기 X×Y×Z: {dimensions['reference_body']['extents']}m\n\n"
            '사진 위 P/G 점·박스·축은 모두 저장된 모델의 투영입니다.\n'
            '초록 점선 = 기존 정답 기준 모델, 청록 = PoseFix, 주황 = 현재.\n'
            '사진 위 자세·축에는 90° 재정렬을 적용하지 않았습니다.\n'
            '회전 오차 숫자만 기존 C2(180° 대칭) 평가를 따릅니다.')
        text_ax.text(.035,.96,text,va='top',fontsize=12.3,linespacing=1.5,color='#203149')
    fig.suptitle(case['id'],fontsize=15,fontweight='bold',y=.975)
    fig.subplots_adjust(top=.875 if not details else .89,bottom=.045,left=.035,right=.985)
    fig.savefig(path,dpi=130,facecolor='white');plt.close(fig)


def _page(cases):
    articles=[]
    for index,case in enumerate(cases,1):
        image=html.escape(case['assets']['details']);basic=html.escape(case['assets']['comparison'])
        name=html.escape(case['id']);metrics=case['metrics']
        articles.append(f'''<article id="case-{index}"><div class="case-head"><h2>{index}/7 · {name}</h2><a href="{image}" target="_blank">큰 이미지 열기 ↗</a></div>
<div class="stats"><span class="pf">PoseFix: {metrics['PoseFix']['translation_cm']:.2f}cm / {metrics['PoseFix']['rotation_deg']:.2f}°</span><span class="current">현재: {metrics['현재']['translation_cm']:.2f}cm / {metrics['현재']['rotation_deg']:.2f}°</span><span>기존 번호 대응 검사 차이 {case['registered_geometry']['maximum_gap_m']:.2f}m</span></div>
<img class="comparison-image" src="{image}" data-details="{image}" data-basic="{basic}" alt="{name} 실제 사진 · PoseFix와 현재 자세 및 코너번호 비교" loading="{'eager' if index==1 else 'lazy'}">
<p class="caption">원본 / PoseFix / 현재를 같은 영상·카메라로 표시했습니다. P0~P7은 각 방법의 모델 코너, G0~G7은 기존 reference를 고정 mapping으로 연결한 모델 코너입니다. 실제 영상에서 검출한 점과 구분합니다.</p></article>''')
    jumps=''.join(f'<a href="#case-{i}">{i}. {html.escape(c["id"].split(":")[0])}</a>' for i,c in enumerate(cases,1))
    return '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실제 7장 · PoseFix / 현재 자세와 코너번호 비교</title>
<style>*{box-sizing:border-box}body{margin:0;background:#eef3f8;color:#17263c;font-family:Arial,'Noto Sans CJK KR','NanumGothic',sans-serif}main{max-width:1680px;margin:auto;padding:32px 22px}header{background:white;padding:26px;border-radius:15px}h1{font-size:28px;margin:0 0 14px}p{line-height:1.7;margin:10px 0}.explain{font-size:18px;max-width:1120px}.legend{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}.legend span{padding:8px 12px;background:#f2f6fa;border-radius:7px}.pf{color:#00647e}.current{color:#b85000}.gt{color:#276711}.controls{position:sticky;top:0;z-index:20;background:#ffffffef;padding:12px;margin:20px 0;display:flex;gap:10px;align-items:center;border:1px solid #cdd9e6;border-radius:9px;flex-wrap:wrap}button{border:1px solid #b5c6d9;background:white;border-radius:7px;padding:9px 14px;color:#253a51;cursor:pointer;font-size:15px}button[aria-pressed=true]{background:#163956;color:white}nav{display:flex;gap:12px;flex-wrap:wrap;font-size:14px}a{color:#1f598e}article{scroll-margin-top:85px;background:white;margin:24px 0;padding:20px 16px;border:1px solid #d8e1ea;border-radius:14px}.case-head{display:flex;justify-content:space-between;gap:15px;align-items:center;flex-wrap:wrap}h2{font-size:19px;margin:0}.stats{display:flex;gap:20px;flex-wrap:wrap;margin:16px 0;font-weight:600}.comparison-image{display:block;width:100%;height:auto;border:1px solid #e3eaf1}.caption{font-size:14px;color:#51637a}.small{font-size:14px;color:#51637a}footer{padding:22px}.intro-box{background:#f1f6fc;border-left:4px solid #557baf;padding:12px 17px}@media(max-width:650px){main{padding:14px 7px}header{padding:18px}h1{font-size:23px}.explain{font-size:16px}article{padding:16px 8px}h2{font-size:15px}.stats{gap:9px;font-size:13px}.case-head a{font-size:13px}}
</style></head><body><main><header><h1>실제 7장: PoseFix와 현재 자세 비교</h1>
<p class="explain">“42장 / 7장”은 정사각형·직사각형 구분이 아닙니다. <strong>같은 KP 번호가 두 3D 모델에서 같은 모서리를 뜻하는지</strong> 검사한 구분입니다. 사진 위 박스와 아래 번호 배치를 함께 보면 차이를 볼 수 있습니다.</p>
<div class="intro-box">이 페이지는 현재 모델 등록에서 그 검사를 통과하지 않은 7장을 모두 보여줍니다. 이 7장에서는 PoseFix도 같은 크기·등록축을 선택했습니다. 사진·정답을 잘못된 데이터라고 판정한 표시는 아닙니다. 성능 비교에는 전체 49장을 포함했습니다.</div>
<div class="legend"><span class="gt">초록 점선 = 기존 정답 모델 투영</span><span class="pf">청록 = PoseFix 원본 모델 투영</span><span class="current">주황 = 현재 방법 모델 투영</span></div>
<p class="small">PoseFix RAW / 현재 최신 B·I 보정 + O·T·U 영역 PnP. 저장된 결과를 시각화했고, 추론·PnP 재계산은 하지 않았습니다. 사진 위 축은 물리 X/Z축이며 원래 저장된 방향을 유지했습니다. 숫자 회전 오차는 C2 대칭 평가입니다.</p>
<nav>'''+jumps+'''</nav></header><div class="controls"><strong>표시:</strong><button id="details" aria-pressed="true">코너번호·축·설명 보기</button><button id="basic" aria-pressed="false">박스 비교만 보기</button><span class="small">각 사례의 “큰 이미지 열기”로 확대할 수 있습니다.</span></div>
'''+''.join(articles)+'''<footer>원본 이미지 및 두 방법의 자세·reference 좌표는 CASES.json에 기록되어 있습니다. GT는 후평가 표시용이며 추론 입력으로 사용하지 않았습니다.</footer></main><script>
function show(mode){document.querySelectorAll('.comparison-image').forEach(img=>{img.src=img.dataset[mode]});document.getElementById('details').setAttribute('aria-pressed',String(mode==='details'));document.getElementById('basic').setAttribute('aria-pressed',String(mode==='basic'));document.body.dataset.mode=mode}document.getElementById('details').onclick=()=>show('details');document.getElementById('basic').onclick=()=>show('basic');document.body.dataset.mode='details';
</script></body></html>'''


def main():
    if OUTPUT.exists():raise FileExistsError('Preserve existing actual image comparison')
    selected,current,pfix,labels,bindings,protocol=_bound_inputs()
    font_manager.fontManager.addfont(str(FONT));plt.rcParams['font.family']='NanumGothic';plt.rcParams['axes.unicode_minus']=False
    OUTPUT.mkdir(parents=True);(OUTPUT/'assets').mkdir()
    cases=[]
    for scored in selected:
        identity=scored['id'];row=current[identity];geometry=row['runtime']['initial_geometry'];label=labels[identity];truth=label['target']['truth']
        pose_cur=row['arms'][U.ARMS[1]]['actual_pose'];pose_pf=pfix[identity]['actual_pose'];K=np.asarray(geometry['K'])
        source=C.ROOT/row['image'];assert C.sha(source)==row['image_sha256'];bindings[str(source)]=C.sha(source)
        bgr=cv2.imread(str(source));assert bgr is not None and list(bgr.shape[:2])==geometry['image_hw'];image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        xyz_cur=np.asarray(geometry['native_xyz']);xyz_pf=C.cuboid(*pose_pf['cf_extents']);xyz_gt=C.cuboid(*truth['body_xyz'])
        qcur=_project(xyz_cur,pose_cur,K);qpf=_project(xyz_pf,pose_pf,K);qgt=C.project(xyz_gt,truth['body_R'],truth['t'],K)
        parity={}
        for name,xyz,pose,q in (('PoseFix',xyz_pf,pose_pf,qpf),('현재',xyz_cur,pose_cur,qcur)):
            Q=np.asarray(pose['R_cf']).T@np.asarray(pose['R_physical'])
            physical=C.project(xyz@Q,pose['R_physical'],pose['centroid'],K)
            parity[name]=float(np.max(np.abs(physical-q)));assert parity[name]<1e-8
        assert np.allclose(qcur,row['arms'][U.ARMS[1]]['keypoint_readout']['rigid_model_corners'],atol=1e-7,rtol=0)
        mapping=np.asarray(label['metadata']['native_to_reference_corner_ids'],int)
        alignment=_reference_alignment(row,label);assert not alignment['compatible']
        assert abs(alignment['geometry_gap_m']-scored['geometry_gap_m'])<1e-12
        Qcur=np.asarray(geometry['physical_registration_Q']);Qref=np.asarray(truth['body_R']).T@np.asarray(truth['R'])
        symmetry=Rotation.from_rotvec([0,alignment['phase']*np.pi,0]).as_matrix()
        registered=xyz_cur@Qcur@symmetry.T;reference=xyz_gt[mapping]@Qref
        gaps=np.linalg.norm(registered-reference,axis=1);assert abs(gaps.max()-alignment['geometry_gap_m'])<1e-12
        axis_length=float(min(truth['xyz'][0],truth['xyz'][2])*.38)
        metrics={'PoseFix':pose_metric(pose_pf,truth),'현재':pose_metric(pose_cur,truth)}
        assert metrics['현재']==scored['metrics'][U.ARMS[1]]
        identifier=identity.replace(':','__')
        case=C.finite(dict(id=identity,image=str(source),image_sha256=row['image_sha256'],image_hw=geometry['image_hw'],K=K,
            current_states=row['states'],metrics=metrics,pose_sources={'PoseFix':pose_pf,'현재':pose_cur,'reference':truth},
            model_details={'PoseFix':{'cf_extents':pose_pf['cf_extents'],'family':pose_pf['selected_hypothesis'],
                'Q':(np.asarray(pose_pf['R_cf']).T@np.asarray(pose_pf['R_physical']))},
                '현재':{'cf_extents':pose_cur['cf_extents'],'Q':Qcur},
                'reference_body':{'extents':truth['body_xyz'],'Q':Qref},'reference_physical':{'extents':truth['xyz']}},
            projected_corners={'PoseFix':qpf,'현재':qcur,'reference_body':qgt,'reference_mapped_native':qgt[mapping]},
            projected_physical_axes={'PoseFix':_physical_axis_projection(pose_pf['R_physical'],pose_pf['centroid'],K,axis_length),
                '현재':_physical_axis_projection(pose_cur['R_physical'],pose_cur['centroid'],K,axis_length),
                'reference':_physical_axis_projection(truth['R'],truth['t'],K,axis_length)},
            display_roi_xyxy=_roi([qcur,qpf,qgt],geometry['image_hw']),
            native_to_reference_corner_ids=mapping,
            registered_geometry={'native_physical_best_C2':registered,'mapped_reference_physical':reference,
                'C2_phase_index':alignment['phase'],'maximum_gap_m':alignment['geometry_gap_m'],'gaps_m':gaps,
                'diagnostic_pose_independent':True,'camera_pose_90deg_realigned':False},
            projection_native_physical_parity_px=parity,
            assets={'details':f'assets/{identifier}_details.png','comparison':f'assets/{identifier}_comparison.png'}))
        _render(case,image,OUTPUT/case['assets']['details'],True)
        _render(case,image,OUTPUT/case['assets']['comparison'],False)
        cases.append(case)
        print('rendered',identity,metrics,flush=True)
    C.write(OUTPUT/'CASES.json',{'schema':'ACTUAL_INCOMPATIBLE_PF_CURRENT_V1','cases':cases,'reference_pose_used_for_display_only':True})
    (OUTPUT/'index.html').write_text(_page(cases),encoding='utf8')
    dest=OUTPUT/'code_snapshot'/Path(__file__).name;dest.parent.mkdir();shutil.copyfile(__file__,dest)
    for path,digest in bindings.items():assert C.sha(path)==digest
    E._verify_sources(protocol['protected_user_sha256'])
    E._verify_sources(protocol['source_sha256'])
    C.write(OUTPUT/'RENDER_RECEIPT.json',dict(schema='ACTUAL_SEVEN_IMAGE_POSE_COMPARE_RENDER_V1',
        cases=len(cases),TEST_all_frames=49,reference_native_incompatible=7,severe=0,
        before_render_both_prediction_seals_verified=True,input_sha256=bindings,
        output_sha256={str(p.relative_to(OUTPUT)):C.sha(p) for p in OUTPUT.rglob('*') if p.is_file()},
        only_frozen_pose_projection=True,new_inference_calls=0,new_PnP_calls=0,
        photographic_90deg_or_C2_pose_realignment=False,
        native_diagram_uses_existing_declared_geometry_C2_phase=True,
        protected_user_and_sealed_source_files_unchanged=True))
    print('OUTPUT',OUTPUT,flush=True)


if __name__=='__main__':main()
