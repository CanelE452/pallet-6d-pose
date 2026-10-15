"""Readable saved comparison: both raw and N3+SubPix receive the same BI API.

Only saved observations, masks, poses and postseal scores are displayed. No
state classifier, image-feature query, correction, pose fit or reference reader
is invoked here. Initial detector channels remain predictions; S/O/T/U are
never promoted to precise B/I observations.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
import html
import json
import shutil

import numpy as np

from . import common as C
from . import render_raw_yolo_N3_subpix_gain10_process_v1 as OLD

B = OLD.B
RUN = B/'new_gain10_paired_BI_v2'
INPUT = B/'raw_ns_BI_gain10_input_v1/INPUTS.json'
OUTPUT = B/'new_gain10_paired_BI_process_v1'
BRANCHES = ('RAW_YOLO', 'N3_SUBPIX')
FIXED_IDS = OLD.FIXED_IDS
FONT = OLD.FONT
EDGES = OLD.EDGES
TITLES = {'RAW_YOLO':'N3+SubPix 없이', 'N3_SUBPIX':'N3+SubPix 사용'}
STATE_COLORS = {'B':'#00a4c4','I':'#8847d9','S':'#7b4cbf','O':'#bd4c86','T':'#b36400','U':'#676767'}


def relative(path):
    return str(Path(path).resolve().relative_to(C.ROOT))


def binding(path):
    return dict(path=relative(path),sha256=C.sha(path))


def finite_xy(value):
    q=np.asarray(value,float)
    return q.shape==(2,) and np.isfinite(q).all() and not np.all(q==-1)


def measurement_rows(branch):
    """State gate precedes raw query access; only saved B/I can be arrows."""
    observation=branch['observation'];states=observation['states']
    source=branch['input'];raw=np.asarray(source['native_points'],float)
    correction=observation['visible_correction']
    records={int(r['native_corner']):r for r in correction['records']}
    recovery={int(r['native_corner']):r for r in correction.get('boundary_recovery_records',[])}
    rows=[];seen=set()
    for measurement in observation['frozen_visible_measurements']:
        k=int(measurement['native_corner'])
        if k in seen or k not in range(8) or states[k] not in ('B','I'):
            raise ValueError('Only unique branch B/I measurements may be displayed')
        seen.add(k)
        q=np.asarray(measurement['xy'],float)
        if measurement['state']!=states[k] or not finite_xy(q) or not finite_xy(raw[k]):
            raise ValueError('Saved actual B/I measurement/query mismatch')
        rec=records[k];extra=recovery.get(k)
        reason=rec.get('reason','not_saved')
        if extra is not None and extra.get('pixel_queried'):
            reason+='; recovery: '+str(extra.get('reason','not_saved'))
        rows.append(dict(native_corner=k,state=states[k],before=raw[k].tolist(),after=q.tolist(),
            movement_px=float(np.linalg.norm(q-raw[k])),source=measurement['source'],
            corrected=bool(rec.get('corrected',False)),reason=reason,
            covariance_px2=deepcopy(measurement['covariance_px2'])))
    if seen!={k for k,state in enumerate(states) if state in ('B','I')}:
        raise ValueError('Every visible B/I, including unchanged fallback, must remain')
    return rows


def projection(frame,pose):
    return OLD.projection(pose,frame['K'],frame['xyz'])[0]


def common_ROI(frame,branches):
    """Same viewport from initial predictions and actual BI only, no final/GT."""
    h,w=frame['raw_hw'];points=[]
    for name in BRANCHES:
        branch=branches[name]
        points.append(np.asarray(branch['input']['native_points'],float)[:8])
        points.append(projection(frame,branch['input']['coarse_pose']))
        points.append(np.asarray([r['after'] for r in measurement_rows(branch)],float).reshape(-1,2))
    q=np.concatenate(points);q=q[np.isfinite(q).all(1)]
    q=q[(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)]
    if not len(q):return [0.,float(w),0.,float(h)]
    lo,hi=q.min(0),q.max(0);center=(lo+hi)/2
    span=np.maximum(hi-lo,[240.,180.]);half=span/2+np.maximum(32.,.2*span)
    return [max(0.,float(center[0]-half[0])),min(float(w),float(center[0]+half[0])),
            max(0.,float(center[1]-half[1])),min(float(h),float(center[1]+half[1]))]


def saved_support(branch):
    """Stored finite RGB inliers, no TLS refit or geometric line extension."""
    correction=branch['observation']['visible_correction'];states=branch['observation']['states']
    records={int(r['native_corner']):r for r in correction['records']}
    recovery={int(r['native_corner']):r for r in correction.get('boundary_recovery_records',[])}
    lines=[]
    for k,state in enumerate(states):
        if state not in ('B','I'):continue
        rec=records[k];extra=recovery.get(k,{})
        rays=extra.get('selected_rays',[]) if extra.get('recovered') else rec.get('rays',[])
        for ray in rays:
            if not ray.get('supported'):continue
            xy=np.asarray(ray.get('observed_points',[]),float)
            keep=np.asarray(ray.get('inlier_mask',[]),bool)
            if xy.ndim!=2 or xy.shape[1]!=2 or keep.shape!=(len(xy),):continue
            xy=xy[keep]
            if len(xy)>=2 and np.isfinite(xy).all():
                lines.append(dict(native_corner=k,edge=ray.get('edge'),xy=xy.tolist()))
    return lines


def state_table(branch):
    observation=branch['observation'];records=observation['state_observation']['records']
    if len(records)!=8:raise ValueError('All eight saved states required')
    confidence=branch['input']['keypoints_conf'];result=[]
    for k,record in enumerate(records):
        state=observation['states'][k]
        if record['native_id']!=k or record['state']!=state:
            raise ValueError('Saved state records not in native order')
        if state=='S' and (record.get('rgb_evaluated') or record.get('p_visible') is not None):
            raise ValueError('S must have no local image classifier query')
        result.append(dict(native_corner=k,state=state,p_visible=record.get('p_visible'),
            original_YOLO_confidence=confidence[k],rgb_evaluated=record.get('rgb_evaluated',False),
            reason=record.get('reason'),initial_role=record.get('initial_role')))
    return result


def draw_tiles(frame,row,folder):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager,patheffects
    from PIL import Image
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False})
    imagepath=C.ROOT/frame['image']
    if C.sha(imagepath)!=frame['image_sha256']:raise ValueError('Exact same original image required')
    image=np.asarray(Image.open(imagepath).convert('RGB'))
    if tuple(image.shape[:2])!=tuple(frame['raw_hw']):raise ValueError('Original camera canvas differs')
    folder.mkdir();shutil.copyfile(imagepath,folder/'source_rgb.png')
    branches=row['branches'];roi=common_ROI(frame,branches);proof={}
    def picture(title):
        fig,ax=plt.subplots(figsize=(10,7.4),layout='constrained');ax.imshow(image,interpolation='nearest')
        ax.set_xlim(roi[:2]);ax.set_ylim(roi[3],roi[2]);ax.set_aspect('equal');ax.set_title(title,fontsize=16,pad=12)
        ax.set_xlabel('원본 x [px]');ax.set_ylabel('원본 y [px]');return fig,ax
    def wire(ax,xy,color,label):
        for i,(a,b) in enumerate(EDGES):ax.plot(xy[[a,b],0],xy[[a,b],1],color=color,lw=1.7,label=label if i==0 else None)
    def point(ax,xy,k,color,marker='o',size=38):
        ax.scatter(*xy,color=color,marker=marker,s=size,zorder=7)
        text=ax.annotate(str(k),xy,xytext=(7,10 if k%2 else -14),textcoords='offset points',fontsize=12,color=color,weight='bold')
        text.set_path_effects([patheffects.withStroke(linewidth=2.3,foreground='white')])
    for name in BRANCHES:
        branch=branches[name];observed=branch['observation'];states=observed['states']
        if branch['input']!=frame['arms'][name]:raise ValueError('Same sealed branch input required')
        if observed['original_raw_points']!=branch['input']['native_points']:
            raise ValueError('Observation belongs to another input')
        before=projection(frame,branch['input']['coarse_pose']);measurements=measurement_rows(branch)
        files={};label=TITLES[name]
        fig,ax=picture('① '+label+' · 자체 초기 예측과 F 자세')
        wire(ax,before,'#8852b6','이 경로의 저장 초기 F 모델')
        raw=np.asarray(branch['input']['native_points'],float)
        support=np.asarray(branch['input']['prediction_support'],bool)
        for k in range(8):
            if support[k] and finite_xy(raw[k]):point(ax,raw[k],k,'#cf5938','x')
        ax.legend(fontsize=11);files['initial']=name+'_01_initial.png';fig.savefig(folder/files['initial'],dpi=135);plt.close(fig)
        fig,ax=picture('② '+label+' · 새 상태 판정')
        wire(ax,before,'#a9a3b1','초기 모델: S는 이 모델의 자기 차폐')
        for k,state in enumerate(states):
            if state in ('B','I'):point(ax,raw[k],k,STATE_COLORS[state])
        ax.legend(fontsize=11);files['states']=name+'_02_states.png';fig.savefig(folder/files['states'],dpi=135);plt.close(fig)
        fig,ax=picture('③ '+label+' · 실제 B/I 보정 관측')
        mb=branch['mask_binding'];maskpath=C.ROOT/mb['path']
        if C.sha(maskpath)!=mb['sha256']:raise ValueError('Saved branch mask bytes changed')
        with np.load(maskpath,allow_pickle=False) as arrays:mask=np.asarray(arrays['pallet_mask']).copy()
        if mask.shape!=image.shape[:2]:raise ValueError('Saved mask canvas differs')
        rgba=np.zeros((*mask.shape,4));rgba[mask>=.5]=(.25,.75,.9,.13);ax.imshow(rgba)
        for line in saved_support(branch):
            xy=np.asarray(line['xy']);ax.scatter(xy[:,0],xy[:,1],color='#c69b00',s=4,alpha=.6)
        for m in measurements:
            a,b=np.asarray(m['before']),np.asarray(m['after']);k=m['native_corner'];color=STATE_COLORS[m['state']]
            ax.scatter(*a,color='#777777',marker='x',s=45,zorder=6)
            if m['movement_px']>1e-8:ax.annotate('',b,a,arrowprops={'arrowstyle':'->','color':color,'lw':2.2},zorder=6)
            point(ax,b,k,color,size=55)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([],[],color='#777777',marker='x',ls='',label='이전 입력'),
            Line2D([],[],color=STATE_COLORS['B'],marker='o',ls='',label='B 실제 측정'),
            Line2D([],[],color=STATE_COLORS['I'],marker='o',ls='',label='I 실제 측정')],fontsize=10)
        files['measurements']=name+'_03_measurements.png';fig.savefig(folder/files['measurements'],dpi=135);plt.close(fig)
        result=branch['pose_result'];final=projection(frame,result['actual_pose'])
        fig,ax=picture('④ '+label+' · 공통 B/I-only PnP 최종 모델')
        wire(ax,final,'#ef841b','최종 모델 투영 — 측정점 아님')
        ax.legend(fontsize=11);files['final']=name+'_04_final.png';fig.savefig(folder/files['final'],dpi=135);plt.close(fig)
        proof[name]=dict(files=files,state_records=state_table(branch),measurements=measurements,
            mask_binding=deepcopy(mb),initial_pose=deepcopy(branch['input']['coarse_pose']),
            initial_projection_xy=before.tolist(),final_pose=deepcopy(result['actual_pose']),
            final_projection_xy=final.tolist(),solution_accepted=bool(result['solution']['accepted']),
            pose_source=result['actual_pose'].get('source'),solver_states=deepcopy(result['states']),
            prompt=deepcopy(observed['mask_audit']),saved_finite_support_count=len(saved_support(branch)))
    return dict(id=frame['id'],ROI_original_xy=roi,branches=proof,all_initial_points_are_predictions=True,
        S_O_T_U_measured_point_markers=0,new_NN_SubPix_query_correction_or_posefit_calls=0)


def _table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def _number(x):
    return '—' if x is None else f'{float(x):.3f}'


def reason_text(row):
    labels={'actual_image_intersection':'영상의 두 선 교점 채택',
        'no_two_supported_nonparallel_lines':'두 선의 안정적인 교차 근거 부족',
        'insufficient_line_resamples':'반복 측정 지지 부족',
        'intersection_outside_image_trust_or_inconsistent':'교점이 범위 밖이거나 선들이 불일치',
        'actual_exterior_RGB_boundary_intersection':'실제 바깥 경계의 교점 채택',
        'intersection_extrapolates_beyond_observed_inlier_support':'실제 관측된 선 지지 범위 초과'}
    found=[label for key,label in labels.items() if key in row['reason']]
    if not found:return '원좌표 유지: 추가 영상 근거 미채택' if row['movement_px']<=1e-8 else '저장된 영상 보정 결과 채택'
    return ' · '.join(found)


def case_HTML(frame,proof,metrics):
    texts={'initial':'원래 입력 좌표와 각 경로의 초기 F 자세입니다. 점은 역사적 예측이며, 뒤에 S/O/T/U로 판정되는 채널도 이 초기 예측 표시에는 포함될 수 있습니다. 그 점을 새 RGB 관측으로 쓰지는 않습니다.',
        'states':'각 경로의 자체 초기 자세·좌표로 새로 판정했습니다. B/I만 점으로 표시합니다. S는 모델 자기 차폐, O는 외부 가림 분류, T는 화면 밖 추정, U는 미확정입니다. 분류 상태는 정답이 아닙니다.',
        'measurements':'회색 ×는 그 경로의 이전 좌표, 색 ○는 실제 B/I 보정 결과입니다. 이동이 없으면 원래 좌표를 유지한 것입니다. 노란 작은 점은 보관된 실제 영상 선 지지입니다. 얇은 파란 영역은 그 경로의 저장 target mask이며 가림 정답은 아닙니다.',
        'final':'동일 B/I-only PnP의 저장 결과입니다. 선은 강체 모델 투영이며 새 측정점이 아닙니다. O/T/U는 정밀 영상점으로 피팅하지 않았고 S도 영상 관측으로 쓰지 않았습니다.'}
    parts=[]
    for stage,title in (('initial','① 각 경로의 입력과 초기 F'),('states','② 각 경로에서 새로 판정한 상태'),('measurements','③ B/I 실제 영상 보정'),('final','④ 같은 B/I-only PnP의 최종 모델')):
        parts+=['<h3>'+title+'</h3><p>'+texts[stage]+'</p><div class="pair">']
        for name in BRANCHES:
            data=proof['branches'][name];src=data['files'][stage]
            part=['<article><h4>'+TITLES[name]+'</h4>',f'<a href="{src}" target="_blank"><img loading="lazy" src="{src}" alt="{html.escape(frame["id"]+name+stage)}"></a><p class="zoom">그림을 누르면 원본 크기로 확대합니다.</p>']
            if stage=='states':
                part.append(_table(('번호','상태','새 p_visible','원 YOLO conf'),[(r['native_corner'],r['state'],_number(r['p_visible']),_number(r['original_YOLO_confidence'])) for r in data['state_records']]))
            if stage=='measurements':
                part.append(_table(('점','이동 px','실제 이동','유지·채택 근거'),[(str(r['state'])+str(r['native_corner']),_number(r['movement_px']),'이동' if r['movement_px']>1e-8 else '원좌표 유지',reason_text(r)) for r in data['measurements']]))
            if stage=='final':
                part.append('<p>PnP 제약: <b>'+('통과' if data['solution_accepted'] else '폴백')+'</b></p>')
                m=metrics.get(name)
                if m is not None:part.append('<p class="metric">저장 사후 평가: T '+_number(m['translation_cm'])+' cm · R '+_number(m['rotation_deg'])+'°<br>평가 기준: '+('통과' if m['success_5cm_5deg'] else '실패')+' (T &lt; 5cm · R &lt; 5°)</p>')
            part.append('</article>');parts+=part
        parts+=['</div>']
    return '\n'.join(parts)


STYLE='body{font-family:sans-serif;max-width:1800px;margin:24px auto;padding:18px;background:#f1f4f7;color:#172633}h1{font-size:30px}h2{padding-top:25px}h3{margin-top:34px}p{line-height:1.7}.pair{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}article,section{background:white;border-radius:12px;padding:15px}img{display:block;width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}.metric{font-weight:bold}.zoom{font-size:13px;color:#556}nav a{display:inline-block;margin:7px}@media(max-width:900px){.pair{grid-template-columns:1fr}}'


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _metadata(run,prediction_sha,score_sha):
    """Exact completed saved data only, no scoring or live observation path."""
    run=Path(run).resolve()
    if not run.is_relative_to(B) or not run.name.startswith('new_gain10_paired_BI_v'):
        raise ValueError('Separate declared gain10 prediction folder required')
    if C.sha(run/'PREDICTION_SEAL.json')!=prediction_sha or C.sha(run/'SCORE_SEAL.json')!=score_sha:
        raise ValueError('Caller-pinned completed prediction and score seals required')
    seal=C.read(run/'PREDICTION_SEAL.json');scoreseal=C.read(run/'SCORE_SEAL.json')
    protocol=C.read(run/'PROTOCOL.json');input_data=C.read(INPUT)
    expected_arms=[name+'_F' for name in BRANCHES]+[name+'_BI_ONLY_FRESH_PNP' for name in BRANCHES]
    if (seal['rows']!=10 or tuple(seal['ordered_ids'])!=FIXED_IDS or seal['arms']!=expected_arms
            or seal['before_scoring'] is not True or seal['GT_inputs'] is not False
            or seal['no_population_drop'] is not True or seal['full_V33_method'] is not False
            or C.sha(run/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']
            or C.sha(run/'PROTOCOL.json')!=seal['protocol_sha256']
            or C.sha(run/'OBSERVATION_SEAL.json')!=seal['observation_seal_sha256']
            or C.sha(INPUT)!=seal['inputs_sha256']
            or C.sha(INPUT.with_name('INPUT_SEAL.json'))!=seal['input_seal_sha256']
            or scoreseal['rows']!=10 or scoreseal['arm_rows']!=40 or scoreseal['arms']!=expected_arms
            or scoreseal['prediction_seal_sha256']!=prediction_sha
            or scoreseal['all_rows_kept'] is not True
            or C.sha(run/'SCORE.json')!=scoreseal['score_sha256']
            or protocol['full_V33_method'] is not False or protocol['reference_or_GT_inputs'] is not False
            or tuple(input_data['ordered_ids'])!=FIXED_IDS or input_data['arms']!=list(BRANCHES)):
        raise ValueError('Complete fixed10/20branch/4arm saved diagnostic must remain exact')
    scores=C.read(run/'SCORE.json')
    if [row['id'] for row in scores['rows']]!=list(FIXED_IDS):raise ValueError('All ten saved score rows required')
    metrics={row['id']:row['metrics'] for row in scores['rows']}
    if any(set(value)!=set(expected_arms) for value in metrics.values()):raise ValueError('All four metrics per case required')
    sources={relative(path):C.sha(path) for path in (Path(__file__),Path(C.__file__),Path(OLD.__file__))}
    if C.sha(OLD.__file__)!='a5d2433c983f14bb9cbd6fe5ca675937805ded6df24b148935a7648e3d329525':
        raise ValueError('Reviewed camera-facing forward projection helper required')
    for path,digest in protocol['source_sha256'].items():
        if C.sha(C.ROOT/path)!=digest:raise ValueError('Saved runtime source changed')
        if path in sources and sources[path]!=digest:raise ValueError('Source map conflict')
        sources[path]=digest
    paths=[run/name for name in ('PROTOCOL.json','PLAN_SEAL.json','OBSERVATION_SEAL.json',
        'PREDICTION_SEAL.json','PREDICTIONS.jsonl.gz','SCORE.json','SCORE_SEAL.json')]
    paths+=[INPUT,INPUT.with_name('INPUT_SEAL.json')]
    inputs={relative(path):C.sha(path) for path in paths}
    return run,protocol,input_data,metrics,sources,inputs


def render(*,root_saved_render_GO=False,prediction_dir=RUN,expected_prediction_seal_sha256=None,
           expected_score_seal_sha256=None):
    if root_saved_render_GO is not True:raise PermissionError('ROOT saved paired-BI render GO before IO')
    if not _sha(expected_prediction_seal_sha256) or not _sha(expected_score_seal_sha256):
        raise ValueError('Explicit prediction and score SHA before IO')
    if OUTPUT.exists():raise FileExistsError('Preserve every new paired BI process attempt')
    run,protocol,data,metrics,sources,inputs=_metadata(prediction_dir,expected_prediction_seal_sha256,expected_score_seal_sha256)
    OUTPUT.mkdir(parents=True)
    C.write(OUTPUT/'ANALYSIS_PLAN.json',dict(schema='SAVED_GAIN10_BOTH_BI_PROCESS_DISPLAY_PLAN_V1',
        UTC=datetime.now(timezone.utc).isoformat(),source_sha256=sources,input_sha256=inputs,
        external_environment_binding={'font':{'path':str(FONT),'sha256':C.sha(FONT)}},
        ordered_ids=list(FIXED_IDS),branches=list(BRANCHES),prediction_dir=relative(run),
        source='postscore user-requested fixed historical gain10; no new case selection',
        ROI='same initial branch q/own coarse/actual BI only; 240x180 minimum span, max32px or20percent margin, canvas clipping',
        initial_q='historical predicted channels, not new S/O/T/U image observations',
        measured_markers='cyan B / violet I; gray x input; arrows actual movement only',
        final_model='orange wire, no precise observed S/O/T/U point',
        fresh_BI_in_this_renderer=0,new_NN_or_PnP_or_optimizer_or_feature_queries=0,
        new_reference_documents_opened=0,full_current_V33_method=False))
    C.write(OUTPUT/'PLAN_SEAL.json',dict(plan_sha256=C.sha(OUTPUT/'ANALYSIS_PLAN.json')))
    cards=[];md=['# 양쪽 모두 B/I 영상보정 적용','',
        '원 YOLO와 N3+SubPix의 같은 열 장을 각각 새 상태·마스크·B/I 영상보정에 넣고 같은 B/I-only PnP로 푼 저장 결과다. 현재 V33의 S/O/U/C와 유한 선 요인을 포함한 전체 방법 재실행은 아니다.','',
        '청록 원은 B 실제 측정, 보라 원은 I 실제 측정, 회색 ×는 그 경로의 이전 입력이다. 화살표는 실제 2D 이동만 나타낸다. 주황 선은 최종 PnP 모델 투영이며 측정점이 아니다. S는 모델 자기 차폐이며 영상 관측점으로 쓰지 않는다.','']
    page=['<!doctype html><meta charset="utf-8"><title>양쪽 모두 B/I 영상보정 적용</title><style>'+STYLE+'</style>',
        '<h1>양쪽 모두 B/I 영상보정 적용</h1>',
        '<p>왼쪽은 N3+SubPix 없이, 오른쪽은 N3+SubPix를 사용한 입력입니다. 두 쪽 모두 각자의 초기 자세로 상태·마스크·B/I를 새로 계산한 뒤 같은 B/I-only PnP를 적용했습니다. 현재 V33 전체 방법을 재실행한 결과는 아닙니다.</p>',
        '<p><b>청록 ○ = B 실제 측정 · 보라 ○ = I 실제 측정 · 회색 × = 이전 입력 · 주황 선 = 최종 모델 투영.</b> S/O/T/U를 정확한 새 영상점으로 만들지 않았습니다. 원 YOLO confidence와 새 보임 확률은 서로 다른 값입니다. 입력 RAW 쪽의 분류 모델 적용도 새 CAL 정확도 보장을 뜻하지 않습니다.</p>',
        '<p>이 열 장은 과거 TEST49에서 N3+SubPix로 실패→성공이 바뀐 사례를 사후 고른 부분집합입니다. 새 전체 성공률이나 일반화 평가가 아닙니다. 그림은 저장된 실제 결과만 표시하며 새 추론·보정·PnP를 실행하지 않습니다.</p>',
        '<details><summary>N3→SubPix→총 이동한도: 앞선 저장 전처리 4단계 보기</summary><p><a href="../raw_yolo_N3_subpix_gain10_process_v1/index.html">이전 10사례 갤러리</a>의 q0/qN/qS/qFinal은 이번 N3+SubPix 입력의 앞단입니다. 아래에서는 양쪽 모두에 추가한 B/I 관측과 공통 PnP를 비교합니다.</p></details><nav>']
    page += [f'<a href="#c{i}">{i+1}. {html.escape(identifier)}</a>' for i,identifier in enumerate(FIXED_IDS)]
    page+=['</nav>']
    try:
        count=0
        for i,(frame,row) in enumerate(zip_longest(data['frames'],C.rows(run/'PREDICTIONS.jsonl.gz'))):
            if frame is None or row is None or i>=10 or frame['id']!=FIXED_IDS[i] or row['id']!=frame['id']:
                raise ValueError('Exact ordered fixed10 rows required; failures never dropped')
            for key in ('image','image_sha256','raw_hw','K','xyz'):
                if row[key]!=frame[key]:raise ValueError('Prediction/input identity differs: '+key)
            folder=OUTPUT/f'case{i:02d}';proof=draw_tiles(frame,row,folder)
            proof['metrics']={name:deepcopy(metrics[frame['id']][name+'_BI_ONLY_FRESH_PNP']) for name in BRANCHES}
            proof['all_four_saved_metrics']=deepcopy(metrics[frame['id']]);proof['prediction_dir']=relative(run)
            C.write(folder/'PROCESS_DATA.json',proof)
            body=case_HTML(frame,proof,proof['metrics'])
            header='<h2>'+str(i+1)+'. '+html.escape(frame['id'])+'</h2><p><a href="source_rgb.png" target="_blank">원본 전체 사진</a> · <a href="PROCESS_DATA.json">저장 값·출처</a> · <a href="../index.html#c'+str(i)+'">전체 목록</a></p>'
            (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>'+STYLE+'</style>'+header+body,encoding='utf8')
            rootbody=body
            for name in BRANCHES:
                for image in proof['branches'][name]['files'].values():
                    rootbody=rootbody.replace('"'+image+'"','"'+folder.name+'/'+image+'"')
            page+=['<section id="c'+str(i)+'"><h2>'+str(i+1)+'. '+html.escape(frame['id'])+'</h2>',
                '<p><a href="'+folder.name+'/index.html">이 사례만 크게 보기</a> · <a href="'+folder.name+'/source_rgb.png">원본 전체 사진</a></p>',rootbody,
                '<details><summary>보관 값과 출처</summary><p><a href="'+folder.name+'/PROCESS_DATA.json">단계별 관측·자세·사유·평가 JSON</a> · <a href="ANALYSIS_PLAN.json">표시 계획과 바이트 바인딩</a></p></details></section>']
            cards.append(dict(id=frame['id'],folder=folder.name,states={name:row['branches'][name]['observation']['states'] for name in BRANCHES},metrics=proof['metrics']))
            md+=['## '+str(i+1)+'. '+frame['id'],'',f'[양쪽 단계별 HTML]({folder.name}/index.html) · [원본 사진]({folder.name}/source_rgb.png) · [보관 값]({folder.name}/PROCESS_DATA.json)','']
            count+=1;del row,proof
            print('RENDERED',count,'/10',frame['id'],str(folder/'index.html'),flush=True)
        if count!=10:raise ValueError('All ten cases must remain')
        page+=['<p><a href="REPORT_KO.md">간단 설명과 개별 사례 링크</a></p>']
        (OUTPUT/'index.html').write_text('\n'.join(page),encoding='utf8');(OUTPUT/'REPORT_KO.md').write_text('\n'.join(md),encoding='utf8')
        for path,digest in sources.items():
            if C.sha(C.ROOT/path)!=digest:raise ValueError('Bound source changed during rendering')
        for path,digest in inputs.items():
            if C.sha(C.ROOT/path)!=digest:raise ValueError('Sealed prediction/score input changed')
        C.write(OUTPUT/'RENDER_SEAL.json',dict(schema='SAVED_GAIN10_BOTH_BI_PROCESS_RENDER_SEAL_V1',
            rows=10,branches=20,cases=cards,source_sha256=sources,input_sha256=inputs,
            prediction_seal_sha256=expected_prediction_seal_sha256,score_seal_sha256=expected_score_seal_sha256,
            plan_sha256=C.sha(OUTPUT/'ANALYSIS_PLAN.json'),
            assets={str(path.relative_to(OUTPUT)):C.sha(path) for path in sorted(OUTPUT.rglob('*')) if path.is_file()},
            new_reference_NN_query_correction_PnP_or_optimizer_calls=0,full_current_V33_method=False,
            posthoc_fixed_subset_only=True,visual_QA_pending=True))
    except BaseException as error:
        C.write(OUTPUT/'FAILED_ATTEMPT_V1.json',dict(completed_cases=len(cards),exception=type(error).__name__,reason=str(error),attempt_preserved=True))
        raise
    return binding(OUTPUT/'RENDER_SEAL.json')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-saved-render-GO',action='store_true')
    parser.add_argument('--prediction-dir',type=Path,default=RUN)
    parser.add_argument('--prediction-seal-sha256',required=True)
    parser.add_argument('--score-seal-sha256',required=True)
    args=parser.parse_args()
    print(render(root_saved_render_GO=args.root_saved_render_GO,prediction_dir=args.prediction_dir,
        expected_prediction_seal_sha256=args.prediction_seal_sha256,expected_score_seal_sha256=args.score_seal_sha256))
