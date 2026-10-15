"""Two fixed user-requested cases, showing only sealed v29 observations/poses.

Source/plan are declared before new scores. Rendering requires explicit root
postscore/audit/memory GO. No new image measurements, model calls, pose fits,
GT projections or S/U point markers. Stages1..4 are common saved evidence;
panels5..9 are distinct final methods, not optimizer intermediate states.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import html
import json
from pathlib import Path
import re

import cv2
import numpy as np

from . import common as C
from . import observe_geometry_guided_edges_v4_29b as OBS
from . import image_edge_observations_v4_29 as P

B=OBS.OUTPUT.parent
PREDICTIONS=B/'geometry_guided_edge_eval_v4_29'
OUTPUT=B/'geometry_guided_process_v4_29_v1'
PLAN=B/'geometry_guided_process_plan_v4_29/PREDECLARATION.json'
CASES=('wood_night_01:031426','eval_pallet09:1778653661653195264')
ARMS=('SAME_FROZEN_E','GEOMETRY_EDGE_ALL_BI_NATIVE','GEOMETRY_EDGE_ALL_BI_CONTAINMENT',
      'GEOMETRY_EDGE_SOURCE_NATIVE','GEOMETRY_EDGE_SOURCE_CONTAINMENT')
LABELS=('기존 e (기본값)','전체 B/I 선 요인 · native','전체 B/I 선 요인 · C',
        'RGB q 보존 · native','RGB q 보존 · C')
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def _exact_rows(path,ids=CASES):
    patterns={ident:re.compile(r'"id"\s*:\s*"'+re.escape(ident)+'"')for ident in ids}
    found={};digests={}
    with gzip.open(path,'rt')as stream:
        for line in stream:
            matching=[ident for ident,p in patterns.items()if ident not in found and p.search(line)]
            if not matching:continue
            row=json.loads(line)
            if row.get('id')not in matching:continue
            found[row['id']]=row;digests[row['id']]=hashlib.sha256(line.encode()).hexdigest()
            if len(found)==len(ids):break
    if set(found)!=set(ids):raise ValueError('Both exact fixed cases required')
    return found,digests


def _gated_visible(row):
    result=[]
    for record in row['visible_measurements']:
        k=P.G.NMS.V3._id(record)
        if row['initial_geometry']['initial_self_hidden'][k]or row['states'][k]not in('B','I'):continue
        if record['state']!=row['states'][k]:raise ValueError('Exact B/I state required')
        # Only now access fields that belong to an actual visible measurement.
        result.append(dict(native_corner=k,state=row['states'][k],xy=record['xy'],source=record['source']))
    return result


def _selected(row):
    observation=row['observation'];output=[]
    for ref in observation['selected_chain_refs']:
        k,mate=ref['native_corner'],ref['native_neighbor']
        if row['initial_geometry']['initial_self_hidden'][k]or row['states'][k]not in('B','I')or row['states'][mate]=='S':
            raise ValueError('Selected local support must retain permanent state gates')
        chain=observation['chain_observation']['edge_records'][ref['ray_index']]['chains'][ref['chain_index']]
        if chain['chain_id']!=ref['chain_id']:raise ValueError('Exact selected actual chain required')
        xy=np.asarray(chain['observed_points'],float)[np.asarray(chain['inlier_mask'],bool)]
        output.append(dict(reference=ref,xy=xy,normal=np.asarray(chain['normal']),offset=float(chain['offset']),
            partial=chain['proposal_grade']==P.PARTIAL_GRADE,actual_span_px=chain['actual_finite_inlier_span_px']))
    return output


def _active_plan_lines(result,row,arm):
    if arm=='SAME_FROZEN_E':return {},[]
    plan=result.get('solution',{}).get('v29_cost_plan')
    if plan is None:plan=result['new_geometry_edge_cost_plans'][result['WD_family']]
    lines=[]
    for group in plan['active_groups']:
        k,mate=group['native_corner'],group['native_neighbor']
        if row['initial_geometry']['initial_self_hidden'][k]or row['states'][k]not in('B','I')or row['states'][mate]=='S':
            raise ValueError('Active factor support must retain permanent state gates')
        lines.append(dict(xy=np.asarray(group['actual_inlier_xy']),normal=np.asarray(group['normal_unit_xy']),
            offset=float(group['offset_px']),partial=group['proposal_grade']==P.PARTIAL_GRADE,
            reference=dict(native_edge=group['physical_native_edge'])))
    return plan,lines


def prepare():
    if PLAN.exists():raise FileExistsError('Preserve fixed case declaration')
    if(PREDICTIONS/'SCORE_SEAL.json').exists():raise ValueError('Declare fixed cases before new scores')
    if C.sha(OBS.V.PLAN)!=P.CENTRAL_PREDECLARATION_SHA256:raise ValueError('Exact central experiment declaration required')
    source={str(Path(__file__).resolve().relative_to(C.ROOT)):C.sha(Path(__file__))}
    for module in(OBS,OBS.V,P):source[str(Path(module.__file__).resolve().relative_to(C.ROOT))]=C.sha(Path(module.__file__))
    PLAN.parent.mkdir(parents=True,exist_ok=False)
    C.write(PLAN,dict(schema='FIXED_TWO_USER_CASES_GEOMETRY_GUIDED_PROCESS_PLAN_V29',
        UTC=datetime.now(timezone.utc).isoformat(),cases=list(CASES),arms=list(ARMS),source_sha256=source,
        central_predeclaration_sha256=P.CENTRAL_PREDECLARATION_SHA256,
        observation_output=str(OBS.OUTPUT.relative_to(C.ROOT)),prediction_output=str(PREDICTIONS.relative_to(C.ROOT)),
        stage_order=['original RGB','frozen initial N3 geometry and gated BI q','actual target-mask exterior; O/U ranges only',
            'graph-selected full/partial NMS candidates before AUTH/source-policy activation','five separately saved final methods with active-plan support'],
        no_stored_optimizer_intermediate_pose_invented=True,full_and_partial_colour_separate=True,
        independent_other_object_mask_available=False,target_mask_absence_alone_is_unknown=True,
        precise_KP_created=False,S_U_observed_point_markers=0,GT_wire=0,
        new_image_measurement_NN_fit_PnP_or_reference_decode=0,source_and_case_selection_before_new_scores=True,
        render_requires_full238_prediction_score_and_independent_audit=True,requires_explicit_root_memory_GO=True,
        default_remains_e=True))
    print(json.dumps(dict(plan=str(PLAN),sha256=C.sha(PLAN))),flush=True)


def _completed():
    plan=C.read(PLAN)
    if plan['cases']!=list(CASES)or plan['arms']!=list(ARMS):raise ValueError('Fixed case/method declaration changed')
    for rel,digest in plan['source_sha256'].items():
        if C.sha(C.ROOT/rel)!=digest:raise ValueError('Declared renderer/producer changed')
    protocol=C.read(PREDICTIONS/'PROTOCOL.json');pred=C.read(PREDICTIONS/'PREDICTION_SEAL.json');score=C.read(PREDICTIONS/'SCORE_SEAL.json')
    if (pred['rows']!=238 or pred['arms']!=list(ARMS)or pred['arm_count']!=5
            or not pred['all_declared_arms_complete']or C.sha(PREDICTIONS/'PREDICTIONS.jsonl.gz')!=pred['predictions_sha256']
            or C.sha(PREDICTIONS/'PROTOCOL.json')!=pred['protocol_sha256']
            or C.sha(PREDICTIONS/'INPUT_SEAL.json')!=pred['input_seal_sha256']
            or C.sha(PREDICTIONS/'PREDICTION_SEAL.json')!=score['prediction_seal']['sha256']):
        raise ValueError('Complete sealed238/fivearm predictions and scores required')
    auditpath=PREDICTIONS/'independent_root_audit_v1/AUDIT.json';audit=C.read(auditpath);aseal=C.read(auditpath.with_name('AUDIT_SEAL.json'))
    if(audit['rows']!=238 or audit['failure_count']or audit['failures']or C.sha(auditpath)!=aseal['receipt_sha256']
            or audit['prediction_seal']['sha256']!=C.sha(PREDICTIONS/'PREDICTION_SEAL.json')):
        raise ValueError('Current independent zero-failure audit required')
    for key in('source_sha256','input_sha256','protected_user_sha256'):OBS.OLD._verify_map(protocol[key])
    oseal=C.read(OBS.OUTPUT/'OBSERVATION_SEAL.json')
    if oseal['rows']!=238 or C.sha(OBS.OUTPUT/'OBSERVATIONS.jsonl.gz')!=oseal['observations_sha256']:
        raise ValueError('Complete current actual observation seal required')
    observations,obs_digests=_exact_rows(OBS.OUTPUT/'OBSERVATIONS.jsonl.gz')
    predictions,pred_digests=_exact_rows(PREDICTIONS/'PREDICTIONS.jsonl.gz')
    tasks,task_digests=_exact_rows(PREDICTIONS/'INPUTS.jsonl.gz')
    scores={}
    for partition,ident in(('TEST',CASES[1]),('REMAINDER',CASES[0])):
        path=PREDICTIONS/(partition+'_SCORE.json')
        if C.sha(path)!=score['score_sha256'][partition]:raise ValueError('Sealed metric archive changed')
        matches=[r for r in C.read(path)['rows']if r['id']==ident]
        if len(matches)!=1:raise ValueError('Exact fixed postseal metric required')
        scores[ident]=matches[0]
    return observations,predictions,tasks,scores,dict(observation_rows=obs_digests,prediction_rows=pred_digests,task_rows=task_digests)


def _wire(ax,q,color='#7b72bd',style='--'):
    q=np.asarray(q)
    for k,mate in EDGES:
        if np.isfinite(q[[k,mate]]).all():ax.plot(q[[k,mate],0],q[[k,mate],1],c=color,ls=style,lw=1.1)


def _lines(ax,lines,alpha=1.):
    for item in lines:
        xy=item['xy'];n=item['normal'];axis=np.array([-n[1],n[0]])
        order=np.argsort(xy@axis);colour='#d3a022'if item['partial']else'#009e8d'
        ax.plot(xy[order,0],xy[order,1],c=colour,lw=2.,alpha=alpha)
        ax.scatter(xy[:,0],xy[:,1],c=colour,s=3,alpha=.65*alpha)
        center=xy.mean(0);edge=item['reference']['native_edge']
        ax.text(*center,' '+str(edge[0])+'–'+str(edge[1]),fontsize=8,color=colour)


def _ranges(ax,row):
    from matplotlib.patches import Ellipse,Rectangle
    for domain in [*row['original_observer_domains'],*row['u_regions']]:
        k=P.G.NMS.V3._id(domain)
        if row['initial_geometry']['initial_self_hidden'][k]or row['states'][k]not in('O','U'):continue
        colour='#cd7042'if row['states'][k]=='O'else'#b69623'
        if domain.get('shape')=='ellipse':
            axes=domain['half_axes_px'];ax.add_patch(Ellipse(domain['center_xy'],2*axes[0],2*axes[1],
                angle=domain.get('angle_deg',0.),facecolor=colour,edgecolor=colour,alpha=.12))
        elif domain.get('shape')=='box'and 'lower_xy'in domain:
            lo=np.asarray(domain['lower_xy']);hi=np.asarray(domain['upper_xy']);ax.add_patch(Rectangle(lo,*(hi-lo),
                facecolor=colour,edgecolor=colour,alpha=.08))
    # Centers are not marked. These are localization/range archives, not
    # measured hidden keypoints or a segmentation of the occluding object.


def render(*,root_postscore_audit_memory_GO=False):
    if not root_postscore_audit_memory_GO:raise ValueError('Wait for explicit root score/audit/memory GO')
    if OUTPUT.exists():raise FileExistsError('Preserve every report generation')
    observations,predictions,tasks,scores,row_digests=_completed()
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties,fontManager
    from . import render_source_preserving_tangent_mask_process_v4_24 as FONT
    font=FONT.DRAW._font();fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False;OUTPUT.mkdir(parents=True)
    assets=[];reports=[]
    for number,ident in enumerate(CASES):
        row=observations[ident];frame=predictions[ident];task=tasks[ident];geometry=row['initial_geometry']
        if set(frame['arms'])!=set(ARMS):raise ValueError('All five uniform methods required')
        image_path=C.ROOT/row['image'];mask_path=C.ROOT/row['mask_binding']['path']
        if C.sha(image_path)!=row['image_sha256']or C.sha(mask_path)!=row['mask_binding']['sha256']:raise ValueError('Actual RGB/mask changed')
        image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
        with np.load(mask_path,allow_pickle=False)as data:mask=data['pallet_mask_clean'].copy()
        selected=_selected(row);visible=_gated_visible(row)
        ys,xs=np.where(np.asarray(mask)>=.5);h,w=image.shape[:2]
        if len(xs):bounds=(max(0,int(xs.min())-25),min(w,int(xs.max())+26),max(0,int(ys.min())-25),min(h,int(ys.max())+26))
        else:bounds=(0,w,0,h)
        l,r,t,b=bounds
        initial=P.G._geometry(geometry);q=C.project(initial['xyz'],initial['R'],initial['t'],initial['K'])
        figures=[];facts=[]
        for stage in range(9):
            fig,ax=plt.subplots(figsize=(7.5,5.7));ax.imshow(image);ax.set_axis_off()
            if stage:ax.set_xlim(l,r);ax.set_ylim(b,t)
            if stage==0:title='① 원본 RGB · 밝기/색 변경 없음';caption='기존 사용자가 지정한 사례이며 전체 성능을 대표하지 않습니다.'
            elif stage==1:
                title='② 저장된 첫 N3 기하와 B/I 관측';caption='점=실제 보관 B/I q · 점선=모델 예측; S/U 점 표시 없음'
                _wire(ax,q)
                for m in visible:
                    colour='#e08a2c'if m['source']=='VISIBLE_BACKBONE_FALLBACK'else'#188d68'
                    ax.scatter(*m['xy'],c=colour,marker='x',s=35);ax.text(m['xy'][0]+2,m['xy'][1],str(m['native_corner'])+' '+m['state'],color=colour,fontsize=9)
            elif stage==2:
                title='③ 실제 target mask와 미확정 가림';caption='독립 other-object mask 없음 · mask 부재만으로 O 확정하지 않음'
                contours,_=cv2.findContours((mask>=.5).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    xy=contour.reshape(-1,2);ax.plot(xy[:,0],xy[:,1],c='#49b5bc',lw=1.)
                _ranges(ax,row)
                for edge in row['observation']['edge_visibility']:
                    for a,z in zip(edge['samples'],edge['samples'][1:]):
                        observed=a['classification']==z['classification']=='observed_support'
                        xy=np.asarray([a['xy'],z['xy']]);ax.plot(xy[:,0],xy[:,1],c='#009e8d'if observed else'#946bbb',ls='-'if observed else'--',lw=1.)
            elif stage==3:
                title='④ graph 선정 후보 (아직 채택 전)';caption='청록=일반 선 · 노랑=부분 선; AUTH/외곽·중복/source정책 후 활성요인과 다를 수 있음'
                _lines(ax,selected)
            else:
                arm=ARMS[stage-4];result=frame['arms'][arm];pose=result['actual_pose'];name=result['WD_family']
                family=task['families'][name]
                if not np.allclose(pose['physical_registration_Q'],family['physical_registration_Q'],atol=1e-12,rtol=0):raise ValueError('Displayed physical family differs')
                if not np.allclose(np.mean(family['native_xyz'],axis=0),0.,atol=1e-12,rtol=0):
                    raise ValueError('This renderer requires centered native XYZ: legacy centroid field stores t')
                active_plan,active_lines=_active_plan_lines(result,row,arm);_lines(ax,active_lines,alpha=.3)
                projected=C.project(family['native_xyz'],pose['R_cf'],pose['centroid'],geometry['K']);_wire(ax,projected,'#6555bf','-')
                outside=int(np.sum(np.any((projected<[l,t])|(projected>[r,b]),axis=1)))
                if outside:ax.text(.02,.02,f'확대창 밖 모델 꼭짓점 {outside}개 · 전체 모델 투영은 JSON',
                    transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.85))
                metric=scores[ident]['metrics'][arm];center=np.asarray(pose['R_cf'])@np.asarray(family['native_xyz']).mean(0)+np.asarray(pose['centroid'])
                title=str(stage+1)+' '+LABELS[stage-4]
                caption=f"{name} · 모델 중심 깊이 {center[2]:.3f}m · T {metric['translation_cm']:.2f}cm · R {metric['rotation_deg']:.2f}°"
                accepted=bool(result['accepted']);owners=active_plan.get('new_owner_ids',[])
                retained=sorted(m['native_corner']for m in visible if m['native_corner']not in owners)
                status='joint 수용'if accepted else'해 없음 → 초기 자세 fallback'
                caption+='\n'+status+(' · 기존 e: 새 활성계획 없음'if arm==ARMS[0]else
                    f" · {'활성선'if accepted else'활성계획 선'} {len(active_lines)} · 새 owner {owners} · 기존 관측 유지 {retained}")
                # Final methods are compared, never an interpolated pose path.
                facts.append(dict(arm=arm,WD_family=name,model_center_camera_m=center.tolist(),
                    metric=metric,joint_feasible=pose.get('joint_feasible'),fallback=pose.get('fallback'),
                    source_cost_plan=active_plan,active_line_count=len(active_lines),new_owner_ids=owners,
                    retained_observation_ids=None if arm==ARMS[0]else retained,
                    model_wire_projection_xy=projected.tolist(),outside_crop_model_vertices=outside,
                    legacy_centroid_field_is_translation_parameter=True,centered_native_model_center_equals_translation=True))
            ax.set_title(ident+'\n'+title,fontsize=12);fig.text(.03,.02,caption,fontsize=9)
            fig.tight_layout(rect=(0,.045,1,1));path=OUTPUT/(f'case{number}_stage{stage+1:02d}.png')
            fig.savefig(path,dpi=140);plt.close(fig);assets.append(path);figures.append(path.name)
        from PIL import Image,ImageOps
        panels=[Image.open(OUTPUT/name).convert('RGB')for name in figures]
        width=max(p.width for p in panels);height=max(p.height for p in panels)
        montage=Image.new('RGB',(width*3,height*3),'white')
        for index,panel in enumerate(panels):montage.paste(ImageOps.pad(panel,(width,height),color='white'),((index%3)*width,(index//3)*height))
        montage_path=OUTPUT/(f'case{number}_montage.png');montage.save(montage_path);assets.append(montage_path)
        reports.append(dict(id=ident,images=figures,montage=montage_path.name,ROI_original_xy=bounds,
            states=row['states'],visible_sources=visible,O_range_count=sum(s=='O'for s in row['states']),
            U_range_count=len(row['u_regions']),initialS_model_only_ids=[k for k,s in enumerate(row['states'])if s=='S'],
            selected_lines=[dict(reference=x['reference'],inlier_count=len(x['xy']),partial=x['partial'],actual_span_px=x['actual_span_px'])for x in selected],
            final_methods=facts,stages1_to4_are_common_saved_evidence_not_new_PnP_steps=True,
            stages5_to9_are_separate_final_methods_not_optimizer_iterations=True,
            other_object_occlusion_mask_available=False,S_U_point_markers=0,GT_wire=0,new_measurement_or_posefit_calls=0))
    data=dict(schema='FIXED_TWO_CASES_GEOMETRY_GUIDED_PROCESS_DATA_V29',cases=reports,exact_row_text_sha256=row_digests,
        source_plan_sha256=C.sha(PLAN),observation_seal=OBS.OLD._binding(OBS.OUTPUT/'OBSERVATION_SEAL.json'),
        prediction_seal=OBS.OLD._binding(PREDICTIONS/'PREDICTION_SEAL.json'),score_seal=OBS.OLD._binding(PREDICTIONS/'SCORE_SEAL.json'),
        default_remains_e=True,reference_scope='derived/reference contract; no independent physical metrology claim',new_GT_projection=0)
    C.write(OUTPUT/'PROCESS_DATA.json',data)
    sections=''.join('<section><h2>'+html.escape(r['id'])+'</h2><img src="'+r['montage']+'">'+''.join(
        '<details><summary>단계 '+str(i+1)+'</summary><img src="'+name+'"></details>'for i,name in enumerate(r['images']))+'</section>'for r in reports)
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v29 실제 선 관측과 최종 자세</title><style>body{font:16px/1.65 sans-serif;color:#203341;max-width:1500px;margin:24px auto;padding:0 16px}img{width:100%;height:auto}details{margin:12px 0}h2{margin-top:36px}</style><h1>기하 가이드로 찾은 실제 선과 저장된 최종 자세</h1><p>사용자가 앞서 지정한 두 사례를 모든 방법에 동일하게 표시했습니다. ①~④는 공통 관측 과정이고 ⑤~⑨는 서로 다른 최종 방법 비교입니다. 저장되지 않은 optimizer 중간 자세를 만들지 않았습니다. ④는 그래프 선정 후보이고, 최종 정책의 AUTH·중복·외곽·source 필터 후 실제 활성선은 각 최종 패널의 옅은 선과 JSON에 따로 표시합니다. fallback은 활성계획이 있어도 수용 해가 없는 초기 자세입니다.</p><p>예상 선은 실제 RGB 측정선을 대신하지 않습니다. 짧은 선도 실제 픽셀과 TLS 지지를 보존하며, 새 정밀 KP를 만들지 않습니다. S는 모델 wire/PnP 역할만, U는 넓은 범위만 사용합니다. 독립 가림 물체 마스크가 없어 target mask 부재는 원인 미확정입니다. O 범위는 예측 위치 범위이며 실제 가림 물체 모양이 아닙니다. native XYZ는 중심이0인 모형임을 확인하고, legacy centroid 필드는 translation t이므로 여기서는 모델 중심과 같습니다.</p><p>기존 e 기본값은 유지됩니다. 두 사례 그림은 전체192 성능을 대표하지 않으며 추가143 참조는 독립 계측이 아닙니다. <a href="PROCESS_DATA.json">표시값과 봉인 출처</a></p>'+sections+'</html>'
    (OUTPUT/'index.html').write_text(page,encoding='utf-8');assets.extend([OUTPUT/'index.html',OUTPUT/'PROCESS_DATA.json'])
    C.write(OUTPUT/'RENDER_SEAL.json',dict(schema='FIXED_TWO_CASES_GEOMETRY_GUIDED_RENDER_SEAL_V29',
        cases=list(CASES),arms=list(ARMS),source_sha256=C.sha(Path(__file__)),plan_sha256=C.sha(PLAN),
        prediction_seal_sha256=C.sha(PREDICTIONS/'PREDICTION_SEAL.json'),
        assets={p.name:C.sha(p)for p in assets},S_U_point_markers=0,GT_wire=0,new_measurement_NN_posefit_calls=0))
    print(json.dumps(dict(report=str(OUTPUT/'index.html'),seal_sha256=C.sha(OUTPUT/'RENDER_SEAL.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','render'))
    parser.add_argument('--root-postscore-audit-memory-GO',action='store_true');args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:render(root_postscore_audit_memory_GO=args.root_postscore_audit_memory_GO)
