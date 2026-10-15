"""Postscore, user-selected 031426 supplement from saved evidence only.

This is not one of the four predeclared cases. No inference, observation
search, pose fit, reference projection, or optimizer-trajectory invention.
"""
from pathlib import Path
import argparse
import gzip
import html
import json
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

from . import common as C
from . import render_coarse_prior_containment_process_v4_25 as R

EXP, PREVIOUS, OLD, DRAW = R.EXP, R.PREVIOUS, R.OLD, R.DRAW
CASE = 'wood_night_01:031426'
STEM = 'wood_night_01__031426'
OUTPUT = EXP.OUTPUT.parent / 'user_requested_031426_process_v4_25_v1'
PREVIOUS_ARM = PREVIOUS.LOCKED
METHODS = (R.TP_NATIVE, R.TP_C, R.FULL_C)
TITLES = (
    '01 원본 사진 · 청록 상자는 확대 위치',
    '02 첫 N3_SUBPIX 모델 · 저장 B/I 관측',
    '03 고정 상태와 역할',
    '04 실제 저장 마스크 · NMS 선 대안',
    '05 SOURCE: 현재 complete 관측 유지·대체',
    '06 O/U 범위 · S는 모델 역할만',
    '07 첫 N3_SUBPIX pose를 향한 CAL prior',
    '08 이전 v24 · 약 176cm 실패',
    '09 이동 prior · native 최종 자세',
    '10 이동 prior · containment 최종 자세',
    '11 6D prior · containment 최종 자세',
    '12 깊이·W/D 비교 · 여전히 큰 실패')
CAPTIONS = (
    '실제 원본 사진. 이후 패널은 같은 청록 상자의 확대이며 새 추론은 하지 않았습니다.',
    '점선은 최초 N3 모델. ×는 complete B/I, +는 부분관측으로 남은 B/I의 보관 q입니다. S/U 점은 없습니다.',
    'B/I는 관측 역할, O/U는 범위 역할입니다. S는 초기 모델 가시성 및 PnP 모델 투영만 사용합니다.',
    '녹색은 실제 마스크. 회색은 보관 NMS 대안, 청록은 이동-native 선택 family의 활성 변입니다. 정밀 KP가 아닙니다.',
    '현재 RGB complete q는 유지합니다. 정확히 backbone fallback인 q만 대체 자격이 있습니다. 과거 부분관측 q를 되살리는 단계는 아닙니다.',
    'O hard domain은 닫힌 캔버스. 주황 타원은 raw 위치의 bounded soft 가정으로 실제 occluder가 아닙니다. U는 넓은 범위만 사용합니다.',
    '첫 N3 모델 중심·자세는 고정입니다. 이동 prior는 marginal Σtt, 6D prior는 전체 Σ6, CAL μ는 한 번만 뺍니다.',
    '이전 방법이 저장한 최종 자세입니다. 공통 처리의 중간 iterate가 아니며 GT wire도 아닙니다.',
    '새 이동 prior native 방법의 최종 저장 자세입니다. 이전 176cm 결과를 직접 수정한 연속 중간 자세가 아닙니다.',
    '이동 prior에 containment를 함께 최적화한 별도 최종 결과입니다. 추가 C는 내부0·외부만의 soft 비용입니다.',
    '전체 6D prior를 사용하는 별도 최종 결과입니다. 약 168cm 오차가 남아 성공으로 볼 수 없습니다.',
    '같은 사진에서 여러 depth/family가 남습니다. 영상 경계 비용과 CAL prior가 참조 정확도를 보장하지 않습니다.')


def _bind(path):
    path = Path(path).resolve()
    return str(path.relative_to(C.ROOT)), C.sha(path)


def _paths():
    return [Path(__file__), Path(R.__file__), Path(PREVIOUS.__file__), Path(OLD.__file__), Path(DRAW.__file__),
        Path(R.CONTEXT.__file__), Path(EXP.__file__), Path(EXP.PRIOR.__file__),
        EXP.OUTPUT/'PROTOCOL.json', EXP.OUTPUT/'INPUT_SEAL.json', EXP.OUTPUT/'PREDICTION_SEAL.json',
        EXP.OUTPUT/'SCORE_SEAL.json', EXP.OUTPUT/'independent_root_audit_v1/AUDIT.json',
        EXP.OUTPUT/'independent_root_audit_v1/AUDIT_SEAL.json',
        PREVIOUS.EXP.OUTPUT/'PREDICTION_SEAL.json', PREVIOUS.EXP.OUTPUT/'SCORE_SEAL.json',
        OLD.OBS.OUTPUT/'OBSERVATION_SEAL.json']


def prepare():
    if OUTPUT.exists():
        raise FileExistsError('Keep existing user-requested reports immutable')
    if not (EXP.OUTPUT/'SCORE_SEAL.json').exists():
        raise ValueError('This case is selected after score completion')
    OUTPUT.mkdir(parents=True)
    source = Path(__file__)
    (OUTPUT/source.name).write_bytes(source.read_bytes())
    C.write(OUTPUT/'PLAN.json', dict(schema='USER_SELECTED_POSTSCORE_031426_PROCESS_PLAN_V25',
        case_id=CASE, case_selected_by_user_after_scores=True, predeclared_four_case=False,
        renderer_source_sha256=C.sha(source), methods=list(METHODS), previous_method=PREVIOUS_ARM,
        steps=list(TITLES), optimizer_intermediate_poses_available=False,
        common_pipeline_and_distinct_final_methods_separated=True,
        S_U_point_queries_or_markers=0, GT_wire_or_projection=0, new_inference_or_pose_fit=0,
        dependency_sha256=dict(_bind(p) for p in _paths())))
    print(dict(plan=str(OUTPUT/'PLAN.json'), source_sha256=C.sha(source)), flush=True)


def _exact_row(path):
    """One sequential pass; decode only the exact case, retain no other row."""
    token = json.dumps(CASE).encode()
    found = None
    with gzip.open(path, 'rb') as stream:
        for line in stream:
            if token not in line:
                continue
            row = json.loads(line)
            if row['id'] == CASE:
                if found is not None:
                    raise ValueError('Duplicate exact case')
                found = row
    if found is None:
        raise ValueError('Exact case absent: '+str(path))
    return found


def _sealed_run(folder, arm_count):
    protocol, pred, score = (C.read(folder/name) for name in ('PROTOCOL.json','PREDICTION_SEAL.json','SCORE_SEAL.json'))
    if (pred['rows'] != 238 or pred['arm_count'] != arm_count or not pred['all_declared_arms_complete']
            or C.sha(folder/'PROTOCOL.json') != pred['protocol_sha256']
            or C.sha(folder/'INPUT_SEAL.json') != pred['input_seal_sha256']
            or C.sha(folder/'PREDICTIONS.jsonl.gz') != pred['predictions_sha256']
            or C.sha(folder/'PREDICTION_SEAL.json') != score['prediction_seal']['sha256']):
        raise ValueError('Completed prediction and score seal required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        EXP.E._verify_sources(protocol[key])
    frame_path = folder/'frames'/(STEM+'.json')
    frame = C.read(frame_path)
    if frame['id'] != CASE or frame['id'] not in protocol['ids'] or set(frame['arms']) != set(pred['arms']):
        raise ValueError('Exact saved complete frame required')
    sealed_frame = _exact_row(folder/'PREDICTIONS.jsonl.gz')
    if frame != sealed_frame:
        raise ValueError('Per-frame file differs from the exact sealed journal row')
    spath = folder/'REMAINDER_SCORE.json'
    if C.sha(spath) != score['score_sha256']['REMAINDER']:
        raise ValueError('Use sealed scalar metrics')
    report = C.read(spath)
    scored = next(row for row in report['rows'] if row['id'] == CASE)
    return frame, scored, [frame_path,spath,folder/'PREDICTIONS.jsonl.gz']


def _axis(ax, image, bounds, title, full=False):
    ax.imshow(image)
    h,w = image.shape[:2]
    ax.set_xlim((0,w) if full else bounds[:2]); ax.set_ylim((h,0) if full else bounds[2:])
    ax.set_aspect('equal'); ax.set_axis_off(); ax.set_title(title,fontsize=12,pad=10)


def _note(ax,text,*,bottom=False):
    ax.text(.025,.03 if bottom else .975,text,transform=ax.transAxes,va='bottom' if bottom else 'top',
        fontsize=9,bbox=dict(facecolor='white',alpha=.91,edgecolor='none'))


def _family_depth(arm):
    pose = arm['actual_pose']
    xyz = C.cuboid(*pose['cf_extents'])
    center = np.asarray(pose['R_cf'])@xyz.mean(0)+np.asarray(pose['centroid'])
    return dict(WD_family=arm.get('WD_family'),cf_extents_m=pose['cf_extents'],
        saved_camera_model_center_xyz_m=center.tolist(),camera_depth_m=float(center[2]),
        accepted=arm['accepted'], R_physical=pose['R_physical'])


def _roles(task,plan):
    by = {r['native_corner']:r for r in plan['corner_plans']}
    partial = {r['native_corner'] for r in task['boundary_merged']}
    visible = {}
    for m in task['spec']['visible']:
        k=m['native_corner']
        if task['initial_self_hidden'][k] or task['states'][k] not in ('B','I'):
            raise ValueError('State/S gate before BI fields')
        archive = by.get(k,{}).get('original_measurement_archive',{})
        visible[k]=dict(state=task['states'][k],source=m.get('source',archive.get('original_measurement_source')),
            observed_xy=m['xy'],partial_preexisting=k in partial,
            current_complete_q_retained=(not by[k]['point_cost_revoked']) if k in by else False,
            effective_fit_covariance_px2=m.get('fit_covariance_px2'),reliability=m.get('fit_reliability_weight'))
    return [dict(native_corner=k,state=s,BI_observation=visible.get(k),
        role=('모델만 · S 영상/2D 없음' if s=='S' else '범위만 · 정밀점 없음' if s=='U' else
              '캔버스 + bounded 위치 가정' if s=='O' else '조건부 화면밖 범위' if s=='T' else '영상 B/I 관측'))
        for k,s in enumerate(task['states'])]


def render():
    plan=C.read(OUTPUT/'PLAN.json')
    if plan['renderer_source_sha256'] != C.sha(__file__) or not plan['case_selected_by_user_after_scores']:
        raise ValueError('Render exact source fixed before this supplement')
    for path,digest in plan['dependency_sha256'].items():
        if C.sha(C.ROOT/path)!=digest: raise ValueError('Dependency changed')
    if (OUTPUT/'RENDER_SEAL.json').exists(): raise FileExistsError('Preserve completed render')
    new,score,paths=_sealed_run(EXP.OUTPUT,9)
    old,oldscore,oldpaths=_sealed_run(PREVIOUS.EXP.OUTPUT,7)
    audit=C.read(EXP.OUTPUT/'independent_root_audit_v1/AUDIT.json')
    aseal=C.read(EXP.OUTPUT/'independent_root_audit_v1/AUDIT_SEAL.json')
    if (audit['rows']!=238 or audit['failure_count'] or audit['failures']
            or aseal['receipt_sha256']!=C.sha(EXP.OUTPUT/'independent_root_audit_v1/AUDIT.json')
            or audit['prediction_seal']['sha256']!=C.sha(EXP.OUTPUT/'PREDICTION_SEAL.json')):
        raise ValueError('Independent full audit required')
    inputpath=EXP.OUTPUT/'INPUTS.jsonl.gz'; obspath=OLD.OBS.OUTPUT/'OBSERVATIONS.jsonl.gz'
    inputseal=C.read(EXP.OUTPUT/'INPUT_SEAL.json');obsseal=C.read(OLD.OBS.OUTPUT/'OBSERVATION_SEAL.json')
    if C.sha(inputpath)!=inputseal['input_sha256'] or C.sha(obspath)!=obsseal['observations_sha256']:
        raise ValueError('Actual sealed input/observation bytes required')
    task=_exact_row(inputpath); observed=_exact_row(obspath)
    if (task['states']!=observed['states'] or task['spec']['visible']!=observed['visible_measurements']
            or task['new_chain_observation']!=observed['chain_observation'] or task['baseline']!=new['arms'][R.BASELINE]):
        raise ValueError('Same original inputs and e baseline required')
    if EXP.E._digest(task)!=new['input_sha256']:
        raise ValueError('Saved frame must bind this exact input')
    imagepath=C.ROOT/task['image']; maskpath=C.ROOT/observed['mask_binding']['path']
    if C.sha(imagepath)!=task['image_sha256'] or C.sha(maskpath)!=observed['mask_binding']['sha256']:
        raise ValueError('Actual RGB/mask changed')
    image=cv2.cvtColor(cv2.imread(str(imagepath)),cv2.COLOR_BGR2RGB)
    with np.load(maskpath,allow_pickle=False) as arrays: mask=arrays['pallet_mask_clean'].copy()
    geometry=observed['initial_geometry']; K=geometry['K']; states=task['states']
    plans={key:PREVIOUS._saved_plan(new['arms'][key]) for key in METHODS}
    active=plans[R.TP_NATIVE]; role_records=_roles(task,active)
    original_O=[r['original_domain'] for r in task['O_localization_abstention']['records']]
    if original_O!=[d for d in observed['domains'] if d['state']=='O']:
        raise ValueError('Archived O provenance changed')
    coarse=R._original_coarse_geometry(new['original_first_pose_prior_inputs'],K)
    h,w=image.shape[:2]
    support=[np.asarray(geometry['initial_projection_xy']),np.asarray([m['xy'] for m in task['spec']['visible']]).reshape(-1,2)]
    xy=np.vstack(support);lo=np.maximum(xy.min(0)-45,0);hi=np.minimum(xy.max(0)+45,[w,h])
    bounds=tuple(map(float,(lo[0],hi[0],hi[1],lo[1])))
    arms={key:new['arms'][key] for key in METHODS}; arms['PREVIOUS_V24']=old['arms'][PREVIOUS_ARM];arms[R.BASELINE]=new['arms'][R.BASELINE]
    metrics={key:score['metrics'][key] for key in (*METHODS,R.BASELINE)};metrics['PREVIOUS_V24']=oldscore['metrics'][PREVIOUS_ARM]
    projections={key:OLD._model(arm,K) for key,arm in arms.items()}
    energies={key:R._energies(arms[key],key) for key in METHODS}
    depths={key:_family_depth(arm) for key,arm in arms.items()}
    DRAW._font();assets=OUTPUT/'assets';assets.mkdir(exist_ok=False)
    stages=[]; plan_counts=None; nms_audit=None
    for index,title in enumerate(TITLES):
        fig,ax=plt.subplots(figsize=(10.4,7.4),dpi=150)
        _axis(ax,image,bounds,title,full=index==0)
        if index==0:
            ax.add_patch(Rectangle((bounds[0],bounds[3]),bounds[1]-bounds[0],bounds[2]-bounds[3],fill=False,ec='#00aeb5',lw=2))
        elif index==1:
            DRAW._wire(ax,np.asarray(geometry['initial_projection_xy']),'#687a8b',ls='--');OLD._frozen_points(ax,task)
            _note(ax,'첫 N3 모델 · 새 PnP를 실행한 중간 단계 아님\nS/U 점·GT wire 없음',bottom=True)
        elif index==2:
            OLD._frozen_points(ax,task)
            _note(ax,'  '.join('KP'+str(k)+':'+s for k,s in enumerate(states))+'\nB/I 관측 · O/U 범위 · S 모델만\n초기 S 좌표/RGB 조회 없음; U 정밀점 없음')
        elif index==3:
            ax.imshow(np.dstack([np.zeros_like(mask),mask,np.zeros_like(mask),mask*.19]))
            nms_audit=OLD._nms(ax,task['new_chain_observation'],dict(groups=active['active_groups']))
            _note(ax,'실제 저장 마스크 + NMS 표본\n활성 physical edge '+str(len(active['active_groups']))+'개 · 새 KP 아님',bottom=True)
        elif index==4:
            plan_counts=PREVIOUS._replacement_panel(ax,task,active,'이동-native 선택 family: '+str(arms[R.TP_NATIVE]['WD_family']))
        elif index==5:
            for d in original_O:
                if d['state']!='O' or task['initial_self_hidden'][d['native_corner']]: raise ValueError('O gate')
                center=np.asarray(d['center_xy']);aa=d['half_axes_px']
                ax.add_patch(Ellipse(center,2*aa[0],2*aa[1],angle=d.get('angle_deg',0),fc='#efaa28',ec='#b88312',alpha=.2,lw=1.5))
                DRAW._label(ax,center,'O'+str(d['native_corner'])+' 위치 가정','#9e6d00',(5,12))
            for d in task['u_regions']:
                k=d['native_corner']
                if states[k]!='U' or task['initial_self_hidden'][k]: raise ValueError('U range gate')
                aa=d['half_axes_px'];ax.add_patch(Ellipse(d['center_xy'],2*aa[0],2*aa[1],angle=d.get('angle_deg',0),fc='#8794a1',ec='#63717e',alpha=.13,ls='--'))
            DRAW._wire(ax,np.asarray(geometry['initial_projection_xy']),'#7c8894',ls='--')
            _note(ax,'O '+str([d['native_corner'] for d in original_O])+' : hard 캔버스 / 타원은 soft 위치 가정\nU '+str([d['native_corner'] for d in task['u_regions']])+' : 범위만\nS '+str([k for k,s in enumerate(states) if s=='S'])+' : 모델 wire만 · 점 없음')
        elif index==6:
            DRAW._wire(ax,np.asarray(coarse['original_first_model_projection_xy']),'#576779',ls='--')
            _note(ax,'원래 첫 N3 pose 고정 · e pose 아님\n이동: marginal Σtt / 6D: 전체 Σ6\nCAL μ 한 번 · λ=β=1\n상관된 coarse prior, 독립 영상 관측 아님')
        elif index==7:
            q=projections['PREVIOUS_V24'];DRAW._wire(ax,q,'#b45347',width=2)
            e=PREVIOUS._energies(arms['PREVIOUS_V24'],PREVIOUS_ARM)
            _note(ax,OLD._metric_text(metrics['PREVIOUS_V24'])+'\n'+PREVIOUS._energy_text(e))
            _note(ax,'WD '+str(depths['PREVIOUS_V24']['WD_family'])+f" · 카메라 중심 깊이 {depths['PREVIOUS_V24']['camera_depth_m']:.3f}m",bottom=True)
        elif index in (8,9,10):
            key=METHODS[index-8]
            R._pose_panel(ax,projections[key],arms[key],key,metrics[key],energies[key],bounds)
            _note(ax,'WD '+str(depths[key]['WD_family'])+f" · 카메라 중심 깊이 {depths[key]['camera_depth_m']:.3f}m\n새 최종 방법 비교 · optimizer 중간 iterate 아님",bottom=True)
        else:
            display=R._containment_display_geometry(projections[R.TP_C],task['outline'])
            R._containment_panel(ax,display,R.TP_C,arms[R.TP_C],energies[R.TP_C])
            lines=['이전176 → 이동-native 약166cm: 작은 개선, 큰 실패 유지']
            for key in ('PREVIOUS_V24',*METHODS):
                label='이전24' if key=='PREVIOUS_V24' else R.NAMES[key]
                lines.append(label+f" · z {depths[key]['camera_depth_m']:.3f}m · {depths[key]['WD_family']}")
            _note(ax,'\n'.join(lines),bottom=True)
        fig.suptitle(CASE+' · 저장 결과 설명',fontsize=10,color='#546274')
        path=assets/f'step_{index+1:02d}.png';DRAW._save(fig,path,CAPTIONS[index])
        stages.append(dict(step=index+1,title=title,caption=CAPTIONS[index],asset=str(path.relative_to(OUTPUT)),
            is_new_PnP_intermediate=False,kind='distinct_final_method' if index in (7,8,9,10) else 'saved_common_evidence_or_explanation'))
    fig,axes=plt.subplots(4,3,figsize=(25.5,23.5),dpi=160)
    for ax,stage in zip(axes.ravel(),stages):
        ax.imshow(plt.imread(OUTPUT/stage['asset']));ax.set_axis_off()
    fig.suptitle(CASE+' · 176.5cm → 165.9cm: 개선은 작고 여전히 실패\n공통 관측 과정과 서로 다른 최종 방법을 구분 · 새 추론/정답 투영 없음',fontsize=17,y=.995)
    fig.subplots_adjust(left=.006,right=.994,bottom=.006,top=.968,wspace=.008,hspace=.012)
    fig.savefig(OUTPUT/'process_montage.png',bbox_inches='tight');plt.close(fig)
    for first,last,name,heading in ((0,6,'sheet_01_observation.png','①~⑥ 실제 사진·관측·범위'),
                                   (6,12,'sheet_02_final_methods.png','⑦~⑫ 첫 pose prior와 최종 결과 비교')):
        fig,axes=plt.subplots(3,2,figsize=(20.5,21.8),dpi=165)
        for ax,stage in zip(axes.ravel(),stages[first:last]):
            ax.imshow(plt.imread(OUTPUT/stage['asset']));ax.set_axis_off()
        fig.suptitle(CASE+' · '+heading+'\n176.5 → 165.9cm: 작은 개선, 약1.66m 실패 유지',fontsize=18,y=.995)
        fig.subplots_adjust(left=.004,right=.996,bottom=.004,top=.962,wspace=.006,hspace=.009)
        fig.savefig(OUTPUT/name,bbox_inches='tight');plt.close(fig)
    bindings=dict(plan['dependency_sha256'])
    paths+=oldpaths+[inputpath,obspath,imagepath,maskpath,OUTPUT/'PLAN.json']
    bindings.update(dict(_bind(p) for p in paths))
    data=dict(schema='USER_SELECTED_031426_POSTSCORE_PROCESS_V25',case_id=CASE,
        selected_by_user_after_scores=True,predeclared_four_case=False,partition=task['partition'],
        image=task['image'],image_sha256=task['image_sha256'],crop_bounds=bounds,crop_from_initial_model_and_BI_not_reference=True,
        per_corner_roles=role_records,state_sequence=states,plan_counts=plan_counts,active_physical_edge_count=len(active['active_groups']),
        original_O_localization_soft_domains=original_O,hard_O_domains=task['spec']['domains'],U_range_count=len(task['u_regions']),
        O_domain_count=len(original_O),S_ids=[k for k,s in enumerate(states) if s=='S'],
        NMS_display_audit=nms_audit,S_U_point_coordinates_or_markers_exported=False,
        S_model_projection_wire_only=True,U_region_only=True,original_first_N3_prior=coarse,
        original_first_N3_camera_center_xyz_m=new['original_first_pose_prior_inputs']['TRANSLATION']['original_first_center_camera_xyz'],
        methods={key:dict(metric=metrics[key],**depths[key],model_wire_projection_xy=projections[key],
            energy=energies.get(key),source_replacement_plan=plans.get(key)) for key in arms},
        stages=stages,optimizer_intermediate_poses_saved=False,no_intermediate_pose_invented=True,
        result='176.5351cm to 165.9262cm is a small exploratory reduction, still a large failure',
        explanation_scope='Saved family/depth and image/prior objectives do not certify semantic correspondence or accurate depth; no isolated causal proof.',
        reference_axes_and_intrinsics_status='UNCONFIRMED in additional143 reference contract; derived annotation/PnP reference, not independent metrology',
        GT_pose_or_corner_reads=0,GT_wire_or_projection=0,new_inference_or_pixel_search_or_pose_fit=0,
        existing_default_e_preserved=True,source_bindings=bindings)
    C.write(OUTPUT/'PROCESS_DATA.json',data)
    rows=[]
    for key in (R.BASELINE,'PREVIOUS_V24',*METHODS):
        label='기존 e 기본값' if key==R.BASELINE else '이전24 · 176cm' if key=='PREVIOUS_V24' else R.NAMES[key]
        m=metrics[key];d=depths[key]
        rows.append(f'<tr><td>{label}</td><td>{m["translation_cm"]:.4f}</td><td>{m["rotation_deg"]:.4f}</td><td>{d["WD_family"]}</td><td>{d["camera_depth_m"]:.4f}</td></tr>')
    rrows=''.join('<tr><td>KP'+str(r['native_corner'])+'</td><td>'+r['state']+'</td><td>'+html.escape(r['role'])+'</td><td>'+html.escape(str((r['BI_observation'] or {}).get('source','—')))+'</td></tr>' for r in role_records)
    sections=''.join('<section><h2>'+html.escape(s['title'])+'</h2><img src="'+s['asset']+'"><p>'+html.escape(s['caption'])+'</p></section>' for s in stages)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>031426 실제 저장 처리 순서</title><style>body{margin:0;background:#edf2f6;color:#21384a;font-family:system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:22px}section{background:white;padding:15px;margin:18px 0;border-radius:8px}img{width:100%;height:auto}p{line-height:1.65}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #bbc9d3;padding:8px;text-align:left}.scroll{overflow:auto}.warning{background:#fff0d9;padding:15px}h1{font-size:27px}h2{font-size:21px}</style><main><h1>wood_night_01:031426 — 176cm 사례의 실제 처리 순서</h1><p class="warning">이전 176.5351cm에서 이동-native 165.9262cm로 약10.61cm 줄었지만, 약1.66m 오차가 남는 큰 실패입니다. 개선 완료나 성공으로 표현하지 않습니다. 이 사례는 점수 공개 후 사용자가 골랐으며 사전등록 네 사례가 아닙니다.</p><p>①~⑦은 저장된 공통 관측·계획의 설명입니다. 이 시점마다 새로 PnP를 실행한 것이 아닙니다. ⑧~⑪은 서로 다른 방법의 최종 저장 결과이며 연속 최적화 중간 자세가 아닙니다. 중간 iterate는 저장되지 않았고 새로 만들지 않았습니다.</p><div class="scroll"><table><tr><th>방법</th><th>이동 cm</th><th>회전 °</th><th>W/D 선택</th><th>카메라 중심 깊이 m</th></tr>'''+''.join(rows)+'''</table></div><p>깊이는 저장 모델 중심의 카메라 z이며 정답 깊이와 비교한 값이 아닙니다. 영상 마스크·선 비용을 만족하는 듯 보여도 W/D·채널 대응·깊이의 정확성은 보장되지 않습니다. 여기서는 저장된 family와 비용, 자세만 보여 주며 특정 항이 오차의 원인이라고 단정하지 않습니다.</p><p>추가143의 참조는 주석/PnP 유도값입니다. 참조 axes/intrinsics는 UNCONFIRMED이며 독립 계측이 아닙니다. S는 모델 wire만, U는 범위만 표시합니다. O 타원은 예측 위치의 경험 가정이고 실제 가림물 모양이 아닙니다. GT wire·S/U 점·새 픽셀 탐색·새 추론·새 pose fit은 없습니다.</p><h2>나눠 보기</h2><p><a href="sheet_01_observation.png">①~⑥ 관측 과정 PNG</a> · <a href="sheet_02_final_methods.png">⑦~⑫ 최종 비교 PNG</a></p><h2>전체 순서</h2><a href="process_montage.png"><img src="process_montage.png" alt="12단계 3열 그림"></a><h2>각 KP의 저장 역할</h2><div class="scroll"><table><tr><th>KP</th><th>state</th><th>역할</th><th>남은 B/I 원출처</th></tr>'''+rrows+'''</table></div><p>활성 변 '''+str(len(active['active_groups']))+'개 · 유지 complete q '+str(plan_counts['retained_q'])+'개 · CAL 접선 '+str(plan_counts['archived_tangent'])+'개 · O 범위 '+str(len(original_O))+'개 · U 범위 '+str(len(task['u_regions']))+'''개. 값은 이동-native가 고른 family의 실제 계획입니다. 다른 방법의 계획은 JSON에 따로 보존했습니다.</p><p><a href="PROCESS_DATA.json">표시값·비용·계획·출처 JSON</a> · <a href="PLAN.json">사용자 요청 후 고정한 렌더 계획</a></p>'''+sections+'</main></html>'
    (OUTPUT/'index.html').write_text(page,encoding='utf-8')
    for relative,digest in bindings.items():
        if C.sha(C.ROOT/relative)!=digest: raise ValueError('Input changed during render')
    C.write(OUTPUT/'RENDER_SEAL.json',dict(schema='USER_REQUESTED_031426_RENDER_SEAL_V25',case_id=CASE,
        renderer_source_sha256=C.sha(__file__),source_bindings=bindings,
        output_sha256={str(p.relative_to(OUTPUT)):C.sha(p) for p in OUTPUT.rglob('*') if p.is_file()},
        selected_after_scores=True,predeclared_four_case=False,new_inference_or_pose_fit=0,GT_projection=0,
        S_U_KP_point_markers=0,steps=len(stages)))
    print(dict(output=str(OUTPUT),case=CASE,stages=len(stages),metrics={key:metrics[key]['translation_cm'] for key in metrics}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','render'))
    if parser.parse_args().command=='prepare': prepare()
    else: render()
