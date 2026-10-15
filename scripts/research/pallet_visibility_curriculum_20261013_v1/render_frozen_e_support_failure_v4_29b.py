"""One fixed postscore V29 guard-failure illustration; no new measurement/fit.

Original RGB, gated archived B/I q, exact saved actual line support and the
two separately saved model poses are displayed. e is a prediction, not GT.
ORIGINAL and SWAPPED native channel correspondences are explicitly distinct.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from . import render_geometry_guided_process_v4_29 as DRAW
from . import diagnose_frozen_e_pose_under_v29_v4_29b as D

C,E,M=D.C,D.E,D.M
CASE='eval_noapril:1775201416944730624'
ARM='GEOMETRY_EDGE_ALL_BI_NATIVE'
EDGE=(0,3)
DIAG=D.OUTPUT
OUTPUT=DIAG/'actual_failure_noapril_1775201416944730624_v1'
DIAG_SHA256='b437981d5c4ddd3af097f632e1588a825309a3a4daf6c1535890947eb659de80'
DRAW_SHA256='ec75d1630f3dcfa7cfcbf740a2f70785c46fe86aa99278177e5a20fbb9633f5f'


def gated_BI(task):
    result=[]
    for record in task['spec']['visible']:
        k=record['native_corner']
        if task['initial_self_hidden'][k] or task['states'][k]not in('B','I'):continue
        result.append(dict(native_corner=k,state=task['states'][k],xy=record['xy'],source=record['source']))
    return result


def display_ROI(actual, e_segment, image_hw, margin=35):
    h,w=map(int,image_hw)
    points=np.vstack([actual,e_segment]);points=points[np.isfinite(points).all(1)]
    if not len(points):raise ValueError('Actual finite support/display model required')
    lo=np.floor(points.min(0)-margin).astype(int);hi=np.ceil(points.max(0)+margin).astype(int)
    l,t=max(0,min(w-1,int(lo[0]))),max(0,min(h-1,int(lo[1])))
    r,b=max(l+1,min(w,int(hi[0])+1)),max(t+1,min(h,int(hi[1])+1))
    return l,r,t,b


def pure_checks():
    assert display_ROI(np.array([[4.,10.],[20.,15.]]),np.array([[-5.,9.],[12.,40.]]),(60,80),5)==(0,26,4,46)
    class Poison(dict):
        def __getitem__(self,key):
            if key=='native_corner':return super().__getitem__(key)
            raise AssertionError('S/U observed field read')
    assert gated_BI(dict(spec=dict(visible=[Poison(native_corner=0),Poison(native_corner=1)]),
        states=['S','U']+['B']*6,initial_self_hidden=[True]+[False]*7))==[]
    return 2


def prepare():
    if OUTPUT.exists():raise FileExistsError('Preserve each diagnostic figure generation')
    if C.sha(DIAG/'DIAGNOSIS.json')!=DIAG_SHA256 or C.sha(DRAW.__file__)!=DRAW_SHA256:
        raise ValueError('Exact saved diagnosis and frozen drawing source required')
    seal=C.read(DIAG/'DIAGNOSIS_SEAL.json')
    if seal['diagnosis_sha256']!=DIAG_SHA256 or seal['count']!=49 or seal['arm_count']!=4:
        raise ValueError('Completed immutable TEST49x4 diagnosis required')
    protocol=C.read(M.OUTPUT/'PROTOCOL.json')
    if CASE not in protocol['ids'][46:95] or ARM not in M.ARMS:
        raise ValueError('Exact fixed already-scored TEST case/arm required')
    paths=[Path(__file__).resolve(),Path(D.__file__).resolve(),Path(DRAW.__file__).resolve(),
        DIAG/'DIAGNOSIS.json',DIAG/'DIAGNOSIS_SEAL.json',DIAG/'PLAN.json',DIAG/'OWNER_COMPLETION_V1.json']
    paths.extend(M.OUTPUT/name for name in ('PROTOCOL.json','INPUT_SEAL.json','PREDICTION_SEAL.json','SCORE_SEAL.json'))
    bindings={str(p.relative_to(C.ROOT)):C.sha(p)for p in paths}
    count=pure_checks();OUTPUT.mkdir()
    C.write(OUTPUT/'PLAN.json',dict(schema='FIXED_POSTSCORE_V29_ACTUAL_SUPPORT_FAILURE_FIGURE_PLAN_V1',
        UTC=datetime.now(timezone.utc).isoformat(),id=CASE,arm=ARM,edge=list(EDGE),
        purpose='Show saved e ORIGINAL good-scored prediction excluded by finite actual-support guard, versus separately saved SWAPPED selected pose. e is not GT.',
        input_sha256=bindings,source_sha256={str(Path(__file__).resolve().relative_to(C.ROOT)):C.sha(__file__)},
        pure_display_state_gate_checks=count,
        actual_read_scope='One fixed matching V29 input/prediction row, saved DIAG and original RGB; other journal lines are not JSON-decoded. No mask/reference/model fitting.',
        legend=['Original RGB unchanged','archived observed B/I q only','saved actual NMS inliers/TLS coefficient, no refit',
            'saved e ORIGINAL model segment0-3 (prediction, not GT)','saved selected SWAPPED model segment0-3 (different native correspondence)'],
        crop='Display-only bounds around actual support and e model segment plus35px, clipped to canvas; whole RGB+model view accompanies it.',
        S_U_point_markers=0,GT_wire=0,new_RGB_sampling_NMS_TLS_NN_posefit_or_GT_decode=0,
        actual_render_requires_root_fixed_case_GO=True,original_runtime_or_V30_changes=0))
    C.write(OUTPUT/'PLAN_SEAL.json',dict(plan_sha256=C.sha(OUTPUT/'PLAN.json'),input_sha256=bindings))
    print(dict(plan_sha256=C.sha(OUTPUT/'PLAN.json'),fixed_case_rows_decoded=0,pure_checks=count),flush=True)


def render(*,root_fixed_case_GO=False):
    if root_fixed_case_GO is not True:raise ValueError('Root fixed postscore case render GO required before IO')
    if (OUTPUT/'PROCESS_DATA.json').exists()or(OUTPUT/'actual_support_failure.png').exists():
        raise FileExistsError('Preserve prior actual render or partial result')
    plan=C.read(OUTPUT/'PLAN.json');seal=C.read(OUTPUT/'PLAN_SEAL.json')
    if C.sha(OUTPUT/'PLAN.json')!=seal['plan_sha256']or plan['id']!=CASE or plan['arm']!=ARM:
        raise ValueError('Exact predeclared fixed case/arm/source required')
    E._verify_sources(plan['input_sha256'])
    source_plan=C.read(DIAG/'PLAN.json')
    E._verify_sources(source_plan['deferred_journal_sha256'])
    tasks,task_digest=DRAW._exact_rows(M.OUTPUT/'INPUTS.jsonl.gz',(CASE,))
    frames,frame_digest=DRAW._exact_rows(M.OUTPUT/'PREDICTIONS.jsonl.gz',(CASE,))
    task,frame=tasks[CASE],frames[CASE]
    if frame['input_sha256']!=E._digest(task)or task['baseline']!=frame['arms'][M.ARMS[0]]:
        raise ValueError('Exact frozen matched input/baseline/prediction required')
    data=C.read(DIAG/'DIAGNOSIS.json');diagnosis=next(r for r in data['rows']if r['id']==CASE)['arms'][ARM]
    e=task['baseline'];selected=frame['arms'][ARM]
    if e['WD_family']!='ORIGINAL_WD'or selected['WD_family']!='SWAPPED_WD' or not selected['accepted']:
        raise ValueError('Predeclared original-e / swapped selected example changed')
    ef=task['families'][e['WD_family']];sf=task['families'][selected['WD_family']]
    er,et=D.validate_saved_pose_basis(e['actual_pose'],ef)
    sr,st=D.validate_saved_pose_basis(selected['actual_pose'],sf)
    gap=D._physical_body_gap(ef,sf)
    if gap>1e-12:raise ValueError('Same registered unordered box dimensions required')
    eplan=selected['new_geometry_edge_cost_plans'][e['WD_family']]
    groups=[g for g in eplan['active_groups']if sorted([g['native_corner'],g['native_neighbor']])==list(EDGE)]
    if len(groups)!=1:raise ValueError('Exactly one named physical-edge factor0-3 required')
    group=groups[0];k,mate=group['native_corner'],group['native_neighbor']
    if task['initial_self_hidden'][k]or task['initial_self_hidden'][mate]or task['states'][k]not in('B','I')or task['states'][mate]=='S':
        raise ValueError('Permanent state gates before actual chain payload')
    ref=group['selected_chain_ref'];observation=task['v29_observation']
    ray=observation['chain_observation']['edge_records'][ref['ray_index']]
    chain=ray['chains'][ref['chain_index']]
    if chain['chain_id']!=ref['chain_id']:raise ValueError('Exact archived selected actual chain')
    actual=np.asarray(chain['observed_points'],float)[np.asarray(chain['inlier_mask'],bool)]
    if not np.array_equal(actual,group['actual_inlier_xy']):raise ValueError('Original actual inliers unchanged')
    n=np.asarray(group['normal_unit_xy']);offset=float(group['offset_px']);axis=np.array([-n[1],n[0]])
    oldn=np.asarray(chain['normal'],float);norm=np.linalg.norm(oldn);oldn=oldn/norm;oldc=float(chain['offset'])/norm
    sign=1. if float(oldn@n)>=0. else -1.
    if not np.allclose(sign*oldn,n,atol=1e-12,rtol=0)or not np.isclose(sign*oldc,offset,atol=1e-10,rtol=0):
        raise ValueError('Exact original TLS coefficients, up to line-sign equivalence')
    eq=C.project(ef['native_xyz'],er,et,task['spec']['K']);sq=C.project(sf['native_xyz'],sr,st,task['spec']['K'])
    if not np.allclose(sq,selected['solution']['projected_native'],atol=1e-8,rtol=0):
        raise ValueError('Selected model projection changed')
    guard=[a for a in diagnosis['primary_e_evaluation']['guard_audits']['actual_edge_support_constraint_audit']if a['component_id']==group['component_id']]
    if len(guard)!=1 or guard[0]['passed']:raise ValueError('Predeclared saved e support guard failure required')
    guard=guard[0]
    span=[float((actual@axis).min()),float((actual@axis).max())]
    projected=np.asarray([axis@eq[k],axis@eq[mate]])
    finite=[span[0]-float(projected.min())+11.,float(projected.max())-span[1]+11.]
    if not np.allclose(finite,guard['finite_actual_support_slacks_px'],atol=1e-8,rtol=0):
        raise ValueError('Saved finite guard slack must match the displayed model/actual support')
    image_path=C.ROOT/task['image']
    if C.sha(image_path)!=task['image_sha256']:raise ValueError('Original RGB bytes changed')
    import cv2
    image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
    visible=gated_BI(task);roi=display_ROI(actual,eq[list(EDGE)],image.shape[:2]);l,r,t,b=roi
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.font_manager import FontProperties,fontManager
    from . import render_source_preserving_tangent_mask_process_v4_24 as FONT
    font=FONT.DRAW._font();fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig=plt.figure(figsize=(17,11));gs=fig.add_gridspec(2,2,height_ratios=[4,1.5],width_ratios=[1,1.25])
    full=fig.add_subplot(gs[0,0]);crop=fig.add_subplot(gs[0,1]);text=fig.add_subplot(gs[1,:]);text.set_axis_off()
    colors=dict(e='#2473b6',selected='#a53987',actual='#00a082',q='#d07b20')
    order=np.argsort(actual@axis)
    for ax in (full,crop):
        ax.imshow(image)
        ax.plot(actual[order,0],actual[order,1],c=colors['actual'],lw=3.,label='보관된 실제 선 지지 (새 fit 없음)')
        ax.scatter(actual[:,0],actual[:,1],c=colors['actual'],s=10,zorder=5)
        ax.plot(eq[list(EDGE),0],eq[list(EDGE),1],c=colors['e'],ls='--',lw=2.4,label='e ORIGINAL 명명변0–3 · 예측, GT 아님')
        ax.plot(sq[list(EDGE),0],sq[list(EDGE),1],c=colors['selected'],lw=2.4,label='선택 SWAPPED 명명변0–3 · 다른 대응')
        for m in visible:
            if ax is crop and m['native_corner']not in EDGE:continue
            ax.scatter(*m['xy'],marker='x',c=colors['q'],s=38,zorder=6)
            ax.text(m['xy'][0]+3,m['xy'][1],str(m['native_corner'])+m['state'],color=colors['q'],fontsize=9)
        ax.set_axis_off()
    full.add_patch(Rectangle((l,t),r-l,b-t,fill=False,color='#dc9e23',lw=1.3))
    full.set_xlim(0,image.shape[1]);full.set_ylim(image.shape[0],0);full.set_title('전체 원 RGB + 저장 선분 · 노랑 상자가 확대 범위')
    crop.set_xlim(l,r);crop.set_ylim(b,t);crop.set_title('실제 선 지지와 서로 다른 WD의 저장 모델 선분')
    crop.legend(loc='best',fontsize=8,framealpha=.94)
    em=diagnosis['saved_scalar_metrics']['e'];sm=diagnosis['saved_scalar_metrics']['selected']
    ev=diagnosis['primary_e_evaluation'];sv=diagnosis['selected_evaluation']
    directed='; '.join(f"k{a['native_corner']} 지지 slack {a['slack_px']:.2f}px" for a in guard['directed_support']if a['hard_order_guard_applied'])
    notes=[CASE+' / '+ARM,
        f"저장 참조 대비 T: e {em['translation_cm']:.3f}cm → 선택 {sm['translation_cm']:.3f}cm. e도 정답이 아닌 저장 예측 자세입니다.",
        f"e guard: 제외 · finite slack [{finite[0]:.3f}, {finite[1]:.3f}]px (음수는 위반); {directed}.",
        f"현재 비용 J: e {ev['total']:.6g} / 선택 {sv['total']:.6g}; N {ev['native']:.6g}/{sv['native']:.6g}, O {ev['O']:.6g}/{sv['O']:.6g}, S {ev['S']:.6g}/{sv['S']:.6g}, C=0.",
        '실제 지지는 정밀 코너가 아닙니다. e ORIGINAL과 선택 SWAPPED는 같은 물리 상자 규격의 다른 native 채널 대응이며 같은 코너 의미로 겹쳐 해석하면 안 됩니다.',
        '주황 ×는 보관 B/I 관측만. S/U 점·GT wire 없음. 원래 finite support가 e를 제외했다는 기록이며 잘못된 선/WD의 인과 증명이나 V30 해결 확인은 아닙니다.']
    text.text(.01,.98,'\n'.join(notes),va='top',fontsize=10,linespacing=1.6)
    fig.tight_layout();png=OUTPUT/'actual_support_failure.png';fig.savefig(png,dpi=160);plt.close(fig)
    product=dict(schema='FIXED_V29_ACTUAL_LINE_SUPPORT_FAILURE_DISPLAY_V1',id=CASE,arm=ARM,edge=list(EDGE),
        image=E._binding(image_path),image_shape_hw=list(image.shape[:2]),ROI_original_xy=list(roi),
        original_RGB_brightness_color_unchanged=True,actual_inlier_xy=actual.tolist(),
        TLS_normal=n.tolist(),TLS_offset=offset,actual_support_tangent_interval_px=span,
        actual_selected_chain_ref=ref,component_id=group['component_id'],gated_archived_BI=visible,
        saved_e_family=e['WD_family'],selected_family=selected['WD_family'],same_unordered_physical_body_gap_m=gap,
        saved_e_pose=e['actual_pose'],saved_selected_pose=selected['actual_pose'],
        saved_e_model_projection_xy=eq.tolist(),selected_model_projection_xy=sq.tolist(),
        native_channel_correspondence_different=True,e_is_prediction_not_GT=True,
        cached_e_actual_support_guard=guard,finite_slack_recalculated_from_display_geometry=finite,
        e_costs=ev,selected_costs=sv,sealed_scalar_metrics=diagnosis['saved_scalar_metrics'],
        exact_row_text_sha256=dict(input=task_digest,prediction=frame_digest),
        plan=E._binding(OUTPUT/'PLAN.json'),diagnosis=E._binding(DIAG/'DIAGNOSIS.json'),
        new_measurement_TLS_NN_posefit_or_GT_decode=0,S_U_point_markers=0,GT_wire=0,
        limitations='One postscore fixed example, not a performance sample. Actual support is conditional observed-line evidence; neither e nor selected is GT. Crop may hide continued predicted segments; full projections are retained here.')
    C.write(OUTPUT/'PROCESS_DATA.json',product)
    md=['# 저장된 실제 선 지지가 e를 제외한 사례','', '![원 RGB와 edge 확대](actual_support_failure.png)','',*notes,'',
        '[정확한 좌표·slack·비용·봉인 출처](PROCESS_DATA.json)','']
    (OUTPUT/'REPORT_KO.md').write_text('\n\n'.join(md),encoding='utf-8')
    E._verify_sources(plan['input_sha256'])
    assets={p.name:C.sha(p)for p in(OUTPUT/'PROCESS_DATA.json',OUTPUT/'REPORT_KO.md',png)}
    C.write(OUTPUT/'RENDER_SEAL.json',dict(schema='FIXED_V29_ACTUAL_SUPPORT_FAILURE_RENDER_SEAL_V1',
        id=CASE,arm=ARM,source_sha256=C.sha(__file__),plan_sha256=C.sha(OUTPUT/'PLAN.json'),
        assets=assets,S_U_point_markers=0,GT_wire=0,new_measurement_or_fit=0))
    print(dict(report=str(OUTPUT/'REPORT_KO.md'),seal_sha256=C.sha(OUTPUT/'RENDER_SEAL.json'),assets=assets),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','render'))
    parser.add_argument('--root-fixed-case-GO',action='store_true');args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:render(root_fixed_case_GO=args.root_fixed_case_GO)
