"""Predeclared v24 actual-image process; no rendering before root audit GO.

Five uniform methods, four unchanged historical cases, ten stored stages.
Only saved images/masks/observations/poses/scores are rendered. No pose fit,
GT wire, S/U point marker, new model forward or reference coordinate read.
"""
from pathlib import Path
import argparse
import html
import os
from urllib.parse import quote

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from . import common as C
from . import evaluate_source_preserving_tangent_mask_v4_24 as EXP
from . import render_complete_bi_bounded_priors_process_v4_23 as PREVIOUS
from . import build_repair_campaign_progress_v3_v18 as ACCOUNT

OLD,DRAW,CONTEXT=PREVIOUS.OLD,PREVIOUS.DRAW,PREVIOUS.PREVIOUS
OUTPUT=EXP.OUTPUT.parent/'source_preserving_tangent_mask_process_v4_24'
PLAN=EXP.OUTPUT.parent/'source_preserving_tangent_mask_report_plan_v4_24'
AUDIT_SUBDIR='independent_root_audit_v1'
CASES=PREVIOUS.CASES
BASELINE,ALL_NATIVE,SOURCE_NATIVE,MASK_FIT,LOCKED=(EXP.ARMS[i]for i in (0,1,3,5,6))
METHODS=(BASELINE,ALL_NATIVE,SOURCE_NATIVE,MASK_FIT,LOCKED)
NAMES={BASELINE:'기본 e',ALL_NATIVE:'전체 대체 · CAL 접선',SOURCE_NATIVE:'원출처 보존 · CAL 접선',
    MASK_FIT:'원출처 보존 · 연속 마스크',LOCKED:'같은 연속 마스크 · e family 고정'}
COLORS={BASELINE:'#2879c3',ALL_NATIVE:'#d39b14',SOURCE_NATIVE:'#1a9c78',MASK_FIT:'#9b57c4',LOCKED:'#df5f42'}
TITLES=('① 초기 모델·고정 B/I','② 실제 마스크·보관된 모든 NMS 대안','③ O 캔버스·U 범위·기존 부분 제약',
    '④ 별도 bounded O/S 모델 prior','⑤ ALL / SOURCE 비용 계획 비교','⑥ 기본 e 저장 자세',
    '⑦ 전체 대체·CAL 접선 저장 자세','⑧ 원출처 보존·CAL 접선 저장 자세',
    '⑨ 연속 마스크 저장 자세','⑩ e family 고정 연속 마스크 저장 자세')
CAPTIONS=(
    '회색 선은 초기 모델 투영. 분홍 ×는 기존 complete B/I, 회색 +는 기존 부분관측의 보관 q. S/U 입력 점은 조회하거나 표시하지 않음.',
    '녹색은 실제 저장 마스크, 회색은 모든 NMS 선 대안, 청록은 활성 physical edge에 속한 보관 NMS 대안. 실제 비용 성분은 ⑤에 표시하며 반복 fit은 독립 관측이 아님.',
    'O의 hard 범위는 닫힌 영상 캔버스. U는 넓은 경험 범위만 사용. 기존 부분 B/I의 선·접선 구간은 그대로이며 새 정밀 KP를 만들지 않음.',
    'O 타원은 raw 예측의 경험 위치 가정이며 bounded 벌점만 사용. S는 초기 모델 가시성 prior로만 사용. S/U 점 표시는 없고 S 영상·2D 조회도 없음.',
    '각 패널은 해당 방법이 선택한 family의 보관 비용 계획. 실패 시 joint 해로 채택되지 않은 계획임. RGB/unknown complete q 보존과 fallback q 대체를 구분하며 주황 점선 길이는 CAL σ.',
    '파란 선은 기존 e의 저장 모델 투영. 현재 비용함수의 에너지와 동등한 것으로 바꾸지 않음. 참조/GT 선이 아님.',
    '전체 대체 방법: 기존20d 자격의 complete B/I 비용을 대체하고 접선에 CAL 분산1+γa 적용. a는 ORIGINAL guide와 원래 유효 Σ의 tᵀΣt.',
    '원출처 보존 방법: 정확히 VISIBLE_BACKBONE_FALLBACK인 complete q만 대체 가능. RGB·unknown complete q/Σ/신뢰도는 유지하며 같은 전역 γ 사용.',
    '같은 native·O/S에 λM을 최적화와 선택에서 한 번만 더함. 모든 기존 후보와 SOURCE의 연속 endpoint도 유지. 더 낮은 비용이 더 정확한 포즈를 보장하지 않음.',
    '앞선 e가 예측한 family 이름만 전체 후보 인증 전에 고정. e의 숫자 자세를 복사하거나 대체하지 않음. 요청 family에서 해가 없으면 own coarse fallback이며 실패로 표시.')


def _binding(path):
    path=Path(path).resolve();return str(path.relative_to(C.ROOT)),C.sha(path)


def _url(path):return quote(os.path.relpath(Path(path).resolve(),OUTPUT),safe='/')


def prepare_plan():
    if PLAN.exists():raise FileExistsError('Preserve each predeclaration')
    if (EXP.OUTPUT/'SCORE_SEAL.json').exists():raise ValueError('Declare before new24 scores')
    PLAN.mkdir(parents=True);source=Path(__file__).resolve();(PLAN/source.name).write_bytes(source.read_bytes())
    paths=[Path(EXP.__file__),Path(PREVIOUS.__file__),EXP.PLAN,
        EXP.TCAL.OUTPUT/'MODEL.json',EXP.TCAL.OUTPUT/'MODEL_SEAL.json',
        EXP.SOURCE_CAL.OUTPUT/'MODEL.json',EXP.SOURCE_CAL.OUTPUT/'MODEL_SEAL.json']
    C.write(PLAN/'PREDECLARATION.json',dict(schema='SOURCE_PRESERVING_TCAL_MASK_PROCESS_PREDECLARATION_V24',
        case_ids=list(CASES),same_cases_as_v23=True,uniform_methods=list(METHODS),declared_all7_arms=list(EXP.ARMS),
        renderer_source_sha256=C.sha(source),declared_before_new24_metric_read=True,stages_per_case=10,
        requires_all238_all7_prediction_seal_score_and_zero_failure_independent_audit=True,
        requires_explicit_root_postscore_audit_memory_GO=True,case_best_method_or_reference_routing=False,
        S_U_point_markers=False,U_regions_only=True,no_GT_wire_or_reference_pose_corner_reads=True,
        no_new_image_model_or_pose_fit=True,default_e_unchanged=True,dependencies=dict(_binding(p)for p in paths)))
    print(dict(predeclared_cases=list(CASES),methods=list(METHODS),stages=10,new24_metrics_read=False),flush=True)


def _load_plan():
    plan=C.read(PLAN/'PREDECLARATION.json')
    if (tuple(plan['case_ids'])!=CASES or tuple(plan['uniform_methods'])!=METHODS
            or plan['declared_all7_arms']!=list(EXP.ARMS)or not plan['declared_before_new24_metric_read']
            or C.sha(__file__)!=plan['renderer_source_sha256']):raise ValueError('Exact predeclared cases/methods/source required')
    for path,digest in plan['dependencies'].items():
        if C.sha(C.ROOT/path)!=digest:raise ValueError('Predeclared source/model dependency changed')
    return plan


def _completed():
    folder=EXP.OUTPUT;protocol=C.read(folder/'PROTOCOL.json');pred=C.read(folder/'PREDICTION_SEAL.json');score=C.read(folder/'SCORE_SEAL.json')
    if (pred['rows']!=238 or not pred.get('all7_arms_complete')or not pred.get('all_declared_arms_complete')
            or pred['arms']!=list(EXP.ARMS)or pred['arm_count']!=7
            or C.sha(folder/'PREDICTIONS.jsonl.gz')!=pred['predictions_sha256']
            or C.sha(folder/'PROTOCOL.json')!=pred['protocol_sha256']
            or C.sha(folder/'INPUT_SEAL.json')!=pred['input_seal_sha256']
            or C.sha(folder/'PREDICTION_SEAL.json')!=score['prediction_seal']['sha256']):raise ValueError('Complete7arm/238 prediction+score required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):EXP.E._verify_sources(protocol[key])
    auditpath=folder/AUDIT_SUBDIR/'AUDIT.json';asealpath=auditpath.with_name('AUDIT_SEAL.json')
    audit=C.read(auditpath);aseal=C.read(asealpath)
    if (audit['rows']!=238 or audit['failure_count']or audit['failures']or C.sha(auditpath)!=aseal['receipt_sha256']
            or audit['prediction_seal']['sha256']!=C.sha(folder/'PREDICTION_SEAL.json')
            or not audit['continuous_mask_factor_count_one']):raise ValueError('Independent zero-failure full238 audit required')
    selected={};ordered=[]
    for row in C.rows(folder/'PREDICTIONS.jsonl.gz'):
        ordered.append(row['id'])
        if row['id']in CASES:selected[row['id']]=row
    if ordered!=protocol['ids']or len(set(ordered))!=238 or set(selected)!=set(CASES):raise ValueError('Same238 and four fixed cases required')
    reports={};scores={}
    for part,count in ACCOUNT.COUNTS.items():
        path=folder/(part+'_SCORE.json')
        if C.sha(path)!=score['score_sha256'][part]:raise ValueError('Sealed score changed')
        report=C.read(path)
        if report['count']!=count or len(report['rows'])!=count:raise ValueError('No frame/fallback/native-flag exclusion')
        reports[part]=report;scores.update({r['id']:r for r in report['rows']})
    if set(scores)!=set(ordered):raise ValueError('All238 score inventory required')
    return protocol,selected,scores,reports,[auditpath,asealpath]


def _energies(arm,key):
    if key==BASELINE:return dict(applied=False,native=None,O=None,S=None,mask=None,weighted_mask=None,total=None)
    if not arm['accepted']:return dict(applied=True,native=None,O=None,S=None,mask=None,weighted_mask=None,total=None,failed_joint=True)
    solution=arm['solution'];prior=solution['regularizer_costs']
    if [r['kind']for r in prior]!=list(EXP.P.REGULARIZER_KINDS):raise ValueError('Separate O/S prior ledger required')
    native=float(solution['native_observation_objective']);ov,sv=[float(r['weighted_cost'])for r in prior]
    if not np.isclose(solution['total_regularized_objective'],native+ov+sv,atol=1e-9,rtol=1e-10):raise ValueError('Native/O/S ledger changed')
    selection=arm['candidate_selection'];selected=next(r for r in selection['candidates']if r['name']==selection['selected_name'])
    continuous=key in (MASK_FIT,LOCKED);weighted=0.;mask=selected['global_mask_cost']
    if continuous:
        weighted=float(solution['weighted_mask_objective'])
        if (not np.isclose(weighted,solution['global_mask_weight']*solution['global_mask_cost'],atol=1e-9,rtol=1e-10)
                or not np.isclose(solution['objective'],native+ov+sv+weighted,atol=1e-9,rtol=1e-10)
                or not np.isclose(selected['total_energy'],solution['objective'],atol=1e-9,rtol=1e-10)
                or solution['mask_count_in_optimizer']not in (0,1)):raise ValueError('Mask enters combined energy exactlyonce')
    elif not np.isclose(solution['objective'],native+ov+sv,atol=1e-9,rtol=1e-10):raise ValueError('Native control has no continuous mask term')
    return dict(applied=True,native=native,O=ov,S=sv,mask=mask,weighted_mask=weighted,
        total=float(solution['objective']),continuous_mask_gradient=continuous and bool(solution.get('continuous_mask_gradient_active')),
        mask_count_in_optimizer=solution.get('mask_count_in_optimizer',0),
        native_mask_tie_break_only=selection['mask_only_machine_numeric_regularized_energy_ties'])


def _energy_text(e):
    if not e['applied']:return '현재 N/O/S/M 에너지 비교 미적용'
    if e['native']is None:return 'fallback: 수용된 joint 비용 없음'
    return f"N {e['native']:.3f} · O {e['O']:.3f} · S {e['S']:.3f} · λM {e['weighted_mask']:.3f} = {e['total']:.3f}"


def _saved_plan(arm):
    bundle=arm['new_image_line_group_authentication'][arm['WD_family']]
    plan=bundle['runtime_complete_BI_replacement_plan']
    if arm['accepted']and EXP.E._digest(plan)!=EXP.E._digest(arm['solution'].get('complete_BI_replacement_audit')):
        raise ValueError('Exact selected-family cost plan required')
    return plan


def _replacement_panel(ax,task,plan,title):
    by_corner={r['native_corner']:r for r in plan['corner_plans']};old_partial={r['native_corner']for r in task['boundary_merged']}
    counts=dict(retained_q=0,archived_tangent=0,lines_only=0)
    for measurement in task['spec']['visible']:
        k=measurement['native_corner']
        if task['initial_self_hidden'][k]or task['states'][k]not in ('B','I'):raise ValueError('S/U forbidden before point access')
        q=np.asarray(measurement['xy']);record=by_corner.get(k)
        if record is None:
            if k not in old_partial:raise ValueError('Every complete measurement has a saved plan')
            DRAW._points(ax,{k:q},'#899099','+',prefix='부분 보관',offset=(5,-10));continue
        archive=record['original_measurement_archive']
        if not np.array_equal(q,np.asarray(archive['original_observed_xy'])):raise ValueError('Exact source q archive required')
        if not record['point_cost_revoked']:
            counts['retained_q']+=1;source=archive['original_measurement_source']
            DRAW._points(ax,{k:q},'#e53d9b','x',prefix=('fallback 유지'if source=='VISIBLE_BACKBONE_FALLBACK'else'원출처 유지'),offset=(5,-10))
        else:
            DRAW._points(ax,{k:q},'#899099','+',prefix='q 보관',offset=(5,-10))
            if 'tangent_kind'in record:
                counts['archived_tangent']+=1;parameters=record['tangent_variance_CAL'];variance=float(record['tangent_variance_px2'])
                if (not np.isclose(variance,parameters['pixel_floor_variance_px2']+parameters['gamma']*parameters['original_variance_axis_px2'],atol=1e-10,rtol=1e-10)
                        or not np.isclose(record['legacy_uncalibrated_variance_px2'],4*parameters['original_variance_axis_px2'],atol=1e-10,rtol=1e-10)):
                    raise ValueError('Stored CAL variance1+gamma*a and exact old4a required')
                tangent=np.asarray(record['tangent_unit_xy']);sigma=np.sqrt(variance);segment=q+np.array([-sigma,sigma])[:,None]*tangent
                ax.plot(segment[:,0],segment[:,1],c='#e99817',ls='--',lw=1.8)
                DRAW._label(ax,q,f'접선 σCAL {sigma:.2f}px','#aa6500',(5,13))
            else:counts['lines_only']+=1;DRAW._label(ax,q,'선-only','#008f91',(5,13))
    for group in plan['active_groups']:
        for component in group['components']:
            n=np.asarray(component['normal_unit_xy']);tangent=np.array([-n[1],n[0]]);lo,hi=component['actual_support_tangent_interval_px']
            segment=-component['offset_px']*n+np.asarray([lo,hi])[:,None]*tangent;pixels=np.asarray(component['actual_inlier_xy'])
            ax.plot(segment[:,0],segment[:,1],c='#00aeb3',lw=1.,alpha=.65);ax.scatter(pixels[:,0],pixels[:,1],s=5,c='#00aeb3',alpha=.55)
    ax.text(.02,.97,title+f"\n기존 q {counts['retained_q']} · CAL 접선 {counts['archived_tangent']} · 선-only {counts['lines_only']}\n공유 변 {len(plan['active_groups'])}개 · 새 KP 없음",transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.9))
    return counts


def _pose_panel(ax,q,arm,key,metric,energy,bounds):
    if q is None:ax.text(.02,.95,'저장 자세 없음',transform=ax.transAxes,va='top');return
    DRAW._wire(ax,q,COLORS[key],width=1.8,ls='-'if arm['accepted']else'--')  # No S/U point markers.
    text=OLD._metric_text(metric)+' · '+('joint 수용'if arm['accepted']else'fallback')+'\n'+_energy_text(energy)
    outside=np.any((q<[bounds[0],bounds[3]])|(q>[bounds[1],bounds[2]]),axis=1)
    if outside.any():text+='\n확대창 밖 모델 꼭짓점 '+str(int(outside.sum()))+'개 · 전체 좌표 JSON'
    if key==LOCKED:
        a=arm['early_family_audit'];text+='\n요청 '+str(a['requested_predicted_family'])+' / 실제 '+str(arm['WD_family'])+' · e 수치자세 미사용'
        if not a['locked_family_constraint_satisfied']:text+='\n요청 family 해 미수용'
    ax.text(.02,.97,text,transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.9))


def _summary(reports):
    rows=[]
    for part,label in (('PRIMARY192','전체192'),('TEST','기존 TEST49'),('REMAINDER','추가143'),('CAL','별도 CAL46')):
        rr=reports['TEST']['rows']+reports['REMAINDER']['rows']if part=='PRIMARY192'else reports[part]['rows']
        for key in METHODS:
            s=ACCOUNT._summary(rr,key)
            if part!='PRIMARY192':ACCOUNT._same_summary(s,reports[part]['summary'][key])
            rows.append(f'<tr><td>{label}</td><td>{NAMES[key]}</td><td>{len(rr)}</td><td>{s["translation_le5"]}</td><td>{s["successes_5cm_5deg"]}</td><td>{s["translation_cm"]["mean"]:.3f}</td><td>{s["translation_cm"]["max"]:.3f}</td><td>{s["fallback_count"]}</td></tr>')
    return '<table><tr><th>집합</th><th>고정 방법</th><th>전체</th><th>T≤5cm</th><th>5cm·5°</th><th>평균 cm</th><th>최대 cm</th><th>fallback</th></tr>'+''.join(rows)+'</table>'


def render(*,root_postscore_audit_memory_GO=False):
    if not root_postscore_audit_memory_GO:raise ValueError('Wait for explicit root score/audit/memory GO')
    if OUTPUT.exists():raise FileExistsError('Preserve completed and partial reports')
    _load_plan();protocol,new,scores,reports,auditpaths=_completed()
    tasks={r['id']:r for r in C.rows(EXP.OUTPUT/'INPUTS.jsonl.gz')if r['id']in CASES}
    observations={r['id']:r for r in C.rows(OLD.OBS.OUTPUT/'OBSERVATIONS.jsonl.gz')if r['id']in CASES}
    if set(tasks)!=set(CASES)or set(observations)!=set(CASES):raise ValueError('Four unchanged actual inputs required')
    prior_models=dict(O=C.read(EXP.OLD.OCAL.OUTPUT/'MODEL.json'),S=C.read(EXP.OLD.SCAL.OUTPUT/'MODEL.json'))
    paths=[Path(__file__),Path(EXP.__file__),Path(PREVIOUS.__file__),Path(OLD.__file__),Path(DRAW.__file__),
        Path(CONTEXT.__file__),Path(ACCOUNT.__file__),PLAN/'PREDECLARATION.json',OLD.OBS.OUTPUT/'OBSERVATIONS.jsonl.gz',OLD.OBS.OUTPUT/'OBSERVATION_SEAL.json']+auditpaths
    paths += [EXP.OUTPUT/name for name in ('PROTOCOL.json','INPUT_SEAL.json','INPUTS.jsonl.gz','PREDICTION_SEAL.json','PREDICTIONS.jsonl.gz','SCORE_SEAL.json','CAL_SCORE.json','TEST_SCORE.json','REMAINDER_SCORE.json')]
    paths += [module.OUTPUT/name for module in (EXP.TCAL,EXP.SOURCE_CAL,EXP.V.CAL,EXP.OLD.OCAL,EXP.OLD.SCAL)for name in ('MODEL.json','MODEL_SEAL.json')]
    bindings=dict(_binding(p)for p in paths);font=DRAW._font()
    if font.is_file():bindings[str(font)]=C.sha(font)
    OUTPUT.mkdir(parents=True);assets=OUTPUT/'assets';assets.mkdir();cases=[];pictures=[]
    for index,fid in enumerate(CASES):
        task,observed=tasks[fid],observations[fid]
        if (task['states']!=observed['states']or task['spec']['visible']!=observed['visible_measurements']
                or task['new_chain_observation']!=observed['chain_observation']or task['baseline']!=new[fid]['arms'][BASELINE]):
            raise ValueError('Exact unchanged observations and e required')
        original_domains=[r['original_domain']for r in task['O_localization_abstention']['records']]
        if original_domains!=[d for d in observed['domains']if d['state']=='O']:raise ValueError('Original O prior provenance differs')
        image_path,mask_path=C.ROOT/task['image'],C.ROOT/observed['mask_binding']['path']
        if C.sha(image_path)!=task['image_sha256']or C.sha(mask_path)!=observed['mask_binding']['sha256']:raise ValueError('Actual RGB/mask bytes differ')
        bindings[task['image']]=task['image_sha256'];bindings[observed['mask_binding']['path']]=observed['mask_binding']['sha256']
        bgr=cv2.imread(str(image_path))
        if bgr is None:raise ValueError('Actual image decode failed')
        image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        with np.load(mask_path,allow_pickle=False)as arrays:mask=arrays['pallet_mask_clean'].copy()
        geometry=observed['initial_geometry'];K=geometry['K'];_,_,prepared_S=EXP.OLD._prior_inputs(task,prior_models)
        eligible_S=[r['native_corner']for r in prepared_S['eligible_initial_S']]
        arms={key:new[fid]['arms'][key]for key in METHODS};projected={key:OLD._model(arm,K)for key,arm in arms.items()}
        metrics={key:scores[fid]['metrics'][key]for key in METHODS};energies={key:_energies(arms[key],key)for key in METHODS}
        plans={key:_saved_plan(arms[key])for key in METHODS[1:]};active_bundle=dict(groups=plans[MASK_FIT]['active_groups'])
        support=[np.asarray(geometry['initial_projection_xy']),np.asarray([m['xy']for m in task['spec']['visible']]).reshape(-1,2),np.asarray([d['center_xy']for d in original_domains]).reshape(-1,2)]
        xy=np.vstack(support);h,w=image.shape[:2];lo,hi=np.maximum(xy.min(0)-35,0),np.minimum(xy.max(0)+35,[w,h]);bounds=tuple(map(float,(lo[0],hi[0],hi[1],lo[1])))
        case=dict(id=fid,partition=task['partition'],image=task['image'],image_sha256=task['image_sha256'],states=task['states'],
            initial_geometry=geometry,frozen_BI=task['spec']['visible'],original_O_domains=original_domains,hard_O_T_domains=task['spec']['domains'],
            U_regions=task['u_regions'],partial_boundary=task['boundary_merged'],actual_NMS_observation=task['new_chain_observation'],mask_binding=observed['mask_binding'],
            eligible_initial_S_ids=eligible_S,S_U_point_markers=False,crop_bounds=bounds,crop_not_based_on_reference_or_final_pose=True,
            methods={key:dict(actual_pose=arm['actual_pose'],accepted=arm['accepted'],WD_family=arm.get('WD_family'),
                model_projection_xy=None if projected[key]is None else projected[key].tolist(),metric=metrics[key],energy_ledger=energies[key],
                early_family_audit=arm.get('early_family_audit'),fallback_audit=arm.get('fallback_audit'),
                runtime_complete_BI_replacement_plan=plans.get(key),
                plan_is_accepted_joint_solution=arm['accepted']if key!=BASELINE else None)for key,arm in arms.items()},stages=[])
        for stage,title in enumerate(TITLES):
            if stage==4:
                fig,axes=plt.subplots(1,2,figsize=(15,6))
                for ax,key in zip(axes,(ALL_NATIVE,SOURCE_NATIVE)):
                    ax.imshow(image);ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.set_aspect('equal');ax.set_axis_off()
                    counts=_replacement_panel(ax,task,plans[key],NAMES[key]+' / '+str(arms[key]['WD_family'])+(' / 미수용 계획'if not arms[key]['accepted']else''))
                    case['methods'][key]['displayed_plan_counts']=counts
            else:
                fig,ax=DRAW._base(image,title);ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:])
                if stage==0:DRAW._wire(ax,np.asarray(geometry['initial_projection_xy']),'#7e8790',ls='--');OLD._frozen_points(ax,task)
                elif stage==1:
                    ax.imshow(np.dstack([np.zeros_like(mask),mask,np.zeros_like(mask),mask*.15]));case['NMS_display_audit']=OLD._nms(ax,task['new_chain_observation'],active_bundle)
                elif stage==2:
                    if original_domains:CONTEXT._new_canvas(ax,image,task['spec']['domains'],bounds)
                    CONTEXT._context(ax,task)
                elif stage==3:
                    CONTEXT._old_o(ax,original_domains)
                    ax.text(.02,.97,'초기 S 모델 prior 대상: '+str(eligible_S)+'\nS/U 점 표시는 없음 · S 영상 관측 없음',transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.9))
                else:
                    key=METHODS[stage-5];_pose_panel(ax,projected[key],arms[key],key,metrics[key],energies[key],bounds)
            path=assets/f'case{index}_stage{stage+1}.png';DRAW._save(fig,path,CAPTIONS[stage]);case['stages'].append(dict(title=title,caption=CAPTIONS[stage],asset=str(path.relative_to(OUTPUT))))
        cases.append(case);pictures.append((image,bounds,projected,metrics,energies))
    fig,axes=plt.subplots(len(METHODS),len(CASES),figsize=(17,18),squeeze=False)
    for col,(image,bounds,projected,metrics,energies)in enumerate(pictures):
        for row,key in enumerate(METHODS):
            ax=axes[row,col];ax.imshow(image)
            if projected[key]is not None:DRAW._wire(ax,projected[key],COLORS[key],width=1.4)
            ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.set_aspect('equal');ax.set_axis_off()
            ax.set_title((CASES[col]+'\n'if row==0 else'')+NAMES[key]+'\n'+OLD._metric_text(metrics[key])+'\n'+_energy_text(energies[key]),fontsize=7)
    fig.suptitle('실제 사진 · 고정 네 사례 / 동일 다섯 방법\n저장 모델 투영만 · CAL 접선/원출처/연속 마스크/family 고정 구분 · GT/S/U 점 표시 없음',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUTPUT/'overview_actual.png',dpi=160,bbox_inches='tight');plt.close(fig)
    C.write(OUTPUT/'PROCESS_DATA.json',dict(schema='SOURCE_PRESERVING_TCAL_MASK_ACTUAL_PROCESS_V24',predeclared_cases=list(CASES),
        uniform_methods=list(METHODS),cases=cases,source_bindings=bindings,S_U_input_or_projection_point_markers=False,
        no_GT_wire_or_reference_pose_corner_reads=True,new_inference_or_fit=0,coarse_priors_not_independent_image_evidence=True,default_e_unchanged=True))
    sections=[]
    for case in cases:
        rows=''.join('<tr><td>'+NAMES[key]+'</td><td>'+OLD._metric_text(case['methods'][key]['metric'])+'</td><td>'+html.escape(str(case['methods'][key]['WD_family']))+'</td><td>'+('joint 수용'if case['methods'][key]['accepted']else'fallback')+'</td><td>'+_energy_text(case['methods'][key]['energy_ledger'])+'</td></tr>'for key in METHODS)
        stages=''.join('<h3>'+s['title']+'</h3><img src="'+s['asset']+'" alt="실제 저장 처리 단계"><p>'+s['caption']+'</p>'for s in case['stages'])
        sections.append('<details open><summary>'+html.escape(case['id'])+'</summary><p>고정 상태: '+' '.join(case['states'])+'</p><div class="scroll"><table><tr><th>고정 방법</th><th>명목 참조 오차</th><th>W/D</th><th>수용</th><th>분리된 비용</th></tr>'+rows+'</table></div><p>ALL/SOURCE 패널은 각 저장 family의 비용 계획입니다. fallback 계획은 joint 해로 채택되지 않았습니다. 모든 수치·출처·family 제한과 실패 기록은 JSON에 보존합니다.</p>'+stages+'</details>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>원출처·CAL 접선·연속 마스크 과정</title><style>body{background:#edf2f6;color:#173044;font-family:system-ui,sans-serif;margin:0}main{max-width:1160px;margin:auto;padding:18px}img{max-width:100%;height:auto;display:block}details{background:white;margin:18px 0;padding:16px;border-radius:10px}summary{font-weight:bold}p{line-height:1.6}.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border:1px solid #cbd6df;text-align:left}@media(max-width:500px){main{padding:8px}details{padding:9px}}</style><main><h1>원출처 보존 · CAL 접선 · 연속 마스크: 실제 저장 과정</h1><p>기존23의 네 사례를 새24 점수 전에 그대로 고정했습니다. 모든 사진에서 e, 전체 대체+CAL 접선, 원출처 보존+CAL 접선, 연속 마스크, e family 고정을 같은 순서로 비교합니다. 사례별 최적 방법을 섞지 않습니다.</p><p>원출처 보존은 정확히 VISIBLE_BACKBONE_FALLBACK인 complete q만 대체하며 RGB/unknown complete q·Σ·신뢰도는 유지합니다. 한 방향 접선 분산은 재사용 CAL30에서 학습한 1+γ(tᵀΣt)이며 원래 ORIGINAL guide를 사용합니다. native 선택기는 기존처럼 수치상 동률에서만 마스크를 참고합니다. 기존 부분관측 T4는 바꾸지 않습니다. fallback 자료는 CAL에2점뿐이라 별도 검증이나 향후 포함률을 주장하지 않습니다. 새 정밀 KP는 만들지 않습니다.</p><p>모든 새 방법은 같은 bounded O/S prior를 사용합니다. S는 영상·2D·범위 관측이 아니며 U는 범위만 사용합니다. S/U 점 표시는 하지 않고 모델 wire만 표시합니다. 연속 마스크 두 방법만 최적화와 선택에 λM을 한 번 더합니다. family 고정은 e의 예측 이름만 인증 전에 제한하며 e의 수치 자세를 복사하지 않습니다. 실패는 own coarse fallback으로 남기고 성공으로 바꾸지 않습니다.</p><p>추가143의 참조는 annotation/PnP 유도값으로 독립 계측이 아니며, 반복 사용한 자료의 탐색 결과입니다. 모든238·실패·fallback·native 플래그를 유지하고 CAL46은 primary192와 분리합니다. 기존 e 기본값은 유지합니다.</p><div class="scroll">'''+_summary(reports)+'''</div><img src="overview_actual.png" alt="같은 네 사례 다섯 고정 방법"><p><a href="PROCESS_DATA.json">전체 표시값과 출처</a> · <a href="'''+_url(PREVIOUS.OUTPUT/'index.html')+'''">이전23 실제 과정</a></p>'''+''.join(sections)+'''</main></html>'''
    (OUTPUT/'index.html').write_text(page,encoding='utf-8')
    for relative,digest in bindings.items():
        path=Path(relative)if Path(relative).is_absolute()else C.ROOT/relative
        if C.sha(path)!=digest:raise ValueError('Bound source/evidence changed while rendering')
    C.write(OUTPUT/'RENDER_SEAL.json',dict(source_sha256=C.sha(__file__),input_sha256=bindings,
        assets={str(p.relative_to(OUTPUT)):C.sha(p)for p in OUTPUT.rglob('*')if p.is_file()},cases=4,stages_per_case=10,
        predeclared_case_order=list(CASES),uniform_methods=list(METHODS),actual_RGB_only=True,
        no_GT_wire_or_S_U_point_markers=True,new_inference_or_fit=0))
    print(dict(output=str(OUTPUT),cases=4,methods=5,stages=10),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','render'))
    parser.add_argument('--after-root-score-audit-memory-go',action='store_true');args=parser.parse_args()
    if args.command=='prepare':prepare_plan()
    else:render(root_postscore_audit_memory_GO=args.after_root_score_audit_memory_go)
