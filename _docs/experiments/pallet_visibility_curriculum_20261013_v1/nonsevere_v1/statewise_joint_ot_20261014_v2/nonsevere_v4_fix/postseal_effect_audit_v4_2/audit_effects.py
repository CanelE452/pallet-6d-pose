"""Fixed postseal interval effect comparisons. No GT lookup, RGB or fitting."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;F=HERE.parent
legacy=F/'postseal_effect_audit_v4_1/audit_effects.py'
spec=importlib.util.spec_from_file_location('preserved_effect_helpers',legacy)
L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
FINAL=F/'evaluation_interval_v4_2';BIAS=F/'evaluation_bias_v4_1';BASE=F/'evaluation_v1';MEAN=F/'coarse_mean_control_v4_1'
REFS=(*L.REFERENCES[:4],'BASE_V4_ZERO_MEAN','FROZEN_BIAS_V4_1','NO_OT','NO_PRIOR','COARSE_MEAN_ONLY')


def prepare():
    plan={'schema':'FIXED_POSTSEAL_INTERVAL_EFFECT_PLAN_V4_2','source':L.binding(__file__),
        'shared_preserved_statistical_helpers':L.binding(legacy),'final_protocol':L.binding(FINAL/'PROTOCOL.json'),
        'interval_MODEL':L.binding(F/'truncation_interval_model_CAL_v4_2/MODEL.json'),
        'interval_MODEL_SEAL':L.binding(F/'truncation_interval_model_CAL_v4_2/MODEL_SEAL.json'),
        'references':list(REFS),'partition':{'CAL':46,'TEST':49,'severe':0,'separate_crops':3},
        'metrics':list(L.METRICS),'cluster_method':'Exact ordered4-session resampling,256draws,frameweighted,2.5/97.5linearquantiles;exploratory',
        'metric_read_gate':'Complete98/all3arms prediction seal AND POSTSEAL_SCORING before any newmetric rows read',
        'attribution':'OldNO_PRIOR→basezero→oldbias→intervalfinal isolates cumulative changes in declared order;newNO_OT isolates remainingOT',
        'fixed_cases':list(L.FIXED_CASES),'additional_cases':'everyoriginalsuccess→finalfailure;allT→U/retainedT/OTposechanges kept',
        'no_new_parameter_or_hypothesis_selection':True,'labels_or_RGB_or_fits':False,
        'limitations':['CALfullmodel scores are not conditionalLOGO selection','TEST reused exploratory repair after olderTESTdiagnosis',
            'Fourcluster intervals do not prove general superiority','Humanstates are historical annotation diagnostic labels',
            'Tinterval is an empiricalCALmaximum, not independentphysical or futurecoveragecertificate',
            'BI2acceptedO-supportedrepresentatives remain ambiguousfamilies','All95fallback/nativeincompatible rows retained']}
    L.write(HERE/'AUDIT_PLAN.json',plan)
    L.write(HERE/'PREPARATION_SEAL.json',{'source':L.binding(__file__),'plan':L.binding(HERE/'AUDIT_PLAN.json'),'new_TEST_metrics_read':False})


def prepared():
    p=L.read(HERE/'PREPARATION_SEAL.json');L.verify_binding(p['source']);L.verify_binding(p['plan'])
    plan=L.read(HERE/'AUDIT_PLAN.json')
    for k in ['source','shared_preserved_statistical_helpers','final_protocol','interval_MODEL','interval_MODEL_SEAL']:L.verify_binding(plan[k])
    return plan


def run():
    plan=prepared()
    if not all((FINAL/n).is_file() for n in ['PREDICTION_SEAL.json','POSTSEAL_SCORING.json']):raise RuntimeError('Wait completefinalseal+POST')
    inputs={str((d/n).relative_to(L.ROOT)):L.sha(d/n) for d in [FINAL,BIAS,BASE,MEAN] for n in ['PROTOCOL.json','PREDICTION_SEAL.json','PREDICTIONS.jsonl.gz','POSTSEAL_SCORING.json']}
    fp,fs,proto=L.load_sealed(FINAL);bp,bs,_=L.load_sealed(BIAS);zp,zs,_=L.load_sealed(BASE);mp,ms,_=L.load_sealed(MEAN,True)
    Fp,Fm,Bp,Bm,Zm,Mm=({r['id']:r for r in rows} for rows in [fp,fs['rows'],bp,bs['rows'],zs['rows'],ms['rows']])
    if not all(set(Fp)==set(x) for x in [Fm,Bp,Bm,Zm,Mm]):raise ValueError('98populationmismatch')
    for p in ['CAL','TEST']:
        if fs['label_bindings'][p]!=bs['label_bindings'][p] or fs['label_bindings'][p]!=zs['label_bindings'][p] or fs['label_bindings'][p]['sha256']!=ms['label_sha256'][p]:raise ValueError('FixedC2labelbindingsdiffer')
    rows=[]
    for fid,p in Fp.items():
        s=Fm[fid];metrics={'FINAL':s['metrics'][L.ALL],**{a:s['metrics'][a] for a in L.REFERENCES[:4] if a in s['metrics']},
            'BASE_V4_ZERO_MEAN':Zm[fid]['metrics'][L.ALL],'FROZEN_BIAS_V4_1':Bm[fid]['metrics'][L.ALL],
            'OLD_BIAS_NO_PRIOR':Bm[fid]['metrics'][L.NO_PRIOR],'NO_OT':s['metrics'][L.NO_OT],
            'NO_PRIOR':s['metrics'][L.NO_PRIOR],'COARSE_MEAN_ONLY':Mm[fid]['metrics']}
        d=L.runtime_record(p,s,Bp[fid]);a=p['runtime'];b=Bp[fid]['runtime'];guard=a['truncation_interval_guard']
        if a['frozen_visible_measurements']!=b['frozen_visible_measurements'] or a['calibrated_visible_measurements']!=b['calibrated_visible_measurements']:raise ValueError('BIfreezeparityfailed')
        if a['ot_domains']['nonself_prediction_scale_px']!=b['ot_domains']['nonself_prediction_scale_px']:raise ValueError('Scalefreezeparityfailed')
        if any(x!=y if x!='T' else y not in ['T','U'] for x,y in zip(b['states'],a['states'])):raise ValueError('NonTroleschanged')
        oldO={q['native_corner']:q for q in b['ot_domains']['domains'] if q['state']=='O'};newO={q['native_corner']:q for q in a['ot_domains']['domains'] if q['state']=='O'}
        if set(oldO)!=set(newO) or any(oldO[k].get(n)!=newO[k].get(n) for k in oldO for n in ['shape','center_xy','half_axes_px','angle_deg','bounds_xyxy','halfplanes']):raise ValueError('Ogeometrychanged')
        d.update(T_interval_guard=guard,truncation_model_binding=a['truncation_model_binding'],
            frozen_BI_effective_cov_O_numeric_and_nonS_scale_exact_vs_old_bias=True,
            only_T_to_U_roles_changed_vs_old_bias=True)
        rows.append({k:p[k] for k in ['id','source_record_id','partition','session','cohort']}|{'metrics':metrics,'diagnostics':d})
    primary=[r for r in rows if r['cohort']=='PRIMARY_NONSEVERE'];crops=[r for r in rows if r['cohort']!='PRIMARY_NONSEVERE']
    if len(primary)!=95 or len(crops)!=3:raise ValueError('95+3populationchanged')
    summary={};paired={};components={};pop={}
    for part,n in [('CAL',46),('TEST',49)]:
        group=[r for r in primary if r['partition']==part]
        if len(group)!=n:raise ValueError('Partitionchanged')
        summary[part]={a:{'frames':n,'success_count':sum(r['metrics'][a]['success_5cm_5deg'] for r in group),
            'metrics':{m:L.stats([r['metrics'][a][m] for r in group]) for m in L.METRICS[:3]}} for a in ['FINAL',*REFS]}
        paired[part]={a:L.paired(group,a) for a in REFS}
        components[part]={'compound_soft_BI_state_search_old_OT_vs_original':L.paired(group,'ORIGINAL_N3_PNP','OLD_BIAS_NO_PRIOR'),
            'zero_mean_prior_given_original_T_roles':L.paired(group,'OLD_BIAS_NO_PRIOR','BASE_V4_ZERO_MEAN'),
            'CAL_median_bias_given_original_T_roles':L.paired(group,'BASE_V4_ZERO_MEAN','FROZEN_BIAS_V4_1'),
            'interval_T_admission_given_frozen_BI_cov_prior_bias_O':L.paired(group,'FROZEN_BIAS_V4_1','FINAL'),
            'remaining_OT_vs_NO_OT':L.paired(group,'NO_OT','FINAL'),
            'current_prior_and_bias_given_interval_roles':L.paired(group,'NO_PRIOR','FINAL'),
            'statewise_vs_mean_only':L.paired(group,'COARSE_MEAN_ONLY','FINAL')}
        pop[part]={'frames':n,'fallback_ids':[r['id'] for r in group if r['diagnostics']['actual_pose_fallback']],
            'native_incompatible_ids':[r['id'] for r in group if not r['diagnostics']['physical_native_compatible']],
            'ambiguous_family_ids':[r['id'] for r in group if r['diagnostics']['material_ambiguity']],
            'demoted_T_count':sum(len(r['diagnostics']['T_interval_guard']['demoted_T_to_U_ids']) for r in group),
            'remaining_T_count':sum(len(r['diagnostics']['T_interval_guard']['retained_T_ids']) for r in group),
            'BI_ge3_OT_pose_changed':sum(r['diagnostics']['BI_count']>=3 and r['diagnostics']['OT_changes_pose_vs_no_OT'] for r in group),
            'BI2_OT_pose_changed':sum(r['diagnostics']['BI_count']==2 and r['diagnostics']['OT_changes_pose_vs_no_OT'] for r in group),
            'active_OT_inequality_categories':{cat:sum(d['support_category']==cat and d['selected_constraint_active'] for r in group for d in r['diagnostics']['OT_domains']) for cat in ['SOURCE_EXTERNAL_OCCLUSION_O','INITIAL_GEOMETRY_CONDITIONAL_T','INDEPENDENTLY_CERTIFIED_T']}}
    regress=[r['id'] for r in primary if r['metrics']['ORIGINAL_N3_PNP']['success_5cm_5deg'] and not r['metrics']['FINAL']['success_5cm_5deg']]
    selected=list(dict.fromkeys([*L.FIXED_CASES,*regress]));lookup={r['id']:r for r in rows}
    detail={'rows':rows,'fixed_actual_photo_and_original_success_regression_cases':[lookup[fid] for fid in selected],
        'original_success_to_final_failure_ids':regress,'all_OT_pose_change_cases':[r for r in rows if r['diagnostics']['OT_changes_pose_vs_no_OT']],
        'all_demoted_or_retained_T_cases':[r for r in rows if r['diagnostics']['T_interval_guard']['original_T_ids']]}
    report={'schema':'POSTSEAL_FIXED_INTERVAL_EFFECT_AUDIT_V4_2','summary':summary,'paired':paired,'components':components,
        'population':pop,'target_checks':fs['target_checks'],'target_achieved':fs['target_achieved'],
        'whole_goal_complete_claim_allowed':bool(fs['target_achieved']),
        'all95primary_fallback_native_mismatch_kept':True,'crops_excluded_from_primary':3,
        'no_parameter_or_reference_selection':True,'no_GT_lookup_RGB_or_fitting':True,
        'original_success_to_final_failure_ids':regress,'limitations':plan['limitations'],
        'remaining_OT_activity_is_solver_slack_diagnostic_not_physical_confidence':True}
    prepared();L.verify_map(inputs)
    L.write(HERE/'EFFECT_AUDIT.json',report);L.write(HERE/'FRAME_DETAILS.json',detail)
    lines=['# V4.2 동일 프레임 봉인 후 효과 감사','','모든 비교는 같은 TEST49·물리 C2 기준입니다. 전체95의 fallback과 native 불일치를 유지하고 crop3은 별도 기록했습니다.','','| 방법 | 성공/49 | 평균 T(cm) | 평균 R(deg) |','|---|---:|---:|---:|']
    for a,v in summary['TEST'].items():lines.append(f'| {a} | {v["success_count"]} | {v["metrics"]["translation_cm"]["mean"]:.6f} | {v["metrics"]["rotation_deg"]["mean"]:.6f} |')
    lines+=['','고정 숫자 목표 충족: **'+str(fs['target_achieved'])+'**. 조건부 T를 U로 보류한 효과와 남은 O/T의 효과를 구분했습니다. 원래 B/I·유효 공분산·nonS scale·O 숫자 영역·λ1β1은 그대로입니다.','','재사용 TEST의 후속 탐색적 수정이며 새 일반화나 통계적 우월성을 증명하지 않습니다. CAL full모델 점수는 세션 제외 LOGO 선택 점수와 다릅니다. 원과 캔버스 교차는 CAL 최대 오차에 따른 보류이며, 미래 포함률이나 물리 T 인증을 뜻하지 않습니다.','','모든 세션별 paired 차이와 256개 exact 세션 복원추출 구간, prior·bias·interval 기여는 EFFECT_AUDIT.json에 있습니다. 실제 사진 및 실패/변화 사례는 FRAME_DETAILS.json에 있습니다.']
    (HERE/'AUDIT_KO.md').write_text('\n'.join(lines)+'\n')
    prepared();L.verify_map(inputs)
    L.write(HERE/'COMPLETION_RECEIPT.json',{'source':L.binding(__file__),'plan':L.binding(HERE/'AUDIT_PLAN.json'),'input_sha256':inputs,
        'outputs':[L.binding(HERE/n) for n in ['EFFECT_AUDIT.json','FRAME_DETAILS.json','AUDIT_KO.md']],
        'new_fits':0,'GT_coordinate_or_pose_lookup':False,'RGB_reads':0,'TEST_parameter_changes':0,'target_achieved':fs['target_achieved']})
    print(json.dumps({'TEST':summary['TEST'],'target':fs['target_checks'],'target_achieved':fs['target_achieved'],'population':pop},ensure_ascii=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run']);a=p.parse_args();{'prepare':prepare,'run':run}[a.phase]()
