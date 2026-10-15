"""Three fixed postseal coarse/selected-pose scalar probes; no pose fitting.

Only the reviewed independent audit helper/native equations, old guard
primitives and frozen O/S/C kernels are used. No V32 consumer or planner,
fresh seed generator, reference, score numbers, RGB or local query is read.
"""
import argparse
from collections import Counter
from copy import copy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np

from . import audit_clean_fresh_BI_poses_v4_32 as A

C=A.C
INPUT=A.OUTPUT
OUTPUT=A.B/'clean_fresh_BI_initial_guard_cost_probe_v4_32_v1'
IDS=('wood_night_01:031344','wood_night_01:031426','eval_noapril:1775201428535961600')
EXPECTED_PRED='eff80efdb1cb9c17f7eb13df50b953467f82bbe3415c8385b9aa146456948b48'
EXPECTED_SCORE='7bcc9bce3e8b14a87021955c513e3061dcb561cc309fabd61964e06efbcc6c80'
EXPECTED_AUDIT='fbe2f120dab0fc361701c9fe2f1166bb8683ba17d798a84f072837764fa78013'


def binding(path):return dict(path=str(Path(path).resolve().relative_to(C.ROOT)),sha256=C.sha(path))


def declare(*,root_probe_GO=False):
    if root_probe_GO is not True:raise PermissionError('ROOT fixed three-case probe GO before IO')
    if OUTPUT.exists():raise FileExistsError('Append-only three-case probe')
    files=('PROTOCOL.json','INPUT_SEAL.json','INPUTS.jsonl.gz','PREDICTIONS.jsonl.gz',
        'PREDICTION_SEAL.json','SCORE_SEAL.json','independent_root_audit_v1/AUDIT.json',
        'independent_root_audit_v1/AUDIT_SEAL.json')
    if (C.sha(INPUT/'PREDICTION_SEAL.json')!=EXPECTED_PRED
            or C.sha(INPUT/'SCORE_SEAL.json')!=EXPECTED_SCORE
            or C.sha(INPUT/'independent_root_audit_v1/AUDIT.json')!=EXPECTED_AUDIT
            or C.sha(A.HELPER)!=A.HELPER_SHA256):raise ValueError('Fixed current seals/helper bytes required')
    pred=C.read(INPUT/'PREDICTION_SEAL.json');seal=C.read(INPUT/'INPUT_SEAL.json')
    if pred['rows']!=238 or pred['predictions_sha256']!=C.sha(INPUT/'PREDICTIONS.jsonl.gz')or seal['input_sha256']!=C.sha(INPUT/'INPUTS.jsonl.gz'):
        raise ValueError('Exact complete sealed238 positive inputs/predictions required')
    sources=A._source_map();sources[str(Path(__file__).resolve().relative_to(C.ROOT))]=C.sha(__file__)
    plan=dict(schema='FIXED_THREE_CASE_COARSE_SCALAR_PROBE_V32_V1',UTC=datetime.now(timezone.utc).isoformat(),
        ids=list(IDS),source_sha256=sources,input_sha256={str((INPUT/name).relative_to(C.ROOT)):C.sha(INPUT/name)for name in files},
        scope='Postseal user/root-selected diagnostics, not prospective performance or runtime selection.',
        measured_poses='Own independent ORIGINAL coarse, same-body transported SWAPPED coarse, and each saved native/C selected pose.',
        scalar_basis='Reviewed helper6d18 positive task/family binding plus independent V29 native energy, frozen original guards and old O/S/C pure kernels.',
        all_family_policy_plans_from_saved_predictions=True,new_G32_consumer_or_planner_calls=0,
        fresh_PnP_seed_optimizer_NN_query_reference_or_GT_calls=0,
        score_file_numeric_decodes=0,old_and_new_runtime_sources_unchanged=True)
    OUTPUT.mkdir(parents=True);C.write(OUTPUT/'PLAN.json',plan)
    for relative,digest in sources.items():
        target=OUTPUT/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((C.ROOT/relative).read_bytes())
        if C.sha(target)!=digest:raise ValueError('Source snapshot changed')
    C.write(OUTPUT/'PLAN_SEAL.json',dict(plan_sha256=C.sha(OUTPUT/'PLAN.json')))
    return binding(OUTPUT/'PLAN.json')


def exact_rows(path,key):
    found={}
    with gzip.open(path,'rt',encoding='utf-8')as stream:
        for text in stream:
            if not any(identifier in text for identifier in IDS):continue
            value=json.loads(text);identifier=value[key]['id']if key else value['id']
            if identifier not in IDS:continue
            if identifier in found:raise ValueError('Duplicate fixed diagnostic case')
            found[identifier]=value
    if set(found)!=set(IDS):raise ValueError('Every fixed case must remain, including rejected coarse poses')
    return found


def guard_detail(fresh,raw,plan,R,t):
    guard=copy(raw);guard.boundary=[record for record in raw.boundary if record['native_corner']not in plan['new_owner_ids']]
    p=np.r_[A.REPLAY.Rotation.from_matrix(R).as_rotvec(),np.asarray(t)/raw.scale]
    branches=[]
    for index,sides in enumerate(guard.sides):
        slack=guard.constraints(p,sides)[0];finite=bool(np.isfinite(slack).all())
        branches.append(dict(index=index,sides=sides,finite=finite,slacks=slack,
            minimum_slack=float(np.min(slack))if finite else None,
            failed_indices=np.flatnonzero(slack < -raw.cfg['feasibility_tolerance']).tolist()if finite else None))
    original=fresh['old_guard_audit'];support=fresh['actual_support']
    best=max((record for record in branches if record['finite']),key=lambda record:record['minimum_slack'],default=None)
    causes=[]
    if not original['accepted']:causes.append('ORIGINAL_HARD_GUARDS')
    if fresh['minimum_actual_support_slack'] < -raw.cfg['feasibility_tolerance']:causes.append('ACTUAL_FINITE_OR_DIRECTED_EDGE_SUPPORT')
    scalar={key:original.get(key)for key in ('visible_protection_audit','joint_constraint_audit','U_region_constraint_audit','boundary_observation_audit')}
    return dict(accepted=bool(fresh['accepted']),original_guard_accepted=bool(original['accepted']),
        minimum_joint_slack=original.get('minimum_joint_slack'),best_branch_index=None if best is None else best['index'],
        all_original_branches=branches,scalar_guard_audits=scalar,actual_support=support,
        minimum_actual_support_slack=fresh['minimum_actual_support_slack']if support else None,
        no_active_support_constraint=not support,minimum_depth=fresh['minimum_depth'],
        feasibility_tolerance=raw.cfg['feasibility_tolerance'],failure_causes=causes,
        guard_legacy_objective_not_reported_as_current_native_energy=True)


def evaluate(packet,family,plan,R,t,raw):
    task=A.REPLAY.positive_task(packet)
    A.REPLAY.check_initial_physical_body(packet,family)
    if (plan['observation_family']!=A.REPLAY.FAMILY or plan['population']!=A.REPLAY.POPULATION
            or plan['selection_profile']!=A.REPLAY.PROFILE):raise ValueError('Same frozen current plan required')
    policy=plan['source_policy'];contract=packet['source_contract'];reg=packet['regularizers']
    gamma=contract['tangent_parameters'][policy]['gamma']if contract['model_availability'][policy]else 0.
    fresh=A.INDEPENDENT.recompute_native(task,family,R,t,plan,gamma,raw=raw)
    failed=[reason for passed,reason in fresh['checks']if not passed]
    if failed:raise ValueError('Independent factor/source typing failed: '+repr(failed))
    p=np.r_[A.REPLAY.Rotation.from_matrix(R).as_rotvec(),np.asarray(t)/raw.scale]
    ov,_,oa=A.O.bounded_o_localization_value_and_gradient(p,raw.xyz,raw.K,raw.scale,reg['o_regions'],**reg['o_parameters'])
    if reg['original_S_prepared']is None:sv=0.;sa=dict(S_image_or_2D_terms=0)
    else:
        rebound=A.S.rebind_bounded_visibility_trial_geometry(reg['original_S_prepared'],raw.xyz)
        sv,_,sa=A.S.coarse_visibility_prior_value_and_gradient(p,raw.xyz,raw.scale,rebound)
    cfg=A.CONFIG._outline_config(reg['outline_configuration'])
    weight=float(cfg['outline_factor_weight'])if reg['outline_observation'].get('supported')else 0.
    cv=0.;ca=dict(valid_projection=True,enabled=False)
    if weight>0:cv,_,ca=A.CONTAINMENT.outline_value_and_gradient(p,raw.xyz,raw.K,raw.scale,reg['outline_observation'],cfg)
    assert oa['O_point_terms']==0 and sa['S_image_or_2D_terms']==0
    guards=guard_detail(fresh,raw,plan,R,t)
    by_kind=Counter()
    for record in fresh['native_costs']:by_kind[record['kind']]+=record['weighted_robust_cost']
    native=float(fresh['objective']);noC=native+float(ov)+float(sv);withC=noC+weight*float(cv)
    return dict(native=native,O=float(ov),S=float(sv),C=float(cv),C_weight=weight,weighted_C=weight*float(cv),
        total_native_arm=noC,total_containment_arm=withC,native_by_kind=dict(by_kind),native_costs=fresh['native_costs'],
        guards=guards,native_feasible=bool(fresh['accepted']),containment_feasible=bool(fresh['accepted']and ca['valid_projection']),
        O_audit=oa,S_audit=sa,C_audit=ca,active_owner_ids=fresh['owner_ids'],
        native_consumer_or_planner_called=False,factor_typing_checks=len(fresh['checks']))


def comparison(coarse,selected,*,containment):
    key='total_containment_arm'if containment else'total_native_arm'
    feasible=coarse['containment_feasible']if containment else coarse['native_feasible']
    gap=coarse[key]-selected[key];tolerance=1e-8*max(1.,abs(coarse[key]),abs(selected[key]))
    reason=('REJECTED_CURRENT_GUARDS'if not feasible else'SELECTED_HAS_LOWER_CURRENT_OBJECTIVE'if gap>tolerance
            else'COARSE_HAS_LOWER_CURRENT_OBJECTIVE_REQUIRES_RETENTION_OR_SEARCH_CHECK'if gap < -tolerance else'NUMERIC_TIE')
    return dict(feasible=feasible,coarse_minus_selected_total=gap,interpretation=reason,
        no_global_optimum_or_neighborhood_claim=True)


def report_md(records):
    out=['# 고정 3사례: 초기 자세의 제약·비용 진단','',
        '봉인된 양성 입력과 저장된 네 방법의 계획을 사용했습니다. 새 PnP·시작점·최적화·영상 질의·정답/점수 숫자 조회는 없습니다. ORIGINAL/SWAPPED coarse는 같은 물리 몸체와 중심을 서로 다른 native 대응으로 표현합니다.','']
    f=lambda value:'미정'if value is None else f'{value:.6g}'
    for item in records:
        out.extend(['## '+item['id'],'','| 정책/자세 | WD | guard 통과 | N | O | S | 가중 C | N+O+S | N+O+S+C | 최소 원 guard slack |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|---:|'])
        for policy,record in item['policies'].items():
            label='ALL'if policy=='ALL_BI_REPLACE'else'SOURCE'
            for name,pose in record['poses'].items():
                q=pose['computed'];out.append('| '+' | '.join((label+'/'+name,pose['family'],str(q['native_feasible']),
                    f(q['native']),f(q['O']),f(q['S']),f(q['weighted_C']),f(q['total_native_arm']),f(q['total_containment_arm']),f(q['guards']['minimum_joint_slack'])))+' |')
            for name,value in record['coarse_vs_selected'].items():
                out.append(f'- {label} {name}: {value["interpretation"]}, 초기−선택 비용 {value["coarse_minus_selected_total"]:.6g}.')
            cross=record['native_endpoint_vs_C_selected']
            out.append(f'- {label} native 최종 자세를 C 비용으로 다시 평가한 차이(native 최종−C 최종): {cross["C_total_native_pose_minus_C_selected"]:.6g}.')
        out.append('')
    out.extend(['위 비용 차이는 현재 관측·가정·제약의 선호를 설명합니다. 참조 오차의 원인이나 전역 최적해, 올바른 모서리 정체를 증명하지 않습니다. 원래 guard의 개별 branch와 실제 선의 finite/directed 지지 slack은 JSON에 보존했습니다.',''])
    return '\n'.join(out)


def run(*,root_probe_GO=False,expected_plan_sha256=None):
    if root_probe_GO is not True:raise PermissionError('ROOT fixed pose scalar probe GO before IO')
    if C.sha(OUTPUT/'PLAN.json')!=expected_plan_sha256 or C.read(OUTPUT/'PLAN_SEAL.json')['plan_sha256']!=expected_plan_sha256:raise ValueError('Exact premeasurement plan required')
    if (OUTPUT/'RESULT.json').exists():raise FileExistsError('Preserve every probe attempt')
    plan=C.read(OUTPUT/'PLAN.json');A._verify(plan['source_sha256'],plan['input_sha256'])
    protocol=C.read(INPUT/'PROTOCOL.json');metadata={record['id']:record for record in protocol['records']}
    packets=exact_rows(INPUT/'INPUTS.jsonl.gz','identifiers');rows=exact_rows(INPUT/'PREDICTIONS.jsonl.gz',None)
    records=[];checks=0;max_difference=0.
    try:
        for identifier in IDS:
            packet,row=packets[identifier],rows[identifier]
            if A.digest(packet)!=metadata[identifier]['positive_packet_sha256']:raise ValueError('Exact canonical positive packet required')
            if packet['source_contract']['ready']is not True:raise ValueError('Fixed case retained but lacks valid initial coarse')
            families=A.geometry_families(packet);task=A.REPLAY.positive_task(packet)
            raw={name:A.INDEPENDENT.prepare_raw(task,family)for name,family in families.items()}
            policies={}
            for policy,arms in A.POLICY_ARMS.items():
                native_entry,C_entry=(row['arms'][arm]for arm in arms)
                for name in A.FAMILIES:
                    if A.digest(native_entry['new_geometry_edge_cost_plans'][name])!=A.digest(C_entry['new_geometry_edge_cost_plans'][name]):raise ValueError('N and C policy plans must have identical evidence')
                poses={}
                for name,family in families.items():
                    coarse=family['coarse_pose'];R=np.asarray(coarse['R']);t=np.asarray(coarse['t'])
                    value=evaluate(packet,family,native_entry['new_geometry_edge_cost_plans'][name],R,t,raw[name])
                    poses['COARSE_'+name]=dict(family=name,R=R,t=t,computed=value)
                for label,entry,containment in (('SELECTED_NATIVE',native_entry,False),('SELECTED_C',C_entry,True)):
                    name=entry['WD_family'];R=np.asarray(entry['R']);t=np.asarray(entry['t'])
                    value=evaluate(packet,families[name],entry['new_geometry_edge_cost_plans'][name],R,t,raw[name])
                    key='total_containment_arm'if containment else'total_native_arm';saved=entry['solution']['objective']
                    diff=abs(value[key]-saved);max_difference=max(max_difference,diff);checks+=1
                    if diff>1e-8*max(1.,abs(saved))or not value['native_feasible']:raise ValueError('Selected independently recomputed energy/guard changed')
                    poses[label]=dict(family=name,R=R,t=t,computed=value,saved_solution_objective=saved,
                        absolute_total_difference=diff,stored_selected_name=entry['candidate_selection']['selected_name'])
                comparisons={}
                for name in A.FAMILIES:
                    for label,containment in (('NATIVE',False),('C',True)):
                        comparisons[name+'/'+label]=comparison(poses['COARSE_'+name]['computed'],
                            poses['SELECTED_C'if containment else'SELECTED_NATIVE']['computed'],containment=containment)
                n,c=poses['SELECTED_NATIVE']['computed'],poses['SELECTED_C']['computed']
                policies[policy]=dict(poses=poses,coarse_vs_selected=comparisons,native_endpoint_vs_C_selected=dict(
                    C_total_native_pose_minus_C_selected=n['total_containment_arm']-c['total_containment_arm'],
                    native_total_C_selected_minus_native=c['total_native_arm']-n['total_native_arm'],
                    native_pose_feasible_under_C=n['containment_feasible']))
            records.append(dict(id=identifier,partition=packet['identifiers']['partition'],policies=policies))
        A._verify(plan['source_sha256'],plan['input_sha256'])
        result=dict(schema='FIXED_THREE_CASE_CURRENT_GUARD_AND_COST_RESULT_V32',records=records,
            selected_total_checks=checks,selected_total_max_absolute_difference=max_difference,
            plan=binding(OUTPUT/'PLAN.json'),new_G32_consumer_planner_PnP_optimizer_NN_query_GT_or_reference_calls=0,
            score_numeric_reads=0,all_fixed_cases_retained=True,no_new_method_or_runtime_candidate_added=True)
        C.write(OUTPUT/'RESULT.json',result)
        (OUTPUT/'REPORT_KO.md').write_text(report_md(records),encoding='utf-8')
        C.write(OUTPUT/'SEAL.json',dict(plan_sha256=expected_plan_sha256,result=binding(OUTPUT/'RESULT.json'),
            report=binding(OUTPUT/'REPORT_KO.md'),all_current_input_source_bindings_verified=True))
        print(dict(result=binding(OUTPUT/'RESULT.json'),report=binding(OUTPUT/'REPORT_KO.md'),seal=binding(OUTPUT/'SEAL.json')))
    except BaseException as error:
        C.write(OUTPUT/'FAILURE_V1.json',dict(exception=type(error).__name__,reason=str(error),
            completed_cases=len(records),attempt_preserved=True,new_posefit_or_GT_reads=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('declare','run'))
    parser.add_argument('--root-probe-GO',action='store_true');parser.add_argument('--plan-sha256')
    args=parser.parse_args()
    if args.stage=='declare':print(declare(root_probe_GO=args.root_probe_GO))
    else:run(root_probe_GO=args.root_probe_GO,expected_plan_sha256=args.plan_sha256)
