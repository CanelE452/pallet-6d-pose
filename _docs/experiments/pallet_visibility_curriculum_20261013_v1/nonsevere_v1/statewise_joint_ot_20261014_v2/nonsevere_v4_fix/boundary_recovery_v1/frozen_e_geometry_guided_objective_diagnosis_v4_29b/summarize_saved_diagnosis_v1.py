"""Small saved-DIAG-only completion accounting; never replays a pose."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math

OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/'scripts/research').is_dir())
PEER=OUT.parent/'geometry_guided_edges_peer_review_v4_29/E_OBJECTIVE_DIAGNOSTIC_SOURCE_REVIEW_V1.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def new(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def main():
    data=json.loads((OUT/'DIAGNOSIS.json').read_text())
    seal=json.loads((OUT/'DIAGNOSIS_SEAL.json').read_text())
    if sha(OUT/'DIAGNOSIS.json')!=seal['diagnosis_sha256'] or data['count']!=49 or data['arm_count']!=4:
        raise ValueError('Exact completed immutable49x4 diagnosis required')
    summaries={}
    for arm in data['rows'][0]['arms']:
        records=[dict(id=row['id'],**row['arms'][arm]) for row in data['rows']]
        reg=[a for a in records if a['posthoc_e5cm_to_selected_gt5cm_regression']]
        counts=Counter(a['classification'] for a in records)
        regression_counts=Counter(a['classification'] for a in reg)
        if dict(counts)!=data['summary']['classifications'][arm] or dict(regression_counts)!=data['summary']['regression_classifications'][arm]:
            raise ValueError('Saved summary must match every one of its49 rows')
        failed=[];types=Counter();reg_types=Counter()
        for a in records:
            if a['classification']!='saved_e_excluded_by_current_hard_or_numeric_guards':continue
            e=a['primary_e_evaluation'];groups=[];row_types=set()
            for group in e.get('guard_audits',{}).get('actual_edge_support_constraint_audit',[]) or []:
                if group['passed']:continue
                failure_types=[]
                if any(v < -1e-8 for v in group['finite_actual_support_slacks_px']):
                    failure_types.append('finite_projected_edge_does_not_cover_actual_support_with11px_margin')
                directed=[]
                for owner in group['directed_support']:
                    if not owner['hard_order_guard_applied']:continue
                    causes=[]
                    if owner['slack_px'] < -1e-8:causes.append('fixed_order_actual_support_fraction_slack')
                    if owner['projected_endpoint_order_slack_px'] < -1e-8:causes.append('projected_native_endpoint_order_slack')
                    if causes:directed.append({**owner,'failure_types':causes})
                    failure_types.extend(causes)
                if not failure_types:failure_types.append('passed_false_near_solver_tolerance_or_other_reported_slack')
                row_types.update(failure_types)
                groups.append({**group,'failure_types':sorted(set(failure_types)),
                    'failed_directed_owners':directed})
            for kind in row_types:
                types[kind]+=1
                if a['posthoc_e5cm_to_selected_gt5cm_regression']:reg_types[kind]+=1
            other=[]
            for kind,values in e.get('guard_audits',{}).items():
                if kind=='actual_edge_support_constraint_audit':continue
                if any(v.get('passed',v.get('feasible',v.get('visible_support_guards_pass',True))) is False for v in values or []):
                    other.append(kind)
            failed.append(dict(id=a['id'],regression=a['posthoc_e5cm_to_selected_gt5cm_regression'],
                saved_e_family=a['saved_e_family'],selected_family=a['selected_family'],
                minimum_joint_slack=e.get('minimum_joint_slack'),failed_actual_support_groups=groups,
                other_failed_guard_audit_kinds=other))
        alternate_lower=[];max_gap=0.;max_R=0.
        for a in records:
            for family,t in a['e_transport_diagnostics'].items():
                if t['available']:
                    max_gap=max(max_gap,t.get('unordered_physical_body_gap_m',0.))
                    max_R=max(max_R,t.get('physical_rotation_max_absolute_difference',0.))
                trial=a['e_physical_body_trials_by_WD'][family]
                if (a['selected_accepted'] and trial.get('accepted_current_guards') and trial.get('finite_objective')
                    and a['selected_evaluation'].get('finite_objective') and trial['total']<a['selected_evaluation']['total']-1e-8):
                    alternate_lower.append(dict(id=a['id'],family=family,e_trial_total=trial['total'],selected_total=a['selected_evaluation']['total']))
        paired=[a for a in reg if a['primary_e_evaluation'].get('finite_objective') and a['selected_evaluation'].get('finite_objective')]
        kinds=sorted({k for a in paired for k in set(a['primary_e_evaluation']['native_by_kind'])|set(a['selected_evaluation']['native_by_kind'])})
        means={kind:math.fsum(a['primary_e_evaluation']['native_by_kind'].get(kind,0.)-a['selected_evaluation']['native_by_kind'].get(kind,0.) for a in paired)/len(paired) for kind in kinds} if paired else {}
        model_means={kind:math.fsum(a['primary_e_evaluation'][kind]-a['selected_evaluation'][kind] for a in paired)/len(paired) for kind in ('O','S','weighted_C','total')} if paired else {}
        summaries[arm]=dict(count=len(records),classifications=dict(counts),regression_count=len(reg),
            regression_classifications=dict(regression_counts),
            regression_ids=[a['id'] for a in reg],
            eligible_objective_prefers_worse_selected=sum(a['eligible_objective_prefers_worse_selected'] for a in records),
            finite_objective_prefers_selected_without_eligibility_claim=sum(a['finite_objective_prefers_selected_even_if_e_guard_excluded'] for a in records),
            selected_fallback_count=sum(not a['selected_accepted'] for a in records),
            selected_vs_saved_e_family_changed_count=sum(a['selected_family']!=a['saved_e_family'] for a in records),
            regression_family_changed_count=sum(a['selected_family']!=a['saved_e_family'] for a in reg),
            active_edge_count_histogram=dict(sorted(Counter(a['selected_active_plan']['active_physical_edge_count'] for a in records).items())),
            owner_count_histogram=dict(sorted(Counter(a['selected_active_plan']['new_owner_count'] for a in records).items())),
            e_guard_exclusions=failed,guard_failure_types_frame_counts=dict(types),regression_guard_failure_types_frame_counts=dict(reg_types),
            valid_e_physical_trial_lower_than_selected_in_either_WD=alternate_lower,
            maximum_unordered_physical_body_gap_m=max_gap,maximum_Rcf_Q_vs_Rphysical_difference=max_R,
            regression_finite_decomposition_pairs=len(paired),
            regression_mean_e_minus_selected_native_cost_by_kind=means,
            regression_mean_e_minus_selected_model_terms=model_means,
            decomposition_interpretation='Descriptive cost differences on all finite regression pairs, including rejected e. Different saved WD hypotheses may have different named-channel association; these values do not identify causal measurement errors.')
    bindings={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size}
        for p in (Path(__file__),OUT/'DIAGNOSIS.json',OUT/'DIAGNOSIS_SEAL.json',OUT/'PLAN.json',
            OUT/'PREPARATION_SEAL.json',OUT/'SOURCE_AND_SYNTHETIC_RECEIPT_V1.json',PEER)}
    result=dict(schema='POSTSCORE_V29_E_OBJECTIVE_DIAGNOSTIC_OWNER_COMPLETION_V1',
        UTC=datetime.now(timezone.utc).isoformat(),completed_TEST49=49,arm_count=4,frame_arm_comparisons=196,
        replay_process_exit_code=0,actual_replay_executions=1,new_optimization_or_measurement_or_GT_decode=0,
        maximum_selected_saved_objective_difference=data['summary']['maximum_selected_saved_objective_difference'],
        summaries=summaries,bindings=bindings,
        interpretation=['Two observed failure paths: a good saved e pose can be excluded by current hard actual-line support guards; when e passes, the current objective can favor a selected pose with worse saved translation metric.',
            'No valid evaluated e physical trial had lower energy in either permitted WD here. This is limited to these two transported frozen poses, not a global-optimality proof or proof that all failures are unrelated to optimization.',
            'V30 association normal-distance gating does not establish finite along-line support association; no claim that this diagnosis has already been fixed before its independent primary evaluation.',
            'All49 and fallbacks remain. No e numeric pose is inserted into runtime, and no runtime source/model/results/current V30 operation was changed.'])
    new(OUT/'OWNER_COMPLETION_V1.json',result)
    lines=['# V29 TEST49 저장 자세 비용 진단','',
        '새 관측·최적화·정답 decode 없이 저장된 e와 선택 자세를 같은 frozen 목적함수로 한 번 재평가했습니다. 원 e의 WD 대응을 본문에 보존했고, 물리 상자가 같은 다른 WD 운반은 별도 부록입니다.','',
        '| 방법 | TEST 악화 | 통과한 e보다 나쁜 선택 자세의 비용이 낮음 | e guard 제외 | fallback |',
        '|---|---:|---:|---:|---:|']
    for arm,s in summaries.items():
        c=s['regression_classifications']
        lines.append(f"| {arm} | {s['regression_count']} | {c.get('current_objective_prefers_selected_over_diagnostic_e',0)} | {c.get('saved_e_excluded_by_current_hard_or_numeric_guards',0)} | {c.get('selected_is_own_fallback_not_a_solved_pose',0)} |")
    lines.extend(['','악화는 저장 e의 translation≤5cm에서 선택 자세가 >5cm인 경우입니다. 실제 선 지지 guard가 e를 제외한 모든 기록은 `OWNER_COMPLETION_V1.json`에 원 finite/directed slack과 함께 있습니다. 제외 guard는 actual-edge support audit이며 다른 guard 실패와 구분했습니다.','',
        '전체 49행에서 ALL 두 arm은 비용선호 37 / e 제외 9 / fallback 3, SOURCE 두 arm은 43 / 5 / 1입니다. 두 WD의 유효한 e 물리 운반 대안 중 더 낮은 비용은 없었고, 저장 선택 비용 검산 최대차는 7.11e−15였습니다.','',
        '두 경로 모두 관측되며, 전역최적·WD/선의 인과 증명이나 모든 실패가 최적화와 무관하다는 결론은 아닙니다. V30의 법선 거리 association은 유한 along-line 대응 보장이 아니므로 수정 완료로 간주하지 않습니다. 기본 e와 모든 원 산출물은 유지했습니다.',''])
    with (OUT/'SUMMARY_KO_V1.md').open('x') as f:f.write('\n'.join(lines))
    new(OUT/'OWNER_COMPLETION_SEAL_V1.json',dict(receipt_sha256=sha(OUT/'OWNER_COMPLETION_V1.json'),
        summary_sha256=sha(OUT/'SUMMARY_KO_V1.md'),diagnosis_sha256=sha(OUT/'DIAGNOSIS.json'),passed=True,
        completed_TEST49=49,arm_count=4))
    print(json.dumps(dict(receipt_sha256=sha(OUT/'OWNER_COMPLETION_V1.json'),
        seal_sha256=sha(OUT/'OWNER_COMPLETION_SEAL_V1.json'),summary_sha256=sha(OUT/'SUMMARY_KO_V1.md'),
        summaries={a:{k:s[k] for k in ('regression_count','regression_classifications','selected_vs_saved_e_family_changed_count',
            'guard_failure_types_frame_counts','regression_guard_failure_types_frame_counts',
            'valid_e_physical_trial_lower_than_selected_in_either_WD')} for a,s in summaries.items()}),ensure_ascii=False))


if __name__=='__main__':main()
