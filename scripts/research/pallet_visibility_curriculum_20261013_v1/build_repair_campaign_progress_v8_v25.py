"""Postseal, append-only uniform coarse-prior/containment campaign accounting.

Source prepared before v25 metric access. Only sealed scores and audit/report
bindings are read at execution; no pose fitting, image or reference decode.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import html

from . import build_repair_campaign_progress_v7_v24 as PREVIOUS
from . import build_repair_campaign_progress_v6_v23 as COMMON
from . import evaluate_coarse_prior_containment_v4_25 as V

P,C=PREVIOUS.P,PREVIOUS.C
OUTPUT=PREVIOUS.OUTPUT
NAMES=('PROGRESS_V8_V25.json','RESULTS_V8_V25.html','PROGRESS_V8_V25_SEAL.json')
LABELS={V.ARMS[1]:'v25 위치 prior · 영상선',V.ARMS[2]:'v25 위치 prior · 영상선 · e WD 유지',
    V.ARMS[3]:'v25 위치 prior · 포함 제약',V.ARMS[4]:'v25 위치 prior · 포함 제약 · e WD 유지',
    V.ARMS[5]:'v25 위치·회전 prior · 영상선',V.ARMS[6]:'v25 위치·회전 prior · 영상선 · e WD 유지',
    V.ARMS[7]:'v25 위치·회전 prior · 포함 제약',V.ARMS[8]:'v25 위치·회전 prior · 포함 제약 · e WD 유지'}


def _render(data):
    lines=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>v25 전체192 비교</title>',
        '<style>body{font:16px/1.6 sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:#192536}'
        'table{border-collapse:collapse;width:100%;margin:20px 0 40px}th,td{border-bottom:1px solid #ccd4df;padding:8px;text-align:right}'
        'th:first-child,td:first-child{text-align:left}th{background:#eef4fa}.scroll{overflow-x:auto}.new{background:#f3faf5}</style>',
        '<p>기본 e는 유지합니다. 각 행은 기존 TEST49와 추가143장 모두에 동일하게 적용한 하나의 방법입니다. 실패·fallback·원래 native 대응 플래그와 관계없이 모든 행을 포함했고 사진별 최선 방법을 섞지 않았습니다. severe 제외, CAL46은 별도입니다.</p>',
        '<p>v25는 SOURCE의 현재 complete RGB/unknown q 보존, 기존 부분관측·CAL 접선·O/S·hard guard·물리 크기·원래 시작점 정책을 유지합니다. 과거에 이미 partial로 바뀐 RGB q를 복구한 것은 아닙니다. 기존 CAL44 공분산과 median 편향을 그대로 재사용하며 새 매개변수 적합은 없습니다.</p>',
        '<p>위치 prior는 초기 모델 중심의 카메라 ray와 log-depth 잔차 및 3×3 주변 공분산을 사용합니다. 위치·회전 prior는 같은 첫 N3_SUBPIX 물리자세에 대한 기존 P1의 6차원 Gaussian 비용입니다. λ=1·편향 강도=1이며 편향은 잔차에서 한 번만 뺍니다. 위치만 쓰는 대조의 계수가 새로 CAL 최적화됐다고 주장하지 않습니다.</p>',
        '<p>새 포함 제약은 고정된 전역 마스크 표본이 투영 모델 hull 안에 있으면 0, 밖이면 거리에 따른 soft 비용을 줍니다. 모델 전체 윤곽을 보이는 마스크에 맞추는 항은 없고, native 동률에서도 이전 전체 마스크를 사용하지 않습니다. 이 가정은 마스크 오염에 취약할 수 있으며 모델 확장·깊이·유일성을 보장하지 않습니다.</p>',
        '<p>e WD 유지는 기존 e가 예측한 가로·세로 family 이름으로 같은 후보은행을 제한하는 별도 선택기입니다. e의 수치 자세를 prior target이나 fallback으로 쓰지 않습니다. 그 family 예측은 과거 마스크에 의존하므로 독립적인 물리 축 판별 정보가 아닙니다. S의 2D 출력은 자세 투영뿐이며 S 영상·2D·영역 관측은 없습니다. 초기 모델에 대한 제한된 coarse-S prior는 계속 활성입니다. U는 대략적인 영역만 사용합니다.</p>']
    for part,title in(('PRIMARY192','전체192 = TEST49 + 추가143'),('TEST','기존 TEST49'),('REMAINDER','추가143'),('CAL','별도 CAL46')):
        lines.append('<p><b>'+title+'</b> — T≤5cm는 위치 오차, 공동은 5cm·5° 모두 만족한 수입니다.</p><div class="scroll"><table><tr>'+
            ''.join('<th>'+s+'</th>'for s in('고정 방법','수','T≤5cm','공동','평균 T(cm)','최대 T(cm)','평균 R(°)','최대 R(°)','R 미확정','fallback'))+'</tr>')
        for method in data['methods']:
            q=method['populations'][part]
            numbers=[q['count'],q['translation_le5'],q['successes_5cm_5deg'],q['translation_cm']['mean'],q['translation_cm']['max'],
                q['rotation_deg']['mean'],q['rotation_deg']['max'],q['rotation_unresolved'],q['fallback_count']]
            cells=['미확정'if x is None else(f'{x:.3f}'if isinstance(x,float)else str(x))for x in numbers]
            lines.append('<tr'+(' class="new"'if method['method']in V.ARMS[1:]else'')+'><td>'+html.escape(method['label_KO'])+'</td>'+''.join('<td>'+s+'</td>'for s in cells)+'</tr>')
        lines.append('</table></div>')
    lines.extend(['<p>추가143장의 참조는 저장 annotation/PnP 유도값이며 독립 물리 계측이 아닙니다. 같은 TEST를 반복 사용한 탐색 결과입니다. 비용 감소·accepted·family 고정이 정확도나 유일성을 증명하지 않습니다. 전체 목표 완료나 새 기본 방법 채택은 이 문서에서 선언하지 않습니다.</p>',
        '<p><a href="PROGRESS_V8_V25.json">전체 수치·대응별 변화·검산 근거</a></p></html>'])
    return '\n'.join(lines)


def build(*,root_postscore_GO=False):
    if root_postscore_GO is not True:raise ValueError('Require full9arm seal, zero-failure independent audit and fixed score')
    paths=[OUTPUT/name for name in NAMES]
    if any(p.exists()for p in paths):raise FileExistsError('Preserve previous progress')
    previouspath=OUTPUT/'PROGRESS_V7_V24.json';previoussealpath=OUTPUT/'PROGRESS_V7_V24_SEAL.json'
    previous=C.read(previouspath);previousseal=C.read(previoussealpath)
    for key in('source','json','html'):
        b=previousseal[key]
        if C.sha(C.ROOT/b['path'])!=b['sha256']:raise ValueError('Frozen V7 source/document changed')
    if previous['method_count']!=15 or len(previous['methods'])!=15:raise ValueError('Exact previous15 uniform methods required')
    protocol,reports=P._load_run(V.OUTPUT);oldprotocol,old=P._load_run(V.V.OUTPUT)
    COMMON._require_declared_arms(V.OUTPUT,V.ARMS);COMMON._same_e(old,reports)
    predseal=C.read(V.OUTPUT/'PREDICTION_SEAL.json')
    if predseal.get('all9_arms_complete')is not True or protocol['ids']!=oldprotocol['ids']:raise ValueError('Nine complete arms on all original238 required')
    auditpaths=COMMON._require_audit(V.OUTPUT)
    methods=list(previous['methods'])
    for arm in V.ARMS[1:]:
        populations=COMMON._population_summaries(reports,arm)
        methods.append(dict(method=arm,label_KO=LABELS[arm],populations=populations,source_output=str(V.OUTPUT.relative_to(C.ROOT)),
            current_default=False,original_TEST49_all_translation_le5=populations['TEST']['translation_le5']==49,
            numerical_primary192_target_met=populations['PRIMARY192']['translation_le5']==192))
    if len(methods)!=23 or len({r['method']for r in methods})!=23:raise ValueError('23 distinct uniform methods required')
    comparisons=((old,reports,V.V.ARMS[3],V.ARMS[1],'SOURCE native → translation prior; old numeric mask tie also removed'),
        (old,reports,V.V.ARMS[3],V.ARMS[5],'SOURCE native → full pose prior; old numeric mask tie also removed'),
        (reports,reports,V.ARMS[1],V.ARMS[3],'translation native → soft containment; same-mode native bank retained'),
        (reports,reports,V.ARMS[5],V.ARMS[7],'full pose native → soft containment; same-mode native bank retained'),
        (reports,reports,V.ARMS[1],V.ARMS[2],'translation native same bank → predicted e family constraint'),
        (reports,reports,V.ARMS[3],V.ARMS[4],'translation containment same bank → predicted e family constraint'),
        (reports,reports,V.ARMS[5],V.ARMS[6],'full pose native same bank → predicted e family constraint'),
        (reports,reports,V.ARMS[7],V.ARMS[8],'full pose containment same bank → predicted e family constraint'))
    paired=[]
    for before,after,a,b,label in comparisons:
        parts={part:COMMON.PAIR._paired(before[part]['rows'],after[part]['rows'],a,b)for part in P.COUNTS}
        parts['PRIMARY192']=COMMON.PAIR._paired(before['TEST']['rows']+before['REMAINDER']['rows'],after['TEST']['rows']+after['REMAINDER']['rows'],a,b)
        paired.append(dict(before=a,after=b,label=label,populations=parts))
    evidence=[Path(__file__),previouspath,previoussealpath,*auditpaths,V.PLAN,V.PLAN.with_name('METADATA_CLARIFICATION_V1.json')]
    evidence.extend(V.OUTPUT/name for name in('PROTOCOL.json','INPUT_SEAL.json','PREDICTION_SEAL.json','SCORE_SEAL.json',
        'CAL_SCORE.json','TEST_SCORE.json','REMAINDER_SCORE.json'))
    evidence.extend(C.ROOT/Path(protocol['coarse_prior_model_binding']['path']).parent/name for name in('MODEL.json','SEAL.json','CAL_LOGO_REPORT.json'))
    for relative in('coarse_pose_prior_source_audit_v4_25/SOURCE_AUDIT_V1.json','global_outline_occlusion_source_audit_v4_25/AUDIT_V1.json',
            'frozen_e_current_objective_diagnosis_v4_25/DIAGNOSIS.json'):
        evidence.append(V.OUTPUT.parent/relative)
    active=C.read(OUTPUT/'ACTIVE_GOAL_PROTOCOL_V1.json');P._verify(active['protected_user_sha256'])
    bindings=[P._binding(path)for path in evidence]
    data=dict(schema='ACTIVE_REPAIR_CAMPAIGN_PROGRESS_V8_COARSE_PRIOR_CONTAINMENT_V25',generated_UTC=datetime.now(timezone.utc).isoformat(),
        goal_status='active',goal_completion_declared=False,default_changed=False,default_method=P.BASELINE,
        methods=methods,method_count=23,paired_controls=paired,exact_baseline_consistency_verified=True,
        primary=dict(TEST49=49,REMAINDER143=143,total=192,severe=0),separate_CAL=46,
        numerical_target_met_any_new_method=any(m['numerical_primary192_target_met']for m in methods[15:]),
        all_methods_uniform_full_population=True,no_per_frame_best_or_GT_ID_partition_routing=True,
        all238_no_frame_fallback_or_native_flag_filter=True,S_image_2D_or_regions=0,U_regions_only=True,
        original_hard_guards_unchanged=True,original_v19_start_names_policy_unchanged=True,
        new_v25_controls_old_full_outline_not_used_even_at_ties=True,no_numeric_e_pose_prior_or_fallback=True,
        new_v25_contract_scope='new8 controls only; previous15 methods retain their original recipes and metrics',
        old_e_predicted_family_depends_on_historical_whole_mask=True,new_parameter_fits=0,
        coarse_prior_model_reused=True,translation_prior_uses_marginal_covariance=True,coarse_bias_subtracted_once=True,
        one_sided_containment_is_conditional_soft_sampled_cue_not_full_subset_certificate=True,
        additional_reference_is_annotation_PnP_derived_not_independent_metrology=True,reused_exploratory_TEST_not_new_holdout=True,
        no_new_inference_pose_fit_images_or_reference_decode=True,independent_root_audits=[P._binding(auditpaths[0])],
        evidence=bindings,protected_user_sha256=active['protected_user_sha256'])
    for binding in bindings:
        if C.sha(C.ROOT/binding['path'])!=binding['sha256']:raise ValueError('Evidence changed during aggregation')
    C.write(paths[0],data);paths[1].write_text(_render(data),encoding='utf-8')
    P._verify(active['protected_user_sha256'])
    C.write(paths[2],dict(source=P._binding(Path(__file__)),json=P._binding(paths[0]),html=P._binding(paths[1]),evidence=bindings,
        exact_e_preserved=True,method_count=23,no_new_inference_pose_fit_images_or_reference_decode=True))
    print(dict(methods=23,json_sha256=C.sha(paths[0]),html_sha256=C.sha(paths[1])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--after-v25-score-and-root-audit-go',action='store_true',required=True)
    args=parser.parse_args();build(root_postscore_GO=args.after_v25_score_and_root_audit_go)
