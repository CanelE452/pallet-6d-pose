"""Append-only uniform v24 accounting, after all238 prediction/audit/score.

Prepare source before new metrics; execution reads sealed scores only.
No image, inference, optimization, reference decode or per-frame method choice.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import html

from . import build_repair_campaign_progress_v6_v23 as PREV
from . import evaluate_source_preserving_tangent_mask_v4_24 as V

P,C=PREV.P,PREV.P.C
OUTPUT=PREV.OUTPUT
NAMES=('PROGRESS_V7_V24.json','RESULTS_V7_V24.html','PROGRESS_V7_V24_SEAL.json')
LABELS={V.ARMS[1]:'v24 전체 B/I 교체 · CAL 접선',V.ARMS[2]:'v24 전체 B/I 교체 · CAL 접선 + 마스크 선택',
    V.ARMS[3]:'v24 RGB q 보존 · CAL 접선',V.ARMS[4]:'v24 RGB q 보존 · CAL 접선 + 마스크 선택',
    V.ARMS[5]:'v24 RGB q 보존 · 연속 마스크 최적화',V.ARMS[6]:'v24 RGB q 보존 · 연속 마스크 · e 예측 WD 유지'}


def _render(data):
    lines=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>v24 전체192 비교</title>',
        '<style>body{font:16px/1.6 sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:#192536}'
        'table{border-collapse:collapse;width:100%;margin:20px 0 40px}th,td{border-bottom:1px solid #ccd4df;padding:8px;text-align:right}'
        'th:first-child,td:first-child{text-align:left}th{background:#eef4fa}.scroll{overflow-x:auto}</style>',
        '<p>기본 e는 그대로입니다. 각 행은 TEST49와 추가143장에 동일하게 적용한 하나의 방법입니다. 모든 실패·fallback을 포함했고 사진별 최선 방법을 섞지 않았습니다. severe 제외, CAL46 별도입니다.</p>',
        '<p>v24는 기존 실제 영상선·마스크를 재사용합니다. 전체 B/I 교체안과 RGB complete q 보존안을 분리했고, 접선 분산은 별도 CAL의 1+γ·tΣt를 사용합니다. RGB 보존안은 적용 집합에 맞는 별도 CAL 모서리 잡음 모델도 사용하므로 두 안의 차이가 단일 요소 효과는 아닙니다.</p>',
        '<p>공통 접선 계수 γ는 CAL 14프레임의 30접선(RGB 28·fallback 2)으로 추정했습니다. SOURCE의 두 fallback 접선만을 별도로 검증한 신뢰구간이 아닙니다. RGB q 보존은 현재 complete인 관측에 적용하며, 앞 단계에서 이미 partial로 바뀐 옛 RGB q를 복구하지 않습니다.</p>',
        '<p>마스크 선택은 고정 후보의 순위를 바꿉니다. 연속 마스크 최적화는 같은 마스크 비용을 자세 미분에 한 번 포함하며, RGB 보존안의 모든 후보와 새 endpoint를 함께 검사합니다. 마지막 행은 같은 전체 후보은행에서 기존 e가 예측한 WD family로 제한합니다. 기존 e의 수치 자세를 fallback으로 사용하지 않습니다.</p>',
        '<p>S는 모델/PnP 투영만 출력합니다. S 영상·2D·영역 관측은 0이며 초기 모델 가시성에 대한 제한된 prior만 사용합니다. U는 영역만 사용하고, 기존 hard guard·원래 q·공분산·물리 크기·v19 시작점 정책을 유지합니다.</p>']
    for part,title in(('PRIMARY192','전체192'),('TEST','기존 TEST49'),('REMAINDER','추가143'),('CAL','별도 CAL46')):
        lines.append('<p><b>'+title+'</b> — T≤5cm: 위치 오차, 공동: 5cm·5° 모두 만족.</p><div class="scroll"><table><tr>'+
            ''.join('<th>'+s+'</th>'for s in('방법','수','T≤5cm','공동','평균 T(cm)','최대 T(cm)','평균 R(°)','최대 R(°)','R 미확정','fallback'))+'</tr>')
        for method in data['methods']:
            q=method['populations'][part]
            numbers=[q['count'],q['translation_le5'],q['successes_5cm_5deg'],q['translation_cm']['mean'],q['translation_cm']['max'],
                q['rotation_deg']['mean'],q['rotation_deg']['max'],q['rotation_unresolved'],q['fallback_count']]
            cells=['미확정'if x is None else(f'{x:.3f}'if isinstance(x,float)else str(x))for x in numbers]
            lines.append('<tr><td>'+html.escape(method['label_KO'])+'</td>'+''.join('<td>'+s+'</td>'for s in cells)+'</tr>')
        lines.append('</table></div>')
    lines.append('<p>추가143장의 참조는 annotation/PnP 유도값이며 독립 물리 계측이 아닙니다. 반복 사용한 TEST의 탐색 결과입니다. 비용 감소·accepted·WD 유지가 정확도나 유일성을 보장하지 않습니다. 전체 목표 완료와 기본 방법 변경을 선언하지 않았습니다.</p>')
    lines.append('<p><a href="PROGRESS_V7_V24.json">전체 수치·대응별 변화·검산 근거</a></p></html>')
    return '\n'.join(lines)


def build(*,root_postscore_GO=False):
    if root_postscore_GO is not True:raise ValueError('Require complete prediction, zero-failure independent audit and fixed score')
    paths=[OUTPUT/name for name in NAMES]
    if any(p.exists()for p in paths):raise FileExistsError('Preserve previous progress')
    previouspath=OUTPUT/'PROGRESS_V6_V23.json';previoussealpath=OUTPUT/'PROGRESS_V6_V23_SEAL.json'
    previous=C.read(previouspath);previousseal=C.read(previoussealpath)
    for key in('source','json','html'):
        b=previousseal[key]
        if C.sha(C.ROOT/b['path'])!=b['sha256']:raise ValueError('Frozen V6 source/document changed')
    if previous['method_count']!=9 or len(previous['methods'])!=9:raise ValueError('Exact previous nine uniform methods required')
    protocol,reports=P._load_run(V.OUTPUT);oldprotocol,old=P._load_run(V.V.OUTPUT)
    PREV._require_declared_arms(V.OUTPUT,V.ARMS);PREV._same_e(old,reports)
    if protocol['ids']!=oldprotocol['ids']:raise ValueError('Original all238 order required')
    auditpath=V.OUTPUT/'independent_root_audit_v1/AUDIT.json';auditsealpath=auditpath.with_name('AUDIT_SEAL.json')
    audit=C.read(auditpath);auditseal=C.read(auditsealpath)
    if (audit['rows']!=238 or audit['failure_count']or audit['failures']or C.sha(auditpath)!=auditseal['receipt_sha256']
            or audit['prediction_seal']['sha256']!=C.sha(V.OUTPUT/'PREDICTION_SEAL.json')):
        raise ValueError('Independent current all238 zero-failure audit required')
    methods=list(previous['methods'])
    for arm in V.ARMS[1:]:
        populations=PREV._population_summaries(reports,arm)
        methods.append(dict(method=arm,label_KO=LABELS[arm],populations=populations,source_output=str(V.OUTPUT.relative_to(C.ROOT)),
            current_default=False,original_TEST49_all_translation_le5=populations['TEST']['translation_le5']==49,
            numerical_primary192_target_met=populations['PRIMARY192']['translation_le5']==192))
    if len(methods)!=15 or len({r['method']for r in methods})!=15:raise ValueError('15 distinct uniform methods required')
    comparisons=((old,reports,V.V.ARMS[2],V.ARMS[1],'v23 OS native → TCAL all'),
        (old,reports,V.V.ARMS[4],V.ARMS[2],'v23 OS fixedmask → TCAL all'),
        (reports,reports,V.ARMS[1],V.ARMS[3],'all → source-preserving; separate edge model'),
        (reports,reports,V.ARMS[2],V.ARMS[4],'all → source-preserving fixedmask; separate edge model'),
        (reports,reports,V.ARMS[4],V.ARMS[5],'source fixedmask → continuousmask; B bank retained'),
        (reports,reports,V.ARMS[5],V.ARMS[6],'same continuousmask bank → predicted e family constraint'))
    paired=[]
    for before,after,a,b,label in comparisons:
        parts={part:PREV.PAIR._paired(before[part]['rows'],after[part]['rows'],a,b)for part in P.COUNTS}
        parts['PRIMARY192']=PREV.PAIR._paired(before['TEST']['rows']+before['REMAINDER']['rows'],after['TEST']['rows']+after['REMAINDER']['rows'],a,b)
        paired.append(dict(before=a,after=b,label=label,populations=parts))
    evidence=[Path(__file__),previouspath,previoussealpath,auditpath,auditsealpath,V.PLAN]
    evidence.extend(V.OUTPUT/name for name in('PROTOCOL.json','INPUT_SEAL.json','PREDICTION_SEAL.json','SCORE_SEAL.json',
        'CAL_SCORE.json','TEST_SCORE.json','REMAINDER_SCORE.json'))
    for module in(V.SOURCE_CAL,V.TCAL):
        evidence.extend(module.OUTPUT/name for name in('MODEL.json','MODEL_SEAL.json','CAL_LOGO_REPORT.json'))
    active=C.read(OUTPUT/'ACTIVE_GOAL_PROTOCOL_V1.json');P._verify(active['protected_user_sha256'])
    bindings=[P._binding(path)for path in evidence]
    data=dict(schema='ACTIVE_REPAIR_CAMPAIGN_PROGRESS_V7_SOURCE_TCAL_MASK_V24',generated_UTC=datetime.now(timezone.utc).isoformat(),
        goal_status='active',goal_completion_declared=False,default_changed=False,default_method=P.BASELINE,
        methods=methods,method_count=15,paired_controls=paired,exact_baseline_consistency_verified=True,
        primary=dict(TEST49=49,REMAINDER143=143,total=192,severe=0),separate_CAL=46,
        numerical_target_met_any_new_method=any(m['numerical_primary192_target_met']for m in methods[9:]),
        all_methods_uniform_full_population=True,no_per_frame_best_or_GT_ID_partition_routing=True,
        all238_no_frame_fallback_or_native_flag_filter=True,S_image_2D_or_regions=0,U_regions_only=True,
        original_hard_guards_unchanged=True,original_v19_start_names_policy_unchanged=True,
        additional_reference_is_annotation_PnP_derived_not_independent_metrology=True,reused_exploratory_TEST_not_new_holdout=True,
        no_new_inference_pose_fit_images_or_reference_decode=True,independent_root_audits=[P._binding(auditpath)],
        evidence=bindings,protected_user_sha256=active['protected_user_sha256'])
    for binding in bindings:
        if C.sha(C.ROOT/binding['path'])!=binding['sha256']:raise ValueError('Evidence changed during aggregation')
    C.write(paths[0],data);paths[1].write_text(_render(data),encoding='utf-8')
    P._verify(active['protected_user_sha256'])
    C.write(paths[2],dict(source=P._binding(Path(__file__)),json=P._binding(paths[0]),html=P._binding(paths[1]),evidence=bindings,
        exact_e_preserved=True,method_count=15,no_new_inference_pose_fit_images_or_reference_decode=True))
    print(dict(methods=15,json_sha256=C.sha(paths[0]),html_sha256=C.sha(paths[1])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--after-v24-score-and-root-audit-go',action='store_true',required=True)
    args=parser.parse_args();build(root_postscore_GO=args.after_v24_score_and_root_audit_go)
