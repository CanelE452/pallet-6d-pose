"""Append four uniform v29 methods after sealed238 audit and scores.

The previous29 methods remain exact. Small score rows only are decoded;
large predictions/observations are bound by hash. No selection by frame or
reference, no fitting, images or optimization. Source written before scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import html
from pathlib import Path

from . import build_repair_campaign_progress_v9_v26 as PREVIOUS
from . import build_repair_campaign_progress_v6_v23 as COMMON
from . import evaluate_geometry_guided_edges_v4_29 as V

C,P=PREVIOUS.C,PREVIOUS.P
OUTPUT=PREVIOUS.OUTPUT
LABELS={V.ARMS[1]:'v29 기하 유도 실제 선 · B/I 방향 교체',
        V.ARMS[2]:'v29 기하 유도 실제 선 · B/I 방향 교체 + 포함 제약',
        V.ARMS[3]:'v29 기하 유도 실제 선 · 기존 RGB q 보존',
        V.ARMS[4]:'v29 기하 유도 실제 선 · 기존 RGB q 보존 + 포함 제약'}


def _paired_definitions():
    pairs=[(V.ARMS[0],arm,'기본 e → '+LABELS[arm]) for arm in V.ARMS[1:]]
    pairs.extend([(V.ARMS[1],V.ARMS[2],'전체 B/I 교체에서 포함 제약 추가'),
                  (V.ARMS[3],V.ARMS[4],'RGB q 보존에서 포함 제약 추가'),
                  (V.ARMS[1],V.ARMS[3],'native에서 RGB q 보존 정책 비교'),
                  (V.ARMS[2],V.ARMS[4],'포함 제약에서 RGB q 보존 정책 비교')])
    return [dict(before=a,after=b,label_KO=label) for a,b,label in pairs]


def _load_new_run(folder):
    protocol=C.read(folder/'PROTOCOL.json');pred=C.read(folder/'PREDICTION_SEAL.json');score=C.read(folder/'SCORE_SEAL.json')
    if (pred['rows']!=238 or pred['arms']!=list(V.ARMS) or pred['arm_count']!=5
            or pred['all_declared_arms_complete'] is not True or pred['before_new_control_metric_read'] is not True
            or C.sha(folder/'PREDICTIONS.jsonl.gz')!=pred['predictions_sha256']
            or C.sha(folder/'PROTOCOL.json')!=pred['protocol_sha256']
            or C.sha(folder/'INPUT_SEAL.json')!=pred['input_seal_sha256']
            or C.sha(folder/'PREDICTION_SEAL.json')!=score['prediction_seal']['sha256']):
        raise ValueError('Complete exact five-arm all238 prediction and score seals required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):P._verify(protocol[key])
    reports={}
    for part,count in P.COUNTS.items():
        path=folder/(part+'_SCORE.json')
        if C.sha(path)!=score['score_sha256'][part]:raise ValueError('Frozen score changed')
        report=C.read(path)
        if report['count']!=count or len(report['rows'])!=count or len({r['id'] for r in report['rows']})!=count:
            raise ValueError('Keep all original rows and failures')
        if any(set(r['metrics'])!=set(V.ARMS) for r in report['rows']):raise ValueError('Every declared arm required')
        reports[part]=report
    if [r['id'] for part in P.COUNTS for r in reports[part]['rows']]!=protocol['ids']:
        raise ValueError('Exact ordered238 inventory required')
    return protocol,reports


def _render(data):
    out=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>v29 전체192 비교</title>',
        '<style>body{font:16px/1.6 sans-serif;max-width:1600px;margin:30px auto;padding:0 20px}'
        'table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #ccd4df;text-align:right}'
        'th:first-child,td:first-child{text-align:left}th{background:#eef4fa}.new{background:#eff8ef}.scroll{overflow:auto}</style>',
        '<p>기본 e는 유지합니다. 기존 29개 방법을 그대로 보존하고 기하 유도 실제 선을 쓰는 4개 균일 방법을 추가했습니다. '
        'TEST49와 추가143 모두에 같은 방법을 적용하며 실패와 fallback도 포함합니다. severe는 제외하고 CAL46은 별도입니다.</p>',
        '<p>초기 키포인트와 기존 native 시작 자세에서 예상 선 방향을 만들고, 그 주변 실제 RGB 경계를 찾습니다. '
        '파렛트 전역 외곽과 내부 구멍을 구분하고, 관측된 짧은 선과 인접 선의 진행 방향을 검사합니다. '
        '모델 선 자체를 영상 관측으로 넣거나 가려진 지점을 새 픽셀 좌표로 만들지 않습니다.</p>',
        '<p>확실한 선 한 방향은 해당 키포인트의 법선 정보를 교체하고 보관된 q의 실제 선 접선 정보만 남깁니다. '
        '비평행인 두 실제 선 방향의 유한·순서 조건이 맞으면 q의 2D 비용을 교체합니다. '
        'RGB q 보존 정책에서는 정확한 fallback 소스만 교체합니다. 정책별 선 잡음과 접선 γ는 별도 CAL 대상으로 두고, '
        '표본이 없어 모델이 준비되지 않은 정책은 새 선을 일괄 미사용합니다. 한 세션만 지지하는 척도는 제한된 기술 통계입니다.</p>',
        '<p>이번 실제 데이터에는 별도 다른 물체 가림 마스크가 없습니다. 파렛트 마스크가 없는 것만으로 O를 판정하지 않으며 '
        '미관측 구간은 unknown입니다. 별도 가림 마스크 입력이 있을 때만 해당 구간을 샘플링에서 제외하는 연결을 지원합니다. '
        'S는 모델/PnP 투영만, U는 대략적인 영역만 사용합니다.</p>',
        '<p>새 방법의 full pose·translation prior 가중치는 0이며 기존 제한된 O/S 항만 유지합니다. '
        '포함 제약은 관측 전역 외곽이 투영 모델 hull 밖에 있을 때만 주는 고정 λ1 항입니다. '
        '옛 전체 대칭 마스크 동률 처리는 쓰지 않습니다.</p>']
    for part,label in (('PRIMARY192','전체192'),('TEST','기존 TEST49'),('REMAINDER','추가143'),('CAL','별도 CAL46')):
        out.append('<p><b>'+label+'</b> — 공동은 위치 5cm 및 회전 5°를 모두 만족한 수입니다.</p><div class="scroll"><table><tr>'+''.join(
            '<th>'+x+'</th>' for x in ('고정 방법','수','위치≤5cm','공동','평균 위치 cm','최대 위치 cm','최대 사례','평균 회전 °','최대 회전 °','fallback'))+'</tr>')
        for method in data['methods']:
            q=method['populations'][part]
            values=[q['count'],q['translation_le5'],q['successes_5cm_5deg'],q['translation_cm']['mean'],
                    q['translation_cm']['max'],q['max_translation_id'],q['rotation_deg']['mean'],q['rotation_deg']['max'],q['fallback_count']]
            cells=['미확정' if x is None else f'{x:.3f}' if isinstance(x,float) else str(x) for x in values]
            out.append('<tr'+(' class="new"' if method['method'] in LABELS else '')+'><td>'+html.escape(method['label_KO'])+
                       '</td>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>')
        out.append('</table></div>')
    out.extend(['<p>동일한 TEST를 반복 사용한 탐색 결과이며 추가143 참조는 annotation/PnP 유도값입니다. '
        '독립 물리 계측이나 새 holdout 성능으로 해석하지 않습니다. 영상선과 q의 통계적 독립, 전역 최적해, 자세의 유일성을 주장하지 않습니다.</p>',
        '<p><a href="PROGRESS_V10_V29.json">전체 수치와 대응 비교</a> · <a href="RESULTS_V9_V26.html">이전 결과</a></p></html>'])
    return '\n'.join(out)


def build(*,root_postscore_GO=False):
    if root_postscore_GO is not True:raise ValueError('Complete prediction238, audit and scores first')
    paths=[OUTPUT/name for name in ('PROGRESS_V10_V29.json','RESULTS_V10_V29.html','PROGRESS_V10_V29_SEAL.json')]
    if any(p.exists() for p in paths):raise FileExistsError('Append-only campaign accounting')
    oldpath=OUTPUT/'PROGRESS_V9_V26.json';oldseal=OUTPUT/'PROGRESS_V9_V26_SEAL.json'
    previous=C.read(oldpath);previousseal=C.read(oldseal)
    for key in ('source','json','html'):
        binding=previousseal[key]
        if C.sha(C.ROOT/binding['path'])!=binding['sha256']:raise ValueError('Previous frozen campaign changed')
    if previous['method_count']!=29:raise ValueError('Retain all previous29 methods')
    protocol,reports=_load_new_run(V.OUTPUT);_,oldreports=P._load_run(PREVIOUS.V.OUTPUT)
    COMMON._require_declared_arms(V.OUTPUT,V.ARMS)
    COMMON._same_e(oldreports,reports)
    PREVIOUS._require_same_reference_bindings(oldreports,reports)
    auditpaths=COMMON._require_audit(V.OUTPUT)
    methods=list(previous['methods'])
    for arm in V.ARMS[1:]:
        populations=COMMON._population_summaries(reports,arm)
        source_policy='ALL_BI_REPLACE' if arm in V.ARMS[1:3] else 'RGB_NONFALLBACK_PRESERVE'
        methods.append(dict(method=arm,label_KO=LABELS[arm],populations=populations,
            source_output=str(V.OUTPUT.relative_to(C.ROOT)),current_default=False,
            original_TEST49_all_translation_le5=populations['TEST']['translation_le5']==49,
            numerical_primary192_target_met=populations['PRIMARY192']['translation_le5']==192,
            intervention_contract=dict(source_policy=source_policy,new_actual_image_line_generation='V29',
                fresh_same_CAL_REST_producer=True,source_specific_CAL_line_and_tangent=True,
                FULL_translation_prior_weight=0.,containment_weight=1. if arm.endswith('_CONTAINMENT') else 0.,
                new_owner_old_partial_locality_replaced_by_original_support_and_immutable_order=True,
                no_precise_new_corner_or_U_S_image_point=True,old_full_mask_tie=False,e_numeric_prior_or_fallback=False)))
    if methods[:29]!=previous['methods'] or len(methods)!=33:raise ValueError('Previous29 exact plus four new controls')
    paired=[]
    for definition in _paired_definitions():
        a,b=definition['before'],definition['after']
        parts={part:COMMON.PAIR._paired(reports[part]['rows'],reports[part]['rows'],a,b) for part in P.COUNTS}
        primary=reports['TEST']['rows']+reports['REMAINDER']['rows']
        parts['PRIMARY192']=COMMON.PAIR._paired(primary,primary,a,b)
        paired.append(dict(**definition,populations=parts))
    active=C.read(OUTPUT/'ACTIVE_GOAL_PROTOCOL_V1.json');P._verify(active['protected_user_sha256'])
    evidence=[Path(__file__),oldpath,oldseal,*auditpaths,V.PLAN]
    evidence.extend(V.OUTPUT/name for name in ('PROTOCOL.json','INPUT_SEAL.json','PREDICTION_SEAL.json',
                                            'SCORE_SEAL.json','CAL_SCORE.json','TEST_SCORE.json','REMAINDER_SCORE.json'))
    evidence.extend(V.CAL.OUTPUT/name for name in ('PROTOCOL.json','PREPARATION_SEAL.json','MODEL_SEAL.json','CAL_LOGO_REPORT.json'))
    bindings=[P._binding(path) for path in dict.fromkeys(evidence)]
    data=dict(schema='ACTIVE_REPAIR_CAMPAIGN_PROGRESS_V10_GEOMETRY_GUIDED_LINES_V29',
        generated_UTC=datetime.now(timezone.utc).isoformat(),goal_status='active',goal_completion_declared=False,
        default_changed=False,default_method=P.BASELINE,method_count=33,previous_method_count=29,new_method_count=4,
        population_table_rows=132,methods=methods,paired_control_count=8,paired_controls=paired,
        original_V9_methods_retained_exactly=True,primary=dict(TEST49=49,REMAINDER143=143,total=192,severe=0),separate_CAL=46,
        exact_baseline_consistency_verified=True,all_methods_uniform_full_population=True,
        numerical_target_met_any_new_method=any(m['numerical_primary192_target_met'] for m in methods[29:]),
        original49_preserved_by_new_method={m['method']:m['original_TEST49_all_translation_le5'] for m in methods[29:]},
        no_GT_ID_error_partition_runtime_routing=True,no_primary_native_compatibility_or_fallback_filter=True,
        original_v19_start_names_verified=True,no_e_numeric_pose_prior_or_fallback=True,
        S_image_2D_or_regions=0,U_regions_only=True,new_precise_2D_corners=0,
        actual_other_instance_occluder_masks_in_current_trial=0,targetmask_absence_is_unknown=True,
        old_global_symmetric_mask_or_tie_used_in_new_methods=False,
        no_new_reference_numeric_binding_changes=True,reused_exploratory_TEST_not_new_holdout=True,
        additional_reference_is_annotation_PnP_derived_not_independent_metrology=True,
        no_per_frame_best_method_mixture=True,no_new_images_labels_inference_or_optimization_in_accounting=True,
        independent_root_audits=[P._binding(auditpaths[0])],evidence=bindings,
        protected_user_sha256=active['protected_user_sha256'])
    C.write(paths[0],data);paths[1].write_text(_render(data),encoding='utf-8')
    P._verify(active['protected_user_sha256'])
    C.write(paths[2],dict(source=P._binding(Path(__file__)),json=P._binding(paths[0]),html=P._binding(paths[1]),
        evidence=bindings,method_count=33,population_table_rows=132,paired_control_count=8,
        original29_methods_retained_exactly=True,exact_e_preserved=True))
    print(dict(methods=33,table_rows=132,paired_controls=8,json_sha256=C.sha(paths[0])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--after-v29-score-and-root-audit-go',action='store_true',required=True)
    build(root_postscore_GO=parser.parse_args().after_v29_score_and_root_audit_go)
