"""Post-seal campaign accounting, without inference, fitting or GT decoding.

Every table row is one uniform method.  Primary192 is the concatenation of
TEST49 and remaining143, never a per-frame best-method or oracle mixture.
Run only after the continuous238 prediction AND score seals are complete.
Existing campaign documents and all experiment artifacts are immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path

import numpy as np

from . import common as C


BOUNDARY=C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
REMAINDER=C.DOC/'nonsevere_v1/remainder_after_test49_v1'
OUTPUT=REMAINDER/'repair_campaign_20261015_v1'
BASELINE='SAME_FROZEN_E'
RUNS=(
    ('old_normal_t4_only_candidates_v4_18',(
        ('OLD_NORMAL_T4_NATIVE','법선·T4'),('OLD_NORMAL_T4_PLUS_MASK','법선·T4+마스크'))),
    ('image_chain_mixture_candidates_v4_18',(
        ('NMS_NATIVE_D_MERGED','NMS'),('NMS_PLUS_MASK_D_MERGED','NMS+마스크'))),
    ('grazing_native_candidates_v4_18',(
        ('GRAZING_U_NATIVE_DMERGED','grazing'),('GRAZING_U_NATIVE_PLUS_MASK_DMERGED','grazing+마스크'))),
    ('grazing_image_chain_mixture_candidates_v4_18b',(
        ('GRAZING_NMS_NATIVE_D_MERGED','grazing·NMS'),('GRAZING_NMS_PLUS_MASK_D_MERGED','grazing·NMS+마스크'))),
    ('refined_grazing_image_chain_mixtures_v4_18c',(
        ('REFINED_GRAZING_NMS_NATIVE','연속 grazing·NMS'),('REFINED_GRAZING_NMS_PLUS_MASK','연속 grazing·NMS+마스크'))),
)
COUNTS={'CAL':46,'TEST':49,'REMAINDER':143}


def _binding(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(C.ROOT)),sha256=C.sha(path),bytes=path.stat().st_size)


def _verify(mapping):
    for relative,digest in mapping.items():
        if C.sha(C.ROOT/relative)!=digest:raise ValueError('Bound bytes changed: '+relative)


def _distribution(values):
    a=np.asarray(values,float)
    if not len(a):return dict(n=0,mean=None,median=None,P90=None,std=None,max=None)
    if not np.isfinite(a).all():raise ValueError('All original finite translation/rotation metrics required')
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),P90=float(np.percentile(a,90)),
        std=float(a.std(ddof=1))if len(a)>1 else None,max=float(a.max()))


def _summary(rows,arm):
    metrics=[r['metrics'][arm]for r in rows]
    translations=[m['translation_cm']for m in metrics]
    rotations=[m['rotation_deg']for m in metrics if m['rotation_deg']is not None]
    return dict(translation_cm=_distribution(translations),rotation_deg=_distribution(rotations),
        translation_le5=sum(t<=5 for t in translations),
        successes_5cm_5deg=sum(bool(m['success_5cm_5deg'])for m in metrics),
        rotation_unresolved=len(metrics)-len(rotations),
        fallback_count=sum(not r['accepted'][arm]for r in rows),
        max_translation_id=rows[int(np.argmax(translations))]['id'])


def _same_summary(actual,stored):
    for key in ('translation_le5','successes_5cm_5deg','rotation_unresolved','fallback_count','max_translation_id'):
        if actual[key]!=stored[key]:raise ValueError('Score count/maximum ID differs: '+key)
    for group in ('translation_cm','rotation_deg'):
        for key,value in actual[group].items():
            other=stored[group][key]
            if value is None:
                if other is not None:raise ValueError('Unresolved rotation summary differs')
            elif not np.isclose(value,other,atol=1e-10,rtol=1e-12):
                raise ValueError('Independent cached-score aggregation differs: '+group+'.'+key)


def _load_run(folder):
    protocol=C.read(folder/'PROTOCOL.json');pred=C.read(folder/'PREDICTION_SEAL.json');score=C.read(folder/'SCORE_SEAL.json')
    if (pred['rows']!=238 or not pred['all3_arms_complete']or not pred['before_new_control_metric_read']
            or C.sha(folder/'PREDICTIONS.jsonl.gz')!=pred['predictions_sha256']
            or C.sha(folder/'PROTOCOL.json')!=pred['protocol_sha256']
            or C.sha(folder/'INPUT_SEAL.json')!=pred['input_seal_sha256']
            or C.sha(folder/'PREDICTION_SEAL.json')!=score['prediction_seal']['sha256']):
        raise ValueError('All238 prediction/score seals required: '+str(folder))
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify(protocol[key])
    compatibility=folder/'SCORER_COMPATIBILITY_RECEIPT.json'
    if compatibility.exists():
        receipt=C.read(compatibility);adapter=C.read(folder/'SCORER_COMPATIBILITY_SEAL.json')
        if (C.sha(compatibility)!=adapter['receipt_sha256']
                or receipt['predictions_unchanged_sha256']!=pred['predictions_sha256']
                or receipt['score_seal']['sha256']!=C.sha(folder/'SCORE_SEAL.json')
                or receipt['field_used_in_numeric_metrics']is not False
                or receipt['numeric_metric_and_reference_code_unchanged']is not True):
            raise ValueError('Metadata-only scorer compatibility closure required')
    reports={}
    for part,n in COUNTS.items():
        path=folder/(part+'_SCORE.json')
        if C.sha(path)!=score['score_sha256'][part]:raise ValueError('Bound post-seal score changed')
        report=C.read(path)
        if report['count']!=n or len(report['rows'])!=n or len({r['id']for r in report['rows']})!=n:
            raise ValueError('No frame/fallback/native-flag filtering allowed')
        reports[part]=report
    ordered=[r['id']for part in COUNTS for r in reports[part]['rows']]
    if ordered!=protocol['ids']:raise ValueError('Original ordered238 inventory changed')
    return protocol,reports


def _evidence():
    paths=[
        REMAINDER/'wd_convention_source_audit_v4_18/SOURCE_AUDIT_V1.json',
        REMAINDER/'wd_convention_source_audit_v4_18/CACHED_PHASE_CUE_INVENTORY_V1.json',
        REMAINDER/'wd_convention_source_audit_v4_18/POSTSEAL_BANK_AND_WD_DIAGNOSTIC_V1.json',
        REMAINDER/'wd_convention_source_audit_v4_18/FRESH_FEASIBLE_WD_AMBIGUITY_V1.json',
        REMAINDER/'cached_coverage_failure_diagnosis_v4_18/R24_MISSING_COVERAGE_DIAGNOSIS_V1.json',
        REMAINDER/'cached_coverage_failure_diagnosis_v4_18/O_DOMAIN_AND_EMPTY_BANK_DETAILS_V1.json',
        BOUNDARY/'CAL_native_edge_mixture_v4_18/MODEL.json',
        BOUNDARY/'CAL_native_edge_mixture_v4_18/MODEL_SEAL.json',
        BOUNDARY/'CAL_grazing_native_edge_mixture_v4_18b/MODEL.json',
        BOUNDARY/'CAL_grazing_native_edge_mixture_v4_18b/MODEL_SEAL.json',
        BOUNDARY/'old_normal_t4_only_candidates_v4_18/INDEPENDENT_OFF_OBJECTIVE_CHECK_V1.json',
        BOUNDARY/'old_normal_t4_only_candidates_v4_18/PAIRED_NMS_ON_OFF_ATTRIBUTION_V1.json',
        BOUNDARY/'independent_continuous_grazing_mixture_review_v4_18c/REVIEW.json',
        BOUNDARY/'independent_continuous_grazing_mixture_review_v4_18c/REVIEW_SEAL.json',
        BOUNDARY/'refined_grazing_mixture_process_v4_18c/RENDER_SEAL.json',
    ]
    for name,_ in RUNS:
        folder=BOUNDARY/name
        for receipt in ('EXECUTION_RECEIPT_V1.json','SCORER_COMPATIBILITY_RECEIPT.json','SCORER_COMPATIBILITY_SEAL.json'):
            path=folder/receipt
            if path.exists():paths.append(path)
        paths.extend(sorted((folder/'independent_root_audit_v1').glob('*.json')))
    return [_binding(p)for p in paths if p.exists()]


def _link(path,output):
    return os.path.relpath(path,output)


def _markdown(data,output):
    lines=['현재 기본 방법 e는 그대로 유지합니다. 원본 TEST49의 49/49는 달성했지만, 추가143장을 포함한 전체192장의 5cm 목표는 계속 진행 중입니다. v18 통제는 모두 같은 원본 사진·기하·저장 자세 후보를 사용했고, 표는 사진별 최적 방법을 섞지 않은 단일 방법 결과입니다.','',
        '표의 오차 단위는 cm이며 `공동`은 5cm·5°를 모두 만족한 수입니다. CAL46은 학습/진단 집합으로 primary192에 합치지 않았습니다. 모든 실패·fallback·원래 native 대응 플래그를 유지했습니다.','']
    for key,title in [('PRIMARY192','Primary192 = TEST49 + 추가143'),('TEST','기존 TEST49'),('REMAINDER','추가143'),('CAL','별도 CAL46')]:
        lines.extend([title,'','| 방법 | T≤5cm | 공동 | 평균 T | 최대 T | fallback |','|---|---:|---:|---:|---:|---:|'])
        for method in data['methods']:
            s=method['populations'][key];lines.append('| '+method['label_KO']+' | '+str(s['translation_le5'])+' | '+str(s['successes_5cm_5deg'])+' | '+f"{s['translation_cm']['mean']:.3f}"+' | '+f"{s['translation_cm']['max']:.3f}"+' | '+str(s['fallback_count'])+' |')
        lines.append('')
    lines.extend([
        'OFF는 기존 법선·T4를 유지하고 NMS 및 기존 projected-edge 각도 비용을 끕니다. ON은 같은 입력에서 실제 NMS 선의 모든 후보를 정규화 혼합으로 넣습니다. 독립 grazing 행은 v17의 기존 각도 모델을 유지합니다. grazing은 초기 비S B/I 중 기존 2° 기하 허용대에 놓인 31점을 U 범위로 바꾸며, S와 기존 U/O/T는 유지합니다. 결합 통제는 그 상태를 먼저 적용한 뒤 실제 RGB를 새로 조회한 NMS 기록과 별도 CAL 모델을 사용합니다. 연속 통제만 최대 6개 저장 시작점과 coarse에서 추가 6차원 자세 최적화를 수행하며, 기존 후보도 전부 보존합니다. 이는 전역 최적·유일성 증명이 아닙니다.','',
        'CAL 모델은 고정 대응이 맞는 44장 중 적용 가능한 19장·22모서리·36후보에서 법선 σ≈6.451px, 사인각 σ≈0.01938을 추정했습니다. 기존 및 grazing 모델의 수치가 같아도 출처·상태 프로필·모델 봉인은 별도입니다. ρ=0과 픽셀/지지길이 하한은 공학적 근사이며 독립 관측·정확한 코너·향후 신뢰구간을 뜻하지 않습니다. wood 제외 LOGO의 선택 시작점 `success=False` 상태도 보존되어 있어 모든 fold의 수렴을 주장하지 않습니다.','',
        '원인 점검에서는 native 채널이 학습 사진에 따라 물리 X/Z에 대응하며, 저장된 Q는 앞선 PnP W/D 선택 결과임을 확인했습니다. 현재 캐시에 독립 W/D 판별 logits는 발견하지 못했습니다. 봉인된 원래 후보은행의 사후 GT-best 진단은 추가143장 중 119장에만 5cm 이내 후보가 있고 24장은 없음을 보였습니다(그중 4장은 빈 은행). 이는 새 연속 후보나 미래 방법의 한계가 아니며, GT-best를 실제 선택기로 쓰지 않았습니다.','',
        'O 원은 관측한 가림 물체 마스크가 아니라 raw N3 위치 예측의 허용 범위입니다. CAL의 O 예시는 5장·7점이고 모두 plastic이며 wood 예시는 없습니다. 추가143장 중 5장의 정확한 명목 참조 자세가 고정 O 범위를 포함한 hard guard를 통과하지 못했습니다. 이는 5cm 근방의 모든 자세가 불가능하다는 증거가 아니며, 이 진단으로 범위나 매개변수를 GT에 맞춰 바꾸지 않았습니다.','',
        '추가143장의 참조는 저장된 annotation/PnP에서 유도한 값으로 독립적인 물리 계측이 아닙니다. 원래 canonical 축/카메라 관례의 미확정 사항과 C2 정책을 유지했으며, 재사용 TEST에서 반복 진단한 탐색 결과입니다. 새 기본 방법 채택이나 전체 목표 완료를 선언하지 않습니다.','',
        '연속 통제의 원래 score 진입점은 `tangent_control_audit` 메타데이터 누락으로 실패했습니다. 별도 scorer 어댑터가 누락 표시만 메모리에서 추가했고, 예측 파일·수치 입력·참조·점수 계산식은 바꾸지 않았습니다. 성공한 채점은 이 어댑터와 호환성 봉인을 기준으로 기록했습니다.','',
        '실제 사진/과정과 독립 검산:',''])
    for report in data['actual_reports']:
        lines.append('- ['+report['label_KO']+']('+_link(C.ROOT/report['binding']['path'],output)+')')
    lines.extend(['','전체 회전 오차·집합별 최대 사례·봉인/소스/검산 근거는 [PROGRESS_V3_V18.json](PROGRESS_V3_V18.json)에 있습니다.'])
    return '\n'.join(lines)+'\n'


def build(output=OUTPUT):
    output=Path(output).resolve();json_path=output/'PROGRESS_V3_V18.json';md_path=output/'RESULTS_V3_V18.md'
    if json_path.exists()or md_path.exists():raise FileExistsError('Append-only campaign version: preserve existing V3 documents')
    # The continuous score is a hard gate before any aggregate output.
    if not (BOUNDARY/RUNS[-1][0]/'SCORE_SEAL.json').exists():
        raise ValueError('Wait for full continuous238 prediction+score seals and root notification')
    active=C.read(output/'ACTIVE_GOAL_PROTOCOL_V1.json');_verify(active['protected_user_sha256'])
    methods=[];bindings=[];baseline_rows=None;canonical_ids=None;audited_runs=[];protocols=[]
    for name,arms in RUNS:
        folder=BOUNDARY/name;protocol,reports=_load_run(folder);protocols.append(protocol)
        ids={p:[r['id']for r in reports[p]['rows']]for p in COUNTS}
        if canonical_ids is None:canonical_ids=ids
        elif ids!=canonical_ids:raise ValueError('Same complete46/49/143 required in every control')
        current_baseline={p:[(r['metrics'][BASELINE],r['accepted'][BASELINE])for r in reports[p]['rows']]for p in COUNTS}
        if baseline_rows is None:baseline_rows=current_baseline;selected=((BASELINE,'기존 e'),)+arms
        else:
            if current_baseline!=baseline_rows:raise ValueError('Archived e baseline changed between controls')
            selected=arms
        for arm,label in selected:
            populations={}
            for part in COUNTS:
                populations[part]=_summary(reports[part]['rows'],arm)
                _same_summary(populations[part],reports[part]['summary'][arm])
            populations['PRIMARY192']=_summary(reports['TEST']['rows']+reports['REMAINDER']['rows'],arm)
            methods.append(dict(method=arm,label_KO=label,output=str(folder.relative_to(C.ROOT)),
                populations=populations,adopted=arm==BASELINE,
                numerical_primary192_target_met=populations['PRIMARY192']['translation_le5']==192,
                original_TEST49_preserved=populations['TEST']['translation_le5']==49))
        for filename in ('PROTOCOL.json','INPUT_SEAL.json','PREDICTION_SEAL.json','SCORE_SEAL.json',
                'CAL_SCORE.json','TEST_SCORE.json','REMAINDER_SCORE.json'):
            bindings.append(_binding(folder/filename))
        audited_runs.append(name)
    reports=[]
    for relative,label in [('native_edge_actual_process_report_v4_17/index.html','v17 실제 사례'),
            ('NMS_ON_OFF_actual_process_report_v4_18/index.html','NMS ON/OFF 실제 과정'),
            ('refined_grazing_mixture_process_v4_18c/index.html','grazing·NMS 저장 후보와 연속 최적화 실제 과정')]:
        path=BOUNDARY/relative
        if path.exists():reports.append(dict(label_KO=label,binding=_binding(path)))
    evidence=_evidence()
    wd=C.read(REMAINDER/'wd_convention_source_audit_v4_18/SOURCE_AUDIT_V1.json')
    bank=C.read(REMAINDER/'wd_convention_source_audit_v4_18/POSTSEAL_BANK_AND_WD_DIAGNOSTIC_V1.json')
    cues=C.read(REMAINDER/'wd_convention_source_audit_v4_18/CACHED_PHASE_CUE_INVENTORY_V1.json')
    missing=C.read(REMAINDER/'cached_coverage_failure_diagnosis_v4_18/R24_MISSING_COVERAGE_DIAGNOSIS_V1.json')
    occlusion=C.read(REMAINDER/'cached_coverage_failure_diagnosis_v4_18/O_DOMAIN_AND_EMPTY_BANK_DETAILS_V1.json')
    render_seal=C.read(BOUNDARY/'refined_grazing_mixture_process_v4_18c/RENDER_SEAL.json')
    _verify(render_seal['input_sha256'])
    data=dict(schema='ACTIVE_REPAIR_CAMPAIGN_PROGRESS_V3_V18',generated_UTC=datetime.now(timezone.utc).isoformat(),
        generation_source=_binding(Path(__file__)),goal_status='active',goal_achieved=False,
        objective='one uniform inference method, all primary192 translation<=5cm and original49 preserved',
        default_method=BASELINE,default_changed=False,no_per_frame_best_method_or_GT_router=True,
        primary=dict(TEST49=49,REMAINDER143=143,total=192,severe=0),CAL46_separate=True,
        methods=methods,numerical_target_met_any_uniform_method=any(m['numerical_primary192_target_met']for m in methods),
        exact_baseline_consistency_verified_across_runs=True,all238_no_frame_or_fallback_filter=True,
        all_source_input_and_protected6_current_PASS=True,run_bindings=bindings,
        source_binding_scope='conservative inherited dependency unions; not every listed module executes per method',
        source_parity_findings=wd['conclusion'],independent_parity_signal_status=cues['independent_parity_signal_status'],
        precontinuous_cached_bank_posthoc_upper_bound=bank['summary'],
        missing_cached_coverage=missing['summary'],
        O_domain_source_findings=dict(O_model=occlusion['O_model'],source_semantics=occlusion['source_semantics'],
            calibration_points=7,calibration_frames=5,calibration_materials=dict(plastic=7,wood=0),
            exact_nominal_reference_hardfail_frames=len(occlusion['five_O_hardfail_frames']),
            no_five_cm_neighborhood_impossibility_claim=True,no_GT_adjusted_parameters=True),
        coverage_is_diagnostic_not_runtime_selector_and_not_continuous_ceiling=True,
        independent_audit_and_review_evidence=evidence,actual_reports=reports,
        continuous_scoring=dict(original_entrypoint_success=False,
            original_failure='KeyError tangent_control_audit; no score reports written',
            actual_adapter_source=_binding(C.ROOT/'scripts/research/pallet_visibility_curriculum_20261013_v1/score_refined_grazing_mixtures_v4_18c.py'),
            compatibility_receipt=_binding(BOUNDARY/RUNS[-1][0]/'SCORER_COMPATIBILITY_RECEIPT.json'),
            compatibility_seal=_binding(BOUNDARY/RUNS[-1][0]/'SCORER_COMPATIBILITY_SEAL.json'),
            metadata_only=True,numeric_metric_and_reference_code_unchanged=True),
        reference_caveats=['Original fixed95 scorer and unchanged derived143 canonical/C2 reader retained',
            'Remaining143 reference is derived annotation/PnP, not independent physical metrology',
            'Original native-index flags and unconfirmed axis/camera conventions do not filter rows',
            'TEST reused exploratory; no claim of new holdout generalization'],
        invariant_contract=['Initial S model-only; no S raw/image queries','U approximate unclipped regions only',
            'Retained B/I q,covariance,reliability immutable','All old OT/U/tangent/depth/canvas/geometry guards retained',
            'No GT/frame-ID/error-based runtime method routing; no new default'],
        script_prepared_before_continuous_results=True,
        final_evidence_links_added_after_continuous_score_and_root_GO=True,new_inference_or_pose_fits=0)
    for protocol in protocols:
        for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify(protocol[key])
    for binding in bindings+evidence+[r['binding']for r in reports]:
        if C.sha(C.ROOT/binding['path'])!=binding['sha256']:raise ValueError('Evidence changed during aggregation')
    output.mkdir(parents=True,exist_ok=True);C.write(json_path,data)
    md_path.write_text(_markdown(data,output),encoding='utf-8')
    _verify(active['protected_user_sha256'])
    C.write(output/'PROGRESS_V3_V18_SEAL.json',dict(source=_binding(Path(__file__)),
        json=_binding(json_path),markdown=_binding(md_path),run_bindings=bindings,
        evidence=evidence,protected_user_sha256=active['protected_user_sha256'],
        all_source_input_score_seals_checked=True,no_new_reference_decodes_or_fits=True))
    print(dict(methods=len(methods),all_runs=audited_runs,json_sha256=C.sha(json_path),markdown_sha256=C.sha(md_path)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--after-continuous-score',action='store_true',required=True)
    args=parser.parse_args();build(args.output)
