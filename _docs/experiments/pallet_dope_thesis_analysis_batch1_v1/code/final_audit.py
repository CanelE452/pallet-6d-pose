"""Final deliverable/constraint checks; no neural inference or raw data reads."""
from common import *
from collections import Counter
import ast

def rows(name):
    with (OUT/'tables'/name).open() as f:return list(csv.DictReader(f))

def main():
    start('final_requirement_audit')
    checks={}
    initial=read(OUT/'preflight/INITIAL_GATES.json')
    assert initial['initial_file_count_gate_pass'] and initial['summary_count_reproduction_exact']
    checks['initial_file_and_summary_gate']=True
    authorization=read(OUT/'preflight/GATE5_RESOLUTION.json')
    assert authorization['status']=='RESOLVED_BY_EXPLICIT_USER_AUTHORIZATION'
    assert authorization['user_answer']=='기존 파생 SCORES 사용만 허용하고 push'
    assert not authorization['raw_protected_FINAL_RGB_opening_allowed']
    assert not authorization['raw_protected_FINAL_annotation_body_opening_allowed']
    for rel,expected in authorization['frozen_quantitative_outputs'].items():
        assert sha(OUT/rel)==expected,('Frozen CSV changed',rel)
    checks['Gate5_existing_derived_scores_only_user_authorized']=True
    checks['quantitative_CSV_unchanged_since_authorization']=18
    metadata=read(OUT/'cache/METADATA_RECORDINGS.json')
    assert len(metadata['protected_final_frame_ids'])==229
    assert len(metadata['protected_final_image_sha256'])==229
    bias=read(OUT/'cache/T1_NATIVE_BIAS.json')
    assert bias['complete'] and len(bias['rows'])==21 and bias['error_coordinates_checked']==14252
    checks['T1a_21rows_14252_coordinate_errors_reproduced']=True
    checks['T1b_status']='SKIPPED_UNPROVEN_ORIGINAL_INFERENCE'
    boot=rows('T2_PRIMARY_CI.csv')+rows('T2_GREEN_MANUAL_CI.csv')+rows('T2_PLASTIC184_CI.csv')
    assert len(boot)==300 and sum(r['bootstrap_status']=='COMPUTED' for r in boot)==150
    assert all(r['seed']=='20261026' and r['bootstrap_requested']=='10000' for r in boot)
    for r in boot:
        if r['unit']=='recording':
            assert r['bootstrap_status'].startswith('NOT_COMPUTED_') and not r['ci_low_pp'] and not r['ci_high_pp']
        else:
            assert r['bootstrap_defined']=='10000' and r['bootstrap_undefined']=='0'
            lo,hi=float(r['ci_low_pp']),float(r['ci_high_pp'])
            assert r['decision']==('높았다' if lo>0 else '낮았다' if hi<0 else '차이 불명확')
    checks['T2_frame_CI_calculated']=150;checks['T2_recording_CI_not_calculated']=150
    eq=read(OUT/'preflight/T3_FILTER_EQUIVALENCE.json')
    assert eq['complete'] and eq['pass_all'] and eq['reference_coordinates_read'] is False
    for mode in ('main','noflip'):
        c=eq['checked'][mode]
        assert c['frames']==876 and c['nine_channel_comparisons']==7884 and c['candidate_frames']==1395
        assert not any(c[k] for k in ('frame_mismatches','mask_mismatches','channel_mismatches','candidate_mismatches'))
    checks['T3_equivalence']=eq['checked']
    stages=read(OUT/'tables/T3_FILTER_STAGES.json')
    assert len(stages['rows'])==132 and stages['annotations_opened'] is False and stages['thresholds_changed'] is False
    assert stages['equivalence_receipt']['sha256']==sha(OUT/'preflight/T3_FILTER_EQUIVALENCE.json')
    assert all(r['annotated_eligible_selected_ref_count']==r['selected_ref_observed_count'] for r in stages['rows'])
    for tau in (5,10,20):assert (OUT/f'tables/T3_DETAIL_{tau}PX.md').is_file()
    checks['T3_stage_rows_primary_and_manual']=132
    pnp=rows('T4_PNP_AVAILABILITY.csv')
    assert len(pnp)==45 and len(rows('T4_PNP_AVAILABILITY_FRAMES.csv'))==4380
    assert Counter(r['scope'] for r in pnp)==Counter(primary_own_kind_or_shared=21,auxiliary_cross_kind=24)
    assert all(r['frames_detected_ge6']==r['frames_PnP_possible'] for r in pnp)
    checks['T4_primary_21_auxiliary_24_no_new_forward']=True
    sel=read(OUT/'cache/T5_SELECTION.json');qa=read(OUT/'cache/T5_RENDER_QA.json')
    assert len(sel['cases'])==sel['total_unique_selected_images']==9 and sel['seed']==20261026
    assert len(rows('T5_ALL_FRAME_DELTAS.csv'))==292 and len(rows('T5_SELECTED_CASES.csv'))==9
    assert len(qa['renders'])==9
    for kind in KINDS:
        assert {c['category'] for c in sel['cases'] if c['kind']==kind}=={'max','median','min'}
    assert len(list((OUT/'figures').glob('*.png')))==len(list((OUT/'figures').glob('*.pdf')))==9
    for r in qa['renders']:
        assert sha(r['pdf'])==r['pdf_sha256'] and sha(r['render_png'])==r['render_sha256']
    t5complete=read(OUT/'cache/T5_COMPLETION.json')
    assert t5complete['complete'] and t5complete['visual_QA_pass'] and t5complete['visual_QA_images_opened']==18
    assert t5complete['marker_successful_invocations']==1 and not t5complete['selection_rule_changed']
    checks['T5_292_deltas_9_fixed_distinct_cases_PNG_and_PDF']=True
    assert len(rows('T6_CROSS_TYPE.csv'))==72
    assert len(rows('T6_CROSS_ONLY.csv'))==36
    checks['T6_existing_score_rows_total']=72;checks['T6_actual_cross_kind_rows']=36
    ledger=[json.loads(s) for s in (OUT/'logs/RUN_LOG.jsonl').read_text().splitlines()]
    forwards=[r for r in ledger if r['event']=='neural_forward']
    attempts=[r for r in ledger if r['event']=='neural_forward_attempt']
    assert len(forwards)==len(attempts)==1752
    assert sum(r['image_count'] for r in forwards)==1752 and all(r['task']=='T3' for r in forwards)
    keys={(r['model'],r['frame_id'],r['view']) for r in forwards}
    assert len(keys)==1752 and keys=={(r['model'],r['frame_id'],r['view']) for r in attempts}
    checks['neural_forwards_by_task']={'T1b':0,'T3':1752,'T4':0,'others':0}
    checks['retraining_threshold_hyperparameter_changes']=0
    comparison=read(OUT/'hashes/COMPARISON.json')
    assert comparison['all_unchanged'] and not comparison['differences']
    before=read(OUT/'hashes/SNAPSHOT_BEFORE.json');after=read(OUT/'hashes/SNAPSHOT_AFTER.json')
    assert before['roots']==after['roots'] and before['R0']==after['R0']
    assert sum(len(r) for r in before['roots'].values())==598
    checks['original_experiment_files_unchanged']=598;checks['R0_unchanged']=True
    audit=read(OUT/'preflight/INDEPENDENT_AUDIT_T0_T2_T6.json')
    assert audit['complete'] and not audit['material_gate_violations']
    assert audit['status']=='PASS_AFTER_EXPLICIT_DERIVED_SCORES_AUTHORIZATION_WITH_LIMITATIONS'
    assert all(audit[k] is True for k in ('audit_receipt_complete','publication_ready','overall_gate_pass'))
    assert audit['pending_scope']==[]
    passed={r['check'] for r in audit['checks'] if r['status']=='PASS'}
    assert {'T3_existing_equivalence_masks_and_precision_contract','T4_cached_PnP_availability_scope',
            'T5_fixed_selection_and_artifact_SHA'}<=passed
    t5review=read(OUT/'preflight/T5_FINAL_REVIEW.json')
    assert t5review['status']=='PASS' and t5review['fixed_selection']['exact']
    for path,expected in t5review['evidence'].items():assert sha(path)==expected,('T5 review binding changed',path)
    review=read(OUT/'preflight/REPORT_PUBLICATION_REVIEW.json')
    assert review['status']=='RESOLVED_NO_ACTIVE_BLOCKING_ISSUES' and review['publication_ready']
    for path,expected in review['bindings'].items():assert sha(path)==expected,('Publication review binding changed',path)
    checks['independent_full_scope_T0_T6_review_passed']=True
    checks['current_report_and_publication_safety_review_passed']=True
    for name in ('README.md','README_KO.md','REQUEST.md','FIXED_PLAN.json'):
        assert (OUT/name).is_file()
    codes={}
    for p in sorted((OUT/'code').glob('*.py')):
        ast.parse(p.read_text());codes[str(p)]=sha(p)
    assert not list(OUT.rglob('*.pth')) and not list(OUT.rglob('*.mp4'))
    write(OUT/'hashes/CODE_SHA256.json',codes)
    result=dict(analysis_complete=True,main_publication_status='pending_separate_verified_publication',
                checked_utc=now(),checks=checks,code_sha256_manifest='hashes/CODE_SHA256.json',
                limitations=['T1b unproven original inference path; not executed',
                             'All recording CI unavailable: missing/conflicting IDs or only2 recordings',
                             'Raw protected FINAL body unopened; existing derived SCORES explicitly authorized; no independent6D truth certification',
                             'Frame CI lacks recording correlation and multiple comparison adjustment'],
                T5_only_GT_case_selection_exception_authorized=True,
                Gate5_derived_SCORES_only_exception_authorized=True,
                authorization_receipt={'path':str(OUT/'preflight/GATE5_RESOLUTION.json'),'sha256':sha(OUT/'preflight/GATE5_RESOLUTION.json')},
                independent_review_receipts={str(OUT/'preflight'/name):sha(OUT/'preflight'/name) for name in
                    ('INDEPENDENT_AUDIT_T0_T2_T6.json','T5_FINAL_REVIEW.json','REPORT_PUBLICATION_REVIEW.json')},
                protected_FINAL_RGB_or_annotation_body_reads=0)
    write(OUT/'COMPLETION.json',result)
    text='# Batch1 최종 요구사항 검증\n\n[확인] 분석 산출물 검사 통과. T1(b)와 녹화 단위 CI는 근거/조건 부족으로 미실행·미계산 상태를 보존했다. main 공개는 별도 remote SHA 검증 대상이다.\n\n'
    text+=mdtable(['검사','결과'],[[k,json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v] for k,v in checks.items()])
    (OUT/'FINAL_AUDIT.md').write_text(text)
    log('step_complete',task='final_requirement_audit',analysis_complete=True,checks=checks)
    print('FINAL_REQUIREMENTS_PASS',len(checks),'checks',flush=True)

if __name__=='__main__':main()
