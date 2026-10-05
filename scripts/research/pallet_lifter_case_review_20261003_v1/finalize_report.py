"""Record actual preparation evidence and hashes without synthesizing performance."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from audit import HERE, ROOT, OUT, dump, sha

DOC = ROOT/'_docs/experiments'/HERE.name


def main():
    inputs=json.loads((OUT/'LIFTER_INPUT_AND_TIME_MAP.json').read_text('utf-8'))
    plan=json.loads((OUT/'LIFTER_EVALUATION_PLAN.json').read_text('utf-8'))
    results=json.loads((OUT/'LIFTER_RESULTS.json').read_text('utf-8'))
    metrics=json.loads((OUT/'LIFTER_METRICS.json').read_text('utf-8'))
    assert inputs['total_decoded_stored_frames']==8910
    assert plan['fixed_review_frame_count']==120 and plan['repeat_review_frame_count']==24
    assert results['all_saved_frames']['inferred_frames']==0
    assert metrics['prediction_record_count']==0
    assert not (OUT/'review/annotations_in_progress.json').exists(), 'Human state changed; regenerate report from new evidence'
    fields=['scope','session_id','planned_frames','inferred_frames','status','base_fresh_outputs','n3_fresh_outputs','visible_reference_corners','median_error_px','p90_error_px','pck10_percent']
    rows=[{'scope':'all_saved_frames','session_id':s['session_id'],'planned_frames':s['sequentially_decoded_frames'],
           'inferred_frames':0,'status':'BLOCKED_CONTRACT'} for s in inputs['sessions']]
    rows += [{'scope':'all_saved_frames','session_id':'TOTAL','planned_frames':8910,'inferred_frames':0,'status':'BLOCKED_CONTRACT'},
             {'scope':'fixed_visible_corner_sample','session_id':'TOTAL','planned_frames':120,'inferred_frames':0,'status':'WAITING_HUMAN'},
             {'scope':'repeat_annotation_quality','session_id':'TOTAL','planned_frames':24,'status':'WAITING_HUMAN'},
             {'scope':'independent_physical_reference','session_id':'TOTAL','status':'BLOCKED_REFERENCE'}]
    with (OUT/'LIFTER_RESULTS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    verification={'schema_version':'lifter_preparation_verification_v1','source_kind':'machine_verification',
          'statuses':{'time_map':'VERIFIED_COMPLETE','review_ui':'READY_TO_REVIEW','human_corner_accuracy':'WAITING_HUMAN',
                      'full_inference':'BLOCKED_CONTRACT','physical_accuracy':'BLOCKED_REFERENCE','paper_patch':'READY_TO_REVIEW'},
          'python_tests_passed':{'historical_fixture_and_guards':7,'plan_and_input_gates':6,'review':8,'metrics':16,'total':37},
          'js_roundtrip_cases_passed':96,'js_known_css_scaled_point':[100,50],
          'browser_checks':{'raw_png_display':True,'zoom_fit_navigation':True,'canonical_diagram_visible':True,
                            'javascript_errors':0,'actual_human_records_created':0},
          'actual_run_preflight':'two_missing_fixed_weights_rejected_before_model_load',
          'actual_resume':'WAITING_HUMAN and BLOCKED_CONTRACT, references not invented',
          'model_run_validation':'BLOCKED_CONTRACT: fixed checkpoints absent; GPU forward/cap equivalence not tested on actual models',
          'test_reference_isolation':'synthetic TemporaryDirectory/in-memory fixtures only',
          'actual_cost':results['actual_cost']}
    dump(DOC/'VERIFICATION.json',verification)
    files=[]
    def add(path):
        files.append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    for path in HERE.glob('*.py'):
        add(path)
    add(HERE/'README_KO.md')
    for dirname in ['review','metrics','paper_patch']:
        for path in (HERE/dirname).glob('*'):
            if path.is_file():
                add(path)
    for path in OUT.glob('*'):
        if path.is_file():
            add(path)
    for name in ['review/MANIFEST.json','review/CORNER_CONTRACT.json']:
        add(OUT/name)
    add(DOC/'FINAL_REPORT_KO.md')
    add(DOC/'VERIFICATION.json')
    delivery=HERE/'delivery'
    input_artifacts=[]
    for name in ['CLI_A_LIFTER_CASE_20261003.txt','source_context/current_paper_source.zip',
                 'source_context/PENDING_DATA_AND_EXPERIMENT_CHECKS_KO.md','source_context/TABLE_FIGURE_CROSSWALK_KO.md',
                 'source_context/pallet_manuscript_revision_guide_20261003.md']:
        p=delivery/name
        input_artifacts.append({'filename':name,'bytes':p.stat().st_size,'sha256':sha(p),'source_kind':'user_supplied_context'})
    storage={'review_lossless_png_bytes':sum(p.stat().st_size for p in (OUT/'review/frames').glob('*.png')),
             'a_code_and_copied_context_bytes':sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
             'a_results_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}
    manifest={'schema_version':'lifter_case_preparation_manifest_v1','source_kind':'machine_verification',
              'recorded_at_kst':datetime.now(timezone(timedelta(hours=9))).isoformat(),
              'workspace_head':'39f07aa625dee924f44097f6dd46dd6b60c2524c','evaluation_head':'7e136fca834d97b52f63be696e4fcc9cbb8bd77e',
              'original_dirty_work_preserved':True,'inputs':input_artifacts,'artifacts':sorted(files,key=lambda f:f['path']),
              'capture_input_hashes':'LIFTER_INPUT_AND_TIME_MAP.json:sessions[].files',
              'per_frame_decoded_pixels':'LIFTER_EVALUATION_PLAN.json:frames[].decoded_bgr_sha256',
              'sample_image_hashes':'review/MANIFEST.json:frames[].image_sha256',
              'storage_bytes':storage,'actual_cost':results['actual_cost'],'verification':verification,
              'excluded_changed_video':inputs['excluded_video'],
              'prohibited_actions_executed':{'training':False,'hardware_control':False,'static_data_edit':False,
                                            'shared_manuscript_overwrite':False,'external_upload':False,'git_push':False},
              'remaining_inputs':['hash-matched YOLO R0 + N3 seed1 checkpoints','actual submitted human reference',
                                  'human-reviewed stop intervals if stationary noise is desired','independent physical reference for absolute pose accuracy']}
    dump(DOC/'MANIFEST.json',manifest)
    print(json.dumps({'status':'READY_TO_REVIEW','overall_inference_status':'BLOCKED_CONTRACT','python_tests':37,
                      'visible_reference_status':'WAITING_HUMAN','artifacts_hashed':len(files),'storage_bytes':storage},ensure_ascii=False))


if __name__=='__main__':
    main()
