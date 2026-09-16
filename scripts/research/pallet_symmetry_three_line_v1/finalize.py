"""Evidence-backed handoff: A awaits decision, B complete. Never label A as trained."""
import io,json,subprocess,unittest
from collections import Counter
import torch
import env as E

def main():
    # Never overwrite completed/resumed A with this historical pre-training report.
    if (E.DOC/'A/MAIN_TRAINING_COMPLETE.json').exists():
        from a_finalize import main as completed_main
        return completed_main()
    torch.set_num_threads(4)
    binding=E.read(E.DOC/'SOURCE_BINDING.json')
    for r in binding['files']:
        path=E.Path(r['path']);path=path if path.is_absolute() else E.ROOT/path
        assert E.sha(path)==r['sha256'],r['path']
    assert E.sha(E.HERE/'audit_math.py')=='d4526e0babe44bd31c7df02cfa9a697c57b7e78e8356b34849666e4baf44f546'
    assert E.sha(E.HERE/'test_audit_math.py')=='6c8bab79a2ab81503baece2efe1588fe66b34e76566da60d91621b31d2e594e3'
    provenance=E.read(E.DOC/'A/PROVENANCE_QA.json')
    assert provenance['input_QA_complete'] and not provenance['missing_or_changed_raw_images']
    amendment=E.read(E.DOC/'A/ANNOTATION_TARGET_AMENDMENT.json')
    assert E.sha(E.ROOT/amendment['target_view']['path'])==amendment['target_view']['sha256']
    assert len(amendment['changed_records'])==2
    assert E.read(E.DOC/'B/ADAPTER_TESTS.json')['PASS']
    assert E.read(E.DOC/'B/runtime.json')['complete'] and E.read(E.DOC/'B/POSE_SECONDARY.json')['complete']
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(E.HERE),pattern='test_*.py'))
    assert result.wasSuccessful()
    E.write(E.DOC/'REGRESSION_TESTS.json',dict(cpu_tests=result.testsRun,cpu_PASS=True,log=stream.getvalue(),
      A_identity=E.bound(E.DOC/'A/IDENTITY_AND_RLE_AUDIT.json'),B_actual_cache=E.bound(E.DOC/'B/ADAPTER_TESTS.json'),
      A_EQUIV='NOT_IMPLEMENTED_PENDING_OBJECTIVE_DECISION',A_main_training_completed=False))
    train=dict(status='AWAITING_USER_OBJECTIVE_DECISION',reason='Stock RLE batch-global clamp conflicts with independent object-level complete-loss min; joint branch selection needs approval.',
      main_fits=0,main_updates=0,max_authorized_fits=12,max_authorized_updates=24000,smoke_optimizer_updates=0,
      diagnostic_detector_forward_images=dict(RECT=16,SQUARE=16),optimizer_created=False,
      changed_loss_or_omitted_RLE=False,actual_A_model_evaluations=0,geometry_gain='NOT_EVALUATED')
    E.write(E.DOC/'A/training_audit.json',train)
    E.write(E.DOC/'A/results_and_intervals.json',dict(status='NOT_EVALUATED',RECT=None,SQUARE=None,
      fits=0,primary_intervals=None,definition_math_and_geometry='PASS',EQUIV_wiring='PENDING',reason=train['reason']))
    # Explicit non-result table, not blank fabricated seed measurements.
    (E.DOC/'A/per_frame_results.csv').write_text('stage,status,model_evaluations,reason\nA_MAIN,NOT_RUN,0,AWAITING_USER_OBJECTIVE_DECISION\n')
    orbits={};available={};null_bias_changed=0;allseed=True
    for split in ['calibration','synth_val','DEV']:
        hist=Counter();count=0
        for path in sorted((E.RAW/f'B/cache/{split}').glob('*.pt')):
            c=torch.load(path,map_location='cpu',weights_only=False)
            if not c['P']:continue
            a=c['P'][1]['alignment'];hist[a['inference_orbit']]+=1;count+=1
            for s in [2,3]:allseed &= c['P'][s]['alignment']==a
            null_bias_changed+=int((c['P'][1]['biases']['B3_THREE_AMBIG'][:,-1]!=0).sum())
        orbits[split]={str(k):v for k,v in hist.items()};available[split]=count
    assert allseed
    E.write(E.DOC/'B/observation_and_role_alignment_audit.json',dict(PASS=True,orbit_counts=orbits,
      objects_with_line_observation=available,all_P_seeds_share_same_GT_free_alignment=True,
      source_checkpoint_SHA=E.DHT_SHA,line_path='stem -> line_head; full2340-bin posterior',WLS_calls=0,utility_calls=0,
      three_means='three distinct incident semantic edges, cuboid degree3',null_bias_changed_corners=null_bias_changed,
      GT_for_inference_orbit=False,evaluation_orbit_stored_separately=True,
      numerical_gate=E.bound(E.DOC/'B/ADAPTER_TESTS.json'),historical_batch_amendment=E.bound(E.DOC/'B/NUMERIC_LOCK_AMENDMENT.json')))
    b=E.read(E.DOC/'B/results_and_intervals.json');b.update(status='COMPLETE_FROZEN_B',
      DEV=E.bound(E.DOC/'B/DEV_RESULTS.json'),pose=E.bound(E.DOC/'B/POSE_SECONDARY.json'),runtime=E.bound(E.DOC/'B/runtime.json'))
    E.write(E.DOC/'B/results_and_intervals.json',b)
    runtime=E.read(E.DOC/'B/runtime.json');assert len(runtime['records'])==520
    run=dict(status='PARTIAL_AWAITING_A_OBJECTIVE_DECISION',A=train,
      A0=dict(square_checkpoint_forwards=775,paper_saved_prediction_rescores=3190,
        square_corrected_annotation_saved_prediction_rescores=775,new_training_updates=0),
      B=dict(status='COMPLETE',source_cache_frames=768,source_P_forwards=2304,source_DHT_forwards=768,
        real_image_R0_forwards=3008,real_P_forwards=5574,real_DHT_forwards=1858,
        calibration_predictions=45*256,source_arm_seed_predictions=15*512,real_arm_seed_predictions=15*3008,
        PnP_pose_frame_evaluations=15*(512+319),new_optimizer_updates=0,
        runtime_R0_forwards=808,runtime_P_forwards=808,runtime_DHT_forwards=606,
        numerical_DHT_diagnostic_forwards=512+12+96+512,
        CPU_scorer_attempts=3,additional_model_forwards_from_scorer_corrections=0,
        CPU_scorer_correction_notes=str((E.DOC/'B/SCORER_CORRECTION.json').relative_to(E.ROOT))),
      FINAL_access=0,external_notifications=0,reboot=False,driver_or_system_changes=False,
      original_paper_or_deployment_changed=False)
    E.write(E.DOC/'RUN_MANIFEST.json',run)
    assert subprocess.check_output(['git','diff','--name-only','--','_docs/paper'],text=True).strip()==''
    sources_unchanged=[r['path'] for r in binding['files']]
    E.write(E.DOC/'FINAL_AUDIT.json',dict(status='B_COMPLETE_A_PENDING_NOT_ALL_DONE',
      completion=dict(A0=True,A_MAIN=False,B=True),missing=['A_EQUIV shared branch objective decision','A_EQUIV adapter/wiring tests','A12 fits x2000 updates and new-model inference/statistics'],
      A_main_fits=0,A_main_updates=0,B_new_updates=0,CPU_tests=31,source_bindings_unchanged=sources_unchanged,
      core_SHA_preserved=True,paper_tracked_diff_empty=True,GT_or_source_cache_modified=False,
      square_original_prepared_label_mismatches=2,
      square_target_sidecar_amendment=E.bound(E.DOC/'A/ANNOTATION_TARGET_AMENDMENT.json'),
      large_weights_features_raw_images_not_for_Git=True,user_unrelated_changes_preserved=True,
      real_line_diagnostics_scope='Synthetic and DEV line-quality subgroup CSVs are separate. The 20 visual panels use synthetic images.',
      no_all_tasks_complete_claim=True))
    print('FINAL AUDIT: B complete; A0 complete; A main 0/12 awaiting decision',flush=True)
if __name__=='__main__':main()
