"""Measured phase ledger; all prior fits and inference remain reuse, not new work."""
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path

def read(p):return json.loads(p.read_text())
def run(args):
    doc=Path(args.public_dir);train=read(doc/'RGB_ONLY_TRAIN_RECEIPTS.json')
    selection=read(doc/'SYNTHETIC_SELECTION.json');cv=read(doc/'CVRANK_COMPUTE_RECEIPT.json')
    rgb=read(doc/'RGB_ONLY_EVALUATION_RECEIPT.json');stages=[read(p) for p in sorted(doc.glob('STAGE_*.json'))]
    GPU=sum(s['device_reserved_upper_seconds'] for s in stages)
    CPU=sum(s['wall_seconds'] for s in stages if s['device']=='cpu')
    # Metadata preparation is outside the wrapped classical stage; synthetic
    # reference scoring is also CPU analysis inside its reserved CUDA process.
    reuse=read(doc/'INPUT_REUSE_CONTRACT.json')
    repairs=read(doc/'IMPLEMENTATION_REPAIRS.json')
    metadata_seconds=reuse.get('wall_seconds',0.)
    feature_audit_seconds=read(doc/'FEATURE_EXCLUSION_AUDIT.json')['wall_seconds']
    CPU+=metadata_seconds+selection['CPU_scoring_seconds']+feature_audit_seconds
    # 13:30 UTC precedes the first task initialization and the preserved
    # 13:36:05 UTC dirty-worktree snapshot. Counting the whole elapsed interval
    # plus every measured CPU phase deliberately double-counts idle/GPU time
    # and covers the uninstrumented preparation repair and overlapping workers.
    conservative_start_epoch=args.analysis_start_epoch
    conservative_end_epoch=time.time()
    conservative_CPU_upper=conservative_end_epoch-conservative_start_epoch+CPU
    costs=[read(doc/f'LATENCY_{m}.json') for m in ('RGB_ONLY','RGB_ONLY_SUBPIX','CVRANK_NATIVE','CVRANK_CAP1','CVRANK_THEN_SUBPIX')]
    benchmark_F=sum(sum(c['counters'].get('F_calls',0) for c in r['per_call']) for r in costs)
    benchmark_OpenCV=sum(r['actual_OpenCV_total'] for r in costs)
    new_real_F=cv['actual_F_calls']+rgb['actual_F_calls']
    new_real_OpenCV=cv['total_OpenCV_calls']+rgb['total_OpenCV_calls']
    synthetic_F=selection['new_pose_F_calls'];synthetic_OpenCV=sum(selection['OpenCV_calls'].values())
    assert train['actual_updates']==6000 and train['fits']==2 and train['existing_model_fits']==0
    assert new_real_F==770 and synthetic_F==1564 and benchmark_F==550
    assert GPU<2700 and CPU<3600 and conservative_CPU_upper<3600
    receipt=dict(status='COMPLETE_WITH_MEASURED_PHASES',new_model='RGB_ONLY',new_fits=2,new_updates=6000,updates_per_seed=3000,
        old_model_fits=0,old_model_new_inference=0,old_metric_rows_reused=2200,
        new_RGB_accuracy_head_forwards=220,new_detector_accuracy_forwards=0,
        CVRANK_native_CPU_image_calls=110,CVRANK_seed_replica_independent_fit_claim=False,
        new_main_metric_rows=3300,new_RGB_metric_rows=440,new_CVRANK_metric_rows=660,
        synthetic_selection_new_F_calls=synthetic_F,synthetic_selection_reused_F_rows=3128,
        REAL_new_F_calls=new_real_F,benchmark_new_F_calls=benchmark_F,total_new_F_calls=synthetic_F+new_real_F+benchmark_F,
        expected_F_calls=dict(synthetic_selection=1564,REAL=770,benchmark=550),
        actual_OpenCV_calls=dict(synthetic_selection=synthetic_OpenCV,REAL=new_real_OpenCV,benchmark=benchmark_OpenCV,
            total=synthetic_OpenCV+new_real_OpenCV+benchmark_OpenCV),
        expected_OpenCV_upper_bound=6*(1564+770+550),
        GPU_device_reserved_seconds=GPU,GPU_budget_seconds=2700,
        CPU_analysis_reserved_seconds=CPU,CPU_analysis_budget_seconds=3600,
        CPU_analysis_conservative_upper_seconds=conservative_CPU_upper,
        CPU_conservative_interval_start_UTC=datetime.fromtimestamp(conservative_start_epoch,timezone.utc).isoformat(),
        CPU_conservative_interval_end_epoch=conservative_end_epoch,
        CPU_conservative_bound_definition='Entire elapsed wall from a start before initialization, plus all measured CPU analysis phases. This overcounts GPU/idle time and covers uninstrumented preparation and measured parallel CPU workers. Add the final audit wall separately.',
        timing_definition='Actual child process wall; CUDA reservations include imports/load/CPU work, not kernel occupancy. CPU synthetic scoring also counted in analysis. Final independent audit has its own receipt and is separately checked against remaining CPU budget.',
        stages=stages,metadata_preparation_seconds=metadata_seconds,feature_audit_seconds=feature_audit_seconds,
        prior_LATENCY_routes_reused=10,old_18_route_benchmark_repeated=0,new_cost_processes=5,new_cost_timed_calls=450,new_cost_warmup_calls=100,
        new_geometry_cache=0,new_synthetic_images=0,new_backbone_updates=0,new_seeds_beyond1_2=0,new_hyperparameter_search=0,
        protected_or_Severe_REAL_access=0,manuscript_changes=0,scientific_stage_failures=[s['stage'] for s in stages if s['returncode']!=0],
        preparation_repair_record='IMPLEMENTATION_REPAIRS.json',
        preparation_failure_uninstrumented=repairs.get('preparation_failure_wall_seconds_not_instrumented',False),
        final_independent_audit_phase_record='STAGE_final_verify.json',
        raw_RGB_and_GT_coordinates_public=False,existing_results_deleted=False)
    with (doc/'COMPUTE_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print('COMPUTE_COMPLETE_GPU_CPU_SECONDS',GPU,CPU,receipt['actual_OpenCV_calls'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public-dir',required=True)
    p.add_argument('--analysis-start-epoch',type=float,default=1791725400,
        help='Conservative pre-initialization start; set a freshly recorded start for a clean replay.')
    run(p.parse_args())
