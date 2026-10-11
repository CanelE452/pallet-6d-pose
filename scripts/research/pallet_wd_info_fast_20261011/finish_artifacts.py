"""Finalize execution status, full transitions and actual compute receipts."""
import csv
import json
import os
import platform
import subprocess
import time
from collections import Counter
from . import common as C


def transitions():
    result=[]
    with (C.DOC/'C_CHANGED_SELECTIONS.csv').open() as f:
        for r in csv.DictReader(f):
            result.append(dict(source='C_SELECTOR',**r,candidate_semantics='unchanged final8 cached poses'))
    d={(r['method'],r['id']):r for r in C.rows(C.DOC/'D_REPAIRED_ROWS.jsonl.gz')}
    with (C.DOC/'D_ALL_PAIRED_TRANSITIONS.csv').open() as f:
        for r in csv.DictReader(f):
            suffix='_SUBPIX' if r['route']=='SubPix' else ''
            # Actual route tokens are verified against the sealed D row keys.
            if ('D_CONTROL'+suffix,r['id']) not in d:
                suffix='_SUBPIX' if 'subpix' in r['route'].lower() else ''
            a=d['D_JOINT_DIM'+suffix,r['id']];b=d['D_CONTROL'+suffix,r['id']]
            known=a['axis_wrong'] is not None and b['axis_wrong'] is not None
            state='UNCLEAR' if not known else ('RECOVERY' if b['axis_wrong'] and not a['axis_wrong'] else
                'DAMAGE' if not b['axis_wrong'] and a['axis_wrong'] else 'UNCHANGED')
            out=dict(source='D_LOCAL',method=a['method'],id=r['id'],seed=1,
                from_hypothesis=b['hypothesis'],to_hypothesis=a['hypothesis'],axis_transition=state,
                old_success=b['strict_success'],new_success=a['strict_success'],
                control_T_cm=r['control_T_cm'],joint_T_cm=r['joint_T_cm'],
                control_R_deg=r['control_R_deg'],joint_R_deg=r['joint_R_deg'],corner_delta_px=r['corner_delta_px'],
                candidate_semantics='initial original selector fit; final8 selected metrics separate')
            for arm,row in [('control',b),('joint',a)]:
                for hyp,val in row['candidate_selector_values'].items():
                    name='long' if hyp.startswith('long') else 'short'
                    out[arm+'_'+name+'_score']=val.get('score')
                    out[arm+'_'+name+'_front_depth_m']=val.get('t',[None,None,None])[2]
                    out[arm+'_'+name+'_candidate_available']=val.get('success')
            result.append(out)
    fields=['source','method','id','seed','from_hypothesis','to_hypothesis','axis_transition',
            'old_success','new_success','long_T','long_R','short_T','short_R',
            'control_T_cm','joint_T_cm','control_R_deg','joint_R_deg','corner_delta_px',
            'candidate_semantics']
    fields+=sorted(set().union(*(r.keys() for r in result))-set(fields))
    destination=C.DOC/'CHANGED_SELECTIONS.csv';assert not destination.exists()
    with destination.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(result)
    return len(result)


def statuses():
    rows=[
      ('A1','DONE','Claims/source audits; protected historical claims not newly rescored','CLAIM_AUDIT.md'),
      ('A1_margin_corner','DONE','Paired margin/corner residual diagnostic; severe retained','A1_PAIRED_MARGIN_RESIDUAL.json'),
      ('A2','DONE','Cached candidate oracles; separate GT axis / best-T / best-R / best-success','A2_ORACLE_METRICS.json'),
      ('A3_exact','DONE','IMPLEMENTATION_OR_CONTRACT_ISSUE:2/16 C1 signed rotation mapping failures','A3_PHYSICAL_ROUNDTRIP_AUDIT.json'),
      ('A3_noise','DONE','FEASIBILITY_ONLY; physical C1 failures limit rotation ceiling interpretation','A3_GEOMETRY_DIAGNOSTIC.json'),
      ('A3_ratio','DONE','Projected matched-footprint ratios; no new RGB','A3_RATIO_PROJECTION_ROWS.jsonl.gz'),
      ('B1','DONE','Dimension sensitivity observed; no selection benefit established','B1_RESULTS.json'),
      ('B1_checkpoint_ablation','REUSED','N0/N1/N2/N3 saved outputs; allowed173 only; checkpoint changes distinct from same-checkpoint intervention','B1_ABLATION_REUSE.json'),
      ('B2','DONE','Whole-object symmetry targets counted; C1 no nonidentity supervision','B2_SYMMETRY.json'),
      ('C1_cache','DONE','Current R0 feature parity; missing confidence filled once with fixed backbone','C_EXTRACTION_RECEIPT.json'),
      ('C2_features','DONE','G188+V768+D5 runtime-only inputs; truth used after seals','C_GEOMETRY_RECEIPT.json'),
      ('C3_fits','DONE','All8 fixed arms; no sweeps/additional seeds','C_FITS.json'),
      ('C4_select','DONE','Single classifier decision p=.5; physical tagged candidate fallback','C_RESULTS.json'),
      ('C5_REAL','DONE','Full allowed173; axis170; seed-within-frame; no best real model promoted','C_RESULTS.json'),
      ('C5_SYNTH','DONE','Exploratory1024; old N3/C-D exposure and C1 mapping limits','C_SYNTH_TEST_RESULTS.json'),
      ('D_contract','DONE','Matched2305 parameters/initialization; original frozen weights','D_CONTRACT.json'),
      ('D_fit','DONE','2x1000 matched updates; common final step; no refit','D_TRAINING_RECEIPT.json'),
      ('D_first_evaluation','DONE','IMPLEMENTATION_OR_CONTRACT_ISSUE: K-list adapter; invalid rows preserved','D_ARTIFACT_INDEX.json'),
      ('D_recovery','DONE','Sealed choices reused; array K; 3 full-F checks on1frame/3routes','D_REPAIRED_POSE_SEAL.json'),
      ('D_final','DONE','UNRESOLVED: success86->85; axis41->41 afterSubPix; damage retained','D_REPAIRED_RESULTS.json'),
      ('D_sensitivity','DONE','Dimension response; mean/shuffle benefit not evidence for correct dimensions','D_REPAIRED_SENSITIVITY.json'),
      ('D_C_joint_holdout','BLOCKED_CONTRACT','C VAL253/TEST271 exposed by D TRAIN; no cross-branch independent TEST claim','D_CROSS_TASK_EXPOSURE.json'),
      ('E_gap','DONE','Separate official/raw-box-normalized confidence; fixed coverage and full-denominator success','E_RISK_COVERAGE.json'),
      ('E_probability','DONE','All8 arms+prior/3 seeds; VAL-only thresholds transferred once','E_PROBABILITY_RISK.json'),
      ('F_input','DONE','8910 continuous frames exist; independent motion/pose alignment absent','F_G_INPUT_AUDIT.json'),
      ('F_causal','BLOCKED_CONTRACT','No measured camera motion/extrinsics/no-face-switch contract','F_G_INPUT_AUDIT.json'),
      ('G1','BLOCKED_CONTRACT','BLOCKED_CALIBRATION: no independent target-matched floor calibration','F_G_INPUT_AUDIT.json'),
      ('G2_existing','REUSED','Existing S3 GT-derived planes remain posthoc feasibility','G2_PLANE_SENSITIVITY.json'),
      ('G2_sensitivity','DONE','FEASIBILITY_ONLY: fixed offsets/normals OFAT; all signs retained','G2_PLANE_SENSITIVITY.json'),
      ('G3_gate','REUSED','Existing Metric3D gate FAIL_STOP_S4; no reinference or new model','G3_DEPTH_GATE_REUSE.json'),
      ('G3_bias','DONE','Allowed depth cached join85; direction/bias diagnostic only','G3_DEPTH_BIAS_ROWS.jsonl.gz'),
      ('G3_S4_policy','BLOCKED_CONTRACT','Depth gate remains failed; no policy performance claim','G3_DEPTH_GATE_REUSE.json'),
      ('SQUARE118','NA','C4 no rectangular long/short task; prior aggregate only','INPUT_CONTRACT.json'),
      ('PROTECTED_FINAL146','NA','No new scoring/features/selection; physical role contract preserved','INPUT_CONTRACT.json'),
      ('PRIMARY_STATISTICS','DONE','8 hypotheses;9-session paired bootstrap10000; raw/Holm separate','PAIRED.json')]
    destination=C.DOC/'EXPERIMENT_STATUS.csv';assert not destination.exists()
    with destination.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['task','execution_status','judgment_or_limit','artifact']);w.writerows(rows)
    assert all((C.DOC/r[3]).exists() for r in rows)


def compute():
    receipts=[json.loads(line) for line in (C.PRIVATE/'PNP_RECEIPTS.jsonl').read_text().splitlines()]
    totals=Counter()
    for r in receipts:totals[r['job']]+=r['total']
    plan=C.read(C.DOC/'PLAN_LOCK.json')
    assert all(totals[j]<=limit for j,limit in plan['pnp_reservations'].items())
    assert sum(totals.values())==48277<=50000
    gpu=[]
    extraction=C.read(C.DOC/'C_EXTRACTION_RECEIPT.json')
    gpu.append(dict(job='C_frozen_extraction',seconds=extraction['GPU_device_seconds'],
                    peak_allocated_bytes=extraction['peak_allocated_GPU_bytes']))
    for filename in ['D_THROUGHPUT.json','D_TRAINING_RECEIPT.json','D_CALIBRATION.json','D_PREDICTION_SEAL.json']:
        r=C.read(C.DOC/filename)
        gpu.append(dict(job=filename[:-5],seconds=r['actual_device_seconds'],peak_allocated_bytes=r['peak_allocated_bytes']))
    elapsed=(time.time()-C.read(C.PRIVATE/'START.json')['start_unix'])/60
    try:
        device=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
    except Exception:device='Not available at receipt time'
    C.write(C.DOC/'COMPUTE_RECEIPT.json',dict(status='ACTUAL_COUNTS_FAILURE_CALLS_INCLUDED',
        hardware=dict(GPU=device,CPU=platform.processor() or platform.machine(),logical_CPU=os.cpu_count()),
        new_PnP_receipts=receipts,new_OpenCV_solve_refine_calls=sum(totals.values()),per_job_calls=dict(totals),
        reservations=plan['pnp_reservations'],GPU_jobs_sequential=True,GPU_measured_jobs=gpu,
        GPU_active_device_minutes=sum(x['seconds'] for x in gpu)/60,
        GPU_peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in gpu),
        GPU_timing_scope='CUDA synchronized active phase walls; startup/source audits excluded',
        new_classifier_fits=8,corrector_fits=2,corrector_updates_per_arm=1000,total_corrector_updates=2000,
        classifier_CPU_fit_wall_seconds=C.read(C.DOC/'C_FITS.json')['CPU_fit_wall_seconds'],
        CPU_analysis_fitting_budget_minutes=60,elapsed_project_wall_minutes_at_receipt=elapsed,
        CPU_threads_per_worker=2,max_CPU_workers=2,CPU_peak_memory='NOT_MEASURED; no retrospective estimate',
        frozen_YOLO_RGB_forwards=6144,frozen_N3_cached_features_forwards=6144,
        new_REAL_expensive_RGB_features=0,current_synthetic_confidence_extraction_images=6144,
        extraction_amendment='C_PREEXEC_AMENDMENT.json: required missing current-checkpoint confidence sidecar',
        new_synthetic_RGB_generation=0,new_depth_download_or_inference=0,
        failed_D_initial_calls_included=6055,D_repair_additional_calls=2440,
        invalid_D_first_outputs_preserved=True,protected_FINAL_new_scored=0,
        private_checkpoint_feature_files_published=False))


if __name__=='__main__':
    n=transitions();statuses();compute();print('FINAL_ARTIFACTS',n,'transition rows')
