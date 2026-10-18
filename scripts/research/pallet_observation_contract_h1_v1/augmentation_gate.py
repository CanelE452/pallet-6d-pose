"""Seal P4 feasibility separately from observed synthetic coverage and budget.

This command does not train, render, forward a model, or read real references.
It records unmet prerequisites, rather than inventing a generator-controllable
cause from the number of failed synthetic actions.
"""
from __future__ import annotations
import argparse,json,time,resource
from pathlib import Path
from scripts.research.pallet_risk_information_v1.common import read,rows,sha,write

def seal_gate(run,coverage_dir,resume_receipt,output):
    start,cpu=time.monotonic(),time.process_time();run,coverage_dir,resume_receipt,output=map(Path,(run,coverage_dir,resume_receipt,output))
    plan_path=run/'EXPERIMENT_PLAN.json';cov_path=coverage_dir/'SYNTHETIC_COVERAGE.json';scene_path=run/'B_SCENE_RENDER_PREFLIGHT.json';ledger_path=run/'GLOBAL_FIT_ATTEMPTS.jsonl'
    plan,cov,scene,resume=map(read,(plan_path,cov_path,scene_path,resume_receipt))
    if plan['status']!='FROZEN_BEFORE_FIRST_OPTIMIZER_UPDATE' or cov['status']!='COMPLETE_DIAGNOSTIC_ONLY':raise ValueError('Frozen plan and complete coverage required')
    if sha(coverage_dir/'SYNTHETIC_COVERAGE_ROWS.jsonl.gz')!=cov['rows_sha256']or cov['experiment_plan_sha256']!=sha(plan_path):raise ValueError('Coverage binding differs')
    ledger=list(rows(ledger_path));attempt_ids=[r['attempt']for r in ledger]
    if len(attempt_ids)!=len(set(attempt_ids)):raise ValueError('Duplicate optimizer reservation')
    failed=resume['prior_failed_attempts'];required_base=plan['fit_count']['required'];cap=plan['fit_count']['total_cap_including_failed_attempts'];additional=plan['fit_count']['conditional_additional']
    projected=max(len(ledger),required_base+len(failed))
    if additional!=6 or required_base!=3 or cap!=9:raise ValueError('Original fixed balanced comparison budget changed')
    if scene['P4']['same_generator_new_train_scenes_runnable']:raise ValueError('Scene availability changed; substantive generator gate requires root review')
    evidence=[plan_path,cov_path,scene_path,resume_receipt,ledger_path]
    result=dict(schema='P4_AUGMENTATION_SCIENTIFIC_AND_BUDGET_GATE_V1',status='NOT_RUN_INPUT_AND_BUDGET_GATES_UNMET',
        bindings={p.name:dict(path=str(p),sha256=sha(p))for p in evidence},
        scientific_gate=dict(passed=False,
            deficiency=dict(status='GENERATOR_CONTROLLABLE_MISSING_CONDITION_NOT_ESTABLISHED',
                evidence='All declared marginal validation occlusion/elevation/azimuth/scene-mode/explicit-occluder bins are represented in train. Low E0-only/full-only counts describe outcome coverage, not a causal scene factor or a success-ratio target.',
                untested='No claim that all joint visual/geometric conditions are covered; no new real GT relation was used to propose a scene factor.'),
            generator_assets_camera_label=dict(passed=False,camera_and_original_label_contract_available=scene['P4']['camera_and_original_label_contract_available'],
                same_generator_new_train_scenes_runnable=False,missing_inputs=scene['missing_replay_inputs']),
            automatic_D1=dict(source_available=scene['P4']['same_D1_producer_source_available'],new_scenes_available=False,new_jobs=0),
            matched_control_and_structural_factor=dict(defined=False,reason='No authenticated reusable generator or demonstrated controllable missing cause; no invented augmentation factors.')),
        optimizer_budget_gate=dict(passed=projected+additional<=cap,cap_including_failures=cap,
            observed_optimizer_reservations=len(ledger),observed_attempt_ids=attempt_ids,
            preserved_failed_attempts=failed,mandatory_successful_base_fits=required_base,
            projected_no_B_total_after_mandatory_completion=projected,
            projected_remaining_slots=cap-projected,required_balanced_additional_fits=additional,
            projected_total_if_full_comparison=projected+additional,
            note='Observed reservations are not silently counted as completed fits. Projection includes mandatory three successful fixed-seed no-B fits and all preserved failed creations.'),
        decision=dict(new_train_render_attempts=0,new_D1_jobs=0,new_optimizer_fits=0,
            no_cap_expansion=True,no_partial_or_unbalanced_seed_comparison=True,no_success_based_rejection_sampling=True,
            no_existing_D1_full_reproduction=True,original_train768_validation235_unchanged=True,
            resampling_not_substituted=True,reason='A matched six-fit augmentation comparison is both scientifically unprepared and outside the remaining optimizer cap; saved coverage and other instructed branches continue.'),
        PRIMARY_or_CAL_reference_reads=0,source_sha256=sha(__file__),
        cost=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1))
    write(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--coverage-dir',type=Path,required=True);p.add_argument('--resume-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=seal_gate(a.run_dir,a.coverage_dir,a.resume_receipt,a.output);print(json.dumps({k:r[k]for k in('status','optimizer_budget_gate','cost')}))
