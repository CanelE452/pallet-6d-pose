"""Assemble measured execution accounting and six distinct validation axes.

Missing measurements remain null or explicit missing scope. This module does
not train, predict, optimize or decode reference poses.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import read,sha,write
from .campaign import progress


def optional(path):return read(path)if Path(path).exists()else None

def write_derived(path,value):
    path=Path(path)
    if path.exists():
        previous=path.with_name(path.stem+'.previous_'+sha(path)[:12]+path.suffix)
        if not previous.exists():path.rename(previous)
        elif sha(previous)!=sha(path):raise ValueError('Derived snapshot hash prefix collision')
        else:
            write(path,value,replace=True);return # exact previous bytes preserved
    write(path,value)


def cost(run_dir):
    root=Path(run_dir);plan=read(root/'PLAN_LOCK.json');branches={}
    bindings={};ledger=[];ids=set()
    def add(batch_id,wall,scope,receipt,wall_is_upper_bound=False):
        if batch_id in ids:raise ValueError('Distinct batch repeated: '+batch_id)
        if wall is None or wall<0:raise ValueError('Measured nonnegative batch wall required')
        ids.add(batch_id);ledger.append(dict(batch_id=batch_id,wall_seconds=wall,scope=scope,receipt=receipt,wall_is_upper_bound=wall_is_upper_bound))
    for branch,name in [('B','B_COST_AND_EXECUTION_RECEIPT.json'),('D1','D1_COST_AND_EXECUTION_RECEIPT.json'),('L','L_COST_AND_EXECUTION_RECEIPT.json')]:
        x=optional(root/name);branches[branch]=x;bindings[name]=sha(root/name)if x else None
        if not x:continue
        if branch=='B':
            # Already sums disjoint production, technical, calibration, audits.
            add('B_MEASURED_BATCH_SUBTOTAL',x['measured_sequential_batch_wall_seconds_subtotal'],
                'B receipt distinct sequential batches; see embedded scope limitations',name)
        else:
            if 'distinct_batch_wall_ledger'not in x:raise ValueError(branch+' receipt lacks distinct measured batch ledger; no zero substitution')
            for item in x['distinct_batch_wall_ledger']:
                add(item['batch_id'],item['wall_seconds'],item.get('scope'),name,item.get('wall_is_upper_bound',False))
    root_batches=[]
    for p in sorted((root/'root_execution_costs').glob('*.json')):
        x=read(p);root_batches.append(x);bindings[str(p.relative_to(root))]=sha(p)
        add(x['batch_id'],x['wall_seconds'],x['scope'],str(p.relative_to(root)))
    artifact_bytes=sum(p.stat().st_size for p in root.rglob('*')if p.is_file()and not p.is_symlink())
    repo=Path(__file__).resolve().parents[3]
    public=repo/'_docs/experiments/pallet_boundary_action_repair_v1'
    new_namespace_bytes=sum(p.stat().st_size for folder in (public,Path(__file__).parent)for p in folder.rglob('*')if p.is_file()and not p.is_symlink())
    artifact_bytes+=new_namespace_bytes
    elapsed=time.time()-plan['campaign_start_epoch'];limits=plan['limits'];measured=sum(x['wall_seconds']for x in ledger)
    attempts=root/'action_training_v1/TRAINING_ATTEMPTS.jsonl'
    attempt_n=sum(bool(line.strip())for line in attempts.open())if attempts.exists()else 0
    obs=optional(root/'D1_OBSERVATION_COST_AUDIT.json');profile=optional(root/'D1_PERSISTENT_HOOK_PROFILE.json')
    GPU_upper=obs['total_observation_GPU_process_wall_upper_bound_seconds']if obs else None
    # Profile is measured separately and charged conservatively via its entire
    # execution host wall if supplied; no missed event is substituted by zero.
    additional_profile_host=None
    if profile:
        additional_profile_host=profile.get('total_host_seconds',profile.get('total_wall_seconds',profile.get('wall_seconds')))
    d1=branches.get('D1')or{}
    GPU_jobs=[j for j in d1.get('unique_execution_batches',[])if j.get('kind')in ('GPU_OBSERVATION','TECHNICAL_GPU_OBSERVATION')]
    GPU_full_upper=sum(j['parent_wall_seconds']for j in GPU_jobs)if GPU_jobs and all(j.get('parent_wall_seconds')is not None for j in GPU_jobs)else None
    checks=dict(campaign_wall=elapsed<=limits['campaign_wall_hours']*3600,
        measured_batch_wall_sum=measured<=limits['solver_analysis_job_wall_sum_hours']*3600,
        new_training_attempts=attempt_n<=limits['new_training_runs'],disk=artifact_bytes<=limits['new_artifact_GiB']*1024**3,
        GPU_full_process_host_upper_bound=None if GPU_full_upper is None else GPU_full_upper<=limits['GPU_activity_hours']*3600)
    value=dict(schema='CAMPAIGN_COST_AND_EXECUTION_V1',epoch=time.time(),branches=branches,
        root_analysis_batches=root_batches,distinct_batch_wall_ledger=ledger,
        measured_batch_wall_sum_seconds=measured,
        wall_sum_scope='Mixed exact measured batches and explicitly marked conservative parent-lifetime upper bounds; unmeasured preparation remains excluded.',
        exact_measured_batch_wall_subtotal_seconds=sum(x['wall_seconds']for x in ledger if not x['wall_is_upper_bound']),
        conservative_batch_wall_upper_bound_subtotal_seconds=sum(x['wall_seconds']for x in ledger if x['wall_is_upper_bound']),
        campaign_elapsed_seconds=elapsed,
        new_artifact_bytes=artifact_bytes,new_public_and_source_namespace_bytes=new_namespace_bytes,new_training_attempts=attempt_n,
        GPU_observation_process_host_seconds_upper_bound=GPU_upper,
        GPU_full_process_host_seconds_upper_bound=GPU_full_upper,
        GPU_upper_bound_unique_batch_ids=[j['execution_batch_id']for j in GPU_jobs],
        GPU_additional_profile_host_seconds=additional_profile_host,
        GPU_actual_CUDA_event_total_seconds=None,
        GPU_capture='Original persistent SAM CUDA hook/event gap preserved; observation call counts from actual provider receipts and first CAL cold/warm profiler are separate. No Jetson FPS inferred.',
        limits=limits,limit_checks=checks,receipt_sha256=bindings,
        unmeasured_scopes=['Initial interactive source inspection, preparation and tiny debugging jobs were not all wall/CPU timed.',
            'Full-campaign CPU seconds and simultaneous total RSS were not measured; per-batch actual subsets are retained.',
            'Original reused training/generation runtime is historical reuse, not zero-cost standalone inference.',
            'B activeCAL33profile timing belongs to main238 and is not added twice; separate small fixed-pose CAL geometry audit timer is unmeasured (B_ACTIVE_CAL_COST_SCOPE.json).'],
        precision='Measured batch sum is a subtotal with scope gaps; missing measurements are never reported as zero.',
        training_cost='18 new small-head optimizer attempts cap;3 historical H0 checkpoints reused; no new backbone training',
        inference_cost='Common D1 producer is charged to pair and full selectors; no pair-only runtime saving was established.')
    write_derived(root/'COST_AND_EXECUTION.json',value);return value


def validation(run_dir):
    root=Path(run_dir);state=progress(root)
    names={
        'input':'P0_BASELINE_INPUT_INVENTORY.json','gate':'ALL_OUTPUTS_SCORING_GO.json',
        'numeric':'independent_numeric_v1/INDEPENDENT_CHECK.json',
        'geometry':'b_finite_independent_v2/INDEPENDENT_FINITE_CHECK.json',
        'geometry_original':'b_finite_independent_v1/INDEPENDENT_FINITE_CHECK.json',
        'learning':'independent_action_full_v1/INDEPENDENT_ACTION_AUDIT.json',
        'learning_original':'independent_action_d0_v1/INDEPENDENT_ACTION_AUDIT.json',
        'h1_failure':'independent_h1_failure_v1/INDEPENDENT_H1_FAILURE_CHECK.json',
        'h1_analysis':'h1_all_failure_v1/ANALYSIS_SEAL.json',
        'visual':'VISUAL_REVIEW.json','publication':'REMOTE_PUBLICATION_RECEIPT.json',
        'privacy':'PRIVACY_REVIEW.json','rights':'RIGHTS_SCOPE_BINDING.json',
        'interruption':'D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json',
        'recovery':'D1_RECOVERY_FINALIZATION_RECEIPT.json'}
    evidence={k:optional(root/v)for k,v in names.items()}
    def axis(status,keys,note):
        return dict(status=status,evidence=[dict(path=names[k],sha256=sha(root/names[k]))for k in keys if evidence[k]],note=note)
    expected={'B','D1_REAL_OBSERVATIONS','D1_SYNTHETIC_OBSERVATIONS','D1_REAL_CANDIDATES','D1_SYNTHETIC_CANDIDATES','ALL_PREDICTIONS','L'}
    complete=set(state['branches'])==expected and all(x['status']=='COMPLETED'for x in state['branches'].values())and evidence['gate']is not None
    execution_gap=optional(root/'D1_TECHNICAL_ATTEMPTS_RETROSPECTIVE.json')
    execution_status='PASS_COMPLETED_OUTPUTS_WITH_DOCUMENTED_TECHNICAL_LOG_GAPS'if complete and execution_gap else'PASS'if complete else'PENDING'
    numeric=evidence['numeric'];geometry=evidence['geometry'];learning=evidence['learning'];visual=evidence['visual'];publication=evidence['publication']
    h1=evidence['h1_failure']
    h1_ok=bool(h1 and h1.get('complete') is True and h1.get('failures')==0 and
        h1.get('status')in ('PASS','PASS_WITH_DOCUMENTED_REPRESENTATION_DIFFERENCES'))
    numeric_status=numeric.get('status','PENDING')if numeric else'PENDING'
    learning_status=learning.get('status','PENDING')if learning else'PENDING'
    if not h1_ok:
        numeric_status=learning_status='H1_ALL_FAILURE_DIAGNOSTIC_PENDING_OR_FAILED'
    privacy=evidence['privacy'];rights=evidence['rights']
    visual_ok=bool(visual and visual.get('pixel_review_complete')is True)
    privacy_ok=bool(privacy and privacy.get('actual_redacted_output_review')=='COMPLETED_MODEL_PIXEL_REVIEW')
    rights_ok=bool(rights and len(set(rights.get('approved_ids',[])))==192)
    remote_ok=bool(publication and publication.get('status')=='PUSHED_MAIN_VERIFIED')
    visual_status='PUSHED_MAIN_VERIFIED_WITH_MODEL_PIXEL_REVIEW'if visual_ok and privacy_ok and rights_ok and remote_ok else'MODEL_PIXEL_REVIEW_COMPLETE_REMOTE_PENDING'if visual_ok and privacy_ok and rights_ok else'VISUAL_OR_RIGHTS_REVIEW_PENDING_OR_FAILED'
    axes={
        'INPUT_AND_VERSION':axis('PASS'if evidence['input']and evidence['input']['status']=='PASS'and evidence['gate']else'PENDING',['input','gate'],'Original fixed references, split, source/weight closures and memoization amendment are bound before scoring.'),
        'EXECUTION_COMPLETENESS':axis(execution_status,['gate','interruption','recovery'],'Completion derives from sealed rows/checkpoints, not process existence; all37 arms preserve238 rows. Initial descriptor/keyword-default errors have retrospective summaries but missing raw traceback files; those gaps are preserved, not reclassified as full trace capture. The externally interrupted shared run and unchanged-procedure recovery retain distinct attempts, unknown partial operation counts, and separate parent-wall accounting.'),
        'NUMERIC_REPRODUCTION':axis(numeric_status,['numeric','h1_failure','h1_analysis'],'Independent formula/aggregation/paired/source replay plus every H1 all-failure table/membership. Original-expression strict directions are preserved; stable-expression sign differences are recorded separately without changing reported directions.'),
        'GEOMETRIC_SUPPORT':axis(geometry.get('status','PENDING')if geometry else'PENDING',['geometry_original','geometry'],'Raw v1 lambda arithmetic FAIL is preserved; unchanged-bound secondary expression reconciliation and finite predicates are separate.'),
        'LEARNING_ACTION_CONTRACT':axis(learning_status,['learning_original','learning','h1_failure'],'D0 original raw arithmetic discrepancies retained; action labels, dedup, roles, same real features, split and actual checkpoint outputs independently checked. H1 no-success-bank membership and continuous harm are a separate independently checked diagnostic.'),
        'VISUAL_AND_PUBLICATION':axis(visual_status,['visual','publication','privacy','rights'],'Actual image pixel inspection is model_review, not human or physical GT certification. Remote verification is separate from offline file/link checks.')}
    if execution_gap:axes['EXECUTION_COMPLETENESS']['technical_log_gap_evidence']=dict(path='D1_TECHNICAL_ATTEMPTS_RETROSPECTIVE.json',sha256=sha(root/'D1_TECHNICAL_ATTEMPTS_RETROSPECTIVE.json'))
    axes['NUMERIC_REPRODUCTION']['H1_all_failure_diagnostic_status']=h1.get('status')if h1 else'PENDING'
    if evidence['interruption']:
        interrupted=evidence['interruption'];recovered=evidence['recovery']
        axes['EXECUTION_COMPLETENESS']['runtime_interruption']=dict(
            status=interrupted.get('status'),termination_cause=interrupted.get('concrete_termination_cause'),
            tool_exit_code=interrupted.get('observed_tool_session_exit_code'),
            partial_operation_counts=interrupted.get('interrupted_partial_operations'),
            recovery_status=recovered.get('status')if recovered else'PENDING',
            completed_outputs_do_not_imply_uninterrupted_execution=True)
    axes['VISUAL_AND_PUBLICATION']['substatus']=dict(pixel_review=visual_ok,privacy_review=privacy_ok,rights_scope=rights_ok,remote_verified=remote_ok,visual_receipt_status=visual.get('status')if visual else None,publication_receipt_status=publication.get('status')if publication else None)
    value=dict(schema='SIX_AXIS_VALIDATION_V1',epoch=time.time(),axes=axes,material_owner='UNVERIFIED',
        actual_physical_6D_ground_truth='UNVERIFIED',human_review=False,model_review=bool(visual and visual.get('reviewer_type')=='model' or privacy and privacy.get('reviewer_type')=='model'),
        operational_defaults_changed=False,paper_changed=False,
        semantic_limit='Correct computation and execution do not independently verify material owner or physical6D GT; one PASS does not cover all six axes.')
    write_derived(root/'VALIDATION.json',value);write_derived(root/'RUN_STATE.json',state);return value


def bindings(run_dir):
    root=Path(run_dir);repo=Path(__file__).resolve().parents[3]
    names=['PLAN_LOCK.json','P0_BASELINE_INPUT_INVENTORY.json','EVALUATION_PROTOCOL.json',
        'D1_CACHE_TECHNICAL_AMENDMENT_LOCK.json','RESOURCE_BUDGET_UNIT_LOCK.json','ALL_OUTPUTS_SCORING_GO.json',
        'PREDICTION_SEAL.json','B_SOURCE_LOCK_v2.json','D1_MEMOIZATION_STATIC_MODEL_REVIEW_L.json',
        'D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json','D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json',
        'D1_RECOVERY_LAUNCH_LOCK.json','D1_RECOVERY_FINALIZATION_RECEIPT.json']
    source={str(p.relative_to(repo)):sha(p)for p in sorted(Path(__file__).parent.glob('*.py'))}
    value=dict(schema='FINAL_SOURCE_AND_INPUT_BINDINGS_V1',execution_base=read(root/'PLAN_LOCK.json')['base_main'],
        source_sha256=source,private_receipt_sha256={n:sha(root/n)for n in names if(root/n).exists()},
        source_scope='New research namespace only; original method source/weight/input/split closure is nested in immutable receipts.',
        result_commit='Read actual publication receipt/final response; no commit self reference')
    write_derived(root/'SOURCE_BINDINGS.json',value);return value


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['cost','validation','bindings']);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    f={'cost':cost,'validation':validation,'bindings':bindings}[a.stage]
    x=f(a.run_dir);print(a.stage,x.get('schema'),flush=True)


if __name__=='__main__':main()
