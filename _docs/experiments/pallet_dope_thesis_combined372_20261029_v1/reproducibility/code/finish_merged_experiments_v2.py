"""Technical repair continuation V2: same372 analysis and all15 count-ratio trials.

Preserves the failed V1 attempt and reuses only its verified baseline-cache stage.
Only the metadata RGB SHA binding sealer changes; science kernels remain frozen.

No publication occurs here. Scientific stopping is amended before outcomes by
the explicit user completion policy, through its separately reviewed adapter.
Input/numerical gates are retained. Any implementation failure is preserved
for the active ROOT goal/heartbeat to diagnose and repair without data tuning.
"""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parent
MAIN=ROOT/'dope-filter-ablation-per-type-20261027-v1'
WOOD=ROOT/'dope-wood640-eval80-20261027-v1'
COMBINED=WOOD/'combined372'
RATIO=ROOT/'dope-r2-ratio-per-type-20261027-v1'
POLICY_SHA='7740a3f2cc778b6a9f9a48c595d260dab38ff66d1bc4296316419ec189aac120'
EXPECTED={
    str(OUT/'code/advance_merged_workflow.py'):'ce2509d14f94e415243820341eac6e1203bbeae5cb6769251757403b5491a098',
    str(OUT/'code/prepare_ratio_evaluation_v2.py'):'11278e1b623c251c98276305390de32cb3cd030cd740c25c52aeeb19da959322',
    str(OUT/'code/prepare_ratio_evaluation.py'):'3b8f18bed78e2bbece33f0488d0bdfe8ee0d0a4890f83af639c4924d6f4d02f7',
    str(OUT/'code/finish_merged_experiments.py'):'4b11205c5cbbe7b29f6a032a1a57313ac4143010cc0b2e946bb73b4910928275',
    str(WOOD/'code/prepare_combined.py'):'9430e48df9ff61da67d3d1df1aaf799deda1c0294d8bf739f2eac19d42c5ab53',
    str(WOOD/'code/combined_metrics.py'):'ce51e0bd3637d47eaae666e3ab1eb50e0a8ae5432aee35e8f0bac659ef861088',
    str(WOOD/'code/merged_pose.py'):'d5a6dbe600b485abdbe5b36f89b100e8b3b09a7127725d9c5aebed11aea5d163',
    str(RATIO/'code/completion_policy.py'):'9e9cb77d418d366a6fba8171045732515d9ac4eb3058a464f26b155cb3b50634',
    str(RATIO/'code/ratio_eval_common.py'):'3f16bcfe9dba1510c81984bb54113b18b2c43728158cf543465d106c03a495ad',
    str(RATIO/'code/evaluate_ratios.py'):'3ae3557ed1ec5dba298511e34de64a59a239a8018e5ae49ec731d6a44c8366ee',
    str(RATIO/'code/plot_ratios.py'):'9b92a83702ae943acceabe5b67c4ad9295deee7d7bfca93eb8f3ad0499569568',
    str(RATIO/'code/future_teacher_diagnostics.py'):'9d13a701f68320c05b74f99d382e34bd03eadec6f33b7ecc79c8a7fc98298b24',
}


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_new(path,value):
    require(path.resolve().is_relative_to(ROOT),'Output outside experiment roots')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def status(stage,**extra):
    path=OUT/'FINISH_EXPERIMENTS_STATUS_V2.json';temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(dict(utc=utc(),stage=stage,**extra),ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def verify_sources():
    review=read(OUT/'preflight/FINISH_WORKFLOW_ROOT_REVIEW_V2.json')
    require(review['passed'] is True and review['workflow_sha256']==sha(__file__)
        and review['source_sha256']==EXPECTED,'ROOT finish-workflow source review stale')
    require(sha(RATIO/'CONTINUATION_POLICY.json')==POLICY_SHA,'Pre-result completion policy changed')
    for path,digest in EXPECTED.items():require(sha(path)==digest,'Reviewed source changed: '+path)
    verify_resume_authority(review)
    return review


RESUME_AUTHORITY=OUT/'preflight/FINISH_WORKFLOW_RESUME_AUTHORITY_V2.json'
BASELINE_PATHS={str(COMBINED/'cache'/f'BASE_{kind}_{role}.json')
    for kind in ('PLASTIC','WOOD','GREEN') for role in ('R0','R1','control_R2','oldfull_R2')}


def verify_resume_authority(review):
    a=read(RESUME_AUTHORITY)
    require(a['schema']=='outer_workflow_resume_authority_v2' and a['passed'] is True
        and a['technical_repair_only'] is True and a['scientific_core_or_target_tags_changed'] is False
        and sha(RESUME_AUTHORITY)==review['resume_authority_sha256'],'ROOT technical repair authority absent/stale')
    entry=a['approved_rebindings'][str(Path(__file__).resolve())]
    producer=review['first_queue_process']
    require(entry['source_sha256']==sha(__file__) and entry['producer']=={
        'source_path':str(OUT/'code/advance_merged_workflow.py'),
        'source_sha256':EXPECTED[str(OUT/'code/advance_merged_workflow.py')]}
        and producer['source_path']==entry['producer']['source_path']
        and producer['source_sha256']==entry['producer']['source_sha256'],'Approved resume/producer identity differs')
    require(str(OUT/'FINISH_EXPERIMENTS_FAILURE.json') in entry['allowed_old_failures']
        and set(entry['allowed_old_failures']).issubset(a['old_failures']),'Original failure must be preserved and mapped')
    for path,digest in a['old_failures'].items():
        require(Path(path).is_relative_to(OUT) and sha(path)==digest,'Recorded old failure changed')
    reused=a['reused_completed_stage']
    require(reused['stage']=='merged_baseline_caches' and reused['journal_path']==str(OUT/'logs/FINISH_COMMANDS.jsonl')
        and sha(reused['journal_path'])==reused['journal_sha256'],'Prior actual command journal differs')
    rows=[json.loads(line) for line in Path(reused['journal_path']).read_text().splitlines() if line.strip()]
    require([r['stage'] for r in rows]==['merged_baseline_caches','ratio_evaluation_seal']
        and [r['returncode'] for r in rows]==[0,1]
        and rows[0]['script_sha256']==EXPECTED[str(WOOD/'code/prepare_combined.py')]
        and rows[1]['script_sha256']==EXPECTED[str(OUT/'code/prepare_ratio_evaluation.py')],
        'Only the successful baseline stage preceding the failed metadata sealer may be reused')
    require(set(reused['output_sha256'])==BASELINE_PATHS,'Exactly12 baseline outputs required')
    for path,digest in reused['output_sha256'].items():require(sha(path)==digest,'Bound baseline output changed')
    for path,digest in reused['stdout_sha256'].items():
        require(path in {r['log'] for r in rows} and sha(path)==digest,'Original stage stdout changed')
    require(set(reused['stdout_sha256'])=={r['log'] for r in rows},'Bind both original success and failure stdout logs')
    return a


def wait_for(path,stage):
    status(stage)
    def ready():
        require(not (OUT/'MERGED_WORKFLOW_FAILURE.json').exists(),'Earlier reviewed queue failed; ROOT repair needed')
        if not path.is_file():return False
        try:
            read(path)  # An exposed, not-yet-flushed JSON file is not ready.
        except (json.JSONDecodeError,UnicodeError,OSError):
            return False
        return True
    while not ready():
        review=verify_sources()
        require(not (OUT/'MERGED_WORKFLOW_FAILURE.json').exists(),'Earlier reviewed queue failed; ROOT repair needed')
        # State files alone are insufficient evidence of a live producer.
        import psutil
        producer=review['first_queue_process']
        try:
            process=psutil.Process(producer['pid'])
            live=(abs(process.create_time()-producer['create_time'])<.01
                and str(OUT/'code/advance_merged_workflow.py') in process.cmdline()
                and process.status()!=psutil.STATUS_ZOMBIE)
            if not live:
                if ready():return
                raise RuntimeError('Earlier queue PID identity/state changed')
        except psutil.NoSuchProcess:
            if ready():return  # Receipt may have appeared just before exit.
            raise RuntimeError('Earlier queue handle is missing before required artifact: '+str(path))
        time.sleep(30)


def run(stage,script,args):
    verify_sources();status(stage)
    command=[sys.executable,'-B',str(script),*args]
    log=OUT/'logs'/('finish_v2_'+stage+'.stdout.log')
    require(not log.exists(),'Preserve existing/partial stage log: '+str(log))
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        MPLCONFIGDIR=str(RATIO/'cache/matplotlib'))
    begin=time.monotonic()
    with log.open('xb',buffering=0) as stream:
        done=subprocess.run(command,cwd=script.parents[1],env=environment,stdout=stream,stderr=subprocess.STDOUT)
    row=dict(utc=utc(),stage=stage,command=command,returncode=done.returncode,elapsed_seconds=time.monotonic()-begin,
        script_sha256=sha(script),log=str(log))
    with (OUT/'logs/FINISH_COMMANDS_V2.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    require(done.returncode==0,'Stage failed: '+stage)
    verify_sources()


def execution_go():
    train=read(RATIO/'preflight/TRAINING_GO.json');evaluate=read(RATIO/'preflight/EVALUATION_GO.json')
    review=read(RATIO/'preflight/COMPLETION_POLICY_ROOT_REVIEW.json')
    tests=read(RATIO/'preflight/COMPLETION_POLICY_PURE_TESTS.json')
    wrapper=RATIO/'code/completion_policy.py'
    require(review['passed'] is True and review['source_sha256'][str(wrapper)]==sha(wrapper)
        and review['policy_sha256']==POLICY_SHA and tests['passed'] is True and tests['check_count']==12
        and tests['source_sha256']==sha(wrapper),'Completion policy static/pure-test review stale')
    sources=dict(train['code_sha256'])
    for name in ('ratio_eval_common.py','evaluate_ratios.py','plot_ratios.py','future_teacher_diagnostics.py'):
        path=str(RATIO/'code'/name);sources[path]=evaluate['source_sha256'][path]
    sources[str(wrapper)]=sha(wrapper)
    require(len(sources)==9,'Expected unchanged4training+4evaluation+wrapper')
    write_new(RATIO/'preflight/EXECUTION_GO.json',dict(passed=True,policy_execution_authorized=True,utc=utc(),
        modes=['train','evaluate'],continuation_policy_sha256=POLICY_SHA,
        training_go_sha256=sha(RATIO/'preflight/TRAINING_GO.json'),
        evaluation_go_sha256=sha(RATIO/'preflight/EVALUATION_GO.json'),source_sha256=sources,
        scientific_stop_only_amended=True,technical_gates_waived=False))


def wood_decision():
    folder=RATIO/'evaluation/ratios_v1/wood';evidence=read(folder/'WOOD_GATE_EVIDENCE.json')
    require(evidence['complete'] is True and evidence['baseline_condition']=='C'
        and sha(Path(evidence['paired_ci_path']))==evidence['paired_ci_sha256'],'WOOD evidence incomplete/changed')
    require(set(evidence['contrasts'])=={'A','B','D','E'},'All four actual WOOD intervals required')
    import math
    all_zero=True
    for value in evidence['contrasts'].values():
        require(value['metric']=='PCK10' and value['reference_condition']=='C'
            and math.isfinite(value['ci_low']) and math.isfinite(value['ci_high'])
            and value['ci_low']<=value['ci_high'],'Invalid WOOD interval')
        all_zero &= value['ci_low']<=0<=value['ci_high']
    require(evidence['all_four_include_zero']==all_zero,'WOOD evidence truth flag differs')
    require(evidence['proposed_decision']==('STOP_REPORT' if all_zero else 'CONTINUE'),'Original pilot recommendation changed')
    bindings=dict(evidence['input_sha256'])
    for condition in 'ABCDE':
        directory=RATIO/'trials/WOOD'/condition
        for name in ('R2.pth','PROTOCOL.json','TRAIN_COMPLETE.json'):
            p=directory/name;require(bindings.get(str(p))==sha(p),'WOOD model binding differs: '+str(p))
    for p in (folder/'WOOD_GATE_EVIDENCE.json',folder/'PAIRED_CI.csv',RATIO/'CONTINUATION_POLICY.json',RATIO/'preflight/EXECUTION_GO.json'):
        bindings[str(p)]=sha(p)
    write_new(RATIO/'WOOD_RATIO_DECISION.json',dict(passed=True,utc=utc(),decision='CONTINUE',
        baseline_condition='C',contrasts=evidence['contrasts'],all_four_include_zero=all_zero,input_sha256=bindings,
        original_pilot_recommendation=evidence['proposed_decision'],
        continuation_authority='Latest user completion request registered before any ratio training/outcome',
        continuation_policy_sha256=POLICY_SHA,statistical_interpretation='inconclusive' if all_zero else 'report_each_actual_interval',
        hyperparameters_or_CI_values_changed=False,technical_gates_waived=False))


def main():
    require(sys.argv[1:]==['--run'],'Explicit --run required')
    require(not (OUT/'FINISH_EXPERIMENTS_COMPLETE.json').exists(),'Preserve completed workflow')
    with (OUT/'logs/FINISH_EXPERIMENTS.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            verify_sources()
            wait_for(WOOD/'results/COMPLETE.json','waiting_for_wood80_scored')
            wait_for(RATIO/'preflight/TRAINING_GO.json','waiting_for_ratio_training_go')
            # The one successful V1 stage is bound by ROOT and never rerun.
            verify_resume_authority(verify_sources())
            run('ratio_evaluation_seal',OUT/'code/prepare_ratio_evaluation_v2.py',['--seal'])
            execution_go()
            run('merged_primary372_points',WOOD/'code/combined_metrics.py',['score','--tag','final_v1','--main-tag','final_v1'])
            run('merged_old45_pose_golden',WOOD/'code/merged_pose.py',['golden','--tag','pose_golden_v1','--main-tag','final_v1'])
            golden=COMBINED/'evaluation/pose_golden_v1/pose-golden/QA.json';proof=read(golden)
            require(proof['passed'] is True and proof['groups']==17 and proof['frames']==45,'Old45 pose golden failed')
            write_new(COMBINED/'preflight/MERGED_POSE_GO.json',dict(authorized=True,utc=utc(),
                scope='supplemental_2D_reconstructed_reference_WOOD125',manifest_sha256=sha(COMBINED/'MERGED_MANIFEST.json'),
                source_sha256=sha(WOOD/'code/merged_pose.py'),golden_receipt_sha256=sha(golden)))
            run('merged_primary372_supplemental_pose',WOOD/'code/merged_pose.py',
                ['score','--tag','final_v1','--main-tag','final_v1','--golden-tag','pose_golden_v1'])
            wait_for(OUT/'MERGED_WORKFLOW_COMPLETE.json','waiting_for_first_five_wood_ratios')
            first=read(OUT/'MERGED_WORKFLOW_COMPLETE.json')
            require(first['complete'] is True and first['WOOD_ratios_trained']==5 and first['other_ratios_trained']==0,
                'First queue completion scope differs')
            run('wood_ratios_infer',RATIO/'code/evaluate_ratios.py',['infer','--tag','ratios_v1','--phase','wood'])
            run('wood_ratios_score',RATIO/'code/evaluate_ratios.py',['score','--tag','ratios_v1','--phase','wood'])
            wood_decision()
            wrapper=RATIO/'code/completion_policy.py'
            for kind in ('GREEN','PLASTIC'):
                for condition in 'ABCDE':
                    run('ratio_'+kind+'_'+condition,wrapper,['train','trial','--kind',kind,'--condition',condition])
            for kind in ('WOOD','GREEN','PLASTIC'):
                for condition in 'ABCDE':
                    receipt=read(RATIO/'trials'/kind/condition/'TRAIN_COMPLETE.json')
                    require(receipt['complete'] is True and receipt['stats']['updates_completed']==828
                        and receipt['optimizer_steps']==[1656],'One of all15 requested trials is incomplete')
            run('all_ratios_infer',wrapper,['evaluate','infer','--tag','ratios_v1','--phase','all'])
            run('all_ratios_score',wrapper,['evaluate','score','--tag','ratios_v1','--phase','all'])
            run('all_ratios_plots',RATIO/'code/plot_ratios.py',['--tag','ratios_v1','--phase','all'])
            run('all_ratios_future_teacher_select',RATIO/'code/future_teacher_diagnostics.py',['select','--tag','ratios_v1','--phase','all'])
            run('all_ratios_future_teacher_score',RATIO/'code/future_teacher_diagnostics.py',['score','--tag','ratios_v1','--phase','all'])
            write_new(OUT/'FINISH_EXPERIMENTS_COMPLETE.json',dict(complete=True,utc=utc(),primary_frames=372,WOOD_frames=125,
                count_ratio_trials_completed=15,primary_results_scored=True,supplemental_reconstructed_pose_scored=True,
                future_R3_probe_completed=True,source_sha256=EXPECTED,workflow_sha256=sha(__file__),
                repair_authority_sha256=sha(RESUME_AUTHORITY),reused_completed_stages=['merged_baseline_caches'],
                original_attempt_journal=str(OUT/'logs/FINISH_COMMANDS.jsonl'),resume_journal=str(OUT/'logs/FINISH_COMMANDS_V2.jsonl'),
                published=False,completion_of_user_goal_proven=False,
                pending='ROOT independent numerical and visual audits, remaining paired comparisons/ablation plots, thesis integration, final integrity, isolated main publication and remote blob verification'))
            status('ready_for_root_final_audits_and_publication')
        except Exception as error:
            write_new(OUT/'FINISH_EXPERIMENTS_FAILURE_V2.json',dict(utc=utc(),error=repr(error),published=False,
                input_or_performance_tuning_attempted=False,root_repair_required=True))
            status('implementation_repair_needed',error=repr(error));raise


if __name__=='__main__':main()
