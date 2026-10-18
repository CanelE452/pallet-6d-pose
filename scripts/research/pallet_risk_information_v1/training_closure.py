"""Reference-free, reporting-only closure of the fixed eighteen completed fits."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import time
from .common import read, rows, sha, write


ARMS=('R_CONT_F0','R_ORDER_F0','R_ORDER_F1_ZERO','R_ORDER_F1','V_ORDER_NULL','V_ORDER_RGB')
SEEDS=(20261018,20261019,20261020)


def process_cost(root,name):
    path=root/(name+'_PROCESS_COST.txt');raw=path.read_text()
    def get(key):
        return re.search(r'^\s*'+re.escape(key)+r':\s*(.+)$',raw,re.M).group(1)
    parts=list(map(float,get('Elapsed (wall clock) time (h:mm:ss or m:ss)').split(':')))
    return dict(batch_id=name,wall_seconds=sum(x*60**i for i,x in enumerate(reversed(parts))),
        CPU_user_seconds=float(get('User time (seconds)')),CPU_system_seconds=float(get('System time (seconds)')),
        peak_rss_kib=int(get('Maximum resident set size (kbytes)')),actual_exit_code=int(get('Exit status')),
        scope='Whole CLI including import, preparation and contained fits/probe; nested timers are not additional batches',
        wall_is_upper_bound=False,process_cost_path=str(path),process_cost_sha256=sha(path),log_sha256=sha(root/(name+'.log')))


def run(root):
    root=Path(root)
    for name in ('ALL18_TRAINING_CLOSURE.json','TRAINING_COST_AND_EXECUTION.json'):
        if (root/name).exists():raise FileExistsError('Preserve previous closure: '+name)
    attempts=list(rows(root/'GLOBAL_FIT_ATTEMPTS.jsonl'))
    assert len(attempts)==18 and {(r['arm'],r['seed'])for r in attempts}=={(a,s)for a in ARMS for s in SEEDS}
    assert not list((root/'training').glob('*/FAILED_TECHNICAL.json'))
    execution=read(root/'V_TRAINING_EXECUTION_RECEIPT.json')
    assert execution['status']=='COMPLETE'and execution['successful_new_fits']==6 and execution['new_optimizer_attempts']==6 and not execution['failed_fits']
    manifest_path=root/'TRAINING_MANIFEST.json';prior=sha(manifest_path);manifest=read(manifest_path)
    run_keys=list(manifest['runs'][0]);runs=[];summaries=[];all_receipts={};all_history={}
    for arm in ARMS:
        for seed in SEEDS:
            folder=root/'training'/f'{arm}_s{seed}';path=folder/'RUN_RECEIPT.json';r=read(path);history=read(folder/'HISTORY.json')
            expected={'R_CONT_F0':16386,'R_ORDER_F0':16321}.get(arm,57977 if arm.startswith('V_') else 34753)
            assert r['status']=='COMPLETE'and r['complete']and r['optimizer_created']and r['first_optimizer_step_executed']
            assert (r['arm'],r['seed'])==(arm,seed)and r['parameters']==expected
            assert [(v['arm'],v['seed'])for v in attempts if v['attempt']==r['attempt']]==[(arm,seed)]
            for field,key in (('checkpoint','checkpoint_sha256'),('last_checkpoint','last_checkpoint_sha256'),('history','history_sha256')):
                assert sha(r[field])==r[key],(arm,seed,key)
            assert r['epochs']==len(history)and [v['epoch']for v in history]==list(range(1,len(history)+1))
            losses=[v['validation']['loss']for v in history];best=min(range(len(losses)),key=losses.__getitem__)
            assert r['best_epoch']==best+1 and r['best_validation_loss']==losses[best]
            assert r['source_sha256']['risk_models.py']=='3d61d0a9a93655e4ace102f84caaac2c9899164f428b7ca3c3e13801e7ec5054'
            assert r['source_sha256']['supervision.py']=='d6c6d4dfbf11592c33df532308b1775c24de24af64b62fe42b6eb276fd4f0fab'
            assert history[best]==r['best_epoch_diagnostics']and history[best]['validation']['frames']==235
            entry={k:deepcopy(r[k])for k in run_keys if k in r};entry.update(receipt=str(path),receipt_sha256=sha(path));runs.append(entry)
            summaries.append(dict(arm=arm,seed=seed,representative=seed==SEEDS[0],epochs=len(history),best_epoch=r['best_epoch'],
                train=history[best]['train'],validation=history[best]['validation'],receipt_sha256=sha(path)))
            all_receipts[arm,seed]=r;all_history[arm,seed]=history
    matched=[]
    for control,treatment in (('R_ORDER_F1_ZERO','R_ORDER_F1'),('V_ORDER_NULL','V_ORDER_RGB')):
        for seed in SEEDS:
            a,b=all_receipts[control,seed],all_receipts[treatment,seed];first,second=all_history[control,seed],all_history[treatment,seed]
            assert a['initial_state_sha256']==b['initial_state_sha256']and a['parameters']==b['parameters']
            for key in ('features_sha256','normalizer_sha256','extra_features_sha256','extra_feature_seal_sha256','extra_normalizer_sha256'):
                assert a['bindings'][key]==b['bindings'][key]
            for x,y in zip(first,second):
                for key in ('training_order_sha256','sampled_pair_sequence_sha256','sampled_pairs'):assert x[key]==y[key]
            matched.append(dict(control=control,treatment=treatment,seed=seed,initial_state_sha256=a['initial_state_sha256'],
                parameters_each=a['parameters'],common_epochs=min(len(first),len(second)),training_frame_order_exact=True,
                sampled_pair_sequence_exact=True,sampled_pair_counts_exact=True))
    backup=root/('TRAINING_MANIFEST_before_final18_'+prior[:12]+'.json')
    if not backup.exists():shutil.copyfile(manifest_path,backup)
    manifest.update(status='ALL18_OF18_COMPLETE_TRAINING_CLOSED',runs=runs,actual_optimizer_attempts=18,successful_new_fits=18,
        failed_optimizer_fits=0,pending_basic_groups=[],conditional_groups=[],conditional_V_executed=True,
        global_attempt_ledger_sha256=sha(root/'GLOBAL_FIT_ATTEMPTS.jsonl'),prior_basic12_manifest_sha256=prior,
        V_execution_receipt_sha256=sha(root/'V_TRAINING_EXECUTION_RECEIPT.json'),V_gate_sha256=sha(root/'V_GATE.json'),
        real_reference_decodes=0,training_closed_no_additional_fits=True,epoch=time.time())
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    closure=dict(schema='ALL18_RISK_INFORMATION_TRAINING_CLOSURE_V1',status='PASS',total_new_optimizer_attempts=18,
        successful_new_fits=18,failed_optimizer_fits=0,non_optimizer_CLI_failures=1,all18_best_last_history_sha_verified=True,
        all18_first_strict_val_min_verified=True,matched_controls=matched,training_manifest_sha256=sha(manifest_path),
        V_execution_receipt_sha256=sha(root/'V_TRAINING_EXECUTION_RECEIPT.json'),
        independent_F1_training_verification_sha256=sha(root/'independent_training_f1_v1/VERIFICATION.json'),
        independent_V_training_verification='PENDING_SEPARATE_POST_TRAINING_AUDIT',real_reference_decodes=0,
        real_RGB_training_reads=0,original_archive_modified=False,training_closed_no_additional_fits=True,summaries=summaries,
        reporting_source_sha256=sha(__file__))
    write(root/'ALL18_TRAINING_CLOSURE.json',closure)
    batches=[]
    for r in read(root/'R_F0_COST_AND_EXECUTION.json')['distinct_batch_wall_ledger']:
        batches.append(dict(batch_id=r['batch_id'],wall_seconds=r['wall_seconds'],CPU_user_seconds=r['user_CPU_seconds'],
            CPU_system_seconds=r['system_CPU_seconds'],peak_rss_kib=r['peak_RSS_KiB'],actual_exit_code=r['actual_exit_code'],
            scope=r['scope'],wall_is_upper_bound=False,process_cost_sha256=r['timer_sha256']))
    for r in read(root/'R_F1_COST_AND_EXECUTION.json')['distinct_batch_wall_ledger']:
        value=dict(r);value['actual_exit_code']=value.pop('exit_code');batches.append(value)
    batches.extend((process_cost(root,'V_THROUGHPUT'),process_cost(root,'TRAIN_V_SIX')))
    assert len({r['batch_id']for r in batches})==len(batches)==8
    cost=dict(schema='ALL18_TRAINING_COST_AND_EXECUTION_V1',status='PASS',actual_new_training_attempts=18,successful_new_fits=18,
        failed_optimizer_fits=0,non_optimizer_CLI_failures=1,actual_global_attempt_limit=18,training_closed=True,
        distinct_batch_wall_ledger=batches,measured_distinct_whole_CLI_wall_sum_seconds=sum(r['wall_seconds']for r in batches),
        measured_training_CLI_wall_sum_seconds=sum(r['wall_seconds']for r in batches if r['batch_id']!='V_THROUGHPUT'),
        measured_CPU_sum_seconds=sum(r['CPU_user_seconds']+r['CPU_system_seconds']for r in batches),
        max_single_process_peak_rss_kib=max(r['peak_rss_kib']for r in batches),
        peak_RSS_scope='Maximum of serial process highwater values; not additive RAM consumption',
        nested_per_fit_measurements={f"{r['arm']}_s{r['seed']}":r['cost']for r in runs},nested_timers_are_not_additional_batches=True,
        GPU_training_host_wall_upper_bound_seconds=batches[-1]['wall_seconds'],GPU_technical_probe_host_wall_upper_bound_seconds=batches[-2]['wall_seconds'],
        total_measured_GPU_host_wall_upper_bound_seconds=batches[-1]['wall_seconds']+batches[-2]['wall_seconds'],exact_CUDA_active_seconds=None,
        GPU_scope='Whole CLI host wall upper bounds include CPU preparation and I/O; CUDA stream probe intervals are not active-kernel integrals and are not added again',
        V_peak_CUDA_allocated_bytes=execution['GPU_peak_allocated_bytes'],unmeasured_startup_for_F0_F1_gradient_probes_seconds=None,
        source_editing_and_manual_review_seconds=None,unmeasured_note='Uninstrumented durations remain unknown; do not replace them with zero',
        historical_training_repeated=False,original_backbone_new_forward_count=0,real_reference_decodes=0,
        training_manifest_sha256=sha(manifest_path),training_closure_sha256=sha(root/'ALL18_TRAINING_CLOSURE.json'),
        V_internal_receipt_sha256=sha(root/'V_TRAINING_EXECUTION_RECEIPT.json'),V_probe_receipt_sha256=sha(root/'TRAIN_FORWARD_BACKWARD_THROUGHPUT_V.json'),
        prior_F0_cost_sha256=sha(root/'R_F0_COST_AND_EXECUTION.json'),prior_F1_cost_sha256=sha(root/'R_F1_COST_AND_EXECUTION.json'),
        original_exit134_technical_failure_receipt_sha256=sha(root/'INTEROP_CLI_FAILURE_V1.json'),reporting_source_sha256=sha(__file__))
    write(root/'TRAINING_COST_AND_EXECUTION.json',cost)
    return dict(manifest_sha256=sha(manifest_path),closure_sha256=sha(root/'ALL18_TRAINING_CLOSURE.json'),
        cost_sha256=sha(root/'TRAINING_COST_AND_EXECUTION.json'),V_execution_sha256=sha(root/'V_TRAINING_EXECUTION_RECEIPT.json'),
        whole_CLI_wall_sum_seconds=cost['measured_distinct_whole_CLI_wall_sum_seconds'],
        GPU_host_training_upper_seconds=cost['GPU_training_host_wall_upper_bound_seconds'],
        peak_CUDA_allocated_bytes=execution['GPU_peak_allocated_bytes'],Vmatched=[v for v in matched if v['control'].startswith('V_')],
        Vsummary=[dict(arm=v['arm'],seed=v['seed'],epochs=v['epochs'],best_epoch=v['best_epoch'],
            success=v['validation']['counts']['successes'],missed=v['validation']['counts']['missed_good_action'])for v in summaries if v['arm'].startswith('V_')])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    print(json.dumps(run(parser.parse_args().run_dir),allow_nan=False))
