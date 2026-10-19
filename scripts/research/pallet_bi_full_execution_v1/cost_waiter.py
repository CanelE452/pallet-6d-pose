"""Metadata-only closure waiter and one-shot launcher for frozen master_costs.

Preparation writes a new V4 catalog/plan. Running requires a fresh ROOT GO,
waits for authenticated scientific+derived and four-service terminal receipts,
captures the one whitelisted fifth service after exit, then invokes only the
frozen metadata auditor. No scientific module is imported or executed here.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

MASTER_SHA='3c5aa52cab971a50d4cae16892636c43304e6b7573b30fdd2a5509a02b1713d9'
UNIT='pallet-bi-derived-completion-20261019-v1.service'
FOUR_UNITS={
 'pallet-bi-full-s2-recovery-20261019-v1.service',
 'pallet-bi-full-f-recovery-20261019-v1.service',
 'pallet-bi-full-s2-h1-20261019-v1.service',
 'pallet-bi-full-final-sequence-20261019-v1.service'}
PROPERTIES=('Id','ActiveState','SubState','MainPID','Result','ExecMainStatus','ExecMainCode',
 'CPUUsageNSec','MemoryPeak','ExecMainStartTimestampMonotonic','ExecMainExitTimestampMonotonic','NRestarts')
DERIVED_STAGES=['contrasts','export','gallery_S2','gallery_F','gallery_F_BI_VS_COMMON',
 'gallery_S2_CP_VS_C0','gallery_S2_CB_VS_CP','scatter']
SEQUENCE_STAGES=['S2_evaluation','S2_independent','S2_bank','S2_selection','S2_core','F_choose',
 'F_finish','F_evaluation','F_independent','F_bank','F_selection','F_core']
CUTOFF_SCOPE='scientific+derivedcomplete;cost/report/package/Gitpending'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def append(path,value):
    with Path(path).open('a')as stream:
        stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def verify_map(bindings):
    for path,digest in bindings.items():require(sha(path)==digest,'Changed bound input: '+str(path))


def source_guard(plan):
    verify_map(plan['sources']);verify_map(plan['fixed_inputs'])
    require(plan['sources'][plan['master_source']]==MASTER_SHA,'Frozen master source differs')
    require(sha(plan['catalog'])==plan['catalog_sha256'],'Reviewed V4 catalog changed during wait/finalization')


def version4(catalog,run):
    """Pure catalog projection; preparation does not execute an audit."""
    run=Path(run).resolve();result=deepcopy(catalog)
    require(result['run']==str(run),'Catalog run differs')
    ledger=str(run/'DERIVED_COMPLETION_V1/ACTUAL_COMMAND_LEDGER.jsonl')
    require(all(x['path']!=ledger for x in result['ledgers']),'Derived ledger already cataloged')
    result['ledgers'].append(dict(name='DERIVED_COMPLETION',path=ledger,parent_scope='DERIVED_COMPLETION_SERVICE',
        relation='Derived child costs are subsets of the separate inclusive fifth-service cgroup CPU; never add both'))
    result['metadata'].append(dict(name='EXTERNAL_DERIVED_SERVICE',
        path=str(run/'ROOT_DERIVED_SERVICE_FINAL_CAPTURE.json'),fields=['complete','CPU_seconds',
        'elapsed_wall_seconds','Result','ExecMainStatus','NRestarts','source_sha256','launch_receipt_sha256'],
        scope='Inclusive fifth-service cgroup CPU/elapsed after exited; derived ledger/controller timers are nested subsets; not campaign elapsed'))
    result['elapsed_measurement']=dict(path=str(run/'ROOT_COST_ELAPSED_CUTOFF.json'),epoch_field='measurement_cutoff_epoch')
    return result


def prepare(run,catalog_v3,output,master_output):
    run,catalog_v3,output,master_output=map(lambda p:Path(p).resolve(),(run,catalog_v3,output,master_output))
    require(not output.exists() and not master_output.exists(),'Require fresh waiter/master output namespaces')
    require(output.is_relative_to(run) and master_output.is_relative_to(run) and output!=master_output
        and not master_output.is_relative_to(output) and not output.is_relative_to(master_output),
        'Waiter/master outputs must be disjoint campaign subdirectories')
    folder=Path(__file__).resolve().parent;master=folder/'master_costs.py'
    require(sha(master)==MASTER_SHA,'Frozen master source changed')
    dp=run/'DERIVED_COMPLETION_V1/PLAN.json';dg=run/'DERIVED_COMPLETION_ROOT_GO_V1.json'
    derived=read(dp);approved=read(dg)
    require(approved['approved'] is True and approved['plan_sha256']==sha(dp)
        and approved['controller_sha256']==derived['controller_sha256']
        and derived['stage_order']==DERIVED_STAGES and derived['run']==str(run), 'Derived prepared PLAN/GO differs')
    sources={str(Path(__file__).resolve()):sha(__file__),str(master):MASTER_SHA,
        str(run/'ROOT_CAPTURE_SERVICE_LIFECYCLE.py'):sha(run/'ROOT_CAPTURE_SERVICE_LIFECYCLE.py')}
    sources.update(derived['sources'])
    fixed={str(p):sha(p)for p in (catalog_v3,dp,dg,run/'ROOT_DERIVED_COMPLETION_ACTUAL_LAUNCH.json')}
    fixed.update(derived['fixed_inputs'])
    sequence=read(derived['sequence_plan']);sources.update(sequence['sources']);fixed.update(sequence['fixed_inputs'])
    catalog_path=run/'MASTER_COSTS_METADATA_CATALOG_V4.json'
    require(not catalog_path.exists(),'Preserve previous V4 catalog')
    catalog=version4(read(catalog_v3),run)
    write(catalog_path,catalog)
    plan=dict(schema='METADATA_COST_WAITER_PLAN_V1',run=str(run),output=str(output),master_output=str(master_output),
        worktree=str(folder.parents[2]),python=sys.executable,master_source=str(master),master_source_sha256=MASTER_SHA,
        wrapper_source_sha256=sha(__file__),catalog=str(catalog_path),catalog_sha256=sha(catalog_path),
        derived_plan=str(dp),derived_GO=str(dg),derived_launch=str(run/'ROOT_DERIVED_COMPLETION_ACTUAL_LAUNCH.json'),
        lifecycle_root=str(run/'ROOT_SERVICE_LIFECYCLE_V1'),sources=sources,fixed_inputs=fixed,
        systemctl_unit=UNIT,systemctl_properties=list(PROPERTIES),poll_seconds=20,
        child_commands_authorized=1,new_scientific_calls=0,new_fits=0,new_reference_decodes=0,
        cutoff_scope=CUTOFF_SCOPE,epoch=time.time())
    output.mkdir(parents=True,exist_ok=False);write(output/'PLAN.json',plan)
    return plan


def verify_controller_seal(plan_path,GO_path,terminal,source_path,stage_order):
    plan,go,seal=read(plan_path),read(GO_path),read(terminal)
    verify_map(plan['sources']);verify_map(plan['fixed_inputs'])
    require(go['approved'] is True and go['plan_sha256']==sha(plan_path)
        and go['controller_sha256']==sha(source_path) and plan['controller_sha256']==sha(source_path),
        'Controller PLAN/GO/source mismatch')
    require(seal['complete'] is True and seal['plan_sha256']==sha(plan_path)
        and seal['ROOT_GO_sha256']==sha(GO_path) and seal['source_sha256']==sha(source_path)
        and seal['stage_order']==stage_order==plan['stage_order'], 'Controller terminal ancestry differs')
    root=Path(plan['output'])
    require(seal['actual_command_ledger_sha256']==sha(root/'ACTUAL_COMMAND_LEDGER.jsonl'), 'Controller ledger changed')
    require(set(seal['checkpoints'])=={s+'.json'for s in stage_order},'Incomplete controller stage set')
    for stage in stage_order:
        cp=root/'checkpoints'/(stage+'.json')
        require(sha(cp)==seal['checkpoints'][cp.name],'Controller checkpoint changed')
        c=read(cp)
        require(c['complete'] is True and c['identity']['input_bindings']['PLAN.json']==sha(plan_path)
            and c['identity']['input_bindings']['ROOT_GO']==sha(GO_path), 'Checkpoint PLAN/GO ancestry differs')
        verify_map({p:v for p,v in c['identity']['input_bindings'].items()if Path(p).is_absolute()})
        verify_map(c['completed_files'])
    return plan,seal


def verify_derived(plan):
    dp=read(plan['derived_plan']);root=Path(dp['output']);terminal=root/'DERIVED_COMPLETION_SEAL.json'
    sequence=read(dp['sequence_plan']);seqterminal=Path(dp['sequence_output'])/'FINAL_SEQUENCE_SEAL.json'
    _,seqseal=verify_controller_seal(dp['sequence_plan'],dp['sequence_GO'],seqterminal,
        Path(plan['master_source']).with_name('final_sequence.py'),SEQUENCE_STAGES)
    _,seal=verify_controller_seal(plan['derived_plan'],plan['derived_GO'],terminal,
        Path(plan['master_source']).with_name('derived_completion.py'),DERIVED_STAGES)
    require(seal['schema']=='FULL_CAMPAIGN_DERIVED_COMPLETION_SEAL_V1'
        and seal['upstream_sequence_seal_sha256']==sha(seqterminal)
        and seal['contrasts']==30 and seal['scalar_arms']==40 and seal['scalar_population_rows']==160
        and seal['scalar_frame_arm_rows']==9520 and seal['gallery_groups']==5 and seal['gallery_seed_count']==3
        and seal['scatter_datasets']==len(dp['scatter_pairs'])*3
        and seal['new_H1_fits']==seal['new_pose_solves']==seal['new_pose_metric_evaluations']==0
        and seal['raw_RGB_copied'] is False and seal['public_poses_or_GT'] is False
        and seal['all_outliers_and_failures_retained'] is True,'Derived full population/scope differs')
    verify_map(seal['generated_inputs_sha256']);verify_map(seal['public_contrasts_files'])
    require(sha(root/'ARTIFACT_SIZES.json')==seal['artifact_sizes_sha256'],'Derived artifact size receipt changed')
    return dict(derived_seal_sha256=sha(terminal),upstream_sequence_seal_sha256=sha(seqterminal),
        actual_command_ledger_sha256=sha(root/'ACTUAL_COMMAND_LEDGER.jsonl'),stage_count=len(DERIVED_STAGES))


def service_record(value,unit,*,successful=True):
    require(value['unit']==unit and value['returncode']==0 and value['Id']==unit
        and value['SubState']in('exited','failed','dead') and int(value['NRestarts'])==0,
        'Service is not terminal or silently restarted')
    if successful:require(value['Result']=='success' and int(value['ExecMainStatus'])==0,'Service exited unsuccessfully')
    cpu=int(value['CPUUsageNSec']);start=int(value['ExecMainStartTimestampMonotonic']);end=int(value['ExecMainExitTimestampMonotonic'])
    require(cpu>=0 and start>0 and end>=start,'Service accounting timestamp/counter unavailable')
    return dict(CPU_seconds=cpu/1e9,elapsed_wall_seconds=(end-start)/1e6,
        Result=value['Result'],ExecMainStatus=value['ExecMainStatus'],NRestarts=int(value['NRestarts']),
        peak_memory_bytes=value.get('MemoryPeak','UNKNOWN_NOT_ZERO'))


def verify_lifecycle(plan):
    root=Path(plan['lifecycle_root']);lock=read(root/'LOCK.json');end=read(root/'COMPLETION.json')
    require(lock['source_sha256']==plan['sources'][str(Path(plan['run'])/'ROOT_CAPTURE_SERVICE_LIFECYCLE.py')]
        and set(lock['units'])==FOUR_UNITS and end['source_sha256']==lock['source_sha256']
        and end['complete'] is True and end['scientific_completion_claim'] is False
        and end['snapshot_sha256']==sha(root/'SNAPSHOTS.jsonl'),'Lifecycle completion/source differs')
    rows=[json.loads(x)for x in (root/'SNAPSHOTS.jsonl').read_text().splitlines()if x.strip()]
    require(rows and rows[-1]==end['last_snapshot'],'Lifecycle terminal snapshot differs')
    units={r['unit']:r for r in rows[-1]['units']}
    require(set(units)==FOUR_UNITS,'Exactly four original units required')
    for unit,value in units.items():service_record(value,unit)
    return dict(completion_sha256=sha(root/'COMPLETION.json'),snapshot_sha256=end['snapshot_sha256'])


def validate_derived_launch(plan):
    launch=read(plan['derived_launch']);args=launch['actual_service_argv']
    expected=[plan['python'],'-B','-m','scripts.research.pallet_bi_full_execution_v1.derived_completion',
        'run','--plan',plan['derived_plan'],'--root-GO',plan['derived_GO']]
    require(launch['service_returncode']==0 and '--unit='+UNIT.removesuffix('.service')in args
        and '--property=Restart=no'in args and '--property=CPUAccounting=yes'in args
        and '--setenv=PYTHONPATH='+plan['worktree']in args
        and all('--setenv='+name+'=1'in args for name in ('MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'))
        and args[-len(expected):]==expected and launch['plan_sha256']==sha(plan['derived_plan'])
        and launch['GO_sha256']==sha(plan['derived_GO'])
        and launch['source_sha256']==read(plan['derived_plan'])['controller_sha256'],
        'Actual fifth-service launch binding differs')
    return sha(plan['derived_launch'])


def query_service(unit,runner=subprocess.run):
    require(unit==UNIT,'Only the reviewed fifth derived service may be queried')
    argv=['systemctl','--user','show',UNIT,'--property='+','.join(PROPERTIES)]
    before=resource.getrusage(resource.RUSAGE_CHILDREN);tick=time.monotonic()
    p=runner(argv,text=True,capture_output=True)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    value=dict(line.split('=',1)for line in p.stdout.splitlines()if '='in line)
    value.update(unit=UNIT,returncode=p.returncode,stderr=p.stderr)
    value.update(query_wall_seconds=time.monotonic()-tick,
        query_child_CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
    return value,argv


def guard_GO(plan_path,GO_path):
    plan,go=read(plan_path),read(GO_path)
    require(go.get('approved')is True and go.get('metadata_only')is True
        and go['wrapper_source_sha256']==sha(__file__)==plan['wrapper_source_sha256']
        and go['master_source_sha256']==MASTER_SHA==plan['master_source_sha256']
        and go['catalog_sha256']==sha(plan['catalog'])==plan['catalog_sha256']
        and go['plan_sha256']==sha(plan_path) and go['systemctl_unit']==UNIT
        and go['master_child_commands_authorized']==1 and go['new_scientific_calls_authorized']==0,
        'Fresh source/plan/catalog-bound ROOT metadata GO required')
    source_guard(plan)
    require(plan['systemctl_unit']==UNIT and plan['systemctl_properties']==list(PROPERTIES)
        and plan['poll_seconds']==20 and plan['cutoff_scope']==CUTOFF_SCOPE,
        'Prepared whitelist/resource/cutoff protocol differs')
    return plan


def completed_return(plan,GO_path):
    root=Path(plan['output']);terminal=root/'COMPLETION.json'
    if not terminal.exists():return None
    value=read(terminal)
    require(value['complete']is True and value['ROOT_GO_sha256']==sha(GO_path)
        and value['plan_sha256']==sha(root/'PLAN.json') and value['source_sha256']==sha(__file__),
        'Completed waiter ancestry differs')
    verify_map(value['output_bindings']);verify_derived(plan);verify_lifecycle(plan)
    return value


def run(plan_path,GO_path):
    plan=guard_GO(plan_path,GO_path);root=Path(plan['output']);runroot=Path(plan['run'])
    done=completed_return(plan,GO_path)
    if done is not None:return done
    # A failed/partially executed metadata child is preserved for explicit ROOT
    # disposition. It is never silently rerun or adopted by path existence.
    require(not(root/'RUN_STARTED.json').exists(),'Partial waiter output preserved; explicit ROOT disposition required')
    require(not Path(plan['master_output']).exists(),'Master output already exists; never overwrite or assume success')
    require(not any((runroot/leaf).exists()for leaf in ('ROOT_DERIVED_SERVICE_FINAL_CAPTURE.json',
        'ROOT_DERIVED_SERVICE_FINAL_CAPTURE_PUBLIC.json','ROOT_COST_ELAPSED_CUTOFF.json')),
        'Preserve prior service capture/cutoff; explicit ROOT disposition required')
    with(root/'OWNER.lock').open('a+')as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not(root/'RUN_STARTED.json').exists(),'Concurrent/partial waiter output detected')
        tick=time.monotonic();cpu=time.process_time()
        write(root/'RUN_STARTED.json',dict(plan_sha256=sha(plan_path),ROOT_GO_sha256=sha(GO_path),
            source_sha256=sha(__file__),actual_argv=[v.decode()for v in Path('/proc/self/cmdline').read_bytes().split(b'\0')if v],
            epoch=time.time(),new_scientific_calls=0))
        try:
            terminal=Path(read(plan['derived_plan'])['output'])/'DERIVED_COMPLETION_SEAL.json'
            lifecycle=Path(plan['lifecycle_root'])/'COMPLETION.json'
            while not(terminal.exists()and lifecycle.exists()):
                source_guard(plan)
                print(json.dumps(dict(event='WAIT_TERMINAL_METADATA',derived_present=terminal.exists(),
                    lifecycle_present=lifecycle.exists(),new_scientific_calls=0)),flush=True)
                time.sleep(plan['poll_seconds'])
            source_guard(plan);derived_binding=verify_derived(plan);lifecycle_binding=verify_lifecycle(plan)
            launch_sha=validate_derived_launch(plan)
            while True:
                value,argv=query_service(plan['systemctl_unit'])
                append(root/'SERVICE_QUERY_LEDGER.jsonl',dict(epoch=time.time(),actual_argv=argv,record=value,new_scientific_calls=0))
                require(value['returncode']==0,'Failed querying the whitelisted fifth service')
                if value.get('SubState')in('exited','failed','dead'):break
                time.sleep(plan['poll_seconds'])
            external=dict(complete=True,unit=UNIT,source_sha256=sha(__file__),launch_receipt_sha256=launch_sha,
                raw_properties=value,query_argv=argv,epoch=time.time(),
                new_scientific_calls=0,relation='Inclusive fifth-service cgroup contains derived ledger/controller costs; never add nested scopes')
            external.update(service_record(value,UNIT,successful=False))
            capture=runroot/'ROOT_DERIVED_SERVICE_FINAL_CAPTURE.json';write(capture,external)
            public_capture=runroot/'ROOT_DERIVED_SERVICE_FINAL_CAPTURE_PUBLIC.json'
            write(public_capture,{k:external[k]for k in ('complete','CPU_seconds','elapsed_wall_seconds','Result',
                'ExecMainStatus','NRestarts','source_sha256','launch_receipt_sha256','relation')})
            service_record(value,UNIT)
            source_guard(plan);verify_derived(plan);verify_lifecycle(plan)
            cutoff=runroot/'ROOT_COST_ELAPSED_CUTOFF.json'
            write(cutoff,dict(schema='ROOT_COST_ELAPSED_MEASUREMENT_CUTOFF_V1',campaign_complete=False,
                measurement_cutoff_epoch=time.time(),scope=CUTOFF_SCOPE,source_sha256=sha(__file__),
                plan_sha256=sha(plan_path),ROOT_GO_sha256=sha(GO_path),
                derived_binding=derived_binding,lifecycle_binding=lifecycle_binding,
                derived_service_capture_sha256=sha(capture),new_scientific_calls=0))
            child_GO=root/'MASTER_CHILD_ROOT_GO.json'
            write(child_GO,dict(approved=True,metadata_only=True,auditor_source_sha256=MASTER_SHA,
                catalog_sha256=sha(plan['catalog']),delegating_ROOT_GO_sha256=sha(GO_path),
                wrapper_source_sha256=sha(__file__),derived_binding=derived_binding,lifecycle_binding=lifecycle_binding,
                cost_cutoff_sha256=sha(cutoff),new_scientific_calls_authorized=0))
            source_guard(plan)
            argv=[plan['python'],'-B','-m','scripts.research.pallet_bi_full_execution_v1.master_costs',
                '--catalog-path',plan['catalog'],'--root-go',str(child_GO),'--output',plan['master_output']]
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=plan['worktree'])
            before=resource.getrusage(resource.RUSAGE_CHILDREN);child_tick=time.monotonic()
            append(root/'ACTUAL_COMMAND_LEDGER.jsonl',dict(event='START',phase='frozen_master_costs',argv=argv,epoch=time.time()))
            with(root/'MASTER_CHILD.log').open('x')as log:
                code=subprocess.call(argv,cwd=plan['worktree'],env=env,stdout=log,stderr=subprocess.STDOUT)
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            child=dict(event='COMPLETE'if code==0 else'TECHNICAL_FAILURE',phase='frozen_master_costs',argv=argv,
                return_code=code,child_CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                wall_seconds=time.monotonic()-child_tick,epoch=time.time(),
                timer_scope='External child start to exit; includes import, output encoding/writes and teardown omitted by internal timer')
            append(root/'ACTUAL_COMMAND_LEDGER.jsonl',child)
            require(code==0,'Frozen metadata auditor child failed; partial output retained')
            source_guard(plan)
            output=Path(plan['master_output']);cost=read(output/'MASTER_COSTS.json');public=read(output/'PUBLIC_COST_SUMMARY.json')
            require(cost['auditor_source_sha256']==MASTER_SHA
                and cost['H1']['all_expected_fits_complete']is True and cost['H1']['completed_new_fits']==33
                and cost['H1']['all_declared_prediction_seals_complete']is True
                and cost['H1']['tiny_head_real_prediction_forwards']==7854
                and public['private_master_receipt_sha256']==sha(output/'MASTER_COSTS.json')
                and cost['campaign_elapsed_wall_seconds']=='UNKNOWN_NOT_ZERO'
                and cost['campaign_elapsed']['measurement_cutoff_scope']==CUTOFF_SCOPE,
                'Final cost/H1 readiness/cutoff output failed verification')
            bindings={str(p):sha(p)for p in (capture,public_capture,cutoff,child_GO,
                root/'ACTUAL_COMMAND_LEDGER.jsonl',root/'SERVICE_QUERY_LEDGER.jsonl',
                output/'MASTER_COSTS.json',output/'MASTER_COSTS.md',output/'PUBLIC_COST_SUMMARY.json')}
            completed=dict(schema='METADATA_COST_WAITER_COMPLETION_V1',complete=True,
                source_sha256=sha(__file__),plan_sha256=sha(plan_path),ROOT_GO_sha256=sha(GO_path),
                output_bindings=bindings,actual_master_child=child,derived_binding=derived_binding,
                lifecycle_binding=lifecycle_binding,controller_wall_seconds=time.monotonic()-tick,
                controller_CPU_seconds=time.process_time()-cpu,new_scientific_calls=0,new_fits=0,
                own_final_receipt_write_CPU='UNKNOWN_NOT_ZERO',campaign_complete=False,
                scope='Cost metadata complete with historical unknowns; reporting/package/Git still pending',epoch=time.time())
            write(root/'COMPLETION.json',completed)
            return completed
        except Exception as error:
            if not(root/'TECHNICAL_FAILURE.json').exists():
                write(root/'TECHNICAL_FAILURE.json',dict(complete=False,error=repr(error),epoch=time.time(),
                    source_sha256=sha(__file__),new_scientific_calls=0,partial_outputs_preserved=True))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for name in ('run','catalog-v3','output','master-output'):a.add_argument('--'+name,required=True)
    a=sub.add_parser('run');a.add_argument('--plan',required=True);a.add_argument('--root-GO',required=True)
    a=p.parse_args()
    if a.command=='prepare':print(json.dumps(prepare(a.run,a.catalog_v3,a.output,a.master_output),sort_keys=True))
    else:print(json.dumps(run(a.plan,a.root_GO),sort_keys=True))


if __name__=='__main__':main()
