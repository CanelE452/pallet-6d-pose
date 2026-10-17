"""Thin proof-driven campaign controller: run/resume/report, no scientific fit.

Runtime commands/paths are separate from CONFIG_FROZEN and PLAN_LOCK. Completed
proofs are verified and reused; independently runnable branches continue when
another branch needs input. Existing external jobs are adopted, never restarted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .common import canonical,read,sha,write

TERMINAL={'DONE','REJECTED_EVIDENCE','BLOCKED_INPUT','BLOCKED_DEPENDENCY',
          'NOT_APPLICABLE','FAILED_TECHNICAL','BUDGET_DEFERRED'}
_HASH_CACHE={}


def _file_sha(path):
    path=Path(path);stat=path.stat();key=(str(path.resolve()),stat.st_ino,stat.st_size,stat.st_mtime_ns)
    if key not in _HASH_CACHE:_HASH_CACHE[key]=sha(path)
    return _HASH_CACHE[key]


def _field(value,name):
    for part in name.split('.'):
        value=value[int(part)] if isinstance(value,list) else value[part]
    return value


def _proc(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':return None
        argv=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
        return dict(pid=int(pid),start_ticks=stat[19],argv=argv,
                    argv_sha256=hashlib.sha256(canonical(argv).encode()).hexdigest())
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def _same_live(handle):
    current=_proc(handle['pid']) if handle else None
    return current if current and all(current[k]==handle[k] for k in ('start_ticks','argv_sha256')) else None


def _external(stage):
    tokens=stage.get('external_match_argv')
    if not tokens:return None
    found=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        value=_proc(path.name)
        if value and all(token in value['argv'] for token in tokens):found.append(value)
    # Fork workers may inherit the parent's argv; only one main job is adopted.
    return min(found,key=lambda v:(int(v['start_ticks']),v['pid'])) if found else None


def _proof(stage):
    if stage.get('proof_alternatives'):
        for option in stage['proof_alternatives']:
            value=_proof(option)
            if value is not None:
                value['__completion_status__']=option.get('status','DONE')
                return value
        return None
    proofs=stage.get('proofs',[])
    if not proofs:return None
    bindings={}
    for item in proofs:
        path=Path(item['path'])
        if not path.is_file():return None
        digest=_file_sha(path)
        if item.get('sha256') and digest!=item['sha256']:raise ValueError('Proof changed: '+str(path))
        bindings[str(path)]=digest
        document=read(path) if item.get('fields') or item.get('bindings') else None
        for key,value in item.get('fields',{}).items():
            if _field(document,key)!=value:raise ValueError('Proof field mismatch: '+str(path)+'/'+key)
        for bound in item.get('bindings',[]):
            artifact=Path(bound['artifact'])
            if not artifact.is_file():return None
            actual=_file_sha(artifact)
            if _field(document,bound['field'])!=actual:raise ValueError('Sealed artifact mismatch: '+str(artifact))
            bindings[str(artifact)]=actual
    return bindings


def _verify_science(config):
    run=Path(config['run_dir']);plan=read(run/'PLAN_LOCK.json')
    if sha(run/'PLAN_LOCK.json')!=config['plan_lock_sha256']:raise ValueError('Frozen plan changed')
    frozen=Path(config.get('scientific_config',run/'CONFIG_FROZEN.json'))
    if sha(frozen)!=config['scientific_config_sha256'] or plan['config_sha256']!=sha(frozen):
        raise ValueError('Scientific configuration changed')
    if sha(run/'ARM_REGISTRY.json')!=plan['arm_registry_sha256']:raise ValueError('Frozen arm registry changed')
    source=Path(config['source_root'])/'scripts/research/pallet_statewise_end_to_end_recovery_v1'
    for name,digest in plan['core_source_sha256'].items():
        current=sha(source/name)
        if current==digest:continue
        # A documented pre-outcome input coverage repair may change the input
        # adapter, never the five modules used by live frozen batch execution.
        allowed=False
        if name=='inputs.py':
            for item in config.get('source_contract_amendments',[]):
                path=Path(item['path'])
                if sha(path)!=item['sha256']:raise ValueError('Input contract amendment changed')
                amendment=read(path)
                allowed|=(amendment.get('plan_lock_sha256')==sha(run/'PLAN_LOCK.json')
                    and amendment.get('scientific_config_sha256')==sha(frozen)
                    and amendment.get('model_or_hyperparameters_changed') is False
                    and amendment.get('primary_outcome_reads_before_repair')==0
                    and amendment.get('original_core_inputs_source_sha256')==digest
                    and amendment.get('corrected_source_sha256')==current)
        if not allowed:raise ValueError('Frozen live core source changed: '+name)
    return run,plan


def _save_state(run,state):
    state['updated_epoch']=time.time()
    write(run/'CONTROLLER_STATE.json',state,replace=True)


def report(run_dir):
    run=Path(run_dir)
    state=read(run/'CONTROLLER_STATE.json') if (run/'CONTROLLER_STATE.json').exists() else dict(stages={})
    verified=None;verification_error=None
    if state.get('runtime_config_path'):
        try:_verify_science(read(state['runtime_config_path']));verified=True
        except (ValueError,KeyError,OSError) as exc:verified=False;verification_error=repr(exc)
    result=dict(schema='PROOF_DRIVEN_CAMPAIGN_REPORT_V1',stages=state.get('stages',{}),
                status_counts={status:sum(v.get('status')==status for v in state.get('stages',{}).values())
                               for status in sorted(TERMINAL|{'RUNNING','PENDING'})},
                scientific_config_unchanged=verified,scientific_config_verification_error=verification_error,
                root_scoring_GO=state.get('root_scoring_GO',False),
                note='DONE requires current artifact proofs; publication/evaluation need explicit supplied hooks and root GO')
    write(run/'CONTROLLER_REPORT.json',result,replace=True)
    return result


def run(config_path,*,wait=True,poll_seconds=15.):
    config_path=Path(config_path);config=read(config_path);run_dir,plan=_verify_science(config)
    stages=config['stages'];names=[s['id'] for s in stages]
    if len(names)!=len(set(names)):raise ValueError('Unique ordered controller stage IDs required')
    if any(dep not in names for s in stages for dep in s.get('depends_on',[])+s.get('optional_depends_on',[])):
        raise ValueError('Unknown dependency stage')
    by_name={s['id']:s for s in stages};visiting=set();visited=set()
    def visit(name):
        if name in visiting:raise ValueError('Cyclic controller dependencies')
        if name in visited:return
        visiting.add(name)
        for dependency in by_name[name].get('depends_on',[])+by_name[name].get('optional_depends_on',[]):visit(dependency)
        visiting.remove(name);visited.add(name)
    for name in names:visit(name)
    state_path=run_dir/'CONTROLLER_STATE.json'
    state=read(state_path) if state_path.exists() else dict(schema='PROOF_DRIVEN_CAMPAIGN_STATE_V1',
        created_epoch=time.time(),runtime_config_path=str(config_path.resolve()),stages={},attempts=[])
    runtime_digest=sha(config_path)
    if state.get('runtime_config_sha256') not in (None,runtime_digest):
        with (run_dir/'RUNTIME_COMMAND_AMENDMENTS.jsonl').open('a') as stream:
            stream.write(canonical(dict(epoch=time.time(),previous_runtime_sha256=state['runtime_config_sha256'],
                new_runtime_sha256=runtime_digest,scientific_config_sha256=config['scientific_config_sha256'],
                method_or_hyperparameters_changed=False))+'\n')
    state.update(runtime_config_path=str(config_path.resolve()),runtime_config_sha256=runtime_digest,
                 scientific_config_verified=True,root_scoring_GO=config.get('primary_scoring_GO',False))
    processes={};wall_limit=float(plan['runtime_budget']['wall_hours'])*3600.
    campaign_start=float(config.get('campaign_start_epoch',plan['frozen_epoch']))
    while True:
        changed=False
        _verify_science(config)
        live_handles={s['id']:(_same_live(state['stages'].get(s['id'],{}).get('process')) or _external(s)) for s in stages}
        active=sum(int(s.get('cpu_workers',1)) for s in stages if live_handles[s['id']])
        for stage in stages:
            name=stage['id'];entry=state['stages'].setdefault(name,dict(status='PENDING'))
            definition=hashlib.sha256(canonical(stage).encode()).hexdigest()
            prior_definition=entry.get('definition_sha256')
            if entry.get('status') in ('DONE','RUNNING') and prior_definition not in (None,definition):
                raise ValueError('Completed/running runtime stage definition changed: '+name)
            if entry.get('status')=='FAILED_TECHNICAL' and prior_definition not in (None,definition):
                state['attempts'].append(dict(stage=name,preserved_failed_state=dict(entry),runtime_definition_amended=True))
                entry['status']='PENDING'
            entry['definition_sha256']=definition
            try:proof=_proof(stage)
            except (ValueError,KeyError,TypeError) as exc:
                entry.update(status='FAILED_TECHNICAL',reason=repr(exc));changed=True;continue
            if proof is not None:
                if entry.get('proof_sha256') and entry['proof_sha256']!=proof:
                    raise ValueError('Completed stage proof changed: '+name)
                completion=proof.get('__completion_status__','DONE')
                if entry['status']!=completion:changed=True
                entry.update(status=completion,proof_sha256=proof,completion_verified_epoch=time.time())
                continue
            if entry['status']=='DONE':raise ValueError('Completed stage artifact disappeared: '+name)
            live=live_handles[name]
            if live:
                entry.update(status='RUNNING',process=live,verified_live_epoch=time.time(),
                             adopted_external=not(name in processes))
                continue
            if entry['status']=='RUNNING':
                exit_code=processes[name].poll() if name in processes else None
                entry.update(status='FAILED_TECHNICAL',reason='Process terminal without required completion proof',exit_code=exit_code)
                state['attempts'].append(dict(stage=name,**entry));changed=True;continue
            if stage.get('skip_if_exists') and Path(stage['skip_if_exists']).exists():
                entry.update(status='NOT_APPLICABLE',reason=stage.get('skip_reason','Explicit prior branch result'))
                continue
            dependencies=[state['stages'].get(d,{}).get('status') for d in stage.get('depends_on',[])]
            if any(v not in ('DONE','NOT_APPLICABLE') for v in dependencies):
                entry.update(status='BLOCKED_DEPENDENCY' if all(v in TERMINAL for v in dependencies) else 'PENDING',
                             reason='Awaiting required proof-producing dependencies')
                continue
            optional=[state['stages'].get(d,{}).get('status') for d in stage.get('optional_depends_on',[])]
            if any(v not in TERMINAL for v in optional):
                entry.update(status='PENDING',reason='Await optional branch terminal evidence before final assembly')
                continue
            if stage.get('requires_primary_scoring_GO') and config.get('primary_scoring_GO') is not True:
                entry.update(status='BLOCKED_INPUT',reason='Root final prediction-seal scoring GO has not been supplied')
                continue
            if stage.get('not_applicable_reason'):
                entry.update(status='NOT_APPLICABLE',reason=stage['not_applicable_reason']);continue
            if not stage.get('command'):
                entry.update(status='BLOCKED_INPUT',reason=stage.get('blocked_reason','Completion hook or input not yet supplied'));continue
            if entry['status']=='FAILED_TECHNICAL':continue  # preserve; no blind reexecution
            if time.time()-campaign_start>=wall_limit:
                entry.update(status='BUDGET_DEFERRED',reason='Predeclared whole campaign wall cap');continue
            requested=int(stage.get('cpu_workers',1))
            if active+requested>int(plan['runtime_budget']['workers']):continue
            attempt=sum(a.get('stage')==name for a in state['attempts'])+1
            log=run_dir/'controller_logs'/f'{name}.attempt{attempt}.log';log.parent.mkdir(exist_ok=True)
            stream=log.open('x')
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',**stage.get('env',{})}
            process=subprocess.Popen(stage['command'],cwd=stage.get('cwd',config['source_root']),env=env,
                                     stdout=stream,stderr=subprocess.STDOUT);stream.close()
            processes[name]=process;handle=_proc(process.pid)
            entry.update(status='RUNNING',process=handle,command=stage['command'],log=str(log),
                         started_epoch=time.time(),adopted_external=False)
            state['attempts'].append(dict(stage=name,attempt=attempt,pid=process.pid,log=str(log),started_epoch=time.time()))
            active+=requested;changed=True
        _save_state(run_dir,state)
        running=[k for k,v in state['stages'].items() if v['status']=='RUNNING']
        pending=[k for k,v in state['stages'].items() if v['status']=='PENDING']
        print(canonical(dict(controller='CHECKPOINT',running=running,pending=pending,
              statuses={k:v['status'] for k,v in state['stages'].items()})),flush=True)
        if not wait or (not running and not pending):break
        if not running and not changed:break
        time.sleep(min(max(float(poll_seconds),1.),30.))
    return report(run_dir)


def default_runtime_config(run_dir,source_root,python):
    """Concrete current-campaign wiring; hooks remain explicit until supplied.

    This reads no reference or candidate labels. Earlier frozen provisioning
    receipts are reused; their absence requests that one input branch only.
    """
    run_dir=Path(run_dir).resolve();source_root=Path(source_root).resolve();syn=run_dir/'synthetic_v1'
    frozen=run_dir/'CONFIG_FROZEN.json';plan=read(run_dir/'PLAN_LOCK.json')
    prefix=[str(python),'-B','-m','scripts.research.pallet_statewise_end_to_end_recovery_v1.']
    def command(module,*args):return prefix[:-1]+[prefix[-1]+module]+list(map(str,args))
    def proof(path,fields=None,bindings=None):return dict(path=str(path),fields=fields or {},bindings=bindings or [])
    def batch_proof(folder,count):return proof(folder/'BATCH_SEAL.json',{'complete':True,'rows':count,'reference_inputs':False},
        [dict(field='predictions_sha256',artifact=str(folder/'PREDICTIONS.jsonl.gz')),
         dict(field='head_features_sha256',artifact=str(folder/'HEAD_FEATURES.jsonl.gz'))])
    def stage(name,phase,proofs,**kwargs):return dict(id=name,phase=phase,proofs=proofs,**kwargs)
    pilot=run_dir/'synthetic_pilot_batch_v1';expanded=run_dir/'synthetic_expansion_batch_v1'
    repair_path=run_dir/'INPUT_CONTRACT_REPAIR_V1.json'
    repaired=repair_path.exists()
    real=run_dir/('real_batch_all_available_v1' if repaired else 'real_batch_v1')
    synth_features=run_dir/'synthetic_features_v1';real_features=run_dir/'real_features_v1'
    labels=run_dir/'head_labels_v1';training=run_dir/'head_training_v1';learned=run_dir/'learned_real_v1'
    pinput=syn/'PILOT512_ROOT_FRAMES_AUTO_MASK.jsonl.gz';einput=syn/'EXPANSION1003_ROOT_FRAMES_AUTO_MASK.jsonl.gz'
    real_input=run_dir/'inputs_all_available_v1/PREDICTION_INPUTS238.jsonl.gz' if repaired else run_dir/'PREDICTION_INPUTS238.jsonl.gz'
    real_binding=real_input.parent/'INPUT_BINDING.json'
    stages=[
        stage('P0_BASELINES','P0',[proof(run_dir/'p0_baseline/BASELINE_REPRODUCTION.json',{'gate':'PASS'})]),
        stage('P1_REAL_INPUTS','P1',[proof(real_binding,{'frames':238,'reference_or_oracle_inputs':False},
            [dict(field='output_sha256',artifact=str(real_input))])],depends_on=['P0_BASELINES']),
        stage('P1_SYNTHETIC_PROVENANCE','P1',[proof(syn/'PROVISION_SEAL.json',{'rows':1003,'no_GT_mask_features':True,'failures':[]})]),
        stage('P1_AUTOMATIC_SYNTHETIC_MASKS','P1',[proof(syn/'EXPANSION1003_ROOT_FRAMES_AUTO_MASK_SEAL.json',
            {'rows':1003,'GT_masks_or_labels_read':False},[dict(field='predictions_sha256',artifact=str(einput))])],
            depends_on=['P1_SYNTHETIC_PROVENANCE']),
        stage('P2_P3_P4_PRECHECKS','P2_P3_P4',[proof(syn/'EXPANSION1003_ALL_FRAME_COMPILE_PRECHECK.json',{'passed':True,'reference_reads':0}),
            proof(syn/'NONE_PRIOR_CALIBRATION.json',{'PRIMARY_reference_reads':0,'real_reference_reads':0},[])],
            depends_on=['P1_AUTOMATIC_SYNTHETIC_MASKS']),
        stage('P5_SYNTHETIC_PILOT','P5',[batch_proof(pilot,512)],depends_on=['P2_P3_P4_PRECHECKS'],cpu_workers=2,
            external_match_argv=['scripts.research.pallet_statewise_end_to_end_recovery_v1.batch','--output',str(pilot)],
            command=command('batch','--inputs',pinput,'--config',frozen,'--output',pilot,'--workers','2')),
        stage('P5_REAL238','P5',[batch_proof(real,238)],depends_on=['P1_REAL_INPUTS','P2_P3_P4_PRECHECKS'],cpu_workers=2,
            external_match_argv=['scripts.research.pallet_statewise_end_to_end_recovery_v1.batch','--output',str(real)],
            command=command('batch','--inputs',real_input,'--config',frozen,'--output',real,'--workers','2')),
        stage('P5_EXTENSION_REUSE','P5',[proof(expanded/'EXTENSION_REUSE_RECEIPT.json',{'complete':True,'reused_frames':512,'new_pending_frames':491,'new_solves':0})],
            depends_on=['P5_SYNTHETIC_PILOT'],cpu_workers=1,command=command('extend_batch','--pilot-inputs',pinput,'--expanded-inputs',einput,
                '--expanded-input-seal',syn/'EXPANSION1003_ROOT_FRAMES_AUTO_MASK_SEAL.json','--config',frozen,'--pilot-batch',pilot,'--output',expanded)),
        stage('P5_SYNTHETIC_EXPANSION','P5',[batch_proof(expanded,1003)],depends_on=['P5_EXTENSION_REUSE'],cpu_workers=2,
            command=command('batch','--inputs',einput,'--config',frozen,'--output',expanded,'--workers','2'),
            external_match_argv=['scripts.research.pallet_statewise_end_to_end_recovery_v1.batch','--output',str(expanded)]),
        stage('P6_SYNTH_FEATURE_BRIDGE','P6',[proof(synth_features/'FEATURE_SEAL.json',{'rows':1003,'reference_reads':0},
            [dict(field='head_features_sha256',artifact=str(synth_features/'HEAD_FEATURES.jsonl.gz'))])],depends_on=['P5_SYNTHETIC_EXPANSION'],cpu_workers=1,
            command=command('feature_bridge','--inputs',einput,'--features',expanded/'HEAD_FEATURES.jsonl.gz','--output',synth_features)),
        stage('P6_REAL_FEATURE_BRIDGE','P6',[proof(real_features/'FEATURE_SEAL.json',{'rows':238,'reference_reads':0},
            [dict(field='head_features_sha256',artifact=str(real_features/'HEAD_FEATURES.jsonl.gz'))])],depends_on=['P5_REAL238'],cpu_workers=1,
            command=command('feature_bridge','--inputs',real_input,'--features',real/'HEAD_FEATURES.jsonl.gz','--output',real_features)),
        stage('P6_PREPARE_LABELS','P6',[proof(labels/'LABEL_SEAL.json',None,[dict(field='labels_sha256',artifact=str(labels/'SYNTHETIC_HEAD_TARGETS.jsonl.gz'))])],
            depends_on=['P6_SYNTH_FEATURE_BRIDGE'],cpu_workers=1,command=command('learned_head','prepare-labels','--features',synth_features/'HEAD_FEATURES.jsonl.gz',
                '--feature-seal',synth_features/'FEATURE_SEAL.json','--targets',syn/'SYNTH_GT_LABELS.jsonl.gz','--target-seal',syn/'PROVISION_SEAL.json',
                '--split-lock',syn/'DATA_SPLIT_LOCK.json','--output',labels)),
        stage('P6_TRAIN_SIX_FIXED_HEADS','P6',[],depends_on=['P6_PREPARE_LABELS'],cpu_workers=1,
            proof_alternatives=[dict(proofs=[proof(training/'TRAINING_MANIFEST.json',{'complete':True,'total_training_runs':6})]),
                dict(proofs=[proof(training/'TRAINING_NOT_RUN.json',{'status':'NOT_APPLICABLE_LABEL_SUPPORT_GATE','training_runs':0})],status='NOT_APPLICABLE')],
            command=command('learned_head','train','--features',synth_features/'HEAD_FEATURES.jsonl.gz','--feature-seal',synth_features/'FEATURE_SEAL.json',
                            '--labels-dir',labels,'--output',training)),
        stage('P6_LEARNED_REAL_PREDICTIONS','P6',[proof(learned/'PREDICTION_SEAL.json',{'rows':238,'reference_reads':0},
            [dict(field='predictions_sha256',artifact=str(learned/'PREDICTIONS.jsonl.gz'))])],
            depends_on=['P6_TRAIN_SIX_FIXED_HEADS','P6_REAL_FEATURE_BRIDGE'],cpu_workers=1,
            skip_if_exists=str(training/'TRAINING_NOT_RUN.json'),skip_reason='Declared synthetic label-support gate did not permit training',
            command=command('learned_head','predict','--features',real_features/'HEAD_FEATURES.jsonl.gz','--feature-seal',real_features/'FEATURE_SEAL.json',
                            '--training-dir',training,'--output',learned)),
        stage('ASSEMBLE_ALL_PREDICTIONS','FINAL',[],depends_on=['P5_REAL238'],optional_depends_on=['P6_LEARNED_REAL_PREDICTIONS'],
              blocked_reason='Root final assembly command/proof hook must be supplied'),
        stage('EVALUATE','FINAL',[],depends_on=['ASSEMBLE_ALL_PREDICTIONS'],requires_primary_scoring_GO=True,
              blocked_reason='Root final-seal GO and evaluator command/proof hook must be supplied'),
        stage('INDEPENDENT_VALIDATE','FINAL',[],depends_on=['EVALUATE'],requires_primary_scoring_GO=True,
              blocked_reason='Independent checker command/proof hook must be supplied'),
        stage('REPORT_AND_PUBLICATION','FINAL',[],depends_on=['INDEPENDENT_VALIDATE'],
              blocked_reason='Root report and safe publication command/proof hook must be supplied'),
    ]
    return dict(schema='STATEWISE_RECOVERY_RUNTIME_COMMAND_CONFIG_V1',run_dir=str(run_dir),source_root=str(source_root),
        scientific_config=str(frozen),scientific_config_sha256=sha(frozen),plan_lock_sha256=sha(run_dir/'PLAN_LOCK.json'),
        primary_scoring_GO=False,campaign_start_epoch=plan['frozen_epoch'],stages=stages,
        source_contract_amendments=[dict(path=str(repair_path),sha256=sha(repair_path))] if repaired else [],
        no_scientific_settings_in_runtime_config=True)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('run');a.add_argument('--config',required=True);a.add_argument('--no-wait',action='store_true')
    a=sub.add_parser('resume');a.add_argument('--run-dir',required=True);a.add_argument('--no-wait',action='store_true')
    a=sub.add_parser('report');a.add_argument('--run-dir',required=True)
    a=sub.add_parser('write-runtime-config');a.add_argument('--run-dir',required=True);a.add_argument('--source-root',required=True)
    a.add_argument('--python',required=True);a.add_argument('--output',required=True)
    a=p.parse_args()
    if a.action=='write-runtime-config':
        value=default_runtime_config(a.run_dir,a.source_root,a.python);write(a.output,value)
    elif a.action=='report':value=report(a.run_dir)
    elif a.action=='resume':value=run(read(Path(a.run_dir)/'CONTROLLER_STATE.json')['runtime_config_path'],wait=not a.no_wait)
    else:value=run(a.config,wait=not a.no_wait)
    print(canonical(value))


if __name__=='__main__':main()
