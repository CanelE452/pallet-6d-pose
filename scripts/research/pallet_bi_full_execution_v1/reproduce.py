"""Read-only reproduction inventory; never launches scientific work.

Literal argv comes only from stored execution records. Plans and supported
resume APIs are separate from executed commands. No RGB, model tensor, target
label, canonical reference or per-frame pose archive is decoded by this tool.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex

PLAN_SHA='e5b965e312fc96e6ef245d868285be2bf48e7ba4ee5a7dc5a612368d9485c023'
INSTRUCTION_SHA='208663999af68480922a6f8d2c9cfc1f26abcc6c267ca62356b6110099f6f88c'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def jsonrows(path):
    with Path(path).open()as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def sealed_metadata(run,relative):
    path=Path(run)/relative
    if not path.exists():return dict(path=str(path),exists=False,status='PENDING')
    value=read(path)
    # This is a metadata checkpoint,not recursive scientific artifact audit.
    return dict(path=str(path),exists=True,sha256=sha(path),status=value.get('status'),
        complete=value.get('complete')is True or value.get('PASS')is True or value.get('status')in ('COMPLETE_NEW','COMPLETE_EXACT_REUSE'),rows=value.get('rows'),
        artifact_hashes_declared=len(value.get('artifacts',value.get('output_artifacts',{}))))


def extension_metadata(run,source_map,record):
    """Explicit metadata paths only; Python API records never become argv."""
    run=Path(run).resolve()
    if source_map is None:return {},[],{},[]
    path=Path(source_map);value=read(path)
    if value.get('schema')!='REPRODUCTION_METADATA_SOURCES_V1':raise ValueError('Explicit reproduction metadata source-map required')
    binding=dict(path=str(path),sha256=sha(path));missing=[];apis=[]
    def metadata_path(relative):
        p=(run/relative).resolve()
        if run not in p.parents or p.suffix not in ('.json','.jsonl')or any(word in p.name.upper()for word in ('REFERENCE','LABEL','NORMALIZER'))or p.name in ('FRAME.jsonl','RESULT.jsonl','PREDICTIONS.jsonl'):
            raise ValueError('Only explicit run metadata paths may be registered')
        if not p.exists():missing.append(str(p))
        return p
    for relative in value.get('ledgers',[]):
        p=metadata_path(relative)
        if not p.exists():continue
        raw=p.read_bytes();captured=hashlib.sha256(raw).hexdigest()
        for index,line in enumerate(raw.splitlines()):
            if not line.strip():continue
            event=json.loads(line)
            if (event.get('subprocess')is False and str(event.get('event','')).startswith('METADATA_'))or event.get('event')=='CHECKPOINT_ACTION':
                apis.append(dict(path=str(p),sha256=captured,event_index=index,
                    captured_source_bytes=len(raw),CLI_invocation_claimed=False,
                    scope='Recorded metadata function call,not a subprocess command',
                    explicit_API_metadata={k:event[k]for k in ('event','phase','function_call','action','checkpoint_sha256','wall_seconds','controller_CPU_seconds','new_metric_calculations','epoch')if k in event}))
                continue
            phase=event.get('phase')
            if phase is None:
                if event.get('event')!='CONTROLLER_START':raise ValueError('Unrecognized phase-free command event')
                phase='CONTROLLER_START'
            argv=event.get('argv')
            if argv is None and event.get('event')=='CONTROLLER_START':argv=event.get('actual_argv')
            record(p,phase,argv,event['event'],event_index=index,
                returncode=event.get('return_code',event.get('returncode')),wall_seconds=event.get('wall_seconds'),child_CPU_seconds=event.get('child_CPU_seconds'),
                captured_source_sha256=captured,captured_source_bytes=len(raw),source_read_scope='exact byte snapshot;future append does not retroactively change this prefix')
    keys=('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','LP_NUM_THREADS','PYOPENGL_PLATFORM','EGL_DEVICE_ID')
    for relative in value.get('process_snapshots',[]):
        p=metadata_path(relative)
        if not p.exists():continue
        snapshot=read(p)
        for role,event in snapshot['processes'].items():
            record(p,role,event['argv'],'ACTUAL_PROCESS_SNAPSHOT_NOT_COMPLETION',pid_at_snapshot=event['pid'],
                cwd=snapshot.get('working_directory'),environment={k:v for k,v in snapshot.get('launch_environment',{}).items()if k in keys})
    for item in value.get('launch_receipts',[]):
        p=metadata_path(item['path'])
        if not p.exists():continue
        receipt=read(p);record(p,item['phase'],receipt[item.get('argv_key','argv')],'RECORDED_LAUNCH_RECEIPT_NOT_COMPLETION',
            metadata_scope='Literal outer controller launch; child API work is not invented as CLI')
    for item in value.get('API_records',[]):
        p=metadata_path(item['path'])
        if not p.exists():continue
        receipt=read(p);apis.append(dict(path=str(p),sha256=sha(p),scope=item['scope'],
            explicit_API_metadata={key:receipt[key]for key in item['fields']},CLI_invocation_claimed=False))
    states={}
    for name,relative in value.get('state_overrides',{}).items():
        p=metadata_path(relative);states[name]=str(p)
    return states,apis,binding,missing


def inventory(run,metadata_sources=None):
    run=Path(run).resolve()
    if sha(run/'CAMPAIGN_PLAN.json')!=PLAN_SHA or sha(run/'CLI_PALLET_BI_FULL_EXECUTION_20261019.txt')!=INSTRUCTION_SHA:
        raise ValueError('Authorized frozen campaign/instruction bytes differ')
    commands=[];command_sources={}
    def record(path,phase,argv,status,**metadata):
        if not isinstance(argv,list)or not argv or not all(isinstance(a,str)for a in argv):raise ValueError('Literal recorded argv list required')
        path=Path(path);digest=metadata.pop('captured_source_sha256',None)or sha(path);command_sources[str(path)]=digest
        commands.append(dict(phase=phase,argv=argv,shell_display=shlex.join(argv),execution_status=status,
            evidence_path=str(path),evidence_sha256=digest,**metadata))
    source=run/'S1_EXECUTED_COMMANDS.json';s1=read(source)
    for row in s1['commands']:
        record(source,row['stage'],row['argv'],'RECORDED_EXECUTED_SUCCESSFUL'if row.get('actual_executed')else'NOT_CONFIRMED_EXECUTED',cwd=row.get('cwd'),
            scope=s1['scientific_argv_scope'])
    material=run/'s3_s4_production/ACTUAL_ARGV.json'
    if material.exists():
        value=read(material);full=run/'s3_s4_production/FULL_RECEIPT.json'
        complete=full.exists()and read(full).get('complete')is True and read(full).get('all1241_completed')is True
        record(material,'S3_S4_FULL_PRODUCER',value['argv'],'COMPLETE_FROM_TERMINAL_PRODUCER_SEAL'if complete else'RECORDED_STARTED_NOT_TERMINALLY_COMPLETE',
            cwd=value.get('cwd'),environment=value.get('environment'),terminal_seal_sha256=sha(full)if full.exists()else None)
    s2=run/'S2_AGENT_HANDOFF.json'
    if s2.exists():
        value=read(s2);argv=value['producer']['argv']
        producer_root=Path(argv[argv.index('--output')+1])if '--output'in argv else Path(value['producer']['root'])
        summary=producer_root/'SUMMARY.json'
        complete=summary.exists()and read(summary).get('status')=='COMPLETE_NEW'
        record(s2,'S2_HISTORICAL_PRODUCER_FROM_HANDOFF',argv,
            'COMPLETE_FROM_TERMINAL_PRODUCER_SEAL'if complete else'LAUNCHED_FROM_HANDOFF_NOT_TERMINALLY_COMPLETE',
            terminal_seal_sha256=sha(summary)if summary.exists()else None)
    frontend=run/'F_PRECOMPUTE_FRONTEND_EXECUTION.json'
    if frontend.exists():
        for index,value in enumerate(read(frontend).get('commands',[])):
            record(frontend,'F_FRONTEND_RECORDED_CHILD_'+str(index),value['argv'],
                'CHILD_RETURNED_ZERO_CONTROLLER_RECEIPT_INCOMPLETE'if value.get('returncode')==0 else'CHILD_TECHNICAL_FAILURE',
                returncode=value.get('returncode'),wall_seconds=value.get('wall_seconds'))
    ledger_binding_supplements=[]
    for relative in ('s3_s4_postprocess/POSTPROCESS_LEDGER.jsonl',
                     'ROOT_S3_S4_POSTSEAL_EXECUTION/POSTPROCESS_LEDGER.jsonl',
                     'DERIVED_VISUAL_EXECUTION/POSTPROCESS_LEDGER.jsonl'):
        ledger=run/relative
        if not ledger.exists():continue
        events=list(jsonrows(ledger))
        if relative.startswith('ROOT_S3_S4_POSTSEAL_EXECUTION/'):
            supplement_path=ledger.parent/'COMPLETION_LEDGER_BINDING_SUPPLEMENT.json'
            supplement=read(supplement_path);original=ledger.parent/'COMPLETE.json'
            snapshot=ledger.parent/supplement['first_four_component_ledger_snapshot']
            if sha(original)!=supplement['original_complete_sha256']or sha(snapshot)!=supplement['snapshot_sha256']:
                raise ValueError('Original root completion/prefix snapshot changed')
            if read(original)['actual_command_ledger_sha256']!=sha(snapshot):
                raise ValueError('Preserved root completion does not bind its exact prefix snapshot')
            if sha(ledger)!=supplement['current_full_ledger_sha256']or len(events)!=supplement['current_full_ledger_events']:
                raise ValueError('Root appended ledger differs from its completion supplement')
            prefix=b''.join(ledger.read_bytes().splitlines(keepends=True)[:supplement['original_completion_prefix_events']])
            if prefix!=snapshot.read_bytes():raise ValueError('Preserved root completion prefix is not exact')
            ledger_binding_supplements.append(dict(path=str(supplement_path),sha256=sha(supplement_path),
                original_completion_sha256=sha(original),original_prefix_sha256=sha(snapshot),
                current_ledger_sha256=sha(ledger),exact_append_only_prefix_verified=True))
        for event_index,event in enumerate(events):
            record(ledger,event['phase'],event['argv'],event['event'],event_index=event_index,
                returncode=event.get('return_code'),wall_seconds=event.get('wall_seconds'),
                child_CPU_seconds=event.get('child_CPU_seconds'))
    live=run/'ACTUAL_LIVE_ARGV_PROVENANCE_01.json'
    if live.exists():
        for event in read(live)['records']:
            record(live,event['role'],event['argv'],'ACTUAL_LIVE_PROCESS_SNAPSHOT_NOT_COMPLETION',
                pid_at_snapshot=event['pid'],cwd=event['cwd'],environment=event['selected_environment'])
    if metadata_sources is None and(run/'REPRODUCE_METADATA_SOURCES.json').exists():metadata_sources=run/'REPRODUCE_METADATA_SOURCES.json'
    overrides,api_records,metadata_source_binding,missing_metadata_sources=extension_metadata(run,metadata_sources,record)
    attempts_path=run/'GLOBAL_H1_FIT_ATTEMPTS.jsonl';attempts=list(jsonrows(attempts_path));fits=[]
    if len(attempts)>36:raise ValueError('Global registered H1 attempt budget exceeded')
    for row in attempts:
        path=Path(row['output'])/f"{row['arm']}_s{row['seed']}"/'RUN_RECEIPT.json';complete=False
        if path.exists():
            receipt=read(path)
            if receipt.get('arm')!=row['arm']or receipt.get('seed')!=row['seed']or receipt.get('attempt')!=row['attempt']:
                raise ValueError('Registered fit and saved execution receipt differ')
            complete=receipt.get('complete')is True
        fits.append(dict(arm=row['arm'],seed=row['seed'],attempt=row['attempt'],complete=complete,receipt=str(path),receipt_sha256=sha(path)if path.exists()else None))
    names={'P0':'P0_REUSE_PROOF.json','S1':'S1_EXECUTION_SUMMARY.json','D_FIXED_GAMMA':'D_FIXED_COMMON_GRAPH_FIXED_GAMMA/SUMMARY.json',
        'S2_PRODUCER':'S2_FULL_UNIT_WEIGHT_THREE_CPU/SUMMARY.json','S2_OWN_H1_PREDICTIONS':'S2_UNIT_WEIGHT_THREE_CPU_H1/ALL_REAL_PREDICTIONS_SEAL.json',
        'S3_S4_PRODUCER':'s3_s4_production/FULL_RECEIPT.json','S3_S4_OWN_H1_PREDICTIONS':'s3_s4_postprocess/FINISH_RECEIPT.json',
        'S3_S4_HANDOFF':'S3_S4_HANDOFF_SEAL.json','F_OBSERVATIONS':'F_OBSERVATIONS/OBSERVATION_MANIFEST.json',
        'F_OWN_CAL_PACKETS':'F_CALIBRATION/PACKET_MANIFEST.json','F_CORE_BANK':'F_CORE_THREE_CPU/SUMMARY.json','F_FINAL_CHOICE':'F_SYSTEM_CHOICE.json'}
    names.update(overrides)
    states={name:sealed_metadata(run,path)for name,path in names.items()}
    return dict(schema='READ_ONLY_REPRODUCTION_INVENTORY_DRAFT_V1',run=str(run),frozen_plan_sha256=PLAN_SHA,instruction_sha256=INSTRUCTION_SHA,
        environment_sha256=sha(run/'ENVIRONMENT_INITIAL.json'),states=states,commands=commands,command_source_hashes=command_sources,
        ledger_binding_supplements=ledger_binding_supplements,
        metadata_source_map_binding=metadata_source_binding,explicit_API_records=api_records,
        registered_missing_metadata_paths=missing_metadata_sources,
        fit_attempts_registered=len(attempts),fit_budget=36,completed_fits_with_actual_receipts=sum(r['complete']for r in fits),fit_receipts=fits,
        no_scientific_commands_executed_by_this_entrypoint=True,recursive_large_artifact_verification_performed=False,
        command_record_count_is_not_process_invocation_count=True,
        remaining_literal_argv_gaps=['P0, D and earlier S1 scoring launcher argv are not completely recorded; the five S3/S4 postseal commands are recorded exactly in the root ledger.',
            'H1/F controller child API calls remain API records; they are not separate invented CLI launches.'],
        incomplete_registered_S2_F_states=[name for name,state in states.items()if name.startswith(('S2_','F_'))and not state.get('complete')],
        whole_campaign_complete=states.get('CAMPAIGN_FINAL',{}).get('complete')is True)


def write_snapshot(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);content=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if path.exists()and path.read_text()!=content:raise ValueError('Snapshot changed;choose a fresh output path')
    path.write_text(content)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--metadata-sources',type=Path)
    parser.add_argument('--show-commands',action='store_true');args=parser.parse_args();value=inventory(args.run,args.metadata_sources)
    if args.snapshot:write_snapshot(args.snapshot,value)
    if args.show_commands:
        for command in value['commands']:print(command['execution_status']+' | '+command['phase']+'\n'+command['shell_display'])
    else:print(json.dumps({k:value[k]for k in ('fit_attempts_registered','fit_budget','completed_fits_with_actual_receipts','no_scientific_commands_executed_by_this_entrypoint','remaining_literal_argv_gaps')},ensure_ascii=False))
