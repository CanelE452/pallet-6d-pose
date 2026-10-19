"""Explicit F core resume/authentication, without changing its running controller.

Completed banks are read-only authenticated. A partial resume requires a new
root resource GO, checks for an active owner, snapshots every mutable prior
aggregate/job receipt, then invokes the unchanged ordered dispatcher. This
adapter never trains a head, chooses a final system, or reads references.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ARMS=('F_COMMON','F_BI')
COUNTS={'real':238,'synthetic':1003}


def read(path):
    path=Path(path)
    with (gzip.open if path.suffix=='.gz'else open)(path,'rt')as stream:return json.load(stream)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=content:raise ValueError('Preserved resume artifact differs: '+str(path))
    else:path.write_text(content)


def rows(path):
    with gzip.open(path,'rt')as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def authenticate_cross_arm_origin(output,lock,domain,ordinal,arm,seal,receipt):
    """Preserved pilot headers keep their original arm; require exact lineage."""
    if receipt.get('exact_reused')is not True:raise ValueError('Cross-arm preserved seal requires an exact-reuse execution envelope')
    reuse=lock.get('exact_reuse_binding')
    if not reuse:raise ValueError('Cross-arm preserved seal requires registered exact-reuse staging')
    root=Path(reuse['root']);mp=root/'REUSE_MANIFEST.json'
    if sha(mp)!=reuse['manifest_sha256']:raise ValueError('Registered exact-reuse manifest changed')
    manifest=read(mp)
    if manifest['packet_seal_sha256']!=lock['packet_seal_sha256']:raise ValueError('Cross-arm reuse packet manifest differs')
    matches=[r for r in manifest['rows']if(r['domain'],r['ordinal'],r['arm'])==(domain,ordinal,arm)]
    if len(matches)!=1:raise ValueError('Unique cross-arm staging/origin proof required')
    row=matches[0];staged=root/arm/domain/f'{ordinal:04d}';origin=Path(row['source'])
    actual=Path(output)/arm/domain/f'{ordinal:04d}';seal_sha=sha(actual/'FRAME_SEAL.json')
    if any(sha(folder/'FRAME_SEAL.json')!=seal_sha for folder in (staged,origin))or row['source_frame_seal_sha256']!=seal_sha:
        raise ValueError('Cross-arm original/staged/current frame seals differ')
    if row['desired_cache_payload_sha256']!=seal['cache_payload_sha256']:raise ValueError('Cross-arm exact requested payload differs')
    for name,value in seal['artifacts'].items():
        if sha(staged/name)!=value or sha(origin/name)!=value:raise ValueError('Cross-arm original/staged actual artifacts differ')
    proof=receipt.get('pilot_reuse_proof')
    if proof is not None:
        if Path(proof['source_folder']).resolve()!=staged.resolve()or proof['source_frame_seal_sha256']!=seal_sha:
            raise ValueError('Execution pilot reuse origin differs from registered staging')
        if proof['copied_artifact_sha256']!=seal['artifacts']or proof.get('dependency_payload_equality_verified')is not True or proof.get('pilot_source_preserved')is not True:
            raise ValueError('Execution pilot exact-copy proof differs')
    if receipt.get('reuse_origin')not in ('SEALED_PILOT','PRIOR_SEALED_CURRENT_ROW'):raise ValueError('Unsupported cross-arm reuse execution origin')
    return dict(original_preserved_arm=seal['arm'],requested_arm=arm,origin=str(origin),staging_manifest_sha256=sha(mp),
        exact_frame_seal_sha256=seal_sha,requested_payload_sha256=seal['cache_payload_sha256'])


def authenticate_bank(output,lock,packets):
    """Byte/identity/receipt audit only; no optimizer or feature evaluator."""
    output=Path(output);lock_sha=sha(output/'PRODUCER_LOCK.json');missing=[];row_bindings=[];features={}
    stored_calls=0;sealed_original_calls=0;unreceipted=[];unsealed=[];cross_arm=[]
    source=lock['numerical_kernel_source_binding'];helper=lock['driver_sha256']
    for domain,ids in lock['static_ordered_row_ids'].items():
        for arm in lock['arms']:
            for ordinal,fid in enumerate(ids):
                folder=output/arm/domain/f'{ordinal:04d}';sp=folder/'FRAME_SEAL.json';rp=folder/'ROW_EXECUTION.json'
                if not sp.exists():
                    missing.append(dict(domain=domain,arm=arm,ordinal=ordinal,id=fid))
                    unsealed.extend(str(path.relative_to(output))for path in folder.glob('*')if path.is_file())
                    if rp.exists():raise ValueError('Execution envelope without its original frame seal')
                    continue
                seal=read(sp)
                if any(seal.get(k)!=v for k,v in dict(id=fid,domain=domain,ordinal=ordinal,complete=True,
                    source_binding=source,numeric_reference_inputs=False).items()):raise ValueError('Completed frame identity/source contract differs')
                if set(seal['artifacts'])!={'RESULT.json.gz','FEATURE.json.gz'}:raise ValueError('Expected actual result and feature artifacts')
                for name,value in seal['artifacts'].items():
                    if sha(folder/name)!=value:raise ValueError('Completed frame artifact changed')
                sealed_original_calls+=seal.get('resource',{}).get('actual_pose_optimizer_calls',0)
                if not rp.exists():
                    unreceipted.append(dict(domain=domain,arm=arm,ordinal=ordinal,id=fid,frame_seal_sha256=sha(sp),
                        original_sealed_calls=seal.get('resource',{}).get('actual_pose_optimizer_calls')))
                    missing.append(dict(domain=domain,arm=arm,ordinal=ordinal,id=fid,reason='EXECUTION_ENVELOPE_MISSING'))
                    continue
                receipt=read(rp);packet=packets[(domain,ordinal)]
                expected=dict(id=fid,arm=arm,domain=domain,ordinal=ordinal,complete=True,driver_sha256=helper,
                    numerical_kernel_source_binding=source,original_frame_seal_sha256=sha(sp),numeric_reference_inputs=False,
                    cache_payload_sha256=seal['cache_payload_sha256'],requested_packet_file_sha256=packet['sha256'],
                    requested_selection_sha256=packet['selection_sha256'][arm])
                if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('Row execution/source/packet/selection binding differs')
                if seal.get('arm')!=arm:cross_arm.append(authenticate_cross_arm_origin(output,lock,domain,ordinal,arm,seal,receipt))
                stored_calls+=receipt['resource']['actual_new_pose_optimizer_calls']
                row_bindings.append(dict(domain=domain,arm=arm,ordinal=ordinal,id=fid,frame_seal_sha256=sha(sp),row_execution_sha256=sha(rp)))
            fp=output/arm/domain/'FEATURES.jsonl.gz';fs=fp.parent/'FEATURE_SEAL.json'
            if not fs.exists():continue
            seal=read(fs);expected=dict(schema='BI_FULL_NORMAL_FEATURE_SEAL_V1',complete=True,domain=domain,arm=arm,
                rows=len(ids),source_binding=source,driver_sha256=helper,producer_lock_sha256=lock_sha,numeric_reference_inputs=False)
            if any(seal.get(k)!=v for k,v in expected.items()):raise ValueError('Completed feature seal contract differs')
            if sha(fp)!=seal['features_sha256']or sha(fp)!=seal['head_features_sha256']:raise ValueError('Completed feature bytes changed')
            expected_keys={f'{i:04d}'for i in range(len(ids))}
            if set(seal['frame_seal_sha256'])!=expected_keys or set(seal['row_execution_sha256'])!=expected_keys:raise ValueError('Full feature ancestry denominator differs')
            for ordinal in expected_keys:
                folder=fp.parent/ordinal
                if sha(folder/'FRAME_SEAL.json')!=seal['frame_seal_sha256'][ordinal]or sha(folder/'ROW_EXECUTION.json')!=seal['row_execution_sha256'][ordinal]:
                    raise ValueError('Completed feature row ancestry changed')
            identity=[];splits={}
            for feature in rows(fp):
                identity.append(feature['id']);splits[feature['split']]=splits.get(feature['split'],0)+1
            if identity!=ids:raise ValueError('Completed feature exact ordered IDs differ')
            if domain=='synthetic'and len(ids)==1003 and splits!={'train':768,'validation':235}:raise ValueError('Canonical synthetic split differs')
            features[arm+'/'+domain]=dict(path=str(fp),sha256=sha(fp),feature_seal_sha256=sha(fs),rows=len(ids))
    complete=not missing and len(features)==len(lock['arms'])*len(lock['static_ordered_row_ids'])
    return dict(bank_complete=complete,completed_rows=len(row_bindings),missing_rows=missing,features=features,
        row_binding_digest=digest(row_bindings),row_bindings=row_bindings,
        stored_ROW_EXECUTION_optimizer_calls=stored_calls,
        represented_original_FRAME_SEAL_optimizer_calls=sealed_original_calls,
        represented_original_cost_scope='sums of original saved fields;may include copied exact-reuse origins;not current calls or a unique campaign cost',
        sealed_frames_missing_execution_envelopes=unreceipted,unsealed_partial_artifacts=sorted(unsealed),
        verified_cross_arm_original_seal_reuses=cross_arm,
        historical_cost_unknown_possible=bool(unsealed or unreceipted),new_pose_calls_during_authentication=0)


def authenticate(campaign,prior_run,current_binding):
    campaign,prior_run=Path(campaign),Path(prior_run);output=campaign/'F_CORE_THREE_CPU'
    go=read(campaign/'F_CORE_PRODUCTION_ROOT_GO.json');lock=read(output/'PRODUCER_LOCK.json')
    if not go.get('approved')or go.get('H1_fits_authorized')!=0 or go.get('final_configuration_selected')is not False:
        raise ValueError('Only previously authorized unconditional F core is supported')
    if go['source_binding']!=current_binding:raise ValueError('Frozen dispatcher/row-helper/kernel source changed')
    for key,path in [('core_dependency_lock_sha256',campaign/'F_CORE_PRECOMPUTE_LOCK.json'),
        ('frontend_audit_sha256',campaign/'F_FRONT_AUDIT/SUMMARY.json'),('raw_intervention_counts_sha256',campaign/'F_FRONT_AUDIT/INTERVENTION_PATH_COUNTS.json'),
        ('packet_manifest_sha256',campaign/'F_CALIBRATION/PACKET_MANIFEST.json'),('models_sha256',campaign/'F_CALIBRATION/MODELS.json')]:
        if sha(path)!=go[key]:raise ValueError('F core GO dependency changed: '+key)
    expected=dict(schema='FULL_NORMAL_ORDERED_DISPATCH_V1',arms=list(ARMS),workers=3,
        driver_sha256=current_binding['row_helper_sha256'],row_helper_sha256=current_binding['row_helper_sha256'],
        orchestration_sha256=current_binding['orchestration_sha256'],numerical_kernel_source_binding=current_binding['numerical_kernel_source_binding'],
        packet_root=str(campaign/'F_CALIBRATION'),packet_seal=str(campaign/'F_CALIBRATION/PACKET_MANIFEST.json'),
        packet_seal_sha256=go['packet_manifest_sha256'],prior_P0_sha256=sha(prior_run/'P0_INPUT_CONTRACT.json'),
        environment_path=str(campaign/'ENVIRONMENT_INITIAL.json'),environment_sha256=sha(campaign/'ENVIRONMENT_INITIAL.json'),
        stop_file=str(campaign/'F_CORE_CHECKPOINT_STOP'))
    if any(lock.get(k)!=v for k,v in expected.items()):raise ValueError('Existing F core producer lock differs')
    if sha(output/'PACKET_VALIDATION.json')!=lock['packet_validation_receipt_sha256']:raise ValueError('Original packet validation receipt changed')
    inventory=read(prior_run/'P0_INPUT_CONTRACT.json')
    if set(lock['static_ordered_row_ids'])!=set(COUNTS):raise ValueError('Original domain inventory differs')
    for domain,n in COUNTS.items():
        ids=lock['static_ordered_row_ids'][domain]
        if len(ids)!=n or len(set(ids))!=n or ids!=[r['id']for r in inventory[domain]['inventory']]:raise ValueError('Canonical full population/order differs')
    controller=read(campaign/'F_CORE_CONTROLLER_LOCK.json');gate=read(campaign/'F_CORE_RESOURCE_GATE_PASSED.json')
    if controller['GO_sha256']!=sha(campaign/'F_CORE_PRODUCTION_ROOT_GO.json')or controller['source_sha256']!=sha(Path(__file__).with_name('f_core.py')):
        raise ValueError('Original controller source/GO receipt differs')
    if controller['output']!=str(output)or controller['workers']!=3 or controller['head_fit_calls']!=0:raise ValueError('Original controller scope differs')
    if not gate.get('complete')or gate['controller_lock_sha256']!=sha(campaign/'F_CORE_CONTROLLER_LOCK.json')or not gate['renderer_original_process_exited']:
        raise ValueError('Original completed resource gate required')
    material=campaign/'s3_s4_production/FULL_RECEIPT.json';receipt=read(material)
    if sha(material)!=gate['material_FULL_RECEIPT_sha256']or not receipt.get('complete')or receipt.get('rows')!=1241 or not receipt.get('all1241_completed'):
        raise ValueError('Original material completion resource gate differs')
    if receipt['method_lock_sha256']!=sha(material.parent/'METHOD_LOCK_RESOURCE_LP4.json'):raise ValueError('Completed material method lock changed')
    manifest=read(campaign/'F_CALIBRATION/PACKET_MANIFEST.json');packets={}
    if manifest.get('complete')is not True or manifest.get('rows')!=1241:raise ValueError('Complete own1241 F packet manifest required')
    for row in manifest['frames']:
        domain,ordinal=row['domain'],int(row['index']);key=(domain,ordinal)
        if key in packets or domain not in COUNTS or not 0<=ordinal<COUNTS[domain]:raise ValueError('Unknown or duplicate packet index')
        if row['id']!=lock['static_ordered_row_ids'][domain][ordinal]:raise ValueError('Packet canonical identity differs')
        path=campaign/'F_CALIBRATION/packets'/domain/f'{ordinal:04d}.json.gz'
        if sha(path)!=row['packet_sha256']:raise ValueError('Own F packet bytes changed')
        packet=read(path)
        if packet['id']!=row['id']:raise ValueError('Packet payload identity differs')
        packets[key]=dict(sha256=sha(path),selection_sha256={arm:digest(packet['selections'][arm])for arm in ARMS})
    if len(packets)!=1241:raise ValueError('Full1241 sealed packet denominator required')
    bank=authenticate_bank(output,lock,packets)
    bank['bindings']=dict(GO_sha256=sha(campaign/'F_CORE_PRODUCTION_ROOT_GO.json'),producer_lock_sha256=sha(output/'PRODUCER_LOCK.json'),
        packet_manifest_sha256=go['packet_manifest_sha256'],packet_digest=digest({d+'/'+str(i):p['sha256']for(d,i),p in sorted(packets.items())}),
        original_controller_sha256=sha(campaign/'F_CORE_CONTROLLER_LOCK.json'),original_resource_gate_sha256=sha(campaign/'F_CORE_RESOURCE_GATE_PASSED.json'))
    return lock,bank


def classify_owner(campaign,output,args,fd_targets):
    """Recognize F ownership without mistaking generic S2/H1 children."""
    controllers={'scripts.research.pallet_bi_full_execution_v1.f_core','scripts.research.pallet_bi_full_execution_v1.f_core_resume'}
    dispatch='scripts.research.pallet_bi_full_execution_v1.fullnormal_dispatch'
    if (controllers&set(args)and str(campaign)in args)or(dispatch in args and str(output)in args):return 'EXPLICIT_F_CONTROLLER_OR_DISPATCHER'
    # An orphan spawned worker loses its informative parent/module argv.
    # It still inherits the F controller's exact stdout/stderr file. Require
    # spawn_main too: the idle multiprocessing resource tracker is not a solve.
    spawned=any('multiprocessing.spawn'in arg and 'spawn_main'in arg for arg in args)
    expected=str((Path(campaign)/'F_CORE_STDOUT.log').resolve())
    if spawned and expected in fd_targets.values():return 'ORPHAN_CAPABLE_SPAWN_WORKER_WITH_EXACT_OWNED_F_LOG_FD'
    return None


def active_owners(campaign,output):
    """Read-only ownership guard; never updates the old observed /proc fields."""
    found=[]
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit()or int(directory.name)==os.getpid():continue
        try:argv=(directory/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        args=[a.decode(errors='replace')for a in argv if a]
        fd_targets={}
        for number in ('1','2'):
            try:fd_targets[number]=os.readlink(directory/'fd'/number)
            except (FileNotFoundError,PermissionError,ProcessLookupError,OSError):pass
        reason=classify_owner(campaign,output,args,fd_targets)
        if reason:found.append(dict(pid=int(directory.name),argv=args,reason=reason,stdout_stderr=fd_targets,
            owned_runtime_directory_present=(Path(output)/'runtime_tmp'/directory.name).is_dir()))
    return found


def preserve_before(output,invocation,bank):
    """Copy mutable partial receipts; retain immutable scientific rows in place."""
    output,invocation=Path(output),Path(invocation);paths=set()
    for name in ('SUMMARY.json','DISPATCH_COMPLETION.json','PACKET_VALIDATION.json','PRODUCER_LOCK.json'):
        if (output/name).exists():paths.add(output/name)
    paths.update(p for p in (output/'row_jobs').rglob('*')if p.is_file())
    paths.update(output/path for path in bank['unsealed_partial_artifacts'])
    for folder in output.glob('*/*'):
        for name in ('FEATURE_SEAL.json','FEATURES.jsonl.gz'):
            if (folder/name).is_file():paths.add(folder/name)
    bindings=[]
    for path in sorted(paths):
        relative=path.relative_to(output);destination=invocation/'before'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        if sha(path)!=sha(destination):raise ValueError('Preserved prior partial artifact copy differs')
        bindings.append(dict(path=str(relative),sha256=sha(path)))
    write(invocation/'PRESERVED_PRIOR_ARTIFACTS.json',bindings);return bindings


@contextmanager
def exclusive(campaign):
    with (Path(campaign)/'F_CORE_RESUME.RUNNING.lock').open('a')as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another explicit F resume adapter is active')
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def run(campaign,prior_run,*,GO=False,resume_partial=False,resource_GO=False,clear_stop=False):
    if GO is not True:raise PermissionError('Explicit root authentication/resume GO required')
    if clear_stop and not resume_partial:raise ValueError('STOP clearing is only an explicit partial-resume action')
    campaign,prior_run=map(lambda p:Path(p).resolve(),(campaign,prior_run));output=campaign/'F_CORE_THREE_CPU'
    # Imported only for an authorized real invocation; constructed tests use
    # standard-library metadata fixtures and never load the dispatcher.
    from . import fullnormal_dispatch as D
    with exclusive(campaign):
        owners=active_owners(campaign,output)
        if owners:raise RuntimeError('Preserve running original controller/dispatcher: '+str([x['pid']for x in owners]))
        lock,before=authenticate(campaign,prior_run,D.source_bindings())
        invocation=campaign/'F_CORE_RESUME_INVOCATIONS'/str(time.time_ns());invocation.mkdir(parents=True)
        write(invocation/'BEFORE_AUTHENTICATION.json',before)
        common=dict(source_sha256=sha(__file__),GO=True,input_bindings=before['bindings'],invocation=str(invocation),
            original_controller_and_gate_preserved=True,final_F_complete=False,new_H1_fits=0,new_reference_decodes=0)
        if before['bank_complete']:
            result=dict(common,status='COMPLETE_EXACT_REUSE',bank_complete=True,new_pose_optimizer_calls_this_invocation=0,
                dispatcher_calls=0,aggregator_calls=0,original_stored_optimizer_costs=before['stored_ROW_EXECUTION_optimizer_calls'])
            write(invocation/'COMPLETE.json',result);return result
        if not resume_partial:
            result=dict(common,status='INCOMPLETE_AUTHENTICATION_ONLY',bank_complete=False,new_pose_optimizer_calls_this_invocation=0,dispatcher_calls=0,aggregator_calls=0)
            write(invocation/'COMPLETE.json',result);return result
        if resource_GO is not True:raise PermissionError('Partial resume requires fresh explicit root resource GO')
        stop=Path(lock['stop_file'])
        if stop.exists()and not clear_stop:raise PermissionError('Existing STOP preserved;explicit --clear-stop root resume is required')
        owners=active_owners(campaign,output)
        if owners:raise RuntimeError('Another scientific owner started before explicit resume')
        preserved=preserve_before(output,invocation,before)
        if stop.exists():
            shutil.copy2(stop,invocation/'PRESERVED_STOP');write(invocation/'EXPLICIT_STOP_CLEAR.json',dict(root_requested=True,sha256=sha(stop)));stop.unlink()
        reuse=lock.get('exact_reuse_binding');reuse_root=Path(reuse['root'])if reuse else None
        write(invocation/'RESUME_GO.json',dict(common,resource_GO=True,clear_stop=clear_stop,preserved_artifacts=len(preserved),
            prior_stored_optimizer_costs=before['stored_ROW_EXECUTION_optimizer_calls'],historical_cost_unknown_possible=before['historical_cost_unknown_possible']))
        started=time.monotonic()
        try:
            summary=D.run(prior_run,campaign/'F_CALIBRATION',campaign/'F_CALIBRATION/PACKET_MANIFEST.json',campaign/'ENVIRONMENT_INITIAL.json',output,
                arms=ARMS,workers=3,reuse_root=reuse_root,root_GO=True,stop_file=stop)
        except BaseException as error:
            write(invocation/'INTERRUPTED.json',dict(error=repr(error),wall_seconds=time.monotonic()-started,
                known_current_optimizer_calls='UNKNOWN_UNTIL_SEALED_ROW_ENVELOPE_AUDIT',prior_artifacts_preserved=True));raise
        _,after=authenticate(campaign,prior_run,D.source_bindings());write(invocation/'AFTER_AUTHENTICATION.json',after)
        for row in before['row_bindings']:
            folder=output/row['arm']/row['domain']/f"{row['ordinal']:04d}"
            if sha(folder/'FRAME_SEAL.json')!=row['frame_seal_sha256']or sha(folder/'ROW_EXECUTION.json')!=row['row_execution_sha256']:
                raise ValueError('Previously completed scientific row/envelope changed during resume')
        for feature in before['features'].values():
            if sha(feature['path'])!=feature['sha256']or sha(Path(feature['path']).parent/'FEATURE_SEAL.json')!=feature['feature_seal_sha256']:
                raise ValueError('Previously completed feature archive/seal changed during partial resume')
        result=dict(common,status='COMPLETE_RESUMED'if after['bank_complete']else'INCOMPLETE_CHECKPOINT_OR_TECHNICAL',bank_complete=after['bank_complete'],
            dispatcher_calls=1,aggregator_calls=1,new_pose_optimizer_calls_this_invocation=summary['current_invocation_new_pose_optimizer_calls'],
            current_failed_job_optimizer_costs_unknown_possible=bool(summary['technical_failures']),
            original_stored_optimizer_costs=before['stored_ROW_EXECUTION_optimizer_calls'],
            after_stored_optimizer_costs=after['stored_ROW_EXECUTION_optimizer_calls'],
            historical_cost_unknown_possible=before['historical_cost_unknown_possible'],
            summary_sha256=sha(output/'SUMMARY.json'),wall_seconds=time.monotonic()-started,STOP_present_after=stop.exists())
        write(invocation/'COMPLETE.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('campaign','prior-run'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('GO','resume-partial','resource-go','clear-stop'):parser.add_argument('--'+name,action='store_true')
    args=parser.parse_args();print(run(args.campaign,args.prior_run,GO=args.GO,resume_partial=args.resume_partial,resource_GO=args.resource_go,clear_stop=args.clear_stop))
