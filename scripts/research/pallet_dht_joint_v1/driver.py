"""Owned sequential train → matched audit → real evaluation → report completion."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pending.json');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def now():return datetime.now(timezone.utc).isoformat()


def audit_training(root, protocol):
    import importlib
    import torch
    if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
    importlib.import_module('integration')
    runs=[];traces={};final_models=[]
    for seed in protocol['training']['seeds']:
        for arm in protocol['arms']:
            folder=root/'runs'/f'{arm}_seed{seed}'
            c=read(folder/'COMPLETION.json')
            if not(c['complete'] and c['PASS'] and c['stage']=='main'
                   and c['epochs_completed']==protocol['training']['epochs']
                   and c['optimizer_steps']==protocol['expected_optimizer_steps_per_cell']
                   and c['train_frames']==55980 and c['val_frames']==4020
                   and c['checkpoint_sha256']==sha(c['checkpoint'])
                   and c['bindings']['protocol_sha256']==sha(root/'TRAIN_PROTOCOL.json')):
                raise ValueError(f'Unfinished or changed training cell {arm}/{seed}')
            trace=sha(folder/'BATCH_TRACE.jsonl')
            if trace!=c['batch_trace_sha256']:raise ValueError('Trace changed')
            traces[(arm,seed)]=trace
            for group in ['backbone_neck','pose_head']+(['hough'] if arm!='point_only' else []):
                if c['final_parameter_BN_changes'][group]['changed_values']<=0:
                    raise ValueError(f'No actual learning in {arm}/{seed}/{group}')
            for group in ['backbone_neck','pose_head']:
                if c['final_parameter_BN_changes'][group]['bn_changed']<=0:
                    raise ValueError(f'Frozen BN in {arm}/{seed}/{group}')
            checkpoint=torch.load(c['checkpoint'],map_location='cpu')
            state=checkpoint['ema'].state_dict()
            tensor_sha=hashlib.sha256()
            for key,tensor in sorted(state.items()):
                tensor_sha.update(key.encode());tensor_sha.update(str(tensor.dtype).encode())
                tensor_sha.update(str(tuple(tensor.shape)).encode())
                tensor_sha.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            ema_sha=tensor_sha.hexdigest();final_models.append(ema_sha)
            del checkpoint,state
            runs.append(dict(arm=arm,seed=seed,checkpoint=c['checkpoint'],
                             checkpoint_sha256=c['checkpoint_sha256'],
                             ema_tensor_sha256=ema_sha,
                             completion_sha256=sha(folder/'COMPLETION.json'),trace_sha256=trace))
        if len({traces[(a,seed)] for a in protocol['arms']})!=1:
            raise ValueError(f'Augmented input or minibatch mismatch across arms for seed {seed}')
    for arm in protocol['arms']:
        if len({traces[(arm,s)] for s in protocol['training']['seeds']})!=len(protocol['training']['seeds']):
            raise ValueError('Seeds did not produce distinct actual data schedules')
    if len(set(final_models))!=len(runs):raise ValueError('Final EMA tensor values duplicated across cells')
    path=root/'TRAINING_AUDIT.json'
    existing=read(path) if path.exists() else None
    result=dict(complete=True,PASS=True,created_at_utc=existing['created_at_utc'] if existing else now(),runs=runs,
                same_seed_augmented_batches_exact=True,seeds_have_distinct_actual_batches=True,
                all_network_groups_and_BN_changed=True,protocol_sha256=sha(root/'TRAIN_PROTOCOL.json'),
                evaluation_weights='final epoch EMA; no real or best-epoch selection')
    if existing is not None:
        if existing!=result:raise ValueError('Completed training audit changed')
    else:write(path,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
    root=args.run_dir.resolve();protocol=read(root/'TRAIN_PROTOCOL.json');t=protocol['training']
    lock_handle=(root/'.driver.lock').open('a+')
    fcntl.flock(lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupted(signum,_frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    for name in ['train.py','evaluate.py','runtime.py','aggregate.py','report.py','finalize.py']:
        if not(HERE/name).exists():raise FileNotFoundError(f'Completion chain missing {name}')
    preflight={}
    for name in protocol['preflight_required']:
        path=root/name;value=read(path)
        if not(value.get('complete') is True and value.get('PASS') is True):raise ValueError(f'Preflight failed: {path}')
        for key in ['source_sha256','input_sha256','output_sha256']:
            for file,digest in value.get(key,{}).items():
                file=Path(file);file=file if file.is_absolute() else path.parent/file
                if sha(file)!=digest:raise ValueError(f'Preflight binding changed: {file}')
        preflight[name]=sha(path)
    bound=dict(protocol_sha256=sha(root/'TRAIN_PROTOCOL.json'),driver_sha256=sha(__file__),
               training_source_sha256=protocol['source_code_sha256'],preflight_sha256=preflight)
    if (root/'DRIVER_BINDING.json').exists():
        if read(root/'DRIVER_BINDING.json')!=bound:raise ValueError('Driver binding changed')
    else:write(root/'DRIVER_BINDING.json',bound)
    def check():
        if sha(root/'TRAIN_PROTOCOL.json')!=bound['protocol_sha256']:raise ValueError('Protocol changed')
        for file,digest in bound['training_source_sha256'].items():
            if sha(file)!=digest:raise ValueError(f'Training source changed: {file}')
    def stage(label,module,extra):
        check();command=[sys.executable,'-m',f'scripts.research.pallet_dht_joint_v1.{module}',
                          '--run-dir',str(root),*extra]
        stamp=time.time();write(root/'STATUS.json',dict(complete=False,stage=label,started_at_utc=now(),command=command))
        print(f'{now()} START {label}',flush=True)
        log=root/'logs'/f'{label}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            child=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                                   env={**os.environ,'PYTHONPATH':str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''),
                                        'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
            write(root/'STATUS.json',dict(complete=False,stage=label,started_at_utc=now(),command=command,
                                         driver_pid=os.getpid(),child_pid=child.pid,log=str(log)))
            try:code=child.wait()
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        if code:raise RuntimeError(f'{label} exited {code}; inspect {log}')
        print(f'{now()} DONE {label} ({time.time()-stamp:.1f}s)',flush=True)
    try:
        for seed in t['seeds']:
            for arm in protocol['arms']:
                extra=['--arm',arm,'--seed',str(seed),'--epochs',str(t['epochs']),
                       '--batch',str(t['batch']),'--lr',str(t['lr']),'--optimizer',t['optimizer'],
                       '--line-weight',str(t['line_weight']),'--workers',str(t['workers']),
                       '--lrf',str(t['lrf']),'--warmup-epochs',str(t['warmup_epochs']),
                       '--warmup-bias-lr',str(t['warmup_bias_lr']),
                       '--amp-init-scale',str(t['amp_init_scale'])]
                if t['amp']:extra.append('--amp')
                folder=root/'runs'/f'{arm}_seed{seed}'
                if (folder/'CELL_CONFIG.json').exists() and not(folder/'COMPLETION.json').exists():
                    if (folder/'weights/last.pt').exists():extra.append('--resume')
                    else:
                        # No complete epoch exists to resume. Preserve every
                        # partial artifact, then replay from common R0/seed.
                        archive=root/'provenance/interrupted_before_first_epoch'/f'{folder.name}_{time.time_ns()}'
                        archive.parent.mkdir(parents=True,exist_ok=True);folder.rename(archive)
                        journal=root/'FIRST_EPOCH_RESTARTS.json';entries=read(journal) if journal.exists() else []
                        entries.append(dict(arm=arm,seed=seed,archive=str(archive),time_utc=now(),
                                            reason='No completed-epoch checkpoint; replay same protocol from R0'))
                        write(journal,entries)
                stage(f'train_{arm}_seed{seed}','train',extra)
        audit_training(root,protocol)
        for seed in t['seeds']:
            for arm in protocol['arms']:
                cp=root/'runs'/f'{arm}_seed{seed}'/'weights/final.pt'
                stage(f'evaluate_{arm}_seed{seed}','evaluate',
                      ['--arm',arm,'--seed',str(seed),'--checkpoint',str(cp),'--phase','all'])
        stage('runtime','runtime',[])
        stage('aggregate','aggregate',[])
        stage('report','report',[])
        stage('audit_outputs','audit_outputs',[])
        stage('visual_qa','visual_qa',[])
        stage('finalize','finalize',[])
        done=read(root/'COMPLETION.json')
        if not(done['complete'] and done['PASS']):raise ValueError('Invalid completion marker')
        write(root/'STATUS.json',dict(complete=True,stage='complete',finished_at_utc=now(),
                                   completion_sha256=sha(root/'COMPLETION.json')))
    except BaseException as exc:
        write(root/'DRIVER_FAILURE.json',dict(complete=False,PASS=False,time_utc=now(),
              type=type(exc).__name__,message=str(exc),last_status=read(root/'STATUS.json') if(root/'STATUS.json').exists() else None))
        raise


if __name__=='__main__':main()
