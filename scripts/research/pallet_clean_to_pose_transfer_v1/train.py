"""Bounded four-arm training; immutable protocols, last-only, full state audit."""
import argparse
import csv
from pathlib import Path
import time

import torch

from . import common as C
from .augmentation import load_paired_labels
from .trainer import CleanPoseTrainer, assert_protected_state


def code_lock():
    path = C.DOC/'CODE_LOCK.json'
    if path.exists():
        for b in C.read(path)['files']: C.verify(b)
        return
    names = ['common.py','prepare.py','augmentation.py','trainer.py','train.py','preflight.py']
    files = [Path(__file__).with_name(n) for n in names]
    files += [C.ROOT/p for p in [
        'scripts/research/pallet_type_selftrain_v1/recovery_pose_trainer.py',
        'scripts/self_training_yolo/v3/true_ignore_trainer.py',
        'scripts/self_training_yolo/v3/true_ignore_pose_loss.py']]
    C.save(path,dict(created=C.now(),files=[C.bind(p) for p in files],
        protocol=C.bind(C.DOC/'PRIMARY_PROTOCOL.json'),locked_before_fit=True),True)


def train(arm, training_seed=42):
    protocol_path=C.DOC/'PRIMARY_PROTOCOL.json'; p=C.read(protocol_path)
    assert p['locked_before_fit'] and arm in p['arms'] and training_seed in p['seeds']
    assert C.read(C.DOC/'PREFLIGHT.json')['passed']
    code_lock()
    for b in p['inputs']+p['sources']+[p['initialization']]: C.verify(b)
    name=f'{arm}_S{training_seed}'; fit_path=C.DOC/f'FIT_{name}.json'
    if fit_path.exists():
        fit=C.read(fit_path); assert fit['complete']; C.verify(fit['checkpoint'])
        print('FIT_ALREADY_COMPLETE',name,flush=True); return
    run_dir=C.RAW/'runs'/name
    assert not run_dir.exists(), 'Incomplete run preserved: no implicit overwrite or restart'
    totals=C.read(C.DOC/'RESOURCE_LEDGER.json')['totals']
    assert totals['student_fits']<10 and totals['GPU_training_seconds']<21600
    remaining_seconds=21600-totals['GPU_training_seconds']
    from scripts.research.pallet_type_selftrain_v1 import train as T
    from scripts.research.pallet_oracle_mechanism_followup_v1.cycle_affine import gpu
    T.C.N.setup(); torch.set_num_interop_threads(1)
    assert torch.cuda.is_available(), 'Host CUDA required; do not silently fall back to CPU'
    gpu()
    spec=p['arms'][arm]; target=spec['target']
    paired=load_paired_labels(*[C.ROOT/p['datasets'][key]['train_list']['path'] for key in ('RAW','REF')])
    manifest=C.read(C.RAW/'PRIMARY_INPUT_BINDINGS_PRIVATE.json')
    recordings={r['train_id']+'.png':r['recording'] for r in manifest['rows'] if r['train_id']+'.png' in paired}
    assert set(recordings)==set(paired), (len(recordings),len(paired))
    args=dict(p['args'],seed=training_seed,model=str(C.ROOT/p['initialization']['path']),
        data=str(C.ROOT/p['datasets'][target]['data']['path']),project=str(C.RAW/'runs'),name=name,exist_ok=False)
    trainer=CleanPoseTrainer(overrides=args,paired_labels=paired,target=target,
        condition=spec['condition'],recordings=recordings)
    base=torch.load(C.ROOT/p['initialization']['path'],map_location='cpu',weights_only=False)['model'].float().state_dict()
    steps=[]; history=[]; start=time.monotonic()
    C.save(C.RAW/f'START_{name}.json',dict(utc=C.now(),protocol=C.bind(protocol_path),
        code=C.bind(C.DOC/'CODE_LOCK.json'),arm=arm,seed=training_seed),True)

    def step_done(opt,args,kwargs):
        steps.append(1)
        assert len(steps)<=320 and time.monotonic()-start<remaining_seconds

    def on_start(t):
        state=t.model.state_dict(); assert set(state)==set(base)
        assert all(torch.equal(base[k],state[k].detach().cpu()) for k in base)
        assert len(t.recovery_trainable)==132 and len(t.transfer_protected_keys)==747
        t.optimizer.register_step_post_hook(step_done)
        print('EXACT_R0',name,len(state),'TRAINABLE',len(t.recovery_trainable),flush=True)

    def on_epoch(t):
        fixed=t.check_frozen(); temperature=gpu()
        history.append(dict(epoch=t.epoch+1,steps=len(steps),protected_tensors=fixed,temperature_C=temperature))
        C.save(C.RAW/f'RUN_STATE_{name}.json',dict(status='RUNNING',arm=arm,
            optimizer_steps=len(steps),history=history,utc=C.now()))
        print('EPOCH',name,t.epoch+1,len(steps),flush=True)

    trainer.add_callback('on_train_start',on_start)
    trainer.add_callback('on_train_epoch_end',on_epoch)
    try:
        trainer.train()
        checkpoint=run_dir/'weights/last.pt'
        final=torch.load(checkpoint,map_location='cpu',weights_only=False)['model'].float().state_dict()
        protected=trainer.transfer_protected_keys
        assert_protected_state(final,base,protected)
        changed=[k for k in base if not torch.equal(base[k],final[k])]
        assert changed and set(changed)<=set(trainer.recovery_trainable)
        assert len(steps)==len(trainer.trace)==320
        assert len(list(csv.DictReader((run_dir/'results.csv').open())))==5
        trace_path=C.RAW/f'TRACE_{name}.json'; C.save(trace_path,trainer.trace,True)
        fit=dict(complete=True,arm=arm,target=target,condition=spec['condition'],seed=training_seed,
            optimizer_steps=320,checkpoint=C.bind(checkpoint),protocol=C.bind(protocol_path),
            initialization=p['initialization'],exact_R0_initialization=True,protected_state_exact=True,
            protected_tensors=len(protected),trainable_tensors=len(trainer.recovery_trainable),
            changed_tensors=changed,trace=C.bind(trace_path),results_csv=C.bind(run_dir/'results.csv'),
            history=history,seconds=time.monotonic()-start,manual_added=0,checkpoint_selection='last only',
            code_lock=C.bind(C.DOC/'CODE_LOCK.json'),evaluation_reference_read=False)
        C.save(fit_path,fit,True)
        C.resource(f'FIT_{name}',fit['seconds'],fits=1,updates=320,details=dict(checkpoint=fit['checkpoint']))
        C.save(C.RAW/f'RUN_STATE_{name}.json',dict(status='COMPLETED',fit=C.bind(fit_path),utc=C.now()))
        print('FIT_COMPLETE',name,fit['seconds'],flush=True)
    except BaseException as exc:
        C.save(C.RAW/f'FAILURE_{name}.json',dict(error=repr(exc),executed_optimizer_steps=len(steps),
            seconds=time.monotonic()-start,utc=C.now()),True)
        C.resource(f'FAILED_{name}',time.monotonic()-start,fits=int(bool(steps)),updates=len(steps),details=repr(exc))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['lock','train'])
    parser.add_argument('--arm'); parser.add_argument('--seed',type=int,default=42)
    a=parser.parse_args()
    if a.phase=='lock': code_lock()
    else: train(a.arm,a.seed)
