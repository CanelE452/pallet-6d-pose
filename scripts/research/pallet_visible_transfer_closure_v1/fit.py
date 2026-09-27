"""One bounded pair. Preserve original five-epoch LR AND E2E criterion schedules."""
import argparse
import csv
from copy import deepcopy
import time
import torch
from ultralytics.utils.torch_utils import one_cycle
from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import PoseOnlyTrainer,pose_parameter
from scripts.research.pallet_type_selftrain_v1 import common as T
from . import common as C

class ExtendedTrainer(PoseOnlyTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.args.epochs==5
        self.epochs=10
    def _setup_scheduler(self):
        original=one_cycle(1,self.args.lrf,5)
        self.lf=lambda x:original(min(x,4))
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda=self.lf)

def train(arm):
    assert arm in ('RAW_NEW','REF_NEW')
    dst=C.DOC/f'FIT_{arm}.json'
    if dst.exists():C.verify(C.read(dst)['checkpoint']);return
    assert not (C.DOC/'EXPERIMENT_BLOCKED.json').exists(),'Preserve failed prefix; do not attempt another arm or intervention'
    lock=C.read(C.DOC/'INTERVENTION_LOCK.json')
    for b in lock['source_files']:C.verify(b)
    for b in C.read(C.DOC/'INPUT_BINDINGS.json')['files']:C.verify(b)
    target=arm.split('_')[0];old=target+'_LR5';protocol=C.read(C.P.REC/'pose_only/PROTOCOL.json')
    assert not (C.RAW/'runs'/arm).exists(),'Preserve incomplete run; no blind retrain'
    T.N.setup();torch.set_num_interop_threads(1);T.N.E.gpu();assert torch.cuda.is_available()
    args=dict(protocol['args'],lr0=1e-5,model=str(C.ROOT/C.checkpoint('R0')['path']),data=str(C.ROOT/protocol['datasets'][target]['data']['path']),project=str(C.RAW/'runs'),name=arm,exist_ok=False)
    trainer=ExtendedTrainer(overrides=args);steps=[];history=[];start=time.monotonic()
    base=torch.load(C.ROOT/C.checkpoint('R0')['path'],map_location='cpu',weights_only=False)['model'].float().state_dict()
    oldstate=torch.load(C.ROOT/C.checkpoint(old)['path'],map_location='cpu',weights_only=False)['model'].float().state_dict()
    oldfit=C.read(C.P.REC/'pose_only'/f'FIT_{old}.json');oldcsv=list(csv.DictReader((C.ROOT/oldfit['results_csv']['path']).open()))
    def begin(t):
        assert all(torch.equal(v,t.model.state_dict()[k].cpu()) for k,v in base.items())
        def step(opt,args,kwargs):
            steps.append(1);assert len(steps)<=640
        t.optimizer.register_step_post_hook(step)
    def epoch(t):
        assert t.batch_size==16 and t.accumulate==1,'No implicit OOM/batch protocol change'
        assert t.args.epochs==5 and t.epochs==10
        fixed=t.check_frozen();history.append(dict(epoch=t.epoch+1,steps=len(steps),fixed=fixed,gpu=T.N.E.gpu(),lr=[g['lr'] for g in t.optimizer.param_groups],o2m=t.model.criterion.o2m))
        # Validation in old final epoch can alter EMA numeric dtype; reproduce that bookkeeping here too.
        if t.epoch==4:
            t.ema.update_attr(t.model,include=['yaml','nc','args','names','stride','class_weights'])
            t.metrics,t.fitness=t.validate()
        print('BOUNDED_EPOCH',arm,t.epoch+1,len(steps),flush=True)
    def saved(t):
        if t.epoch!=4:return
        current=deepcopy(t.ema.ema).half().float().cpu().state_dict()
        changed=[k for k in oldstate if not torch.equal(oldstate[k],current[k])]
        csvnew=list(csv.DictReader(t.csv.open()))
        columns=[k for k in oldcsv[0] if k.strip().startswith(('train/','lr/'))]
        parity=all(float(x[k])==float(y[k]) for x,y in zip(csvnew,oldcsv) for k in columns)
        report=dict(arm=arm,steps=len(steps),model_bit_exact=not changed,different_tensors=changed,loss_lr_csv_exact=parity,
            first320_pass=not changed and parity,full_augmented_tensor_audit=False)
        C.save(C.DOC/f'PREFIX_{arm}.json',report,True)
        assert report['first320_pass'],'Prefix mismatch: preserve run and stop entire paired experiment'
    trainer.add_callback('on_train_start',begin);trainer.add_callback('on_train_epoch_end',epoch);trainer.add_callback('on_model_save',saved)
    try:trainer.train()
    except Exception as e:
        C.save(C.DOC/'EXPERIMENT_BLOCKED.json',dict(arm=arm,steps=len(steps),reason=repr(e),utc=C.now(),retry_for_performance=False,other_intervention_forbidden=True))
        raise
    ck=C.RAW/'runs'/arm/'weights/last.pt';final=torch.load(ck,map_location='cpu',weights_only=False)['model'].float().state_dict()
    protected=[k for k in base if not pose_parameter(k) or k.endswith(('.running_mean','.running_var','.num_batches_tracked'))]
    assert all(torch.equal(base[k],final[k]) for k in protected)
    changed=[k for k in base if not torch.equal(base[k],final[k])];assert changed and all(pose_parameter(k) for k in changed)
    assert len(steps)==640
    C.save(dst,dict(complete=True,arm=arm,checkpoint=C.bind(ck),optimizer_steps=len(steps),history=history,seconds=time.monotonic()-start,
        prefix=C.bind(C.DOC/f'PREFIX_{arm}.json'),protected_state_exact=True,changed_tensors=changed,results_csv=C.bind(trainer.csv),intervention_lock=C.bind(C.DOC/'INTERVENTION_LOCK.json')),True)
    print('FIT_COMPLETE',arm,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['RAW_NEW','REF_NEW']);train(p.parse_args().arm)
