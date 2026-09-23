import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import PoseModel
from scripts.research.pallet_clean19_structured_easyhard_v1.train import FrozenDataset,PairedTrainer
from . import common as E
from .geo_diag_loss import criterion,diag_total

class DiagDataset(FrozenDataset):
    def __init__(self,material):
        super().__init__(material,'S1');self.weights=dict(np.load(E.RAW/f'WEIGHTS_{material}.npz'))
    def __getitem__(self,index):
        sample=super().__getitem__(index);occ=sample['occ']
        sample['diag_weight']=torch.from_numpy(self.weights['w'][occ].copy())
        sample['diag_enabled']=torch.tensor(bool(self.weights['enabled'][occ]))
        assert not (self.records[occ]['real'] and sample['diag_enabled'])
        return sample

def collate(samples):
    b=YOLODataset.collate_fn(samples)
    for k in ('diag_weight','diag_enabled'):b[k]=torch.stack(b[k])
    return b

class DiagModel(PoseModel):
    def init_criterion(self):return criterion(self)

class Trainer(PairedTrainer):
    def validate(self):return self.metrics,0.0
    def final_eval(self):
        from ultralytics.utils.torch_utils import strip_optimizer
        if self.last.exists():strip_optimizer(self.last)
    def _handle_nan_recovery(self,epoch):
        assert torch.isfinite(self.loss).all(),'NaN/Inf: rescue prohibited'
        return False
    def get_model(self,cfg=None,weights=None,verbose=True):
        if self.experiment_arm=='M0':return super().get_model(cfg,weights,verbose)
        m=DiagModel(cfg,ch=self.data['channels'],nc=self.data['nc'],data_kpt_shape=self.data['kpt_shape'],verbose=verbose)
        if weights:m.load(weights)
        m.lambda_diag=self.diag_lambda;m.collect_diag=True
        return m
    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
        return DataLoader(DiagDataset(self.material),batch_size=batch_size,shuffle=False,num_workers=self.args.workers,persistent_workers=False,collate_fn=collate,pin_memory=True,generator=torch.Generator().manual_seed(42))

def calibrate():
    import inspect
    from .test_contracts import fresh,batch
    assert E.read(E.DOC/'GPU_GRADIENT_PARITY.json')['passed']
    assert E.read(E.DOC/'DIAGONAL_MATH_TEST.json')['passed']
    m=fresh();m.collect_diag=True;b,idx,hashes=batch();loss=criterion(m)
    total,_=loss(m(b['img']),b);geo=diag_total(loss,len(idx))
    head=m.model[-1];ids=set()
    for branch in (head.one2many,head.one2one):
        for key in ('pose_head','kpts_head'):
            ids.update(id(p) for p in branch[key].parameters())
    params=[(n,p) for n,p in m.named_parameters() if id(p) in ids and p.requires_grad];assert params
    baseg=torch.autograd.grad(total.sum(),[p for _,p in params],retain_graph=True,allow_unused=True)
    geog=torch.autograd.grad(geo,[p for _,p in params],allow_unused=True)
    norm=lambda gs:float(torch.sqrt(sum(g.detach().double().square().sum() for g in gs if g is not None)))
    gb,gd=norm(baseg),norm(geog);lam=.25*gb/gd;assert np.isfinite([gb,gd,lam]).all() and min(gb,gd,lam)>0
    E.save(E.DOC/'LAMBDA_CALIBRATION.json',dict(lambda_diag=lam,g_base=gb,g_diag=gd,target_ratio=.25,parameter_names=[n for n,_ in params],head_forward_source=inspect.getsource(type(head).forward_head),microbatch_slots=idx,batch_hashes=hashes,base_definition='existing TOTAL loss; same coordinate-producing pose_head+kpts_head parameters',diag_definition='E2E weighted diagonal loss times batch size; after pose gain',optimizer_steps=0,DEV_used=False))
    E.save(E.DOC/'TRAINING_PROTOCOL.json',dict(fits=4,steps_each=320,maximum_total_steps=1280,init=E.bind(E.C.H.C.N.E.R0),args=E.read(E.C.DOC/'PREFLIGHT.json')['args'],lambda_diag=lam,eval_during_train=False,checkpoint='final last only',rescue=False))
    print('CALIBRATION_READY',lam,gb,gd,flush=True)

def train(material,arm):
    E.deterministic();E.C.guard();E.immutable();assert not (E.DOC/'STOP.json').exists()
    assert E.read(E.DOC/'GPU_GRADIENT_PARITY.json')['passed']
    cal=E.read(E.DOC/'LAMBDA_CALIBRATION.json');args=E.read(E.C.DOC/'PREFLIGHT.json')['args']
    Trainer.material=material;Trainer.arm='S1';Trainer.experiment_arm=arm;Trainer.diag_lambda=cal['lambda_diag']
    name=material+'_'+arm;run=E.RAW/'runs'/name;assert not run.exists(),'No restart/rescue'
    t=Trainer(overrides=dict(args,model=str(E.C.H.C.N.E.R0),data=E.read(E.C.DOC/'PREFLIGHT.json')['datasets'][material],project=str(E.RAW/'runs'),name=name,exist_ok=False))
    t.trace=[];steps=[];epochs=[];initial={};inventory=[];geo=[];start=time.monotonic()
    def begin(t):
        initial.update({k:E.C.digest(v.cpu().numpy()) for k,v in t.model.state_dict().items()})
        assert initial==E.read(E.C.DOC/f'FIT_{material}_S1.json')['initial_state']
        inventory.extend(dict(name=n,shape=list(p.shape),trainable=p.requires_grad) for n,p in t.model.named_parameters())
        def pre(*_):
            assert len(steps)<320
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in t.model.parameters())
        t.optimizer.register_step_pre_hook(pre)
        t.optimizer.register_step_post_hook(lambda *_:steps.append(len(steps)+1))
    def epochstart(t):t.train_loader.dataset.epoch=t.epoch;E.C.guard()
    def batchend(t):
        assert torch.isfinite(t.loss).all()
        if arm=='M1':
            vals=[float(p.latest_diag.detach()) if p.latest_diag is not None else 0. for p in (t.model.criterion.one2many,t.model.criterion.one2one)]
            assert np.isfinite(vals).all();geo.append(vals)
    def epochend(t):
        epochs.append(dict(epoch=t.epoch,steps=len(steps),loss=t.tloss.cpu().tolist(),diag_mean=np.mean(geo[-64:],axis=0).tolist() if geo else None,gpu=E.C.guard()))
        print('FIT_PROGRESS',name,t.epoch+1,len(steps),flush=True)
    for key,fn in [('on_train_start',begin),('on_train_epoch_start',epochstart),('on_train_batch_end',batchend),('on_train_epoch_end',epochend)]:t.add_callback(key,fn)
    t.train();assert len(steps)==320 and len(t.trace)==5120
    assert t.trace==E.read(E.C.RAW/f'TRACE_{material}_S1.json')
    E.save(E.RAW/f'TRACE_{name}.json',t.trace)
    old=torch.load(E.ROOT/E.read(E.C.DOC/f'FIT_{material}_S1.json')['checkpoint']['path'],map_location='cpu',weights_only=False)
    new=torch.load(run/'weights/last.pt',map_location='cpu',weights_only=False)
    os=(old.get('ema') or old['model']).state_dict();ns=(new.get('ema') or new['model']).state_dict()
    E.save(E.DOC/f'FIT_{name}.json',dict(complete=True,steps=320,epochs=epochs,checkpoint=E.bind(run/'weights/last.pt'),initial_state=initial,inventory=inventory,real_exposures=2560,synthetic_exposures=2560,trace=E.bind(E.RAW/f'TRACE_{name}.json'),old_S1_trace_exact=True,old_S1_final_state_exact=all(torch.equal(ns[k],v) for k,v in os.items()),lambda_diag=cal['lambda_diag'] if arm=='M1' else 0,seconds=time.monotonic()-start,diag_batch_history=geo,no_eval_during_train=True,no_rescue=True))
    print('FIT_COMPLETE',name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['calibrate','train']);p.add_argument('--material',choices=E.MATS);p.add_argument('--arm',choices=E.ARMS);a=p.parse_args()
    calibrate() if a.action=='calibrate' else train(a.material,a.arm)
