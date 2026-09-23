import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import PoseModel
from scripts.research.pallet_clean19_structured_easyhard_v1.train import FrozenDataset,PairedTrainer
from . import common as E
from .geo_loss import criterion

class GeoDataset(FrozenDataset):
    def __init__(self,material):
        super().__init__(material,'S1')
        self.H=dict(np.load(E.RAW/f'H_MODEL_{material}.npz'))
    def __getitem__(self,index):
        sample=super().__getitem__(index);occ=sample['occ']
        sample['geo_H']=torch.from_numpy(self.H['H'][occ].copy())
        sample['geo_enabled']=torch.tensor(bool(self.H['enabled'][occ]))
        assert not (self.records[occ]['real'] and sample['geo_enabled'])
        return sample

def collate(samples):
    batch=YOLODataset.collate_fn(samples)
    batch['geo_H']=torch.stack(batch['geo_H']);batch['geo_enabled']=torch.stack(batch['geo_enabled'])
    return batch

class GeoModel(PoseModel):
    def init_criterion(self):return criterion(self)

class Trainer(PairedTrainer):
    def validate(self):
        # Stock runs final-epoch validation even when val=False. Explicitly disable it.
        return self.metrics,0.0
    def final_eval(self):
        from ultralytics.utils.torch_utils import strip_optimizer
        if self.last.exists():strip_optimizer(self.last)
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=GeoModel(cfg,ch=self.data['channels'],nc=self.data['nc'],data_kpt_shape=self.data['kpt_shape'],verbose=verbose)
        if weights:model.load(weights)
        model.lambda_geo=self.geo_lambda;model.collect_geo=True
        return model
    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
        return DataLoader(GeoDataset(self.material),batch_size=batch_size,shuffle=False,num_workers=self.args.workers,persistent_workers=False,collate_fn=collate,pin_memory=True,generator=torch.Generator().manual_seed(42))

def train(material,arm):
    E.C.setup();E.C.guard()
    assert not (E.DOC/'STOP.json').exists()
    assert E.read(E.DOC/'LOSS_CONTRACT_TEST.json')['passed']
    cal=E.read(E.DOC/'LAMBDA_CALIBRATION.json');args=E.read(E.C.DOC/'PREFLIGHT.json')['args']
    Trainer.material=material;Trainer.arm='S1';Trainer.geo_lambda=0. if arm=='M0' else cal['lambda_geo']
    name=material+'_'+arm;run=E.RAW/'runs'/name;assert not run.exists()
    trainer=Trainer(overrides=dict(args,model=str(E.C.H.C.N.E.R0),data=E.read(E.C.DOC/'PREFLIGHT.json')['datasets'][material],project=str(E.RAW/'runs'),name=name,exist_ok=False))
    trainer.trace=[];steps=[];epochrows=[];initial={};geo=[];start=time.monotonic()
    def begin(t):
        src=torch.load(E.C.H.C.N.E.R0,map_location='cpu',weights_only=False)['model'].float().state_dict()
        for k,v in t.model.state_dict().items():
            if k in src and src[k].shape==v.shape:assert torch.equal(v.cpu(),src[k])
        initial.update({k:E.C.digest(v.cpu().numpy()) for k,v in t.model.state_dict().items()})
        assert initial==E.read(E.C.DOC/f'FIT_{material}_S1.json')['initial_state']
        t.optimizer.register_step_post_hook(lambda *x:steps.append(len(steps)+1))
    def epochstart(t):t.train_loader.dataset.epoch=t.epoch;E.C.guard()
    def batchend(t):
        crit=t.model.criterion
        vals=[float(p.latest_geo.detach()) if p.latest_geo is not None else 0. for p in [crit.one2many,crit.one2one]]
        assert np.isfinite(vals).all();geo.append(vals)
    def epochend(t):
        assert torch.isfinite(t.tloss).all()
        epochrows.append(dict(epoch=t.epoch,steps=len(steps),loss=t.tloss.cpu().tolist(),geo_mean=np.mean(geo[-64:],axis=0).tolist(),gpu=E.C.guard()))
        print('FIT_PROGRESS',name,t.epoch+1,len(steps),flush=True)
    trainer.add_callback('on_train_start',begin);trainer.add_callback('on_train_epoch_start',epochstart);trainer.add_callback('on_train_batch_end',batchend);trainer.add_callback('on_train_epoch_end',epochend)
    trainer.train()
    assert len(steps)==320 and len(trainer.trace)==5120
    old=E.read(E.C.RAW/f'TRACE_{material}_S1.json');assert trainer.trace==old
    E.save(E.RAW/f'TRACE_{name}.json',trainer.trace)
    last=run/'weights/last.pt';oldstate=torch.load(E.ROOT/E.read(E.C.DOC/f'FIT_{material}_S1.json')['checkpoint']['path'],map_location='cpu',weights_only=False)
    newstate=torch.load(last,map_location='cpu',weights_only=False)
    oldmodel=(oldstate.get('ema') or oldstate['model']).state_dict();newmodel=(newstate.get('ema') or newstate['model']).state_dict()
    exact=all(torch.equal(newmodel[k],v) for k,v in oldmodel.items())
    E.save(E.DOC/f'FIT_{name}.json',dict(complete=True,steps=320,epochs=epochrows,checkpoint=E.bind(last),initial_state=initial,real_exposures=2560,synthetic_exposures=2560,trace=E.bind(E.RAW/f'TRACE_{name}.json'),old_S1_trace_exact=True,old_S1_final_state_exact=exact,lambda_geo=Trainer.geo_lambda,seconds=time.monotonic()-start,geo_batch_history=geo))
    print('FIT_COMPLETE',name,'old_S1_exact',exact,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('material',choices=['PLASTIC','WOOD']);p.add_argument('arm',choices=['M0','M1']);a=p.parse_args();train(a.material,a.arm)
