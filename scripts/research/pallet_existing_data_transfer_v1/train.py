import argparse
import time
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.torch_utils import strip_optimizer
from scripts.self_training_yolo.v3.true_ignore_trainer import TrueIgnorePoseTrainer
from scripts.research.pallet_clean19_structured_easyhard_v1 import augmentation as A
from . import common as E

class Frozen(Dataset):
    def __init__(self,arm):
        self.arm=arm;self.records=E.read(E.RAW/'OCCURRENCES.json');self.epoch=0;self.labels=[]
        for i in range(1024):
            with np.load(self.path(self.records[i])) as z:self.labels.append(dict(cls=z['cls'],bboxes=z['bboxes']))
    def path(self,r):return E.ROOT/(r['cache'][{'T0_EASY_PSEUDO':'E','T1_HARD_PSEUDO':'H','T2_HARD_MANUAL':'M'}[self.arm]] if r['role']=='REPLACEMENT' else r['cache'])['path']
    def __len__(self):return 1024
    def __getitem__(self,i):
        occ=self.epoch*1024+i;r=self.records[occ]
        with np.load(self.path(r)) as z:s={k:torch.from_numpy(z[k].copy()) for k in ('img','keypoints','bboxes','cls','batch_idx')}
        if r['role']=='C0':s['img']=torch.from_numpy(A.apply(s['img'].numpy(),r['old']['plan'],'S1'))
        s.update(im_file=str(self.path(r)),ori_shape=(640,640),resized_shape=(640,640),ratio_pad=((1.,1.),(0.,0.)),occ=occ)
        return s

class Trainer(TrueIgnorePoseTrainer):
    def validate(self):return self.metrics,0.
    def final_eval(self):
        if self.last.exists():strip_optimizer(self.last)
    def _handle_nan_recovery(self,epoch):
        assert torch.isfinite(self.loss).all(),'Nonfinite: no rescue run';return False
    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
        return DataLoader(Frozen(self.arm),batch_size=batch_size,shuffle=False,num_workers=self.args.workers,persistent_workers=False,collate_fn=YOLODataset.collate_fn,pin_memory=True,generator=torch.Generator().manual_seed(42))
    def preprocess_batch(self,batch):
        for i,occ in enumerate(batch['occ']):
            k=batch['keypoints'][batch['batch_idx']==i].numpy();b=batch['bboxes'][batch['batch_idx']==i].numpy();r=self.train_loader.dataset.records[occ]
            self.trace.append(dict(occ=occ,role=r['role'],RGB=E.C.digest(batch['img'][i].numpy()),bbox=E.C.digest(b),target=E.C.digest(k),mask=E.C.digest(k[:,:,2]),supervised=int((k[:,:,2]==2).sum())))
        return super().preprocess_batch(batch)

def main(arm):
    E.immutable();E.G.deterministic();E.C.guard();audit=E.read(E.DOC/'INPUT_EXPOSURE_AUDIT.json');assert E.read(E.DOC/'PRETRAIN_TESTS.json')['passed'];assert E.read(E.C.DOC/'LOSS_TEST.json')['passed']
    args=audit['args'];run=E.RAW/'runs'/arm;assert not run.exists(),'No silent restart'
    Trainer.arm=arm;t=Trainer(overrides=dict(args,model=str(E.C.H.C.N.E.R0),data=audit['datasets']['E'],project=str(E.RAW/'runs'),name=arm,exist_ok=False));t.trace=[];steps=[];epochs=[];initial={};inventory=[];start=time.monotonic()
    def begin(t):
        initial.update({k:E.C.digest(v.cpu().numpy()) for k,v in t.model.state_dict().items()});assert initial==E.read(E.C.DOC/'FIT_PLASTIC_S1.json')['initial_state']
        inventory.extend(dict(name=n,shape=list(p.shape),trainable=p.requires_grad) for n,p in t.model.named_parameters())
        def pre(*_):
            assert len(steps)<320 and all(p.grad is None or torch.isfinite(p.grad).all() for p in t.model.parameters())
        t.optimizer.register_step_pre_hook(pre);t.optimizer.register_step_post_hook(lambda *_:steps.append(len(steps)+1))
    def epochstart(t):t.train_loader.dataset.epoch=t.epoch;E.C.guard()
    def epochend(t):
        assert torch.isfinite(t.tloss).all();epochs.append(dict(epoch=t.epoch,steps=len(steps),loss=t.tloss.cpu().tolist(),gpu=E.C.guard()));print('FIT_PROGRESS',arm,t.epoch+1,len(steps),flush=True)
    for k,f in [('on_train_start',begin),('on_train_epoch_start',epochstart),('on_train_epoch_end',epochend)]:t.add_callback(k,f)
    t.train();assert len(steps)==320 and len(t.trace)==5120 and [r['occ'] for r in t.trace]==list(range(5120))
    E.save(E.RAW/f'TRACE_{arm}.json',t.trace);E.save(E.DOC/f'FIT_{arm}.json',dict(complete=True,steps=320,epochs=epochs,initial_state=initial,inventory=inventory,checkpoint=E.bind(run/'weights/last.pt'),trace=E.bind(E.RAW/f'TRACE_{arm}.json'),seconds=time.monotonic()-start,real_exposures=2560,synthetic_exposures=2560,C0_exposures=1280,replacement_exposures=1280,no_eval_during_train=True,no_rescue=True,last_only=True))
    print('FIT_COMPLETE',arm,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=E.ARMS);a=p.parse_args();main(a.arm)
