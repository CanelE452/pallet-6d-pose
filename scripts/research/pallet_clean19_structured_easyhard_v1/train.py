import argparse
import csv
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from ultralytics.data.dataset import YOLODataset
from scripts.self_training_yolo.v3.true_ignore_trainer import TrueIgnorePoseTrainer
from . import common as C
from . import augmentation as A


def plans(material):
    return [json.loads(line) for line in (C.DOC/'AUGMENTATION_PLAN.jsonl').read_text().splitlines() if json.loads(line)['material']==material]


class FrozenDataset(Dataset):
    def __init__(self,material,arm):
        self.records=plans(material);self.arm=arm;self.epoch=0
        assert len(self.records)==5120
        self.labels=[]
        for r in self.records[:1024]:
            with np.load(C.ROOT/r['cache']['path']) as z:self.labels.append(dict(cls=z['cls'],bboxes=z['bboxes']))

    def __len__(self):return 1024

    def __getitem__(self,index):
        r=self.records[self.epoch*1024+index]
        assert r['slot']==index and r['epoch']==self.epoch
        with np.load(C.ROOT/r['cache']['path']) as z:
            d={k:torch.from_numpy(z[k].copy()) for k in ('keypoints','bboxes','cls','batch_idx')};img=z['img'].copy()
        assert C.digest(img)==r['base_RGB_sha256']
        assert C.digest(d['keypoints'].numpy())==r['target_sha256']
        d['img']=torch.from_numpy(A.apply(img,r['plan'],self.arm))
        d.update(im_file=r['image'],ori_shape=(640,640),resized_shape=(640,640),ratio_pad=((1.,1.),(0.,0.)),occ=self.epoch*1024+index)
        return d


class PairedTrainer(TrueIgnorePoseTrainer):
    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
        ds=FrozenDataset(self.material,self.arm)
        return DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=self.args.workers,persistent_workers=False,
            collate_fn=YOLODataset.collate_fn,pin_memory=True,generator=torch.Generator().manual_seed(42))

    def preprocess_batch(self,batch):
        for i,occ in enumerate(batch['occ']):
            r=self.train_loader.dataset.records[occ]
            kp=batch['keypoints'][batch['batch_idx']==i].numpy()
            box=batch['bboxes'][batch['batch_idx']==i].numpy()
            assert C.digest(kp)==r['target_sha256'] and C.digest(box)==r['box_sha256']
            self.trace.append(dict(occ=occ,real=r['real'],base_RGB_sha256=r['base_RGB_sha256'],input_sha256=C.digest(batch['img'][i].numpy()),
                target_sha256=C.digest(kp),box_sha256=C.digest(box),applied=self.arm!='S0' and r['plan']['applied']))
        return super().preprocess_batch(batch)


def main(material,arm):
    C.setup();C.guard();C.immutable()
    assert C.read(C.DOC/'LOSS_TEST.json')['passed']
    assert C.read(C.DOC/'PROBE_BEFORE.json')['complete']
    assert C.read(C.DOC/'AUGMENTATION_AUDIT.json')['status']=='READY'
    protocol=C.read(C.DOC/'PROTOCOL.json')
    for b in protocol['bindings']:C.verify(b)
    name=f'{material}_{arm}';run=C.RAW/'runs'/name
    assert not run.exists(),'Interrupted fits must not be silently restarted'
    args=C.read(C.DOC/'PREFLIGHT.json')['args']
    PairedTrainer.material=material;PairedTrainer.arm=arm
    trainer=PairedTrainer(overrides=dict(args,model=str(C.H.C.N.E.R0),data=C.read(C.DOC/'PREFLIGHT.json')['datasets'][material],project=str(C.RAW/'runs'),name=name,exist_ok=False))
    trainer.trace=[];steps=[];epochs=[];inventory=[];initial={};start=time.monotonic()
    def on_start(t):
        src=torch.load(C.H.C.N.E.R0,map_location='cpu',weights_only=False)['model'].float().state_dict()
        for k,v in t.model.state_dict().items():
            if k in src and v.shape==src[k].shape:assert torch.equal(v.cpu(),src[k])
        inventory.extend(dict(name=n,shape=list(p.shape),trainable=p.requires_grad) for n,p in t.model.named_parameters())
        initial.update({k:C.digest(v.cpu().numpy()) for k,v in t.model.state_dict().items()})
        t.optimizer.register_step_post_hook(lambda opt,args,kwargs:steps.append(len(steps)+1))
    def epoch_start(t):t.train_loader.dataset.epoch=t.epoch;C.guard()
    def epoch_end(t):
        assert torch.isfinite(t.tloss).all(),'NaN/Inf training loss'
        epochs.append(dict(epoch=t.epoch,steps=len(steps),loss=t.tloss.detach().cpu().tolist(),gpu=C.guard()))
        print('FIT_PROGRESS',name,t.epoch+1,len(steps),flush=True)
    trainer.add_callback('on_train_start',on_start)
    trainer.add_callback('on_train_epoch_start',epoch_start)
    trainer.add_callback('on_train_epoch_end',epoch_end)
    trainer.train()
    assert len(steps)==320 and len(trainer.trace)==5120
    assert [r['occ'] for r in trainer.trace]==list(range(5120))
    assert sum(r['real'] for r in trainer.trace)==2560
    C.save(C.RAW/f'TRACE_{name}.json',trainer.trace)
    C.save(C.DOC/f'FIT_{name}.json',dict(complete=True,steps=len(steps),seconds=time.monotonic()-start,epochs=epochs,
        checkpoint=C.bind(run/'weights/last.pt'),trace=C.bind(C.RAW/f'TRACE_{name}.json'),trainable_inventory=inventory,
        initial_state=initial,real_exposures=2560,synthetic_exposures=2560,protocol=C.bind(C.DOC/'PROTOCOL.json')))
    print('FIT_COMPLETE',name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('material',choices=C.MATERIALS);p.add_argument('arm',choices=C.ARMS);a=p.parse_args();main(a.material,a.arm)
