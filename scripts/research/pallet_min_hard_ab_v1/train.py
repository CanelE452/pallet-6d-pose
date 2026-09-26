"""Matched original-R0 initialization, S1 budget, and hard xy-only supervision."""
import argparse
import time
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from ultralytics.data.dataset import YOLODataset
from scripts.self_training_yolo.v3.true_ignore_trainer import TrueIgnorePoseTrainer,TrueIgnorePoseModel
from scripts.research.pallet_clean19_structured_easyhard_v1 import common as S,augmentation as A
from . import common as C
from .loss import criterion


class HardModel(TrueIgnorePoseModel):
    def init_criterion(self):return criterion(self)


class FrozenDataset(Dataset):
    def __init__(self,arm):
        lock=C.read(C.DOC/'TRAIN_OCCURRENCE_LOCK.json');C.verify(lock['occurrences'])
        self.records=C.read(C.ROOT/lock['occurrences']['path']);self.arm=arm;self.epoch=0;self.labels=[]
        for r in self.records[:1024]:
            with np.load(C.ROOT/r['hard_cache' if r['hard'] else 'cache']['path']) as z:
                self.labels.append(dict(cls=z['cls'],bboxes=z['bboxes']))
    def __len__(self):return 1024
    def __getitem__(self,index):
        r=self.records[self.epoch*1024+index];hard=r['hard']
        with np.load(C.ROOT/r['hard_cache' if hard else 'cache']['path']) as z:
            d={k:torch.from_numpy(z[k].copy()) for k in ('bboxes','cls','batch_idx')}
            kp=z[self.arm if hard else 'keypoints'].copy();img=z['img'].copy()
        if hard:
            assert S.digest(img)==r['hard_RGB_sha256'] and S.digest(kp)==r['hard_targets'][self.arm]
        else:
            assert S.digest(img)==r['base_RGB_sha256'] and S.digest(kp)==r['target_sha256']
            img=A.apply(img,r['plan'],'S1')
        d.update(img=torch.from_numpy(img),keypoints=torch.from_numpy(kp),hard=hard,occ=self.epoch*1024+index,
                 im_file=r['image'],ori_shape=(640,640),resized_shape=(640,640),ratio_pad=((1.,1.),(0.,0.)))
        return d


class Trainer(TrueIgnorePoseTrainer):
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=HardModel(cfg,ch=self.data['channels'],nc=self.data['nc'],data_kpt_shape=self.data['kpt_shape'],verbose=verbose)
        if weights:model.load(weights)
        return model
    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
        return DataLoader(FrozenDataset(self.arm),batch_size=batch_size,shuffle=False,num_workers=self.args.workers,
                          persistent_workers=False,collate_fn=YOLODataset.collate_fn,pin_memory=True,generator=torch.Generator().manual_seed(42))
    def preprocess_batch(self,batch):
        for i,occ in enumerate(batch['occ']):
            r=self.train_loader.dataset.records[occ];kp=batch['keypoints'][batch['batch_idx']==i].numpy();box=batch['bboxes'][batch['batch_idx']==i].numpy()
            assert S.digest(kp)==(r['hard_targets'][self.arm] if r['hard'] else r['target_sha256'])
            assert S.digest(box)==r['hard_box_sha256' if r['hard'] else 'box_sha256']
            self.trace.append(dict(occ=occ,hard=r['hard'],kind=r['kind'],input_sha256=S.digest(batch['img'][i].numpy()),target_sha256=S.digest(kp),box_sha256=S.digest(box)))
        return super().preprocess_batch(batch)


def main(arm):
    if (C.DOC/f'FIT_{arm}.json').exists():
        C.verify(C.read(C.DOC/f'FIT_{arm}.json')['checkpoint']);return
    assert C.read(C.DOC/'PRETRAIN_TESTS.json')['passed']
    S.H.P.N.setup()
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    S.guard();lock=C.read(C.DOC/'TRAIN_OCCURRENCE_LOCK.json')
    for k in ('original_init','base_checkpoint','fit_binding','label_lock','teacher_lock','occurrences'):C.verify(lock[k])
    baseline=C.read(C.ROOT/lock['fit_binding']['path']);run=C.RAW/'runs'/arm
    assert not run.exists(),'Do not silently restart an interrupted fit'
    Trainer.arm=arm
    trainer=Trainer(overrides=dict(lock['args'],model=str(C.ROOT/lock['original_init']['path']),
        data=C.read(S.DOC/'PREFLIGHT.json')['datasets']['PLASTIC'],project=str(C.RAW/'runs'),name=arm,exist_ok=False))
    trainer.trace=[];steps=[];epochs=[];inventory=[];start=time.monotonic()
    def started(t):
        inventory.extend(dict(name=n,shape=list(p.shape),trainable=p.requires_grad) for n,p in t.model.named_parameters())
        assert inventory==baseline['trainable_inventory']
        initial={k:S.digest(v.cpu().numpy()) for k,v in t.model.state_dict().items()}
        assert initial==baseline['initial_state'],'original init parity'
        t.optimizer.register_step_post_hook(lambda *_:steps.append(len(steps)+1))
    def epoch_start(t):t.train_loader.dataset.epoch=t.epoch;S.guard()
    def epoch_end(t):
        assert torch.isfinite(t.tloss).all()
        epochs.append(dict(epoch=t.epoch,steps=len(steps),loss=t.tloss.detach().cpu().tolist(),gpu=S.guard()))
        C.save(C.OUT/f'PROGRESS_{arm}.json',dict(epochs=epochs,complete=False))
        print('HARD_FIT_PROGRESS',arm,t.epoch+1,len(steps),flush=True)
    trainer.add_callback('on_train_start',started);trainer.add_callback('on_train_epoch_start',epoch_start);trainer.add_callback('on_train_epoch_end',epoch_end)
    C.set_state('TRAINING',arm=arm,training='RUNNING')
    trainer.train()
    assert len(steps)==320 and [r['occ'] for r in trainer.trace]==list(range(5120))
    assert sum(r['hard'] for r in trainer.trace)==320
    C.save(C.RAW/f'TRACE_{arm}.json',trainer.trace,immutable=True)
    C.save(C.DOC/f'FIT_{arm}.json',dict(complete=True,steps=len(steps),epochs=epochs,seconds=time.monotonic()-start,
           checkpoint=C.bind(run/'weights/last.pt'),trace=C.bind(C.RAW/f'TRACE_{arm}.json'),trainable_inventory=inventory,
           initial_state_verified=True,checkpoint_selection='last',actual_args=str(trainer.args),
           protocol=C.bind(C.DOC/'TRAIN_OCCURRENCE_LOCK.json')),immutable=True)
    C.set_state('FIT_COMPLETE_EVALUATION_PENDING',arm=arm,training='FIT_COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['H_PSEUDO','H_MANUAL']);main(p.parse_args().arm)
