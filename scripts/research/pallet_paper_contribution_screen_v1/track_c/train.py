"""Paired 900 *optimizer update* screen, no best checkpoint or real GT loader.

Each update uses the same 24 synthetic images; C1/C2 add the same 8 real images.
Separate domain microbatches keep synthetic mosaic/RNG/exposure identical to C0.
This is a newly locked bounded comparison, not a reproduction of V3 epoch counts.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.contracts import ROOT, RAW, DOC, R0, sha, write, tensor_sha
from track_c.wiring import load_model, freeze_detection_only, detection_parameters
sys.path.insert(0, str(ROOT/'scripts/self_training_yolo/v3'))
from true_ignore_pose_loss import make_criterion

HYP = dict(task='pose',imgsz=640,batch=32,epochs=10,optimizer='SGD',lr0=.002,lrf=.01,
    momentum=.937,weight_decay=.0005,warmup_epochs=1.,box=7.5,cls=.5,dfl=1.5,
    pose=12.,kobj=1.,hsv_h=.015,hsv_s=.5,hsv_v=.35,degrees=0.,translate=.1,
    scale=.25,shear=0.,perspective=0.,flipud=0.,fliplr=0.,mosaic=.15,
    close_mosaic=3,mixup=0.,copy_paste=0.,erasing=.4,deterministic=True)
DATA = dict(names={0:'item'},kpt_shape=[9,3],flip_idx=[1,0,3,2,5,4,7,6,8],channels=3)

def prepare():
    base=RAW/'C_geometry_preserving_da/dataset'
    receipt=base/'BINDINGS.json'
    if receipt.exists(): return json.loads(receipt.read_text())
    original=ROOT/'challenge/yolo_pose_one_model/datasets/paper_selftrain_v3/V3A_TRUE_IGNORE'
    names=(ROOT/'data/pallet/results/paper_selftrain_v1/SYNTHETIC_REPLAY_SUBSET.txt').read_text().splitlines()
    sources={'synthetic':[original/'images/train'/('replay__'+n) for n in names],
             'real':sorted((original/'images/train').glob('pseudo__*'))}
    assert len(sources['synthetic'])==1440 and len(sources['real'])==273
    bindings={}
    for domain,images in sources.items():
        out=base/domain
        (out/'images').mkdir(parents=True,exist_ok=True);(out/'labels').mkdir(exist_ok=True)
        records=[]
        for image in images:
            assert image.is_file(),image
            label=original/'labels/train'/image.with_suffix('.txt').name
            target=out/'images'/image.name
            if not target.exists(): target.symlink_to(image.resolve())
            dest=out/'labels'/label.name
            if not dest.exists(): shutil.copyfile(label,dest)
            assert sha(dest)==sha(label)
            records.append(dict(image=str(image.resolve()),image_sha256=sha(image),label=str(label),label_sha256=sha(label)))
        bindings[domain]=records
    assert not {r['image_sha256'] for r in bindings['synthetic']} & {r['image_sha256'] for r in bindings['real']}
    write(receipt,bindings)
    return bindings

class PairedData(Dataset):
    def __init__(self,domain,seed,epoch):
        self.seed,self.domain,self.epoch=seed,domain,epoch
        self.hyp=get_cfg(overrides=HYP)
        if epoch>=7:self.hyp.mosaic=0.
        self.dataset=YOLODataset(img_path=str(RAW/'C_geometry_preserving_da/dataset'/domain/'images'),
            imgsz=640,batch_size=24 if domain=='synthetic' else 8,augment=True,hyp=self.hyp,
            rect=False,cache=False,stride=32,pad=0.,data=DATA,task='pose',prefix=domain+': ')
    def __len__(self):return len(self.dataset)
    def __getitem__(self,key):
        index,position=key
        seed=self.seed*10000000+self.epoch*100000+position+(0 if self.domain=='synthetic' else 50000)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        return self.dataset[index]

def batches(n,batch,seed,epoch):
    rng=np.random.default_rng(seed*1000+epoch)
    indices=[]
    while len(indices)<90*batch:indices.extend(rng.permutation(n).tolist())
    return [[(indices[j],j) for j in range(i,i+batch)] for i in range(0,90*batch,batch)]

def loader(domain,seed,epoch):
    data=PairedData(domain,seed,epoch)
    batch=24 if domain=='synthetic' else 8
    return DataLoader(data,batch_sampler=batches(len(data),batch,seed,epoch),num_workers=2,
        collate_fn=YOLODataset.collate_fn,pin_memory=True)

def device_batch(batch):
    return {k:(v.cuda(non_blocking=True).float()/255 if k=='img' else v.cuda(non_blocking=True))
            if torch.is_tensor(v) else v for k,v in batch.items()}

def batch_digest(batch):
    return tensor_sha({k:batch[k] for k in ['img','batch_idx','cls','bboxes','keypoints']})

def optimizer(model):
    bias,norm,weight=[],[],[]
    for mod in model.modules():
        for name,p in mod.named_parameters(recurse=False):
            if not p.requires_grad:continue
            (bias if name=='bias' else norm if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm) else weight).append(p)
    return torch.optim.SGD([dict(params=bias,weight_decay=0.,is_bias=True),
        dict(params=weight,weight_decay=.0005,is_bias=False),dict(params=norm,weight_decay=0.,is_bias=False)],
        lr=.002,momentum=.937,nesterov=True)

def frozen_buffers(model):
    result={}
    for prefix,module in model.named_modules():
        if module.training:continue
        for buffer_name,value in module.named_buffers(recurse=False):
            key=f'{prefix}.{buffer_name}' if prefix else buffer_name
            assert key in model.state_dict()
            result[key]=value.detach().cpu().clone()
    return result

def train(arm,seed):
    torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    out=RAW/f'C_geometry_preserving_da/{arm}_seed{seed}'
    if (out/'TRAINING_AUDIT.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'last.pt').exists(),'Never overwrite a checkpoint'
    m=load_model().cuda();m.args=get_cfg(overrides=HYP)
    initial=tensor_sha(m.state_dict())
    if arm=='C2':freeze_detection_only(m)
    else:
        m.train()
        # Stock DFL integral is fixed, not a learned geometry layer.
        if hasattr(m.model[-1],'dfl'):m.model[-1].dfl.requires_grad_(False)
    allowed={id(p) for p in m.parameters() if p.requires_grad}
    opt=optimizer(m)
    assert {id(p) for g in opt.param_groups for p in g['params']}==allowed
    if arm=='C2':assert allowed==detection_parameters(m)
    frozen={n:v.detach().cpu().clone() for n,v in m.named_parameters() if not v.requires_grad}
    bn=frozen_buffers(m)
    probe=torch.linspace(0,1,3*64*64,device='cuda').reshape(1,3,64,64).repeat(2,1,1,1)
    raw_before=None
    if arm=='C2':
        with torch.no_grad():raw_before={k:v['kpts'].clone() for k,v in m(probe).items()}
    criterion=make_criterion(m);det_criterion=E2ELoss(m,v8DetectionLoss)
    updates=0;exposure=[];start=time.time();losslog=[]
    for epoch in range(10):
        syn=loader('synthetic',seed,epoch)
        real=iter(loader('real',seed,epoch)) if arm!='C0' else None
        for si,batch in enumerate(syn):
            step=epoch*90+si
            ratio=(1+math.cos(math.pi*epoch/10))/2*.99+.01
            for group in opt.param_groups:
                group['lr']=float(np.interp(step,[0,90],[.1 if group['is_bias'] else 0.,.002*ratio])) if step<=90 else .002*ratio
                group['momentum']=float(np.interp(step,[0,90],[.8,.937])) if step<=90 else .937
            opt.zero_grad(set_to_none=True)
            record=dict(step=step,synthetic=batch_digest(batch))
            sb=device_batch(batch)
            loss=criterion(m(sb['img']),sb)[0].sum()
            assert torch.isfinite(loss),'Nonfinite synthetic loss'
            loss.backward();value=float(loss.detach())
            if real is not None:
                rb=next(real);record['real']=batch_digest(rb);rb=device_batch(rb)
                # C2 real loss never reads keypoints, including invisible labels.
                if arm=='C2':rb.pop('keypoints')
                loss=(det_criterion if arm=='C2' else criterion)(m(rb['img']),rb)[0].sum()
                assert torch.isfinite(loss),'Nonfinite real loss'
                loss.backward();value+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(m.parameters(),10.,error_if_nonfinite=True)
            opt.step();updates+=1;exposure.append(record);losslog.append(value)
            if updates%30==0:print(json.dumps(dict(arm=arm,seed=seed,updates=updates,loss=value,elapsed_s=round(time.time()-start),allocated_mb=round(torch.cuda.memory_allocated()/2**20))),flush=True)
        criterion.update();det_criterion.update()
    assert updates==900
    # Persist recoverable final weights BEFORE reporting assertions. An audit
    # failure must never require repeating a scientifically budgeted fit.
    saved=copy.deepcopy(m).cpu().eval();saved.args=vars(saved.args)
    if hasattr(saved,'criterion'):delattr(saved,'criterion')
    pending=out/'last_unverified.pt'
    assert not pending.exists()
    torch.save(dict(model=saved,ema=None,train_args=HYP,epoch=9,optimizer=None),pending)
    state=m.state_dict()
    assert all(torch.equal(v,state[n].cpu()) for n,v in frozen.items())
    assert all(torch.equal(v,state[n].cpu()) for n,v in bn.items())
    if arm=='C2':
        with torch.no_grad():assert all(torch.equal(raw_before[k],v['kpts']) for k,v in m(probe).items())
    pending.rename(out/'last.pt')
    write(out/'EXPOSURE.json',exposure)
    write(out/'TRAINING_AUDIT.json',dict(status='PASS',arm=arm,seed=seed,optimizer_updates=updates,
        synthetic_exposure=21600,real_exposure=0 if arm=='C0' else 7200,
        init_state_sha256=initial,checkpoint_sha256=sha(out/'last.pt'),
        frozen_parameters_equal=True,frozen_bn_buffers_equal=True,
        raw_keypoints_bit_exact=arm=='C2' or None,finite_gradients=True,loss_first=losslog[0],loss_last=losslog[-1],
        elapsed_s=time.time()-start,trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['C0','C1','C2'],required=True);ap.add_argument('--seed',type=int,required=True)
    args=ap.parse_args();assert torch.cuda.is_available();prepare();train(args.arm,args.seed)

if __name__=='__main__':main()
