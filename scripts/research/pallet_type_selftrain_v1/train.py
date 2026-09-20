"""Fixed last-epoch single-seed R0 copies, separated by pallet type."""
import argparse
import csv
import gc
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from . import common as C
from .pseudo import top

SOURCE=C.ROOT/'challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k'
REPLAY=C.ROOT/'data/pallet/results/paper_selftrain_v1/SYNTHETIC_REPLAY_SUBSET.txt'
ARMS=['SYN_ONLY','PLASTIC','GREEN']
ARGS=dict(epochs=5,imgsz=640,batch=16,nbs=16,device='0',optimizer='AdamW',lr0=.0001,lrf=.1,cos_lr=True,
    weight_decay=.0001,warmup_epochs=0.,patience=0,seed=42,deterministic=True,
    box=7.5,cls=.5,dfl=1.5,pose=12.,kobj=1.,hsv_h=.015,hsv_s=.5,hsv_v=.35,
    degrees=0.,translate=.1,scale=.25,shear=0.,perspective=0.,flipud=0.,fliplr=0.,
    mosaic=0.,close_mosaic=0,mixup=0.,copy_paste=0.,erasing=0.,
    workers=2,cache=False,amp=False,val=False,plots=False,save=True,save_period=-1,verbose=False)


def label(candidate,hw,pad=100):
    """Pad transformation only. Ignore uncertain pseudo points via existing v3 sentinel."""
    h,w=hw;hh,ww=h+2*pad,w+2*pad
    box=np.asarray(candidate['box_xyxy'],float).copy();box[[0,2]]=np.clip(box[[0,2]],0,w);box[[1,3]]=np.clip(box[[1,3]],0,h)
    if np.any(box[2:]-box[:2]<=1):return None
    center=(box[:2]+box[2:])/2+pad;size=box[2:]-box[:2]
    values=[0,*list(center/[ww,hh]),*list(size/[ww,hh])]
    points=np.asarray(candidate['keypoints_xy'],float);conf=np.asarray(candidate['keypoints_conf'])
    for i,(x,y) in enumerate(points):
        supervise=bool(np.isfinite([x,y]).all() and 0<=x<w and 0<=y<h and conf[i]>=.5)
        if supervise:xy=(np.array([x,y])+pad)/[ww,hh];v=2
        else:xy=np.array([.5,.5]);v=1  # untrusted coordinate has no gradient
        values.extend([*xy,v])
    assert len(values)==32 and np.isfinite(values).all()
    return ' '.join(f'{x:.9f}' for x in values)+'\n'


def link(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert dest.resolve()==source.resolve()
    else:dest.symlink_to(source.resolve())


def dataset():
    done=C.read(C.DOC/'PSEUDO_COMPLETE.json')
    for b in done['artifacts']+[done['protocol']]:C.verify(b)
    accepted=C.read(C.RAW/'PSEUDO_ACCEPTED.json')
    replay=[x for x in REPLAY.read_text().splitlines() if x]
    assert len(replay)>=512
    rng=np.random.default_rng(9020);replay=sorted(rng.choice(replay,512,replace=False).tolist())
    real={kind:[] for kind in ['PLASTIC','GREEN']};bindings=[]
    shared=C.RAW/'dataset'
    for name in replay:
        image=SOURCE/'images/train'/name;lbl=(SOURCE/'labels/train'/name).with_suffix('.txt')
        text=lbl.read_text()
        for row in text.splitlines():assert set(np.array(row.split()[5:],float).reshape(9,3)[:,2])<={0.,2.}
        dest=shared/'images'/('syn__'+name);link(image,dest)
        C.write_text((shared/'labels'/('syn__'+name)).with_suffix('.txt'),text)
        bindings.extend([C.bound(image),C.bound(lbl)])
    for row in accepted:
        candidate=top(row['refined']);text=label(candidate,row['raw_hw'])
        if text is None:continue
        imagepath=shared/'images'/(row['id']+'.png')
        C.verify(row['image'])
        if not imagepath.exists():
            image=cv2.imread(str(C.ROOT/row['image']['path']));assert image is not None
            padded=cv2.copyMakeBorder(image,100,100,100,100,cv2.BORDER_REFLECT_101)
            imagepath.parent.mkdir(parents=True,exist_ok=True);assert cv2.imwrite(str(imagepath),padded)
        C.write_text(shared/'labels'/(row['id']+'.txt'),text)
        real[row['kind']].append(str(imagepath))
        bindings.extend([C.bound(imagepath),C.bound(shared/'labels'/(row['id']+'.txt'))])
    # Synthetic-only small validation for framework final-epoch bookkeeping, never real eval.
    val=sorted((SOURCE/'images/val').glob('*.png'))[:32];assert len(val)==32
    C.write_text(shared/'val.txt','\n'.join(map(str,val))+'\n')
    for p in val:bindings.extend([C.bound(p),C.bound((SOURCE/'labels/val'/p.name).with_suffix('.txt'))])
    syn=[str(shared/'images'/('syn__'+n)) for n in replay]
    datasets={}
    for arm in ARMS:
        if arm!='SYN_ONLY' and not real[arm]:continue
        replacement=syn if arm=='SYN_ONLY' else np.random.default_rng(9021).choice(sorted(real[arm]),512,replace=True).tolist()
        slots=syn+replacement;assert len(slots)==1024
        trainlist=shared/f'{arm}_train.txt';C.write_text(trainlist,'\n'.join(slots)+'\n')
        yaml=shared/f'{arm}.yaml'
        C.write_text(yaml,f'path: {shared}\ntrain: {trainlist}\nval: {shared/"val.txt"}\nnc: 1\nnames: [pallet]\nkpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n')
        datasets[arm]=dict(data=C.bound(yaml),train_list=C.bound(trainlist),slots=1024,pseudo_slots=0 if arm=='SYN_ONLY' else 512,
                           pseudo_unique=len(real.get(arm,[])),sampled_pseudo_unique=0 if arm=='SYN_ONLY' else len(set(replacement)))
    protocol=dict(args=ARGS,arms=list(datasets),initialization=C.bound(C.N.E.R0),selection='fixed last.pt after5epochs; no eval checkpoint or threshold selection',
        budget='5epochs x1024slots /batch16,nbs16 =320 optimizer steps expected per arm; actual optimizer hook count recorded',
        real_supervision='Replay-refined fixed pseudo keypoints; never real annotations. Original predicted R0 boxes. No second teacher refresh.',
        loss='Reuse existing TrueIgnorePoseLoss26: uncertain points sentinel1 have no coordinate/RLE/keypoint-objectness gradient; supervised2. Synthetic0/2 stock-equivalent.',
        padding='Real image reflect101 border100 exactly once; box/valid coordinates shifted100 then normalized. Original stored pseudo unchanged. Synthetic already prepared source.',
        synthetic_replay='Same fixed512 synthetic examples including any original empty labels, for every student; mixed pallet source replay',
        exposure='per-type real pseudo, shared mixed-source replay1:1 by epoch slots (not guaranteed within each shuffled minibatch)',
        synthetic_control='SYN_ONLY replaces512 pseudo slots with same synthetic512; same optimizer budget',
        validation='32 original synthetic val images solely framework bookkeeping; no real eval used during training',
        auto_promote=False,independent_confirmation=False,system_changes=False,temperature_limit_C=80,
        datasets=datasets,inputs=bindings,sources=[C.bound(C.DOC/'PSEUDO_COMPLETE.json'),C.bound(REPLAY),
          C.bound(C.ROOT/'scripts/self_training_yolo/v3/true_ignore_trainer.py'),C.bound(C.ROOT/'scripts/self_training_yolo/v3/true_ignore_pose_loss.py')],
        code=C.bound(__file__),common=C.bound(C.HERE/'common.py'),test_code=C.bound(C.HERE/'test_contract.py'))
    C.freeze(C.DOC/'TRAIN_PROTOCOL.json',protocol)
    print('DATASETS',datasets,flush=True)
    return protocol


def train(arm):
    from scripts.self_training_yolo.v3.true_ignore_trainer import TrueIgnorePoseTrainer
    protocol=C.read(C.DOC/'TRAIN_PROTOCOL.json')
    for b in protocol['inputs']+protocol['sources']+[protocol['code'],protocol['common'],protocol['test_code'],protocol['initialization']]:C.verify(b)
    fit=C.DOC/f'FIT_{arm}.json'
    if fit.exists():
        C.verify(C.read(fit)['checkpoint']);print('COMPLETE_ALREADY',arm,flush=True);return
    assert arm in protocol['datasets']
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))//1024
    assert available>=6000,('RAM guard',available)
    C.N.setup();torch.set_num_interop_threads(1);C.N.E.gpu()
    assert torch.cuda.is_available()
    run_dir=C.RAW/'runs'/arm
    assert not run_dir.exists(),'Incomplete run exists; preserve it and explicitly resume before rerun'
    trainer=TrueIgnorePoseTrainer(overrides=dict(protocol['args'],model=str(C.ROOT/protocol['initialization']['path']),
        data=str(C.ROOT/protocol['datasets'][arm]['data']['path']),project=str(C.RAW/'runs'),name=arm,exist_ok=False))
    steps=[];epochs=[];start=time.monotonic()
    def on_start(t):
        # Initialisation parity checked before first optimizer step (framework transfers raw R0 weights).
        ck=torch.load(C.N.E.R0,map_location='cpu',weights_only=False);src=ck['model'].float().state_dict();dst=t.model.state_dict()
        shared=[k for k,v in src.items() if k in dst and v.shape==dst[k].shape and torch.is_floating_point(v)]
        assert shared and max(float((src[k]-dst[k].detach().cpu()).abs().max()) for k in shared)==0
        t.optimizer.register_step_post_hook(lambda opt,args,kwargs:steps.append(len(steps)+1))
        print('R0_INITIALIZATION_EXACT',arm,len(shared),flush=True)
    def on_epoch(t):
        gpu=C.N.E.gpu();epochs.append(dict(epoch=t.epoch+1,optimizer_steps=len(steps),gpu=gpu))
        print('TYPE_TRAIN_PROGRESS',arm,t.epoch+1,'/',protocol['args']['epochs'],'steps',len(steps),flush=True)
    trainer.add_callback('on_train_start',on_start);trainer.add_callback('on_train_epoch_end',on_epoch)
    trainer.train()
    checkpoint=run_dir/'weights/last.pt';rows=list(csv.DictReader((run_dir/'results.csv').open()))
    assert checkpoint.is_file() and len(rows)==5 and len(steps)==320,(len(rows),len(steps))
    base=torch.load(C.N.E.R0,map_location='cpu',weights_only=False)['model'].float().state_dict()
    final=torch.load(checkpoint,map_location='cpu',weights_only=False)['model'].float().state_dict()
    changed=sum(not torch.equal(v,final[k]) for k,v in base.items() if k in final and v.shape==final[k].shape)
    assert changed>0
    C.freeze(fit,dict(complete=True,arm=arm,epochs=5,optimizer_steps=len(steps),checkpoint=C.bound(checkpoint),
        protocol=C.bound(C.DOC/'TRAIN_PROTOCOL.json'),seconds=time.monotonic()-start,epoch_history=epochs,changed_tensors=changed,
        final_csv=C.bound(run_dir/'results.csv'),initialization_exact=True))
    print('TYPE_TRAIN_COMPLETE',arm,round(time.monotonic()-start,1),flush=True)
    del trainer;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['dataset',*ARMS]);a=p.parse_args()
    dataset() if a.stage=='dataset' else train(a.stage)
