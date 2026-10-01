"""Fixed source-only ResNet18 pallet estimator; exact resumable training state."""
import argparse
import math
import signal
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from base_common import *
from model import SimpleBaselineResNet18
from input_data import SourceDataset,masked_mse,worker_init_fn

STOP=False
def request_stop(signum,frame):
    global STOP
    STOP=True
    print('GRACEFUL_STOP_REQUESTED',signum,flush=True)

def initialize():
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    state=torch.load(PRETRAINED,map_location='cpu',weights_only=True)
    return SimpleBaselineResNet18(pretrained_state=state).cuda()

def photometric(image):
    mean=image.new_tensor([.485,.456,.406])[None,:,None,None]
    std=image.new_tensor([.229,.224,.225])[None,:,None,None]
    rgb=image*std+mean
    factors=.8+.4*torch.rand((3,len(image),1,1,1),device=image.device)
    rgb=rgb*factors[0]
    grey=(rgb*rgb.new_tensor([.2989,.5870,.1140])[None,:,None,None]).sum(1,keepdim=True)
    center=grey.mean((2,3),keepdim=True)
    rgb=(rgb-center)*factors[1]+center
    grey=(rgb*rgb.new_tensor([.2989,.5870,.1140])[None,:,None,None]).sum(1,keepdim=True)
    rgb=(rgb-grey)*factors[2]+grey
    return (rgb.clamp(0,1)-mean)/std

def one_update(model,optimizer,batch,physical_batch,augment=True):
    n=len(batch['image']);optimizer.zero_grad(set_to_none=True);total=0.
    for first in range(0,n,physical_batch):
        last=min(first+physical_batch,n)
        x=batch['image'][first:last].cuda(non_blocking=True)
        target=batch['heatmaps'][first:last].cuda(non_blocking=True)
        mask=batch['target_valid'][first:last].cuda(non_blocking=True)
        if augment:x=photometric(x)
        pred=model(x);assert pred.shape==target.shape and torch.isfinite(pred).all()
        value=masked_mse(pred,target,mask);assert torch.isfinite(value)
        weight=(last-first)/n;(value*weight).backward();total+=float(value.detach())*weight
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step();assert all(torch.isfinite(p).all() for p in model.parameters())
    return total

def smoke():
    destination=DOC/'BASELINE_GPU_SMOKE.json'
    assert not destination.exists()
    assert sha(PRETRAINED)==PRETRAINED_SHA
    dataset=SourceDataset(SOURCE,partition='train')
    batch=next(iter(DataLoader(Subset(dataset,list(range(16))),batch_size=16,num_workers=0)))
    model=initialize();optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    model.train();losses=[];durations=[]
    torch.cuda.reset_peak_memory_stats()
    # These discarded updates test finite gradients and measure implementation
    # throughput; they do not select architecture/epochs/accuracy thresholds.
    for step in range(8):
        torch.cuda.synchronize();start=time.perf_counter()
        value=one_update(model,optimizer,batch,physical_batch=16)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        losses.append(value);durations.append(elapsed)
        print('R18_BASELINE_SMOKE',step+1,value,elapsed,flush=True)
    forward=model(batch['image'][:1].cuda(),return_features=True)
    assert tuple(forward['heatmaps'].shape)==(1,9,96,128)
    a,b=forward['features'];assert a.shape==(1,128,48,64) and b.shape==(1,256,24,32)
    estimate=float(np.mean(durations[2:]))*math.ceil(len(dataset)/16)*60
    write(destination,dict(PASS=True,time=now(),physical_batch=16,effective_batch=16,
       optimizer_updates=8,discarded=True,seed=42,source_local_rows=list(range(16)),
       source_rows=batch['index'].tolist(),source_ids=list(batch['id']),
       losses=losses,step_seconds=durations,steady_mean_step_seconds=float(np.mean(durations[2:])),
       compute_only_60epoch_seconds=estimate,estimate_excludes='PNG loading/hash/decode, validation, checkpoint writes, downstream refiner fits; no completion guarantee',
       peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=gpu(),pretrained=bound(PRETRAINED),
       code=[bound(HERE/n) for n in ('model.py','input_data.py','baseline_train.py')]))

@torch.no_grad()
def validate(model,loader):
    model.eval();numerator=0.;frames=0;valid=0;counts={}
    for batch in loader:
        pred=model(batch['image'].cuda(non_blocking=True))
        value=masked_mse(pred,batch['heatmaps'].cuda(non_blocking=True),batch['target_valid'].cuda(non_blocking=True))
        assert torch.isfinite(value)
        n=len(batch['image']);numerator+=float(value)*n;frames+=n;valid+=int(batch['target_valid'].sum())
        for key, value in batch['counts'].items():counts[key]=counts.get(key,0)+int(value.sum())
    model.train();return dict(frames=frames,loss=numerator/frames,supervised_channels=valid,counts=counts)

def train():
    protocol=lock();status=gpu();directory=RAW/'baseline';directory.mkdir(parents=True,exist_ok=True)
    destination=directory/'last.pt';complete=DOC/'BASELINE_TRAINING_COMPLETE.json'
    assert not complete.exists(),'Baseline already complete'
    model=initialize();o=protocol['optimizer'];optimizer=torch.optim.Adam(model.parameters(),lr=o['lr'],weight_decay=o['weight_decay'],betas=tuple(o['betas']))
    train_data=SourceDataset(SOURCE,partition='train');val_data=SourceDataset(SOURCE,partition='calibration')
    assert len(train_data)==55980 and len(val_data)==1004
    val_loader=DataLoader(val_data,batch_size=16,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True,
        worker_init_fn=worker_init_fn,generator=torch.Generator().manual_seed(500042))
    begin_epoch=1;begin_batch=0;global_step=0;history=[];curves=[];elapsed_before=0.;resumes=[];images_seen=0
    training_counts={};epoch_loss_numerator=0.;epoch_frames=0
    if destination.exists():
        state=torch.load(destination,map_location='cpu',weights_only=False)
        assert state['protocol_sha256']==sha(DOC/'BASELINE_PROTOCOL.json')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        begin_epoch=state['next_epoch'];begin_batch=state['next_batch'];global_step=state['step'];images_seen=state['images_seen']
        history=state['history'];curves=state['curves'];elapsed_before=state['elapsed_seconds']
        resumes=state['resumes']+[dict(time=now(),from_step=global_step,epoch=begin_epoch,batch=begin_batch)]
        training_counts=state['training_counts'];epoch_loss_numerator=state['epoch_loss_numerator'];epoch_frames=state['epoch_frames']
    begin=time.perf_counter()
    def checkpoint(next_epoch,next_batch,finished=False):
        value=dict(complete=finished,model=model.state_dict(),optimizer=optimizer.state_dict(),
            next_epoch=next_epoch,next_batch=next_batch,step=global_step,images_seen=images_seen,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            protocol_sha256=sha(DOC/'BASELINE_PROTOCOL.json'),history=history,curves=curves,
            training_counts=training_counts,epoch_loss_numerator=epoch_loss_numerator,epoch_frames=epoch_frames,
            resumes=resumes,elapsed_seconds=elapsed_before+time.perf_counter()-begin)
        temporary=destination.with_name('last.pending.pt');torch.save(value,temporary);temporary.replace(destination)
        return value
    for epoch in range(begin_epoch,61):
        order=np.random.default_rng(42+epoch).permutation(len(train_data)).tolist()
        offset=begin_batch if epoch==begin_epoch else 0
        batches=[order[i:i+16] for i in range(0,len(order),16)]
        loader=DataLoader(train_data,batch_sampler=batches[offset:],num_workers=4,pin_memory=True,
            worker_init_fn=worker_init_fn,generator=torch.Generator().manual_seed(100042+epoch))
        lr=o['lr']*(o['drop_factor']**sum(epoch>=m for m in o['drop_epochs']))
        for group in optimizer.param_groups:group['lr']=lr
        model.train()
        for batch_index,batch in enumerate(loader,offset):
            value=one_update(model,optimizer,batch,protocol['physical_batch']);global_step+=1;images_seen+=len(batch['image'])
            epoch_loss_numerator+=value*len(batch['image']);epoch_frames+=len(batch['image'])
            for key, count in batch['counts'].items():training_counts[key]=training_counts.get(key,0)+int(count.sum())
            row=dict(step=global_step,epoch=epoch,batch=batch_index+1,loss=value,lr=lr,images_seen=images_seen)
            if global_step==1 or global_step%100==0:
                row.update(elapsed_seconds=elapsed_before+time.perf_counter()-begin,gpu=gpu());history.append(row)
                write(RAW/'BASELINE_PROGRESS.json',row,freeze=False)
                print('R18_BASELINE',epoch,batch_index+1,len(batches),'step',global_step,'loss',round(value,6),'sec',round(row['elapsed_seconds'],1),flush=True)
            if global_step%500==0 or STOP:checkpoint(epoch,batch_index+1)
            if STOP:raise SystemExit('Graceful stop: model/optimizer/RNG/batch position saved')
        assert epoch_frames==len(train_data)
        metrics=validate(model,val_loader);curve=dict(epoch=epoch,step=global_step,calibration=metrics,lr=lr,
            training_loss=epoch_loss_numerator/epoch_frames,training_frames=epoch_frames)
        curves.append(curve);write(RAW/'BASELINE_CURVE.json',curves,freeze=False)
        epoch_loss_numerator=0.;epoch_frames=0
        state=checkpoint(epoch+1,0,finished=epoch==60)
        print('R18_EPOCH_COMPLETE',json.dumps(curve),flush=True)
    assert images_seen==55980*60 and global_step==math.ceil(55980/16)*60
    path=RAW/'baseline_final.pt'
    if path.exists():
        frozen=torch.load(path,map_location='cpu',weights_only=False)
        assert frozen['epoch']==60 and frozen['step']==global_step and frozen['protocol_sha256']==sha(DOC/'BASELINE_PROTOCOL.json')
        assert frozen['architecture']=='SimpleBaselineResNet18' and frozen['num_keypoints']==9
        assert frozen['model_state_dict'].keys()==model.state_dict().keys()
        for key,value in model.state_dict().items():assert torch.equal(frozen['model_state_dict'][key],value.cpu()),key
    else:
        temp=path.with_name('baseline_final.pending.pt')
        with temp.open('wb') as f:torch.save(dict(model_state_dict=model.state_dict(),epoch=60,step=global_step,
            architecture='SimpleBaselineResNet18',num_keypoints=9,protocol_sha256=sha(DOC/'BASELINE_PROTOCOL.json')),f)
        temp.replace(path)
    write(complete,dict(complete=True,epochs=60,updates=global_step,source_exposures=images_seen,
       train_images=55980,calibration_images=1004,final_checkpoint=bound(path),resume_checkpoint=bound(destination),
       protocol=bound(DOC/'BASELINE_PROTOCOL.json'),curves=curves,resumes=resumes,real_training=0,
       training_counts=training_counts,training_counts_scope='All60 epochs; exposures, not unique-image counts',
       final_checkpoint_only=True,elapsed_seconds=state['elapsed_seconds'],seed=42,
       parameters=sum(p.numel() for p in model.parameters()),source=bound(SOURCE),after=gpu()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','train']);args=p.parse_args()
    torch.set_num_threads(1);cv2.setNumThreads(1)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    print('GPU_START',gpu(),flush=True)
    smoke() if args.phase=='smoke' else train()
