"""Explicit execution phases; no automatic tuning or real-based selection."""
import argparse
from collections import defaultdict
import time
import cv2
import numpy as np
import torch
from common import *
from dope_adapter import FrozenDopeAdapter,selfcheck,recipe
from data import load_image,target,iou,cache_source
from train import CLASSES,LOSSES,forward,train

def smoke(adapter):
    destination=DOC/'GPU_SMOKE.json'
    if destination.exists():
        r=read(destination)
        for b in r['bindings']:assert sha(ROOT/b['path'])==b['sha256']
        assert r['PASS'];return
    selfcheck();records=read(SOURCE)['records'];groups=defaultdict(list)
    for r in records:
        key=tuple(r['prepared_shape_hw'])
        if r['partition']=='train' and len(groups[key])<4:groups[key].append(r)
    assert len(groups)==4 and all(len(v)==4 for v in groups.values())
    selected=[];predictions=[];features=[];differences=[];start=time.perf_counter()
    for shape,rs in sorted(groups.items()):
        images=[load_image(r) for r in rs]
        preds=adapter.infer_batch(images,source_pre_padded=True,return_features=True)
        prepared=[adapter.prepare(im,True) for im in images]
        x=torch.stack([p['tensor'] for p in prepared]).to(adapter.device)
        a,b=adapter.features_only(x)
        for j,p in enumerate(preds):
            for f,g in zip(p['features'],(a[j],b[j])):
                error=float((f-g).abs().max());differences.append(error)
                torch.testing.assert_close(f,g,rtol=1e-5,atol=1e-5)
            features.append(tuple(f.detach().to(torch.float16) for f in p.pop('features')))
        selected.extend(rs);predictions.extend(preds)
        del x,a,b,preds
    batch={}
    for k in ('points','boxes','point_valid','input_shape','gt_points','gt_valid'):batch[k]=[]
    matched=[]
    for r,p in zip(selected,predictions):
        aff=np.array(p['affine_input_to_net']);gt,gv,bb=target(r,np.diag(aff)[:2],aff[:2,2])
        box=np.full(4,np.nan) if p['bbox_net'] is None else np.array(p['bbox_net'])
        match=iou(box,bb)>=.5;matched.append(match)
        values=dict(points=p['points_net'],boxes=box,point_valid=p['valid'],input_shape=p['input_shape'][-2:],gt_points=gt,gt_valid=gv&match)
        for k,v in values.items():batch[k].append(v)
    for k,v in batch.items():batch[k]=torch.from_numpy(np.array(v)).to(adapter.device)
    for level,key in enumerate(('p3','p4')):
        h=max(fs[level].shape[-2] for fs in features);w=max(fs[level].shape[-1] for fs in features)
        out=torch.zeros((len(features),features[0][level].shape[0],h,w),dtype=torch.float16,device=adapter.device)
        for j,fs in enumerate(features):out[j,:,:fs[level].shape[-2],:fs[level].shape[-1]]=fs[level]
        batch[key]=out
    assert any(matched),'No matched rows in fixed smoke group'
    metrics={}
    for arm in ('P','D'):
        torch.manual_seed(20261001);torch.cuda.manual_seed_all(20261001)
        model=CLASSES[arm](**CONFIG).to(adapter.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        rows=[]
        for step in (1,2):
            optimizer.zero_grad(set_to_none=True);out=forward(model,batch)
            assert torch.equal(torch.isnan(out['points']),torch.isnan(batch['points']))
            torch.testing.assert_close(out['points'],batch['points'].float(),equal_nan=True,rtol=0,atol=0)
            value=LOSSES[arm](out,batch['gt_points'],batch['gt_valid']);assert torch.isfinite(value)
            value.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            assert model.adapt3[0].weight.grad.norm()>0 and model.adapt4[0].weight.grad.norm()>0
            optimizer.step();assert all(torch.isfinite(p).all() for p in model.parameters())
            rows.append(dict(step=step,loss=float(value),gradient_norm=float(norm)))
        metrics[arm]=rows;del model,optimizer,out,value
    assert all(p.grad is None and not p.requires_grad for p in adapter.network.parameters())
    write(destination,dict(PASS=True,time=now(),source_rows=[r['index'] for r in selected],
        source_shapes=[list(k) for k in sorted(groups)],matched_rows=sum(matched),feature_full_prefix_max_abs=max(differences),
        optimizer_updates={'P':2,'D':2},heads_discarded=True,seed=20261001,
        contract_only_no_performance_selection=True,metrics=metrics,elapsed_seconds=time.perf_counter()-start,gpu=gpu(),
        adapter_recipe=recipe(),bindings=[bound(HERE/f) for f in ('dope_adapter.py','refiner.py','data.py','run.py')]))
    print('DOPE_GPU_SMOKE_PASS',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['smoke','cache','train','validation','select'])
    args=parser.parse_args();torch.set_num_threads(1);cv2.setNumThreads(1)
    if args.phase=='select':
        from selection import select
        select();return
    status=gpu();print('GPU_START',status,flush=True)
    adapter=FrozenDopeAdapter(device='cuda')
    if args.phase=='smoke':smoke(adapter)
    elif args.phase=='cache':lock_protocol();cache_source(adapter)
    elif args.phase=='train':train(adapter)
    elif args.phase=='validation':
        from selection import validation
        validation(adapter)

if __name__=='__main__':main()
