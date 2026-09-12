"""Small real optimizer run on GENERATED wiring fixtures, not a pallet experiment."""
from __future__ import annotations
import argparse,copy,time,platform
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from .fixtures import generated_images
from .model import ModelConfig,PointLinePoseModel,ToyRGBBackbone,initialize_matched
from .adapters import MultiScaleROI
from .contracts import Supervision
from .objective import compute_loss
from .io import write_json,tensor_hash,source_hashes


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True)
    ap.add_argument('--steps',type=int,default=8);ap.add_argument('--device',default='cpu');args=ap.parse_args()
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Preserve old outputs; choose a new directory')
    if not 1<=args.steps<=100:raise ValueError('Wiring demo only: 1..100 optimizer steps')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(11)
    batch=generated_images(6);cfg=ModelConfig(in_channels=16,channels=16,grid=12,theta_bins=12,rho_bins=25,starts=4,solver_iterations=3)
    heads={arm:PointLinePoseModel(ModelConfig(**{**asdict(cfg),'arm':arm})) for arm in ('point','direct','hough')}
    common=initialize_matched(heads)
    baseback=ToyRGBBackbone(16,24);baseadapter=MultiScaleROI((16,),16,12)
    report={'scope':'GENERATED_WIRING_ONLY_NOT_PALLET_ACCURACY','input':'procedural cuboid RGB fixtures',
            'roi_origin':'GT-derived diagnostic only; forbidden for real inference',
            'steps_per_arm':args.steps,'device':args.device,'torch':torch.__version__,'python':platform.python_version(),
            'source_sha256':source_hashes(),'arms':{}}
    for arm,head in heads.items():
        backbone=copy.deepcopy(baseback).to(args.device);adapter=copy.deepcopy(baseadapter).to(args.device);head=head.to(args.device)
        params=list(backbone.parameters())+list(adapter.parameters())+list(head.parameters())
        before={'backbone':tensor_hash(backbone.state_dict()),'adapter':tensor_hash(adapter.state_dict()),'head':tensor_hash(head.state_dict())}
        opt=torch.optim.AdamW(params,lr=2e-4);history=[];start=time.perf_counter();maxgrad={}
        for step in range(args.steps):
            ix=torch.tensor([(2*step)%6,(2*step+1)%6])
            rgb=batch['rgb'][ix].to(args.device);f=backbone(rgb)
            A=torch.eye(3,device=args.device).repeat(len(ix),1,1);A[:,0,0]=.5;A[:,1,1]=.5
            obs=adapter((f,),(A,),(torch.ones_like(f[:,:1]),),batch['box'][ix].to(args.device),batch['K'][ix].to(args.device),
                        batch['dims'][ix].to(args.device),batch['image_hw'][ix].to(args.device))
            gt=Supervision(batch['points'][ix].to(args.device),torch.ones(len(ix),9,dtype=torch.bool,device=args.device),
                           batch['R'][ix].to(args.device),batch['t'][ix].to(args.device),[batch['specs'][int(i)] for i in ix])
            opt.zero_grad(set_to_none=True);prediction=head(obs)
            loss,detail=compute_loss(prediction,obs,gt,head.lattice.lines);loss.backward()
            for name,module in [('backbone',backbone),('roi_adapter',adapter),('point_head',head.point_map),('seed_head',head.seed_head),('line_head',head.line_head)]:
                if module is None:continue
                norm=sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None)**.5
                maxgrad[name]=max(maxgrad.get(name,0.),norm)
            norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True);opt.step()
            history.append({'step':step+1,'indices':ix.tolist(),'loss':float(loss.detach()),'gradnorm':float(norm),**detail})
        after={'backbone':tensor_hash(backbone.state_dict()),'adapter':tensor_hash(adapter.state_dict()),'head':tensor_hash(head.state_dict())}
        report['arms'][arm]={'actual_optimizer_steps':args.steps,'history':history,'before':before,'after':after,
                              'all_groups_changed':all(before[k]!=after[k] for k in before),'max_gradient_norms':maxgrad,
                              'head_parameters':sum(p.numel() for p in head.parameters()),'elapsed_seconds':time.perf_counter()-start}
    report['success']=all(v['all_groups_changed'] and all(x>0 for x in v['max_gradient_norms'].values()) for v in report['arms'].values())
    report['scientific_success']=None;write_json(out/'DEMO.json',report)
    print('actual optimizer steps:',len(heads)*args.steps,'wiring checks:',report['success'])
    if not report['success']:raise RuntimeError('Wiring check failed')
if __name__=='__main__':main()
