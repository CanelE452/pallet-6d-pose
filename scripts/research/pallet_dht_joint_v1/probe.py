"""Synthetic tensor resource/parity probe; this is not an accuracy evaluation."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from scripts.research.pallet_dht_joint_v1.integration import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1)
    device = 'cuda'
    base = build_model('point_only').to(device).eval()
    joint = build_model('hough_features').to(device).eval()
    parity = []
    with torch.no_grad():
        for h, w in [(640, 640), (480, 640), (384, 640)]:
            x = torch.rand(1, 3, h, w, device=device)
            a, b = base(x)[0], joint(x)[0]
            parity.append(dict(input_hw=[h, w], exact=torch.equal(a, b),
                               max_abs_delta=float((a-b).abs().max())))
    del base, joint, x, a, b
    torch.cuda.empty_cache()
    rows = []
    for arm in ['point_only', 'hough_features']:
        for batch in [8, 16]:
            model = build_model(arm).to(device).train()
            model.args.epochs = 2
            image = torch.rand(batch, 3, 640, 640, device=device)
            corners = torch.tensor([[.2,.3,2],[.6,.3,2],[.6,.7,2],[.2,.7,2],
                                    [.35,.2,2],[.75,.2,2],[.75,.6,2],[.35,.6,2],
                                    [.475,.45,2]], device=device)
            targets = dict(img=image, batch_idx=torch.arange(batch,device=device).float(),
                           cls=torch.zeros(batch,1,device=device),
                           bboxes=torch.tensor([.475,.45,.55,.5],device=device).repeat(batch,1),
                           keypoints=corners[None].repeat(batch,1,1))
            optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
            scaler = torch.cuda.amp.GradScaler()
            torch.cuda.reset_peak_memory_stats()
            elapsed=[]
            for step in range(12):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); started=time.perf_counter()
                with torch.autocast('cuda'):
                    # Stock loss may alter batch tensors; isolate each step.
                    loss, items = model.loss({k:v.clone() for k,v in targets.items()})
                    total = loss.sum()
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                scaler.step(optimizer); scaler.update()
                torch.cuda.synchronize()
                if step>=3: elapsed.append(time.perf_counter()-started)
            rows.append(dict(arm=arm,batch=batch,mean_step_seconds=float(np.mean(elapsed)),
                             peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                             last_loss_items=items.detach().cpu().tolist(),
                             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
            del model,optimizer,image,targets,loss,items,total,scaler
            torch.cuda.empty_cache()
    result=dict(PASS=all(x['exact'] for x in parity),accuracy_evidence=False,
                scope='Random tensor plumbing and hardware resource measurement only',
                parity=parity,measurements=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
