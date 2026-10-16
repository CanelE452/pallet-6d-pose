"""No optimizer updates: isolate residual numeric variation under strict flags."""
from context34 import *
from data32 import training,plan
from model33 import Correspondence,objective
from cli_train import state_sha
import torch.nn.functional as F

def main():
    deterministic();O,S,g,v,phys,virt,sup,ids,n=training(1);ix=torch.tensor(plan(ids,n,1)[0]);O={k:x[ix].cuda() for k,x in O.items()};S={k:x[ix].cuda() for k,x in S.items()};g=g[ix].cuda();v=v[ix].cuda();phys=phys[ix].cuda();sup=sup[ix].cuda();model=Correspondence(1).cuda();model.load_state_dict(torch.load(RUN/'models/line_only_r0_seed1/final.pt',weights_only=True,map_location='cpu')['state_dict']);outputs=[];grads=[];flags=[]
    for repeat in range(3):
        model.zero_grad(set_to_none=True);out=model(O,S,'line_only');loss,terms=objective(out,g,v,phys,sup,'visible_hough');loss.backward();outputs.append({k:x.detach().cpu().clone() for k,x in out.items()});grads.append({k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None});flags.append(dict(enabled=torch.are_deterministic_algorithms_enabled(),cudnn_deterministic=torch.backends.cudnn.deterministic))
    def diffs(arr):return {k:max(float((arr[0][k]-x[k]).abs().max()) for x in arr[1:]) for k in arr[0] if any(not torch.equal(arr[0][k],x[k]) for x in arr[1:])}
    operator=[]
    def check(name,make,fn):
        x=make();weight=None;values=[]
        for _ in range(3):
            xx=x.detach().clone().requires_grad_(True);y=fn(xx)
            if weight is None:weight=torch.randn_like(y)
            grad=torch.autograd.grad((y*weight).sum(),xx)[0];values.append((y.detach(),grad.detach()))
        operator.append(dict(name=name,forward_max=max(float((values[0][0]-z[0]).abs().max()) for z in values[1:]),gradient_max=max(float((values[0][1]-z[1]).abs().max()) for z in values[1:])))
    grid=torch.rand(16,8,9,2,device='cuda')*2-1
    check('grid_sample input gradient',lambda:torch.randn(16,32,48,48,device='cuda'),lambda x:F.grid_sample(x,grid,align_corners=True))
    check('bilinear upsample',lambda:torch.randn(16,32,24,24,device='cuda'),lambda x:F.interpolate(x,size=(48,48),mode='bilinear',align_corners=True))
    check('bilinear downsample',lambda:torch.randn(16,32,48,48,device='cuda'),lambda x:F.interpolate(x,size=(24,24),mode='bilinear',align_corners=True))
    check('fixed avg_pool',lambda:torch.randn(16,32,48,48,device='cuda'),lambda x:F.avg_pool2d(x,12))
    write(RUN/'NUMERIC_REPLAY.json',dict(complete=True,optimizer_updates=0,flags=flags,repeated_forward_differences=diffs(outputs),repeated_gradient_differences=diffs(grads),operators=operator,limits='Repeated fixed trained state and one actual batch in the same process; does not isolate every cross-process source of variation.'))
    print('NUMERIC REPLAY',diffs(outputs),diffs(grads),operator,flush=True)

if __name__=='__main__':main()
