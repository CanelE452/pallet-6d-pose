"""Small generated-input checks. No source/current/held/official model reads."""
import hashlib
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch

H=Path(__file__).absolute().parent

def load(name):
    path=H/(name+'.py')
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def run():
    out=H/'INVENTED_V2_CHECKS.json'
    assert not out.exists()
    s=load('spatial_head_v2'); t=load('train_spatial')
    torch.set_num_threads(1);torch.manual_seed(17)
    a=s.SpatialPointHead();torch.manual_seed(17);b=s.SpatialPointHead()
    assert t.state_sha(a.state_dict())==t.state_sha(b.state_dict())
    assert not any('offset' in k for k in a.state_dict())
    x=torch.randn(4,256,384).half();support=torch.ones(4,16,16,dtype=torch.bool);support[3]=False
    native=torch.arange(4,dtype=torch.int64)
    o=a(x,support,native)
    assert o['probability'].dtype==torch.float32 and o['uv_subpixel'].shape==(4,2)
    assert torch.allclose(o['uv_subpixel'][:3],torch.full((3,2),.5))
    assert o['probability'][3].count_nonzero()==0 and s.semantic_visibility(o)[3]==-1
    q=torch.zeros(4,2);L=torch.ones(4)*96;r=torch.tensor([[0.,0.],[0.,0.],[60.,0.],[0.,0.]])
    available=torch.tensor([True,True,True,False]);cue=torch.tensor([True,False,True,True])
    known=torch.ones(4,dtype=torch.bool)
    loc,uv,state=t.fixed_targets(s,available,known,cue,known,known,r,q,L)
    assert loc.tolist()==[True,False,False,False]
    assert state.tolist()==[0,-1,2,-1]
    # Weak-cue D is still a semantic D and receives VIS supervision.
    vis=torch.tensor([0,0,1,-1]);loss,parts=s.losses(o,uv,loc,vis,state,'VIS')
    loss.backward();assert a.visibility.weight.grad is not None and a.heatmap.weight.grad is None
    a.zero_grad(set_to_none=True);o=a(x,support,native)
    loss,_=s.losses(o,uv,loc,vis,state,'LOC');loss.backward()
    assert a.visibility.weight.grad is None and a.heatmap.weight.grad is not None
    # One generated optimizer step verifies graph/update mechanics only.
    opt=torch.optim.AdamW(a.parameters(),lr=1e-3,weight_decay=1e-4);opt.step()
    assert all(torch.isfinite(p).all() for p in a.parameters())
    # A single subpixel heatmap and an ambiguous two-peak heatmap.
    yy,xx=torch.meshgrid(torch.arange(16),torch.arange(16),indexing='ij')
    p=torch.exp(-((xx+.5-8.2)**2+(yy+.5-6.7)**2)/(2*.7**2));p/=p.sum()
    z={**o,'probability':p[None].expand(4,-1,-1).clone()}
    d=s.diagnostics(z)
    assert torch.max(torch.abs(d['heatmap_subpixel_uv'][0]-torch.tensor([8.2,6.7])/16))<1e-4
    bimodal=torch.zeros(16,16);bimodal[4,4]=.5;bimodal[11,11]=.5
    z['probability']=bimodal[None].expand(4,-1,-1)
    assert torch.allclose(s.diagnostics(z)['nonlocal_peak_ratio'],torch.ones(4))
    # GT-free rejection preserves original q exactly, irrespective of VIS logit.
    predicted=s.image_points(o,q,L,torch.zeros(4,dtype=torch.bool));assert torch.equal(predicted,q)
    K=torch.tensor([[[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]]])
    T=torch.eye(4)[None];T[:,2,3]=5
    dims=torch.ones(1,3)*2
    side,ok=s.geometric_crop_side(T,K,dims,1.)
    assert ok.item() and abs(side.item()-50)<1e-5
    simple,valid=s.simple_crop_side(torch.tensor([24.]),2.)
    selected,valid,fallback=s.choose_crop_side(simple,valid,torch.tensor([float('nan')]),torch.tensor([False]),'initial_geometry')
    assert selected.item()==48 and fallback.item() and valid.item()
    # Exact RAW96 half-open footprint, not clamped out-of-range GT.
    mask,_,_=s.fixed_loc_mask(torch.tensor([True]),torch.tensor([True]),torch.tensor([True]),torch.tensor([[48.,0.]]),torch.zeros(1,2),torch.tensor([96.]))
    assert not mask.item()
    result={'completed':True,'PASS':True,'groups':['shared_seed17_initialization_and_no_offset',
        'FP16_to_FP32_shape_and_unavailable_support','fixed_LOC_mask_weak_D_and_outside_unknown',
        'separate_VIS_LOC_gradients','generated_optimizer_step_finite','heatmap_subpixel_and_two_peak',
        'q0_fallback_independent_VIS','initial_geometry_and_simple_fallback','half_open_crop_boundary'],
        'head_parameters':sum(p.numel() for p in a.parameters()),'invented_optimizer_updates':1,
        'actual_source_current_held_reads':0,'official_model_reads':0,
        'code':{name:{'sha256':hashlib.sha256((H/name).read_bytes()).hexdigest(),'size_bytes':len((H/name).read_bytes())}
                for name in ['spatial_head_v2.py','train_spatial.py','check_invented_v2.py']}}
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))

if __name__=='__main__':
    run()
