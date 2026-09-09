from dataclasses import fields
import torch
from pointline_v4.model import Observation
from pointline_v4.geometry import EDGES

torch.set_num_threads(1)


def make_observation(seed=3,batch=1,candidates=4,requires_grad=False):
    g=torch.Generator().manual_seed(seed)
    pts=torch.tensor([[12.,12.],[26.,13.],[26.,22.],[12.,22.],[8.,8.],[22.,9.],[22.,18.],[8.,18.],[17.,15.]])
    base=pts[None].repeat(batch,1,1)
    layouts=base[:,None].repeat(1,candidates,1,1)
    for j in range(1,candidates):layouts[:,j,:8]+=torch.tensor([float(j),-.5*j])
    h=[]
    for u,v in EDGES:
        a,b=pts[u],pts[v];d=b-a;n=torch.tensor([-d[1],d[0]]);n=n/n.norm()
        line=torch.cat((n,-(n*a).sum().view(1)))
        h.append(torch.stack([line+torch.tensor([0,0,i*.3]) for i in range(4)]))
    features=(torch.randn(batch,2,40,40,generator=g,requires_grad=requires_grad),
              torch.randn(batch,3,20,20,generator=g,requires_grad=requires_grad))
    return Observation(features=features,raw_to_feature=(torch.tensor([[[1.,0,0],[0,1.,0]]]).repeat(batch,1,1),
              torch.tensor([[[.5,0,0],[0,.5,0]]]).repeat(batch,1,1)),
              content_valid=(torch.ones(batch,1,40,40,dtype=torch.bool),torch.ones(batch,1,20,20,dtype=torch.bool)),
              baseline=base,point_valid=torch.ones(batch,9,dtype=torch.bool),point_conf=torch.linspace(.2,.9,9)[None].repeat(batch,1),
              layouts=layouts,candidate_valid=torch.ones(batch,candidates,dtype=torch.bool),
              line_h=torch.stack(h)[None].repeat(batch,1,1,1),line_logits=torch.randn(batch,12,4,generator=g),
              line_valid=torch.ones(batch,12,4,dtype=torch.bool),diagonal=torch.full((batch,),56.5685))


def clone_observation(o):
    return Observation(**{f.name:tuple(x.clone() for x in getattr(o,f.name)) if isinstance(getattr(o,f.name),tuple)
                          else getattr(o,f.name).clone() for f in fields(o)})
