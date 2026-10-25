import copy,json
from pathlib import Path
import torch
from torch.nn import functional as F
from relational_query_head import *
torch.set_num_threads(1)
with torch.random.fork_rng(devices=[]):
 torch.manual_seed(237)
 m=RelationalQueryHead();assert sum(p.numel() for p in m.base.parameters())==76098
 assert sum(p.numel() for p in m.relational.parameters())==9956
 assert sum(p.numel() for p in m.parameters())==86054
 n=3;x=torch.randn(n,1186);nodes=F.normalize(torch.randn(n,8,384),dim=-1);context=F.normalize(torch.randn(n,384),dim=-1)
 idx=torch.tensor([0,3,7],dtype=torch.int64);available=torch.ones(n,8,dtype=torch.bool);support=available.clone();cs=torch.tensor([1,1,0],dtype=torch.bool);xy=torch.randn(n,8,2).clamp(-8,8)
 x[:,:384]=nodes[torch.arange(n),idx];available[2,7]=False
 a=verify_active_node_alignment(x,nodes,idx,available,support);assert a['checked']==2 and a['max_abs']==0
 bad=nodes.clone();bad[0,0,0]+=1
 try:verify_active_node_alignment(x,bad,idx,available,support)
 except ValueError:pass
 else:raise AssertionError('mismatch accepted')
 # Verify nontrivial existing base preserved exactly under zero residual.
 with torch.no_grad():m.base.head[-1].weight.normal_();m.base.head[-1].bias.normal_()
 z,info=m(x,nodes,context,idx,available,support,cs,xy);assert torch.equal(z,m.base(x))
 assert (info['attention_other7'][torch.arange(n),idx]==0).all()
 # Explicit invented nonzero residual output fixture, not a learned/source state.
 with torch.no_grad():m.relational.readout[-1].weight.normal_();m.relational.readout[-1].bias.zero_()
 z,_=m(x,nodes,context,idx,available,support,cs,xy)
 altered=nodes.clone();altered[0,1]*=-1
 zo,_=m(x,altered,context,idx,available,support,cs,xy);assert not torch.equal(z[0],zo[0])
 xa=x.clone();xa[0,:384]*=-1;za,_=m(xa,nodes,context,idx,available,support,cs,xy);assert not torch.equal(z[0],za[0])
 av=available.clone();av[:,2]=False
 z0,_=m(x,nodes,context,idx,av,support,cs,xy);noise=nodes.clone();noise[:,2]=torch.randn(3,384)*100
 z1,_=m(x,noise,context,idx,av,support,cs,xy);assert torch.equal(z0,z1)
 only=available.clone();only[:]=False;only[torch.arange(n),idx]=available[torch.arange(n),idx]
 zz,ii=m(x,nodes,context,idx,only,support,cs,xy);assert torch.isfinite(zz).all() and not ii['attention_other7'].any()
 pred=fixed_family_decode(zz,torch.tensor([0,1,0]),ii['original_active_available']);assert pred[2]==-1
 m.zero_grad();z.sum().backward();assert m.relational.project.weight.grad is not None and m.relational.project.weight.grad.abs().sum()>0
 assert m.base.head[0].weight.grad is not None
 try:m(x,nodes,context,idx,available,support,cs,xy,target_id=torch.tensor([1,2,3]))
 except TypeError:pass
 else:raise AssertionError('GT keyword accepted')
result={'completed':True,'PASS':True,'actual_source_features_models_weights_read':0,'invented_CPU_only':True,'parameters':{'base':76098,'relational_residual':9956,'total':86054},'tests':['strict base shape/count and zero-init residual exact equivalence','active node alignment and mismatch rejection','active and other image evidence affect invented nonzero branch','input-unavailable othernode masked; all-other-missing finite/zero weights','original input U preserved by fixed0 decoder','head gradients and GT keyword exclusion'],'actual_feature_or_training_run':0}
(Path(__file__).parent/'INVENTED_CPU_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
