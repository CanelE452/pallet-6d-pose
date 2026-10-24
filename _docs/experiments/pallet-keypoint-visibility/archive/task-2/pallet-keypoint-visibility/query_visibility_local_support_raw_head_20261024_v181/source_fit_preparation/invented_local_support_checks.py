"""Invented CPU checks, no input files, SAM, weights, or actual labels."""
import numpy as np
import torch
from torch.nn import functional as F


def run_checks(t):
    torch.set_num_threads(1)
    checks=[]
    def passed(name,ok):
        t.require(ok,name);checks.append(dict(name=name,PASS=True))
    def rejects(thunk):
        try:thunk()
        except (ValueError,AssertionError):return True
        return False
    torch.manual_seed(1810);model=t.head.Raw96SupportHead()
    passed('Frozen115157 parameters with six GN and no BN/dropout',sum(p.numel() for p in model.parameters())==115157 and sum(isinstance(m,torch.nn.GroupNorm) for m in model.modules())==6 and not any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout)) for m in model.modules()))
    rgb=torch.arange(2*96*96*3).remainder(256).to(torch.uint8).reshape(2,96,96,3)
    valid=torch.ones((2,96,96),dtype=torch.bool);valid[1,:7,:11]=False
    original=rgb.clone();original_valid=valid.clone();raw=t.head.assemble_local_input(rgb,valid,use_sam=False)
    passed('Original RGB and valid unmodified; only divide255 normalization',torch.equal(rgb,original) and torch.equal(valid,original_valid) and torch.equal(raw[:,:3],rgb.permute(0,3,1,2).float()/255) and torch.equal(raw[:,3],valid.float()))
    grid=t.head.relative_xy_grid()
    passed('Fixed XY endpoints95over96 and center47point5, no qL input',abs(float(grid[0,0,0])+95/96)<1e-7 and abs(float(grid[0,0,-1])-95/96)<1e-7 and abs(float(grid[0,0,47])+1/96)<1e-7 and abs(float(grid[0,0,48])-1/96)<1e-7 and torch.equal(grid[0],grid[1].T))
    sm=torch.ones((2,96,96),dtype=torch.bool);sa=torch.zeros_like(sm)
    missing=t.head.assemble_local_input(rgb,valid,use_sam=True,sam_mask96=sm,sam_available=sa)
    passed('Missing SAM leaves exact RAW8 input while original valid unchanged',torch.equal(raw,missing))
    sa[0,:48,:]=True;sam=t.head.assemble_local_input(rgb,valid,use_sam=True,sam_mask96=sm,sam_available=sa)
    passed('SAM only occupies dedicated pixelwise mask and availability channels',torch.equal(sam[:,:6],raw[:,:6]) and torch.equal(sam[:,6],sa.float()) and torch.equal(sam[:,7],sa.float()) and raw.dtype==torch.float32 and raw.is_contiguous())
    torch.manual_seed(1810);other=t.head.Raw96SupportHead()
    a=model(raw);b=other(missing)
    passed('Same seed1810 initial states and SAMmissing outputs byteexact',t.object_sha(model.state_dict())==t.object_sha(other.state_dict()) and all(torch.equal(a[k],b[k]) for k in a))
    alone=model(raw[:1]);together=model(raw)
    passed('GN independent of batch companions',all(torch.allclose(alone[k],together[k][:1],atol=1e-6,rtol=1e-6) for k in alone))
    passed('Separate support2 and owner3 logits',a['support_logits'].shape==(2,2) and a['owner_logits'].shape==(2,3))
    with torch.no_grad():
        previous=model.support_head.bias.clone();model.support_head.bias.fill_(10);both=model(raw)['support_logits'];model.support_head.bias.copy_(previous)
    passed('Two support events nonexclusive, both may be positive',bool((both>0).all()))
    passed('Input schema and SAM availability fail closed',all([rejects(lambda:t.head.assemble_local_input(rgb.float(),valid,use_sam=False)),rejects(lambda:t.head.assemble_local_input(rgb,valid,use_sam=True)),rejects(lambda:t.head.assemble_local_input(rgb,valid,use_sam=False,sam_mask96=sm)),rejects(lambda:model(torch.zeros((1,7,96,96)))),rejects(lambda:model(torch.full((1,8,96,96),float('nan'))))]))
    # Test the actual five-loss helper directly, with independent invented logits.
    sl=torch.zeros((192,2),requires_grad=True);ol=torch.zeros((192,3),requires_grad=True)
    total,components=t.five_losses(dict(support_logits=sl,owner_logits=ol));total.backward()
    binary=torch.tensor([1.]*16+[0.]*16);expected=torch.zeros_like(sl);expected[:32,0]=(.5-binary)/(5*32);expected[32:64,1]=(.5-binary)/(5*32)
    owner_expected=torch.zeros_like(ol)
    for start,size,classes in [(64,48,3),(112,48,3),(160,32,2)]:
        labels=torch.arange(classes).repeat_interleave(16);owner_expected[start:start+size]=(torch.full((size,3),1/3)-F.one_hot(labels,3))/(5*size)
    passed('Five equal weights and exact per-head/domain gradient masks',torch.equal(total,torch.stack(components).mean()) and torch.equal(sl.grad,expected) and torch.allclose(ol.grad,owner_expected,atol=1e-9,rtol=0) and not ol.grad[:64].any())
    available=np.array([1,1,1,0],bool)
    soft=np.array([[.2,.1,0],[.1,.2,.7],[0,0,0],[np.nan]*3],np.float64)
    target=dict(support_target=np.array([[1,0],[0,1],[-1,-1],[-1,-1]],np.int8),owner_soft=soft,owner_target=np.full(4,-1,np.int8),owner_ambiguous_mass=np.array([.3,0,1,np.nan],np.float64),owner_invalid_mass=np.array([.4,0,0,np.nan],np.float64))
    before=soft.copy();t.validate_targets(target,available)
    passed('Partialknown and fullUNKNOWN owner mass retained without row renormalization',np.array_equal(before,soft,equal_nan=True) and soft[0].sum()<1 and soft[2].sum()==0)
    def hard_cache(counts):
        labels=np.concatenate([np.full(n,c,np.int8) for c,n in enumerate(counts)])
        return dict(x=dict(original_query_available=np.ones(len(labels),bool)),y=dict(owner_target=labels,support_target=np.full((len(labels),2),-1,np.int8)))
    caches={'SYNTH_TRAIN4096':dict(x=dict(original_query_available=available),y=target),'PUBLIC_TRAIN1480':hard_cache([53,52,148]),'CF3816':hard_cache([157,157])}
    pools=t.pools_for(caches)
    passed('CDF uses original partialclass mass; unknown and inputU weightzero',np.allclose(pools['conditional'][0],[2/3,1/3,0,0],rtol=0,atol=1e-15) and np.allclose(pools['conditional'][1],[1/3,2/3,0,0],rtol=0,atol=1e-15) and np.array_equal(pools['conditional'][2],[0,1,0,0]))
    z=torch.tensor([[.2,-.4,.7],[-.1,.9,.3]],dtype=torch.float64,requires_grad=True)
    conditional=torch.stack([sum(float(pools['conditional'][c][i])*F.cross_entropy(z[i:i+1],torch.tensor([c])) for i in range(2)) for c in range(3)]).mean()
    known=torch.tensor(soft[:2]);direct=-(known*F.log_softmax(z,dim=1)/known.sum(0)).sum()/3
    gc=torch.autograd.grad(conditional,z,retain_graph=True)[0];gd=torch.autograd.grad(direct,z)[0]
    passed('Classconditional CE expected objective and analytic gradient match original soft classbalance',torch.allclose(conditional,direct,atol=1e-15,rtol=0) and torch.allclose(gc,gd,atol=1e-15,rtol=0))
    r1=np.random.default_rng(1810);r2=np.random.default_rng(1810);same=True
    for _ in range(5):
        d1=t.draw_step(r1,pools);d2=t.draw_step(r2,pools)
        same &= all(np.array_equal(d1[k],d2[k]) for k in d1) and [len(x) for x in d1.values()]==[32,32,48,48,32]
        same &= all(3 not in d1[k] and 2 not in d1[k] for k in ['visible','foreign','synthetic_owner'])
    passed('Same192 class/domain draws, no originalU or zero-known-mass owner draws',same)
    target_bad={k:v.copy() for k,v in target.items()};target_bad['support_target'][3,0]=0
    passed('Unavailable support target cannot silently train',rejects(lambda:t.validate_targets(target_bad,available)))
    # Actual head + actual AdamW/clip program, no source files or training cache.
    torch.manual_seed(1810);ma=t.head.Raw96SupportHead();torch.manual_seed(1810);mb=t.head.Raw96SupportHead()
    opts=[torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4) for m in [ma,mb]]
    parity=True
    for _ in range(3):
        state=torch.get_rng_state().clone()
        for m,op,inp in zip([ma,mb],opts,[raw,missing]):
            op.zero_grad(set_to_none=True);out=m(inp);loss=F.binary_cross_entropy_with_logits(out['support_logits'],torch.tensor([[1.,1.],[0.,1.]]))+F.cross_entropy(out['owner_logits'],torch.tensor([0,2]));loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);op.step()
        parity &= t.object_sha(ma.state_dict())==t.object_sha(mb.state_dict()) and t.object_sha(opts[0].state_dict())==t.object_sha(opts[1].state_dict()) and torch.equal(state,torch.get_rng_state())
    passed('Three invented AdamW updates RAW versus SAMmissing allparameter/optimizer/RNG byte parity',parity)
    passed('Both output heads and shared RGB trunk get finite gradients',all(p.grad is not None and torch.isfinite(p.grad).all() for p in ma.parameters()) and float(ma.encoder[0].weight.grad.abs().sum())>0 and float(ma.support_head.weight.grad.abs().sum())>0 and float(ma.owner_head.weight.grad.abs().sum())>0)
    passed('Preparation does not initialize CUDA',not torch.cuda.is_initialized())
    return checks
