"""Invented metadata/prefix tensors and fake backbone only; zero optimizer steps."""
import copy
import hashlib
import json
import os
from pathlib import Path
import types
import numpy as np
import torch
from torch import nn
import train_crop_adaptation as T
from crop_modules import CropFamilyHead,crop_features,family_loss,optimizer_groups


def code_module(path,sha,name):
    path=Path(path)
    assert 'v112' not in str(path).casefold()
    for q in reversed([path,*path.parents]):assert not q.is_symlink()
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__);return m


def run():
    torch.set_num_threads(1);torch.manual_seed(17)
    rng=np.random.Generator(np.random.PCG64(233));n=16
    family=np.repeat([0,1],8).astype(np.int8)
    target=np.tile(np.asarray([0,1,0,1,-1,0,1,-1],np.int8),2)
    fit=np.tile([True]*4+[False]*4,2);available=np.ones(n,bool);available[[2,6,10,14]]=False
    fractions=np.ones((n,2),np.float32);fractions[0,1]=0
    central4=np.ones(n,bool);central4[1]=False
    condition=np.zeros((n,32),np.float32);condition[np.arange(8),np.arange(8)]=1
    data={'full_view':np.arange(n,dtype=np.int64)*2,'central_view':np.arange(n,dtype=np.int64)*2+1,'fractions':fractions,'central4':central4,'condition':condition,'target':target,'family':family,'fit':fit,'available':available,'row_id':np.asarray([f'FAKE_{i}' for i in range(n)])}
    T.validate_samples(data,n*2)
    draws,eligible=T.fixed_draws(data);again,eligible2=T.fixed_draws(data)
    assert draws.shape==(1000,32) and np.array_equal(draws,again) and np.array_equal(eligible,eligible2)
    assert eligible[1] and 1 in draws  # Q0 does not alter the fixed availability/loss mask.
    assert not np.isin([2,6,10,14],draws).any() and not np.isin([4,7,12,15],draws).any()
    for f in (0,1):
        for y in (0,1):assert ((family[draws]==f)&(target[draws]==y)).sum(1).min()==8

    class SpyPrefix:
        def __init__(self,array):self.array=array;self.reads=[]
        def __getitem__(self,index):self.reads.append(np.array(index));return self.array[index]
    spy=SpyPrefix(rng.normal(size=(n*2,261,384)).astype(np.float32));store=T.PrefixViews(spy,data)
    try:store.fetch([5],'cpu')
    except ValueError:pass
    else:raise AssertionError('DEV must fail before numeric prefix fetch')
    assert spy.reads==[]

    A=code_module(T.ADAPTER_CODE,T.ADAPTER_SHA,'toy_adapter_code')
    P=code_module(T.PREFIX_CODE,T.PREFIX_SHA,'toy_prefix_code')
    class FakeAttention(nn.Module):
        def __init__(self):super().__init__();self.qkv=nn.Linear(384,1152)
    class FakeBlock(nn.Module):
        def __init__(self):super().__init__();self.attn=FakeAttention()
        def forward(self,x):return x+.01*self.attn.qkv(x.mean(1,keepdim=True)).reshape(len(x),1,3,384).mean(2)
    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__();self.chunked_blocks=False;self.embed_dim=384;self.num_heads=6;self.num_register_tokens=4;self.patch_size=14
            self.blocks=nn.ModuleList([FakeBlock() for _ in range(12)]);self.norm=nn.LayerNorm(384)
        def prepare_tokens_with_masks(self,*args):raise AssertionError('Toy uses invented prefix, not RGB')
    control=FakeBackbone();treatment=copy.deepcopy(control)
    P.prepare_backbone(control,treatment=False);P.prepare_backbone(treatment,treatment=True,adapter_module=A)
    indices=draws[0]
    xc=T.features_for(indices,store,control,P,A,'cpu',False)
    xt=T.features_for(indices,store,treatment,P,A,'cpu',True)
    assert xc.shape==(32,1186) and torch.equal(xc,xt)
    assert not torch.count_nonzero(xc[indices==1,384:768])
    assert not torch.count_nonzero(xc[indices==0,768:1152])
    assert eligible[list(store.prebarrier_sample_indices)].all()
    y=torch.from_numpy(target[indices]);f=torch.from_numpy(family[indices])
    raw=torch.tensor(rng.normal(size=(32,2)),dtype=torch.float32,requires_grad=True)
    loss,parts=family_loss(raw,y,f)
    assert torch.equal(loss,.5*(parts[0]+parts[1]));loss.backward()
    assert not torch.count_nonzero(raw.grad[f==0,1]) and not torch.count_nonzero(raw.grad[f==1,0])
    assert raw.grad[f==0,0].abs().sum()>0 and raw.grad[f==1,1].abs().sum()>0

    head=CropFamilyHead();clone=copy.deepcopy(head)
    assert sum(p.numel() for p in head.parameters())==76098 and torch.equal(head(xc),torch.zeros(32,2))
    assert T.tensor_state_sha(head.state_dict())==T.tensor_state_sha(clone.state_dict())
    groups=optimizer_groups(head,treatment,True);base_ids={id(p) for name,p in treatment.named_parameters() if name not in P.ADAPTER_NAMES}
    selected={id(p) for g in groups for p in g['params']}
    assert not base_ids&selected and [g['lr'] for g in groups]==[1e-3,1e-4]
    assert sum(p.numel() for g in groups for p in g['params'])==82242
    assert len(optimizer_groups(clone,control,False))==1
    # A diagnostic nonzero head readout, not an update/fit, tests adapter reachability.
    with torch.no_grad():head.head[-1].weight.copy_(torch.from_numpy(rng.normal(size=(2,64)).astype(np.float32))*.01)
    base_before=T.tensor_state_sha(treatment.state_dict(),base=True)
    loss,_=family_loss(head(xt),y,f);loss.backward()
    assert treatment.blocks[11].attn.qkv.lora_B.grad.abs().sum()>0
    assert all(p.grad is None for name,p in treatment.named_parameters() if name not in P.ADAPTER_NAMES)
    assert T.tensor_state_sha(treatment.state_dict(),base=True)==base_before

    store.final_barrier=True;store.fetch([5],'cpu')
    rows=[{'family':0,'role':'DEV','target':0,'prediction':'U','available':False}, {'family':0,'role':'DEV','target':-1,'prediction':'E','available':True}, {'family':1,'role':'FIT','target':1,'prediction':'E','available':True}]
    reduced=T.reduce_rows(rows)
    assert reduced['family0_DEV']['CM_D_E_by_D_E_U']==[[0,0,1],[0,0,0]]
    assert reduced['family0_DEV']['target_UNKNOWN']==1 and reduced['family1_FIT']['E']['recall']==1
    result={'completed':True,'PASS':True,'actual_SOURCE_DINO_weights_RGB_GT_cache_reads_or_fit':0,'optimizer_steps':0,'toy_manual_readout_perturbation':True,'fake_backbone_only':True,'head_parameters':76098,'treatment_adapter_parameters':6144,'trainable_treatment_total':82242,'draw_shape':[1000,32],'draw_sha256':hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest(),'checks':['metadata masks retain U/UNKNOWN; central4 only gates descriptor','paired PCG64 fixed1000 family8D8E draws exact repeat','DEV blocked before any numeric prefix fetch; postbarrier permitted','selected FIT view fetch and zero-adapter CONTROL/TREAT1186 equality','Q central4 and respective CLS fraction gates; YCB condition0','two-output family-only gradient routing and equal-family BCE mean','head clone/zeroinit76098, adapter6144, correct separate LR groups/frozenbase exclusion','manual toy readout permits LoRA B gradient with unchanged frozen base','separate family/role CM preserves unavailable U and targetUNKNOWN']}
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':run()
