"""Random tensors and small random nn modules only; no vendor model or weights."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from adapter_modules import LoRAQKV, install_last_qkv_lora, fixed_q_raw, fixed_q_read, assemble_c11_features, SpatialTargetQueryHead


def run():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(232)
    rand = lambda *shape: torch.randn(*shape,generator=generator)
    base = nn.Linear(384,1152)
    before = {k:v.clone() for k,v in base.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    adapter = LoRAQKV(base,seed=17)
    assert torch.equal(rng_before,torch.get_rng_state())
    x = rand(2,261,384)
    assert torch.equal(adapter(x),base(x))
    assert sum(p.numel() for p in adapter.adapter_parameters()) == 6144
    (adapter(x)*rand(2,261,1152)).sum().backward()
    assert adapter.lora_B.grad.abs().sum()>0 and adapter.lora_A.grad is not None and adapter.lora_A.grad.abs().sum()==0
    assert all(p.grad is None and not p.requires_grad for p in base.parameters())
    assert all(torch.equal(before[k],v) for k,v in base.state_dict().items())
    selected = {id(p) for p in adapter.parameters() if p.requires_grad}
    assert selected == {id(adapter.lora_A),id(adapter.lora_B)}
    # Diagnostic manual toy B perturbation only, not an optimizer step or fit.
    with torch.no_grad():adapter.lora_B.copy_(rand(1152,4)*.001)
    adapter.zero_grad(set_to_none=True)
    adapter(x).square().sum().backward()
    assert adapter.lora_A.grad.abs().sum()>0 and adapter.lora_B.grad.abs().sum()>0

    class FakeAttention(nn.Module):
        def __init__(self):super().__init__();self.qkv=nn.Linear(384,1152)
    class FakeBlock(nn.Module):
        def __init__(self):super().__init__();self.attn=FakeAttention()
    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__();self.chunked_blocks=False;self.embed_dim=384;self.num_heads=6;self.num_register_tokens=4
            self.blocks=nn.ModuleList([FakeBlock() for _ in range(12)])
    fake=FakeBackbone();last=fake.blocks[11].attn.qkv
    a=install_last_qkv_lora(fake,seed=17)
    assert a.base is last and all(isinstance(b.attn.qkv,nn.Linear) for b in fake.blocks[:11])
    assert [name for name,p in fake.named_parameters() if p.requires_grad] == ['blocks.11.attn.qkv.lora_A','blocks.11.attn.qkv.lora_B']

    n=6;patch=rand(n,256,384)
    q=torch.tensor([[0.,0.],[1.,1.],[.5,.5],[-.1,.5],[-1.,-1.],[float('nan'),.5]],dtype=torch.float64)
    boxes=torch.tensor([[0.,0.,1.,1.]]*n,dtype=torch.float64);support=torch.ones(n,dtype=torch.bool)
    raw,inside=fixed_q_raw(patch,q,boxes,support)
    desc,inside2=fixed_q_read(patch,q,boxes,support)
    assert torch.equal(inside,torch.tensor([1,1,1,0,0,0],dtype=torch.bool)) and torch.equal(inside,inside2)
    assert torch.equal(raw[0],patch[0,0]) and torch.equal(raw[1],patch[1,-1])
    assert torch.allclose(raw[2],patch[2].reshape(16,16,384)[7:9,7:9].mean((0,1)),atol=1e-7,rtol=1e-6)
    assert not torch.count_nonzero(raw[~inside]) and torch.allclose(desc[inside].norm(dim=1),torch.ones(3))
    assert torch.equal(desc,F.normalize(raw,p=2,dim=1,eps=1e-12))
    central=torch.tensor([1,1,1,0,1,1],dtype=torch.bool);frac=torch.ones(n,3);frac[3]=0
    cs,fs,fp,cc=rand(n,384),rand(n,384),rand(n,256,384),rand(n,384)
    features,_=assemble_c11_features(cs,patch,fs,fp,cc,q,boxes,frac,central)
    head=SpatialTargetQueryHead();control=copy.deepcopy(head)
    assert sum(p.numel() for p in head.parameters())==221825
    assert torch.equal(head(features),torch.zeros(n)) and torch.equal(head(features),control(features))
    head(features).sum().backward()
    assert head.head[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in head.head[0].parameters())

    original=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_spatial_target_query_features_20261023_v111/train_spatial_target_query_head.py')
    assert 'v112' not in str(original).casefold()
    for p in reversed([original,*original.parents]):assert not p.is_symlink()
    old=next(n for n in ast.parse(original.read_bytes()).body if isinstance(n,ast.ClassDef) and n.name=='SpatialTargetQueryHead')
    own=Path(__file__).parent/'adapter_modules.py'
    new=next(n for n in ast.parse(own.read_bytes()).body if isinstance(n,ast.ClassDef) and n.name=='SpatialTargetQueryHead')
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
    reader=original.parent/'feature_preparation/prepare_spatial_query_features.py'
    assert 'v112' not in str(reader).casefold()
    for p in reversed([reader,*reader.parents]):assert not p.is_symlink()
    reader_bytes=reader.read_bytes()
    assert hashlib.sha256(reader_bytes).hexdigest()=='cbf2f1cad553673b727c39fbe5c9c4263dfbdac35dba0aeea086f956e74ff19e'
    nodes=[n for n in ast.parse(reader_bytes).body if isinstance(n,ast.FunctionDef) and n.name in ('sample_Q_raw','assemble')]
    oldns={'np':np,'torch':torch,'F':F}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_C11_pure_CODE_only','exec'),oldns)
    oldraw,oldinside=oldns['sample_Q_raw'](patch,q.numpy(),boxes.numpy(),support.numpy())
    oldfeatures,oldinside2=oldns['assemble'](cs,patch,fs,fp,cc,q.numpy(),boxes.numpy(),frac.numpy(),central.numpy())
    assert torch.equal(raw,oldraw) and np.array_equal(inside.numpy(),oldinside)
    assert torch.equal(features,oldfeatures) and np.array_equal(oldinside,oldinside2)
    result={'completed':True,'PASS':True,'random_tensors_and_fake_linear_modules_only':True,'actual_DINO_instantiation_or_forward':0,'actual_SOURCE_or_protected_payload_or_weights':0,'optimizer_objects_or_steps':0,'LoRA_rank':4,'adapter_trainable_parameters':6144,'unchanged_scalar_head_parameters':221825,'checks':['zero-init exact output identity and unchanged base tensors','local init RNG leaves global RNG unchanged','B gradient at zero init, A zero until B nonzero; manual toy perturbation permits A gradient','frozen base excluded from trainable parameter list','exact final fused qkv location on fake12-block structure only','fixed original-q closedbox/border alignFalse/rawtoken->readL2 including invalid zero','pinned C11 raw reader and full1928 assembly bit-exact on invented tensors','original C11 scalar class AST exact/zero output/first backward staging'],'dual_target_architecture_selected':False}
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':run()
