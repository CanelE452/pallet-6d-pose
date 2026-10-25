"""Independent random CPU tensors/fake backbone only; no actual DINO or data."""
import ast
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

H = Path(__file__).parent
R = H.parent
P = R.parent
A = R / 'V232_ADAPTER_PREPARATION_A/adapter_modules.py'
OLD = P / 'query_visibility_spatial_target_query_features_20261023_v111/feature_preparation/prepare_spatial_query_features.py'
OLD_HEAD = P / 'query_visibility_spatial_target_query_features_20261023_v111/train_spatial_target_query_head.py'

def read_code(path, expected):
    text = str(path)
    assert 'v112' not in text.casefold()
    cursor = Path('/')
    for part in path.parts[1:]:
        cursor /= part
        assert not stat.S_ISLNK(os.lstat(cursor).st_mode)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        data = stream.read()
    assert hashlib.sha256(data).hexdigest() == expected
    return data.decode()

code = read_code(A, '9558f48279891f192eb9664968b41e9c03022a475b9666b5dc95229195fa8b9a')
old = read_code(OLD, 'cbf2f1cad553673b727c39fbe5c9c4263dfbdac35dba0aeea086f956e74ff19e')
oldhead = read_code(OLD_HEAD, 'f34ce3b44cc491593bbe6b63fa498c0417a5ba70f54318787094ac0802efa5de')
namespace = {'torch': torch, 'nn': nn, 'F': F}
exec(compile(ast.parse(code), str(A), 'exec'), namespace)
oldns = {'torch': torch, 'np': np, 'F': F}
functions = [node for node in ast.parse(old).body
             if isinstance(node, ast.FunctionDef) and node.name in ('sample_Q_raw', 'assemble')]
exec(compile(ast.Module(body=functions, type_ignores=[]), str(OLD) + '::pure', 'exec'), oldns)
torch.set_num_threads(1)
torch.manual_seed(2387)
checks = {}

# No vendor import or actual ViT construction: independently written toy SDPA.
class ToyLastBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(384)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(384, 1152)
        self.out = nn.Linear(384, 384)
    def forward(self, x):
        b, n, c = x.shape
        qkv = self.attn.qkv(self.norm(x)).reshape(b, n, 3, 6, 64)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        value = F.scaled_dot_product_attention(q, k, v, dropout_p=0)
        return x + self.out(value.transpose(1, 2).contiguous().reshape(b, n, c))

class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.chunked_blocks = False
        self.embed_dim = 384
        self.num_heads = 6
        self.num_register_tokens = 4
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(11)] + [ToyLastBlock()])

fake = FakeBackbone()
original = {k: v.clone() for k, v in fake.state_dict().items()}
fake.load_state_dict(original, strict=True)
x = torch.randn(2, 9, 384)
baseline = fake.blocks[11](x).detach()
rng = torch.random.get_rng_state().clone()
adapter = namespace['install_last_qkv_lora'](fake, seed=43)
assert torch.equal(rng, torch.random.get_rng_state())
trainable = {n for n, p in fake.named_parameters() if p.requires_grad}
assert trainable == {'blocks.11.attn.qkv.lora_A', 'blocks.11.attn.qkv.lora_B'}
assert sum(p.numel() for p in adapter.adapter_parameters()) == 6144
assert all(not m.training for m in fake.modules())
assert torch.equal(fake.blocks[11](x).detach(), baseline)
for key, value in fake.state_dict().items():
    if 'lora_' in key:
        continue
    oldkey = key.replace('attn.qkv.base.', 'attn.qkv.')
    assert torch.equal(value, original[oldkey])
try:
    fake.load_state_dict(original, strict=True)
except RuntimeError:
    pass
else:
    raise AssertionError('Original strict state unexpectedly accepted after wrapper')
checks['strict_load_before_wrapper_6144_only_base_frozen_eval_rng_and_zero_delta'] = True

loss = (fake.blocks[11](x) * torch.randn_like(x)).mean()
loss.backward()
assert adapter.lora_B.grad.abs().sum() > 0
assert adapter.lora_A.grad is not None and not adapter.lora_A.grad.any()
assert all(p.grad is None for n, p in fake.named_parameters() if 'lora_' not in n)
fake.zero_grad(set_to_none=True)
with torch.no_grad():
    adapter.lora_B.fill_(0.001)  # invented perturbation; no optimizer or real state
fake.blocks[11](x).square().mean().backward()
assert adapter.lora_A.grad.abs().sum() > 0 and adapter.lora_B.grad.abs().sum() > 0
checks['eval_mode_still_allows_last_attention_adapter_gradient_and_no_base_gradient'] = True

n = 6
patch = torch.randn(n, 256, 384, requires_grad=True)
q = torch.tensor([[0.,0.],[1.,1.],[0.3,0.7],[-1.,-1.],[2.,1.],[0.5,0.5]], dtype=torch.float64)
boxes = torch.tensor([[0.,0.,1.,1.]]*n, dtype=torch.float64)
support = torch.tensor([True,True,True,True,True,False])
expected, old_inside = oldns['sample_Q_raw'](patch, q.numpy(), boxes.numpy(), support.numpy())
actual, inside = namespace['fixed_q_raw'](patch,q,boxes,support)
assert torch.equal(expected, actual) and np.array_equal(old_inside, inside.numpy())
assert inside.dtype == torch.bool and inside.tolist() == [True,True,True,False,False,False]
read, _ = namespace['fixed_q_read'](patch,q,boxes,support)
assert torch.equal(read, F.normalize(expected,p=2,dim=1,eps=1e-12))
read.sum().backward()
assert patch.grad.abs().sum() > 0 and not patch.grad[~inside].any()
try:
    namespace['fixed_q_raw'](patch,q.clone().requires_grad_(True),boxes,support)
except ValueError:
    pass
else:
    raise AssertionError('Trainable query accepted')
checks['C11_raw_sample_then_L2_bitexact_inside_bool_and_immutable_q_gradient_boundary'] = True

ctxcls, fullcls, centralcls = [torch.randn(n,384) for _ in range(3)]
fullpatch = torch.randn(n,256,384)
fractions = torch.rand(n,3)
fractions[-1] = 0
c4 = torch.tensor([True,True,False,True,False,False])
new, new_inside = namespace['assemble_c11_features'](ctxcls,patch.detach(),fullcls,fullpatch,centralcls,q,boxes,fractions,c4)
expected, old_inside = oldns['assemble'](ctxcls,patch.detach(),fullcls,fullpatch,centralcls,q.numpy(),boxes.numpy(),fractions.numpy(),c4.numpy())
assert torch.equal(new, expected) and np.array_equal(old_inside,new_inside.numpy())
checks['1928_full_descriptor_layout_and_scalar_normalization_bitexact'] = True

def head_ast(text):
    return ast.dump(next(n for n in ast.parse(text).body
                         if isinstance(n,ast.ClassDef) and n.name=='SpatialTargetQueryHead'),include_attributes=False)
assert head_ast(code)==head_ast(oldhead)
head = namespace['SpatialTargetQueryHead']()
assert sum(p.numel() for p in head.parameters())==221825
features = new.detach().requires_grad_(True)
z = head(features)
assert not z.any()
z.sum().backward()
assert head.head[-1].weight.grad.abs().sum()>0
assert features.grad is not None and not features.grad.any()
checks['original_221825_head_AST_and_zero_initial_output_staged_first_gradient'] = True

result = {'completed':True,'PASS':True,'checks':checks,'CPU_random_fake_modules_only':True,
          'actual_DINO_vendor_model_instantiations':0,'actual_weights_SOURCE_or_protected_reads':0,
          'optimizer_objects_or_steps':0,'C11_max_descriptor_error':float((new-expected).abs().max())}
out=H/'RANDOM_CPU_CHECKS.json'
fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
with os.fdopen(fd,'w') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps(result))
