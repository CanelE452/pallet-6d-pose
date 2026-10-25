"""Torch-only V232 module preparation. No I/O, vendor import, model load or fit.

Reader: C11 raw context patches -> original-q bilinear -> descriptor L2.
Only last fused qkv receives rank4 LoRA; source target/trainer contract is pending.
No dual-target architecture or target names are selected by this module.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LoRAQKV(nn.Module):
    """Frozen Linear + B(A(x)); rank4, scale1, A~N(0,.02), B=0.

    Construct only AFTER strict loading of the original base state. The wrapper
    changes state keys to base.weight/base.bias and adds lora_A/lora_B.
    """
    def __init__(self, base, *, seed):
        super().__init__()
        if not isinstance(base, nn.Linear) or type(seed) is not int:
            raise ValueError('Existing Linear and explicit integer init seed required')
        if base.weight.dtype != torch.float32:
            raise ValueError('FP32 base only; no implicit precision change')
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = 4
        self.scale = 1.0
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(base.weight.new_empty((4, self.in_features)))
        self.lora_B = nn.Parameter(base.weight.new_zeros((self.out_features, 4)))
        generator = torch.Generator(device=base.weight.device).manual_seed(seed)
        with torch.no_grad():
            self.lora_A.normal_(mean=0.0, std=0.02, generator=generator)

    def forward(self, x):
        if x.dtype != torch.float32 or x.shape[-1] != self.in_features:
            raise ValueError('Unchanged FP32 qkv input shape required')
        return self.base(x) + self.scale * F.linear(F.linear(x, self.lora_A), self.lora_B)

    def adapter_parameters(self):
        return (self.lora_A, self.lora_B)


def install_last_qkv_lora(backbone, *, seed):
    """Attach to pinned unchunked ViT-S/14 reg4 structure; never instantiate it.

    All base tensors are frozen/eval. The returned 6144 adapter parameters are
    the only trainable backbone tensors. The caller must not put the adapted
    last-block forward under no_grad/inference_mode during a future fit.
    """
    if (getattr(backbone, 'chunked_blocks', None) is not False
            or getattr(backbone, 'embed_dim', None) != 384
            or getattr(backbone, 'num_heads', None) != 6
            or getattr(backbone, 'num_register_tokens', None) != 4
            or len(backbone.blocks) != 12):
        raise ValueError('Exact nonchunked12-block dim384 heads6 reg4 backbone required')
    qkv = backbone.blocks[11].attn.qkv
    if not isinstance(qkv, nn.Linear) or (qkv.in_features, qkv.out_features) != (384,1152) or qkv.bias is None:
        raise ValueError('Only final fused qkv Linear384->1152 with bias; no fallback block')
    backbone.requires_grad_(False)
    backbone.eval()
    adapter = LoRAQKV(qkv, seed=seed).eval()
    backbone.blocks[11].attn.qkv = adapter
    expected = {'blocks.11.attn.qkv.lora_A', 'blocks.11.attn.qkv.lora_B'}
    actual = {name for name, parameter in backbone.named_parameters() if parameter.requires_grad}
    if actual != expected:
        raise ValueError('Unexpected trainable base/adapter scope')
    return adapter


def fixed_q_raw(context_patch, queries, boxes, support):
    """Torch-only C11 math. Geometry F64; raw token samples FP32; no q moves.

    Inside is torch.bool[N], unlike the historical numpy bool return. Geometry
    is fixed input, never an optimizable target/reference or recenter variable.
    """
    n = len(context_patch)
    if context_patch.shape != (n,256,384) or context_patch.dtype != torch.float32:
        raise ValueError('Raw FP32 context patches[N,256,384] required')
    if (queries.shape != (n,2) or boxes.shape != (n,4)
            or queries.dtype != torch.float64 or boxes.dtype != torch.float64
            or support.shape != (n,) or support.dtype != torch.bool):
        raise ValueError('Original geometry F64[N,2]/[N,4], bool support[N] required')
    if any(x.device != context_patch.device for x in (queries, boxes, support)):
        raise ValueError('Same-device geometry and raw tokens required')
    if queries.requires_grad or boxes.requires_grad:
        raise ValueError('Original q/box are immutable inputs, not optimized coordinates')
    finite = torch.isfinite(queries).all(1) & torch.isfinite(boxes).all(1) & ~(queries == -1).all(1)
    positive = (boxes[:,2:] > boxes[:,:2]).all(1)
    good = finite & positive & support
    u = torch.zeros((n,2), dtype=torch.float64, device=context_patch.device)
    u[good] = (queries[good] - boxes[good,:2]) / (boxes[good,2:] - boxes[good,:2])
    inside = good & (u >= 0).all(1) & (u <= 1).all(1)
    grid = (2*u-1).to(torch.float32).view(n,1,1,2)
    plane = context_patch.reshape(n,16,16,384).permute(0,3,1,2)
    value = F.grid_sample(plane,grid,mode='bilinear',padding_mode='border',align_corners=False)[:,:,0,0]
    value = torch.where(inside[:,None],value,torch.zeros_like(value))
    return value, inside


def fixed_q_read(context_patch, queries, boxes, support):
    """Raw context token -> original-q bilinear -> L2. No tokenwise pre-L2."""
    value, inside = fixed_q_raw(context_patch,queries,boxes,support)
    return F.normalize(value,p=2,dim=1,eps=1e-12), inside


def assemble_c11_features(ctxcls,ctxpatch,fullcls,fullpatch,centralcls,queries,boxes,fractions,central4):
    """The existing1928 descriptor/scalar layout; no new native/GT features."""
    n = len(ctxcls)
    if any(t.dtype != torch.float32 for t in (ctxcls,ctxpatch,fullcls,fullpatch,centralcls)):
        raise ValueError('FP32 raw DINO descriptors only')
    if (ctxcls.shape != fullcls.shape or ctxcls.shape != centralcls.shape or ctxcls.shape != (n,384)
            or fullpatch.shape != (n,256,384)):
        raise ValueError('Unchanged C11 DINO output shapes')
    if (fractions.shape != (n,3) or fractions.dtype != torch.float32
            or central4.shape != (n,) or central4.dtype != torch.bool
            or not torch.isfinite(fractions).all() or not ((fractions>=0)&(fractions<=1)).all()):
        raise ValueError('Fixed support fractions and central4 required')
    rawq, inside = fixed_q_raw(ctxpatch,queries,boxes,fractions[:,0]>0)
    rawp = fullpatch.reshape(n,16,16,384)[:,7:9,7:9].mean((1,2))
    normalize = lambda x,g: torch.where(g[:,None],F.normalize(x.float(),p=2,dim=1,eps=1e-12),torch.zeros_like(x))
    T = normalize(ctxcls,fractions[:,0]>0)
    Q = normalize(rawq,inside)
    Fv = normalize(fullcls,fractions[:,1]>0)
    C = normalize(centralcls,fractions[:,2]>0)
    Pv = normalize(rawp,central4)
    cosine = lambda a,b: F.cosine_similarity(a,b,dim=1,eps=1e-12)
    scalars = torch.stack((cosine(T,C),cosine(T,Fv),cosine(Q,C),cosine(Q,Pv),fractions[:,0],fractions[:,1],fractions[:,2],inside.to(torch.float32)),1)
    result = torch.cat((T,Q,Fv,C,Pv,scalars),1)
    if result.shape != (n,1928) or not torch.isfinite(result).all():
        raise ValueError('Nonfinite C11 layout')
    return result, inside


class SpatialTargetQueryHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head=torch.nn.Sequential(torch.nn.Linear(3464,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
        torch.nn.init.zeros_(self.head[-1].weight);torch.nn.init.zeros_(self.head[-1].bias)
    def forward(self,features):
        if not isinstance(features,torch.Tensor) or features.ndim!=2 or features.shape[1]!=1928 or features.dtype!=torch.float32:
            raise ValueError('Only FP32 frozen image features[N,1928] accepted.')
        T,Q,C,P=features[:,:384],features[:,384:768],features[:,1152:1536],features[:,1536:1920]
        x=torch.cat((features,torch.abs(T-C),T*C,torch.abs(Q-P),Q*P),1)
        return self.head(x).squeeze(-1)
