"""Fixed V233 common-local-crop features/head. Torch-only, no file access."""
import torch
from torch import nn
from torch.nn import functional as F


class CropFamilyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(1186,64),nn.ReLU(),nn.Linear(64,2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self,features):
        if features.ndim!=2 or features.shape[1]!=1186 or features.dtype!=torch.float32:
            raise ValueError('FP32 common-local-crop1186 only')
        return self.head(features)


def crop_features(full,central,fractions,central4,condition,adapter_module):
    """FullCLS384 + raw fixed-Q384 + centreCLS384 + fractions2 + condition32.

    central4 gates the Q descriptor, not the packet's availability or label.
    The two CLS descriptors use their respective fraction>0 gates. Original
    raw full96 patch tokens are read by the exact fixed-centre four-token sum before the unchanged L2.
    """
    cls=full['x_norm_clstoken'];patch=full['x_norm_patchtokens'];centre=central['x_norm_clstoken']
    n=len(cls)
    if cls.shape!=(n,384) or centre.shape!=(n,384) or patch.shape!=(n,256,384):
        raise ValueError('Fixed DINO384/256 full96 and central32 view outputs required')
    if fractions.shape!=(n,2) or condition.shape!=(n,32) or central4.shape!=(n,):
        raise ValueError('Fixed scalar/conditioning shapes')
    if any(t.dtype!=torch.float32 for t in (cls,patch,centre,fractions,condition)) or central4.dtype!=torch.bool:
        raise ValueError('FP32 features/fractions/condition and bool central4 required')
    read=fixed_center_q_read(patch,central4)
    full_cls=torch.where((fractions[:,0]>0)[:,None],F.normalize(cls,p=2,dim=1,eps=1e-12),torch.zeros_like(cls))
    central_cls=torch.where((fractions[:,1]>0)[:,None],F.normalize(centre,p=2,dim=1,eps=1e-12),torch.zeros_like(centre))
    out=torch.cat((full_cls,read,central_cls,fractions,condition),1)
    if out.shape!=(n,1186) or not torch.isfinite(out).all():raise ValueError('Finite crop1186 required')
    return out


def fixed_center_q_read(patch,central4):
    """Exact grid=(0,0), bilinear align_corners=False read without grid backward.

    The full 16x16 patch grid centre is (7.5,7.5). Keep the pinned CUDA
    forward's top-left, top-right, bottom-left, bottom-right accumulation order.
    Original central4 masks raw descriptor before the unchanged unit-L2 math.
    """
    if patch.ndim!=3 or tuple(patch.shape[1:])!=(256,384) or patch.dtype!=torch.float32:
        raise ValueError('FP32 raw 16x16x384 patch tokens required')
    if central4.shape!=(len(patch),) or central4.dtype!=torch.bool or central4.device!=patch.device:
        raise ValueError('Original central4 bool on same device required')
    grid=patch.reshape(-1,16,16,384)
    raw=torch.zeros_like(grid[:,7,7])
    raw=raw+grid[:,7,7]*.25
    raw=raw+grid[:,7,8]*.25
    raw=raw+grid[:,8,7]*.25
    raw=raw+grid[:,8,8]*.25
    raw=torch.where(central4[:,None],raw,torch.zeros_like(raw))
    return F.normalize(raw,p=2,dim=1,eps=1e-12)


def family_loss(logits,target,family):
    """Only family0->output0, family1->output1; .5*(mean BCE0+mean BCE1)."""
    if logits.shape!=(32,2) or target.shape!=(32,) or family.shape!=(32,):
        raise ValueError('Fixed32 draw and two outputs required')
    parts=[]
    for k in (0,1):
        take=family==k
        if int(take.sum())!=16 or int(((target==0)&take).sum())!=8 or int(((target==1)&take).sum())!=8:
            raise ValueError('Each family requires exactly8D+8E, never S/U/missing as E')
        parts.append(F.binary_cross_entropy_with_logits(logits[take,k],target[take].to(logits.dtype)))
    return .5*(parts[0]+parts[1]),parts


def optimizer_groups(head,backbone,treatment):
    head_parameters=list(head.parameters())
    adapter=[p for p in backbone.parameters() if p.requires_grad]
    if len(adapter)!=(2 if treatment else 0) or sum(p.numel() for p in adapter)!=(6144 if treatment else 0):
        raise ValueError('Only treatment qkv A/B can join head optimizer')
    groups=[{'params':head_parameters,'lr':1e-3}]
    if treatment:groups.append({'params':adapter,'lr':1e-4})
    return groups
