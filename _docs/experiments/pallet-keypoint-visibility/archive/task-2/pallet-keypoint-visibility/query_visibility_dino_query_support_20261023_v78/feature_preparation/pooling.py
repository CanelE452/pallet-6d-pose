"""Input-only fixed DINO preprocessing/readout. No model, file, GT or network access."""
import torch
from torch.nn import functional as F

def image_input(rgb):
    assert rgb.dtype==torch.uint8 and rgb.ndim==4 and rgb.shape[1]==3
    x=F.interpolate(rgb.float()/255.,size=(224,224),mode='bilinear',align_corners=False)
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
    std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    return (x-mean)/std

def token_weights(valid):
    assert valid.ndim==4 and valid.shape[1]==1
    assert valid.dtype==torch.bool
    return F.avg_pool2d(F.interpolate(valid.float(),size=(224,224),mode='bilinear',align_corners=False),14,14)[:,0]

def weighted_mean(tokens,weights):
    # tokens [B,H,W,C], weights [B,H,W]; zero support remains exactly zero.
    mass=weights.sum((1,2))
    pooled=(tokens*weights[...,None]).sum((1,2))/mass.clamp_min(1e-12)[:,None]
    return torch.where((mass>0)[:,None],pooled,torch.zeros_like(pooled))

def pool_local(tokens,weights):
    assert tokens.ndim==3 and tokens.shape[1:]==(256,384)
    assert weights.shape==(tokens.shape[0],16,16)
    t=tokens.reshape(-1,16,16,384)
    center=weighted_mean(t[:,7:9,7:9],weights[:,7:9,7:9])
    whole=weighted_mean(t,weights)
    return torch.stack((center,whole),1),torch.stack((weights[:,7:9,7:9].mean((1,2)),weights.mean((1,2))),1)

def pool_target(tokens,weights):
    assert tokens.ndim==3 and tokens.shape[1:]==(256,384)
    assert weights.shape==(tokens.shape[0],16,16)
    return weighted_mean(tokens.reshape(-1,16,16,384),weights),weights.mean((1,2))
