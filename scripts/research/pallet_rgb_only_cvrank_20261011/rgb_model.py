"""M3 input ablation with both inherited feature-score entrances removed.

The serialized modules and initialization match NativeEvidence. Geometry,
role/context, coverage and support follow the frozen GenericPointRefiner
formula; adapt3/adapt4/patch_body are never called. RGB is native M3 evidence.
"""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[3]
PRIOR_NAMESPACE=ROOT/'scripts/research/pallet_nonsevere_precision_20261011'
if str(PRIOR_NAMESPACE) not in sys.path:sys.path.insert(0,str(PRIOR_NAMESPACE))
import precision_model as prior_model

UNUSED_MODULES=('adapt3','adapt4','patch_body')


class RGBOnlyEvidence(prior_model.NativeEvidence):
    def __init__(self,base,native_template=None):
        if native_template is None:
            super().__init__(base)
        else:
            # Transfer the modules created by the exact prior setup. No second
            # module initialization and no extra CPU/CUDA RNG consumption.
            nn.Module.__init__(self)
            assert native_template.base is base
            self.base=base;self.rgb=native_template.rgb;self.residual=native_template.residual
        for name in UNUSED_MODULES:getattr(self.base,name).requires_grad_(False)

    @classmethod
    def from_native(cls,template):return cls(template.base,native_template=template)

    def forward(self,batch,use_rgb=True,tiles=None):
        base=self.base;dtype=base.role_embedding.weight.dtype
        points,boxes,input_shape=(batch[k].to(dtype) for k in ('points','boxes','input_shape'))
        point_valid=batch['point_valid'].bool()
        valid=point_valid&torch.isfinite(points).all(-1)&~(points==-1).all(-1)
        box_valid=torch.isfinite(boxes).all(-1)&(boxes[:,2:]>boxes[:,:2]).all(-1)
        safe=torch.where(valid[...,None],points,torch.zeros_like(points))
        bb=torch.where(box_valid[:,None],boxes,boxes.new_tensor([0,0,1,1]))
        size=bb[:,2:]-bb[:,:2];diag=size.norm(dim=-1).clamp_min(1);center=(bb[:,2:]+bb[:,:2])*.5
        candidates=diag[:,None,None]*base.displacements[None]
        locations=safe[:,:8,None]+candidates[:,None,:-1]
        positions=locations[:,:,:,None]+diag[:,None,None,None,None]*base.stencil_fraction*base.stencil[None,None,None]
        inside=(positions[...,0]>=0)&(positions[...,1]>=0)&(positions[...,0]<input_shape[:,None,None,None,1])&(positions[...,1]<input_shape[:,None,None,None,0])
        b,k,c=locations.shape[:3]
        # Explicit 48 zeros, never a zero-image convolution with learned bias.
        pooled=points.new_zeros((b,k,c,48))
        own=(safe[:,:8]-center[:,None])/diag[:,None,None]
        geom=torch.cat([own,(size/diag[:,None])[:,None].expand(-1,8,-1),
            (size[:,0]/size[:,1].clamp_min(1e-6)).log()[:,None,None].expand(-1,8,1)],-1)
        context=torch.cat([geom,base.role_embedding.weight[None].expand(b,-1,-1)],-1)
        displacement=base.displacements[None,None,:-1].expand(b,8,-1,-1)/.08
        regular=torch.cat([pooled,context[:,:,None].expand(-1,-1,c,-1),displacement,inside.to(dtype).mean(-1,keepdim=True)],-1)
        null=torch.cat([pooled.mean(-2),context],-1)
        base_logits=torch.cat([base.scorer(regular).squeeze(-1),base.null_scorer(null)],-1)
        regular=torch.cat([regular,regular.new_zeros((*regular.shape[:-1],1))],-1)
        null=torch.cat([null,null.new_zeros((*null.shape[:-1],3)),null.new_ones((*null.shape[:-1],1))],-1)
        features=torch.cat([regular,null[:,:,None]],-2)
        if use_rgb:
            assert tiles is not None
            rgb,grid=tiles;encoded=self.rgb(rgb)
            sampled=F.grid_sample(encoded,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
            rgb_features=sampled[:,:,0].transpose(1,2).reshape(b,k,222,16)
            rgb_features=rgb_features*torch.cat([features.new_ones(221),features.new_zeros(1)])[None,None,:,None]
        else:rgb_features=features.new_zeros((b,k,222,16))
        residual_logits=self.residual(torch.cat([features,rgb_features],-1)).squeeze(-1)
        out=dict(logits=base_logits+residual_logits,points_raw=points,point_valid=valid,
                 point_support=valid[:,:8]&box_valid[:,None]&inside.any(-1).any(-1),
                 candidate_displacements=candidates,box_diagonal=diag,coverage=inside.to(dtype).mean(-1),boxes=boxes,
                 base_logits=base_logits,residual_logits=residual_logits)
        # The CE/coordinate loss decodes actual logits explicitly. lam=0 is
        # retained only for compatibility with the prior wrapper output field.
        out['points']=points.clone()
        return out

    def parameter_counts(self):
        total=sum(p.numel() for p in self.parameters())
        skipped=sum(p.numel() for name in UNUSED_MODULES for p in getattr(self.base,name).parameters())
        zero_columns=sum(layer.weight[:,:48].numel() for layer in (self.base.scorer[0],self.base.null_scorer[0],self.residual[0]))
        trainable=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return dict(serialized_parameters=total,frozen_feature_module_parameters=skipped,
                    trainable_parameters=trainable,executed_parameter_tensors_elements=total-skipped,
                    zero_input_feature_scorer_column_parameters=zero_columns,
                    input_active_parameter_elements=total-skipped-zero_columns,
                    RGB_CNN_parameters=sum(p.numel() for p in self.rgb.parameters()),
                    inherited_base_parameters=sum(p.numel() for p in self.base.parameters()),
                    added_parameters=total-sum(p.numel() for p in self.base.parameters()),
                    interpretation='Zero-input columns execute in the same Linear tensors; data gradient is zero but AdamW weight decay may change their bytes. Frozen feature modules are neither executed nor optimized.')
