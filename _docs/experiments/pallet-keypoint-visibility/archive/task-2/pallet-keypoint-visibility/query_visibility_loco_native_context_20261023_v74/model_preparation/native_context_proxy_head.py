"""Untrained native-context component; three conditional renderer proxy states.

No physical semantic gold is implied. No GT/reference/perm/matched/source IDs
enter forward. A complete initial query set and slot identity are precomputed
input-only context42; no slot-to-GT reassignment takes place in this module.
"""
import torch
from torch import nn
from torch.nn import functional as F

PROXY_STATES = ('D_UNBLOCKED_RAY_PROXY', 'E_EXTERNAL_RAY_PROXY', 'S_CONVEX_SELF_PROXY')
FORWARD_FIELDS = ('p3_roi','local_rgb96','local_valid96','target_rgb128','target_valid128','context42')


class NativeContextProxyHead(nn.Module):
    """Six input-only tensors; flat N or whole-frame F,8 query order.

    Flat mode requires each query's own frame target crop already repeated.
    Whole mode repeats its F target crops internally to the same frame-major
    flat batch. No batchnorm/dropout makes these two paths exactly identical.
    Cached uint8 RGB normalization is fixed inside the module.
    """
    def __init__(self):
        super().__init__()
        self.p3_encoder=nn.Sequential(nn.Conv2d(65,24,3,padding=1),nn.GELU())
        self.local_encoder=nn.Sequential(nn.Conv2d(4,8,3,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(8,12,3,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(12,16,3,stride=2,padding=1),nn.GELU())
        self.target_encoder=nn.Sequential(nn.Conv2d(4,8,5,stride=2,padding=2),nn.GELU(),
            nn.Conv2d(8,16,3,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(16,24,3,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(24,24,3,stride=2,padding=1),nn.GELU())
        self.context_encoder=nn.Sequential(nn.Linear(42,16),nn.GELU(),nn.Linear(16,16),nn.GELU())
        # 24 P3 +16 localRGB +24 target +16 context +1 local-valid
        # +1 native-valid +1 target-valid fraction +2 fixed grid offsets.
        self.fusion=nn.Sequential(nn.Conv2d(85,32,1),nn.GELU(),nn.Conv2d(32,24,3,padding=1),nn.GELU())
        self.reference_localizer=nn.Conv2d(24,1,1)
        # Attention24 +max24 +target24 +context16 +2 factual valid fractions.
        self.classifier=nn.Sequential(nn.Linear(90,32),nn.GELU(),nn.Linear(32,3))
        yy,xx=torch.meshgrid(torch.arange(9),torch.arange(9),indexing='ij')
        self.register_buffer('native_offsets',torch.stack((xx-4,yy-4)).float()[None]/9)

    @staticmethod
    def _binary(value,name):
        if not torch.all((value==0)|(value==1)):
            raise ValueError(name+' must be factual binary input validity')

    def forward(self,p3_roi,local_rgb96,local_valid96,target_rgb128,target_valid128,context42):
        """Return proxy logits, separate reference response/attention and validity.

        Flat: P3[N,65,9,9], localRGB[N,3,96,96], localvalid[N,1,96,96],
        targetRGB[N,3,128,128], targetvalid[N,1,128,128], context[N,42].
        Whole: P3[F,8,65,9,9], localRGB[F,8,3,96,96],
        localvalid[F,8,1,96,96], targetRGB[F,3,128,128],
        targetvalid[F,1,128,128], context[F,8,42]. Output leading shape
        follows N or F,8; target association is a caller input contract.
        """
        values=(p3_roi,local_rgb96,local_valid96,target_rgb128,target_valid128,context42)
        if not all(isinstance(x,torch.Tensor) for x in values):
            raise TypeError('Only six Tensor inputs are accepted; no metadata dicts')
        if not all(x.device==p3_roi.device for x in values):
            raise ValueError('All inputs must share the model device')
        whole=p3_roi.ndim==5
        if whole:
            f,k=p3_roi.shape[:2]
            if k!=8 or tuple(p3_roi.shape)!=(f,8,65,9,9) or tuple(local_rgb96.shape)!=(f,8,3,96,96) or tuple(local_valid96.shape)!=(f,8,1,96,96) or tuple(context42.shape)!=(f,8,42) or tuple(target_rgb128.shape)!=(f,3,128,128) or tuple(target_valid128.shape)!=(f,1,128,128):
                raise ValueError('Whole-frame inputs require complete ordered eight slots')
            leading=(f,8);n=f*8
            p3_roi=p3_roi.reshape(n,65,9,9);local_rgb96=local_rgb96.reshape(n,3,96,96)
            local_valid96=local_valid96.reshape(n,1,96,96);context42=context42.reshape(n,42)
            target_rgb128=target_rgb128.repeat_interleave(8,dim=0);target_valid128=target_valid128.repeat_interleave(8,dim=0)
        else:
            if p3_roi.ndim!=4:raise ValueError('P3 input must be N65x9x9 or F8x65x9x9')
            n=p3_roi.shape[0];leading=(n,)
            if tuple(p3_roi.shape)!=(n,65,9,9) or tuple(local_rgb96.shape)!=(n,3,96,96) or tuple(local_valid96.shape)!=(n,1,96,96) or tuple(context42.shape)!=(n,42) or tuple(target_rgb128.shape)!=(n,3,128,128) or tuple(target_valid128.shape)!=(n,1,128,128):
                raise ValueError('Flat inputs must preserve the same per-query target association')
        if n==0:raise ValueError('Empty batches are not a model input')
        if not p3_roi.is_floating_point() or not context42.is_floating_point():
            raise TypeError('P3/context must be floating point cached inputs')
        if local_rgb96.dtype!=torch.uint8 or target_rgb128.dtype!=torch.uint8:
            raise TypeError('Both RGB inputs must be raw uint8; normalization is internal')
        if not torch.isfinite(p3_roi).all() or not torch.isfinite(context42).all():
            raise ValueError('Nonfinite forward features; use numeric-safe q context and valid flags')
        self._binary(p3_roi[:,64:65],'P3 channel64')
        self._binary(local_valid96,'local_valid96');self._binary(target_valid128,'target_valid128')
        dtype=self.classifier[0].weight.dtype
        p3=p3_roi.to(dtype);lv=local_valid96.to(dtype);tv=target_valid128.to(dtype)
        native=p3[:,64:65]
        p3_features=self.p3_encoder(torch.cat((p3[:,:64]*native,native),1))
        local_features=self.local_encoder(torch.cat((local_rgb96.to(dtype)/255*lv,lv),1))
        local_features=F.interpolate(local_features,size=(9,9),mode='bilinear',align_corners=False)
        local_valid9=F.adaptive_avg_pool2d(lv,(9,9))
        target_features=self.target_encoder(torch.cat((target_rgb128.to(dtype)/255*tv,tv),1))
        target_valid8=F.adaptive_avg_pool2d(tv,target_features.shape[-2:])
        target=(target_features*target_valid8).sum((2,3))/target_valid8.sum((2,3)).clamp_min(1e-6)
        target_fraction=tv.mean((1,2,3))[:,None]
        context=self.context_encoder(context42.to(dtype))
        broadcast=lambda x:x[:,:,None,None].expand(-1,-1,9,9)
        fused=self.fusion(torch.cat((p3_features,local_features,broadcast(target),broadcast(context),
                                    local_valid9,native,broadcast(target_fraction),self.native_offsets.to(dtype).expand(n,-1,-1,-1)),1))
        heatmap=self.reference_localizer(fused).squeeze(1)
        response_valid=local_valid9.squeeze(1)>0
        attention=F.softmax(heatmap.masked_fill(~response_valid,-1e4).flatten(1),1).reshape(n,9,9)*response_valid
        attention=attention/attention.sum((1,2),keepdim=True).clamp_min(1e-8)
        pooled=(fused*attention[:,None]).sum((2,3))
        mx=fused.masked_fill(~response_valid[:,None],-1e4).amax((2,3))
        mx=torch.where(response_valid.any((1,2))[:,None],mx,torch.zeros_like(mx))
        factual_fractions=torch.stack((lv.mean((1,2,3)),native.mean((1,2,3))),1)
        logits=self.classifier(torch.cat((pooled,mx,target,context,factual_fractions),1))
        # Factual availability only: no trained semantic U/absence detector.
        available=response_valid.any((1,2)) & native.bool().any((1,2,3)) & tv.bool().any((1,2,3))
        return dict(proxy_logits=logits.reshape(*leading,3),reference_response_map=heatmap.reshape(*leading,9,9),
                    reference_attention_weights=attention.reshape(*leading,9,9),response_input_valid=response_valid.reshape(*leading,9,9),
                    input_available=available.reshape(*leading))
