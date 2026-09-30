"""Unmerged pointwise adapters and explicit limited-adaptation contracts."""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from scripts.research.pallet_sensors_submission_v1.prior_model import expectation, target_distribution

TARGETS = tuple(f'blocks.3.{i}.{c}.conv' for i in range(3) for c in ('conv1','conv3'))


class PointwiseLoRA(nn.Module):
    def __init__(self, base, rank=4, alpha=4):
        super().__init__()
        if not (type(base) is nn.Conv2d and base.kernel_size==(1,1) and base.stride==(1,1)
                and base.groups==1 and base.padding==(0,0) and base.dilation==(1,1)):
            raise ValueError('Only explicitly audited pointwise Conv2d supported')
        if rank<=0 or alpha<=0:raise ValueError('Invalid rank/alpha')
        self.base=base.requires_grad_(False)  # exact reference: never reset/copy pretrained weight
        self.A=nn.Conv2d(base.in_channels,rank,1,bias=False,device=base.weight.device,dtype=base.weight.dtype)
        self.B=nn.Conv2d(rank,base.out_channels,1,bias=False,device=base.weight.device,dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.A.weight,a=5**.5)
        nn.init.zeros_(self.B.weight)
        self.rank,self.alpha,self.enabled=rank,alpha,True

    def forward(self,x):
        y=self.base(x)
        return y+(self.alpha/self.rank)*self.B(self.A(x)) if self.enabled else y


class AdaptedPoseFix(nn.Module):
    """train() always re-freezes BN statistics, including after parent mode changes."""
    def __init__(self,model,mode,rank=4,alpha=4):
        super().__init__();self.model=model.requires_grad_(False);self.mode=mode
        if mode=='LORA':
            for name in TARGETS:
                parent,key=name.rsplit('.',1)
                setattr(model.get_submodule(parent),key,PointwiseLoRA(model.get_submodule(name),rank,alpha))
        elif mode=='SAME_LAYER':
            for name in TARGETS:model.get_submodule(name).weight.requires_grad_(True)
        elif mode=='HEAD_ONLY':model.out.requires_grad_(True)
        elif mode=='FULL':
            for m in model.modules():
                if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):m.requires_grad_(True)
        elif mode!='BASE':raise ValueError(mode)
        self.train(False)

    def train(self,mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm):
                m.eval();m.requires_grad_(False)
        return self

    def forward(self,*args,**kwargs):return self.model(*args,**kwargs)

    def enable(self,enabled):
        for m in self.modules():
            if isinstance(m,PointwiseLoRA):m.enabled=bool(enabled)

    def adapter_state(self):
        return OrderedDict((k,v.detach().cpu().clone()) for k,v in self.state_dict().items() if '.A.' in k or '.B.' in k)

    def load_adapter(self,state):
        expected=self.adapter_state()
        if set(state)!=set(expected):raise ValueError('Adapter keys differ')
        for k,v in state.items():
            if v.shape!=expected[k].shape or v.dtype!=expected[k].dtype:raise ValueError(k)
        # Do NOT partial-load the whole model: BatchNorm compatibility loading
        # can insert zero num_batches_tracked when state metadata is absent.
        parameters=dict(self.named_parameters())
        with torch.no_grad():
            for k,v in state.items():parameters[k].copy_(v)


def trainable_contract(model):
    rows=[dict(name=n,shape=list(p.shape),count=p.numel()) for n,p in model.named_parameters() if p.requires_grad]
    return dict(mode=model.mode,trainable=sum(r['count'] for r in rows),total=sum(p.numel() for p in model.parameters()),parameters=rows)


def original_state(model):
    """Canonical state without A/B, preserving all base weights/BN buffers."""
    return OrderedDict((k.replace('.base.','.'),v) for k,v in model.model.state_dict().items() if '.A.' not in k and '.B.' not in k)


def supervised_loss(logits,target,valid):
    valid=valid.clone();valid[:,8]=False
    q,mask=target_distribution(target,valid)
    safe=torch.where(mask[...,None],target,torch.zeros_like(target))
    ce=(-(q*logits.flatten(2).log_softmax(-1)).sum(-1)*mask).mean()
    coord=((expectation(logits)/4-safe/4).abs()*mask[...,None]).mean()
    return ce+coord


def original_weight_l2(model):
    # Original weights only. A/B are Conv2d too; old losses() WOULD include them.
    adapter_ids={id(m.A.weight) for m in model.modules() if isinstance(m,PointwiseLoRA)}
    adapter_ids|={id(m.B.weight) for m in model.modules() if isinstance(m,PointwiseLoRA)}
    return sum(m.weight.square().sum() for m in model.modules()
               if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)) and id(m.weight) not in adapter_ids)*.5e-5


def preserve_mask(base_logits,target,valid,scales,provenance,mapping):
    """No evaluation loader dependency. Explicit TRAIN/normal + fixed object map required."""
    if any(r.get('source_kind')!='synthetic' or r.get('partition')!='train' or r.get('input_kind')!='normal' for r in provenance):
        raise ValueError('Preservation only accepts normal synthetic TRAIN')
    if len(provenance)!=len(target):raise ValueError('Batch provenance length')
    if mapping!={'rule':'FIXED_NATIVE_SOURCE_GT','permutation':list(range(9)),'student_reselection':False}:
        raise ValueError('Only fixed whole-object source-native mapping; no pointwise/student branch')
    if not torch.isfinite(scales).all() or not (scales>0).all():raise ValueError('Scale')
    with torch.no_grad():
        _,support=target_distribution(target,valid)
        error=(expectation(base_logits.detach())-target).norm(dim=-1)/scales[:,None]
        mask=support & torch.isfinite(error) & (error<=5)
        mask[:,8]=False
    return mask


def preservation_kl(student,teacher,mask):
    # T=1, average per selected corner, teacher stop-gradient, zero if none.
    tq=teacher.detach().flatten(2).softmax(-1)
    per=F.kl_div(student.flatten(2).log_softmax(-1),tq,reduction='none').sum(-1)
    return (per*mask).sum()/mask.sum().clamp_min(1)
