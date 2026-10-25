"""V232 pure prefix/last-block runtime. No cache/file/RGB/weight I/O or trainer.

An authorized future caller supplies preprocessed views plus abstract view IDs,
strictly loaded backbone(s), and the pinned adapter module. The returned CPU
FP32 prefix packet is ready for that caller to save after its root entry.
Target routing, data packets, dual heads and a training recipe are not chosen.
"""
import torch
from torch import nn

ADAPTER_MODULE_PATH = '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/V232_ADAPTER_PREPARATION_A/adapter_modules.py'
ADAPTER_MODULE_SHA256 = '9558f48279891f192eb9664968b41e9c03022a475b9666b5dc95229195fa8b9a'
ADAPTER_SEED = 17
ADAPTER_NAMES = frozenset(('blocks.11.attn.qkv.lora_A','blocks.11.attn.qkv.lora_B'))


def _structure(backbone):
    if (getattr(backbone,'chunked_blocks',None) is not False
            or getattr(backbone,'embed_dim',None) != 384
            or getattr(backbone,'num_heads',None) != 6
            or getattr(backbone,'num_register_tokens',None) != 4
            or getattr(backbone,'patch_size',None) != 14
            or len(backbone.blocks) != 12):
        raise ValueError('Exact unchunked12-block ViT-S/14 reg4 structure required')
    if not callable(getattr(backbone,'prepare_tokens_with_masks',None)):
        raise ValueError('Original prepare_tokens_with_masks required')


def _is_adapter(qkv):
    return (hasattr(qkv,'lora_A') and hasattr(qkv,'lora_B')
            and isinstance(getattr(qkv,'base',None),nn.Linear)
            and getattr(qkv,'rank',None)==4 and getattr(qkv,'scale',None)==1.0
            and tuple(qkv.lora_A.shape)==(4,384) and tuple(qkv.lora_B.shape)==(1152,4))


def _trainable_scope(backbone):
    _structure(backbone)
    qkv=backbone.blocks[11].attn.qkv
    expected=ADAPTER_NAMES if _is_adapter(qkv) else frozenset()
    if not expected and not isinstance(qkv,nn.Linear):
        raise ValueError('Original final qkv or the pinned rank4 adapter only')
    actual=frozenset(name for name,p in backbone.named_parameters() if p.requires_grad)
    if actual != expected:
        raise ValueError('Only treatment final qkv A/B may be trainable in backbone')
    return expected


def prepare_backbone(backbone, *, treatment, adapter_module=None):
    """Freeze/eval base and optionally attach the caller's exact pinned adapter.

    Strict pretrained loading must happen before this call. No loading or
    module-source validation is performed here: the future entry must pin the
    caller-provided adapter_module to ADAPTER_MODULE_SHA256 above.
    """
    _structure(backbone)
    if type(treatment) is not bool:
        raise ValueError('Explicit control/treatment flag required')
    qkv=backbone.blocks[11].attn.qkv
    if not isinstance(qkv,nn.Linear) or (qkv.in_features,qkv.out_features)!=(384,1152) or qkv.bias is None:
        raise ValueError('Fresh original final Linear384->1152 required; no reattach/reset')
    if treatment and adapter_module is None:
        raise ValueError('Caller must supply the pinned adapter module')
    backbone.requires_grad_(False)
    backbone.eval()
    if treatment:
        adapter=adapter_module.install_last_qkv_lora(backbone,seed=ADAPTER_SEED)
        if not _is_adapter(adapter):raise ValueError('Pinned adapter rank4/scale1 schema')
    else:
        if adapter_module is not None:raise ValueError('CONTROL takes no adapter module')
    names=_trainable_scope(backbone)
    if bool(names) != treatment:raise ValueError('Control/treatment trainable scope mismatch')
    return backbone


def extract_prefix(backbone, preprocessed_views, input_manifest):
    """Original prepare_tokens_with_masks + blocks[0:11], detached CPU cache.

    input_manifest is an in-memory mapping with view_ids only required here.
    No filename, label, instance, GT, ROI or split is followed/interpreted.
    Future dataset/entry code owns the exact view preprocessing and role checks.
    """
    _trainable_scope(backbone)
    if (preprocessed_views.dtype!=torch.float32 or preprocessed_views.ndim!=4
            or tuple(preprocessed_views.shape[1:])!=(3,224,224)
            or len(preprocessed_views)<1 or not torch.isfinite(preprocessed_views).all()):
        raise ValueError('Already-preprocessed finite FP32 views[N,3,224,224] only')
    ids=input_manifest.get('view_ids')
    if not isinstance(ids,(list,tuple)) or len(ids)!=len(preprocessed_views) or not all(isinstance(x,str) and x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError('Exact unique abstract view_ids for this batch required')
    backbone.eval()
    with torch.no_grad():
        value=backbone.prepare_tokens_with_masks(preprocessed_views,None)
        if value.shape!=(len(ids),261,384) or value.dtype!=torch.float32:
            raise ValueError('Expected CLS1+register4+patch256 initial tokens')
        for block in backbone.blocks[:11]:value=block(value)
        if value.shape!=(len(ids),261,384) or value.dtype!=torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Finite FP32 prefix[N,261,384] required')
        prefix=value.detach().to(device='cpu').contiguous()
    return {'view_ids':tuple(ids),'prefix_tokens':prefix,'prefix_shape':tuple(prefix.shape)}


def last_block_features(backbone, prefix_tokens, *, adapter_grad):
    """blocks[11] + norm; output schema equals vendor forward_features.

    Pass the SAME saved prefix batch to both arms. Caller moves its unchanged
    FP32 CPU cache rows to the backbone device. adapter_grad=True is explicit
    future treatment-fit mode; False uses no_grad for CONTROL or frozen replay.
    eval() freezes stochastic base behavior without disabling adapter autograd.
    """
    names=_trainable_scope(backbone)
    if type(adapter_grad) is not bool:
        raise ValueError('Explicit adapter gradient mode required')
    if (prefix_tokens.ndim!=3 or tuple(prefix_tokens.shape[1:])!=(261,384)
            or prefix_tokens.dtype!=torch.float32 or prefix_tokens.requires_grad
            or not torch.isfinite(prefix_tokens).all()):
        raise ValueError('Immutable finite FP32 prefix[N,261,384] required')
    if prefix_tokens.device != next(backbone.parameters()).device:
        raise ValueError('Caller must place unchanged prefix rows on backbone device')
    if adapter_grad and (names!=ADAPTER_NAMES or not torch.is_grad_enabled() or torch.is_inference_mode_enabled()):
        raise ValueError('Treatment fit requires enabled autograd and final A/B only')
    backbone.eval()
    def finish():
        value=backbone.blocks[11](prefix_tokens)
        normed=backbone.norm(value)
        if normed.shape!=prefix_tokens.shape or normed.dtype!=torch.float32 or not torch.isfinite(normed).all():
            raise ValueError('Finite last-block output schema')
        return {'x_norm_clstoken':normed[:,0],
                'x_norm_regtokens':normed[:,1:5],
                'x_norm_patchtokens':normed[:,5:],
                'x_prenorm':value,'masks':None}
    if adapter_grad:return finish()
    with torch.no_grad():return finish()


def toy_smoke():
    """Small compatible fake backbone only; no actual DINO creation/forward."""
    import copy
    import hashlib
    import json
    import os
    from pathlib import Path
    import types
    torch.set_num_threads(1)
    generator=torch.Generator().manual_seed(232)
    rand=lambda *shape:torch.randn(*shape,generator=generator)
    class FakeAttention(nn.Module):
        def __init__(self):super().__init__();self.qkv=nn.Linear(384,1152)
    class FakeBlock(nn.Module):
        def __init__(self):super().__init__();self.attn=FakeAttention()
        def forward(self,x):
            # Deliberately cheap fake math, not vendor self-attention.
            change=self.attn.qkv(x.mean(1,keepdim=True)).reshape(len(x),1,3,384).mean(2)
            return x+.01*change
    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__();self.chunked_blocks=False;self.embed_dim=384;self.num_heads=6
            self.num_register_tokens=4;self.patch_size=14;self.pre=nn.Linear(3,384)
            self.position=nn.Parameter(rand(1,261,384)*.01)
            self.blocks=nn.ModuleList([FakeBlock() for _ in range(12)]);self.norm=nn.LayerNorm(384)
        def prepare_tokens_with_masks(self,x,masks=None):
            assert masks is None
            return self.pre(x.mean((2,3)))[:,None,:]+self.position
        def forward_features(self,x):
            value=self.prepare_tokens_with_masks(x,None)
            for block in self.blocks:value=block(value)
            normed=self.norm(value)
            return {'x_norm_clstoken':normed[:,0],'x_norm_regtokens':normed[:,1:5],
                    'x_norm_patchtokens':normed[:,5:],'x_prenorm':value,'masks':None}
    path=Path(ADAPTER_MODULE_PATH)
    if 'v112' in str(path).casefold():raise ValueError('Lexical rejection before filesystem')
    for parent in reversed((path,*path.parents)):
        if parent.is_symlink():raise ValueError('No code-source symlink')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:raw=f.read()
    assert hashlib.sha256(raw).hexdigest()==ADAPTER_MODULE_SHA256
    adapter_module=types.ModuleType('pinned_adapter_CODE_only');exec(compile(raw,str(path),'exec'),adapter_module.__dict__)
    original=FakeBackbone();control=copy.deepcopy(original);treatment=copy.deepcopy(original)
    prepare_backbone(control,treatment=False)
    prepare_backbone(treatment,treatment=True,adapter_module=adapter_module)
    views=rand(2,3,224,224)
    packet=extract_prefix(control,views,{'view_ids':['invented_context','invented_local']})
    prefix=packet['prefix_tokens']
    assert prefix.shape==(2,261,384) and prefix.dtype==torch.float32 and prefix.device.type=='cpu' and not prefix.requires_grad
    before=prefix.clone()
    with torch.no_grad():full=control.forward_features(views)
    split=last_block_features(control,prefix,adapter_grad=False)
    adapted=last_block_features(treatment,prefix,adapter_grad=True)
    for key in ('x_norm_clstoken','x_norm_regtokens','x_norm_patchtokens','x_prenorm'):
        assert torch.equal(full[key],split[key]) and torch.equal(split[key],adapted[key])
    assert torch.equal(prefix,before) and adapted['x_norm_patchtokens'].requires_grad
    (adapted['x_norm_patchtokens']*rand(2,256,384)).sum().backward()
    qkv=treatment.blocks[11].attn.qkv
    assert qkv.lora_B.grad.abs().sum()>0 and qkv.lora_A.grad is not None and qkv.lora_A.grad.abs().sum()==0
    assert all(p.grad is None for name,p in treatment.named_parameters() if name not in ADAPTER_NAMES)
    assert not control.training and not treatment.training
    try:
        with torch.no_grad():last_block_features(treatment,prefix,adapter_grad=True)
    except ValueError:pass
    else:raise AssertionError('Treatment fit must reject no_grad')
    result={'completed':True,'PASS':True,'fake_random_backbone_only':True,
            'prefix_shape':list(prefix.shape),'adapter_seed':17,'adapter_rank':4,'adapter_scale':1,
            'adapter_parameters':sum(p.numel() for p in qkv.adapter_parameters()),
            'checks':['fake full forward == prepare+first11+cached prefix+last+norm exactly',
                      'CONTROL == zero-init treatment from same immutable prefix exactly',
                      'FP32 detached CPU prefix; abstract view IDs preserved',
                      'eval/frozen base has no gradients; treatment B gradient reaches last-block output',
                      'explicit treatment grad rejects no_grad/inference use'],
            'actual_DINO_instantiation_or_forward':0,'actual_data_RGB_weights_arrays_GT_or_protected_access':0,
            'prefix_cache_file_writes':0,'optimizer_or_training_steps':0,'head_or_target_recipe_selected':False}
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--toy-smoke',action='store_true',required=True)
    parser.parse_args();toy_smoke()
