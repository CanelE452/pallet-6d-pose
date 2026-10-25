"""V233 fixed common-local-crop paired adaptation; root --entry only.

No dataset/model/cache access on import. No raw SOURCE images, original GT or
old source NPZ is followed. Input body is selected memmap prefix rows only.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

import numpy as np
import torch
from crop_modules import CropFamilyHead,crop_features,family_loss,optimizer_groups

HERE=Path(__file__).absolute().parent
R=HERE.parent
PREFIX_CODE=R/'V232_PREFIX_TRAIN_CODE_A/prefix_runtime.py'
PREFIX_SHA='5b29e75eaa59c5be6091714a7999ada5272c3090fd4681f80979221984018be4'
ADAPTER_CODE=R/'V232_ADAPTER_PREPARATION_A/adapter_modules.py'
ADAPTER_SHA='9558f48279891f192eb9664968b41e9c03022a475b9666b5dc95229195fa8b9a'
OFFICIAL_SHA='f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb'
VENDOR_REVISION='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
ARMS=(('CONTROL_LOCAL_CROP',False),('TREAT_LAST_QKV_LORA4',True))
KEYS=('full_view','central_view','fractions','central4','condition','target','family','fit','available','row_id')


def safe(value):
    text=str(value)
    for token in ('v112','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation'):
        if token in text.casefold():raise ValueError('Forbidden path before FS: '+token)
    p=Path(os.path.abspath(text))
    for q in reversed((p,*p.parents)):
        if q.is_symlink():raise ValueError('Symlink rejected: '+str(q))
    return p


def read(path):
    fd=os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:return f.read()


def identity(b):return str(Path(os.path.abspath(b['path']))),b['sha256']


def verify(b,whitelist,pin=None):
    if identity(b) not in {identity(x) for x in whitelist}:raise ValueError('Exact root whitelist before file read')
    if pin is not None and identity(b)!=pin:raise ValueError('Pinned path/SHA mismatch')
    digest=hashlib.sha256();size=0
    fd=os.open(safe(b['path']),os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block);size+=len(block)
    if digest.hexdigest()!=b['sha256'] or ('size_bytes' in b and size!=b['size_bytes']):raise ValueError('Bound bytes mismatch')
    return safe(b['path'])


def binding(path):
    raw=read(path);return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}


def write(path,obj):
    with safe(path).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return binding(path)


def module(b,whitelist,name,pin=None):
    path=verify(b,whitelist,pin);m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    exec(compile(read(path),str(path),'exec'),m.__dict__);return m


def tensor_state_sha(state,base=False):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        if base and (name.endswith('.lora_A') or name.endswith('.lora_B')):continue
        if base:name=name.replace('blocks.11.attn.qkv.base.','blocks.11.attn.qkv.')
        a=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(a.dtype).encode());digest.update(json.dumps(list(a.shape)).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu_copy(v) for v in value)
    return value


def validate_samples(data,view_count):
    if set(data)!=set(KEYS):raise ValueError('Exact samples metadata keys')
    n=len(data['target'])
    for key in ('full_view','central_view','target','family','fit','available','row_id','central4'):
        if data[key].shape!=(n,):raise ValueError('Sample vector shape: '+key)
    for key in ('full_view','central_view'):
        if data[key].dtype.kind not in 'iu' or not ((data[key]>=0)&(data[key]<view_count)).all():raise ValueError('Bound view indices')
    if data['target'].dtype!=np.int8 or not np.isin(data['target'],[-1,0,1]).all():raise ValueError('Unknown target retained as -1')
    if data['family'].dtype!=np.int8 or not np.isin(data['family'],[0,1]).all():raise ValueError('Family0 SYN /1 YCB only')
    if any(data[k].dtype!=bool for k in ('fit','available','central4')):raise ValueError('Fixed boolean metadata')
    if data['row_id'].dtype.kind!='U' or len(set(data['row_id'].tolist()))!=n:raise ValueError('Unique Unicode row IDs')
    if data['fractions'].shape!=(n,2) or data['fractions'].dtype!=np.float32 or not np.isfinite(data['fractions']).all() or not ((data['fractions']>=0)&(data['fractions']<=1)).all():raise ValueError('Fixed validfractions')
    c=data['condition']
    if c.shape!=(n,32) or c.dtype!=np.float32 or not np.isfinite(c).all():raise ValueError('FP32 condition32')
    if np.count_nonzero(c[data['family']==1]):raise ValueError('YCB nativeUNKNOWN condition32 must be allzero')
    s=c[data['family']==0]
    if not np.isin(s[:,:8],[0,1]).all() or not (s[:,:8].sum(1)==1).all() or not np.isin(s[:,8:16],[0,1]).all() or not (np.abs(s[:,16:])<=8).all():raise ValueError('Original SYN native/mask/XY contract')
    if len(s):
        native=s[:,:8].argmax(1);xy=s[:,16:].reshape(-1,8,2);mask=s[:,8:16].astype(bool)
        if mask[np.arange(len(s)),native].any() or np.count_nonzero(xy[~mask]):raise ValueError('Self/unavailable other context exactzero')
    return data


def fixed_draws(data):
    eligible=data['fit']&data['available']&(data['target']>=0)
    pools=[np.flatnonzero(eligible&(data['family']==f)&(data['target']==y)) for f in (0,1) for y in (0,1)]
    if any(len(p)==0 for p in pools):raise ValueError('Empty required SOURCEFIT class pool; no fallback')
    rng=np.random.Generator(np.random.PCG64(17))
    draws=np.stack([np.concatenate([rng.choice(p,8,replace=True) for p in pools]) for _ in range(1000)]).astype(np.int64)
    return draws,eligible


class PrefixViews:
    def __init__(self,prefix,data):
        self.prefix=prefix;self.data=data;self.final_barrier=False;self.prebarrier_view_indices=set();self.prebarrier_sample_indices=set()
    def fetch(self,indices,device):
        indices=np.asarray(indices,np.int64)
        if not self.data['available'][indices].all():raise ValueError('Original unavailable rows must not forward')
        if not self.final_barrier and not (self.data['fit'][indices]&(self.data['target'][indices]>=0)).all():raise ValueError('Before final2 only known available SOURCEFIT fetch')
        views=np.concatenate((self.data['full_view'][indices],self.data['central_view'][indices]))
        unique,inverse=np.unique(views,return_inverse=True)
        # Fancy indexing materializes ONLY these authorized view rows; never np.asarray(full memmap).
        selected=np.array(self.prefix[unique],dtype=np.float32,copy=True)
        if selected.shape!=(len(unique),261,384) or not np.isfinite(selected).all():raise ValueError('Selected prefix body shape/finite')
        if not self.final_barrier:self.prebarrier_view_indices.update(map(int,unique));self.prebarrier_sample_indices.update(map(int,indices))
        return torch.as_tensor(selected,device=device),inverse,len(indices)


def features_for(indices,store,backbone,prefix_runtime,adapter_module,device,adapter_grad):
    selected,inverse,n=store.fetch(indices,device)
    output=prefix_runtime.last_block_features(backbone,selected,adapter_grad=adapter_grad)
    full={key:value[torch.as_tensor(inverse[:n],device=device)] for key,value in output.items() if isinstance(value,torch.Tensor)}
    centre={key:value[torch.as_tensor(inverse[n:],device=device)] for key,value in output.items() if isinstance(value,torch.Tensor)}
    data=store.data
    return crop_features(full,centre,torch.as_tensor(data['fractions'][indices],device=device),torch.as_tensor(data['central4'][indices],device=device),torch.as_tensor(data['condition'][indices],device=device),adapter_module)


def reduce_rows(rows):
    out={}
    for family in (0,1):
        for role in ('FIT','DEV'):
            rr=[r for r in rows if r['family']==family and r['role']==role];known=[r for r in rr if r['target']>=0]
            cm=[[sum(r['target']==y and r['prediction']==p for r in known) for p in ('D','E','U')] for y in (0,1)]
            item={'rows':len(rr),'known_DE':len(known),'target_UNKNOWN':sum(r['target']==-1 for r in rr),'original_input_U':sum(not r['available'] for r in rr),'CM_D_E_by_D_E_U':cm}
            for name,k in (('D',0),('E',1)):
                truth=sum(cm[k]);pred=cm[0][k]+cm[1][k];tp=cm[k][k]
                item[name]={'truth':truth,'predicted':pred,'precision':tp/pred if pred else None,'recall':tp/truth if truth else None}
            out[f'family{family}_{role}']=item
    return out


def configure_runtime():
    os.environ['XFORMERS_DISABLED']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.manual_seed(17)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(17)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)


def load_backbone(config,device,*,treatment=False):
    """Shared strict official-base loader for later root prefix build and fit.

    Config is an in-memory exact root-approved entry. All assets must occur in
    its explicit code/payload whitelists. This function is inert until called.
    Prefix builder uses treatment=False. No SOURCE data is opened by it.
    """
    if not (config.get('approved_official_backbone_load') is True or config.get('approved_source_local_crop_paired_adaptation') is True):
        raise ValueError('Explicit root official-model load approval required')
    if type(treatment) is not bool:raise ValueError('Explicit treatment flag')
    code=config['code_whitelist'];payload=config['payload_whitelist']
    asset=config['asset_identity']
    if config['official_state']['sha256']!=OFFICIAL_SHA or asset['sha256']!=OFFICIAL_SHA or asset['vendor_revision']!=VENDOR_REVISION:
        raise ValueError('Only original official f433 state and pinned7764 vendor')
    configure_runtime()
    vendor=safe(config['vendor_root'])
    for name,loaded in tuple(sys.modules.items()):
        if name=='dinov2' or name.startswith('dinov2.'):
            path=getattr(loaded,'__file__',None)
            if path is not None and not safe(path).is_relative_to(vendor):raise ValueError('Alternate preloaded DINO vendor')
    for item in config['backbone_code_bindings']:verify(item,code)
    factory=module(config['backbone_factory'],code,'v233_root_factory')
    for name in ('dinov2.layers.attention','dinov2.layers.block'):
        loaded=sys.modules.get(name)
        if loaded is not None and getattr(loaded,'XFORMERS_AVAILABLE',False):raise ValueError('Pinned math SDPA requires xFormers disabled at import')
    prefix_runtime=module(config['prefix_runtime'],code,'v233_loader_prefix',(str(PREFIX_CODE),PREFIX_SHA))
    adapter_module=module(config['adapter_module'],code,'v233_loader_adapter',(str(ADAPTER_CODE),ADAPTER_SHA))
    state_path=verify(config['official_state'],payload)
    official=torch.load(state_path,map_location='cpu',weights_only=True)
    if not isinstance(official,dict) or not all(isinstance(v,torch.Tensor) for v in official.values()):raise ValueError('Direct strict official state dict only')
    model=factory.build_backbone()
    model.load_state_dict(official,strict=True)
    model.float()
    original_sha=tensor_state_sha(model.state_dict(),base=True)
    prefix_runtime.prepare_backbone(model,treatment=treatment,adapter_module=adapter_module if treatment else None)
    model=model.to(device)
    if tensor_state_sha(model.state_dict(),base=True)!=original_sha:raise ValueError('Wrapper/device changed official base')
    model._v233_original_base_sha256=original_sha
    return model


def run(entry_path):
    e=json.loads(read(entry_path))
    if e['approved_source_local_crop_paired_adaptation'] is not True or e['source_only'] is not True or e['automatic_retry'] is not False:raise ValueError('Root true SOURCE entry required')
    if e['steps_each']!=1000 or e['threshold_logit']!=0:raise ValueError('Fixed1000/zero threshold only')
    code=e['code_whitelist'];payload=e['payload_whitelist']
    verify(e['code'],code,(str(safe(__file__)),binding(__file__)['sha256']))
    verify(e['crop_module'],code,(str(HERE/'crop_modules.py'),binding(HERE/'crop_modules.py')['sha256']))
    verify(e['protocol'],code)
    prefix_runtime=module(e['prefix_runtime'],code,'v233_pinned_prefix',(str(PREFIX_CODE),PREFIX_SHA))
    adapter_module=module(e['adapter_module'],code,'v233_pinned_adapter',(str(ADAPTER_CODE),ADAPTER_SHA))
    if e['official_state']['sha256']!=OFFICIAL_SHA or e['asset_identity']['sha256']!=OFFICIAL_SHA or e['asset_identity']['vendor_revision']!=VENDOR_REVISION:raise ValueError('Original official asset/vendor identity')
    for asset in e['backbone_code_bindings']:verify(asset,code)
    prefix_path=verify(e['prefix_npy'],payload);sample_path=verify(e['samples_npz'],payload);state_path=verify(e['official_state'],payload)
    if prefix_path.suffix!='.npy' or sample_path.suffix!='.npz':raise ValueError('Exact prefix.npy/samples.npz format')
    out=safe(e['output_directory'])
    if not out.is_relative_to(R) or 'v233' not in str(out).casefold() or out.exists():raise ValueError('Fresh root V233 output directory only')
    configure_runtime()
    # No RGB/source original arrays are available to this trainer.
    prefix=np.load(prefix_path,mmap_mode='r',allow_pickle=False)
    if prefix.dtype!=np.float32 or prefix.ndim!=3 or tuple(prefix.shape[1:])!=(261,384):raise ValueError('Prefix header schema')
    with np.load(sample_path,allow_pickle=False) as z:
        if set(z.files)!=set(KEYS):raise ValueError('Samples metadata schema')
        data=validate_samples({k:z[k] for k in KEYS},len(prefix))
    draws,eligible=fixed_draws(data);draw_sha=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    torch.manual_seed(17);initial_head=CropFamilyHead().float();initial_state=cpu_copy(initial_head.state_dict());initial_head_sha=tensor_state_sha(initial_state)
    if sum(p.numel() for p in initial_head.parameters())!=76098:raise ValueError('Fixed1186-64-2 head capacity')
    out.mkdir();started=time.monotonic();store=PrefixViews(prefix,data);pending={};base_sha=None
    write(out/'STARTED.json',{'entry':binding(entry_path),'code':e['code'],'draw_sha256':draw_sha,'steps_each':1000,'automatic_retry':False,'protected_data_or_model_selection':False})
    write(out/'COMMON_DRAW1000.json',{'seed':17,'RNG':'PCG64; family0D8,E8,family1D8,E8; replacement','indices':draws.tolist(),'sha256':draw_sha})
    try:
        def backbone_for(treatment):
            model=load_backbone(e,e['device'],treatment=treatment)
            nonlocal base_sha
            before=tensor_state_sha(model.state_dict(),base=True)
            if base_sha is None:base_sha=before
            if before!=base_sha:raise ValueError('Both arms must strict-load identical official base')
            return model
        fit_indices=np.flatnonzero(eligible)
        for arm,treatment in ARMS:
            model=backbone_for(treatment);head=CropFamilyHead().float().to(e['device']);head.load_state_dict(initial_state,strict=True)
            if tensor_state_sha(head.state_dict())!=initial_head_sha:raise ValueError('Cloned head initialization mismatch')
            opt=torch.optim.AdamW(optimizer_groups(head,model,treatment),weight_decay=1e-4)
            cached=None;cache_index=np.full(len(eligible),-1,np.int64)
            if not treatment:
                pieces=[]
                for start in range(0,len(fit_indices),64):pieces.append(features_for(fit_indices[start:start+64],store,model,prefix_runtime,adapter_module,e['device'],False).detach())
                cached=torch.cat(pieces);cache_index[fit_indices]=np.arange(len(fit_indices))
            adapter_initial={k:cpu_copy(v) for k,v in model.state_dict().items() if k.endswith('.lora_A') or k.endswith('.lora_B')}
            log_path=out/f'{arm}_TRAIN1000.jsonl';head.train()
            with log_path.open('x') as log:
                for step,indices in enumerate(draws,1):
                    if not eligible[indices].all():raise ValueError('Every actual loss draw must remain SOURCEFIT available known')
                    x=features_for(indices,store,model,prefix_runtime,adapter_module,e['device'],True) if treatment else cached[torch.as_tensor(cache_index[indices],device=e['device'])]
                    opt.zero_grad(set_to_none=True);logits=head(x)
                    loss,parts=family_loss(logits,torch.as_tensor(data['target'][indices],device=e['device']),torch.as_tensor(data['family'][indices],device=e['device']))
                    if not torch.isfinite(loss):raise ValueError('Finite paired objective')
                    loss.backward();opt.step()
                    log.write(json.dumps({'update':step,'sample_indices':indices.tolist(),'draws_sha256':draw_sha,'family0_BCE':float(parts[0].detach()),'family1_BCE':float(parts[1].detach()),'total_loss':float(loss.detach()),'family_counts':[16,16],'all_fit_available_known':True},allow_nan=False)+'\n')
                    if step%100==0:print(json.dumps({'arm':arm,'updates':step,'fixed_final_updates':1000}),flush=True)
            final_base=tensor_state_sha(model.state_dict(),base=True)
            if final_base!=base_sha:raise ValueError('Official base state changed')
            if any(not torch.isfinite(p).all() for p in list(head.parameters())+list(model.parameters())):raise ValueError('Finite final state')
            adapter_state={k:cpu_copy(v) for k,v in model.state_dict().items() if k.endswith('.lora_A') or k.endswith('.lora_B')}
            checkpoint={'arm':arm,'updates':1000,'head_state_dict':cpu_copy(head.state_dict()),'adapter_state_dict':adapter_state,'optimizer_state_dict':cpu_copy(opt.state_dict()),'initial_head_state_dict':initial_state,'initial_head_sha256':initial_head_sha,'initial_adapter_state_dict':adapter_initial,'official_base':e['official_state'],'base_state_sha256_before_after':base_sha,'vendor_revision':VENDOR_REVISION,'use_lora':treatment,'adapter_seed':17,'rank':4,'scale':1,'draws_sha256':draw_sha,'entry':binding(entry_path),'head_parameters':76098,'adapter_parameters':6144 if treatment else 0}
            path=out/f'{arm}_FINAL1000.pt';torch.save(checkpoint,path)
            pending[arm]={'checkpoint':binding(path),'log':binding(log_path),'head_state':checkpoint['head_state_dict'],'adapter_state':adapter_state,'head_state_sha256':tensor_state_sha(head.state_dict())}
            del cached,opt,head,model
            if torch.cuda.is_available():torch.cuda.empty_cache()
        pre_fit_samples=sorted(store.prebarrier_sample_indices);pre_fit_views=sorted(store.prebarrier_view_indices)
        if not eligible[pre_fit_samples].all():raise ValueError('Unexpected pre-barrier sample access')
        freeze=write(out/'ALL2_FINAL1000_FROZEN_BEFORE_DEV_FORWARD.json',{'completed':True,'PASS':True,'checkpoints':{k:v['checkpoint'] for k,v in pending.items()},'steps_each':1000,'same_head_initial_sha256':initial_head_sha,'same_draws_sha256':draw_sha,'same_base_sha256':base_sha,'DEV_forward_calls_before_freeze':0,'prebarrier_only_known_available_FIT_samples':pre_fit_samples,'prebarrier_view_indices':pre_fit_views})
        store.final_barrier=True;all_indices=np.flatnonzero(data['available']);summaries={};outputs={}
        for arm,treatment in ARMS:
            model=backbone_for(treatment);head=CropFamilyHead().float().to(e['device']);head.load_state_dict(pending[arm]['head_state'],strict=True);head.eval().requires_grad_(False)
            if treatment:
                actual={k:v for k,v in model.state_dict().items() if k.endswith('.lora_A') or k.endswith('.lora_B')}
                if set(actual)!=set(pending[arm]['adapter_state']):raise ValueError('Exact trained adapter keys')
                with torch.no_grad():
                    for k,v in actual.items():v.copy_(pending[arm]['adapter_state'][k].to(v.device))
            before=tensor_state_sha(model.state_dict());scores={}
            with torch.no_grad():
                for start in range(0,len(all_indices),64):
                    indices=all_indices[start:start+64];features=features_for(indices,store,model,prefix_runtime,adapter_module,e['device'],False);logits=head(features).cpu().numpy()
                    if not np.isfinite(logits).all():raise ValueError('Finite available SOURCE outputs')
                    scores.update({int(i):z for i,z in zip(indices,logits)})
            rows=[]
            for i in range(len(data['target'])):
                family=int(data['family'][i]);z=scores.get(i);value=None if z is None else float(z[family])
                rows.append({'sample_index':i,'row_id':str(data['row_id'][i]),'family':family,'role':'FIT' if data['fit'][i] else 'DEV','available':bool(data['available'][i]),'target':int(data['target'][i]),'logit0':None if z is None else float(z[0]),'logit1':None if z is None else float(z[1]),'selected_family_logit':value,'prediction':'U' if value is None else 'E' if value>=0 else 'D'})
            csvpath=out/f'{arm}_ALL_SOURCE_ROWS.csv'
            with csvpath.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            if tensor_state_sha(model.state_dict())!=before or tensor_state_sha(model.state_dict(),base=True)!=base_sha or tensor_state_sha(head.state_dict())!=pending[arm]['head_state_sha256']:raise ValueError('Frozen output state changed')
            summaries[arm]=reduce_rows(rows);outputs[arm]=binding(csvpath);del head,model
            if torch.cuda.is_available():torch.cuda.empty_cache()
        summary=write(out/'SOURCE_RESULTS.json',{'completed':True,'PASS':True,'arms':summaries,'target_authority':{'family0':'SYN saved intended-native witness0 D/E MAIN','family1':'YCB associated-r surface D/E AUX'},'two_referents_not_merged':True,'unknown_target_retained':True,'no_current_or_practical_transfer_claim':True})
        complete={'completed':True,'PASS':True,'entry':binding(entry_path),'code':e['code'],'crop_module':e['crop_module'],'protocol':e['protocol'],'prefix_runtime':e['prefix_runtime'],'adapter_module':e['adapter_module'],'official_base':e['official_state'],'asset_identity':e['asset_identity'],'prefix_npy':e['prefix_npy'],'samples_npz':e['samples_npz'],'checkpoints':{k:v['checkpoint'] for k,v in pending.items()},'logs':{k:v['log'] for k,v in pending.items()},'all2_freeze':freeze,'steps_each':1000,'draws_sha256':draw_sha,'head_initial_sha256':initial_head_sha,'base_state_sha256_before_after':base_sha,'outputs':outputs,'summary':summary,'prefix_whole_file_SHA_streamed_opaque':True,'numeric_prefix_rows_before_freeze_only_FIT':True,'DEV_forward_before_freeze':0,'new_RAW_RGB_GT_or_old_SOURCE_NPZ_reads':0,'new_real_inference_or_model_threshold_selection':0,'actual_seconds':time.monotonic()-started}
        write(out/'COMPLETE.json',complete);print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(out/'COMPLETE.json')}),flush=True)
    except BaseException:
        write(out/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entry',required=True);args=parser.parse_args();run(args.entry)
