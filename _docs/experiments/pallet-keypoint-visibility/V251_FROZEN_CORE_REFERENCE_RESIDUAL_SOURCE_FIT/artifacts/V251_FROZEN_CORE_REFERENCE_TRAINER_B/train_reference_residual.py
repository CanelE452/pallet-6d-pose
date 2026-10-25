#!/usr/bin/env python3
"""V251 prospective frozen SOURCE core + matched reference residual; no pixel loader."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback
import numpy as np
import torch
from torch import nn

CONTROL='CONTROL_NO_REFERENCE'
TREAT='TREAT_REFERENCE'
ARMS=(CONTROL,TREAT)
VIEWS=('full_CLS','fixed_centre_Q','centre_CLS')
DRAW_SHA='20a1fc74ea15a04a16fe29f8e5b9a114b4629677710ed43c1628b2a1ac00b734'
CORE_CP_SHA='e645b50a2b9a41f6a06b502040c6683a7a5c3955093c07128b1095992409e65a'
CORE_STATE_SHA='5988479c4ccfa82a50b91ddc82d235c0009cc78280845830d0ddc4868d53cbf4'
CODE_PINS={'utility_code':'ce3380b5938132717d349b7be27545e5347110da8663d2b698ec44c96dbfe419','head_module':'dfd26ab7ccfa7037f0c3dca548d59c8574aa8b78c50699f70f01e03bf48935a9','selector_code':'0e08c0b8d1ba82795aa56be9326ff57f9bdcdb29a300b1de5d74f22d412b421a'}


def safe(value):
    p=Path(value)
    if not p.is_absolute() or '..' in p.parts or any('v112' in x.lower() or 'cache' in x.lower() or x.lower().endswith('.pyc') for x in p.parts):raise ValueError('Forbidden path before filesystem access')
    for x in reversed([p,*p.parents]):
        if x.is_symlink():raise ValueError('Bound artifact or ancestor symlink')
    return p


def binding(path):
    p=safe(path);h=hashlib.sha256();n=0
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
    return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':n}


def checked(b):
    if set(b)!= {'path','sha256','size_bytes'} or binding(b['path'])!=b:raise ValueError('Exact binding mismatch')
    return safe(b['path'])


def jread(path):return json.loads(safe(path).read_text(encoding='utf-8'))


def jwrite(path,value):
    safe(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return binding(path)


def load_code(b,name,pin):
    p=checked(b)
    if b['sha256']!=pin:raise ValueError('Frozen pure/helper CODE pin')
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class ReferenceResidual(nn.Module):
    """2434 trainable parameters; all3 fixed views jointly, no IDs or teacher API."""
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(73,32),nn.ReLU(),nn.Linear(32,2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,frozen_hidden64,reference9):
        n=len(frozen_hidden64)
        if frozen_hidden64.dtype!=torch.float32 or reference9.dtype!=torch.float32 or frozen_hidden64.shape!=(n,64) or reference9.shape!=(n,9) or not torch.isfinite(frozen_hidden64).all() or not torch.isfinite(reference9).all():raise ValueError('Finite frozen hidden64 and reference9')
        return self.net(torch.cat((frozen_hidden64.detach(),reference9),1))


def normalize(x):
    v=np.asarray(x,np.float64)
    if not np.isfinite(v).all():return None
    n=float(np.linalg.norm(v));return v/n if np.isfinite(n) and n>0 else None


def build_SYN_banks(features,data,slots):
    """New supervised FIT-D statistics by ORIGINAL input address, no class invention."""
    proto=np.zeros((3,8,384),np.float64);supported=np.zeros((3,8),bool);rows=[]
    eligible=data['fit']&data['available']&(data['family']==0)&(data['target']==0)
    for slot in range(8):
        ids=np.flatnonzero(eligible&(slots==slot))
        for vi,view in enumerate(VIEWS):
            total=np.zeros(384,np.float64);accepted=[]
            for i in ids:
                v=np.asarray(features[i,384*vi:384*(vi+1)],np.float64)
                if normalize(v) is not None:total=total+v;accepted.append(int(i))
            p=None if not accepted else normalize(total/len(accepted))
            if p is not None:proto[vi,slot]=p;supported[vi,slot]=True
            rows.append({'family':0,'view':view,'original_native_address':slot,'FIT_D_available_rows':len(ids),'finite_nonzero_reference_rows':len(accepted),'sample_indices':accepted,'supported':bool(supported[vi,slot]),'physical_canonical_anatomy':'NOT_CERTIFIED','pallet_type_or_material_filter':False})
    return proto,supported,rows


def reference_inputs(indices,features,nodes,group,slot,geometry,syn_proto,syn_support,ycb_proto,ycb_support,select_bank,family,available):
    """No target/fit/object/owner/r/pose/depth arguments; input-side choice only."""
    result=np.zeros((len(indices),9),np.float32);records=[]
    for k,i in enumerate(indices):
        i=int(i);g=int(group[i]);a=int(slot[i]);f=int(family[i]);detail=None
        if f==0:b=a;other=(a+1)%8;proto=syn_proto;support=syn_support
        elif f==1:
            detail=select_bank(nodes[g],geometry['original_available8'][g],geometry['node_available'][g],a,ycb_proto[0])
            b=detail['bank_index'] if bool(ycb_support[0].all()) else None
            other=None if b is None else (b+1)%4;proto=ycb_proto;support=ycb_support
        else:raise ValueError('Only original family0/1')
        good=[]
        for vi in range(3):
            q=normalize(features[i,vi*384:(vi+1)*384]);p=None if b is None else normalize(proto[vi,b]);o=None if other is None else normalize(proto[vi,other])
            ok=bool(available[i]) and b is not None and q is not None and p is not None and o is not None and bool(support[vi,b]) and bool(support[vi,other])
            if ok:result[k,vi]=np.dot(q,p);result[k,3+vi]=np.dot(q,o);result[k,6+vi]=1
            good.append(bool(ok))
        records.append({'sample_index':i,'family':f,'original_active_slot':a,'selected_bank_index':b,'next_bank_index':other,'per_view_supported':good,'choice':'ORIGINAL_NATIVE_ADDRESS' if f==0 else 'V250_INPUT_ONLY_OTHER_NODES','selector_UNKNOWN':b is None,'YCB_selector_detail':detail,'SYN_next_is_not_wrong_object_or_negative_gold':f==0})
    if not np.isfinite(result).all():raise ValueError('Finite reference9')
    return result,records


def core_read(core,indices,features,geometry,utility,device):
    x=torch.as_tensor(np.array(features['ORIGINAL_ALL_FEATURES.npy'][indices],copy=True),device=device)
    with torch.no_grad():
        z=core(x,*utility.context_args(indices,features,geometry,device))[0]
        hidden=core.base.head[1](core.base.head[0](x))
    return hidden.detach(),z.detach()


def save_csv(path,rows):
    with safe(path).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return binding(path)


def source_rows(data,logits):
    rows=[]
    for i,z in enumerate(logits):
        f=int(data['family'][i]);a=bool(data['available'][i]);v=float(z[f])
        rows.append({'sample_index':i,'row_id':str(data['row_id'][i]),'family':f,'role':'FIT' if data['fit'][i] else 'DEV','target':int(data['target'][i]),'available':a,'logit0':float(z[0]),'logit1':float(z[1]),'selected_family_E_logit':v,'prediction':'U' if not a else ('E' if v>=0 else 'D')})
    return rows


def train_arm(arm,residual,hidden,corelogit,ref9,data,draws,fit_map,utility,device,out,entry_binding,bank_binding):
    initial=utility.cpu_copy(residual.state_dict());initial_sha=utility.state_sha(initial)
    opt=torch.optim.AdamW(residual.parameters(),lr=.001,weight_decay=.0001)
    if sum(p.numel() for p in residual.parameters())!=2434:raise ValueError('Only2434 optimizer parameters')
    log=out/(arm+'_TRAIN1000.jsonl');residual.train()
    with log.open('x',encoding='utf-8') as f:
        for step,ids in enumerate(draws):
            ii=fit_map[ids]
            if (ii<0).any() or not data['fit'][ids].all() or not data['available'][ids].all():raise ValueError('SOURCEFIT available known optimizer scope')
            take=torch.as_tensor(ii,device=device);r=ref9[take] if arm==TREAT else torch.zeros_like(ref9[take])
            opt.zero_grad(set_to_none=True);z=corelogit[take]+residual(hidden[take],r)
            loss,parts=utility.family_loss(z,torch.as_tensor(data['target'][ids],device=device),torch.as_tensor(data['family'][ids],device=device))
            if not torch.isfinite(loss):raise ValueError('Finite objective')
            loss.backward();opt.step()
            f.write(json.dumps({'update':step+1,'sample_indices':ids.tolist(),'draws_sha256':DRAW_SHA,'family0_BCE':float(parts[0].detach()),'family1_BCE':float(parts[1].detach()),'total_loss':float(loss.detach()),'all_fit_available_known':True,'core_optimizer_updates':0,'reference_input':'ZERO' if arm==CONTROL else 'INPUT_SELECTED'},allow_nan=False)+'\n')
    if len(opt.state)!=4 or any(float(s['step'])!=1000 or any(not torch.isfinite(v).all() for v in s.values() if isinstance(v,torch.Tensor)) for s in opt.state.values()):raise ValueError('Four residual optimizer states finite@1000')
    if any(not torch.isfinite(p).all() for p in residual.parameters()):raise ValueError('Finite final residual')
    cp=out/(arm+'_FINAL1000.pt')
    torch.save({'arm':arm,'updates':1000,'core_optimizer_updates':0,'trainable_parameters':2434,'frozen_core_parameters':86054,'total_parameters':88488,'residual_state_dict':utility.cpu_copy(residual.state_dict()),'initial_residual_state_dict':initial,'initial_residual_sha256':initial_sha,'optimizer_state_dict':utility.cpu_copy(opt.state_dict()),'frozen_core_state_sha256':CORE_STATE_SHA,'frozen_core_checkpoint_sha256':CORE_CP_SHA,'entry':entry_binding,'reference_banks':bank_binding,'draws_sha256':DRAW_SHA,'threshold_raw_logit':0.,'use_lora':False},cp)
    return binding(cp),binding(log),utility.state_sha(residual.state_dict()),initial_sha

ROOT_PROTOCOL_SHA='630e3916466b0521a86eb8885853cc47c49600058b3a1fec2d46efc0a10b398c'
RECIPE={'steps':1000,'seed':17,'batch':32,'lr':.001,'weight_decay':.0001,'threshold_raw_logit':0.,'frozen_core_parameters':86054,'trainable_residual_parameters':2434,'CONTROL_reference9':'EXACT_ZERO','TREAT_reference9':'SELECTED_COS3_NEXT_COS3_SUPPORT3','new_CF_or_public_pixel_AUX':False,'historical_comparator':'BASE_FROZEN_V239','matched_min_mean_BR_gain_pp':1.}


def authorize(entry_path):
    e=jread(entry_path)
    if e.get('approved_source_reference_residual_training') is not True or e.get('source_only') is not True or e.get('automatic_retry') is not False:raise PermissionError('Root true SOURCE entry required before payload access')
    if checked(e['code'])!=safe(__file__) or e['fixed_recipe']!=RECIPE:raise ValueError('Self-code and fixedrecipe')
    p=jread(checked(e['root_protocol']))
    if e['root_protocol']['sha256']!=ROOT_PROTOCOL_SHA:raise ValueError('Root prospective protocol pin')
    input_alias={'original_features':e['features']['ORIGINAL_ALL_FEATURES.npy'],'node_cls':e['features']['NODE_CLS.npy'],'context_cls':e['features']['CONTEXT_CLS.npy'],'sample_group_index':e['features']['SAMPLE_GROUP_INDEX.npy'],'sample_node_index':e['features']['SAMPLE_NODE_INDEX.npy'],'relative_xy8':e['features']['RELATIVE_XY8.npy'],'samples':e['samples_npz'],'geometry':e['geometry_npz'],'feature_complete':e['feature_complete'],'YCB_prototypes':e['YCB_prototypes'],'warmstart_checkpoint':e['warmstart_checkpoint'],'warmstart_complete':e['warmstart_complete'],'head_module':e['head_module'],'selector_code':e['selector_code']}
    if input_alias!=p['inputs']:raise ValueError('Exact14 prospective input bindings')
    checked(e['implementation_protocol'])
    scope=jread(checked(e['root_scope']))
    if scope.get('approved') is not True or scope.get('entry_contract')!={k:v for k,v in e.items() if k!='root_scope'}:raise ValueError('Exact Root read+fit scope')
    for key,pin in CODE_PINS.items():
        if e[key]['sha256']!=pin:raise ValueError('Frozen CODE pin')
        checked(e[key])
    packet=jread(checked(e['feature_complete']));warm=jread(checked(e['warmstart_complete']))
    if packet.get('completed') is not True or packet.get('PASS') is not True or warm.get('completed') is not True or warm.get('PASS') is not True:raise ValueError('Completed canonical packet/warm source fit')
    for k in ('samples_npz','geometry_npz'):
        if e[k]!=packet[k]:raise ValueError('Original source metadata identity')
    for k,b in e['features'].items():
        if b!=packet['features'][k]:raise ValueError('Canonical feature identity')
    if e['warmstart_checkpoint']!=warm['checkpoints']['TREAT_CANONICAL_CONTEXT'] or e['warmstart_checkpoint']['sha256']!=CORE_CP_SHA or warm['final_head_sha256']['TREAT_CANONICAL_CONTEXT']!=CORE_STATE_SHA:raise ValueError('Exact frozen core identity')
    if e['feature_complete']!=warm['feature_complete']:raise ValueError('Warm core canonical feature identity')
    if e['YCB_prototypes']['sha256']!='d8aa3853a39efcf16f3d26ac74e3d58c3cb6e36a61c81b1fedf6674a77076d39':raise ValueError('Frozen YCB bank pin')
    out=safe(e['output_directory'])
    if out.exists():raise ValueError('Fresh Root output required; no automatic retries')
    return e,p,packet,warm,out


def load_payload(e,utility):
    if set(e['features'])!=set(utility.FEATURE_LAYOUT):raise ValueError('Six exact original canonical arrays')
    features={}
    for k,(dtype,sh) in utility.FEATURE_LAYOUT.items():
        a=np.load(checked(e['features'][k]),mmap_mode='r',allow_pickle=False)
        if a.dtype!=dtype or a.shape!=sh:raise ValueError('Original canonical shape/dtype')
        features[k]=a
    with np.load(checked(e['samples_npz']),allow_pickle=False) as z:data=utility.validate_samples({k:z[k] for k in utility.SAMPLE_KEYS})
    with np.load(checked(e['geometry_npz']),allow_pickle=False) as z:geometry=utility.validate_geometry({k:z[k] for k in utility.GEOMETRY_KEYS},1767)
    with np.load(checked(e['YCB_prototypes']),allow_pickle=False) as z:ycb=z['prototypes'];ys=z['supported'];views=z['view']
    if ycb.dtype!=np.float64 or ycb.shape!=(3,4,384) or ys.dtype!=bool or ys.shape!=(3,4) or tuple(map(str,views))!=VIEWS:raise ValueError('Frozen YCB3view4bank; no object ID decode')
    draws,eligible=utility.fixed_draws(data)
    if len(data['target'])!=7617 or int(data['fit'].sum())!=5876 or int((~data['fit']).sum())!=1741 or int(eligible.sum())!=5876:raise ValueError('Original7617/FIT5876/DEV1741')
    if hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()!=DRAW_SHA:raise ValueError('Original PCG17 ledger exact')
    return data,features,geometry,ycb,ys,draws,eligible


def mean_balanced_recall(metrics):
    a=[metrics[f'family{f}_DEV'][c]['recall'] for f in (0,1) for c in ('D','E')]
    return None if any(v is None for v in a) else sum(a)/4


def auc_positive_E(rows):
    pos=np.asarray([r['selected_family_E_logit'] for r in rows if r['target']==1 and r['available']],np.float64)
    neg=np.asarray([r['selected_family_E_logit'] for r in rows if r['target']==0 and r['available']],np.float64)
    if not len(pos) or not len(neg):return {'auc':None,'D_supported':len(neg),'E_supported':len(pos),'reason':'BOTH_CLASSES_REQUIRED'}
    return {'auc':sum(float((p>neg).sum())+.5*float((p==neg).sum()) for p in pos)/(len(pos)*len(neg)),'D_supported':len(neg),'E_supported':len(pos),'E_direction':'POSITIVE_RAW_SELECTED_FAMILY_LOGIT','ties':'half'}


def reduce_metrics(rows,utility):
    result=utility.reduce_rows(rows)
    for family in (0,1):
        for role in ('FIT','DEV'):
            r=[r for r in rows if r['family']==family and r['role']==role];m=result[f'family{family}_{role}'];a=m['D']['recall'];b=m['E']['recall']
            m.update(family=family,role=role,CM_D_E=[a[:2] for a in m['CM_D_E_by_D_E_U']],precisionE=m['E']['precision'],recallD=a,recallE=b,BR=None if a is None or b is None else (a+b)/2,AUC=auc_positive_E(r),U=sum(x['prediction']=='U' for x in r))
    return result


def reference_support(records,data):
    result={}
    for family in (0,1):
        for role in ('FIT','DEV'):
            rr=[r for r in records if r['family']==family and bool(data['fit'][r['sample_index']])==(role=='FIT')];n=len(rr)
            all3=sum(all(r['per_view_supported']) for r in rr);unknown=sum(r['selector_UNKNOWN'] for r in rr)
            result[f'family{family}_{role}']={'family':family,'role':role,'original_rows':n,'all3_supported':all3,'all3_supported_fraction':all3/n if n else None,'selector_UNKNOWN':unknown,'selector_UNKNOWN_fraction':unknown/n if n else None,'per_view_supported':{v:sum(r['per_view_supported'][i] for r in rr) for i,v in enumerate(VIEWS)},'reference_UNKNOWN_does_not_change_model_availability':True}
    return result


def gates(base,control,treat,utility):
    legacy=utility.source_gate(base,treat)
    a=mean_balanced_recall(control);b=mean_balanced_recall(treat);gain=None if a is None or b is None else (b-a)*100
    matched=gain is not None and gain>=1
    return {'PASS':bool(legacy['PASS'] and matched),'historical_SOURCE_TREAT_vs_BASE_FROZEN_V239':legacy,'matched_TREAT_minus_CONTROL_mean_family_BR_gain_pp':gain,'matched_min_gain_pp':1.,'matched_PASS':matched,'public_pixel80_gate_applied':False,'current_admission':False,'automatic_current_execution':False}


def run(entry_path):
    e,protocol,packet,warm,out=authorize(entry_path)
    utility=load_code(e['utility_code'],'v251_old_source_utility',CODE_PINS['utility_code'])
    module=load_code(e['head_module'],'v251_frozen_relational_module',CODE_PINS['head_module'])
    selector=load_code(e['selector_code'],'v251_fixed_input_selector',CODE_PINS['selector_code'])
    data,features,geometry,ycb,ys,draws,eligible=load_payload(e,utility)
    out.mkdir(parents=False);entry_binding=binding(entry_path)
    jwrite(out/'STARTED.json',{'entry':entry_binding,'automatic_retry':False,'fixed_recipe':RECIPE})
    try:
        globalqa=jwrite(out/'GLOBAL_CANONICAL_INPUT_QA_BEFORE_OPTIMIZERS.json',utility.global_feature_QA(features,data,geometry))
        utility.configure_runtime();device=e['device']
        if device not in ('cpu','cuda'):raise ValueError('Declared CPU or CUDA only')
        cp=torch.load(checked(e['warmstart_checkpoint']),weights_only=True,map_location='cpu')
        if cp.get('arm')!='TREAT_CANONICAL_CONTEXT' or cp.get('updates')!=1000 or cp.get('head_parameters')!=86054 or cp.get('use_lora') is not False or utility.state_sha(cp['head_state_dict'])!=CORE_STATE_SHA:raise ValueError('Frozen V239 core state')
        core=module.RelationalQueryHead().float();core.load_state_dict(cp['head_state_dict'],strict=True);core.eval().requires_grad_(False);core.to(device);del cp
        if sum(p.numel() for p in core.parameters())!=86054 or any(p.requires_grad for p in core.parameters()) or core.training:raise ValueError('Frozen core no gradients/no optimizer')
        models={}
        for arm in ARMS:
            torch.manual_seed(17);models[arm]=ReferenceResidual().float().to(device)
        if utility.state_sha(models[CONTROL].state_dict())!=utility.state_sha(models[TREAT].state_dict()):raise ValueError('Matched seed17 residual initialization')
        slots=features['SAMPLE_NODE_INDEX.npy'];groups=features['SAMPLE_GROUP_INDEX.npy'];x=features['ORIGINAL_ALL_FEATURES.npy'];nodes=features['NODE_CLS.npy']
        syn,ss,stats=build_SYN_banks(x,data,slots)
        bank_path=out/'REFERENCE_BANKS_FROZEN_BEFORE_OPTIMIZERS.npz'
        np.savez(bank_path,SYN_prototypes=syn,SYN_supported=ss,YCB_prototypes=ycb,YCB_supported=ys,view=np.asarray(VIEWS))
        bank_binding=binding(bank_path);bankstats=jwrite(out/'SYN_FIT_D_REFERENCE_SUPPORT.json',{'completed':True,'rows':stats,'label_used_only_for_SOURCE_FIT_statistics':'SYN intended-native D0; no new teacher labels','original_address_not_physical_canonical':True,'YCB_bank_unchanged':e['YCB_prototypes']})
        fit=np.flatnonzero(eligible);fit_map=np.full(7617,-1,np.int64);fit_map[fit]=np.arange(len(fit))
        hfit=[];zfit=[]
        for start in range(0,len(fit),64):
            h,z=core_read(core,fit[start:start+64],features,geometry,utility,device);hfit.append(h);zfit.append(z)
        hfit=torch.cat(hfit);zfit=torch.cat(zfit)
        rf,fitrecords=reference_inputs(fit,x,nodes,groups,slots,geometry,syn,ss,ycb,ys,selector.select_reference_bank,data['family'],data['available'])
        rf=torch.as_tensor(rf,device=device)
        if any(p.grad is not None for p in core.parameters()):raise ValueError('No frozen core gradients')
        initial_equal={}
        with torch.no_grad():
            for arm in ARMS:
                rr=rf if arm==TREAT else torch.zeros_like(rf)
                initial_equal[arm]=bool(torch.equal(zfit+models[arm](hfit,rr),zfit))
        if not all(initial_equal.values()):raise ValueError('Initial total logits must equal frozen core on allFIT')
        initqa=jwrite(out/'MATCHED_INITIAL_FROZEN_CORE_AND_DRAW_QA.json',{'completed':True,'PASS':True,'core_state_sha256':utility.state_sha(core.state_dict()),'same_initial_residual_sha256':utility.state_sha(models[CONTROL].state_dict()),'initial_total_logits_exact_frozen_core':initial_equal,'residual_output_weight_bias_zero':all(int(torch.count_nonzero(models[a].net[-1].weight))==0 and int(torch.count_nonzero(models[a].net[-1].bias))==0 for a in ARMS),'frozen_core_parameters':86054,'trainable_parameters':2434,'control_unused_reference_columns':288,'draws_sha256':DRAW_SHA,'FIT_core_descriptor_rows':len(fit),'new_DEV_core_reference_or_head_outputs':0})
        if initqa and utility.state_sha(core.state_dict())!=CORE_STATE_SHA:raise ValueError('Frozen core before optimizers')
        ledger=out/'COMMON_DRAW_LEDGER1000.jsonl'
        with ledger.open('x',encoding='utf-8') as f:
            for step,ids in enumerate(draws,1):f.write(json.dumps({'update':step,'sample_indices':ids.tolist(),'draws_sha256':DRAW_SHA,'all_fit_available_known':True})+'\n')
        cps={};logs={};final={};initial={}
        for arm in ARMS:
            cps[arm],logs[arm],final[arm],initial[arm]=train_arm(arm,models[arm],hfit,zfit,rf,data,draws,fit_map,utility,device,out,entry_binding,bank_binding)
            if utility.state_sha(core.state_dict())!=CORE_STATE_SHA or any(p.grad is not None for p in core.parameters()):raise ValueError('Frozen core unchanged across residual learning')
        if initial[CONTROL]!=initial[TREAT]:raise ValueError('Both arms identical initial residual')
        checked(bank_binding)
        ledger_binding=binding(ledger)
        freeze=jwrite(out/'BOTH_FINAL1000_LOGS_BANKS_FROZEN_BEFORE_ANY_DEV_OUTPUT.json',{'completed':True,'PASS':True,'checkpoints':cps,'logs':logs,'bank':bank_binding,'bank_support':bankstats,'core_checkpoint':e['warmstart_checkpoint'],'core_state_sha256':CORE_STATE_SHA,'initial_QA':initqa,'global_QA':globalqa,'draw_ledger':ledger_binding,'draws_sha256':DRAW_SHA,'new_updates_each':1000,'core_updates':0,'new_DEV_core_reference_or_head_outputs_before_freeze':0,'code':e['code'],'root_protocol':e['root_protocol'],'selector_code':e['selector_code'],'utility_code':e['utility_code']})
        for m in models.values():m.eval().requires_grad_(False)
        refall,records=reference_inputs(np.arange(7617),x,nodes,groups,slots,geometry,syn,ss,ycb,ys,selector.select_reference_bank,data['family'],data['available'])
        if not np.array_equal(refall[fit],rf.cpu().numpy()):raise ValueError('FIT reference input stable after freeze')
        np.save(out/'REFERENCE_INPUTS_ALL7617.npy',refall,allow_pickle=False)
        with (out/'REFERENCE_SELECTION_ALL7617.jsonl').open('x',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n')
        logits={a:np.zeros((7617,2),np.float32) for a in ARMS};base=np.zeros((7617,2),np.float32)
        with torch.no_grad():
            for start in range(0,7617,64):
                ids=np.arange(start,min(start+64,7617));h,z=core_read(core,ids,features,geometry,utility,device);r=torch.as_tensor(np.array(refall[ids],copy=True),device=device);base[ids]=z.cpu().numpy()
                for arm in ARMS:logits[arm][ids]=(z+models[arm](h,r if arm==TREAT else torch.zeros_like(r))).cpu().numpy()
        if not all(np.isfinite(z).all() for z in [base,*logits.values()]):raise ValueError('Finite saved frozen logits')
        outputs={};metrics={}
        for arm in ARMS:
            rr=source_rows(data,logits[arm]);outputs[arm]=save_csv(out/(arm+'_ALL_SOURCE_ROWS.csv'),rr);metrics[arm]=reduce_metrics(rr,utility)
            if utility.state_sha(models[arm].state_dict())!=final[arm]:raise ValueError('Frozen final residual unchanged during evaluation')
        br=source_rows(data,base);base_csv=save_csv(out/'BASE_FROZEN_V239_ALL_SOURCE_ROWS.csv',br);bm=reduce_metrics(br,utility)
        if utility.state_sha(core.state_dict())!=CORE_STATE_SHA:raise ValueError('Frozen core unchanged after all outputs')
        summary=jwrite(out/'SOURCE_RESULTS.json',{'completed':True,'PASS':True,'BASE_FROZEN_V239':bm,'arms':metrics,'SOURCE_gate':gates(bm,metrics[CONTROL],metrics[TREAT],utility),'reference_support':reference_support(records,data),'SYN_MAIN_vs_YCB_associated_r_AUX_separate':True,'all3views_joint_input_no_best_view':True,'current_admission':False,'exposed_SOURCE_DEV_not_blind':True})
        for frozen in [*cps.values(),*logs.values(),bank_binding,ledger_binding,freeze,e['code'],e['root_protocol'],e['selector_code'],e['utility_code']]:checked(frozen)
        complete={'completed':True,'PASS':True,'schema':'V251_FROZEN_SOURCE_CORE_REFERENCE_RESIDUAL_PAIR_V1','entry':entry_binding,'code':e['code'],'root_protocol':e['root_protocol'],'root_scope':e['root_scope'],'implementation_protocol':e['implementation_protocol'],'utility_code':e['utility_code'],'head_module':e['head_module'],'selector_code':e['selector_code'],'feature_complete':e['feature_complete'],'features':e['features'],'samples_npz':e['samples_npz'],'geometry_npz':e['geometry_npz'],'YCB_prototypes':e['YCB_prototypes'],'frozen_core_checkpoint':e['warmstart_checkpoint'],'frozen_core_complete':e['warmstart_complete'],'frozen_core_state_sha256_before':CORE_STATE_SHA,'frozen_core_state_sha256_after':utility.state_sha(core.state_dict()),'checkpoints':cps,'logs':logs,'outputs':outputs,'BASE_FROZEN_V239_CSV':base_csv,'summary':summary,'freeze':freeze,'reference_banks':bank_binding,'SYN_reference_support':bankstats,'reference_inputs':binding(out/'REFERENCE_INPUTS_ALL7617.npy'),'reference_selection':binding(out/'REFERENCE_SELECTION_ALL7617.jsonl'),'draw_ledger':ledger_binding,'draws_sha256':DRAW_SHA,'initial_residual_sha256':initial,'final_residual_sha256':final,'optimizer_updates':{a:1000 for a in ARMS},'core_optimizer_updates':0,'trainable_parameters':2434,'frozen_core_parameters':86054,'total_parameters':88488,'optimizer_states_each':4,'threshold_raw_logit':0.,'SYN_FIT_D_reference_statistics_computed':True,'cached_SOURCE_DEV_features_and_nonpixel_labels_loaded_for_global_integrity_QA':True,'SOURCE_DEV_used_for_bank_fit_gradient_optimizer_or_selection':False,'new_DEV_core_reference_or_head_outputs_before_bothfinalfreeze':0,'original_rows':15040,'known_rows':7617,'original_masked_S_U':7423,'FIT_known':5876,'DEV_known':1741,'new_raw_RGB_depth_annotation_official_encoder_weights_or_current_held_reads':0,'new_encoder_forward':0,'new_frozen_SOURCE_core_forward':True,'teacher_native_gold_created':False,'current_admission':False,'automatic_retry':False}
        jwrite(out/'COMPLETE.json',complete);print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(out/'COMPLETE.json')},ensure_ascii=False))
    except BaseException:
        jwrite(out/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'partial_outputs_preserved':True,'traceback':traceback.format_exc()});raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);a=p.parse_args();run(a.entry)


if __name__=='__main__':main()
