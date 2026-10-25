#!/usr/bin/env python3
"""V252 fixed SOURCE direct41 reference classifiers; no old model/CP/pixel loader."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import traceback
import types
import numpy as np
import torch
from torch import nn

CONTROL='CONTROL_GENERIC_REFERENCE'
TREAT='TREAT_TARGET_REFERENCE'
ARMS=(CONTROL,TREAT)
VIEWS=('full_CLS','fixed_centre_Q','centre_CLS')
ROOT_PROTOCOL_SHA='35a49f0628e48f35fbacdbfb40f44aca5fd22d47f58d9cf82dbc9dcff2034a71'
UTILITY_SHA='ce3380b5938132717d349b7be27545e5347110da8663d2b698ec44c96dbfe419'
DRAW_SHA='20a1fc74ea15a04a16fe29f8e5b9a114b4629677710ed43c1628b2a1ac00b734'
RECIPE={'updates_each':1000,'seed':17,'batch':32,'lr':.001,'weight_decay':.0001,'threshold_raw_logit':0.,'input_width':41,'parameters':1410,'no_old_core':True,'generic_FIT_D_self_inclusion':True,'historical_comparator':'BASE_FROZEN_V239','matched_min_BR_gain_pp':1.}


def safe(value):
    p=Path(value)
    if not p.is_absolute() or '..' in p.parts or any('v112' in a.lower() or 'cache' in a.lower() or a.lower().endswith('.pyc') for a in p.parts):raise ValueError('Forbidden path before FS')
    for a in reversed([p,*p.parents]):
        if a.is_symlink():raise ValueError('Symlink bound artifact/ancestor')
    return p


def binding(path):
    p=safe(path);h=hashlib.sha256();n=0
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
    return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':n}


def checked(b):
    if set(b)!= {'path','sha256','size_bytes'} or binding(b['path'])!=b:raise ValueError('Closed exact binding mismatch')
    return safe(b['path'])


def jread(p):return json.loads(safe(p).read_text(encoding='utf-8'))


def jwrite(p,j):
    p=safe(p);tmp=safe(str(p)+'.partial')
    if p.exists() or tmp.exists():raise ValueError('Fresh atomic JSON leaf required')
    tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p);return binding(p)


def load_utility(b):
    p=checked(b)
    if b['sha256']!=UTILITY_SHA:raise ValueError('Exact pure utility CODE')
    m=types.ModuleType('v252_pinned_source_utility');m.__file__=str(p)
    exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)
    return m


class DirectReferenceHead(nn.Module):
    """1410 params. Forward accepts only reference9 + original condition32."""
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(41,32),nn.ReLU(),nn.Linear(32,2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,input41):
        if input41.dtype!=torch.float32 or input41.ndim!=2 or input41.shape[1]!=41 or not torch.isfinite(input41).all():raise ValueError('Finite FP32 reference9+condition32 only')
        return self.net(input41)


def build_model():return DirectReferenceHead()


def assemble_inputs(reference9,condition32):
    n=len(reference9)
    if reference9.dtype!=np.float32 or condition32.dtype!=np.float32 or reference9.shape!=(n,9) or condition32.shape!=(n,32) or not np.isfinite(reference9).all() or not np.isfinite(condition32).all():raise ValueError('Original FP32 ref9/condition32')
    return np.concatenate((reference9,condition32),axis=1)


def normalize(x):
    x=np.asarray(x,np.float64)
    if not np.isfinite(x).all():return None
    n=float(np.linalg.norm(x));return x/n if np.isfinite(n) and n>0 else None


def build_generic_banks(features,data):
    """Supervised raw descriptor means, NOT means of normalized target banks."""
    bank=np.zeros((2,3,384),np.float64);support=np.zeros((2,3),bool);rows=[]
    for family in (0,1):
        ids=np.flatnonzero(data['fit']&data['available']&(data['target']==0)&(data['family']==family))
        for vi,view in enumerate(VIEWS):
            total=np.zeros(384,np.float64);accepted=[]
            for i in ids:
                v=np.asarray(features[i,384*vi:384*(vi+1)],np.float64)
                if normalize(v) is not None:total=total+v;accepted.append(int(i))
            p=None if not accepted else normalize(total/len(accepted))
            if p is not None:bank[family,vi]=p;support[family,vi]=True
            rows.append({'family':family,'view':view,'FIT_available_D_rows':len(ids),'accepted':len(accepted),'sample_indices':accepted,'supported':bool(support[family,vi]),'raw_stored_descriptor_equal_weight':True,'full_FIT_D_self_inclusion':True,'DEV_used':False,'address_or_object_specific_filter':False})
    return bank,support,rows


def generic_reference_inputs(indices,features,family,available,prototypes,supported):
    """Input-only generic cosine3 duplicated as next3, then pair-support3."""
    out=np.zeros((len(indices),9),np.float32)
    for k,i in enumerate(indices):
        i=int(i);f=int(family[i])
        if f not in (0,1):raise ValueError('Original family route0/1')
        for vi in range(3):
            q=normalize(features[i,vi*384:(vi+1)*384]);p=normalize(prototypes[f,vi])
            if available[i] and supported[f,vi] and q is not None and p is not None:
                v=np.float32(np.dot(q,p));out[k,vi]=v;out[k,3+vi]=v;out[k,6+vi]=1
    return out


def validate_ref9(r):
    if r.dtype!=np.float32 or r.shape!=(7617,9) or not np.isfinite(r).all() or not np.isin(r[:,6:],[0.,1.]).all():raise ValueError('Frozen targetref9 schema/support')
    if (np.abs(r[:,:6])>1.000002).any():raise ValueError('Cosine numeric range')
    for vi in range(3):
        if np.count_nonzero(r[r[:,6+vi]==0,vi]) or np.count_nonzero(r[r[:,6+vi]==0,3+vi]):raise ValueError('Unsupported pair must retain0/0')


def authorize(entry_path):
    e=jread(entry_path)
    if e.get('approved_direct_reference_source_training') is not True or e.get('source_only') is not True or e.get('automatic_retry') is not False:raise PermissionError('Exact Root true entry required before payloads')
    if checked(e['code'])!=safe(__file__) or e['fixed_recipe']!=RECIPE:raise ValueError('Self-code and frozen recipe')
    protocol=jread(checked(e['root_protocol']))
    if e['root_protocol']['sha256']!=ROOT_PROTOCOL_SHA:raise ValueError('Root prospective protocol pin')
    expected={k:{'path':b['original_local_path'],'sha256':b['sha256'],'size_bytes':b['size_bytes']} for k,b in protocol['inputs'].items()}
    if e['inputs']!=expected or len(expected)!=9:raise ValueError('Exact nine prospective bindings')
    checked(e['implementation_protocol']);scope=jread(checked(e['root_scope']))
    if scope.get('approved') is not True or scope.get('entry_contract')!={k:v for k,v in e.items() if k!='root_scope'}:raise ValueError('Root exact entry scope')
    # BASE bytes/body are deliberately deferred until both final checkpoints freeze.
    for key,b in e['inputs'].items():
        if key!='BASE_FROZEN_V239_CSV':checked(b)
    old=jread(e['inputs']['V251_complete']['path']);qa=jread(e['inputs']['V251_independent_QA']['path']);packet=jread(e['inputs']['feature_complete']['path'])
    if not all(j.get('completed') is True and j.get('PASS') is True for j in (old,qa,packet)):raise ValueError('Pinned completed provenance and independent cache QA')
    for key,previous in [('V251_saved_reference_inputs','reference_inputs'),('V251_saved_banks','reference_banks'),('BASE_FROZEN_V239_CSV','BASE_FROZEN_V239_CSV')]:
        if e['inputs'][key]!=old[previous]:raise ValueError('V251 frozen ref/bank/baseline lineage')
    if e['inputs']['samples']!=packet['samples_npz'] or e['inputs']['original_features']!=packet['features']['ORIGINAL_ALL_FEATURES.npy']:raise ValueError('Original canonical SOURCE identity')
    out=safe(e['output_directory'])
    expected_out=Path(__file__).absolute().parent.parent/'V252_ROOT_DIRECT_REFERENCE/matched_direct_reference_v1'
    if out!=expected_out or out.exists():raise ValueError('Exact fresh Root output; no retry or overwrite')
    return e,protocol,out


def load_payload(e,utility):
    inputs=e['inputs'];x=np.load(safe(inputs['original_features']['path']),mmap_mode='r',allow_pickle=False)
    if x.dtype!=np.float32 or x.shape!=(7617,1186):raise ValueError('Original frozen3view feature schema')
    with np.load(safe(inputs['samples']['path']),allow_pickle=False) as z:data=utility.validate_samples({k:z[k] for k in utility.SAMPLE_KEYS})
    target=np.load(safe(inputs['V251_saved_reference_inputs']['path']),mmap_mode='r',allow_pickle=False);validate_ref9(target)
    if len(data['target'])!=7617 or int(data['fit'].sum())!=5876 or int((~data['fit']).sum())!=1741:raise ValueError('Fixed7617/FIT5876/DEV1741')
    if np.array(x[:,-32:],copy=True).tobytes()!=data['condition'].tobytes() or np.array(x[:,1152:1154],copy=True).tobytes()!=data['fractions'].tobytes():raise ValueError('Original condition/fraction bytes unchanged')
    draws,eligible=utility.fixed_draws(data)
    if int(eligible.sum())!=5876 or hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()!=DRAW_SHA:raise ValueError('Original exact SOURCE FIT PCG17 ledger')
    return x,data,target,draws,eligible


def train_arm(arm,model,inputFIT,data,draws,fit_map,utility,device,out,entry,bank):
    initial=utility.cpu_copy(model.state_dict());initial_sha=utility.state_sha(initial)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);log=out/(arm+'_TRAIN1000.jsonl');model.train()
    if sum(p.numel() for p in model.parameters())!=1410:raise ValueError('Only1410 optimizer parameters')
    with log.open('x',encoding='utf-8') as f:
        for step,ids in enumerate(draws):
            ii=fit_map[ids]
            if (ii<0).any() or not data['fit'][ids].all() or not data['available'][ids].all():raise ValueError('Only known available FIT optimized')
            opt.zero_grad(set_to_none=True);z=model(inputFIT[torch.as_tensor(ii,device=device)])
            loss,parts=utility.family_loss(z,torch.as_tensor(data['target'][ids],device=device),torch.as_tensor(data['family'][ids],device=device))
            if not torch.isfinite(loss):raise ValueError('Finite fixed family objective')
            loss.backward();opt.step()
            f.write(json.dumps({'update':step+1,'sample_indices':ids.tolist(),'draws_sha256':DRAW_SHA,'family0_BCE':float(parts[0].detach()),'family1_BCE':float(parts[1].detach()),'total_loss':float(loss.detach()),'all_fit_available_known':True},allow_nan=False)+'\n')
    if len(opt.state)!=4 or any(float(s['step'])!=1000 or any(not torch.isfinite(v).all() for v in s.values() if isinstance(v,torch.Tensor)) for s in opt.state.values()) or any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Four optimizerstates finite1000/finalparameters')
    cp=out/(arm+'_FINAL1000.pt')
    torch.save({'arm':arm,'updates':1000,'parameters':1410,'input_width':41,'model_state_dict':utility.cpu_copy(model.state_dict()),'initial_model_state_dict':initial,'initial_model_sha256':initial_sha,'optimizer_state_dict':utility.cpu_copy(opt.state_dict()),'draws_sha256':DRAW_SHA,'entry':entry,'generic_bank':bank,'threshold_raw_logit':0.,'old_core_used':False,'use_lora':False},cp)
    return binding(cp),binding(log),utility.state_sha(model.state_dict()),initial_sha


def source_rows(data,logits):
    return [{'sample_index':i,'row_id':str(data['row_id'][i]),'family':int(data['family'][i]),'role':'FIT' if data['fit'][i] else 'DEV','target':int(data['target'][i]),'available':bool(data['available'][i]),'logit0':float(z[0]),'logit1':float(z[1]),'selected_family_E_logit':float(z[int(data['family'][i])]),'prediction':'U' if not data['available'][i] else ('E' if z[int(data['family'][i])]>=0 else 'D')} for i,z in enumerate(logits)]


def save_csv(path,rows):
    with safe(path).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return binding(path)


def read_BASE_after_freeze(b,data):
    with checked(b).open(newline='',encoding='utf-8') as f:raw=list(csv.DictReader(f))
    if len(raw)!=7617:raise ValueError('Exact saved BASE7617')
    rows=[];seen=set()
    for r in raw:
        i=int(r['sample_index']);f=int(r['family']);y=int(r['target']);av=r['available']=='True'
        if i in seen or not 0<=i<7617 or r['available'] not in ('True','False') or r['row_id']!=str(data['row_id'][i]) or f!=int(data['family'][i]) or y!=int(data['target'][i]) or av!=bool(data['available'][i]) or r['role']!=('FIT' if data['fit'][i] else 'DEV'):raise ValueError('BASE original identity/label/role')
        z=[float(r['logit0']),float(r['logit1'])];v=float(r['selected_family_E_logit']);pred='U' if not av else 'E' if v>=0 else 'D'
        if not np.isfinite(z+[v]).all() or v!=z[f] or r['prediction']!=pred:raise ValueError('BASE fixed0 decoding')
        rows.append(dict(r,sample_index=i,family=f,target=y,available=av,logit0=z[0],logit1=z[1],selected_family_E_logit=v));seen.add(i)
    if seen!=set(range(7617)):raise ValueError('BASE complete row identities')
    return sorted(rows,key=lambda r:r['sample_index'])


def auc_E(rows):
    p=np.asarray([r['selected_family_E_logit'] for r in rows if r['target']==1 and r['available']],np.float64);n=np.asarray([r['selected_family_E_logit'] for r in rows if r['target']==0 and r['available']],np.float64)
    if not len(p) or not len(n):return {'auc':None,'D_supported':len(n),'E_supported':len(p),'reason':'BOTH_CLASSES_REQUIRED'}
    return {'auc':sum(float((v>n).sum())+.5*float((v==n).sum()) for v in p)/(len(p)*len(n)),'D_supported':len(n),'E_supported':len(p),'E_direction':'POSITIVE_RAW_SELECTED_FAMILY_LOGIT','ties':'half'}


def reduce_metrics(rows,utility):
    result=utility.reduce_rows(rows)
    for family in (0,1):
        for role in ('FIT','DEV'):
            rr=[r for r in rows if r['family']==family and r['role']==role];m=result[f'family{family}_{role}'];d=m['D']['recall'];e=m['E']['recall'];u=sum(r['prediction']=='U' for r in rr)
            m.update(family=family,role=role,CM_D_E=[r[:2] for r in m['CM_D_E_by_D_E_U']],precisionD=m['D']['precision'],precisionE=m['E']['precision'],recallD=d,recallE=e,BR=None if d is None or e is None else (d+e)/2,AUC=auc_E(rr),U=u,U_rate=u/len(rr) if rr else None)
    return result


def gates(base,control,treat,utility):
    legacy=utility.source_gate(base,treat)
    def br(m):
        a=[m[f'family{f}_DEV'][c]['recall'] for f in (0,1) for c in ('D','E')];return None if any(v is None for v in a) else sum(a)/4
    a=br(control);b=br(treat);gain=None if a is None or b is None else (b-a)*100;matched=gain is not None and gain>=1
    return {'PASS':bool(legacy['PASS'] and matched),'historical_SOURCE_TREAT_vs_BASE_FROZEN_V239':legacy,'matched_TREAT_minus_CONTROL_mean_family_BR_gain_pp':gain,'matched_min_gain_pp':1.,'matched_PASS':matched,'Pallox80_gate_applied':False,'current_admission':False,'automatic_current_execution':False}


def support_summary(refs,data):
    out={}
    for arm,ref in refs.items():
        out[arm]={}
        for family in (0,1):
            for role in ('FIT','DEV'):
                ix=np.flatnonzero((data['family']==family)&(data['fit']==(role=='FIT')));all3=int((ref[ix,6:]==1).all(1).sum());n=len(ix)
                out[arm][f'family{family}_{role}']={'family':family,'role':role,'original_rows':n,'all3_reference_supported':all3,'all3_supported_fraction':all3/n if n else None,'per_view_supported':dict(zip(VIEWS,map(int,(ref[ix,6:]==1).sum(0)))),'reference_missing_does_not_change_model_U':True}
    return out


def paired_effects(control,treat):
    out={}
    for family in (0,1):
        for role in ('FIT','DEV'):
            ix=[i for i,r in enumerate(control) if r['family']==family and r['role']==role];a=[control[i] for i in ix];b=[treat[i] for i in ix];delta=np.asarray([y['selected_family_E_logit']-x['selected_family_E_logit'] for x,y in zip(a,b)],np.float64)
            out[f'family{family}_{role}']={'rows':len(ix),'prediction_changes':sum(x['prediction']!=y['prediction'] for x,y in zip(a,b)),'recoveries':sum(x['prediction']!=('E' if x['target'] else 'D') and y['prediction']==('E' if y['target'] else 'D') for x,y in zip(a,b)),'damages':sum(x['prediction']==('E' if x['target'] else 'D') and y['prediction']!=('E' if y['target'] else 'D') for x,y in zip(a,b)),'E_logit_delta_mean':float(delta.mean()) if len(delta) else None,'E_logit_delta_abs_max':float(np.abs(delta).max()) if len(delta) else None}
    return out


def run(entry_path):
    e,protocol,out=authorize(entry_path);utility=load_utility(e['inputs']['utility_code']);x,data,targetref,draws,eligible=load_payload(e,utility)
    out.mkdir(parents=False);eb=binding(entry_path);jwrite(out/'STARTED.json',{'entry':eb,'automatic_retry':False,'fixed_recipe':RECIPE})
    try:
        utility.configure_runtime();device=e['device']
        if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
        if device not in ('cuda','cpu'):raise ValueError('Declared device auto/cuda/cpu')
        models={}
        for arm in ARMS:
            torch.manual_seed(17);models[arm]=build_model().float().to(device)
        if utility.state_sha(models[CONTROL].state_dict())!=utility.state_sha(models[TREAT].state_dict()) or sum(p.numel() for p in models[CONTROL].parameters())!=1410:raise ValueError('Matched1410 seed17 initialization')
        bank,bank_support,bankrows=build_generic_banks(x,data);bankp=out/'GENERIC_BANKS_FROZEN_BEFORE_OPTIMIZERS.npz';np.savez(bankp,prototypes=bank,supported=bank_support,view=np.asarray(VIEWS),family=np.asarray([0,1],np.int8));bb=binding(bankp)
        bs=jwrite(out/'GENERIC_FIT_D_BANK_SUPPORT.json',{'completed':True,'rows':bankrows,'supervised_generic_not_no_reference':True,'full_FIT_D_self_inclusion':True,'no_DEV_or_target_bank_averaging':True})
        fit=np.flatnonzero(eligible);fit_map=np.full(7617,-1,np.int64);fit_map[fit]=np.arange(len(fit))
        genericFIT=generic_reference_inputs(fit,x,data['family'],data['available'],bank,bank_support)
        fit_inputs={CONTROL:torch.as_tensor(assemble_inputs(genericFIT,data['condition'][fit]),device=device),TREAT:torch.as_tensor(assemble_inputs(np.array(targetref[fit],copy=True),data['condition'][fit]),device=device)}
        with torch.no_grad():initial_zero={a:bool(torch.equal(models[a](fit_inputs[a]),torch.zeros((len(fit),2),device=device))) for a in ARMS}
        if not all(initial_zero.values()):raise ValueError('Both initial logits exactzero')
        iq=jwrite(out/'MATCHED_INITIAL_AND_FIT_INPUT_QA_BEFORE_OPTIMIZERS.json',{'completed':True,'PASS':True,'initial_model_sha256':utility.state_sha(models[CONTROL].state_dict()),'params':1410,'initial_zero_outputs_allFIT':initial_zero,'FIT_rows':len(fit),'same_draws_sha256':DRAW_SHA,'model_input_width':41,'condition32_original':True,'raw_query_latent_core_CP_or_logits_input':False,'targetref9_byteexact_cache':True,'new_DEV_head_output':0})
        ledger=out/'COMMON_DRAW_LEDGER1000.jsonl'
        with ledger.open('x',encoding='utf-8') as f:
            for step,ids in enumerate(draws,1):f.write(json.dumps({'update':step,'sample_indices':ids.tolist(),'draws_sha256':DRAW_SHA,'all_fit_available_known':True})+'\n')
        lb=binding(ledger);cps={};logs={};final={};initial={}
        for arm in ARMS:cps[arm],logs[arm],final[arm],initial[arm]=train_arm(arm,models[arm],fit_inputs[arm],data,draws,fit_map,utility,device,out,eb,bb)
        if initial[CONTROL]!=initial[TREAT]:raise ValueError('Same full initial state')
        frozen=[*cps.values(),*logs.values(),bb,bs,lb,e['code'],e['root_protocol'],e['implementation_protocol']]
        for b in frozen:checked(b)
        freeze=jwrite(out/'BOTH_FINAL1000_LOGS_BANKS_FROZEN_BEFORE_ANY_DEV_OUTPUT.json',{'completed':True,'PASS':True,'checkpoints':cps,'logs':logs,'generic_bank':bb,'generic_bank_support':bs,'draw_ledger':lb,'draws_sha256':DRAW_SHA,'code':e['code'],'root_protocol':e['root_protocol'],'implementation_protocol':e['implementation_protocol'],'inputs':e['inputs'],'verified_input_SHA_before_freeze':[k for k in e['inputs'] if k!='BASE_FROZEN_V239_CSV'],'BASE_CSV_bytes_or_interpretation_before_freeze':0,'new_DEV_head_outputs_before_freeze':0,'initial_QA':iq,'updates_each':1000})
        for m in models.values():m.eval().requires_grad_(False)
        genericall=generic_reference_inputs(np.arange(7617),x,data['family'],data['available'],bank,bank_support)
        if not np.array_equal(genericall[fit],genericFIT):raise ValueError('GenericFIT inputs stable')
        np.save(out/'GENERIC_REFERENCE_INPUTS_ALL7617.npy',genericall,allow_pickle=False)
        refs={CONTROL:genericall,TREAT:targetref};outputs={};rows_by_arm={};metrics={}
        with torch.no_grad():
            for arm in ARMS:
                logits=np.zeros((7617,2),np.float32)
                for first in range(0,7617,64):
                    ids=np.arange(first,min(first+64,7617));inp=assemble_inputs(np.array(refs[arm][ids],copy=True),data['condition'][ids]);logits[ids]=models[arm](torch.as_tensor(inp,device=device)).cpu().numpy()
                if not np.isfinite(logits).all():raise ValueError('Finite final saved logits')
                rows_by_arm[arm]=source_rows(data,logits);outputs[arm]=save_csv(out/(arm+'_ALL_SOURCE_ROWS.csv'),rows_by_arm[arm]);metrics[arm]=reduce_metrics(rows_by_arm[arm],utility)
                if utility.state_sha(models[arm].state_dict())!=final[arm]:raise ValueError('Final state unchanged')
        base=read_BASE_after_freeze(e['inputs']['BASE_FROZEN_V239_CSV'],data);bm=reduce_metrics(base,utility)
        sr=jwrite(out/'SOURCE_RESULTS.json',{'completed':True,'PASS':True,'BASE_FROZEN_V239':bm,'arms':metrics,'SOURCE_gate':gates(bm,metrics[CONTROL],metrics[TREAT],utility),'paired_effects':paired_effects(rows_by_arm[CONTROL],rows_by_arm[TREAT]),'reference_support':support_summary(refs,data),'SYN_MAIN_vs_YCB_associated_r_AUX_separate':True,'all3views_joint_no_best_view':True,'generic_FIT_D_self_inclusion':True,'current_admission':False,'reused_SOURCE_DEV_not_blind':True})
        for b in [*frozen,freeze]:checked(b)
        complete={'completed':True,'PASS':True,'schema':'V252_DIRECT41_REFERENCE_SOURCE_HEAD_PAIR_V1','entry':eb,'code':e['code'],'implementation_protocol':e['implementation_protocol'],'root_protocol':e['root_protocol'],'root_scope':e['root_scope'],'inputs':e['inputs'],'checkpoints':cps,'logs':logs,'outputs':outputs,'BASE_FROZEN_V239_CSV':e['inputs']['BASE_FROZEN_V239_CSV'],'summary':sr,'freeze':freeze,'generic_banks':bb,'generic_bank_support':bs,'generic_reference_inputs':binding(out/'GENERIC_REFERENCE_INPUTS_ALL7617.npy'),'target_reference_inputs':e['inputs']['V251_saved_reference_inputs'],'target_banks_provenance':e['inputs']['V251_saved_banks'],'draw_ledger':lb,'draws_sha256':DRAW_SHA,'initial_model_sha256':initial,'final_model_sha256':final,'optimizer_updates':{a:1000 for a in ARMS},'optimizer_states_each':4,'parameters_each':1410,'input_width':41,'threshold_raw_logit':0.,'device':device,'known_rows':7617,'FIT_known':5876,'DEV_known':1741,'original_rows':15040,'original_masked_S_U':7423,'new_DEV_head_outputs_before_bothfinalfreeze':0,'BASE_CSV_read_after_freeze':True,'generic_supervised_FIT_D_statistics_computed':True,'full_FIT_D_self_inclusion':True,'whole_SOURCE_nonpixel_labels_and_targetref9_loaded':True,'SOURCE_DEV_used_for_gradient_bankfit_or_selection':False,'old_core_CP_module_hidden_logits_or_preds_forward_inputs':0,'new_RGB_DINO_weights_GT_annotation_archives_current_held':0,'teacher_native_gold_created':False,'current_admission':False,'automatic_retry':False}
        jwrite(out/'COMPLETE.json',complete);print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(out/'COMPLETE.json')}))
    except BaseException:
        jwrite(out/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'partial_outputs_preserved':True,'traceback':traceback.format_exc()});raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);a=p.parse_args();run(a.entry)


if __name__=='__main__':main()
