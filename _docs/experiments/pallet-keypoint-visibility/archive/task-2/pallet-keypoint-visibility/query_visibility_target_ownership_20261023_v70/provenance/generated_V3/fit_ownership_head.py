#!/usr/bin/env python3
"""One locked source-only binary ownership pilot; no fitting without ENTRYLOCK.

GT exact V/E choose supervised rows and TRAIN pair lineage only. GT never
selects appearance-prototype sites or enters a model input. Original/composite
paired views are never supplied together as inference features.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,random,shutil,sys,time
import numpy as np
import torch
from torch.nn import functional as F

from ownership_head import TargetOwnershipHead,parameter_count,save_checkpoint,load_model
from input_prototype import input_prototypes

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('SYNTH','ORIGINAL_YCB','CONTROLLED_YCB','PALLOX')
PLAN=dict(candidate='OWNERSHIP_LOO_PAIR_V1',updates=2000,validation_every=100,batch_size=256,source_rows_per_batch=64,
    changed_pairs_per_update=32,pair_weight=.25,seed=20261023,optimizer='AdamW',lr=.001,weight_decay=.0001,
    BCE_class_weights='all1',threshold=.5,checkpoint_selection='earliest strict minimum unweighted4source knownVE macroNLL')
INPUT_KEYS=('x','rgb_native','sam','box_grid','prototype','prototype_info')
CACHE_KEYS=('x','rgb_native','sam','context','box_grid','target_index','sample_id','source','group_id','image_sha256',
    'global_id','global_source','global_image_sha256','dense_allowed','dense_valid','dense_exact_state')
LABEL_SEAL_SHA='45ffa631781e1efef4f5137393b3340d4fd91a1ca137486c8638dbc59fe83b01'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,j):Path(p).write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
def require(v,m):
    if not v:raise ValueError(m)


def known_ve(allowed,valid):
    a=np.asarray(allowed);v=np.asarray(valid)
    require(a.shape==v.shape and v.dtype==bool and np.isin(a,np.arange(1,64)).all(),'Partial compatible masks1..63 and bool valid required')
    V=v&(a==1);E=v&np.isin(a,[2,4,6])
    return V|E,V.astype(np.float32)


def source_macro_bce(logits,labels,source_codes):
    """All arguments here are already known V/E supervised rows only."""
    values=F.binary_cross_entropy_with_logits(logits,labels,reduction='none')
    return torch.stack([values[source_codes==s].mean() for s in torch.unique(source_codes)]).mean()


def pair_ranking_loss(logits,original_positions,controlled_positions):
    return F.softplus(-(logits[original_positions]-logits[controlled_positions])).mean()


def build_batch(rng,pools,pairs):
    selected=rng.integers(0,len(pairs),size=32)
    chunks=[rng.choice(pools['SYNTH'],64,replace=True),rng.choice(pools['PALLOX'],64,replace=True),
        pairs[selected,0],pairs[selected,1],rng.choice(pools['ORIGINAL_YCB'],32,replace=True),rng.choice(pools['CONTROLLED_YCB'],32,replace=True)]
    rows=np.concatenate(chunks);order=rng.permutation(256);inverse=np.empty(256,np.int64);inverse[order]=np.arange(256)
    return rows[order],inverse[np.arange(128,160)],inverse[np.arange(160,192)]


def load_cache(path,deny):
    with np.load(path,allow_pickle=False) as z:
        require(all(k in z.files for k in CACHE_KEYS),'Missing frozen cached source input/identity/GT key')
        d={k:z[k] for k in CACHE_KEYS}
    n=len(d['x']);t=len(d['global_id'])
    for k,s in dict(x=(n,65,9,9),rgb_native=(n,75,9,9),sam=(n,2,9,9),context=(n,64),box_grid=(n,4,9,9),
        target_index=(n,),sample_id=(n,),source=(n,),group_id=(n,),image_sha256=(n,),dense_allowed=(n,9,9),
        dense_valid=(n,9,9),dense_exact_state=(n,9,9)).items():require(d[k].shape==s,'Cache shape '+k)
    require(d['rgb_native'].dtype==np.uint8 and d['target_index'].dtype==np.int64,'Original native RGB/target grouping dtypes')
    require(set(d['source'].astype(str))==set(SOURCES),'Four fixed source domains required')
    require(not(set(d['image_sha256'].astype(str))&deny),'Private image SHA in source cache')
    require(len(set(d['sample_id']))==n and len(set(d['global_id']))==t,'Unique source identities')
    ti=d['target_index'];require(ti.min()>=0 and ti.max()<t,'Target grouping range')
    require(np.array_equal(d['global_image_sha256'][ti],d['image_sha256']) and np.array_equal(d['global_source'][ti],d['source']),'Each prototype group must be one current image/target')
    allowed=d['dense_allowed'][:,4,4];valid=d['dense_valid'][:,4,4]
    require(d['dense_valid'].dtype==bool and np.isin(allowed,np.arange(1,64)).all() and (allowed[~valid]==63).all(),'Native compatible mask contract')
    known,y=known_ve(allowed,valid);state=d['dense_exact_state'][:,4,4]
    require(np.array_equal(valid&(state==0),valid&(allowed==1)) and np.array_equal(valid&(state==1),valid&np.isin(allowed,[2,4,6])), 'V/E bitmask projection must match stored exact states')
    # ALL ROIs feed the GT-free prototype; do not filter on known/y/state first.
    proto,info,audit=input_prototypes(d['x'],d['sam'],d['box_grid'],d['context'],ti)
    d.update(prototype=proto,prototype_info=info,known_ve=known,y=y,center_allowed=allowed,center_valid=valid,center_state=state)
    return d,audit


def load_pairs(path,d):
    with Path(path).open(newline='') as f:raw=list(csv.DictReader(f))
    rows=[]
    for r in raw:
        i,j=int(r['original_index']),int(r['controlled_index'])
        require(0<=i<len(d['x']) and 0<=j<len(d['x']),'TRAIN pair row range')
        require(str(d['sample_id'][i])==r['original_sample_id'] and str(d['sample_id'][j])==r['controlled_sample_id'],'TRAIN pair identity mismatch')
        require(d['source'][i]=='ORIGINAL_YCB' and d['source'][j]=='CONTROLLED_YCB','Pair source lineage')
        require(d['known_ve'][i] and d['known_ve'][j] and d['y'][i]==1 and d['y'][j]==0,'Only known originalV to controlledE changed pairs')
        require(d['target_index'][i]!=d['target_index'][j] and d['image_sha256'][i]!=d['image_sha256'][j],'Original/composite prototypes must remain separate current images')
        rows.append((i,j))
    require(len(rows)==1781 and len(set(rows))==1781,'Frozen full TRAIN1781 changed-pair cohort required')
    return np.asarray(rows,np.int64)


def tensor_inputs(d,ix,device):
    result={k:torch.as_tensor(np.ascontiguousarray(d[k][ix]),dtype=torch.float32,device=device) for k in INPUT_KEYS}
    result['rgb_native']=result['rgb_native']/255
    return result


def preflight(entry_path,out):
    require(entry_path is not None,'Frozen root fit ENTRYLOCK required')
    entry=json.loads(Path(entry_path).read_text())
    require(entry['schema']=='V70_SOURCE_OWNERSHIP_FIT_ENTRYLOCK_V1','Fit entry schema')
    for k,value in dict(approved=True,source_only=True,current223_allowed=False,new_PnP_allowed=False).items():require(entry.get(k) is value,'Fit scope '+k)
    require(Path(out).resolve()==Path(entry['output_directory']).resolve() and not Path(out).exists(),'Preserve locked run output')
    require(Path(out).resolve().is_relative_to(ROOT),'V70 output only')
    require(set(entry['code'])=={'ownership_head.py','input_prototype.py','fit_ownership_head.py'},'All model/prototype/trainer code must be locked')
    for name,b in entry['code'].items():require(Path(b['path']).resolve()==ROOT/'model'/name and sha(b['path'])==b['sha256'],'Code changed '+name)
    require(set(entry['inputs'])=={'protocol','train','val','train_changed_pairs','source_label_contract','denylist'},'Only registered source fit inputs')
    for role,digest in (('train','8c7935a55ac9df911ca7670c718e94a466bce46fe245f82047e5f796b4bf27dc'),
                        ('val','386fa51f11427464ceed6c6c07e39c33f701aa60b3aac794a46616b42581e17d')):
        expected=ROOT.parent/'query_visibility_controlled_real_20261022_v66/joined_source_cache'/('CONTROLLED_PLUS_PALLOX_'+role.upper()+'.npz')
        require(Path(entry['inputs'][role]['path']).resolve()==expected and entry['inputs'][role]['sha256']==digest,'Unchanged frozen source cache required: '+role)
    label_binding=entry['inputs']['source_label_contract']
    require(Path(label_binding['path']).resolve()==ROOT/'contract/LABEL_COHORT_SEAL.json' and label_binding['sha256']==LABEL_SEAL_SHA,'Frozen source label/pair contract required')
    for b in entry['inputs'].values():require(sha(b['path'])==b['sha256'],'Fit input/provenance digest changed')
    label_seal=json.loads(Path(label_binding['path']).read_text());require(label_seal['completed'] is True and label_seal['GT_pair_index_not_model_input'] is True,'Complete GT/pair lineage audit required')
    pair_binding=entry['inputs']['train_changed_pairs']
    require(Path(pair_binding['path']).resolve()==ROOT/'contract/TRAIN_CHANGED_PAIR_INDEX.csv','TRAIN changed pairs only, never VAL or preserved pair list')
    require(any(Path(b['path']).resolve()==Path(pair_binding['path']).resolve() and b['sha256']==pair_binding['sha256'] for b in label_seal['artifacts']),'TRAIN pair index must be bound by label seal')
    protocol=json.loads(Path(entry['inputs']['protocol']['path']).read_text());require(protocol['training']==PLAN,'One fixed pilot plan; no tuning/sweep')
    require(protocol['source_only'] is True and protocol['current_inference_allowed'] is False,'Source-only fit protocol')
    require(protocol['model_input_whitelist']==list(INPUT_KEYS),'Frozen model input whitelist')
    require(protocol['prototype']['leave_one_entire_ROI_out'] is True and protocol['prototype']['mass_epsilon']==1e-6 and
        protocol['prototype']['positive_logit_threshold']==0 and protocol['prototype']['positive_comparison']=='>','Frozen GT-free predicted-foreground prototype recipe')
    require(entry['runtime']==dict(python='/home/minjae/anaconda3/bin/python',torch_prefix='2.5.1',device='cuda'),'Fixed runtime recipe')
    require(sys.executable==entry['runtime']['python'] and str(torch.__version__).startswith('2.5.1'),'Use registered Torch2.5 runtime')
    return entry,protocol


def fit(entry_path,out):
    entry,protocol=preflight(entry_path,out);bindings=entry['inputs']
    deny=set(json.loads(Path(bindings['denylist']['path']).read_text())['image_sha256']);require(len(deny)==223,'Complete evaluation exclusion list')
    tr,trp=load_cache(bindings['train']['path'],deny);va,vap=load_cache(bindings['val']['path'],deny)
    require(len(tr['x'])==31617 and len(va['x'])==8590,'Frozen source cache sizes')
    for key in ('sample_id','group_id','image_sha256','global_id','global_image_sha256'):
        require(not(set(tr[key].astype(str))&set(va[key].astype(str))),'TRAIN/VAL overlap '+key)
    pairs=load_pairs(bindings['train_changed_pairs']['path'],tr)
    pools={s:np.flatnonzero((tr['source']==s)&tr['known_ve']) for s in SOURCES};require(all(len(v) for v in pools.values()),'Nonempty four-source V/E pools required')
    source_code={s:i for i,s in enumerate(SOURCES)}
    random.seed(PLAN['seed']);np.random.seed(PLAN['seed']);torch.manual_seed(PLAN['seed']);torch.cuda.manual_seed_all(PLAN['seed']);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    device='cuda';model=TargetOwnershipHead().to(device);require(parameter_count(model)<100000,'Small component parameter budget')
    out=Path(out);out.mkdir(parents=True);(out/'code').mkdir()
    for name in entry['code']:shutil.copy2(ROOT/'model'/name,out/'code'/name)
    shutil.copy2(bindings['protocol']['path'],out/'PROTOCOL_AT_START.json')
    audit=dict(entry_lock=ref(entry_path),inputs=bindings,code=entry['code'],prototype_TRAIN=trp,prototype_VAL=vap,
        parameters=parameter_count(model),training_plan=PLAN,source_only=True,current223_used=False,PnP_calls=0,
        input_GT_fields=[],multiquery_set_required=True,model_output='Conditional exact pVisible_given_V_or_E; no causal O/self output',
        source_label_cohort_counts={s:dict(TRAIN_V=int(((tr['source']==s)&tr['known_ve']&(tr['y']==1)).sum()),TRAIN_E=int(((tr['source']==s)&tr['known_ve']&(tr['y']==0)).sum()),
            VAL_V=int(((va['source']==s)&va['known_ve']&(va['y']==1)).sum()),VAL_E=int(((va['source']==s)&va['known_ve']&(va['y']==0)).sum())) for s in SOURCES})
    write(out/'RUN_STARTED.json',audit);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(PLAN['seed']);best=float('inf');best_update=None;history=[];start=time.monotonic();loss_history=[]
    def validation():
        model.eval();zs=[]
        with torch.inference_mode():
            for b in range(0,len(va['x']),256):zs.append(model(**tensor_inputs(va,np.arange(b,min(len(va['x']),b+256)),device)).cpu().numpy())
        z=np.concatenate(zs).astype(np.float64);loss=np.logaddexp(0,z)-va['y']*z
        by={s:float(loss[(va['source']==s)&va['known_ve']].mean()) for s in SOURCES}
        require(np.isfinite(list(by.values())).all(),'Every source needs known V/E NLL support')
        return float(np.mean(list(by.values()))),by,z
    for update in range(1,2001):
        require(sha(bindings['protocol']['path'])==bindings['protocol']['sha256'],'Protocol changed during fit')
        model.train();ix,op,cp=build_batch(rng,pools,pairs)
        z=model(**tensor_inputs(tr,ix,device));y=torch.as_tensor(tr['y'][ix],device=device);sc=torch.as_tensor([source_code[s] for s in tr['source'][ix]],device=device)
        bce=source_macro_bce(z,y,sc);ranking=pair_ranking_loss(z,torch.as_tensor(op,device=device),torch.as_tensor(cp,device=device))
        total=bce+.25*ranking;require(torch.isfinite(total).item(),'Finite binary ownership loss required')
        optimizer.zero_grad(set_to_none=True);total.backward();optimizer.step();loss_history.append([float(bce.detach()),float(ranking.detach()),float(total.detach())])
        if update%100==0:
            metric,by,zv=validation();improved=metric<best
            if improved:
                best,best_update=metric,update;save_checkpoint(out/'BEST_OWNERSHIP_HEAD.pt',model,updates=update,selection_macro_knownVE_NLL=metric,
                    selection_by_source=by,threshold=.5,parameters=parameter_count(model),prototype_mass_epsilon=1e-6)
            row=dict(update=update,train_BCE=float(np.mean(np.asarray(loss_history)[:,0])),train_pair_rank=float(np.mean(np.asarray(loss_history)[:,1])),
                train_total=float(np.mean(np.asarray(loss_history)[:,2])),VAL_macro_knownVE_NLL=metric,by_source=by,saved_best=improved,seconds=time.monotonic()-start)
            history.append(row);write(out/'PROGRESS.json',row);print(json.dumps(row),flush=True);loss_history=[]
    model,payload=load_model(out/'BEST_OWNERSHIP_HEAD.pt',device);metric,by,z=validation()
    p=np.empty_like(z);positive=z>=0;p[positive]=1/(1+np.exp(-z[positive]));e=np.exp(z[~positive]);p[~positive]=e/(1+e)
    np.savez_compressed(out/'VALIDATION_SOURCE_OUTPUTS.npz',raw_logit_visible_given_VE=z,raw_probability_visible_given_VE=p,
        native_valid=va['center_valid'],allowed=va['center_allowed'],exact_GT=va['center_state'],known_VE=va['known_ve'],
        sample_id=va['sample_id'],source=va['source'],group_id=va['group_id'],image_sha256=va['image_sha256'],global_id=va['global_id'][va['target_index']],
        prototype_info=va['prototype_info'])
    write(out/'HISTORY.json',history)
    for b in list(entry['inputs'].values())+list(entry['code'].values()):require(sha(b['path'])==b['sha256'],'Frozen source fit binding changed')
    freeze=dict(**audit,completed=True,optimizer_updates=2000,best_update=best_update,best_source_macro_knownVE_NLL=best,
        checkpoint_sha256=sha(out/'BEST_OWNERSHIP_HEAD.pt'),threshold=.5,source_only_selection=True,
        checkpoint_selection_uses_quality_gates=False,current_inference_allowed=False,eval_data_used=False,PnP_used=False,
        seconds=time.monotonic()-start,outputs=[ref(out/'VALIDATION_SOURCE_OUTPUTS.npz'),ref(out/'HISTORY.json')])
    write(out/'MODEL_FREEZE.json',freeze);print(json.dumps(dict(completed=True,freeze=ref(out/'MODEL_FREEZE.json'))),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--entry-lock',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();fit(args.entry_lock,args.out)
