"""V183 source TRAIN crop-jitter/paired-query raw96/SAM ablation. --prepare uses invented CPU arrays only.

Cache bodies and fitting require a separately bound root entry. No old head,
encoder, reference crop, private/held data, or native visibility labels are used.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time, traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.nn import functional as F

H = Path(__file__).resolve().parent
V, P = H.parent, H.parent.parent
ORIGINAL = P/'query_visibility_local_support_raw_head_20261024_v181'
SCOPE = V/'ROOT_SCOPE_BEFORE_NEW_AUGMENTATION_OR_FIT.json'
SCOPE_SHA = 'd8394deb1c0a5844ecb987447391d7fd61ce5aa80d073cf33adf76ff385fcb95'
HEAD = ORIGINAL/'head_preparation/local_support_head.py'
HEAD_SHA = '14a40c4dd14d416c1648e62b408fd7acdf74f06d4113c69785bd737b45aa848b'
require_head_sha = hashlib.sha256(HEAD.read_bytes()).hexdigest()
if require_head_sha != HEAD_SHA: raise AssertionError('Frozen original head source SHA')
spec = importlib.util.spec_from_file_location('v181_local_support_head', HEAD)
head = importlib.util.module_from_spec(spec);sys.modules[spec.name]=head;spec.loader.exec_module(head)
PROTOCOL = H/'PROTOCOL_BEFORE_AUGMENTED_CACHE_OR_FIT_BODIES.json'
CHECKS = H/'GENERATED_INVENTED_CHECKS.json'
CHECK_SOURCE = H/'invented_local_support_checks.py'
POOL_REPAIR=ORIGINAL/'ROOT_DETERMINISTIC_POOL_RUNTIME_REPAIR_BEFORE_NEW_FIT.json'
POOL_REPAIR_SHA='fd9e6a750a6ffb86d6ad34741302c797f9f25760eef255bd659f9016b2110dcb'
POOL_CHECKS=ORIGINAL/'source_fit_preparation'/'GENERATED_POOL_RUNTIME_REPAIR_CHECKS.json'
POOL_CHECK_SOURCE=ORIGINAL/'source_fit_preparation'/'invented_pool_runtime_repair_checks.py'
RUN = V/'runs/C52_C53_JITTER_PAIRED_SOURCE_FIXED8000'
PYTHON = '/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
ARMS = [('C52_RAW96_JITTER_PAIRED_SOURCE',False),('C53_RAW96_SAM_JITTER_PAIRED_SOURCE',True)]
DOMAIN_ROWS = {'SYNTH_AUG65536':65536,'PUBLIC_TRAIN1480':1480,'CF3816':3816}
INPUT_DTYPES = {'rgb96':np.uint8,'valid96':np.bool_,'sam_mask96':np.bool_,'sam_available96':np.bool_,
                'original_query_available':np.bool_,'original_global_index':np.int64,
                'native_slot':np.int8,'query_xy':np.float64,'L':np.float64}
META_KEYS = ['row_id','original_global_index','native_slot','query_xy','L','original_query_available']
CACHE_SCHEMA = 'V181_LOCAL96_PIXEL_CACHE_WITH_SEPARATE_TARGETS_v1'
ORIGINAL_TRAIN_COMPLETE = ORIGINAL/'source_raw96_cache_v1/COMPLETE.json'
ORIGINAL_TRAIN_COMPLETE_SHA = 'b49f641551190aa27fd520acce51c5a1f0a9acc002b91c5b0b1709d4946cd3e4'
EVALUATION_ROWS = {'COMMON4096':4096,'SYNTH_VAL2048':2048}


def require(ok,message):
    if not bool(ok):raise AssertionError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def progress(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,allow_nan=False)+'\n');os.replace(tmp,path)


def check(binding):
    p=Path(binding['path']).resolve()
    require(any(p.is_relative_to(root) for root in [V,ORIGINAL,P/'query_visibility_source_same_query_causal_diagnosis_20261024_v182']),'Only root-bound V181/V182/V183 source cache paths')
    require(not any(s in str(p).lower() for s in ['_v112','current223','_v169','_v177','_v178','source_validation']),'Forbidden path')
    require(sha(p)==binding['sha256'],'Binding SHA '+str(p));return p


def object_sha(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous();h.update(str((str(a.dtype),tuple(a.shape))).encode());h.update(a.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for a in x:visit(a)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def cpu_copy(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return value


def validate_inputs(x,n):
    require(set(x)==set(INPUT_DTYPES)|{'row_id'},'Exact input cache keys; no targets/GT/features in input cache')
    for k,dtype in INPUT_DTYPES.items():require(x[k].dtype==dtype,'Input dtype '+k)
    shapes={'rgb96':(n,96,96,3),'valid96':(n,96,96),'sam_mask96':(n,96,96),'sam_available96':(n,96,96),
            'query_xy':(n,2)}
    for k in x:require(x[k].shape==shapes.get(k,(n,)),'Input shape '+k)
    require(x['row_id'].dtype.kind=='U' and len(set(x['row_id'].tolist()))==n,'Unique original row IDs')
    require(((x['native_slot']>=0)&(x['native_slot']<8)).all(),'Original input native metadata range')
    available=x['original_query_available']
    require(np.isfinite(x['query_xy'][available]).all() and np.isfinite(x['L'][available]).all() and (x['L'][available]>0).all(),'Available original q/L')
    return x


def validate_targets(y,available):
    n=len(available)
    require(set(y)=={'support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass'},'Exact separate targets')
    require(y['support_target'].dtype==np.int8 and y['support_target'].shape==(n,2) and np.isin(y['support_target'],[-1,0,1]).all(),'Independent support targets')
    require(y['owner_target'].dtype==np.int8 and y['owner_target'].shape==(n,) and np.isin(y['owner_target'],[-1,0,1,2]).all(),'Owner targets')
    soft=y['owner_soft'];require(soft.dtype==np.float64 and soft.shape==(n,3),'Original owner3 float64 mass')
    finite=np.isfinite(soft).all(1);require((finite|np.isnan(soft).all(1)).all(),'Missing soft rows allNaN')
    for k in ['owner_ambiguous_mass','owner_invalid_mass']:
        require(y[k].dtype==np.float64 and y[k].shape==(n,),'Unknown mass schema '+k)
        require(np.isfinite(y[k][finite]).all() and ((y[k][finite]>=0)&(y[k][finite]<=1)).all(),'Unknown mass range '+k)
    require(((soft[finite]>=0)&(soft[finite]<=1)).all() and (soft[finite].sum(1)<=1+1e-12).all(),'Known owner mass without row normalization')
    total=soft.sum(1)+y['owner_ambiguous_mass']+y['owner_invalid_mass']
    require(np.allclose(total[finite],1,atol=1e-12,rtol=0),'Known plus ambiguous plus invalid mass preserved')
    require((y['support_target'][~available]==-1).all() and (y['owner_target'][~available]==-1).all(),'Original U cannot be admitted to fit')
    hard=np.flatnonzero(y['owner_target']>=0)
    if len(hard):require(np.array_equal(soft[hard],np.eye(3,dtype=np.float64)[y['owner_target'][hard]]),'Hard owner is exact onehot, never soft argmax')
    return y


def pools_for(caches):
    s,p,c=(caches[k] for k in DOMAIN_ROWS)
    av=s['x']['original_query_available'];t=s['y']['support_target']
    support=[[np.flatnonzero(av&(t[:,j]==label)) for label in [1,0]] for j in range(2)]
    require(all(len(v)>0 for group in support for v in group),'Both classes for each fixed support')
    mass=s['y']['owner_soft'];eligible=av&np.isfinite(mass).all(1)
    weights=np.where(eligible[:,None],np.nan_to_num(mass,nan=0),0.0)
    require((weights.sum(0)>0).all(),'Nonzero original soft class masses')
    conditional=[weights[:,j]/weights[:,j].sum(dtype=np.float64) for j in range(3)]
    hard=[]
    for cache,classes in [(p,range(3)),(c,range(2))]:
        require((cache['y']['support_target']==-1).all(),'Public/CF owner labels never support targets')
        hard.append([np.flatnonzero(cache['x']['original_query_available']&(cache['y']['owner_target']==j)) for j in classes])
    require([sum(len(z) for z in group) for group in hard]==[253,314],'Public253/CF314 owner rows only')
    require([len(z) for z in hard[0]]==[53,52,148] and [len(z) for z in hard[1]]==[157,157],'Frozen public/CF class counts')
    return dict(support=support,conditional=conditional,public=hard[0],CF=hard[1])


def draw_step(rng,pools):
    draw={}
    for j,k in enumerate(['visible','foreign']):draw[k]=np.concatenate([rng.choice(a,16,replace=True) for a in pools['support'][j]])
    draw['synthetic_owner']=np.concatenate([rng.choice(len(p),16,replace=True,p=p) for p in pools['conditional']])
    draw['public_owner']=np.concatenate([rng.choice(a,16,replace=True) for a in pools['public']])
    draw['CF_owner']=np.concatenate([rng.choice(a,16,replace=True) for a in pools['CF']])
    require(sum(len(v) for v in draw.values())==192,'Fixed total192 draw rows')
    return draw


def five_losses(outputs):
    support,owner=outputs['support_logits'],outputs['owner_logits']
    require(support.shape==(192,2) and owner.shape==(192,3),'One192 forward, independent output schema')
    binary=torch.tensor([1.]*16+[0.]*16,dtype=torch.float32,device=support.device)
    owner3=torch.arange(3,device=owner.device).repeat_interleave(16)
    owner2=torch.arange(2,device=owner.device).repeat_interleave(16)
    losses=[F.binary_cross_entropy_with_logits(support[:32,0],binary),
            F.binary_cross_entropy_with_logits(support[32:64,1],binary),
            F.cross_entropy(owner[64:112],owner3),F.cross_entropy(owner[112:160],owner3),
            F.cross_entropy(owner[160:192],owner2)]
    return torch.stack(losses).mean(),losses


def validate_metadata(metadata,x,n,expected_split):
    require(len(metadata)==n,'All original rows metadata preserved')
    for i,r in enumerate(metadata):
        require(r['split']==expected_split,'Frozen source TRAIN/VAL separation')
        require(r['row_id']==str(x['row_id'][i]) and r['original_global_index']==int(x['original_global_index'][i]) and r['native_slot']==int(x['native_slot'][i]),'Cache metadata original identity')
        require(r['original_query_available']==bool(x['original_query_available'][i]),'Original query availability')
        q=np.asarray([np.nan,np.nan] if r['query_xy'] is None else r['query_xy'],np.float64)
        L=np.nan if r['L'] is None else r['L']
        require(np.array_equal(q,x['query_xy'][i],equal_nan=True),'Original q exact metadata join')
        require((np.isnan(L) and np.isnan(x['L'][i])) or L==x['L'][i],'Original L exact metadata join')


def load_domain(record,n,with_targets,expected_split):
    with np.load(check(record['inputs']),allow_pickle=False) as z:x={k:z[k] for k in z.files}
    validate_inputs(x,n)
    metadata=[json.loads(a) for a in check(record['metadata']).read_text().splitlines() if a]
    validate_metadata(metadata,x,n,expected_split)
    value=dict(x=x,record=record,metadata=metadata)
    if with_targets:
        with np.load(check(record['targets']),allow_pickle=False) as z:y={k:z[k] for k in z.files}
        value['y']=validate_targets(y,x['original_query_available'])
    return value


def validate_augmented_row(r,i,states):
    require(r['diagnostic_row_index']==i and r['original_global_index']==i,'Augmented diagnostic identity')
    if i<61440:
        original=i//15;state=i%15
        require(r['family']=='SELF_SCENE15' and r['state_index']==state and r['state_name']==states[state]['name'],'Exact SELF15 family/state enumeration')
        require(r['source_image_original_global_index']==original and r['anchor_original_global_index']==original and r['native_slot']==original%8,'SELF source/anchor metadata identity')
        require(r['dx']==states[state]['dx'] and r['dy']==states[state]['dy'] and r['L_scale']==states[state]['scale'],'Frozen perturbation metadata')
    else:
        j=i-61440;base=j//32;variant=(j%32)//8;native=j%8
        require(r['family']=='BEFORE_ANCHOR4' and r['native_slot']==native,'All BEFORE-anchor4 rows retained')
        require(r['anchor_original_global_index']==base*32+native and r['source_image_original_global_index']==base*32+variant*8+native,'BEFORE source and anchor IDs distinct/preserved')


def validate_augmented_metadata(metadata,x):
    require(len(metadata)==65536 and np.array_equal(x['original_global_index'],np.arange(65536,dtype=np.int64)),'Exact new augmented row indices0..65535')
    states=read(SCOPE)['training_synthetic']['states15'];require(len(states)==15,'Frozen15 input-only crop states')
    for i,r in enumerate(metadata):validate_augmented_row(r,i,states)
    require(int((~x['original_query_available']).sum())==1264,'Original augmented inputU1264 retained')


def merge_common_inputs(raw,sam):
    require(set(raw)=={'rgb96','valid96'}|set(META_KEYS),'COMMON raw exact8 keys')
    require(set(sam)=={'sam_mask96','sam_available96'}|set(META_KEYS),'COMMON SAM exact2+META6 keys')
    for key in META_KEYS:
        a,b=raw[key],sam[key]
        same=a.dtype==b.dtype and a.shape==b.shape and (np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b))
        require(same,'COMMON SAM/raw exact original identity '+key)
    return validate_inputs(dict(raw,sam_mask96=sam['sam_mask96'],sam_available96=sam['sam_available96']),len(raw['row_id']))


def load_common_evaluation(descriptor_binding):
    descriptor=read(check(descriptor_binding));require(descriptor['domain']=='COMMON4096','Root fixed COMMON descriptor')
    raw_complete=read(check(descriptor['raw_complete']));require(raw_complete['completed'] and raw_complete['PASS'],'Original COMMON complete metadata')
    require(raw_complete['inputs']==descriptor['raw_inputs8'] and raw_complete['metadata']==descriptor['metadata'] and raw_complete['targets']==descriptor['targets5'],'Original COMMON input/metadata/target bindings')
    with np.load(check(descriptor['raw_inputs8']),allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    with np.load(check(descriptor['sam_inputs']),allow_pickle=False) as z:sam={k:z[k] for k in z.files}
    x=merge_common_inputs(raw,sam)
    metadata=[json.loads(a) for a in check(descriptor['metadata']).read_text().splitlines() if a]
    validate_metadata(metadata,x,4096,'TRAIN')
    return dict(x=x,metadata=metadata,record=dict(inputs=descriptor['raw_inputs8'],metadata=descriptor['metadata']),SAM_input_binding=descriptor['sam_inputs'],descriptor=descriptor)


def gather_batch_arrays(caches,draw,use_sam):
    # Keep large immutable caches on CPU. Gather exactly192 rows, then move only
    # this batch to CUDA; no new random draws, normalization, or input selection.
    names=['visible','foreign','synthetic_owner','public_owner','CF_owner']
    domains=['SYNTH_AUG65536']*3+['PUBLIC_TRAIN1480','CF3816']
    keys=['rgb96','valid96']+(['sam_mask96','sam_available96'] if use_sam else [])
    return {k:np.concatenate([caches[d]['x'][k][draw[name]] for name,d in zip(names,domains)],axis=0) for k in keys}


def batch_input(caches,draw,use_sam):
    g={k:torch.as_tensor(a,device='cuda') for k,a in gather_batch_arrays(caches,draw,use_sam).items()}
    kwargs=dict(sam_mask96=g['sam_mask96'],sam_available=g['sam_available96']) if use_sam else {}
    return head.assemble_local_input(g['rgb96'],g['valid96'],use_sam=use_sam,**kwargs)


def save_outputs(model,cache,path,use_sam):
    x=cache['x'];n=len(x['row_id']);support=np.full((n,2),np.nan,np.float32);owner=np.full((n,3),np.nan,np.float32)
    available=x['original_query_available'];ids=np.flatnonzero(available)
    with torch.inference_mode():
        for start in range(0,len(ids),96):
            ii=ids[start:start+96];rgb=torch.as_tensor(x['rgb96'][ii],device='cuda');valid=torch.as_tensor(x['valid96'][ii],device='cuda')
            extra=dict(sam_mask96=torch.as_tensor(x['sam_mask96'][ii],device='cuda'),sam_available=torch.as_tensor(x['sam_available96'][ii],device='cuda')) if use_sam else {}
            inp=head.assemble_local_input(rgb,valid,use_sam=use_sam,**extra);a=model(inp);b=model(inp)
            require(all(torch.equal(a[k],b[k]) for k in a),'Final same-input repeat exact')
            support[ii]=a['support_logits'].cpu().numpy();owner[ii]=a['owner_logits'].cpu().numpy()
    sp=np.full((n,2),-1,np.int8);op=np.full(n,-1,np.int8);proxy=np.full(n,-1,np.int8)
    sp[ids]=(support[ids]>0).astype(np.int8);op[ids]=owner[ids].argmax(1).astype(np.int8)
    proxy[ids]=np.where(sp[ids,1]==1,1,np.where(sp[ids,0]==1,0,-1)).astype(np.int8)
    np.savez_compressed(path,raw_support_logits=support,raw_owner_logits=owner,support_prediction=sp,owner_prediction=op,
        regional_support_proxy_D0_E1_Uminus1=proxy,support_order=np.asarray(['visible_end_support','foreign_front_support']),
        owner_order=np.asarray(['TARGET','FOREIGN','BACKGROUND']),**{k:x[k] for k in META_KEYS})
    return bind(path)


def fit(entry_path):
    e=read(entry_path)
    require(e.get('approved_jitter_paired_source_fit') is True and e.get('source_only') is True,'Root fit entry required')
    for k,p in [('scope',SCOPE),('code',Path(__file__)),('head',HEAD),('protocol',PROTOCOL)]:require(e[k]==bind(p),'Entry '+k)
    p=read(PROTOCOL)
    for key,path in [('checks',CHECKS),('code',Path(__file__)),('head',HEAD),('check_source',CHECK_SOURCE)]:require(p[key]==bind(path),'Frozen preparation '+key)
    require(sha(SCOPE)==SCOPE_SHA and p['scope']==bind(SCOPE) and sha(HEAD)==HEAD_SHA,'Frozen source design/head')
    require(sha(POOL_REPAIR)==POOL_REPAIR_SHA and p['inherited_pool_runtime_repair']['scope']==bind(POOL_REPAIR),'Inherited exact fixed-pool repair')
    require(p['inherited_pool_runtime_repair']['CUDA_checks']==bind(POOL_CHECKS) and read(POOL_CHECKS)['PASS'],'Inherited strict CUDA backward checks')
    require(Path(sys.executable).resolve()==Path(PYTHON).resolve() and torch.__version__=='2.1.1+cu118' and np.__version__=='1.26.4' and torch.cuda.is_available(),'Pinned CUDA runtime')
    require(e['original_training_cache_complete']==p['unchanged_public_CF_cache_complete']==bind(ORIGINAL_TRAIN_COMPLETE) and sha(ORIGINAL_TRAIN_COMPLETE)==ORIGINAL_TRAIN_COMPLETE_SHA,'Unchanged public/CF cache authority')
    original=read(ORIGINAL_TRAIN_COMPLETE)
    require(original['completed'] and original['PASS'] and original['source_only'],'Original source TRAIN cache complete metadata')
    augmentation=read(check(e['augmentation_cache_complete']))
    require(augmentation['completed'] and augmentation['PASS'] and augmentation['rows65536']==65536,'Root-composed complete all65536 input rows')
    for key in ['inputs','targets','metadata']:require(isinstance(augmentation[key],dict),'Augmented cache binding '+key)
    require(not RUN.exists(),'Existing attempt retained, no automatic retry');RUN.mkdir(parents=True)
    write(RUN/'STARTED.json',dict(completed=False,PID=os.getpid(),entry=bind(entry_path),code=bind(__file__),head=bind(HEAD),protocol=bind(PROTOCOL)))
    start=time.monotonic()
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        caches={'SYNTH_AUG65536':load_domain(augmentation,65536,True,'TRAIN')}
        validate_augmented_metadata(caches['SYNTH_AUG65536']['metadata'],caches['SYNTH_AUG65536']['x'])
        for name,n in [('PUBLIC_TRAIN1480',1480),('CF3816',3816)]:caches[name]=load_domain(original['domains'][name],n,True,'SOURCE_DEVELOPMENT')
        require([int(caches[k]['x']['original_query_available'].sum()) for k in DOMAIN_ROWS]==[64272,1286,3456],'All inputavailability unchanged')
        require((caches['SYNTH_AUG65536']['y']['owner_target']==-1).all(),'Synthetic owner remains raw softmass only')
        pools=pools_for(caches)
        write(RUN/'FIT_INPUT_VALIDATION.json',dict(completed=True,PASS=True,rows=DOMAIN_ROWS,input_available=[64272,1286,3456],inputU=[1264,194,360],support_counts=[[len(a) for a in g] for g in pools['support']],public253=[53,52,148],CF314=[157,157],synthetic_known_owner_mass=[float(np.nansum(caches['SYNTH_AUG65536']['y']['owner_soft'][caches['SYNTH_AUG65536']['x']['original_query_available'],j])) for j in range(3)],CPU_cache_batch_gather_only=True,unknown_mass_not_imputed=True,partial_known_mass_preserved=True,old_synthetic_TRAIN4096_loaded_for_fit=False))
        pending={};initial=None;common_draw=None
        for arm,use_sam in ARMS:
            directory=RUN/arm;directory.mkdir();torch.manual_seed(1810);torch.cuda.manual_seed_all(1810)
            model=head.Raw96SupportHead().cuda().float();ish=object_sha(model.state_dict());require(sum(t.numel() for t in model.parameters())==115157,'Frozen architecture')
            if initial is None:initial=ish
            else:require(ish==initial,'Same initialization both arms')
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);rng=np.random.default_rng(1810);dh=hashlib.sha256()
            logpath=directory/'ALL8000_DRAWS_AND_LOSSES.jsonl'
            with logpath.open('x') as log:
                for update in range(1,8001):
                    draw=draw_step(rng,pools)
                    for key in draw:dh.update(key.encode());dh.update(draw[key].astype('<i8').tobytes())
                    inp=batch_input(caches,draw,use_sam);opt.zero_grad(set_to_none=True);loss,components=five_losses(model(inp))
                    require(torch.isfinite(loss),'Finite total loss');loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);require(torch.isfinite(norm),'Finite gradient');opt.step()
                    require(all(torch.isfinite(t).all() for t in model.parameters()),'Finite updated state')
                    log.write(json.dumps(dict(update=update,draw_indices={k:v.tolist() for k,v in draw.items()},five_losses=[float(v.detach()) for v in components],total_loss=float(loss.detach()),grad_norm=float(norm),cumulative_draw_sha=dh.hexdigest()),allow_nan=False)+'\n')
                    if update%250==0:
                        item=dict(arm=arm,update=update,elapsed_seconds=time.monotonic()-start,PID=os.getpid());progress(RUN/'PROGRESS.json',item);print(json.dumps(item),flush=True)
            draw_sha=dh.hexdigest()
            if common_draw is None:common_draw=draw_sha
            else:require(draw_sha==common_draw,'All8000 exact same domain/class draws')
            final=object_sha(model.state_dict());checkpoint=directory/'FINAL_CHECKPOINT.pt'
            torch.save(dict(state_dict=cpu_copy(model.state_dict()),optimizer_state=cpu_copy(opt.state_dict()),update=8000,seed=1810,arm=arm,use_sam=use_sam,protocol=bind(PROTOCOL)),checkpoint)
            model.eval().requires_grad_(False);pending[arm]=dict(model=model,use_sam=use_sam,updates=8000,initial_state_sha=ish,final_state_sha=final,draw_sha=draw_sha,checkpoint=bind(checkpoint),step_losses=bind(logpath));del opt
        freeze=RUN/'ALL2_FROZEN_BEFORE_SAVED_OUTPUTS_OR_EVALUATION_ACCESS.json'
        write(freeze,dict(completed=True,checkpoints={a:v['checkpoint'] for a,v in pending.items()},same_initial_state_sha=initial,same_draw_sha=common_draw,validation_cache_or_targets_read=False,common_diagnostic_inputs_or_targets_read=False,saved_outputs_emitted=False))
        # Open evaluation receipt/input bodies ONLY after both final8000 states.
        evaluation=read(check(e['evaluation_cache_complete']))
        require(evaluation['completed'] and evaluation['PASS'] and evaluation['cache_contract_schema']==CACHE_SCHEMA and set(evaluation['domains'])=={'SYNTH_VAL2048'},'Fixed original VAL evaluation cache')
        evals={'COMMON4096':load_common_evaluation(e['common_evaluation_descriptor']),'SYNTH_VAL2048':load_domain(evaluation['domains']['SYNTH_VAL2048'],2048,False,'VALIDATION')}
        require(int((~evals['COMMON4096']['x']['original_query_available']).sum())==96 and int((~evals['SYNTH_VAL2048']['x']['original_query_available']).sum())==8,'Fixed evalinputU96/8')
        for arm,item in pending.items():
            model=item.pop('model');item['outputs']={}
            for name,cache in list(caches.items())+list(evals.items()):item['outputs'][name]=save_outputs(model,cache,RUN/arm/f'FINAL_{name}_OUTPUTS.npz',item['use_sam'])
            require(object_sha(model.state_dict())==item['final_state_sha'],'Frozen state unchanged');del model
        for cache in list(caches.values())+list(evals.values()):
            for key in ['inputs','metadata']+(['targets'] if 'y' in cache else []):check(cache['record'][key])
        check(evals['COMMON4096']['SAM_input_binding'])
        write(RUN/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,code=bind(__file__),head=bind(HEAD),protocol=bind(PROTOCOL),entry=bind(entry_path),augmentation_cache_complete=e['augmentation_cache_complete'],original_training_cache_complete=e['original_training_cache_complete'],evaluation_cache_complete=e['evaluation_cache_complete'],common_evaluation_descriptor=e['common_evaluation_descriptor'],all2_freeze=bind(freeze),all2_frozen_before_outputs_or_evaluation_access=True,arms=pending,same_initial_state_sha=initial,same_draw_sha=common_draw,updates_each=8000,total_updates=16000,automatic_retry=False,seconds=time.monotonic()-start,selected_model=None,adopted=False,fully_solved=False,all_original_and_augmented_qL_native_inputU_preserved=True,evaluation_targets_read_by_trainer=False,private_or_publicheld_access=0,native_visibility_or_quality_relabel=0))
    except BaseException:
        write(RUN/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise


def prepare():
    require(sha(SCOPE)==SCOPE_SHA and sha(HEAD)==HEAD_SHA,'Frozen scope/head source SHA')
    require(sha(POOL_REPAIR)==POOL_REPAIR_SHA,'Inherited fixed-pool runtime repair scope')
    pool=read(POOL_CHECKS);require(pool['PASS'] and pool['new_head_source']==bind(HEAD) and pool['code']==bind(POOL_CHECK_SOURCE),'Inherited strict CPU/CUDA fixed-pool checks')
    require(Path(sys.executable).resolve()==Path(PYTHON).resolve() and torch.__version__=='2.1.1+cu118' and np.__version__=='1.26.4','Pinned CPU preparation runtime')
    require(sha(ORIGINAL_TRAIN_COMPLETE)==ORIGINAL_TRAIN_COMPLETE_SHA,'Original immutable public/CF cache metadata')
    from invented_local_support_checks import run_checks
    checks=run_checks(sys.modules[__name__])
    write(CHECKS,dict(completed=True,PASS=True,code=bind(__file__),head=bind(HEAD),check_source=bind(CHECK_SOURCE),checks=checks,inherited_pool_runtime_checks=bind(POOL_CHECKS),actual_raw_cache_label_or_weight_bodies_read=0,CUDA_initialized=torch.cuda.is_initialized()))
    protocol=dict(completed=True,prepared_only=True,scope=bind(SCOPE),code=bind(__file__),head=bind(HEAD),checks=bind(CHECKS),check_source=bind(CHECK_SOURCE),
        inherited_pool_runtime_repair=dict(scope=bind(POOL_REPAIR),CUDA_checks=bind(POOL_CHECKS),check_source=bind(POOL_CHECK_SOURCE)),
        unchanged_public_CF_cache_complete=bind(ORIGINAL_TRAIN_COMPLETE),arms=[dict(name=a,use_sam=b) for a,b in ARMS],seed=1810,steps_each=8000,parameters=115157,
        optimizer=dict(name='AdamW',lr=3e-4,weight_decay=1e-4,clip=5),runtime=dict(executable=PYTHON,torch='2.1.1+cu118',numpy='1.26.4',FP32=True,AMP=False,TF32=False,deterministic=True),
        draws=dict(visible=[16,16],foreign=[16,16],synthetic_owner=[16,16,16],public_owner=[16,16,16],CF_owner=[16,16],total192=True,replacement=True),
        loss='Unchanged V181 arithmetic mean of five balanced sample means; synthetic owner class-conditional draws use original raw partialknown classmass, never per-row normalized',
        training_domains=DOMAIN_ROWS,augmented_U=1264,unchanged_public_CF_U=[194,360],training_cache='Root-composed simple completion with rows65536/input10/target5/metadata/scene_metadata; actual root entry binds exact completed SHA',
        CPU_memory='Load each NPZ member once without extra copy, retain immutable CPU arrays; gather selected192 rows per update then transfer to CUDA. Never upload all65536 pixels to GPU.',
        evaluation_domains=EVALUATION_ROWS,common_eval_U=96,VAL_eval_U=8,all2_freeze_before_any_outputs_or_evaluation_receipt_or_input_body=True,evaluation_targets_never_loaded_by_trainer=True,common_evaluation='After freeze only: root descriptor raw8 + SAM2/META6 exactmerge including NaN identity. No new crop sampling or SAMforward. Targets5 binding preserved, never neural inputs.',
        raw_input='Original saveduint8RGB/255,valid,constantXY and optional frozen same-grid SAMmask/availability; RAW SAMchannelszero. q/L/native/GT/targets never neural inputs.',
        support_order=['visible_end_support','foreign_front_support'],owner_order=['TARGET','FOREIGN','BACKGROUND'],support_decision='strict>0 each independently; originalinputU first/rawNaN',owner_decision='firsttie argmax; inputU first',
        regional_proxy='foreignYES->E elsevisibleYES->D elseU; support diagnostic, not native physical gold',new_old_checkpoint_encoder_or_crop_sampler=False,output=str(RUN),selected_model=None,adopted=False,fully_solved=False)
    write(PROTOCOL,protocol)
    template=dict(approved_jitter_paired_source_fit=False,source_only=True,scope=bind(SCOPE),code=bind(__file__),head=bind(HEAD),protocol=bind(PROTOCOL),augmentation_cache_complete=None,original_training_cache_complete=bind(ORIGINAL_TRAIN_COMPLETE),evaluation_cache_complete=None,common_evaluation_descriptor=None)
    entry=H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(entry,template)
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),head=bind(HEAD),protocol=bind(PROTOCOL),checks=bind(CHECKS),entry_template=bind(entry),inherited_pool_runtime_CUDA_checks=bind(POOL_CHECKS),cache_dependencies_pending=True,actual_fit_performed=False,actual_raw_cache_label_or_weight_bodies_read=0))
    print(json.dumps(dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),preparation=bind(H/'PREPARATION_COMPLETE.json'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--entry',type=Path);args=parser.parse_args()
    prepare() if args.prepare else fit(args.entry)
