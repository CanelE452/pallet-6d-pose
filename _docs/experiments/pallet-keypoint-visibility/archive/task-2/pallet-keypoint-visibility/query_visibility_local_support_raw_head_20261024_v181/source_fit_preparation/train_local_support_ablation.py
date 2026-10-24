"""V181 fixed raw96/SAM ablation. --prepare uses invented CPU arrays only.

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
SCOPE = V/'ROOT_SCOPE_BEFORE_NEW_RAW_OR_SAM_CACHE_OR_FIT.json'
SCOPE_SHA = '8cd21d5eb4caf9663927859ead0cbcbd823e701fd4a2164a66aa765e046bfcdc'
HEAD = V/'head_preparation/local_support_head.py'
spec = importlib.util.spec_from_file_location('v181_local_support_head', HEAD)
head = importlib.util.module_from_spec(spec);sys.modules[spec.name]=head;spec.loader.exec_module(head)
PROTOCOL = H/'PROTOCOL_BEFORE_NEW_RAW_CACHE_OR_FIT_BODIES.json'
CHECKS = H/'GENERATED_INVENTED_CHECKS.json'
CHECK_SOURCE = H/'invented_local_support_checks.py'
POOL_REPAIR=V/'ROOT_DETERMINISTIC_POOL_RUNTIME_REPAIR_BEFORE_NEW_FIT.json'
POOL_REPAIR_SHA='fd9e6a750a6ffb86d6ad34741302c797f9f25760eef255bd659f9016b2110dcb'
POOL_CHECKS=H/'GENERATED_POOL_RUNTIME_REPAIR_CHECKS.json'
POOL_CHECK_SOURCE=H/'invented_pool_runtime_repair_checks.py'
RUN = V/'runs/C50_C51_LOCAL_SUPPORT_FIXED8000_v2'
PYTHON = '/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
ARMS = [('C50_RAW96_DUAL_SUPPORT_OWNER_AUX',False),('C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX',True)]
DOMAIN_ROWS = {'SYNTH_TRAIN4096':4096,'PUBLIC_TRAIN1480':1480,'CF3816':3816}
INPUT_DTYPES = {'rgb96':np.uint8,'valid96':np.bool_,'sam_mask96':np.bool_,'sam_available96':np.bool_,
                'original_query_available':np.bool_,'original_global_index':np.int64,
                'native_slot':np.int8,'query_xy':np.float64,'L':np.float64}
META_KEYS = ['row_id','original_global_index','native_slot','query_xy','L','original_query_available']
CACHE_SCHEMA = 'V181_LOCAL96_PIXEL_CACHE_WITH_SEPARATE_TARGETS_v1'


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
    require(p.is_relative_to(V),'Cache/entry binding must be under frozen V181')
    require(not any(s in str(p).lower() for s in ['_v112','current223','_v169','_v178']),'Forbidden path')
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


def load_domain(record,n,with_targets,expected_split):
    with np.load(check(record['inputs']),allow_pickle=False) as z:x={k:z[k].copy() for k in z.files}
    validate_inputs(x,n)
    metadata=[json.loads(a) for a in check(record['metadata']).read_text().splitlines() if a]
    require(len(metadata)==n,'All original rows metadata preserved')
    for i,r in enumerate(metadata):
        require(r['split']==expected_split,'Frozen source TRAIN/VAL separation')
        require(r['row_id']==str(x['row_id'][i]) and r['original_global_index']==int(x['original_global_index'][i]) and r['native_slot']==int(x['native_slot'][i]),'Cache metadata original identity')
        require(r['original_query_available']==bool(x['original_query_available'][i]),'Original query availability')
        q=np.asarray([np.nan,np.nan] if r['query_xy'] is None else r['query_xy'],np.float64)
        L=np.nan if r['L'] is None else r['L']
        require(np.array_equal(q,x['query_xy'][i],equal_nan=True),'Original q exact metadata join')
        require((np.isnan(L) and np.isnan(x['L'][i])) or L==x['L'][i],'Original L exact metadata join')
    value=dict(x=x,record=record)
    if with_targets:
        with np.load(check(record['targets']),allow_pickle=False) as z:y={k:z[k].copy() for k in z.files}
        value['y']=validate_targets(y,x['original_query_available'])
    return value


def gpu_pixels(cache):
    return {k:torch.as_tensor(cache['x'][k],device='cuda') for k in ['rgb96','valid96','sam_mask96','sam_available96']}


def batch_input(gpu,draw,use_sam):
    names=['visible','foreign','synthetic_owner','public_owner','CF_owner']
    domains=['SYNTH_TRAIN4096']*3+['PUBLIC_TRAIN1480','CF3816']
    gathered={k:[] for k in ['rgb96','valid96','sam_mask96','sam_available96']}
    for key,domain in zip(names,domains):
        ix=torch.as_tensor(draw[key],device='cuda',dtype=torch.int64)
        for k in gathered:gathered[k].append(gpu[domain][k][ix])
    g={k:torch.cat(v,0) for k,v in gathered.items()}
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
    require(e.get('approved_local_support_source_fit') is True and e.get('source_only') is True,'Root fit entry required')
    for k,p in [('scope',SCOPE),('code',Path(__file__)),('head',HEAD),('protocol',PROTOCOL)]:require(e[k]==bind(p),'Entry '+k)
    p=read(PROTOCOL);require(p['checks']==bind(CHECKS) and p['code']==bind(__file__) and p['head']==bind(HEAD) and p['check_source']==bind(CHECK_SOURCE),'Frozen preparation')
    require(sha(SCOPE)==SCOPE_SHA and p['scope']==bind(SCOPE),'Original prospective scope unchanged')
    require(sha(POOL_REPAIR)==POOL_REPAIR_SHA and e['pool_runtime_repair']==p['pool_runtime_repair']['scope']==bind(POOL_REPAIR),'Exact root runtime-only pool repair')
    require(p['pool_runtime_repair']['CUDA_checks']==bind(POOL_CHECKS) and p['pool_runtime_repair']['check_source']==bind(POOL_CHECK_SOURCE) and read(POOL_CHECKS)['PASS'],'Pool-equivalence checks frozen')
    require(Path(sys.executable).resolve()==Path(PYTHON).resolve() and torch.__version__=='2.1.1+cu118' and np.__version__=='1.26.4' and torch.cuda.is_available(),'Pinned CUDA runtime')
    complete=read(check(e['training_cache_complete']))
    require(complete['completed'] and complete['PASS'] and complete['source_only'] and complete['cache_contract_schema']==CACHE_SCHEMA,'Frozen source cache complete')
    require(set(complete['domains'])==set(DOMAIN_ROWS),'Exact three training domains')
    require(not RUN.exists(),'Existing attempt retained, no automatic retry');RUN.mkdir(parents=True)
    write(RUN/'STARTED.json',dict(completed=False,PID=os.getpid(),entry=bind(entry_path),code=bind(__file__),head=bind(HEAD),protocol=bind(PROTOCOL)))
    start=time.monotonic()
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        caches={k:load_domain(complete['domains'][k],n,True,'TRAIN' if k=='SYNTH_TRAIN4096' else 'SOURCE_DEVELOPMENT') for k,n in DOMAIN_ROWS.items()}
        require([int(caches[k]['x']['original_query_available'].sum()) for k in DOMAIN_ROWS]==[4016,1286,3456],'Original input availability preserved')
        pools=pools_for(caches);gpu={k:gpu_pixels(v) for k,v in caches.items()}
        write(RUN/'FIT_INPUT_VALIDATION.json',dict(completed=True,PASS=True,rows=DOMAIN_ROWS,input_available=[4016,1286,3456],support_counts=[[len(a) for a in g] for g in pools['support']],public253=[53,52,148],CF314=[157,157],synthetic_known_owner_mass=[float(np.nansum(caches['SYNTH_TRAIN4096']['y']['owner_soft'][caches['SYNTH_TRAIN4096']['x']['original_query_available'],j])) for j in range(3)],unknown_mass_not_imputed=True,partial_known_mass_preserved=True))
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
                    inp=batch_input(gpu,draw,use_sam);opt.zero_grad(set_to_none=True);loss,components=five_losses(model(inp))
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
        freeze=RUN/'ALL2_FROZEN_BEFORE_SAVED_OUTPUTS_OR_VAL_ACCESS.json'
        write(freeze,dict(completed=True,checkpoints={a:v['checkpoint'] for a,v in pending.items()},same_initial_state_sha=initial,same_draw_sha=common_draw,validation_cache_or_targets_read=False,saved_outputs_emitted=False))
        del gpu
        # Evaluation inputs are first opened only after the two final heads freeze.
        evaluation=read(check(e['evaluation_cache_complete']))
        require(evaluation['completed'] and evaluation['PASS'] and evaluation['cache_contract_schema']==CACHE_SCHEMA and set(evaluation['domains'])=={'SYNTH_VAL2048'},'Fixed source VAL evaluation inputs only')
        eval_cache=load_domain(evaluation['domains']['SYNTH_VAL2048'],2048,False,'VALIDATION')
        for arm,item in pending.items():
            model=item.pop('model');item['outputs']={}
            for name,cache in list(caches.items())+[('SYNTH_VAL2048',eval_cache)]:item['outputs'][name]=save_outputs(model,cache,RUN/arm/f'FINAL_{name}_OUTPUTS.npz',item['use_sam'])
            require(object_sha(model.state_dict())==item['final_state_sha'],'Frozen state unchanged');del model
        # Rehash only the separately root-admitted cache/input files and metadata.
        for receipt in [complete,evaluation]:
            for record in receipt['domains'].values():
                for k in ['inputs','targets','metadata']:
                    if k in record:check(record[k])
        write(RUN/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,code=bind(__file__),head=bind(HEAD),protocol=bind(PROTOCOL),entry=bind(entry_path),training_cache_complete=e['training_cache_complete'],evaluation_cache_complete=e['evaluation_cache_complete'],all2_freeze=bind(freeze),all2_frozen_before_outputs_or_VAL_access=True,arms=pending,same_initial_state_sha=initial,same_draw_sha=common_draw,updates_each=8000,total_updates=16000,automatic_retry=False,seconds=time.monotonic()-start,selected_model=None,adopted=False,fully_solved=False,original_qL_native_and_inputU_preserved=True,private_or_publicheld_access=0,native_visibility_or_quality_relabel=0))
    except BaseException:
        write(RUN/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise


def prepare():
    require(sha(SCOPE)==SCOPE_SHA,'Frozen scope SHA')
    require(sha(POOL_REPAIR)==POOL_REPAIR_SHA,'Root runtime repair scope')
    pool=read(POOL_CHECKS);require(pool['PASS'] and pool['new_head_source']==bind(HEAD) and pool['code']==bind(POOL_CHECK_SOURCE),'Invented CPU/CUDA pool checks completed')
    require(Path(sys.executable).resolve()==Path(PYTHON).resolve() and torch.__version__=='2.1.1+cu118' and np.__version__=='1.26.4','Pinned CPU preparation runtime')
    from invented_local_support_checks import run_checks
    checks=run_checks(sys.modules[__name__])
    write(CHECKS,dict(completed=True,PASS=True,code=bind(__file__),head=bind(HEAD),check_source=bind(CHECK_SOURCE),checks=checks,pool_runtime_checks=bind(POOL_CHECKS),actual_raw_cache_label_or_weight_bodies_read=0,CUDA_initialized=torch.cuda.is_initialized()))
    protocol=dict(completed=True,prepared_only=True,scope=bind(SCOPE),code=bind(__file__),head=bind(HEAD),checks=bind(CHECKS),check_source=bind(CHECK_SOURCE),
        pool_runtime_repair=dict(scope=bind(POOL_REPAIR),CUDA_checks=bind(POOL_CHECKS),check_source=bind(POOL_CHECK_SOURCE),operator='Fixed encoder12x12→AvgPool2d4stride4→3x3; oldAdaptiveCPU/GPU forward byteequal and strictCUDA backward verified; zero parameters/statekeys changed'),
        arms=[dict(name=a,use_sam=b) for a,b in ARMS],seed=1810,steps_each=8000,parameters=115157,optimizer=dict(name='AdamW',lr=3e-4,weight_decay=1e-4,clip=5),
        runtime=dict(executable=PYTHON,torch='2.1.1+cu118',numpy='1.26.4',FP32=True,AMP=False,TF32=False,deterministic=True),
        draws=dict(visible=[16,16],foreign=[16,16],synthetic_owner=[16,16,16],public_owner=[16,16,16],CF_owner=[16,16],total192=True,replacement=True),
        loss='Arithmetic mean of five domain-specific balanced sample means; soft3 class-conditional draws preserve partial known mass and ignore unknown mass without per-row renormalization',
        cache_contract_schema=CACHE_SCHEMA,training_domains=DOMAIN_ROWS,evaluation_domain={'SYNTH_VAL2048':2048},
        cache_complete_dependencies='PENDING: separately bound root-admitted A cache receipts required in actual entry; no actual cache opened during preparation',
        all2_freeze_before_outputs_or_VAL_access=True,raw_input='Originaluint8 RGB/255,valid,constantXY;SAMrawlogit exact96bilinear>0 and pixelwiseavailability; RAWonlySAMchannels0',
        support_order=['visible_end_support','foreign_front_support'],owner_order=['TARGET','FOREIGN','BACKGROUND'],
        support_decision='Eachlogit>0 independently; originalinputU first',owner_decision='argmax; originalinputU first',
        regional_proxy='foreign>0→E else visible>0→D else U; regional support diagnostic only, not certified physical native visibility',
        no_qL_native_GT_reference_or_labels_in_forward=True,sourceVAL_evaluation_only=True,no_old_checkpoint_or_encoder=True,output=str(RUN),selected_model=None,fully_solved=False)
    write(PROTOCOL,protocol)
    template=dict(approved_local_support_source_fit=False,source_only=True,scope=bind(SCOPE),pool_runtime_repair=bind(POOL_REPAIR),code=bind(__file__),head=bind(HEAD),protocol=bind(PROTOCOL),training_cache_complete=None,evaluation_cache_complete=None)
    write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),head=bind(HEAD),protocol=bind(PROTOCOL),checks=bind(CHECKS),pool_runtime_repair=bind(POOL_REPAIR),pool_runtime_CUDA_checks=bind(POOL_CHECKS),cache_dependencies_pending=True,actual_fit_performed=False,actual_raw_cache_label_or_weight_bodies_read=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--entry',type=Path);args=parser.parse_args()
    prepare() if args.prepare else fit(args.entry)
