"""V222 continued SOURCE pair. No payload/model/probe/fit without explicit root entry.

Same frozen C65 initialization/optimizer reset/common source recipe. Treatment only adds
controlled expected-finite-part weak BCE; this is not native point gold or an isolated
addressability causal test. Both new final states precede every new SOURCE DEV output.
"""
import argparse, ast, hashlib, importlib.util, io, json, os, re, stat, sys, time
from pathlib import Path
import numpy as np
H=Path(__file__).parent
TARGET5=('support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass')
INPUT8=('rgb96','valid96','row_id','original_global_index','native_slot','query_xy','L','original_query_available')

def require(ok,message):
    if not bool(ok):raise ValueError(message)

def raw(path):
    p=Path(path);require(p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).lower(),'literal non-V112 before filesystem')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        leaf=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    with os.fdopen(leaf,'rb') as f:require(stat.S_ISREG(os.fstat(f.fileno()).st_mode),'regular file');return f.read()

def binding(p):
    b=raw(p);return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))

def checked(b,allowed,allow_C65_checkpoint=False):
    C.need(any(x['path']==b['path'] and x['sha256']==b['sha256'] for x in allowed),'explicit path/SHA whitelist before payload')
    C.need(Path(b['path']).suffix.lower() not in ('.pt','.pth','.ckpt') or allow_C65_checkpoint is True,'only the separately designated C65 checkpoint may be decoded')
    z=raw(b['path']);C.need(hashlib.sha256(z).hexdigest()==b['sha256'],'bound SHA');C.need('size_bytes' not in b or len(z)==b['size_bytes'],'optional bound size');return z

def js(b,allowed):
    z=checked(b,allowed).decode();return [json.loads(x) for x in z.splitlines() if x.strip()] if Path(b['path']).suffix=='.jsonl' else json.loads(z)

def module(b,allowed,name):
    z=checked(b,allowed);m=type(sys)(name);m.__file__=b['path'];sys.modules[name]=m;exec(compile(z,b['path'],'exec'),m.__dict__);return m

def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
    return binding(p)

def arrays(b,keys,allow):
    C.need(Path(b['path']).suffix=='.npz','root-declared input/target container only')
    with np.load(io.BytesIO(checked(b,allow)),allow_pickle=False) as z:return {k:z[k] for k in keys}

def same_meta(k,a,b):
    if k in ('query_xy','L'):
        missing=[np.nan,np.nan] if k=='query_xy' else np.nan
        return np.array_equal(np.asarray(missing if a is None else a,np.float64),np.asarray(missing if b is None else b,np.float64),equal_nan=True)
    return np.array_equal(np.asarray(a),np.asarray(b))

def state_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy();h.update(k.encode());h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()

def cpu_copy(x):
    import torch
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_copy(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_copy(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_copy(v) for v in x)
    return x

def weak_RGB(r,allow):
    from PIL import Image
    b=checked(r['RGB96'],allow);im=Image.open(io.BytesIO(b));C.need(im.format=='PNG' and im.mode=='RGB' and im.size==(96,96),'unmarked owned RGB96');im.load()
    valid=np.zeros(9216,bool)
    for start,length in r['valid96_RLE']:
        C.need(type(start) is int and type(length) is int and start>=0 and length>0 and start+length<=9216,'strict validity RLE');C.need(not valid[start:start+length].any(),'nonoverlapping validity runs');valid[start:start+length]=True
    return np.asarray(im).copy(),valid.reshape(96,96)

def digest_draw(h,draw,point):
    for k,v in draw.items():h.update(k.encode());h.update(v.astype('<i8').tobytes())
    h.update(point.astype('<i8').tobytes())


def designated_C65(e,meta,binary):
    ref=js(e['reference_C65_fit_complete'],meta)
    old=js(e['reference_C65_training_entry'],meta)
    freeze=js(e['reference_C65_all2_freeze'],meta)
    recipe=js(e['reference_C65_recipe'],meta)
    C.need(ref['completed'] is True and ref['PASS'] is True and ref['source_only'] is True and ref['from_scratch'] is True,'completed original source fit')
    C.need(ref['parameters_each']==118359 and ref['steps_each']==4000 and ref['protected_access']==0,'original same C61 source model')
    arm=ref['arms']['STRONG_PLUS_EXPECTED_PART_WEAK']
    C.need(C.same_binding(e['warmstart_checkpoint'],arm['checkpoint']) and C.same_binding(freeze['checkpoints']['STRONG_PLUS_EXPECTED_PART_WEAK'],arm['checkpoint']),'exact C65 state, never select a checkpoint from scores')
    C.need(C.same_binding(ref['all2_freeze'],e['reference_C65_all2_freeze']) and C.same_binding(ref['entry'],e['reference_C65_training_entry']) and C.same_binding(ref['root_recipe'],e['reference_C65_recipe']),'original source closure')
    C.need(C.same_binding(ref['code'],e['reference_V219_runner']) and C.same_binding(ref['contract'],e['reference_V219_contract']),'original exact training code lineage')
    for d in C.DOMAINS:
        for k in ('inputs','targets','metadata'):C.need(C.same_binding(e['old_caches'][d][k],old['old_caches'][d][k]),'same fit-only6 caches and metadata')
    for k in ('point_metadata7040','weak_rows360','lineage_assembly_QA'):C.need(C.same_binding(e[k],old[k]),'same SOURCE targets/lineage '+k)
    C.need(set(e['new_DEV_component_ids'])==set(old['new_DEV_component_ids']),'same original DEV exclusion')
    C.need(e['kernels']==old['kernels'] and recipe['weak_weight']==.25,'same architecture/loss kernels/natural weak weight')
    import torch
    started=time.monotonic()
    payload=torch.load(io.BytesIO(checked(e['warmstart_checkpoint'],binary,allow_C65_checkpoint=True)),map_location='cpu',weights_only=True)
    C.need(payload['update']==4000 and payload['parameters']==118359 and payload['spatial_q8'] is True and payload['use_sam'] is False and payload['use_weak'] is True and payload['weak_weight']==.25,'strict C65 checkpoint type')
    C.need(C.same_binding(payload['source_entry'],e['reference_C65_training_entry']) and C.same_binding(payload['recipe'],e['reference_C65_recipe']),'checkpoint source entry/recipe')
    state=payload['state_dict'];C.need(state_sha(state)==arm['final'],'same frozen C65 tensor-state hash')
    del payload  # Optimizer tensors may be serialized in this source checkpoint; none are reused.
    return state,recipe,time.monotonic()-started


def run(entry_path):
    global C
    prepared_started=time.monotonic()
    e=json.loads(raw(entry_path));require(e['approved_source_pair_fit_or_probe'] is True and e['source_only'] is True and e['automatic_retry'] is False,'root actual source entry')
    require(e['code']==binding(__file__) and e['contract']['path']==str(H/'source_training_contract.py'),'exact self/companion path before code execution')
    protocol=json.loads(raw(H/'PROTOCOL_BEFORE_ANY_V222_SOURCE_PAYLOAD_OR_FIT.json'))
    require(e['protocol']==binding(H/'PROTOCOL_BEFORE_ANY_V222_SOURCE_PAYLOAD_OR_FIT.json') and protocol['code']==e['code'] and protocol['contract']==e['contract'],'frozen protocol and SHA before contract import')
    contract_bytes=raw(e['contract']['path']);require(hashlib.sha256(contract_bytes).hexdigest()==e['contract']['sha256'] and ('size_bytes' not in e['contract'] or len(contract_bytes)==e['contract']['size_bytes']),'companion SHA before execution')
    C=type(sys)('v222_contract');exec(compile(contract_bytes,e['contract']['path'],'exec'),C.__dict__)
    C.need(e['approved_load_exact_C65_into_both_arms'] is True,'explicit same SOURCE continuation authority')
    C.need(isinstance(e['warmstart_checkpoint'],dict) and e['mode'] in ('THROUGHPUT_PROBE','FIXED_PAIRED_FIT'),'same C65 continuation and bounded mode')
    meta=e['metadata_whitelist'];code=e['code_whitelist'];binary=e['fit_binary_whitelist']
    C.need(e['code']==binding(__file__) and e['contract']==binding(H/'source_training_contract.py'),'frozen current code')
    protocol=json.loads(raw(H/'PROTOCOL_BEFORE_ANY_V222_SOURCE_PAYLOAD_OR_FIT.json'))
    C.need(e['protocol']==binding(H/'PROTOCOL_BEFORE_ANY_V222_SOURCE_PAYLOAD_OR_FIT.json') and protocol['code']==e['code'] and protocol['contract']==e['contract'] and protocol['kernels']==e['kernels'],'exact frozen protocol/kernel closure')
    C.need(C.same_binding(e['reference_V219_runner'],protocol['reference_V219_runner']) and C.same_binding(e['reference_V219_contract'],protocol['reference_V219_contract']),'pinned original source recipe code')
    scope=js(e['root_scope'],meta);recipe=js(e['root_recipe'],meta)
    C.need(scope['approved_source_pair_fit_or_probe'] is True and recipe['approved_frozen_recipe'] is True,'separate root scope/recipe')
    C.need(recipe['budget_selection_basis']=='THROUGHPUT_ONLY_NO_SCORE_OR_ARM_SELECTION','throughput-only budget rationale')
    steps=recipe['steps'];C.need(type(steps) is int and steps>0,'root-frozen update budget')
    C.need(recipe['natural_weak_weight']==.25 and recipe['CF_weak_weight']==.25 and recipe['CF_parents_per_step']==4 and recipe['CF_sampling']=='UNIFORM_WITH_REPLACEMENT','root fixed weak BCE recipe')
    if e['mode']=='THROUGHPUT_PROBE':C.need(steps==32,'exact32 disposable updates per arm')
    else:
        probe=js(e['throughput_probe_complete'],meta)
        C.need(probe['completed'] is True and probe['PASS'] is True and probe['steps_each']==32 and probe['checkpoint_saved'] is False and C.same_binding(probe['code'],e['code']) and C.same_binding(probe['contract'],e['contract']) and C.same_binding(probe['C65_checkpoint'],e['warmstart_checkpoint']),'same-code same-checkpoint disposable probe')
        C.need(C.same_binding(probe['CF_rows144'],e['CF_rows144']) and probe['old_caches']==e['old_caches'],'same source payloads as throughput measurement')
        C.need(steps==C.fixed_budget_from_probe(probe['step_seconds_by_arm']),'exact prospective60sec bounded budget; no score choice')
    for key,flag,authority,sourcekey in [('strong_target_admission','approved_strong_synthetic_point_targets',C.STRONG_AUTHORITY,'point_metadata7040'),('weak_target_admission','approved_public_expected_part_weak_targets',C.WEAK_AUTHORITY,'weak_rows360')]:
        admission=js(e[key],meta);C.need(admission[flag] is True and admission['authority']==authority and admission['target_metadata']==e[sourcekey],'separate bound target authority/admission '+key)
    lineage=js(e['lineage_assembly_QA'],meta);C.need(lineage['completed'] is True and lineage['PASS'] is True and lineage['all_old_public_CF_descendants_of_new_DEV_excluded'] is True and lineage['q8_input_provenance_verified_no_target_geometry_features'] is True,'exact joined lineage QA before source payload')
    joined=js(e['weak_rows360'],meta);weak,dev=C.weak_plan(joined)
    C.need(set(e['new_DEV_component_ids'])==dev,'frozen DEV4 exact')
    cfrows=js(e['CF_rows144'],meta);cfpairs,cfstatuses=C.cf_plan(cfrows,weak,joined);cf_fit=[p for p in cfpairs if p['fit_eligible']]
    fit_manifest=js(e['CF_FIT_manifest'],meta);fit_CF_allow=C.role_manifest(fit_manifest,'SOURCE_FIT',cfpairs,cfrows)
    C.need(all(any(C.same_binding(b,x) for x in binary) for b in fit_CF_allow),'all and only role-declared CF images in FIT whitelist')
    admission=js(e['CF_target_admission'],meta);proof=js(e['CF_independent_data_QA'],meta)
    C.need(admission['completed'] is True and admission['approved_CF_FIT_weak_only'] is True and admission['training_admission'] is True and admission['new_native_point_gold'] is False and admission['allowed_role']=='SOURCE_FIT' and admission['allowed_target_field']=='weak_finite_part_target_D0_E1','separate root CF weak-only authority')
    C.need(set(admission['allowed_parent_ids'])=={p['parent_row_id'] for p in cf_fit} and C.same_binding(admission['CF_FIT_manifest'],e['CF_FIT_manifest']) and all(r['family']==admission['approved_family'] for r in cfrows),'exact admitted FIT39 parent/family set')
    C.need(admission['actual_probe_or_fit_execution_authorized_by_this_file'] is False,'admission alone is not an execution entry')
    producer=js(e['CF_producer_complete'],meta)
    C.need(producer['completed'] is True and producer['PASS'] is True and C.same_binding(producer['variant_rows'],e['CF_rows144']) and C.same_binding(producer['attempts'],e['CF_attempts48']) and C.same_binding(producer['role_manifests']['SOURCE_FIT'],e['CF_FIT_manifest']) and C.same_binding(producer['role_manifests']['SOURCE_DEV'],e['CF_DEV_manifest']),'exact current producer closure')
    C.need(proof['completed'] is True and proof['PASS'] is True and C.same_binding(admission['actual_pixel_QA'],e['CF_independent_data_QA']) and C.same_binding(admission['producer_complete'],e['CF_producer_complete']),'root binds independent constructed-pixel/validity/SHAM proof before fit')
    C.need(len(cf_fit)>0 and all(p['component_id'] not in dev for p in cf_fit),'constructed CF FIT only, no fallback')
    initial_state,original_recipe,checkpoint_read_seconds=designated_C65(e,meta,binary)
    C.need(recipe['learning_rate']==original_recipe['learning_rate'] and recipe['weight_decay']==original_recipe['weight_decay'],'same C65 optimizer hyperparameters, reset state only')
    base=module(e['kernels']['base_head'],code,'v219_base_head');candidate=module(e['kernels']['candidate'],code,'v219_candidate');spatial=module(e['kernels']['spatial_head'],code,'v219_spatial');point_contract=module(e['kernels']['point_contract'],code,'v219_point_contract')
    import torch
    from torch.nn import functional as F
    tree=ast.parse(checked(e['kernels']['old_trainer'],code).decode());names={'validate_targets','draw_step','five_losses'};nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names];C.need({x.name for x in nodes}==names,'pure old loss/draw kernels only')
    pure=dict(np=np,torch=torch,F=F,require=C.need);exec(compile(ast.Module(body=nodes,type_ignores=[]),'BOUND_OLD_FUNCTIONS','exec'),pure)
    caches={};masks={};contexts={}
    for domain in C.DOMAINS:
        rec=e['old_caches'][domain];rr=js(rec['metadata'],meta);fit=C.fit_mask(rr,dev);C.need(fit.all(),'only pre-extracted original TRAIN/nominal/lineage/nonDEV numerical rows allowed before NPZ decode');x=arrays(rec['inputs'],INPUT8,binary);y=pure['validate_targets'](arrays(rec['targets'],TARGET5,binary),x['original_query_available']);n=len(rr)
        C.need(x['rgb96'].dtype==np.uint8 and x['rgb96'].shape==(n,96,96,3) and x['valid96'].dtype==bool and x['valid96'].shape==(n,96,96),'same RAW96 input')
        for i,r in enumerate(rr):
            for k in C.META:C.need(same_meta(k,r[k],x[k][i]),'exact row META '+k)
            C.need(r['context_origin']=='ORIGINAL_INITIALIZER','old context cannot use teacher reference')
        contexts[domain]=np.asarray([C.context32(r['query_xy'],r['L'],r['native_slot'],r['original_query_available'],r['context_q8'],r['context_L8'],r['context_available8']) for r in rr],np.float32)
        caches[domain]=dict(x=x,y=y,metadata=rr);masks[domain]=C.fit_mask(rr,dev)
    pools=C.old_pools(caches,masks);syn=caches[C.DOMAINS[0]];pointrows=js(e['point_metadata7040'],meta);C.need(len(pointrows)==7040,'full original point census')
    byid={r['row_id']:i for i,r in enumerate(syn['metadata'])};C.need(len(byid)==len(syn['metadata']),'unique source row IDs');pt=np.full(len(byid),-1,np.int8);seen=set()
    for r in pointrows:
        C.need(r['row_id'] in byid and r['row_id'] not in seen,'unique point row in source RGB');seen.add(r['row_id']);i=byid[r['row_id']];m=syn['metadata'][i]
        for k in C.META:C.need(same_meta(k,m[k],r[k]),'point META '+k)
        C.need(r['source_teacher_key']==[m['source_image_original_global_index']//8,m['native_slot']] and r['source_image_original_global_index']==m['source_image_original_global_index'] and r['anchor_original_global_index']==m['anchor_original_global_index'],'own image point vs BEFORE anchor')
        t,_=point_contract.point_loss_target(r);pt[i]=t if masks[C.DOMAINS[0]][i] else -1
    C.need(seen=={r['row_id'] for i,r in enumerate(syn['metadata']) if masks[C.DOMAINS[0]][i]},'all original TRAIN nominal point rows retained')
    pp=C.balanced_pools(pt,masks[C.DOMAINS[0]]);C.need([len(p) for p in pp]==e['expected_strong_D_E_counts'],'root admitted actual synthetic point class census')
    wt=np.asarray([r['target'] for r in weak],np.int8);wm=np.asarray([r['fit_eligible'] for r in weak],bool);wp=C.balanced_pools(wt,wm)
    C.need(not any(C.same_binding(r['RGB96'],b) for r in weak if r['source_role']=='SOURCE_DEV' and r['target']>=0 for b in binary),'natural weak DEV12 physical inputs absent from FIT whitelist')
    weakpixels={i:weak_RGB(weak[i],binary) for i in np.flatnonzero(wm)}  # DEV pixels deferred until final2 barrier.
    cfpixels={(j,k):weak_RGB(r,fit_CF_allow) for j,p in enumerate(cf_fit) for k,r in enumerate(p['rows'])}  # Only constructed FIT control pixels before barrier.
    out=Path(e['output_directory']);C.need(str(out)==protocol['planned_outputs'][e['mode']] and out.is_absolute() and '..' not in out.parts and 'v222' in str(out).lower() and 'v112' not in str(out).lower(),'exact fresh V222 output only before FS');fd=os.open(out.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in out.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        os.mkdir(out.name,dir_fd=fd)
    finally:os.close(fd)
    write(out/'ENTRY_AND_INPUT_CENSUS.json',dict(entry=binding(entry_path),new_DEV_components=sorted(dev),strong_class_counts=[len(p) for p in pp],weak_FIT_D_E=[len(p) for p in wp],weak_DEV_D_E=[9,3],CF_constructed_parent_counts={role:sum(p['source_role']==role for p in cfpairs) for role in ('SOURCE_FIT','SOURCE_DEV')},CF_all48_statuses=cfstatuses,all144_CF_status_rows_preserved=True,all360_preserved=True,point7040_preserved=True,source_DEV_fit_and_norm_excluded=True,C65_checkpoint=e['warmstart_checkpoint'],optimizer_state_reused=False,normalization_reference_entry=e['reference_C65_training_entry']))
    write(out/'ALL144_CF_STATUS_METADATA.json',dict(completed=True,rows144=cfrows,failed_COVER_SHAM_remain_UNKNOWN_inputunavailable=True,new_native_point_labels=0))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=e['device'];newclass=spatial.make_spatial_q8_head_class(base,candidate);pending={};initial=None;common=None;common_natural=None;timings={};start=time.monotonic();preparation_seconds=start-prepared_started
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize()
    for arm,use_cf in zip(C.ARMS,(False,True)):
        load_started=time.monotonic();torch.manual_seed(1810);torch.cuda.manual_seed_all(1810);model=newclass(spatial_q8=True).float().to(device);C.need(sum(p.numel() for p in model.parameters())==118359,'same C61 capacity');model.load_state_dict(initial_state,strict=True);sync();load_seconds=time.monotonic()-load_started
        ish=state_sha(model.state_dict());initial=ish if initial is None else initial;C.need(ish==initial==state_sha(initial_state),'same exact C65 initialization, no parameter reset')
        opt=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay']);C.need(not opt.state,'fresh optimizer state');rng=np.random.default_rng(1810);prng=np.random.default_rng(1811);wrng=np.random.default_rng(1812);crng=np.random.default_rng(1813);hh=hashlib.sha256();ww=hashlib.sha256();ch=hashlib.sha256();folder=out/arm;folder.mkdir();update_seconds=0.;log_seconds=0.
        with (folder/'ALL_DRAWS_AND_LOSSES.jsonl').open('x') as log:
            for update in range(1,steps+1):
                sync();tick=time.monotonic();draw,pi,wi,ci=C.draw_common_and_cf(pure['draw_step'],rng,prng,wrng,crng,pools,pp,wp,len(cf_fit),use_cf)
                selections=[(C.DOMAINS[0],draw[k]) for k in ('visible','foreign','synthetic_owner')]+[(C.DOMAINS[1],draw['public_owner']),(C.DOMAINS[2],draw['CF_owner']),(C.DOMAINS[0],pi)]
                for d,ii in selections:C.need(masks[d][ii].all(),'all loss draws TRAIN/nominal/nonDEV/lineage')
                if wi is not None:C.need(wm[wi].all(),'weak FIT only')
                rgb=np.concatenate([caches[d]['x']['rgb96'][ii] for d,ii in selections]);valid=np.concatenate([caches[d]['x']['valid96'][ii] for d,ii in selections]);ctx=np.concatenate([contexts[d][ii] for d,ii in selections])
                rgb=np.concatenate((rgb,np.asarray([weakpixels[i][0] for i in wi])));valid=np.concatenate((valid,np.asarray([weakpixels[i][1] for i in wi])));ctx=np.concatenate((ctx,np.asarray([weak[i]['condition32'] for i in wi],np.float32)))
                if ci is not None:
                    C.need(all(cf_fit[int(j)]['fit_eligible'] for j in ci),'CF parents FIT only')
                    rgb=np.concatenate((rgb,np.asarray([cfpixels[int(j),k][0] for j in ci for k in range(3)])));valid=np.concatenate((valid,np.asarray([cfpixels[int(j),k][1] for j in ci for k in range(3)])));ctx=np.concatenate((ctx,np.asarray([cf_fit[int(j)]['rows'][k]['condition32'] for j in ci for k in range(3)],np.float32)))
                digest_draw(hh,draw,pi);ww.update(wi.astype('<i8').tobytes())
                if ci is not None:ch.update(ci.astype('<i8').tobytes())
                inp=base.assemble_local_input(torch.as_tensor(rgb,device=device),torch.as_tensor(valid,device=device),use_sam=False);ct=torch.as_tensor(ctx,device=device);native=ct[:,:8].argmax(1).to(torch.int64)
                opt.zero_grad(set_to_none=True);a=model(inp,native,ct[:,8:16],ct[:,16:32]);loss,parts,strong,local,cf=C.matched_objective(a,torch.as_tensor(pt[pi],device=device),torch.as_tensor(wt[wi],device=device),use_cf,pure['five_losses']);C.need(torch.isfinite(loss),'finite source objective');loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);C.need(torch.isfinite(gn),'finite source gradient');opt.step();sync();update_seconds+=time.monotonic()-tick;log_tick=time.monotonic()
                log.write(json.dumps(dict(update=update,old192={k:v.tolist() for k,v in draw.items()},strong32=pi.tolist(),naturalweak16=wi.tolist(),CF4parents=None if ci is None else ci.tolist(),CF_parent_row_ids=None if ci is None else [cf_fit[int(j)]['parent_row_id'] for j in ci],old5=[float(x.detach()) for x in parts],strong=float(strong.detach()),naturalweak=float(local.detach()),CF_weighted_parent_BCE=float(cf.detach()),total=float(loss.detach()),gradient_norm=float(gn),old_point_sha=hh.hexdigest(),naturalweak_sha=ww.hexdigest(),CF_sha=ch.hexdigest(),all_fit_masks_valid=True),allow_nan=False)+'\n');log_seconds+=time.monotonic()-log_tick
        common=hh.hexdigest() if common is None else common;common_natural=ww.hexdigest() if common_natural is None else common_natural;C.need(hh.hexdigest()==common and ww.hexdigest()==common_natural,'paired exact old+point+naturalweak draws');timings[arm]=dict(optimizer_update_seconds=update_seconds,step_seconds=update_seconds/steps,log_write_seconds=log_seconds,model_construct_and_C65_load_seconds=load_seconds)
        model.eval().requires_grad_(False);pending[arm]=dict(model=model,initial=ish,final=state_sha(model.state_dict()),old_point_draw_sha=hh.hexdigest(),naturalweak_draw_sha=ww.hexdigest(),CF_draw_sha=ch.hexdigest(),use_CF=use_cf)
        if e['mode']=='FIXED_PAIRED_FIT':
            save_started=time.monotonic();cp=folder/'FINAL_CHECKPOINT.pt';torch.save(dict(state_dict=cpu_copy(model.state_dict()),optimizer_state=cpu_copy(opt.state_dict()),update=steps,parameters=118359,spatial_q8=True,use_sam=False,use_weak=True,weak_weight=.25,use_CF=use_cf,CF_weight=.25 if use_cf else 0.,CF_authority=C.CF_AUTHORITY,parent_C65_checkpoint=e['warmstart_checkpoint'],recipe=e['root_recipe'],source_entry=binding(entry_path)),cp);pending[arm]['checkpoint']=binding(cp);timings[arm]['checkpoint_save_seconds']=time.monotonic()-save_started
        del opt
    if e['mode']=='THROUGHPUT_PROBE':
        for v in pending.values():del v['model']
        write(out/'PROBE_COMPLETE.json',dict(completed=True,PASS=True,code=e['code'],contract=e['contract'],entry=binding(entry_path),C65_checkpoint=e['warmstart_checkpoint'],CF_rows144=e['CF_rows144'],old_caches=e['old_caches'],steps_each=steps,updates=2*steps,timings=timings,step_seconds_by_arm={k:v['step_seconds'] for k,v in timings.items()},budget_by_prospective_rule=C.fixed_budget_from_probe({k:v['step_seconds'] for k,v in timings.items()}),preparation_seconds=preparation_seconds,checkpoint_read_seconds=checkpoint_read_seconds,seconds=time.monotonic()-start,DEV_pixels_or_predictions_read=0,checkpoint_saved=False,probe_state_and_optimizer_discarded=True,for_throughput_only_no_metric_selection=True));return
    freeze=write(out/'ALL2_FINAL_FROZEN_BEFORE_ANY_DEV_OR_REAL_OUTPUT.json',dict(completed=True,PASS=True,checkpoints={k:v['checkpoint'] for k,v in pending.items()},same_initial_state_sha=initial,same_old_point_draw_sha=common,same_naturalweak_draw_sha=common_natural,parent_C65_checkpoint=e['warmstart_checkpoint'],DEV_or_real_output_emitted=False,selected_arm=None))
    post=e['postfreeze_binary_whitelist']
    dev_manifest=js(e['CF_DEV_manifest'],meta+e['postfreeze_metadata_whitelist'])
    dev_CF_allow=C.role_manifest(dev_manifest,'SOURCE_DEV',cfpairs,cfrows)
    C.need(all(any(C.same_binding(b,x) for x in post) for b in dev_CF_allow),'DEV images only in post-freeze whitelist')
    C.need(not any(C.same_binding(a,b) for a in dev_CF_allow for b in binary),'physical CF DEV separation before barrier')
    known=[i for i,r in enumerate(weak) if r['target']>=0]
    for i in known:
        if i not in weakpixels:weakpixels[i]=weak_RGB(weak[i],post)
    CF_eval=cfrows
    CF_eval_allow=fit_CF_allow+dev_CF_allow
    cf_eval_pixels={r['row_id']:weak_RGB(r,CF_eval_allow) for r in CF_eval if r['RGB96'] is not None}
    for arm,r in pending.items():
        model=r.pop('model');rr=[];cr=[]
        with torch.inference_mode():
            point_indices=np.concatenate(pp);point_outputs=[]
            for start2 in range(0,len(point_indices),96):
                ii=point_indices[start2:start2+96];ct=torch.as_tensor(contexts[C.DOMAINS[0]][ii],device=device)
                a=model(base.assemble_local_input(torch.as_tensor(syn['x']['rgb96'][ii],device=device),torch.as_tensor(syn['x']['valid96'][ii],device=device),use_sam=False),ct[:,:8].argmax(1).to(torch.int64),ct[:,8:16],ct[:,16:]);z=a['direct_end_external_logit'].cpu().numpy();C.need(np.isfinite(z).all(),'finite strong source predictions')
                point_outputs += [dict(row_id=str(syn['x']['row_id'][i]),authority=C.STRONG_AUTHORITY,source_role='TRAIN',target=int(pt[i]),raw_logit=float(v),prediction_D0_E1=int(v>=0)) for i,v in zip(ii,z)]
            for start2 in range(0,len(known),32):
                ii=known[start2:start2+32];rgb=np.asarray([weakpixels[i][0] for i in ii]);valid=np.asarray([weakpixels[i][1] for i in ii]);ctx=torch.as_tensor(np.asarray([weak[i]['condition32'] for i in ii],np.float32),device=device)
                a=model(base.assemble_local_input(torch.as_tensor(rgb,device=device),torch.as_tensor(valid,device=device),use_sam=False),ctx[:,:8].argmax(1).to(torch.int64),ctx[:,8:16],ctx[:,16:]);z=a['direct_end_external_logit'].cpu().numpy();C.need(np.isfinite(z).all(),'finite source predictions')
                rr += [dict(row_id=weak[i]['row_id'],source_role=weak[i]['source_role'],authority=C.WEAK_AUTHORITY,target=weak[i]['target'],raw_logit=float(v),prediction_D0_E1=int(v>=0)) for i,v in zip(ii,z)]
        with torch.inference_mode():
            for x in CF_eval:
                z=None;prediction=-1
                if x['RGB96'] is not None:
                    rgb,valid=cf_eval_pixels[x['row_id']];ct=torch.as_tensor(np.asarray([x['condition32']],np.float32),device=device)
                    a=model(base.assemble_local_input(torch.as_tensor(rgb[None],device=device),torch.as_tensor(valid[None],device=device),use_sam=False),ct[:,:8].argmax(1).to(torch.int64),ct[:,8:16],ct[:,16:]);zz=a['direct_end_external_logit'].cpu().numpy();C.need(zz.shape==(1,) and np.isfinite(zz).all(),'finite existing CF/factual source output');z=float(zz[0]);prediction=int(z>=0)
                cr.append(dict(row_id=x['row_id'],parent_row_id=x['parent_row_id'],paired_id=x['paired_id'],source_role=x['source_role'],component_id=x['component_id'],native_slot=x['native_slot'],query_xy=x['query_xy'],L=x['L'],query_available=x['query_available'],variant=x['variant'],authority=C.CF_AUTHORITY,target=x['weak_finite_part_target_D0_E1'],native_point_DE_target=-1,input_available=x['RGB96'] is not None,raw_logit=z,prediction_D0_E1_Uminus1=prediction,constructed=x['constructed']))
        r['CF_all144_outputs']=write(out/arm/'CF_ALL144_SOURCE_OUTPUTS.json',dict(completed=True,rows144=cr,all144_status_metadata='../ALL144_CF_STATUS_METADATA.json',missing_COVER_SHAM_UNKNOWN_U_not_new_E=True,primary_native_gold=False))
        r['strong_outputs']=write(out/arm/'STRONG_POINT_TRAIN_OUTPUTS.json',dict(completed=True,rows=point_outputs,source_apparent_not_validation=True));r['weak57_outputs']=write(out/arm/'WEAK57_SOURCE_OUTPUTS.json',dict(completed=True,rows57=rr,no_primary_native_gold=True,no_original_initializer_performance_claim=True));C.need(state_sha(model.state_dict())==r['final'],'no state change during outputs')
    write(out/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,from_scratch=False,continued_from_same_C65_checkpoint=e['warmstart_checkpoint'],optimizer_state_reset=True,entry=binding(entry_path),code=binding(__file__),contract=binding(H/'source_training_contract.py'),factory_closure=e['kernels'],root_recipe=e['root_recipe'],all2_freeze=freeze,steps_each=steps,parameters_each=118359,arms=pending,DEV_not_fit=True,weak57_source_only_not_freshblind=True,CF_weak_finite_part_not_point_gold=True,GT_derived_q8_training_family_not_original_initializer=True,old_point_and_natural_draws_paired=True,same_naturalweak_draw_sha=common_natural,timings=timings,preparation_seconds=preparation_seconds,checkpoint_read_seconds=checkpoint_read_seconds,protected_access=0,selected_arm=None,adopted=False,fully_solved=False,seconds=time.monotonic()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);a=p.parse_args();run(a.entry)
