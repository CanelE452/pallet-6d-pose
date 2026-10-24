"""CODEONLY prospective runner. No source access/model construction without --entry approval."""
import argparse, ast, hashlib, importlib.util, io, json, os, re, stat, sys, time
from pathlib import Path
import numpy as np
import source_training_contract as C
H=Path(__file__).parent
TARGET5=('support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass')
INPUT8=('rgb96','valid96')+C.META

def raw(path):
    p=Path(path);C.need(p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).lower(),'literal non-V112 before filesystem')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        leaf=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    with os.fdopen(leaf,'rb') as f:C.need(stat.S_ISREG(os.fstat(f.fileno()).st_mode),'regular file');return f.read()

def binding(p):
    b=raw(p);return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))

def checked(b,allowed):
    C.need(any(x['path']==b['path'] and x['sha256']==b['sha256'] for x in allowed),'explicit path/SHA whitelist before payload')
    C.need(Path(b['path']).suffix.lower() not in ('.pt','.pth','.ckpt'),'no old checkpoint/foundation weights read')
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

def run(entry_path):
    e=json.loads(raw(entry_path));C.need(e['approved_source_pair_fit_or_probe'] is True and e['source_only'] is True and e['automatic_retry'] is False,'root actual source entry')
    C.need(e['warmstart_checkpoint'] is None and e['mode'] in ('THROUGHPUT_PROBE','FIXED_PAIRED_FIT'),'fresh-only bounded mode')
    meta=e['metadata_whitelist'];code=e['code_whitelist'];binary=e['binary_whitelist']
    C.need(e['code']==binding(__file__) and e['contract']==binding(H/'source_training_contract.py'),'frozen current code')
    protocol=json.loads(raw(H/'PROTOCOL_BEFORE_ANY_V219_SOURCE_PAYLOAD_OR_FIT.json'))
    C.need(e['protocol']==binding(H/'PROTOCOL_BEFORE_ANY_V219_SOURCE_PAYLOAD_OR_FIT.json') and protocol['code']==e['code'] and protocol['contract']==e['contract'] and protocol['kernels']==e['kernels'],'exact frozen protocol/kernel closure')
    scope=js(e['root_scope'],meta);recipe=js(e['root_recipe'],meta)
    C.need(scope['approved_source_pair_fit_or_probe'] is True and recipe['approved_frozen_recipe'] is True,'separate root scope/recipe')
    C.need(recipe['budget_selection_basis']=='THROUGHPUT_ONLY_NO_SCORE_OR_ARM_SELECTION','throughput-only budget rationale')
    steps=recipe['steps'];C.need(type(steps) is int and steps>0 and (e['mode']!='THROUGHPUT_PROBE' or steps<=32),'root-frozen update budget')
    for key,flag,authority,sourcekey in [('strong_target_admission','approved_strong_synthetic_point_targets',C.STRONG_AUTHORITY,'point_metadata7040'),('weak_target_admission','approved_public_expected_part_weak_targets',C.WEAK_AUTHORITY,'weak_rows360')]:
        admission=js(e[key],meta);C.need(admission[flag] is True and admission['authority']==authority and admission['target_metadata']==e[sourcekey],'separate bound target authority/admission '+key)
    lineage=js(e['lineage_assembly_QA'],meta);C.need(lineage['completed'] is True and lineage['PASS'] is True and lineage['all_old_public_CF_descendants_of_new_DEV_excluded'] is True and lineage['q8_input_provenance_verified_no_target_geometry_features'] is True,'exact joined lineage QA before source payload')
    joined=js(e['weak_rows360'],meta);weak,dev=C.weak_plan(joined)
    C.need(set(e['new_DEV_component_ids'])==dev,'frozen DEV4 exact')
    base=module(e['kernels']['base_head'],code,'v219_base_head');candidate=module(e['kernels']['candidate'],code,'v219_candidate');spatial=module(e['kernels']['spatial_head'],code,'v219_spatial');point_contract=module(e['kernels']['point_contract'],code,'v219_point_contract')
    import torch
    from torch.nn import functional as F
    tree=ast.parse(checked(e['kernels']['old_trainer'],code).decode());names={'validate_targets','draw_step','five_losses'};nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names];C.need({x.name for x in nodes}==names,'pure old loss/draw kernels only')
    pure=dict(np=np,torch=torch,F=F,require=C.need);exec(compile(ast.Module(body=nodes,type_ignores=[]),'BOUND_OLD_FUNCTIONS','exec'),pure)
    caches={};masks={};contexts={}
    for domain in C.DOMAINS:
        rec=e['old_caches'][domain];rr=js(rec['metadata'],meta);x=arrays(rec['inputs'],INPUT8,binary);y=pure['validate_targets'](arrays(rec['targets'],TARGET5,binary),x['original_query_available']);n=len(rr)
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
    weakpixels={i:weak_RGB(weak[i],binary) for i in np.flatnonzero(wm)}  # DEV pixels deferred until final2 barrier.
    out=Path(e['output_directory']);C.need(out.is_absolute() and 'v219' in str(out).lower() and 'v112' not in str(out).lower() and not any(x.is_symlink() for x in [out,*out.parents]),'fresh V219 output only');out.mkdir(exist_ok=False)
    write(out/'ENTRY_AND_INPUT_CENSUS.json',dict(entry=binding(entry_path),new_DEV_components=sorted(dev),strong_class_counts=[len(p) for p in pp],weak_FIT_D_E=[len(p) for p in wp],weak_DEV_D_E=[9,3],all360_preserved=True,point7040_preserved=True,source_DEV_fit_and_norm_excluded=True,old_checkpoint_loaded=False))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=e['device'];newclass=spatial.make_spatial_q8_head_class(base,candidate);pending={};initial=None;common=None;start=time.monotonic()
    for arm,use_weak in [('STRONG_ONLY_CONTROL',False),('STRONG_PLUS_EXPECTED_PART_WEAK',True)]:
        torch.manual_seed(1810);torch.cuda.manual_seed_all(1810);model=newclass(spatial_q8=True).float().to(device);C.need(sum(p.numel() for p in model.parameters())==118359,'same C61 capacity')
        ish=state_sha(model.state_dict());initial=ish if initial is None else initial;C.need(ish==initial,'same fresh initialization')
        opt=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay']);rng=np.random.default_rng(1810);prng=np.random.default_rng(1811);wrng=np.random.default_rng(1812);hh=hashlib.sha256();ww=hashlib.sha256();folder=out/arm;folder.mkdir()
        with (folder/'ALL_DRAWS_AND_LOSSES.jsonl').open('x') as log:
            for update in range(1,steps+1):
                draw=pure['draw_step'](rng,pools);pi=C.draw_balanced(prng,pp,16);wi=C.draw_balanced(wrng,wp,8) if use_weak and recipe['weak_weight']>0 else None
                selections=[(C.DOMAINS[0],draw[k]) for k in ('visible','foreign','synthetic_owner')]+[(C.DOMAINS[1],draw['public_owner']),(C.DOMAINS[2],draw['CF_owner']),(C.DOMAINS[0],pi)]
                for d,ii in selections:C.need(masks[d][ii].all(),'all loss draws TRAIN/nominal/nonDEV/lineage')
                if wi is not None:C.need(wm[wi].all(),'weak FIT only')
                rgb=np.concatenate([caches[d]['x']['rgb96'][ii] for d,ii in selections]);valid=np.concatenate([caches[d]['x']['valid96'][ii] for d,ii in selections]);ctx=np.concatenate([contexts[d][ii] for d,ii in selections])
                if wi is not None:rgb=np.concatenate((rgb,np.asarray([weakpixels[i][0] for i in wi])));valid=np.concatenate((valid,np.asarray([weakpixels[i][1] for i in wi])));ctx=np.concatenate((ctx,np.asarray([weak[i]['condition32'] for i in wi],np.float32)))
                for k,v in draw.items():hh.update(k.encode());hh.update(v.astype('<i8').tobytes())
                hh.update(pi.astype('<i8').tobytes())
                if wi is not None:ww.update(wi.astype('<i8').tobytes())
                inp=base.assemble_local_input(torch.as_tensor(rgb,device=device),torch.as_tensor(valid,device=device),use_sam=False);ct=torch.as_tensor(ctx,device=device);native=ct[:,:8].argmax(1).to(torch.int64)
                opt.zero_grad(set_to_none=True);a=model(inp,native,ct[:,8:16],ct[:,16:32]);loss,parts,strong,local=C.objective(a,torch.as_tensor(pt[pi],device=device),torch.as_tensor(wt[wi],device=device) if wi is not None else None,recipe['weak_weight'] if use_weak else 0.,pure['five_losses']);C.need(torch.isfinite(loss),'finite source objective');loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);C.need(torch.isfinite(gn),'finite source gradient');opt.step()
                log.write(json.dumps(dict(update=update,old192={k:v.tolist() for k,v in draw.items()},strong32=pi.tolist(),weak16=None if wi is None else wi.tolist(),old5=[float(x.detach()) for x in parts],strong=float(strong.detach()),weak=float(local.detach()),total=float(loss.detach()),gradient_norm=float(gn),old_point_sha=hh.hexdigest(),weak_sha=ww.hexdigest(),all_fit_masks_valid=True),allow_nan=False)+'\n')
        common=hh.hexdigest() if common is None else common;C.need(hh.hexdigest()==common,'paired exact old+strong draws')
        model.eval().requires_grad_(False);pending[arm]=dict(model=model,initial=ish,final=state_sha(model.state_dict()),old_point_draw_sha=hh.hexdigest(),weak_draw_sha=ww.hexdigest())
        if e['mode']=='FIXED_PAIRED_FIT':
            cp=folder/'FINAL_CHECKPOINT.pt';torch.save(dict(state_dict=cpu_copy(model.state_dict()),optimizer_state=cpu_copy(opt.state_dict()),update=steps,parameters=118359,spatial_q8=True,use_sam=False,use_weak=use_weak,weak_weight=recipe['weak_weight'] if use_weak else 0.,recipe=e['root_recipe'],source_entry=binding(entry_path)),cp);pending[arm]['checkpoint']=binding(cp)
        del opt
    if e['mode']=='THROUGHPUT_PROBE':
        write(out/'PROBE_COMPLETE.json',dict(completed=True,PASS=True,updates=2*steps,seconds=time.monotonic()-start,DEV_pixels_or_predictions_read=0,checkpoint_saved=False,for_throughput_only_no_metric_selection=True));return
    freeze=write(out/'ALL2_FINAL_FROZEN_BEFORE_ANY_DEV_OR_REAL_OUTPUT.json',dict(completed=True,PASS=True,checkpoints={k:v['checkpoint'] for k,v in pending.items()},same_initial_state_sha=initial,same_old_point_draw_sha=common,DEV_or_real_output_emitted=False,selected_arm=None))
    known=[i for i,r in enumerate(weak) if r['target']>=0]
    for i in known:
        if i not in weakpixels:weakpixels[i]=weak_RGB(weak[i],binary)
    for arm,r in pending.items():
        model=r.pop('model');rr=[]
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
        r['strong_outputs']=write(out/arm/'STRONG_POINT_TRAIN_OUTPUTS.json',dict(completed=True,rows=point_outputs,source_apparent_not_validation=True));r['weak57_outputs']=write(out/arm/'WEAK57_SOURCE_OUTPUTS.json',dict(completed=True,rows57=rr,no_primary_native_gold=True,no_original_initializer_performance_claim=True));C.need(state_sha(model.state_dict())==r['final'],'no state change during outputs')
    write(out/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,from_scratch=True,entry=binding(entry_path),code=binding(__file__),contract=binding(H/'source_training_contract.py'),root_recipe=e['root_recipe'],all2_freeze=freeze,steps_each=steps,parameters_each=118359,arms=pending,DEV_not_fit=True,weak57_source_only_not_freshblind=True,protected_access=0,selected_arm=None,adopted=False,fully_solved=False,seconds=time.monotonic()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);a=p.parse_args();run(a.entry)
