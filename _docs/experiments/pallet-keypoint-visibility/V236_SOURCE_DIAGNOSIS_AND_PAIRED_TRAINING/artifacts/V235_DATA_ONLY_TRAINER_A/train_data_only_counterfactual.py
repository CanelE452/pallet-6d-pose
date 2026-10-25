"""V235 fixed-encoder/head data-only SOURCE trial; explicit root --entry only.

Import is inert: no source/prefix/weights/model access. Original CONTROL is
byte-copied, not optimized. CF is FIT-only; new head DEV runs after final freeze.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types
import numpy as np
import torch
from torch.nn import functional as F

R=Path(__file__).absolute().parent.parent
CONTROL='CONTROL_LOCAL_CROP'
TREAT='TREAT_CF_DATA_ONLY'
PINS={
 'source_trainer':(str(R/'V233_DETERMINISTIC_CROP_FIX_A/train_crop_adaptation.py'),'ae6e8ceb66ef29a52633ab98511e23a659d7f5b7e8268d300a53b8f4f12d71b1'),
 'crop_module':(str(R/'V233_DETERMINISTIC_CROP_FIX_A/crop_modules.py'),'44f54dfcbf1671a401e748e66ec30463096e6c24b0779f07dda85d7c9f38ae88'),
 'prefix_runtime':(str(R/'V232_PREFIX_TRAIN_CODE_A/prefix_runtime.py'),'5b29e75eaa59c5be6091714a7999ada5272c3090fd4681f80979221984018be4'),
 'preprocess_CODE':(str(R.parent/'query_visibility_frozen_rgb_head_20261023_v105/frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py'),'cc1e2c331d41aaea5b3d42d144c9abbc40231dd3784d24f52dcc58f41d607578'),
 'root_protocol':(str(R/'V235_ROOT_PUBLIC_COUNTERFACTUAL/PROTOCOL_BEFORE_DONOR_PIXELS_OR_TREATMENT.json'),'61a354e42b5cf9c85ca3ff554de85750afafaad456f730e307e9cf1e58474880')}


def safe(value):
    text=str(value)
    if any(x in text.casefold() for x in ('v112','__pycache__','.pyc','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation')):
        raise ValueError('Forbidden path before filesystem access')
    path=Path(os.path.abspath(text))
    for part in reversed((path,*path.parents)):
        if part.is_symlink():raise ValueError('Symlink rejected: '+str(part))
    return path


def read(path):
    fd=os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:return stream.read()


def identity(b):
    return os.path.abspath(b['path']),b['sha256']


def verify(b,whitelist,pin=None):
    if identity(b) not in {identity(x) for x in whitelist}:raise ValueError('Binding outside exact root whitelist')
    if pin is not None and identity(b)!=pin:raise ValueError('Pinned path/SHA mismatch')
    digest=hashlib.sha256();size=0
    fd=os.open(safe(b['path']),os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk);size+=len(chunk)
    if digest.hexdigest()!=b['sha256'] or ('size_bytes' in b and size!=b['size_bytes']):raise ValueError('Changed bound bytes')
    return safe(b['path'])


def binding(path):
    raw=read(path)
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}


def write_json(path,obj):
    fd=os.open(safe(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(obj,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return binding(path)


def load_code(b,whitelist,name,pin=None):
    path=verify(b,whitelist,pin);module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(read(path),str(path),'exec'),module.__dict__)
    return module


def load_sources(e):
    crop=load_code(e['crop_module'],e['code_whitelist'],'crop_modules',PINS['crop_module'])
    source=load_code(e['source_trainer'],e['code_whitelist'],'v235_original_source_trainer',PINS['source_trainer'])
    runtime=load_code(e['prefix_runtime'],e['code_whitelist'],'v235_original_prefix_runtime',PINS['prefix_runtime'])
    return source,crop,runtime


def load_preprocess(e):
    path=verify(e['preprocess_CODE'],e['code_whitelist'],PINS['preprocess_CODE'])
    nodes=[n for n in ast.parse(read(path)).body if isinstance(n,ast.FunctionDef) and n.name=='preprocess_views']
    if len(nodes)!=1:raise ValueError('Exactly pinned pure preprocess_views AST')
    namespace={'torch':torch,'F':F,'MEAN':(.485,.456,.406),'STD':(.229,.224,.225)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['preprocess_views']


def build_augmented_draws(draws,data,eligible,parents,success):
    """Pure original32 route + PCG235 COVER replacements; no feature/data I/O."""
    expected=np.flatnonzero(eligible&(data['target']==0))
    if parents.dtype!=np.int64 or parents.shape!=expected.shape or not np.array_equal(np.sort(parents),expected):raise ValueError('Every FIT-D parent exactly once, preserving declared pair-array order')
    if success.dtype!=bool or success.shape!=parents.shape:raise ValueError('Exact success mask, failures retained')
    attempt_for_parent=np.full(len(eligible),-1,np.int64)
    attempt_for_parent[parents[success]]=np.flatnonzero(success)
    pools=[np.flatnonzero(success&(data['family'][parents]==family)) for family in (0,1)]
    if any(len(pool)==0 for pool in pools):raise ValueError('Empty successful FIT-D COVER pool; no fallback semantic E')
    pair_indices=np.full(draws.shape,-1,np.int64);variants=np.full(draws.shape,-1,np.int8);actual_parent=draws.copy()
    rng=np.random.Generator(np.random.PCG64(235))
    for step,indices in enumerate(draws):
        if indices.shape!=(32,) or not eligible[indices].all():raise ValueError('Original FIT known available draw only')
        for family,offset in ((0,0),(1,16)):
            if not ((data['family'][indices[offset:offset+16]]==family).all() and (data['target'][indices[offset:offset+8]]==0).all() and (data['target'][indices[offset+8:offset+16]]==1).all()):
                raise ValueError('Fixed original family8D8E draw layout')
            # First4 D original; last4 same-parent SHAM if successful, else original.
            for position in range(offset+4,offset+8):
                attempt=attempt_for_parent[indices[position]]
                if attempt>=0:pair_indices[step,position]=attempt;variants[step,position]=1
            # First4 E original; last4 COVER from independent successful FIT-D pool.
            chosen=rng.choice(pools[family],4,replace=True)
            pair_indices[step,offset+12:offset+16]=chosen;variants[step,offset+12:offset+16]=0
            actual_parent[step,offset+12:offset+16]=parents[chosen]
    changed=pair_indices>=0
    if not eligible[actual_parent].all() or not (data['family'][actual_parent]==data['family'][draws]).all():raise ValueError('No role/family transfer')
    if not success[pair_indices[changed]].all() or not (data['target'][actual_parent[changed]]==0).all():raise ValueError('Only successful FIT-D pair images')
    return pair_indices,variants,actual_parent


def augmentation_draw_sha(pair_indices,variants,actual_parent):
    h=hashlib.sha256()
    for key,value in [('pair_indices',pair_indices.astype('<i8')),('variants',variants.astype('i1')),('actual_parent',actual_parent.astype('<i8'))]:
        h.update(key.encode());h.update(value.tobytes())
    return h.hexdigest()


def mix_draw_features(original_features,fit_map,pair_features,compact,indices,pair_indices,variants,device):
    if np.any(fit_map[indices]<0):raise ValueError('Original feature fetch only from FIT cache')
    x=original_features[torch.as_tensor(fit_map[indices],device=device)].clone()
    augmented=pair_indices>=0
    if augmented.any():
        selected=compact[pair_indices[augmented]]
        if (selected<0).any() or not np.isin(variants[augmented],[0,1]).all():raise ValueError('Only successful compact COVER0/SHAM1 features')
        x[torch.as_tensor(augmented,device=device)]=pair_features[torch.as_tensor(selected,device=device),torch.as_tensor(variants[augmented].astype(np.int64),device=device)]
    return x


def parse_control_rows(path,data):
    with os.fdopen(os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW),'r',newline='') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=len(data['target']):raise ValueError('Original CONTROL CSV row conservation')
    parsed=[];expected=np.zeros((len(rows),2),np.float32)
    for i,row in enumerate(rows):
        available=row['available']=='True';family=int(row['family']);target=int(row['target'])
        if int(row['sample_index'])!=i or row['row_id']!=str(data['row_id'][i]) or family!=int(data['family'][i]) or target!=int(data['target'][i]) or available!=bool(data['available'][i]) or row['role']!=('FIT' if data['fit'][i] else 'DEV'):
            raise ValueError('Original CONTROL CSV exact metadata identity')
        values=[row['logit0'],row['logit1']]
        if available:
            expected[i]=np.array([float(x) for x in values],np.float32)
            if not np.isfinite(expected[i]).all():raise ValueError('Finite original CONTROL both logits')
            value=float(row['selected_family_logit'])
            if np.float32(value)!=expected[i,family] or row['prediction']!=('E' if value>=0 else 'D'):raise ValueError('Original CONTROL family decode')
        elif any(values) or row['selected_family_logit'] or row['prediction']!='U':raise ValueError('Unavailable CONTROL remains null/U')
        parsed.append({'sample_index':i,'row_id':row['row_id'],'family':family,'role':row['role'],'available':available,'target':target,'prediction':row['prediction']})
    return parsed,expected


def copy_bound(src,dst):
    fd=os.open(safe(dst),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(os.open(safe(src),os.O_RDONLY|os.O_NOFOLLOW),'rb') as inp,os.fdopen(fd,'wb') as out:
        for chunk in iter(lambda:inp.read(8*1024*1024),b''):out.write(chunk)
    return binding(dst)


def run(entry_path):
    e=json.loads(read(entry_path))
    if e.get('approved_data_only_training') is not True or e.get('source_only') is not True or e.get('automatic_retry') is not False:raise ValueError('New explicit root SOURCE training entry required')
    if e['steps']!=1000 or e['threshold_logit']!=0 or e.get('original_half_up_reference_equivalence_confirmed') is not True:raise ValueError('Fixed1000/zero threshold and root original-reference equivalence QA')
    codes=e['code_whitelist'];payload=e['payload_whitelist'];meta=e['metadata_whitelist']
    verify(e['code'],codes,(str(safe(__file__)),binding(__file__)['sha256']))
    verify(e['root_protocol'],meta,PINS['root_protocol'])
    verify(e['root_scope'],meta)
    verify(e['implementation_protocol'],codes)
    metadata={k:json.loads(read(verify(e[k],meta))) for k in ('source_complete','source_entry','prefix_entry','pairs_complete','pair_QA')}
    original=metadata['source_complete'];pairs=metadata['pairs_complete'];prefix_entry=metadata['prefix_entry']
    if not all(metadata[k].get('completed') is True and metadata[k].get('PASS') is True for k in ('source_complete','pairs_complete','pair_QA')):raise ValueError('SOURCE completion and independent pair QA PASS required')
    if identity(original['entry'])!=identity(e['source_entry']):raise ValueError('Original source entry closure')
    for key,oldkey in [('control_checkpoint','checkpoints'),('control_log','logs'),('control_csv','outputs')]:
        if identity(e[key])!=identity(original[oldkey][CONTROL]):raise ValueError('Exact original CONTROL reuse binding')
    for key in ('prefix_npy','samples_npz'):
        if identity(e[key])!=identity(original[key]):raise ValueError('Same original packet/cache')
    for key,name in [('aug_parents','parent_indices.npy'),('aug_success','success.npy'),('aug_rgb','PAIR_RGB96.npy')]:
        if identity(e[key])!=identity(pairs['outputs'][name]):raise ValueError('Exact constructed pair packet binding')
    if identity(e['original_valid96'])!=identity(pairs['source_bindings']['valid96']) or identity(e['samples_npz'])!=identity(pairs['source_bindings']['samples']):raise ValueError('Pair original valid/metadata identity')
    if identity(e['preprocess_CODE'])!=identity(prefix_entry['preprocess_CODE']):raise ValueError('Original preprocessing code identity')
    if identity(e['original_valid96'])!=identity(prefix_entry['data_bindings']['valid96']) or identity(e['samples_npz'])!=identity(prefix_entry['data_bindings']['samples']):raise ValueError('Original prefix pixel-support/metadata lineage')
    model_config=e['model_loader_config']
    if identity(model_config['official_state'])!=identity(original['official_base']) or model_config['asset_identity']!=original['asset_identity']:raise ValueError('Same original fixed official encoder')
    source,crop,runtime=load_sources(e);source.configure_runtime();preprocess=load_preprocess(e)
    if identity(model_config['crop_module'])!=PINS['crop_module'] or identity(model_config['prefix_runtime'])!=PINS['prefix_runtime']:raise ValueError('Unchanged loader feature modules')
    # All source bodies are gated above. Whole prefix/NPY SHA is opaque streaming.
    paths={k:verify(e[k],payload) for k in ('control_checkpoint','control_log','control_csv','prefix_npy','samples_npz','original_valid96','aug_parents','aug_success','aug_rgb')}
    prefix=np.load(paths['prefix_npy'],mmap_mode='r',allow_pickle=False)
    if prefix.dtype!=np.float32 or prefix.ndim!=3 or tuple(prefix.shape[1:])!=(261,384):raise ValueError('Original prefix header')
    with np.load(paths['samples_npz'],allow_pickle=False) as z:data=source.validate_samples({k:z[k] for k in z.files},len(prefix))
    draws,eligible=source.fixed_draws(data);draw_sha=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    parents=np.load(paths['aug_parents'],allow_pickle=False);success=np.load(paths['aug_success'],allow_pickle=False)
    pair_indices,variants,actual_parents=build_augmented_draws(draws,data,eligible,parents,success)
    aug_draw_sha=augmentation_draw_sha(pair_indices,variants,actual_parents)
    rgb=np.load(paths['aug_rgb'],mmap_mode='r',allow_pickle=False);valid=np.load(paths['original_valid96'],mmap_mode='r',allow_pickle=False)
    if rgb.dtype!=np.uint8 or rgb.shape!=(len(parents),2,3,96,96) or valid.dtype!=bool or valid.shape!=(len(data['target']),1,96,96):raise ValueError('Exact original-valid/paired RGB schema')
    control=torch.load(paths['control_checkpoint'],weights_only=True,map_location='cpu')
    control_records=[json.loads(line) for line in read(paths['control_log']).decode().splitlines()]
    control_rows,control_expected=parse_control_rows(paths['control_csv'],data)
    initial=source.cpu_copy(control['initial_head_state_dict']);initial_sha=source.tensor_state_sha(initial)
    if initial_sha!=original['head_initial_sha256'] or draw_sha!=original['draws_sha256']:raise ValueError('Same original initial head and1000draw ledger')
    out=safe(e['output_directory'])
    if not out.is_relative_to(R) or 'v235' not in str(out).casefold() or out.exists():raise ValueError('Fresh owned V235 output only')
    out.mkdir();started=time.monotonic()
    write_json(out/'STARTED.json',{'entry':binding(entry_path),'source_only':True,'steps':1000,'automatic_retry':False,'original_draw_sha256':draw_sha,'CF_draw_sha256':aug_draw_sha})
    try:
        model=source.load_backbone(model_config,e['device'],treatment=False)
        if any(p.requires_grad for p in model.parameters()):raise ValueError('No encoder/adapter training')
        base_sha=source.tensor_state_sha(model.state_dict(),base=True)
        reuse_config={'official_state':model_config['official_state'],'reuse_control':{'original_entry':original['control_reuse']['original_entry']}}
        source.validate_control_reuse(control,control_records,initial,initial_sha,draws,draw_sha,base_sha,reuse_config)
        copies={}
        for name,key in [('checkpoint','control_checkpoint'),('log','control_log'),('CSV','control_csv')]:
            destination=out/(CONTROL+{'checkpoint':'_FINAL1000.pt','log':'_TRAIN1000.jsonl','CSV':'_ALL_SOURCE_ROWS.csv'}[name])
            copies[name]=copy_bound(paths[key],destination)
            if copies[name]['sha256']!=e[key]['sha256']:raise ValueError('CONTROL reused bytes changed')
        reuse=write_json(out/'CONTROL_REUSE_QA.json',{'completed':True,'PASS':True,'original_checkpoint':e['control_checkpoint'],'original_log':e['control_log'],'original_CSV':e['control_csv'],'copied':copies,'new_CONTROL_optimizer_updates':0,'same_initial_head_sha256':initial_sha,'same_original_draw_sha256':draw_sha,'same_base_sha256':base_sha})
        store=source.PrefixViews(prefix,data);fit=np.flatnonzero(eligible);fit_map=np.full(len(eligible),-1,np.int64);fit_map[fit]=np.arange(len(fit))
        pieces=[]
        for first in range(0,len(fit),64):
            pieces.append(source.features_for(fit[first:first+64],store,model,runtime,None,e['device'],False).detach())
            if first%1024==0:print(json.dumps({'stage':'original_FIT_features','rows':min(first+64,len(fit)),'total':len(fit)}),flush=True)
        original_fit_features=torch.cat(pieces);del pieces
        successful=np.flatnonzero(success);compact=np.full(len(parents),-1,np.int64);compact[successful]=np.arange(len(successful))
        image_attempt=np.repeat(successful,2);image_variant=np.tile(np.array([0,1],np.int64),len(successful));newpieces=[]
        for first in range(0,len(image_attempt),32):
            attempts=image_attempt[first:first+32];variant=image_variant[first:first+32];parent=parents[attempts]
            if not eligible[parent].all() or not (data['target'][parent]==0).all():raise ValueError('CF pixels from successful FIT-D parents only')
            pixels=torch.from_numpy(np.array(rgb[attempts,variant],copy=True))
            support=torch.from_numpy(np.array(valid[parent],copy=True))
            views,fractions=preprocess(pixels,support)
            if not torch.equal(fractions,torch.from_numpy(data['fractions'][parent])) or not np.array_equal(support.numpy()[:,0,47:49,47:49].all((1,2)),data['central4'][parent]):raise ValueError('Original fractions/central4 unchanged')
            with torch.no_grad():
                output=model.forward_features(views.to(e['device']));n=len(parent)
                full={k:v[:n] for k,v in output.items() if isinstance(v,torch.Tensor)};centre={k:v[n:] for k,v in output.items() if isinstance(v,torch.Tensor)}
                x=crop.crop_features(full,centre,torch.as_tensor(data['fractions'][parent],device=e['device']),torch.as_tensor(data['central4'][parent],device=e['device']),torch.as_tensor(data['condition'][parent],device=e['device']),None)
            newpieces.append(x.detach())
            if first%1024==0 or first+32>=len(image_attempt):print(json.dumps({'stage':'new_CF_features','images':min(first+32,len(image_attempt)),'total':len(image_attempt)}),flush=True)
        pair_features=torch.cat(newpieces).reshape(len(successful),2,1186);del newpieces
        if source.tensor_state_sha(model.state_dict(),base=True)!=base_sha:raise ValueError('Frozen feature extraction changed base')
        ledger_path=out/'TREATMENT_DRAW_SOURCE1000.jsonl'
        with ledger_path.open('x') as stream:
            for i,indices in enumerate(draws):stream.write(json.dumps({'update':i+1,'original_sample_indices':indices.tolist(),'actual_parent_indices':actual_parents[i].tolist(),'augmentation_attempt_indices':pair_indices[i].tolist(),'augmentation_variant_minus1Original_0COVER_1SHAM':variants[i].tolist(),'family':data['family'][indices].tolist(),'target':data['target'][indices].tolist(),'original_draws_sha256':draw_sha,'CF_draws_sha256':aug_draw_sha},allow_nan=False)+'\n')
        head=crop.CropFamilyHead().float().to(e['device']);head.load_state_dict(initial,strict=True)
        if sum(p.numel() for p in head.parameters())!=76098 or source.tensor_state_sha(head.state_dict())!=initial_sha:raise ValueError('Exact original initial76098 head')
        optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4);head.train();log_path=out/(TREAT+'_TRAIN1000.jsonl')
        with log_path.open('x') as stream:
            for step,indices in enumerate(draws):
                x=mix_draw_features(original_fit_features,fit_map,pair_features,compact,indices,pair_indices[step],variants[step],e['device'])
                optimizer.zero_grad(set_to_none=True)
                loss,parts=crop.family_loss(head(x),torch.as_tensor(data['target'][indices],device=e['device']),torch.as_tensor(data['family'][indices],device=e['device']))
                if not torch.isfinite(loss):raise ValueError('Finite original family-balanced objective')
                loss.backward();optimizer.step()
                stream.write(json.dumps({'update':step+1,'original_sample_indices':indices.tolist(),'actual_parent_indices':actual_parents[step].tolist(),'augmentation_attempt_indices':pair_indices[step].tolist(),'augmentation_variant':variants[step].tolist(),'family0_BCE':float(parts[0].detach()),'family1_BCE':float(parts[1].detach()),'total_loss':float(loss.detach()),'original_draws_sha256':draw_sha,'CF_draws_sha256':aug_draw_sha,'all_FIT_available_known':True,'family_counts':[16,16]},allow_nan=False)+'\n')
                if (step+1)%100==0:print(json.dumps({'stage':'head_fit','updates':step+1,'fixed_final':1000}),flush=True)
        if len(optimizer.state)!=4 or any(float(state['step'])!=1000 or any(not torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)) for state in optimizer.state.values()):raise ValueError('Four finite head optimizer states at exact1000')
        if any(not torch.isfinite(p).all() for p in head.parameters()) or source.tensor_state_sha(model.state_dict(),base=True)!=base_sha:raise ValueError('Finite head and unchanged base')
        checkpoint={'arm':TREAT,'updates':1000,'use_lora':False,'head_parameters':76098,'adapter_parameters':0,'head_state_dict':source.cpu_copy(head.state_dict()),'optimizer_state_dict':source.cpu_copy(optimizer.state_dict()),'initial_head_state_dict':initial,'initial_head_sha256':initial_sha,'official_base':model_config['official_state'],'base_state_sha256_before_after':base_sha,'original_draws_sha256':draw_sha,'CF_draws_sha256':aug_draw_sha,'entry':binding(entry_path),'source_referents':{'0':'SYN witness0 MAIN','1':'YCB associated-r AUX'}}
        checkpoint_path=out/(TREAT+'_FINAL1000.pt');torch.save(checkpoint,checkpoint_path);final_head_sha=source.tensor_state_sha(head.state_dict())
        freeze=write_json(out/'FINAL1000_FROZEN_BEFORE_NEW_HEAD_DEV_FORWARD.json',{'completed':True,'PASS':True,'checkpoints':{CONTROL:copies['checkpoint'],TREAT:binding(checkpoint_path)},'CONTROL_reuse_QA':reuse,'new_CONTROL_optimizer_updates':0,'new_TREAT_optimizer_updates':1000,'same_initial_head_sha256':initial_sha,'same_original_draws_sha256':draw_sha,'CF_draws_sha256':aug_draw_sha,'base_sha256':base_sha,'new_head_DEV_forward_before_freeze':0,'original_prefix_numeric_rows_before_freeze_only_FIT':True,'original_FIT_sample_indices':sorted(store.prebarrier_sample_indices),'original_FIT_view_indices':sorted(store.prebarrier_view_indices),'CF_successful_attempt_indices_only_FIT':successful.tolist()})
        store.final_barrier=True;head.eval().requires_grad_(False);control_head=crop.CropFamilyHead().float().to(e['device']);control_head.load_state_dict(control['head_state_dict'],strict=True);control_head.eval().requires_grad_(False)
        all_indices=np.flatnonzero(data['available']);scores={};replayed=hashlib.sha256();expected_hash=hashlib.sha256()
        with torch.no_grad():
            for first in range(0,len(all_indices),64):
                indices=all_indices[first:first+64]
                x=source.features_for(indices,store,model,runtime,None,e['device'],False)
                actual=control_head(x).cpu().numpy();expected=control_expected[indices]
                if not np.array_equal(actual,expected) or actual.tobytes()!=expected.tobytes():
                    write_json(out/'CONTROL_REPLAY_FAILURE.json',{'completed':False,'PASS':False,'sample_indices':indices.tolist(),'max_abs_logit_difference':float(np.max(np.abs(actual-expected))),'byte_equal':False,'no_tolerance_relaxation_or_CONTROL_training':True})
                    raise ValueError('Original CONTROL both-logit byte replay mismatch; stop, no retraining')
                replayed.update(actual.tobytes());expected_hash.update(expected.tobytes())
                z=head(x).cpu().numpy()
                if not np.isfinite(z).all():raise ValueError('Finite frozen available logits')
                scores.update({int(i):v for i,v in zip(indices,z)})
        replay=write_json(out/'CONTROL_ALL_SOURCE_REPLAY_QA.json',{'completed':True,'PASS':True,'all_packet_rows':len(data['target']),'available_replayed':len(all_indices),'both_logits_byte_exact':True,'actual_FP32_logit_bytes_sha256':replayed.hexdigest(),'original_CSV_FP32_logit_bytes_sha256':expected_hash.hexdigest(),'original_features_for_batch64_schedule':True,'original_CSV_copied_bytes':copies['CSV']})
        rows=[]
        for i in range(len(data['target'])):
            family=int(data['family'][i]);z=scores.get(i);value=None if z is None else float(z[family])
            rows.append({'sample_index':i,'row_id':str(data['row_id'][i]),'family':family,'role':'FIT' if data['fit'][i] else 'DEV','available':bool(data['available'][i]),'target':int(data['target'][i]),'logit0':None if z is None else float(z[0]),'logit1':None if z is None else float(z[1]),'selected_family_logit':value,'prediction':'U' if value is None else 'E' if value>=0 else 'D'})
        csv_path=out/(TREAT+'_ALL_SOURCE_ROWS.csv')
        with csv_path.open('x',newline='') as stream:writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        if source.tensor_state_sha(head.state_dict())!=final_head_sha or source.tensor_state_sha(model.state_dict(),base=True)!=base_sha:raise ValueError('Frozen output state changed')
        result=write_json(out/'SOURCE_RESULTS.json',{'completed':True,'PASS':True,'arms':{CONTROL:source.reduce_rows(control_rows),TREAT:source.reduce_rows(rows)},'two_referents_not_merged':True,'all_original_rows_preserved':True,'new_current_inference':0,'model_or_threshold_selection':False})
        complete={'completed':True,'PASS':True,'entry':binding(entry_path),'code':e['code'],'root_protocol':e['root_protocol'],'root_scope':e['root_scope'],'implementation_protocol':e['implementation_protocol'],'source_complete':e['source_complete'],'pairs_complete':e['pairs_complete'],'pair_QA':e['pair_QA'],'original_half_up_reference_equivalence_confirmed':True,'source_trainer':e['source_trainer'],'crop_module':e['crop_module'],'prefix_runtime':e['prefix_runtime'],'preprocess_CODE':e['preprocess_CODE'],'official_base':model_config['official_state'],'asset_identity':model_config['asset_identity'],'prefix_npy':e['prefix_npy'],'samples_npz':e['samples_npz'],'CF_feature_mapping':{'parent_attempts':len(parents),'successful':len(successful),'failed_masked':int((~success).sum()),'compact_successful_attempt_order':successful.tolist(),'variant0':'COVER targetE','variant1':'SHAM targetD'},'checkpoints':{CONTROL:copies['checkpoint'],TREAT:binding(checkpoint_path)},'logs':{CONTROL:copies['log'],TREAT:binding(log_path)},'draw_source_ledger':binding(ledger_path),'original_draws_sha256':draw_sha,'CF_draws_sha256':aug_draw_sha,'initial_head_sha256':initial_sha,'base_state_sha256_before_after':base_sha,'new_CONTROL_optimizer_updates':0,'new_TREAT_optimizer_updates':1000,'freeze':freeze,'CONTROL_reuse_QA':reuse,'CONTROL_replay_QA':replay,'outputs':{CONTROL:copies['CSV'],TREAT:binding(csv_path)},'summary':result,'new_head_DEV_forward_before_freeze':0,'numeric_original_prefix_before_freeze_only_FIT':True,'CF_feature_reads_only_successful_FIT':True,'no_LoRA_or_new_input_channels':True,'original_source_RGB_or_GT_or_protected_payload_reads':0,'automatic_retry':False,'actual_seconds':time.monotonic()-started}
        write_json(out/'COMPLETE.json',complete);print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(out/'COMPLETE.json')}),flush=True)
    except BaseException:
        write_json(out/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entry',required=True);run(parser.parse_args().entry)
