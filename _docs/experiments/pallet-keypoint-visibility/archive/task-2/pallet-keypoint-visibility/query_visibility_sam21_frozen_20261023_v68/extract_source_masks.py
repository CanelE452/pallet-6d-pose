#!/usr/bin/env python3
"""Locked, frozen, box-only SAM2.1 source mask extraction.

No source manifest/CSV/RGB is opened before the immutable ENTRYLOCK, code,
protocol, runtime and input-seal checks pass. GT/depth/pose/features are never
loaded. --mock uses generated scalar/logit arrays only, with no model forward.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT.parent/'foundation_runtime_sam21_20261023'
PROTOCOL_SHA='1e6a99030faf8b2c6cd1144b70b09395d55609c2b746c9fd7d12ad6f7e622128'
RUNTIME_SEAL_SHA='e5655309c2c99ee18e76fa0ebe85541194445895673d9b64f915384130bcbf82'
INPUT_SEAL_SHA='081c4a58cce0a20830872358d1559b59ef75a57df333bdc357a164d35abec979'
MANIFEST_SHA='0a347a477e8d1d69ab0e9b956103f9b657b4c4b9c4505242d76849f8e25a3fc8'
CHECKPOINT_SHA='a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'
DENY_SHA='58ef8b2ca79b9cb511fe7b5813d22b31f6ff82ef5049a2be340ec293bdbef987'
SOURCES=('SYNTH','ORIGINAL_YCB','CONTROLLED_YCB','PALLOX')
TARGET_KEYS={'id','global_id','joined_target_index','source','source_metadata_label','group_id','split',
             'image','image_sha256','raw_hw','box','side','queries'}
QUERY_KEYS={'sample_id','joined_query_index','native_corner','query','pixel','native_input_valid'}
MAP_KEYS=('joined_query_index','joined_target_index','sample_id','global_id','source','group_id','native_corner',
          'query_x','query_y','pixel_x','pixel_y','native_input_valid','side','image','image_sha256')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def ref(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)


def require(condition,message):
    if not condition:raise ValueError(message)


def read_json(path):return json.loads(Path(path).read_text())


def bound(binding,expected_path=None,expected_sha=None):
    p=Path(binding['path']).resolve()
    if expected_path is not None:require(p==Path(expected_path).resolve(),'Unexpected bound path: '+str(p))
    if expected_sha is not None:require(binding['sha256']==expected_sha,'Unapproved digest: '+str(p))
    require(sha(p)==binding['sha256'],'Changed bound bytes: '+str(p));return p


def boolean(value):
    if value in (True,'True','true',1,'1'):return True
    if value in (False,'False','false',0,'0'):return False
    raise ValueError('Explicit boolean required')


def validate_entry_lock(path,out):
    """Metadata/code verification precedes every source-file body access."""
    require(path is not None,'A root-frozen ENTRYLOCK is mandatory before source access')
    lock=read_json(path)
    require(lock['schema']=='V68_FROZEN_SAM21_SOURCE_EXTRACTION_ENTRYLOCK_V1','Unknown entry lock schema')
    for key,value in dict(approved=True,source_only=True,current223_allowed=False,model_training_allowed=False).items():
        require(lock.get(key) is value,'Invalid entry lock scope: '+key)
    bound(lock['extractor'],__file__)
    require(Path(lock['output_directory']).resolve()==Path(out).resolve(),'Different locked output directory')
    require(not Path(out).exists(),'Preserve existing source output directory')
    require(Path(out).resolve().is_relative_to(ROOT),'Source outputs must remain inside V68')
    require(lock['expected']==dict(targets=1091,queries=8590),'Fixed source cohort changed')
    require(lock['seed']==20261023,'Fixed runtime seed expected')
    require(lock['runtime']==dict(python=str(RUNTIME/'venv/bin/python'),device='cuda',autocast='bfloat16',batch_size=1,compile=False),
            'Different source runtime recipe')
    require(Path(sys.prefix).resolve()==RUNTIME/'venv' and sys.prefix!=sys.base_prefix,'Use isolated SAM2 runtime')
    require(Path(sys.executable).absolute()==RUNTIME/'venv/bin/python','Locked Python executable required')
    inputs=lock['inputs']
    require(set(inputs)=={'source_protocol','runtime_seal','input_seal','manifest','query_id_map','checkpoint','denylist'},'Unexpected/missing input roles')
    pp=bound(inputs['source_protocol'],ROOT/'PROTOCOL.json',PROTOCOL_SHA);protocol=read_json(pp)
    rp=bound(inputs['runtime_seal'],RUNTIME/'RUNTIME_SEAL.json',RUNTIME_SEAL_SHA);runtime=read_json(rp)
    require(runtime['READY'] is True and runtime['repo_revision']==protocol['model']['code_revision'],'Runtime/model recipe mismatch')
    require(protocol['model']['torch_compile'] is False and protocol['model']['training'] is False,'Frozen no-compile/no-train recipe required')
    require(protocol['provenance']['no_current_access'] is True and protocol['provenance']['no_current_inference_authorization'] is True,'Source protocol cannot allow current evaluation')
    for binding in runtime['bindings']:
        require(Path(binding['path']).resolve().is_relative_to(RUNTIME),'Runtime binding escapes isolated bundle')
        bound(binding)
    for binding in runtime['wheels']:bound(binding)
    for package,version in runtime['pinned_versions'].items():
        require(importlib.metadata.version(package)==version,'Runtime package drift: '+package)
    cp=bound(inputs['checkpoint'],RUNTIME/'checkpoints/sam2.1_hiera_base_plus.pt',CHECKPOINT_SHA)
    require(cp.stat().st_size==323606802 and runtime['checkpoint']['sha256']==CHECKPOINT_SHA,'Official base-plus checkpoint mismatch')
    dp=bound(inputs['denylist'],ROOT.parent/'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json',DENY_SHA)
    deny=set(read_json(dp)['image_sha256']);require(len(deny)==223,'Complete 223-image SHA exclusion list required')
    sp=bound(inputs['input_seal'],ROOT/'source_inputs/SEAL.json',INPUT_SEAL_SHA);input_seal=read_json(sp)
    require(input_seal.get('completed',input_seal.get('complete')) is True,'Complete source input seal required')
    # Source identity files are still unopened here; their locked digests/paths
    # must first be registered by the source-input seal.
    require(inputs['manifest']['sha256']==MANIFEST_SHA,'Frozen source manifest digest changed')
    for role,name in (('manifest','INPUTS_RGB_ONLY.jsonl'),('query_id_map','QUERY_ID_MAP.csv')):
        expected=ROOT/'source_inputs'/name
        require(Path(inputs[role]['path']).resolve()==expected,'Different source identity input path')
        hits=[b for b in input_seal['outputs'] if Path(b['path']).resolve()==expected and b['sha256']==inputs[role]['sha256']]
        require(len(hits)==1,'Identity input not bound exactly once by source seal: '+role)
    return lock,protocol,runtime,deny,cp


def native_pixel(query,hw):
    q=np.asarray(query,dtype=np.float64);require(q.shape==(2,),'A native query must be xy')
    h,w=map(int,hw);require(h>0 and w>0,'Positive native H/W required')
    if not np.isfinite(q).all():return False,(-1,-1),'NONFINITE_QUERY'
    x,y=map(int,np.floor(q+.5))
    if not(0<=q[0]<w and 0<=q[1]<h):return False,(x,y),'CONTINUOUS_OUTSIDE_NATIVE'
    if not(0<=x<w and 0<=y<h):return False,(x,y),'ROUNDED_OUTSIDE_NATIVE'
    return True,(x,y),'VALID'


def sample_query(chosen,query,recorded_valid,recorded_pixel):
    """Native exact and clipped5x5 max; strict >0, zero is nonvisible."""
    a=np.asarray(chosen)
    require(a.ndim==2 and a.dtype==np.float32 and np.isfinite(a).all(),'Finite native float32 chosen logits required')
    valid,(x,y),reason=native_pixel(query,a.shape)
    require(valid==boolean(recorded_valid),'Recorded native-input validity disagrees with actual H/W/query')
    require([x,y]==list(map(int,recorded_pixel)),'Recorded query pixel differs from floor(q+.5)')
    if not valid:
        return dict(native_valid=False,pixel_x=x,pixel_y=y,status='U',invalid_reason=reason,
            exact_logit=float('nan'),radius2_logit=float('nan'),exact_visible=-1,radius2_visible=-1,
            radius2_window_pixels=0,radius2_clipped=False)
    h,w=a.shape;x0,x1=max(0,x-2),min(w,x+3);y0,y1=max(0,y-2),min(h,y+3)
    exact=float(a[y,x]);near=float(a[y0:y1,x0:x1].max())
    return dict(native_valid=True,pixel_x=x,pixel_y=y,status='VALID',invalid_reason='',
        exact_logit=exact,radius2_logit=near,exact_visible=int(exact>0),radius2_visible=int(near>0),
        radius2_window_pixels=(x1-x0)*(y1-y0),radius2_clipped=(x1-x0)*(y1-y0)!=25)


def select_mask(masks,quality,hw):
    masks=np.asarray(masks);quality=np.asarray(quality)
    require(masks.shape==(3,int(hw[0]),int(hw[1])) and quality.shape==(3,),'Three full-native mask logits/quality scores required')
    require(masks.dtype==np.float32,'Official returned full-native logits must remain float32, with no half-storage roundtrip')
    require(np.isfinite(masks).all() and np.isfinite(quality).all(),'Finite predicted masks/quality required')
    selected=int(np.argmax(quality))
    return np.ascontiguousarray(masks[selected],dtype=np.float32),np.ascontiguousarray(quality,dtype=np.float32),selected


def load_source_identity(lock,protocol,deny):
    """Only after ENTRYLOCK: open the two hashed RGB-only identity files."""
    mp=bound(lock['inputs']['manifest']);qp=bound(lock['inputs']['query_id_map'])
    with mp.open() as f:targets=[json.loads(line) for line in f if line.strip()]
    with qp.open(newline='') as f:
        reader=csv.DictReader(f);require(tuple(reader.fieldnames)==MAP_KEYS,'Unexpected query-ID-map columns')
        mapping=list(reader)
    require(len(targets)==1091 and len(mapping)==8590,'Frozen source cohort counts')
    require([int(r['joined_query_index']) for r in mapping]==list(range(8590)),'Source query map must preserve exact joined-array order')
    ids=set();target_ids=set();indices=set();source_counts=Counter();target_counts=Counter();native_valid=0
    for target in targets:
        require(set(target)==TARGET_KEYS,'Only frozen RGB/box/query identity fields may enter a source target')
        require(target['id']==target['global_id'] and target['global_id'] not in target_ids,'Unique identical target/global IDs required')
        target_ids.add(target['global_id'])
        ti=int(target['joined_target_index']);require(ti not in indices and 0<=ti<1091,'Unique joined target index required');indices.add(ti)
        source=target['source'];require(source in SOURCES and target['split'] in ('VAL','VALIDATION'),'Source validation-only cohort required')
        target_counts[source]+=1;require(target['image_sha256'] not in deny,'Private RGB SHA in source identities')
        require(isinstance(target['raw_hw'],list) and len(target['raw_hw'])==2,'Original native H/W required')
        h,w=map(int,target['raw_hw']);require(h>0 and w>0,'Native shape')
        box=np.asarray(target['box'],np.float64)
        require(box.shape==(4,) and np.isfinite(box).all() and box[2]>box[0] and box[3]>box[1],'Finite nondegenerate original box')
        require(math.isfinite(float(target['side'])) and float(target['side'])>0,'Original source ROI side required')
        require(1<=len(target['queries'])<=8,'Keep only originally included queries, no SYNTH slot padding')
        corners=set()
        for q in target['queries']:
            require(set(q)==QUERY_KEYS,'Query whitelist violation')
            idx=int(q['joined_query_index']);require(0<=idx<8590,'Query index range')
            require(q['sample_id'] not in ids,'Duplicate query sample ID');ids.add(q['sample_id'])
            corner=int(q['native_corner']);require(0<=corner<8 and corner not in corners,'Unique native query corner slot');corners.add(corner)
            valid,pixel,_=native_pixel(q['query'],[h,w]);require(valid==boolean(q['native_input_valid']),'Native manifest validity mismatch')
            require(tuple(map(int,q['pixel']))==pixel,'Native manifest pixel mismatch')
            r=mapping[idx]
            for key,value in (('sample_id',q['sample_id']),('global_id',target['global_id']),('source',source),('group_id',target['group_id']),
                              ('image',target['image']),('image_sha256',target['image_sha256'])):
                require(r[key]==value,'Manifest/query-ID-map mismatch: '+key)
            require(int(r['joined_target_index'])==ti and int(r['native_corner'])==corner,'Source query target/corner mismatch')
            require(float(r['query_x'])==float(q['query'][0]) and float(r['query_y'])==float(q['query'][1]),'Source query coordinates changed')
            require(int(r['pixel_x'])==pixel[0] and int(r['pixel_y'])==pixel[1] and boolean(r['native_input_valid'])==valid,'Source map native validity/pixel mismatch')
            require(float(r['side'])==float(target['side']),'Source query ROI side changed')
            source_counts[source]+=1;native_valid+=int(valid)
    require(len(ids)==8590 and indices==set(range(1091)),'Complete unmodified source query/target identity coverage')
    require(native_valid==8584,'Frozen source native-valid count changed')
    for source,spec in protocol['fixed_inputs']['source_counts'].items():
        require(source_counts[source]==spec['queries'] and target_counts[source]==spec['targets'],'Frozen source domain counts changed')
    return targets,mapping


def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def state_digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        a=np.ascontiguousarray(tensor.detach().cpu().numpy())
        h.update(name.encode()+b'\0'+str(a.dtype).encode()+str(a.shape).encode()+b'\0'+a.tobytes())
    return h.hexdigest()


def extract(lock_path,out):
    lock,protocol,runtime,deny,checkpoint=validate_entry_lock(lock_path,out)
    targets,mapping=load_source_identity(lock,protocol,deny)
    # Model imports/construction are intentionally after every identity guard.
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    require(torch.__version__=='2.5.1+cu121' and torch.cuda.is_available(),'Frozen CUDA runtime required')
    torch.manual_seed(lock['seed']);np.random.seed(lock['seed']);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    compile_calls=0
    def no_compile(*a,**kw):
        nonlocal compile_calls
        compile_calls+=1;raise AssertionError('torch.compile is forbidden by the source protocol')
    torch.compile=no_compile;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    model=build_sam2(protocol['model']['config'],str(checkpoint),device='cuda',mode='eval',
        apply_postprocessing=False,hydra_overrides_extra=['++model.compile_image_encoder=false'])
    model.requires_grad_(False)
    require(not model.training and all(not p.requires_grad and p.dtype==torch.float32 for p in model.parameters()),'Frozen eval FP32 weights required')
    initial_state_sha=state_digest(model)
    predictor=SAM2ImagePredictor(model,mask_threshold=0.,max_hole_area=0.,max_sprinkle_area=0.)
    out=Path(out);out.mkdir(parents=True);(out/'masks').mkdir()
    bindings=dict(entry_lock=ref(lock_path),extractor=ref(__file__),source_protocol=lock['inputs']['source_protocol'],
        runtime_seal=lock['inputs']['runtime_seal'],input_seal=lock['inputs']['input_seal'],manifest=lock['inputs']['manifest'],
        query_id_map=lock['inputs']['query_id_map'],checkpoint=lock['inputs']['checkpoint'],denylist=lock['inputs']['denylist'])
    write_json(out/'RUN_STARTED.json',dict(bindings=bindings,targets=len(targets),queries=len(mapping),source_only=True,
        current_RGB_GT_scores_inputs_read=False,optimizer_updates=0,PnP_calls=0,model_parameters=sum(p.numel() for p in model.parameters())))
    (out/'extract_source_masks.py').write_bytes(Path(__file__).read_bytes())
    started=time.monotonic();query_rows=[None]*8590;target_rows=[];mask_outputs=[];decoded=0
    for number,target in enumerate(targets):
        body=Path(target['image']).read_bytes()
        digest=hashlib.sha256(body).hexdigest()
        require(digest==target['image_sha256'] and digest not in deny,'Source RGB body differs from sealed SHA')
        with Image.open(io.BytesIO(body)) as image:rgb=np.asarray(image.convert('RGB'),dtype=np.uint8).copy()
        require(list(rgb.shape[:2])==list(map(int,target['raw_hw'])) and rgb.ndim==3 and rgb.shape[2]==3,'RGB actual shape differs from manifest native H/W')
        box=np.asarray(target['box'],dtype=np.float32)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictor.set_image(rgb)
            masks,quality,_=predictor.predict(point_coords=None,point_labels=None,box=box,
                                             multimask_output=True,return_logits=True)
        chosen,quality,selected=select_mask(masks,quality,rgb.shape[:2]);decoded+=1
        samples=[]
        for q in target['queries']:
            s=sample_query(chosen,q['query'],q['native_input_valid'],q['pixel'])
            row=dict(joined_query_index=int(q['joined_query_index']),joined_target_index=int(target['joined_target_index']),
                sample_id=q['sample_id'],global_id=target['global_id'],source=target['source'],group_id=target['group_id'],
                native_corner=int(q['native_corner']),query_x=float(q['query'][0]),query_y=float(q['query'][1]),
                native_H=rgb.shape[0],native_W=rgb.shape[1],side=float(target['side']),image_sha256=digest,
                selected_mask_index=selected,predicted_quality_0=float(quality[0]),predicted_quality_1=float(quality[1]),predicted_quality_2=float(quality[2]),**s)
            require(query_rows[row['joined_query_index']] is None,'Duplicate output query position')
            query_rows[row['joined_query_index']]=row;samples.append((q,s))
        mask_path=out/'masks'/f'{number:06d}.npz'
        with mask_path.open('wb') as f:
            np.savez_compressed(f,chosen_logit=chosen,quality_scores=quality,selected_index=np.int64(selected),
                original_prompt_box=np.asarray(target['box'],np.float64),predictor_prompt_box=box,raw_hw=np.asarray(rgb.shape[:2],np.int64),
                sample_id=np.asarray([q['sample_id'] for q,_ in samples]),joined_query_index=np.asarray([q['joined_query_index'] for q,_ in samples],np.int64),
                query=np.asarray([q['query'] for q,_ in samples],np.float64),pixel=np.asarray([q['pixel'] for q,_ in samples],np.int64),
                native_valid=np.asarray([s['native_valid'] for _,s in samples],bool),
                exact_logit=np.asarray([s['exact_logit'] for _,s in samples],np.float32),radius2_logit=np.asarray([s['radius2_logit'] for _,s in samples],np.float32),
                exact_visible=np.asarray([s['exact_visible'] for _,s in samples],np.int8),radius2_visible=np.asarray([s['radius2_visible'] for _,s in samples],np.int8))
        b=ref(mask_path);mask_outputs.append(b)
        target_rows.append(dict(manifest_index=number,joined_target_index=int(target['joined_target_index']),global_id=target['global_id'],source=target['source'],
            group_id=target['group_id'],image=target['image'],image_sha256=digest,raw_hw=list(rgb.shape[:2]),box=target['box'],
            predictor_box_float32=box.tolist(),side=target['side'],query_count=len(samples),selected_mask_index=selected,
            predicted_quality=quality.tolist(),mask_output=b))
        if (number+1)%25==0 or number+1==len(targets):
            write_json(out/'PROGRESS.json',dict(targets_completed=number+1,targets_total=len(targets),seconds=time.monotonic()-started,
                current_RGB_GT_scores_inputs_read=False,optimizer_updates=0,PnP_calls=0))
            print(json.dumps(dict(targets_completed=number+1,targets_total=len(targets),seconds=time.monotonic()-started)),flush=True)
    require(all(r is not None for r in query_rows),'Missing original source query output')
    require(sum(r['native_valid'] for r in query_rows)==8584,'Source valid cohort drift')
    write_csv(out/'PER_QUERY.csv',query_rows);write_json(out/'TARGET_OUTPUTS.json',target_rows)
    # Keep invalid raw values NaN in NPZ/CSV; no finite negative masquerades as U.
    np.savez_compressed(out/'SOURCE_QUERY_OUTPUTS.npz',sample_id=np.asarray([r['sample_id'] for r in query_rows]),
        source=np.asarray([r['source'] for r in query_rows]),group_id=np.asarray([r['group_id'] for r in query_rows]),
        global_id=np.asarray([r['global_id'] for r in query_rows]),native_valid=np.asarray([r['native_valid'] for r in query_rows],bool),
        exact_logit=np.asarray([r['exact_logit'] for r in query_rows],np.float32),radius2_logit=np.asarray([r['radius2_logit'] for r in query_rows],np.float32),
        exact_visible=np.asarray([r['exact_visible'] for r in query_rows],np.int8),radius2_visible=np.asarray([r['radius2_visible'] for r in query_rows],np.int8),
        selected_mask_index=np.asarray([r['selected_mask_index'] for r in query_rows],np.int8))
    torch.cuda.synchronize()
    for binding in bindings.values():bound(binding)
    require(sha(checkpoint)==CHECKPOINT_SHA and compile_calls==0,'Frozen model/code/checkpoint changed')
    final_state_sha=state_digest(model);require(final_state_sha==initial_state_sha,'Model state changed during frozen source extraction')
    report=dict(schema='V68_FROZEN_SAM21_SOURCE_MASK_EXTRACTION_SEAL_V1',completed=True,bindings=bindings,
        targets=1091,queries=8590,native_valid=8584,native_invalid=6,source_query_counts=dict(Counter(r['source'] for r in query_rows)),
        source_target_counts=dict(Counter(t['source'] for t in targets)),model_parameters=sum(p.numel() for p in model.parameters()),
        runtime=dict(python=sys.executable,python_version=sys.version,torch=str(torch.__version__),torchvision=importlib.metadata.version('torchvision'),
            triton=importlib.metadata.version('triton'),CUDA=torch.version.cuda,device=torch.cuda.get_device_name(0),seed=lock['seed'],
            float32_weights=True,cuda_autocast='bfloat16',TF32=False,batch_targets=1,compile_calls=compile_calls,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()),
        mask_outputs=mask_outputs,outputs=[ref(out/p) for p in ('PER_QUERY.csv','TARGET_OUTPUTS.json','SOURCE_QUERY_OUTPUTS.npz','RUN_STARTED.json','extract_source_masks.py')],
        output_mask_bytes=sum(x['bytes'] for x in mask_outputs),seconds=time.monotonic()-started,
        source_RGB_decode_count=decoded,model_weights_unchanged=True,model_state_sha256=initial_state_sha,
        final_model_state_sha256=final_state_sha,checkpoint_unchanged=True,
        model_input_whitelist=['sealed original RGB','unchanged float box'],query_is_prompt=False,GT_branch_selection=False,
        mask_branch_selection='argmax predicted quality, first index on exact ties',storage='full-native chosen logits float32; no half quantization',
        exact_decision='finite native exact logit >0 strict; zero nonvisible',radius2_decision='clipped native5x5 max of same mask >0; separate diagnostic',
        invalid='NaN raw logits, visible=-1 and statusU; never padded negative',
        GT_depth_pose_features_annotations_read=False,current_RGB_GT_scores_inputs_read=False,
        source_only=True,training_calls=0,optimizer_updates=0,PnP_calls=0,current_inference_allowed=False,
        interpretation='Visible-only segmentation baseline. A negative mask/query is not a causal external, outside, self, or physical-absence classification.')
    write_json(out/'EXTRACTION_SEAL.json',report)
    print(json.dumps(dict(completed=True,seal=ref(out/'EXTRACTION_SEAL.json'),seconds=report['seconds'])),flush=True)


def mock(out):
    """Generated blank RGB and analytical native logits; zero model/data calls."""
    path=Path(out);require(not path.exists(),'Preserve prior generated mock report')
    rng=np.random.default_rng(20261023);tests=0
    for h,w in ((480,640),(540,960),(480,720),(560,560),(5,7)):
        blank=np.full((h,w,3),127,np.uint8);require(blank.shape==(h,w,3),'Generated blank shape')
        logit=rng.standard_normal((h,w)).astype(np.float32)
        for q in ((0.,0.),(.49,.49),(w-1.,h-1.),(w-.1,1.),(-.1,1.),(-1.,-1.)):
            valid,pixel,_=native_pixel(q,[h,w]);r=sample_query(logit,q,valid,pixel)
            if valid:
                x,y=pixel;window=[float(logit[yy,xx]) for yy in range(max(0,y-2),min(h,y+3)) for xx in range(max(0,x-2),min(w,x+3))]
                require(r['exact_logit']==float(logit[y,x]) and r['radius2_logit']==max(window),'Independent scalar native sampling')
            else:require(r['status']=='U' and math.isnan(r['exact_logit']) and r['exact_visible']==-1,'Invalid must stay UNKNOWN/NaN')
            tests+=1
    zero=np.zeros((5,7),np.float32);r=sample_query(zero,[3,2],True,[3,2]);require(r['exact_visible']==0 and r['radius2_visible']==0,'Zero tie must be nonvisible')
    zero[2,3]=np.nextafter(np.float32(0),np.float32(1));r=sample_query(zero,[3,2],True,[3,2]);require(r['exact_visible']==1,'Positive float32 subnormal must remain positive')
    masks=np.stack([np.full((5,7),i,np.float32) for i in range(3)])
    chosen,scores,index=select_mask(masks,np.array([.9,.9,.1],np.float32),(5,7));require(index==0 and (chosen==0).all(),'Quality argmax exact tie must choose first')
    chosen,_,index=select_mask(masks,np.array([.1,.2,.9],np.float32),(5,7));require(index==2 and (chosen==2).all(),'Quality argmax branch')
    try:sample_query(zero,[3,2],False,[3,2])
    except ValueError:pass
    else:raise AssertionError('Recorded native validity mismatch must fail')
    try:validate_entry_lock(None,ROOT/'unused_mock_output')
    except ValueError:pass
    else:raise AssertionError('Missing ENTRYLOCK must fail before source access')
    report=dict(schema='V68_GENERATED_NATIVE_SAMPLER_AND_GUARD_MOCK_V1',PASS=True,code=ref(__file__),
        generic_native_HW_cases=5,independent_scalar_query_cases=tests,strict_zero_negative_PASS=True,float32_positive_sign_preserved_PASS=True,
        quality_argmax_tie_first_PASS=True,native_continuous_and_rounded_bounds_PASS=True,invalid_UNKNOWN_NaN_PASS=True,
        native_validity_mismatch_rejected_PASS=True,missing_entrylock_rejected_before_source_access_PASS=True,
        actual_source_identity_RGB_GT_features_reads=0,current_RGB_GT_scores_inputs_read=False,model_forward_calls=0,
        optimizer_updates=0,PnP_calls=0,generated_blank_and_analytical_arrays_only=True)
    path.parent.mkdir(parents=True,exist_ok=True);write_json(path,report);print(json.dumps(dict(PASS=True,report=ref(path))),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--entry-lock',type=Path)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--mock',type=Path)
    args=ap.parse_args()
    if args.mock:
        require(args.entry_lock is None and args.out is None,'Generated mock cannot use source input/output arguments')
        mock(args.mock)
    else:
        if args.entry_lock is None or args.out is None:ap.error('--entry-lock and --out required for source extraction')
        extract(args.entry_lock,args.out)


if __name__=='__main__':main()
