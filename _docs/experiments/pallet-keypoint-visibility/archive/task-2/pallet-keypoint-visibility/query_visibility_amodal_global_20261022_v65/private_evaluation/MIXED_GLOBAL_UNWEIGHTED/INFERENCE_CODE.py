"""Gate-controlled v65 frozen inference from unchanged N0 query inputs.

No reference points, human states, performance table or pose optimizer is read.
New SAM replay is generic/frozen/box-only. Geometry is a separate saved-pose
proxy and cannot change the learned visible/full/joint probabilities.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SELECTED='MIXED_GLOBAL_UNWEIGHTED'
SELECTION_SHA='6d54028ea2247571f4c2dbc4beb4380b741c06633400f8c74b11e8396eef6261'
JOINT_PAIRS=np.array(((0,0),(1,0),(1,1),(2,0),(2,1),(2,2)),np.int64)
NAMES=('VISIBLE_TARGET','EXTERNAL_SUPPORT_PROXY','OUTSIDE_FULL_SUPPORT')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def check_binding(entry,role):
    if not isinstance(entry,dict)or len(entry.get('sha256',''))!=64 or sha(entry['path'])!=entry['sha256']:
        raise PermissionError('Immutable '+role+' binding mismatch')


def import_bound(entry,role):
    check_binding(entry,role)
    name='_v65_frozen_'+role+'_'+entry['sha256'][:12]
    spec=importlib.util.spec_from_file_location(name,entry['path']);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def validate_gate(path):
    """All source/code/runtime checks precede any private cache/RGB open."""
    gate=json.loads(Path(path).read_text())
    for key in ('approved','source_selection_complete','allow_current_evaluation_inference'):
        if gate.get(key)is not True:raise PermissionError('Gate requires '+key+'=true')
    if gate.get('current_eval_used_for_selection')is not False:raise PermissionError('Current evaluation cannot select the model')
    sealbinding=gate['source_selection_seal'];check_binding(sealbinding,'source selection seal')
    if sealbinding['sha256']!=SELECTION_SHA:raise PermissionError('Only the pre-private v65 source seal is accepted')
    seal=json.loads(Path(sealbinding['path']).read_text())
    if seal['primary']!=SELECTED or seal['current_eval_used_for_selection']is not False:
        raise PermissionError('Frozen source primary mismatch')
    if seal['thresholds']!={'binary_visible':.5,'logical_full':.5,'joint_confidence':.5}:
        raise PermissionError('Only source-frozen .5 decoders are accepted')
    selected=[c for c in seal['candidates']if c['candidate']==SELECTED]
    if len(selected)!=1 or selected[0]['release']['RELEASE']is not True or selected[0]['release']['HOLD']is not False:
        raise PermissionError('Selected source candidate has not qualified')
    selected=selected[0]
    check_binding(seal['source_input_lock'],'source input lock')
    lock=json.loads(Path(seal['source_input_lock']['path']).read_text())
    for name,digest in lock['bindings'].items():
        check_binding({'path':str(ROOT/name),'sha256':digest},'source lock '+name)
    for name,binding in seal['reports'].items():check_binding(binding,'source report '+name)
    candidates=gate['approved_candidates']
    if len(candidates)!=1 or candidates[0]['name']!=SELECTED:raise PermissionError('Exactly the selected MIXED candidate is permitted')
    candidate=candidates[0]
    if candidate['inference_code_sha256']!=sha(__file__):raise PermissionError('Inference code is not approved')
    for field in ('checkpoint','freeze','model_code','geometry_helper','global_recipe'):
        check_binding(candidate[field],field)
    if candidate['checkpoint']!=selected['checkpoint']or candidate['freeze']!=selected['freeze']:
        raise PermissionError('Gate checkpoint/freeze differs from source selection seal')
    freeze=json.loads(Path(candidate['freeze']['path']).read_text())
    if freeze.get('completed')is not True or freeze.get('eval_data_used')is not False or freeze.get('PnP_used')is not False:
        raise PermissionError('Source-only complete freeze required')
    if freeze['checkpoint_sha256']!=candidate['checkpoint']['sha256']or freeze['threshold']!=.5:
        raise PermissionError('Frozen checkpoint/threshold mismatch')
    if freeze['code_sha256']['query_global_amodal_head.py']!=candidate['model_code']['sha256']:
        raise PermissionError('Frozen model code mismatch')
    check_binding(gate['v63_input_gate'],'prior input gate')
    prior=json.loads(Path(gate['v63_input_gate']['path']).read_text())
    if set(gate['inputs'])!={'roi_cache','prepared_inputs','feature_metadata'}or gate['inputs']!=prior['inputs']:
        raise PermissionError('Only unchanged v63 N0 input artifact bindings are permitted')
    # Actual private files are not hashed/opened until run() after this returns.
    for entry in gate['inputs'].values():
        if not entry.get('path')or len(entry.get('sha256',''))!=64:raise PermissionError('Unbound private input')
    if candidate.get('sam_mode')!='frozen_replay':raise PermissionError('Registered mode is frozen SAM replay')
    assets=candidate['sam_assets']
    for key in ('loader','checkpoint','runtime_audit','source_recipe'):check_binding(assets[key],'SAM '+key)
    runtime=json.loads(Path(assets['runtime_audit']['path']).read_text())
    for binding in runtime['runtime_files_verified']:check_binding(binding,'SAM runtime file')
    check_binding(candidate['sam_native9_reference_manifest'],'old native9 manifest')
    ref=json.loads(Path(candidate['sam_native9_reference_manifest']['path']).read_text())
    native=candidate['sam_native9_reference']
    # Bind the sparse cached input to the previously completed v63 inference,
    # rather than accepting an arbitrary new map under the gate.
    listed=ref.get('outputs',[])
    if isinstance(listed,dict):listed=list(listed.values())
    matching=[e for e in listed if isinstance(e,dict)and e.get('path')==native['path']and e.get('sha256')==native['sha256']]
    if not matching:raise PermissionError('Old native9 cache is not bound by its prior inference seal')
    if not native.get('path')or len(native.get('sha256',''))!=64:raise PermissionError('Old native9 cache is unbound')
    if gate.get('thresholds')!=seal['thresholds']:raise PermissionError('Gate changed a source-frozen threshold')
    runtime=gate['runtime']
    if runtime['stages']!='SAM_prepare_2.1_then_head_CPU_2.5'or runtime['head_device']!='cpu':raise PermissionError('Registered two-stage CPU head recipe required')
    for role,binding in gate.get('preflight_audits',{}).items():check_binding(binding,'preflight '+role)
    return gate,candidate,freeze


def context64(rois):
    x=np.asarray(rois)
    if x.ndim!=5 or x.shape[1:]!=(8,65,9,9):raise ValueError('Expected F,8,65,9,9 ROIs')
    v=x[:,:,64].astype(np.float32)
    return (x[:,:,:64].astype(np.float32)*v[:,:,None]).sum((1,3,4))/np.maximum(v.sum((1,2,3))[:,None],1)


def initial_query_box(points,hw):
    q=np.asarray(points,float)[:8];h,w=hw;good=np.isfinite(q).all(1)&~(q==-1).all(1)
    if good.sum()<2:return None
    lo=np.maximum(q[good].min(0),[0.,0.]);hi=np.minimum(q[good].max(0),[w-1.,h-1.])
    return np.r_[lo,hi]if(hi-lo>=2).all()else None


def query_validity(points,hw,pixel_valid):
    q=np.asarray(points,float)[:8];h,w=hw
    finite=np.isfinite(q).all(1)&~(q==-1).all(1)
    p=np.floor(np.where(np.isfinite(q),q,-1e8)+.5).astype(np.int64)
    valid=finite&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    valid&=(p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)&np.asarray(pixel_valid,bool)
    return valid,p


def input_features(rgb,points,side,box,logit,recipe):
    """Pure input-only features shared by source fixture audit and private run."""
    import cv2
    rgb=np.asarray(rgb);q=np.asarray(points,np.float64)[:8];h,w=rgb.shape[:2]
    if rgb.shape!=(h,w,3)or rgb.dtype!=np.uint8 or q.shape!=(8,2):raise ValueError('uint8 RGB and query8 required')
    if side<=0 or not np.isfinite(side)or np.asarray(logit).shape!=(h,w)or not np.isfinite(logit).all():raise ValueError('Finite full SAM logits and positive side required')
    x0,y0,x1,y1=np.asarray(box,np.float64);bw,bh=x1-x0,y1-y0
    if min(bw,bh)<2:raise ValueError('Input prompt box must be nondegenerate')
    a=(np.arange(9)-4)*side/9;xx,yy=np.meshgrid(a,a)
    xy=q[:,None,None]+np.stack((xx,yy),-1)[None]
    p=np.floor(np.where(np.isfinite(xy),xy,-1e8)+.5).astype(np.int64)
    valid=recipe.native_valid(xy,p,[h,w])
    patches=[]
    for dy in range(-2,3):
        for dx in range(-2,3):
            px,py=p[...,0]+dx,p[...,1]+dy;inside=(px>=0)&(px<w)&(py>=0)&(py<h)
            values=rgb[np.clip(py,0,h-1),np.clip(px,0,w-1)].copy();values[~inside]=0
            patches.append(values.transpose(0,3,1,2))
    native=np.stack(patches,axis=2).reshape(8,75,9,9)
    bg=np.stack((2*(xy[...,0]-x0)/bw-1,2*(xy[...,1]-y0)/bh-1,
        np.full(xy.shape[:-1],np.log2(side/bw)),np.full(xy.shape[:-1],np.log2(side/bh))),1)
    bg=np.nan_to_num(np.clip(bg,-4,4)).astype(np.float32)
    grids=[]
    for full in (np.asarray(logit,np.float32),cv2.dilate(np.asarray(logit,np.float32),np.ones((5,5),np.uint8))):
        sampled=full[np.clip(p[...,1],0,h-1),np.clip(p[...,0],0,w-1)]
        grids.append(np.where(valid,np.clip(sampled/8,-4,4),0).astype(np.float16))
    globalxy,gp,_,_=recipe.grid(box);gv=recipe.native_valid(globalxy,gp,[h,w])
    globalrgb=cv2.remap(rgb,globalxy[...,0].astype(np.float32),globalxy[...,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    globalsam=np.where(gv,np.clip(logit[np.clip(gp[...,1],0,h-1),np.clip(gp[...,0],0,w-1)]/8,-4,4),0).astype(np.float16)
    prompt=(globalxy[...,0]>=x0)&(globalxy[...,0]<=x1)&(globalxy[...,1]>=y0)&(globalxy[...,1]<=y1)&gv
    safeq=np.where(np.isfinite(q),q,-1e8)
    i64,i8,qv=recipe.local_mapping(safeq,side,box,[h,w])
    return dict(rgb_native=native,box_grid=bg,sam=np.stack(grids,1),
        global_rgb=globalrgb.transpose(2,0,1)[None],global_sam=globalsam[None,None],global_prompt=prompt[None,None],
        global_index64=i64,global_index8=i8,global_query_valid=qv,native_valid=valid,global_valid=gv[None])


def self_flags(frame,geometry):
    pose=frame.get('initial_pose',{})
    if not pose.get('available',False):return [None]*8,'UNAVAILABLE_SAVED_INITIAL_POSE'
    try:return geometry.conditional_self_hidden(pose['cf_extents'],pose['R_cf'],pose['centroid']).astype(bool).tolist(),'SAVED_INITIAL_REGISTERED_CUBOID_PROXY'
    except(KeyError,ValueError,TypeError)as exc:return [None]*8,'INVALID_SAVED_INITIAL_GEOMETRY:'+str(exc)


def forward_frame(model,x,context,features,device):
    import torch
    tensor=lambda a,dtype=torch.float32:torch.as_tensor(np.ascontiguousarray(a),device=device,dtype=dtype)
    inputs=dict(x=tensor(x),rgb_native=tensor(features['rgb_native'])/255,sam=tensor(features['sam']),
        context=tensor(np.repeat(np.asarray(context)[None],8,0)),box_grid=tensor(features['box_grid']),
        global_input=torch.cat((tensor(features['global_rgb'])/255,tensor(features['global_sam']),tensor(features['global_prompt'])),1),
        target_inverse=torch.zeros(8,device=device,dtype=torch.long),global_index64=tensor(features['global_index64'],torch.long),
        global_index8=tensor(features['global_index8'],torch.long),global_query_valid=tensor(features['global_query_valid'],torch.bool))
    with torch.inference_mode():return model(**inputs).cpu().numpy().astype(np.float64)


def probabilities(logits):
    p=np.exp(logits-np.logaddexp.reduce(logits,axis=1)[:,None])
    visible=np.stack((p[:,0],p[:,0]+p[:,1]+p[:,3]),1)
    full=np.stack((p[:,:3].sum(1),p[:,:5].sum(1)),1)
    external=np.stack((p[:,1]+p[:,2],p[:,2]+p[:,4]),1)
    outside=np.stack((p[:,3]+p[:,4]+p[:,5],p[:,5]),1)
    logical=np.where(visible>=.5,0,np.where(full>=.5,1,2))
    confident=np.where((p.max(1)>=.5)[:,None],JOINT_PAIRS[p.argmax(1)],-1)
    return dict(joint=p,visible=visible,external=external,full=full,outside=outside,logical=logical,confident=confident)


def output_rows(frame,meta,pixel_valid,x,geometry):
    q=np.asarray(frame['initial_points'],float)[:8]
    valid,pixel=query_validity(q,frame['raw_hw'],pixel_valid)
    valid&=x[:,64].astype(bool).any((1,2))
    flags,reason=self_flags(frame,geometry);rows=[]
    for k in range(8):
        row=dict(id=frame['id'],native_corner=k,query_x=float(q[k,0])if np.isfinite(q[k,0])else None,
            query_y=float(q[k,1])if np.isfinite(q[k,1])else None,pixel_x=int(pixel[k,0]),pixel_y=int(pixel[k,1]),
            image_sha256=frame['image_sha256'],valid=bool(valid[k]),saved_roi_pixel_valid=bool(pixel_valid[k]),roi_side=float(meta['side']),
            geometry_self_hidden=flags[k],geometry_self_source=reason,geometry_overrides_learned_probability=False,
            status='OK'if valid[k]else'UNKNOWN',reason=''if valid[k]else'INVALID_QUERY_OR_EMPTY_ROI',
            logit=None,p_visible=None,present_at_frozen_threshold=None,p_visible_exact=None,p_external_support_exact=None,
            p_outside_full_exact=None,p_full_exact=None,p_visible_radius2=None,p_external_support_radius2=None,
            p_outside_full_radius2=None,p_full_radius2=None,max_joint_probability=None,confidence_U=True,
            logical_exact='UNKNOWN',logical_radius2='UNKNOWN',confident_exact='UNKNOWN',confident_radius2='UNKNOWN',
            raw_network_p_visible_exact=None,raw_network_p_visible_radius2=None,raw_SAM_exact_logit=None,raw_SAM_radius2_logit=None)
        row.update({f'joint_p{j}':None for j in range(6)});rows.append(row)
    return rows,pixel


def prepare(args):
    gate,candidate,freeze=validate_gate(args.gate)
    output=Path(candidate['output_directory']).resolve()
    if output.exists():raise FileExistsError('Preserve prior complete or partial private preparation')
    for key,entry in gate['inputs'].items():check_binding(entry,'private input '+key)
    check_binding(candidate['sam_native9_reference'],'old native9 cache')
    import torch
    if str(torch.__version__)!=gate['runtime']['SAM_torch']or Path(sys.executable).resolve()!=Path(gate['runtime']['SAM_executable']).resolve():raise PermissionError('SAM runtime differs from gate')
    if args.sam_device!=gate['runtime']['SAM_device']:raise PermissionError('SAM device differs from gate')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    recipe=import_bound(candidate['global_recipe'],'global_recipe');geometry=import_bound(candidate['geometry_helper'],'geometry')
    loader=import_bound(candidate['sam_assets']['loader'],'sam_loader')
    if Path(loader.CHECKPOINT).resolve()!=Path(candidate['sam_assets']['checkpoint']['path']).resolve()or loader.CHECKPOINT_SHA!=candidate['sam_assets']['checkpoint']['sha256']:
        raise PermissionError('SAM loader differs from approved checkpoint')
    sam_model,predictor=loader.load_model(args.sam_device)
    sam_before={k:v.detach().cpu().clone()for k,v in sam_model.state_dict().items()}
    inputs=gate['inputs']
    with np.load(inputs['roi_cache']['path'],allow_pickle=False)as z:x=z['x'];pixel_valid=z['pixel_valid']
    prep=Path(inputs['prepared_inputs']['path'])
    with(gzip.open if prep.suffix=='.gz'else open)(prep,'rt')as f:frames=json.load(f)
    metadata=json.loads(Path(inputs['feature_metadata']['path']).read_text())
    with np.load(candidate['sam_native9_reference']['path'],allow_pickle=False)as z:old_sam=z['sam']
    n=len(frames)
    if x.shape!=(n,8,65,9,9)or pixel_valid.shape!=(n,8)or old_sam.shape!=(n,8,1,9,9)or len(metadata)!=n:raise ValueError('Registered private dimensions disagree')
    if len({r['id']for r in frames})!=n or not np.isfinite(x).all():raise ValueError('Invalid saved ROI/identity')
    context=context64(x);rows=[];records=[];box_available=np.zeros(n,bool)
    fields={k:[]for k in ('rgb_native','box_grid','sam','global_rgb','global_sam','global_prompt','global_index64','global_index8','global_query_valid')}
    output.mkdir(parents=True);(output/'SAM_MASKS').mkdir();shutil.copy2(__file__,output/'INFERENCE_CODE.py')
    write(output/'PREPARATION_STARTED.json',dict(candidate=SELECTED,gate_sha256=sha(args.gate),source_selection_seal=gate['source_selection_seal'],private_ref_or_human_GT_read=False,new_PnP_calls=0))
    start=time.monotonic();parity_cells=0;sam_forwards=0;rgb_reads=0
    for i,(frame,meta)in enumerate(zip(frames,metadata)):
        if meta['id']!=frame['id']:raise ValueError('Metadata order differs')
        q=np.asarray(frame['initial_points'],float)[:8];hw=frame['raw_hw']
        frame_rows,pixel=output_rows(frame,meta,pixel_valid[i],x[i],geometry)
        box=initial_query_box(q,hw);features=None
        if box is not None:
            check_binding({'path':frame['image'],'sha256':frame['image_sha256']},'private RGB')
            from PIL import Image
            rgb=np.array(Image.open(frame['image']).convert('RGB'));rgb_reads+=1
            if list(rgb.shape[:2])!=hw:raise ValueError('Private RGB canvas mismatch')
            with torch.inference_mode():
                predictor.set_image(rgb)
                alternatives,quality,_=predictor.predict(box=box.astype(np.float32),multimask_output=True,return_logits=True)
            if not np.isfinite(quality).all():raise ValueError('Nonfinite SAM quality')
            selected=int(np.argmax(quality));logit=np.asarray(alternatives[selected],np.float32)
            if logit.shape!=tuple(hw)or not np.isfinite(logit).all():raise ValueError('Invalid SAM full logits')
            features=input_features(rgb,q,float(meta['side']),box,logit,recipe)
            if not np.array_equal(features['sam'][:,0:1],old_sam[i]):raise AssertionError('Frozen SAM exact native9 replay differs from v63 input')
            parity_cells+=old_sam[i].size;sam_forwards+=1;box_available[i]=True
            maskpath=output/'SAM_MASKS'/f'{i:04d}.npz'
            np.savez_compressed(maskpath,mask=(logit>0).astype(np.uint8),image_sha256=frame['image_sha256'])
            records.append(dict(id=frame['id'],available=True,box=box.tolist(),selected_index=selected,uncalibrated_mask_quality=np.asarray(quality).tolist(),mask_path=str(maskpath),mask_sha256=sha(maskpath)))
            for k,row in enumerate(frame_rows):
                if row['valid']:
                    px,py=pixel[k];h,w=hw
                    row['raw_SAM_exact_logit']=float(logit[py,px]);row['raw_SAM_radius2_logit']=float(logit[max(0,py-2):min(h,py+3),max(0,px-2):min(w,px+3)].max())
        else:
            for row in frame_rows:row.update(valid=False,status='UNKNOWN',reason='MISSING_OR_DEGENERATE_QUERY_BOX')
            records.append(dict(id=frame['id'],available=False,reason='MISSING_OR_DEGENERATE_UNCHANGED_INITIAL_QUERY_BOX'))
        for key in fields:
            if features is not None:fields[key].append(features[key])
            else:
                shapes={'rgb_native':(8,75,9,9),'box_grid':(8,4,9,9),'sam':(8,2,9,9),
                    'global_rgb':(1,3,64,64),'global_sam':(1,1,64,64),'global_prompt':(1,1,64,64),
                    'global_index64':(8,9,9),'global_index8':(8,9,9),'global_query_valid':(8,9,9)}
                dtype=np.uint8 if key in ('rgb_native','global_rgb')else np.int64 if key.startswith('global_index')else bool if key in ('global_prompt','global_query_valid')else np.float16 if key in ('sam','global_sam')else np.float32
                fields[key].append(np.zeros(shapes[key],dtype))
        rows.extend(frame_rows)
        if(i+1)%20==0:print(json.dumps({'prepared_frames':i+1,'total':n,'seconds':time.monotonic()-start}),flush=True)
    if not all(torch.equal(p.detach().cpu(),sam_before[k])for k,p in sam_model.state_dict().items()):raise AssertionError('SAM state changed')
    np.savez_compressed(output/'REAL_V65_INPUTS.npz',**{key:np.concatenate(value,0)for key,value in fields.items()},context=np.repeat(context,8,0),
        target_index=np.repeat(np.arange(n,dtype=np.int64),8),id=np.repeat([r['id']for r in frames],8),image_sha256=np.repeat([r['image_sha256']for r in frames],8),box_available=box_available)
    write(output/'REAL_QUERY_METADATA.json',rows);write(output/'SAM_TARGET_RECORDS.json',records)
    for name,entry in inputs.items():check_binding(entry,'unchanged private input '+name)
    files=[output/name for name in ('REAL_V65_INPUTS.npz','REAL_QUERY_METADATA.json','SAM_TARGET_RECORDS.json','INFERENCE_CODE.py')]
    seal=dict(schema='V65_GATED_PRIVATE_INPUT_PREPARATION_SEAL',candidate=SELECTED,completed=True,frames=n,slots=len(rows),
        source_selection_seal=gate['source_selection_seal'],gate={'path':str(Path(args.gate).resolve()),'sha256':sha(args.gate)},input_artifacts=inputs,
        inference_code_sha256=sha(__file__),SAM_exact_native9_replay_cells=parity_cells,SAM_exact_native9_replay_mismatches=0,
        SAM_forwards=sam_forwards,new_backbone_forward=0,real_RGB_images_opened=rgb_reads,torch=str(torch.__version__),sam_device=args.sam_device,
        private_ref_or_human_GT_read=False,performance_tables_read=False,SAM_state_unchanged=True,original_query_coordinates_preserved=True,
        geometry_self_overrides_learned_probability=False,optimizer_updates=0,threshold_or_model_selection=False,PnP_calls=0,seconds=time.monotonic()-start,
        outputs=[{'path':str(p.resolve()),'sha256':sha(p)}for p in files])
    write(output/'PREPARATION_SEAL.json',seal);print(json.dumps(seal),flush=True)


def infer(args):
    gate,candidate,freeze=validate_gate(args.gate);output=Path(candidate['output_directory']).resolve()
    if (output/'INFERENCE_SEAL.json').exists()or(output/'REAL_QUERY_SCORES.csv').exists():raise FileExistsError('Preserve prior predictions')
    prep_path=output/'PREPARATION_SEAL.json';prep=json.loads(prep_path.read_text())
    if not prep['completed']or prep['candidate']!=SELECTED or prep['gate']['sha256']!=sha(args.gate)or prep['inference_code_sha256']!=sha(__file__):raise PermissionError('Preparation seal/gate mismatch')
    if prep['private_ref_or_human_GT_read']or prep['performance_tables_read']or prep['PnP_calls']!=0 or prep['optimizer_updates']!=0:raise PermissionError('Input preparation scope mismatch')
    for entry in prep['outputs']:check_binding(entry,'prepared model input')
    for name,entry in gate['inputs'].items():check_binding(entry,'private input '+name)
    import torch
    if str(torch.__version__)!=gate['runtime']['head_torch']or Path(sys.executable).resolve()!=Path(gate['runtime']['head_executable']).resolve():raise PermissionError('Head runtime differs from source frozen CPU inference')
    if args.device!=gate['runtime']['head_device']:raise PermissionError('Head device differs from source CPU gate')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    head=import_bound(candidate['model_code'],'head');model,payload=head.load_model(candidate['checkpoint']['path'],args.device)
    model.eval().requires_grad_(False)
    if payload['head_config']!=freeze['head_config']:raise ValueError('Checkpoint/freeze head config mismatch')
    before={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    with np.load(gate['inputs']['roi_cache']['path'],allow_pickle=False)as z:x=z['x']
    with np.load(output/'REAL_V65_INPUTS.npz',allow_pickle=False)as z:data={k:z[k]for k in z.files}
    rows=json.loads((output/'REAL_QUERY_METADATA.json').read_text());n=len(x)
    if len(rows)!=8*n or not np.array_equal(data['target_index'],np.repeat(np.arange(n),8)):raise ValueError('Prepared query ownership mismatch')
    logits=np.full((n,8,6),np.nan,np.float64);repeat_max=0.;start=time.monotonic()
    local=('rgb_native','box_grid','sam','global_index64','global_index8','global_query_valid')
    for i in range(n):
        if not data['box_available'][i]:continue
        ix=slice(8*i,8*i+8);features={key:data[key][ix]for key in local}
        features.update({key:data[key][i:i+1]for key in ('global_rgb','global_sam','global_prompt')})
        logits[i]=forward_frame(model,x[i],data['context'][8*i],features,args.device)
        if not np.isfinite(logits[i]).all():raise FloatingPointError('Nonfinite frozen learned logits')
        if i==0:
            repeat=forward_frame(model,x[i],data['context'][8*i],features,args.device);repeat_max=float(np.max(np.abs(repeat-logits[i])))
            if repeat_max!=0:raise AssertionError('Repeated frozen forward differs')
        scores=probabilities(logits[i])
        for k,row in enumerate(rows[ix]):
            row['raw_network_p_visible_exact']=float(scores['visible'][k,0]);row['raw_network_p_visible_radius2']=float(scores['visible'][k,1])
            if not row['valid']:continue
            pv=float(scores['visible'][k,0]);row.update(p_visible=pv,logit=float(np.log(np.clip(pv,1e-15,1-1e-15)/np.clip(1-pv,1e-15,1))),
                present_at_frozen_threshold=bool(pv>=.5),max_joint_probability=float(scores['joint'][k].max()),confidence_U=bool(scores['joint'][k].max()<.5))
            for scale,name in enumerate(('exact','radius2')):
                row.update({f'p_visible_{name}':float(scores['visible'][k,scale]),f'p_external_support_{name}':float(scores['external'][k,scale]),
                    f'p_outside_full_{name}':float(scores['outside'][k,scale]),f'p_full_{name}':float(scores['full'][k,scale]),
                    f'logical_{name}':NAMES[int(scores['logical'][k,scale])],f'confident_{name}':NAMES[int(scores['confident'][k,scale])]if scores['confident'][k,scale]>=0 else'UNKNOWN'})
            row.update({f'joint_p{j}':float(scores['joint'][k,j])for j in range(6)})
    if not all(torch.equal(p.detach().cpu(),before[k])for k,p in model.state_dict().items()):raise AssertionError('Head state changed')
    scorespath=output/'REAL_QUERY_SCORES.csv'
    with scorespath.open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(output/'REAL_QUERY_OUTPUTS.npz',joint_logits=logits,valid=np.array([r['valid']for r in rows]).reshape(n,8),
        geometry_self_hidden=np.array([-1 if r['geometry_self_hidden']is None else int(r['geometry_self_hidden'])for r in rows]).reshape(n,8))
    check_binding(candidate['checkpoint'],'unchanged checkpoint')
    for entry in prep['outputs']:check_binding(entry,'unchanged prepared input')
    seal=dict(schema='V65_GATED_SELECTED_FROZEN_PRIVATE_INFERENCE_SEAL',candidate=SELECTED,completed=True,frames=n,slots=len(rows),valid_queries=sum(r['valid']for r in rows),
        source_selection_seal=gate['source_selection_seal'],gate={'path':str(Path(args.gate).resolve()),'sha256':sha(args.gate)},checkpoint=candidate['checkpoint'],model_code=candidate['model_code'],
        inference_code_sha256=sha(__file__),preparation_seal={'path':str(prep_path.resolve()),'sha256':sha(prep_path)},input_artifacts=gate['inputs'],thresholds=gate['thresholds'],device=args.device,
        head_torch=str(torch.__version__),SAM_torch=prep['torch'],SAM_exact_native9_replay_cells=prep['SAM_exact_native9_replay_cells'],SAM_exact_native9_replay_mismatches=0,
        new_backbone_forward=0,SAM_forwards=prep['SAM_forwards'],real_RGB_images_opened=prep['real_RGB_images_opened'],
        private_ref_or_human_GT_read=False,semantic_annotation_reads=0,real_CAL_model_reads=0,performance_tables_read=False,model_state_unchanged=True,SAM_state_unchanged=True,
        repeat_max_logit_difference=repeat_max,original_query_coordinates_preserved=True,geometry_self_is_conditional_cuboid_proxy=True,geometry_self_overrides_learned_probability=False,
        optimizer_updates=0,new_optimizer_updates=0,threshold_or_model_selection=False,PnP_calls=0,new_PnP_calls=0,seconds=time.monotonic()-start,
        outputs=[{'path':str(p.resolve()),'sha256':sha(p)}for p in (scorespath,output/'REAL_QUERY_OUTPUTS.npz')])
    write(output/'INFERENCE_SEAL.json',seal);print(json.dumps(seal),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','infer'))
    p.add_argument('--gate',required=True);p.add_argument('--device',default='cpu');p.add_argument('--sam-device',default='cuda:0')
    a=p.parse_args();prepare(a)if a.command=='prepare'else infer(a)

if __name__=='__main__':main()
