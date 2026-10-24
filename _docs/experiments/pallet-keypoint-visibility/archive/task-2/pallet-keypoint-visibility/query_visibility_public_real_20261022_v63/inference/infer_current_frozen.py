"""Gate-controlled frozen query-head inference. No semantic labels or pose solve.

Real input files are opened only after an approved source-selection gate and all
candidate/source-code hashes pass. The only geometry output is an independently
saved conditional cuboid self flag; it never changes the learned probability.
"""
import argparse,csv,gzip,hashlib,importlib.util,inspect,json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def import_bound(path,expected_sha,role):
    path=Path(path).resolve()
    if sha(path)!=expected_sha:raise ValueError('Bound '+role+' code changed')
    name='_v63_'+role+'_'+expected_sha[:12]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def validate_gate(path,name):
    """This function deliberately does not read any private data artifact."""
    gate=json.loads(Path(path).read_text())
    for key in ['approved','source_selection_complete','allow_current_evaluation_inference']:
        if gate.get(key)is not True:raise PermissionError('Current-eval gate requires '+key+'=true')
    if gate.get('current_eval_used_for_selection')is not False:raise PermissionError('Gate must exclude current evaluation from source selection')
    seal=gate['source_selection_seal']
    if sha(seal['path'])!=seal['sha256']:raise PermissionError('Source-selection seal changed')
    candidates=[r for r in gate['approved_candidates']if r['name']==name]
    if len(candidates)!=1:raise PermissionError('Exactly one immutable approved candidate required')
    candidate=candidates[0]
    if candidate['inference_code_sha256']!=sha(__file__):raise PermissionError('Inference implementation is not approved')
    for field in ['checkpoint','freeze','model_code','geometry_helper']:
        entry=candidate[field]
        if sha(entry['path'])!=entry['sha256']:raise PermissionError('Approved '+field+' changed')
    if candidate.get('rgb_recipe'):
        e=candidate['rgb_recipe']
        if sha(e['path'])!=e['sha256']:raise PermissionError('Approved RGB recipe changed')
    freeze=json.loads(Path(candidate['freeze']['path']).read_text())
    needs_sam=bool(candidate.get('compute_sam_baseline',False)or freeze.get('head_config',{}).get('sam_branch',False))
    if needs_sam:
        if not candidate.get('sam_assets'):raise PermissionError('Approved SAM assets required')
        for field in ['loader','checkpoint','runtime_audit','source_recipe']:
            e=candidate['sam_assets'][field]
            if sha(e['path'])!=e['sha256']:raise PermissionError('Approved SAM '+field+' changed')
        runtime=json.loads(Path(candidate['sam_assets']['runtime_audit']['path']).read_text())
        for binding in runtime['runtime_files_verified']:
            if sha(binding['path'])!=binding['sha256']:raise PermissionError('Approved SAM vendor runtime changed')
    if freeze.get('checkpoint_sha256')!=candidate['checkpoint']['sha256']:raise PermissionError('Freeze does not bind approved checkpoint')
    if freeze.get('eval_data_used')is not False or freeze.get('PnP_used')is not False:
        raise PermissionError('Candidate freeze must declare eval_data_used=false and PnP_used=false')
    if 'threshold'not in freeze or not 0<=float(freeze['threshold'])<=1:raise ValueError('Frozen source threshold missing')
    if set(gate['inputs'])!={'roi_cache','prepared_inputs','feature_metadata'}:raise PermissionError('Only the three registered prediction-input artifacts are permitted')
    for key in ['roi_cache','prepared_inputs','feature_metadata']:
        entry=gate['inputs'][key]
        if not entry.get('path')or len(entry.get('sha256',''))!=64:raise PermissionError('Missing registered input '+key)
    return gate,candidate,freeze


def context64(rois):
    x=np.asarray(rois)
    if x.ndim!=5 or x.shape[1:]!=(8,65,9,9):raise ValueError('Expected frames,8,65,9,9 saved ROI')
    v=x[:,:,64].astype(np.float32)
    return(x[:,:,:64].astype(np.float32)*v[:,:,None]).sum(axis=(1,3,4))/np.maximum(v.sum(axis=(1,2,3))[:,None],1)


def query_validity(points,hw,pixel_valid):
    q=np.asarray(points,float)[:8];h,w=hw
    finite=np.isfinite(q).all(1)&~(q==-1).all(1)
    safe=np.where(np.isfinite(q),q,-1e8);pxy=np.floor(safe+.5).astype(np.int64)
    valid=finite&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)&(pxy[:,0]>=0)&(pxy[:,0]<w)&(pxy[:,1]>=0)&(pxy[:,1]<h)&np.asarray(pixel_valid,bool)
    return valid,pxy


def self_flags(frame,geometry):
    pose=frame.get('initial_pose',{})
    if not pose.get('available',False):return[None]*8,'UNAVAILABLE_SAVED_INITIAL_POSE'
    try:
        flags=geometry.conditional_self_hidden(pose['cf_extents'],pose['R_cf'],pose['centroid'])
        return flags.astype(bool).tolist(),'SAVED_INITIAL_REGISTERED_CUBOID_PROXY'
    except(KeyError,ValueError,TypeError)as e:
        return[None]*8,'INVALID_SAVED_INITIAL_GEOMETRY:'+str(e)


def initial_query_box(points,hw):
    q=np.asarray(points,float)[:8];h,w=hw
    finite=np.isfinite(q).all(1)&~(q==-1).all(1)
    if finite.sum()<2:return None
    lo=np.maximum(q[finite].min(0),[0.,0.]);hi=np.minimum(q[finite].max(0),[w-1.,h-1.])
    return np.r_[lo,hi]if(hi-lo>=2).all()else None


def sam_grid_from_logits(logit,points,side):
    q=np.asarray(points,float)[:8];h,w=logit.shape
    off=(np.arange(9)-4)*side/9;dx,dy=np.meshgrid(off,off)
    xy=q[:,None,None,:]+np.stack([dx,dy],-1)[None]
    finite=np.isfinite(xy).all(-1);safe=np.where(np.isfinite(xy),xy,-1e8)
    pixel=np.floor(safe+.5).astype(np.int64)
    valid=finite&(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)&(pixel[...,0]>=0)&(pixel[...,0]<w)&(pixel[...,1]>=0)&(pixel[...,1]<h)
    sampled=logit[np.clip(pixel[...,1],0,h-1),np.clip(pixel[...,0],0,w-1)]
    grid=np.where(valid,np.clip(sampled/8,-4,4),0).astype(np.float16)[:,None]
    return grid,valid


def sam_neighborhood_logits(logit,px,py):
    """Exact pixel and predeclared clipped-square radius1/radius2 readouts."""
    h,w=logit.shape
    if not(0<=px<w and 0<=py<h):return[None,None,None]
    return[float(logit[max(0,py-r):min(h,py+r+1),max(0,px-r):min(w,px+r+1)].max())for r in[0,1,2]]


def predict(model,x,context,rgb,device,batch_size,sam=None):
    import torch
    before={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    def one(a,b,c,s):
        xx=torch.from_numpy(np.asarray(a)).float().to(device)
        rr=torch.from_numpy(np.asarray(b)).float().to(device)/255 if b is not None else None
        cc=torch.from_numpy(np.asarray(c)).float().to(device)if c is not None else None
        kwargs={'context':cc}if'context'in inspect.signature(model.forward).parameters else{}
        if'sam'in inspect.signature(model.forward).parameters:kwargs['sam']=torch.from_numpy(np.asarray(s)).float().to(device)if s is not None else None
        return model(xx,rr,**kwargs).detach().cpu().numpy()
    values=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            end=start+batch_size
            values.append(one(x[start:end],rgb[start:end]if rgb is not None else None,context[start:end]if context is not None else None,sam[start:end]if sam is not None else None))
        logits=np.concatenate(values)
        repeat=one(x[:min(batch_size,len(x))],rgb[:min(batch_size,len(x))]if rgb is not None else None,context[:min(batch_size,len(x))]if context is not None else None,sam[:min(batch_size,len(x))]if sam is not None else None)
    if not np.array_equal(repeat,logits[:len(repeat)]):raise AssertionError('Repeated frozen batch differs')
    if not all(torch.equal(v.detach().cpu(),before[k])for k,v in model.state_dict().items()):raise AssertionError('Frozen state changed')
    return logits,dict(frozen_state_unchanged=True,repeat_batch_size=len(repeat),repeat_max_logit_difference=0.,forward_calls=len(values)+1)


def run(args):
    import torch
    # No private input or RGB file access can precede this complete gate check.
    gate,candidate,freeze=validate_gate(args.gate,args.candidate)
    output=Path(candidate['output_directory']).resolve()
    if args.out and Path(args.out).resolve()!=output:raise PermissionError('Output directory differs from approved gate')
    if output.exists():raise FileExistsError('Preserve prior complete or partial evaluation output')
    output.mkdir(parents=True)
    write(output/'INFERENCE_STARTED.json',dict(candidate=args.candidate,gate_sha256=sha(args.gate),checkpoint=candidate['checkpoint'],source_selection_seal=gate['source_selection_seal'],private_input_reads_at_start=0,new_PnP_calls=0))
    inputs=gate['inputs']
    for name,entry in inputs.items():
        if sha(entry['path'])!=entry['sha256']:raise PermissionError('Frozen input changed: '+name)
    model_code=import_bound(candidate['model_code']['path'],candidate['model_code']['sha256'],'query_head')
    geometry=import_bound(candidate['geometry_helper']['path'],candidate['geometry_helper']['sha256'],'geometry_self')
    model,payload=model_code.load_model(candidate['checkpoint']['path'],args.device)
    model=model.eval().requires_grad_(False)
    if payload['head_config']!=freeze['head_config']:raise ValueError('Checkpoint config differs from source freeze')
    rgb_helper=None
    if model.config.rgb_branch:
        if not candidate.get('rgb_recipe'):raise PermissionError('RGB model needs approved source-shared RGB recipe')
        rgb_helper=import_bound(candidate['rgb_recipe']['path'],candidate['rgb_recipe']['sha256'],'rgb_recipe')
    need_sam=bool(candidate.get('compute_sam_baseline',False)or getattr(model.config,'sam_branch',False))
    sam_predictor=None;sam_asset=None;sam_before=None
    if need_sam:
        if not candidate.get('sam_assets'):raise PermissionError('SAM head/baseline requires approved generic SAM assets')
        sam_asset=candidate['sam_assets']
        sam_loader=import_bound(sam_asset['loader']['path'],sam_asset['loader']['sha256'],'sam_loader')
        if Path(sam_loader.CHECKPOINT).resolve()!=Path(sam_asset['checkpoint']['path']).resolve()or sam_loader.CHECKPOINT_SHA!=sam_asset['checkpoint']['sha256']:raise PermissionError('SAM loader checkpoint differs from approved generic model')
        sam_model,sam_predictor=sam_loader.load_model(args.sam_device or args.device)
        sam_before={k:v.detach().cpu().clone()for k,v in sam_model.state_dict().items()}
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    with np.load(inputs['roi_cache']['path'],allow_pickle=False)as z:x=z['x'];pixel_valid=z['pixel_valid']
    prepared=Path(inputs['prepared_inputs']['path'])
    with(gzip.open if prepared.suffix=='.gz'else open)(prepared,'rt')as f:frames=json.load(f)
    metadata=json.loads(Path(inputs['feature_metadata']['path']).read_text())
    if x.shape!=(len(frames),8,65,9,9)or pixel_valid.shape!=(len(frames),8)or len(metadata)!=len(frames):raise ValueError('Frozen input dimensions disagree')
    if len({r['id']for r in frames})!=len(frames):raise ValueError('Duplicate input frame ID')
    if not np.isfinite(x).all():raise ValueError('Nonfinite saved ROI')
    features_before=x.copy();ctx=context64(x)
    flat_context=np.repeat(ctx,8,axis=0)if getattr(model.config,'target_context',False) else None
    rgb=np.zeros((len(frames)*8,3,65,65),np.uint8)if model.config.rgb_branch else None
    rows=[];rgb_images=0;self_counts={'valid_frames':0,'unknown_frames':0};start=time.monotonic()
    sam_features=np.zeros((len(frames)*8,1,9,9),np.float16)if need_sam else None
    sam_rows=[];sam_records=[];sam_forwards=0
    if need_sam:(output/'SAM_MASKS').mkdir()
    for i,(frame,meta)in enumerate(zip(frames,metadata)):
        if meta['id']!=frame['id']:raise ValueError('Saved metadata frame order differs')
        points=np.array(frame['initial_points'],float)[:8]
        valid,pixels=query_validity(points,frame['raw_hw'],pixel_valid[i]);flags,reason=self_flags(frame,geometry)
        self_counts['valid_frames'if any(v is not None for v in flags)else'unknown_frames']+=1
        image=None;pil_rgb=None;sam_logit=None;sam_available=False
        if rgb is not None or need_sam:
            from PIL import Image
            if sha(frame['image'])!=frame['image_sha256']:raise ValueError('RGB image hash differs')
            pil_rgb=np.array(Image.open(frame['image']).convert('RGB'))
            if list(pil_rgb.shape[:2])!=frame['raw_hw']:raise ValueError('RGB canvas mismatch')
            rgb_images+=1
            if rgb is not None:
                import cv2
                decoded=cv2.imread(frame['image'])
                if decoded is None:raise ValueError('OpenCV could not decode RGB')
                image=cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
                if image.shape!=pil_rgb.shape:raise ValueError('RGB decoders disagree on canvas')
        if need_sam:
            box=initial_query_box(points,frame['raw_hw'])
            if box is not None:
                with torch.inference_mode():
                    sam_predictor.set_image(pil_rgb)
                    alternatives,quality,_=sam_predictor.predict(box=box.astype(np.float32),multimask_output=True,return_logits=True)
                quality=np.asarray(quality,float)
                if not np.isfinite(quality).all():raise ValueError('Nonfinite SAM predicted quality')
                selected=int(np.argmax(quality));sam_logit=alternatives[selected]
                if not np.isfinite(sam_logit).all():raise ValueError('Nonfinite SAM mask logits')
                if tuple(sam_logit.shape)!=tuple(frame['raw_hw']):raise ValueError('SAM must preserve original canvas')
                grid,_=sam_grid_from_logits(sam_logit,points,float(meta['side']))
                sam_features[i*8:(i+1)*8]=grid;sam_available=True;sam_forwards+=1
                maskpath=output/'SAM_MASKS'/f'{i:04d}.npz'
                np.savez_compressed(maskpath,mask=(sam_logit>0).astype(np.uint8),image_sha256=frame['image_sha256'])
                sam_records.append(dict(id=frame['id'],available=True,box=box.tolist(),selected_index=selected,uncalibrated_mask_quality=quality.tolist(),mask_path=str(maskpath),mask_sha256=sha(maskpath)))
            else:sam_records.append(dict(id=frame['id'],available=False,reason='NO_VALID_UNCHANGED_INITIAL_QUERY_BOX'))
        for k,q in enumerate(points):
            center_ok=np.isfinite(q).all()
            if rgb is not None and center_ok:
                crop=rgb_helper.query_rgb_crop(image,float(q[0]),float(q[1]),float(meta['side']))
                px,py=pixels[k];h,w=frame['raw_hw']
                expected=pil_rgb[py,px]if 0<=px<w and 0<=py<h else np.zeros(3,np.uint8)
                if not np.array_equal(crop[32,32],expected):raise AssertionError('Shared RGB recipe center differs from independent PIL')
                rgb[i*8+k]=crop.transpose(2,0,1)
            if need_sam:
                px,py=pixels[k]
                center_valid=bool(valid[k]and sam_available)
                radii=sam_neighborhood_logits(sam_logit,int(px),int(py))if center_valid else[None,None,None]
                raw=radii[0]
                sam_rows.append(dict(id=frame['id'],native_corner=k,query_x=float(q[0]),query_y=float(q[1]),image_sha256=frame['image_sha256'],valid=center_valid,raw_sam_logit=raw,p_visible=float(1/(1+np.exp(-np.clip(raw,-80,80))))if raw is not None else None,SAM_present_at_logit0=bool(raw>0)if raw is not None else None,geometry_self_hidden=flags[k],raw_sam_radius1_logit=radii[1],p_visible_radius1=float(1/(1+np.exp(-np.clip(radii[1],-80,80))))if radii[1]is not None else None,SAM_radius1_present_at_logit0=bool(radii[1]>0)if radii[1]is not None else None,raw_sam_radius2_logit=radii[2],p_visible_radius2=float(1/(1+np.exp(-np.clip(radii[2],-80,80))))if radii[2]is not None else None,SAM_radius2_present_at_logit0=bool(radii[2]>0)if radii[2]is not None else None))
            rows.append(dict(id=frame['id'],native_corner=k,query_x=float(q[0]),query_y=float(q[1]),pixel_x=int(pixels[k,0]),pixel_y=int(pixels[k,1]),image_sha256=frame['image_sha256'],valid=bool(valid[k]),saved_roi_pixel_valid=bool(pixel_valid[i,k]),roi_side=float(meta['side']),geometry_self_hidden=flags[k],geometry_self_source=reason,SAM_feature_available=sam_available if need_sam else None))
    logits,model_audit=predict(model,x.reshape(-1,65,9,9),flat_context,rgb,args.device,args.batch_size,sam=sam_features if getattr(model.config,'sam_branch',False)else None)
    if not np.isfinite(logits).all():raise FloatingPointError('Nonfinite frozen head output')
    if need_sam and not all(torch.equal(v.detach().cpu(),sam_before[k])for k,v in sam_model.state_dict().items()):raise AssertionError('Frozen SAM state changed')
    probabilities=1/(1+np.exp(-np.clip(logits.astype(np.float64),-80,80)))
    if not np.array_equal(x,features_before):raise AssertionError('Frozen ROI mutated')
    for row,z,p in zip(rows,logits,probabilities):
        row['logit']=float(z);row['p_visible']=float(p);row['present_at_frozen_threshold']=bool(p>=float(freeze['threshold']))
    csvpath=output/'REAL_QUERY_SCORES.csv'
    with csvpath.open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(output/'REAL_QUERY_OUTPUTS.npz',logits=logits.reshape(-1,8),probability=probabilities.reshape(-1,8),valid=np.array([r['valid']for r in rows]).reshape(-1,8),geometry_self_hidden=np.array([-1 if r['geometry_self_hidden']is None else int(r['geometry_self_hidden'])for r in rows]).reshape(-1,8))
    for name,entry in inputs.items():
        if sha(entry['path'])!=entry['sha256']:raise AssertionError('Input artifact mutated: '+name)
    if sha(candidate['checkpoint']['path'])!=candidate['checkpoint']['sha256']:raise AssertionError('Checkpoint mutated')
    extra_outputs=[]
    if need_sam:
        path=output/'RAW_SAM_QUERY_SCORES.csv'
        with path.open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(sam_rows[0]));w.writeheader();w.writerows(sam_rows)
        recordpath=output/'SAM_TARGET_RECORDS.json';write(recordpath,sam_records)
        gridpath=output/'SAM_NATIVE9_FEATURES.npz';np.savez_compressed(gridpath,sam=sam_features.reshape(-1,8,1,9,9))
        extra_outputs=[path,recordpath,gridpath]
    audit=dict(schema='GATED_FROZEN_CURRENT_QUERY_INFERENCE_V1',candidate=args.candidate,frames=len(frames),slots=len(rows),device=args.device,threshold=freeze['threshold'],source_selection_gate=dict(path=str(Path(args.gate).resolve()),sha256=sha(args.gate)),checkpoint=candidate['checkpoint'],model_code=candidate['model_code'],inference_code_sha256=sha(__file__),input_artifacts=inputs,input_only_context='all8unchangedROIs valid-weighted64featuremean',original_query_coordinates_preserved=True,context_no_GT=True,real_RGB_images_opened=rgb_images,RGB_crop_recipe=candidate.get('rgb_recipe'),geometry_self=self_counts,geometry_self_is_conditional_cuboid_proxy=True,geometry_self_overrides_learned_probability=False,head_state=model_audit,new_backbone_forward=0,SAM_used=need_sam,SAM_weights_state_unchanged=True if need_sam else None,SAM_image_encoder_forwards=sam_forwards,SAM_prompt_decoder_calls=sam_forwards,SAM_assets=sam_asset,SAM_feature_recipe='native9 nearest selectedlogits/8 clipped[-4,4], continuous+nearest invalid0',SAM_raw_output_targets={'p_visible':'primary exact nearest query pixel presence','p_visible_radius1':'separate diagnostic: any support in clipped3x3 square','p_visible_radius2':'separate diagnostic: any support in clipped5x5 square'},SAM_raw_sigmoid_is_calibrated_probability=False,SAM_prompt='unchanged initialq8bbox only; no positive query/GT mask prompts',new_optimizer_updates=0,new_PnP_calls=0,semantic_annotation_reads=0,real_CAL_model_reads=0,seconds=time.monotonic()-start,outputs=[dict(path=str(p),sha256=sha(p))for p in[csvpath,output/'REAL_QUERY_OUTPUTS.npz',*extra_outputs]])
    write(output/'INFERENCE_SEAL.json',audit);print(json.dumps(audit),flush=True)


def smoke():
    """Mock tensors/images/poses only; no current-evaluation artifact opens."""
    import tempfile
    import torch
    model_path=ROOT/'model/query_head.py';rgb_path=ROOT/'model/build_rgb_source_cache.py';geom_path=ROOT/'contract/geometry_self_rule.py'
    module=import_bound(model_path,sha(model_path),'mock_query_head');geometry=import_bound(geom_path,sha(geom_path),'mock_geometry');rgb_helper=import_bound(rgb_path,sha(rgb_path),'mock_rgb')
    torch.set_num_threads(1);torch.manual_seed(20261022)
    rng=np.random.default_rng(20261022);x=rng.normal(size=(2,8,65,9,9)).astype(np.float16);x[:,:,64]=(rng.random((2,8,9,9))>.15)
    ctx=context64(x);manual=(x[:,:,:64].astype(np.float32)*x[:,:,64,None].astype(np.float32)).sum((1,3,4))/np.maximum(x[:,:,64].astype(np.float32).sum((1,2,3))[:,None],1)
    assert np.array_equal(ctx,manual)
    image=rng.integers(0,256,(90,110,3),dtype=np.uint8);crop=rgb_helper.query_rgb_crop(image,48.2,42.7,32.)
    assert np.array_equal(crop[32,32],image[43,48])
    rgb=np.repeat(crop.transpose(2,0,1)[None],16,axis=0)
    configs=[module.HeadConfig(),module.HeadConfig(target_context=True),module.HeadConfig(target_context=True,rgb_branch=True,rgb_sampling_version='native9'),module.HeadConfig(target_context=True,rgb_branch=True,rgb_sampling_version='native9',sam_branch=True,sam_prior_mode='learned_raw_residual')]
    checks=[]
    for config in configs:
        model=module.QueryHead(config).eval().requires_grad_(False)
        mock_sam=np.ones((16,1,9,9),np.float16)if getattr(config,'sam_branch',False)else None
        z,audit=predict(model,x.reshape(-1,65,9,9),np.repeat(ctx,8,axis=0)if config.target_context else None,rgb if config.rgb_branch else None,'cpu',8,sam=mock_sam)
        assert z.shape==(16,)and np.isfinite(z).all();checks.append(dict(config=config.__dict__,**audit))
    flags,reason=self_flags(dict(initial_pose=dict(available=True,cf_extents=[1,.2,.6],R_cf=np.eye(3).tolist(),centroid=[0,0,2.])),geometry)
    assert len(flags)==8 and reason=='SAVED_INITIAL_REGISTERED_CUBOID_PROXY'
    valid,pixels=query_validity([[2.,3.],[109.8,3.],[-1.,-1.],[0.,0.],[4.,5.],[7.,8.],[9.,9.],[10.,10.]],[90,110],[True]*8)
    assert valid.tolist()==[True,False,False,True,True,True,True,True]
    mocklogit=np.arange(90*110,dtype=np.float32).reshape(90,110)-4000
    mockq=np.array([[20.+k,30.+k]for k in range(8)])
    mockgrid,mockvalid=sam_grid_from_logits(mocklogit,mockq,32.)
    assert mockgrid.shape==(8,1,9,9)and mockvalid.all()
    assert np.array_equal(mockgrid[:,0,4,4],np.clip(mocklogit[np.floor(mockq[:,1]+.5).astype(int),np.floor(mockq[:,0]+.5).astype(int)]/8,-4,4).astype(np.float16))
    assert initial_query_box([[-1.,-1.]]*8,[90,110])is None
    radii=sam_neighborhood_logits(mocklogit,20,30)
    assert radii[0]==mocklogit[30,20]and radii[1]==mocklogit[29:32,19:22].max()and radii[2]==mocklogit[28:33,18:23].max()
    assert sam_neighborhood_logits(mocklogit,0,0)[2]==mocklogit[:3,:3].max()
    assert sam_neighborhood_logits(mocklogit,-1,0)==[None,None,None]
    with tempfile.TemporaryDirectory()as tmp:
        gate=Path(tmp)/'gate.json';write(gate,dict(approved=False))
        try:validate_gate(gate,'mock');raise AssertionError('Unapproved gate accepted')
        except PermissionError:pass
    result=dict(schema='FROZEN_INFERENCE_MOCK_SMOKE_V1',model_checks=checks,RGB_center_parity=True,context_input_only_parity=True,geometry_self_proxy_valid=True,nearest_pixel_and_query_validity=True,unapproved_gate_rejected_before_private_inputs=True,SAM_native9_center_parity=True,SAM_predeclared_radius012_clipped_square_readout_parity=True,SAM_public_model_loads_in_mock=0,private_RGB_cache_metadata_or_score_reads=0,new_PnP_calls=0,new_training=0)
    write(HERE/'MOCK_SMOKE.json',result);print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','smoke']);parser.add_argument('--gate',default=str(ROOT/'CURRENT_EVAL_GATE.json'));parser.add_argument('--candidate');parser.add_argument('--out');parser.add_argument('--device',default='cpu');parser.add_argument('--batch-size',type=int,default=64);parser.add_argument('--sam-device');args=parser.parse_args()
    if args.command=='run':
        if not args.candidate:parser.error('--candidate required for gated inference')
        run(args)
    else:smoke()
