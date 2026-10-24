"""Root-entry-only original SOURCE TRAIN jitter/BEFORE-anchor raw/target producer.

Preparation reads text/metadata and invented arrays only. GT is supervision only.
"""
from pathlib import Path
from collections import Counter
import argparse,ast,hashlib,importlib.util,json,time,traceback,zipfile
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
V180=P/'query_visibility_frozen_sam_source_baseline_20261024_v180'
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_AUGMENTATION_OR_FIT.json';SCOPE_SHA='d8394deb1c0a5844ecb987447391d7fd61ce5aa80d073cf33adf76ff385fcb95'
OP=V180/'source_end_supervision_preparation_a/source_end_support.py';OP_SHA='928cd3707afa56b9cdb09ad36b1cb3b6bbc8b874a75bfd1c33866a84ed135d73'
SAMPLER=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py';SAMPLER_SHA='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
SCENES=H/'SCENE512_ORIGINALS_INPUT_ONLY.jsonl';ROWS=H/'AUG65536_ROW_REFS_INPUT_ONLY.jsonl'
PROTOCOL=H/'PROTOCOL_BEFORE_NEW_SOURCE_AUGMENTATION_PIXEL_BODIES.json';CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
OUT=V/'augmentation_source_train_v1';VARIANTS=['BEFORE','COVER','NEAR_SHAM','FAR_SHAM'];N=65536
def require(ok,msg):
    if not bool(ok):raise AssertionError(msg)
def permitted(p):
    p=Path(p).resolve();require(p.is_relative_to(P),'Source project only')
    require(not any(x in str(p).lower() for x in ['_v112','_v169','_v177','_v178','current223','synth_val','validation2048','/private_','source_validation']),'TRAIN-only whitelist')
    return p
def sha(p):
    h=hashlib.sha256()
    with permitted(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):p=permitted(b['path']);require(sha(p)==b['sha256'],'SHA '+str(p));return p
def read(p):return json.loads(permitted(p).read_text())
def lines(p):return [json.loads(s) for s in permitted(p).read_text().splitlines() if s]
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def jlines(p,rr):
    with Path(p).open('x') as f:
        for r in rr:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
def ah(a):
    b=memoryview(np.ascontiguousarray(a)).cast('B');h=hashlib.sha256()
    for i in range(0,len(b),1048576):h.update(b[i:i+1048576])
    return h.hexdigest()
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def qarray(q):return np.asarray([np.nan,np.nan] if q is None else q,np.float64)
def validq(q,L):return q is not None and L is not None and np.isfinite(q).all() and np.isfinite(L) and L>0 and not np.array_equal(q,[-1.,-1.])
def functions():
    require(sha(OP)==OP_SHA and sha(SAMPLER)==SAMPLER_SHA,'Pinned fixed operator/original sampler')
    spec=importlib.util.spec_from_file_location('v183_frozen_end_operator',OP);op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
    nodes=[n for n in ast.parse(SAMPLER.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='crop'];require(len(nodes)==1,'Exact original crop only')
    ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SAMPLER)+'::crop_only','exec'),ns);return op,ns['crop']
def transform(q,L,state):
    # Missing source queries stay missing in every state. Factual U is never recomputed.
    outq=None if q is None else (np.asarray(q,np.float64)+[state['dx'],state['dy']]).tolist()
    return outq,None if L is None else float(np.float64(L)*np.float64(state['scale']))
def original_sources():
    return dict(original_manifest=V180/'ALL4096_TRAIN_INPUTS_METADATA_ONLY.jsonl',end_plan=V180/'source_end_supervision_preparation_a/TRAIN4096_END_SUPERVISION_INPUT_ONLY_PLAN.jsonl',geometry=V180/'source_end_supervision_preparation_a/TRAIN1024_FIXED_END_PATCH_GEOMETRY_METADATA.jsonl',old_TRAIN_SAM_complete=V180/'frozen_sam_allTRAIN4096_predictions_v2/COMPLETE.json')
def metadata():
    scope=read(SCOPE);states=scope['training_synthetic']['states15'];require(len(states)==15 and states[0]==dict(name='NOMINAL',dx=0.,dy=0.,scale=1.),'Root ordered15states')
    src=original_sources();original=lines(src['original_manifest']);plan=lines(src['end_plan']);geo=lines(src['geometry']);sam=read(src['old_TRAIN_SAM_complete']);old={g['scene_index']:g for g in sam['scenes']}
    require(len(original)==len(plan)==4096 and len(geo)==1024 and len(old)==512,'All TRAIN source only')
    require([r['original_global_index'] for r in original]==list(range(4096)) and all(r['split']=='TRAIN' for r in original),'Original orderedTRAIN4096')
    scenes=[]
    for si in range(512):
        eight=original[si*8:si*8+8];r=eight[0];require([x['native_slot'] for x in eight]==list(range(8)) and all(x['scene_index']==si for x in eight),'Original512 native8 groups')
        require(old[si]['original_input_metadata8']==eight,'Original SAM source metadata exactly aligned')
        scenes.append(dict(scene_index=si,base_index=r['base_index'],variant=r['variant'],split='TRAIN',factors=r['factors'],source_RGB_binding=r['full_RGB_packet'],source_RGB_key=r['full_RGB_key'],decoded_RGB_SHA=r['decoded_RGB_sha256_inherited'],raw_hw=r['raw_hw'],original_RGB_PNG=r['raw_RGB_PNG'],original_input_metadata8=eight,old_SAM_output=old[si]['outputs'],old_SAM_scene_metadata={k:v for k,v in old[si].items() if k!='original_input_metadata8'}))
    rr=[]
    def row(im,anchor,state,st,family):
        i=len(rr);si=im['scene_index'];base=im['base_index'];k=im['native_slot'];q,L=transform(anchor['original_query_xy'],anchor['original_L'],state)
        sid='SELF_SCENE_%04d'%si if family=='SELF_SCENE15' else 'BEFORE_ANCHOR_BASE%03d/%s'%(base,im['variant'])
        rid='V183_SELF_%04d_NATIVE%d_%s'%(si,k,state['name']) if family=='SELF_SCENE15' else 'V183_BEFORE_BASE%03d_%s_NATIVE%d'%(base,im['variant'],k)
        return dict(row_id=rid,original_global_index=i,diagnostic_row_index=i,native_slot=k,query_xy=q,L=L,original_query_available=anchor['original_input_flags']['query_available'],split='TRAIN',family=family,scene_index=si,base_index=base,variant=im['variant'],source_image_original_global_index=im['original_global_index'],anchor_original_global_index=anchor['original_global_index'],anchor_scene_index=anchor['scene_index'],source_image_original_query_id=im['original_occurrence_id'],anchor_original_query_id=anchor['original_occurrence_id'],sam_group_id=sid,state_index=st,state_name=state['name'],dx=state['dx'],dy=state['dy'],L_scale=state['scale'],original_flags_ref=[anchor['scene_index'],k],image_flags_ref=[si,k])
    for im in original:
        for st,state in enumerate(states):rr.append(row(im,im,state,st,'SELF_SCENE15'))
    for base in range(128):
        for vi,var in enumerate(VARIANTS):
            for k in range(8):rr.append(row(original[base*32+vi*8+k],original[base*32+k],states[0],0,'BEFORE_ANCHOR4'))
    require(len(rr)==N and len({r['row_id'] for r in rr})==N and [r['original_global_index'] for r in rr]==list(range(N)),'All65536distinct row identities')
    require(sum(not r['original_input_flags']['query_available'] for r in original)==80,'OriginalU80')
    require(sum(not r['original_input_flags']['query_available'] for r in original if r['variant']=='BEFORE')==16,'BEFOREU16')
    require(sum(not r['original_query_available'] for r in rr)==1264,'U80x15+U16x4')
    require(Counter(r['family'] for r in rr)==dict(SELF_SCENE15=61440,BEFORE_ANCHOR4=4096),'Full family census')
    return scenes,rr,geo,states,{k:bind(p) for k,p in src.items()}
def crop_one(crop,rgb,q,L):
    if q is None or L is None:return np.zeros((96,96,3),np.uint8),np.zeros((96,96),bool),False
    im,valid=crop(rgb,qarray(q),L,96);return im.transpose(1,2,0).copy(),valid[0].copy(),True
def owner_classes(scene):
    c=np.where(scene['part_id']<0,2,np.where(scene['part_id']==999,1,0)).astype(np.int8);c[scene['ambiguity']]=3;return c
def measured(op,end_classes,owner_cls,patch,q,L,available):
    if not validq(q,L):return [-1,-1],np.full(7,np.nan),np.full(5,np.nan),'ORIGINAL_Q_OR_L_MISSING'
    positions=op.original_crop_positions(q,L);s=op.summarize(op.sample_classes(end_classes,positions),patch,available)
    center=positions.reshape(96,96,2)[47:49,47:49].reshape(4,2);own=op.sample_classes(owner_cls,center).mean(0)[[0,1,2,3,6]]
    return [s['visible_support_target'],s['foreign_front_support_target']],np.asarray(s['full96_average_mass'],np.float64),own,'|'.join(s['reasons'])
def nominal_parity(rgb,valid,old_rgb,old_valid):require(exact(rgb,old_rgb) and exact(valid,old_valid),'All4096 ownnominal savedRGB96/valid fullbyteparity admission gate')
def savez_verified(p,data):
    with p.open('xb') as f:np.savez_compressed(f,**data)
    # Streaming NPY payload checks avoid allocating a second multi-GB raw packet.
    with zipfile.ZipFile(p) as z:
        require(set(z.namelist())=={k+'.npy' for k in data},'Serialized exact keys')
        for key,a in data.items():
            with z.open(key+'.npy') as f:
                version=np.lib.format.read_magic(f)
                header=np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f)
                shape,fortran,dtype=header;require(shape==a.shape and dtype==a.dtype and not fortran,'Serialized dtype/shape/order')
                h=hashlib.sha256()
                for b in iter(lambda:f.read(1048576),b''):h.update(b)
                require(h.hexdigest()==ah(a),'Serialized payload fullbytes')
    return bind(p)
def invented(op,crop,states):
    tests={};rgb=np.arange(7*9*3).reshape(7,9,3).astype(np.uint8);q=[.25,2.75];L=8.;im,valid,built=crop_one(crop,rgb,q,L);truth=np.zeros_like(im);tv=np.zeros_like(valid)
    for yy in range(96):
        for xx in range(96):
            x=q[0]+((xx+.5)/96-.5)*L;y=q[1]+((yy+.5)/96-.5)*L;i=int(np.floor(x));j=int(np.floor(y));fx=x-i;fy=y-j;v=np.zeros(3,np.float64)
            for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
                wt=(fx if dx else 1-fx)*(fy if dy else 1-fy)
                if 0<=i+dx<9 and 0<=j+dy<7:v+=rgb[j+dy,i+dx].astype(np.float64)*wt
            tv[yy,xx]=-.5<=x<8.5 and -.5<=y<6.5
            if tv[yy,xx]:truth[yy,xx]=np.floor(np.clip(v,0,255)+.5).astype(np.uint8)
    tests['exact_original_sampler_independent_scalar4tap_and_round_domain']=exact(im,truth) and exact(valid,tv) and built
    expected=[(0.,0.,1.)]+[(float(x),float(y),1.) for y in [-1,0,1] for x in [-1,0,1] if (x,y)!=(0,0)]+[(-.5,0.,1.),(.5,0.,1.),(0.,-.5,1.),(0.,.5,1.),(0.,0.,.99),(0.,0.,1.01)]
    tests['root15state_order_nominal_dymajor_halfaxes_scale']=len(states)==15 and [(s['dx'],s['dy'],s['scale']) for s in states]==expected
    tests['FP64_halfpixel_and_scale_not_clipped']=transform([-.25,.25],10.,states[9])==([-.75,.25],10.) and transform([1.,2.],10.,states[13])==([1.,2.],9.9)
    tests['missing_original_query_not_manufactured']=all(transform(None,None,s)==(None,None) for s in states)
    nominal_parity(im,valid,truth,tv);bad=truth.copy();bad[0,0,0]^=1
    try:nominal_parity(im,valid,bad,tv)
    except AssertionError:tests['nominal_gate_rejects_single_RGB_byte']=True
    else:tests['nominal_gate_rejects_single_RGB_byte']=False
    bv=tv.copy();bv[0,0]^=1
    try:nominal_parity(im,valid,truth,bv)
    except AssertionError:tests['nominal_gate_rejects_single_valid_bit']=True
    else:tests['nominal_gate_rejects_single_valid_bit']=False
    p=dict(geometry_valid=True,projected_region_out_of_frame=False);end=np.zeros((5,5),np.int8);end[1,1]=1;owner=np.zeros((5,5),np.int8)
    a=measured(op,end,owner,p,[2.,1.],.1,True);b=measured(op,end,owner,p,[3.,1.],.1,True)
    tests['jitter_recomputes_boundary_support_not_copied']=a[0]==[1,0] and b[0]==[0,-1] and not np.array_equal(a[1],b[1])
    end[:]=3;a=measured(op,end,owner,p,[2.,2.],.1,True);tests['self_UNKNOWN_not_external_negative']=a[0]==[-1,-1] and a[1][3]==1
    end[:]=1;a=measured(op,end,owner,p,[2.,2.],.1,False);tests['original_inputU_preserved_even_valid_RGB']=a[0]==[-1,-1]
    end[:]=0;owner[:]=1;a=measured(op,end,owner,p,[2.,2.],.1,True);tests['literal_ownerF_not_renamed_endforeign']=a[0]==[0,-1] and a[2].tolist()==[0.,1.,0.,0.,0.]
    a=measured(op,end,owner,p,[-.25,2.],1.,True);tests['owner_INVALID_mass_retained_no_renormalization']=a[2][4]>0 and np.isclose(a[2].sum(),1,rtol=0,atol=2e-15)
    a=measured(op,end,owner,p,None,None,False);tests['missing_all_NaN_owner_end_not_BG']=a[0]==[-1,-1] and np.isnan(a[1]).all() and np.isnan(a[2]).all()
    # Verify cached-classmap runtime path equals direct frozen process_one for all15 invented states.
    cam=dict(K=[[10.,0.,4.],[0.,10.,4.],[0.,0.,1.]],R=np.eye(3).tolist(),t=[0.,0.,0.]);quad=[[-.1,-.1,2.],[.1,-.1,2.],[.1,.1,2.],[-.1,.1,2.]]
    g=dict(base_index=0,native_slot=0,canonical_index=0,primary_part_id=0,geometry_key='invented',quads_world=[quad]*3,camera=cam,raw_hw=[8,8],native8_CF_xy=[[0.,0.]]*8,permCF_to_canonical=list(range(8)))
    before=dict(part_id=np.zeros((8,8),np.int32),camera_depth=np.full((8,8),2.,np.float64),global_triangle_id=np.zeros((8,8),np.int32),ambiguity=np.zeros((8,8),bool),camera_K=np.array(cam['K']),camera_R=np.array(cam['R']),camera_t=np.array(cam['t']),native8_CF_xy=np.zeros((8,2)),permCF_to_canonical=np.arange(8,dtype=np.int32));scene={k:v.copy() for k,v in before.items()};scene['part_id'][4,4]=999;scene['camera_depth'][4,4]=1.
    patch=dict(op.expected_patch_depth(g['quads_world'],cam,[8,8]),geometry_key='invented');cls=op.classify_end_pixels(before,scene,patch,0);own=owner_classes(scene);parity=True
    for state in states:
        q,L=transform([4.,4.],1.,state);record=dict(split='TRAIN',base_index=0,native_slot=0,declared_canonical_index=0,primary_part_id=0,query_xy=q,L=L,original_global_index=0,original_occurrence_id='invented',scene_index=0,variant='COVER',original_input_flags={'query_available':True},original_semantic_state='UNCHANGED')
        direct=op.process_one(record,g,before,scene,patch);cached=measured(op,cls,own,patch,q,L,True)
        parity&=cached[0]==[direct['visible_support_target'],direct['foreign_front_support_target']] and np.array_equal(cached[1],direct['full96_average_mass'])
    tests['cached_classmap_sampling_matches_direct_frozen_operator_15states']=parity
    tmp=H/'INVENTED_SERIALIZATION_ONLY.npz';a=dict(toy=np.array([[np.nan,1.]],np.float64),flags=np.array([False,True]),ids=np.array(['a','b']));savez_verified(tmp,a);tests['actual_streaming_serializer_NaN_dtype_shape_byteparity']=True
    require(all(tests.values()),'Invented checks');return {k:bool(v) for k,v in tests.items()}
def prepare():
    require(sha(SCOPE)==SCOPE_SHA,'Frozen root scope');op,crop=functions();scenes,rr,geo,states,sources=metadata();jlines(SCENES,scenes);jlines(ROWS,rr);tests=invented(op,crop,states)
    write(CHECKS,dict(completed=True,PASS=True,invented_only=True,code=bind(__file__),operator=bind(OP),sampler=bind(SAMPLER),tests=tests,checks=len(tests),source_pixel_GT_model_NPZ_header_hash_or_forward_reads=0))
    p=dict(completed=True,prepared_only=True,scope=bind(SCOPE),code=bind(__file__),operator=bind(OP),sampler=bind(SAMPLER),sources=sources,scenes=bind(SCENES),rows=bind(ROWS),checks=bind(CHECKS),states15=states,
        counts=dict(rows=65536,SELF_SCENE15=61440,BEFORE_ANCHOR4=4096,original_scenes=512,original_rows=4096,original_inputU=80,original_BEFORE_inputU=16,augmentation_inputU=1264),
        identity='SELF i*15+state0..61439; BEFORE61440+base32+variant8+native. Allnew augmentation IDs. Resolve image-original andanchor q/L/flags/IDs in SCENE512.original_input_metadata8[native]. No originalcache modification/GTquery; no V182diagnostic payload fitinput.',
        q_and_availability='RootFP64 q+dxdy/L*scale; originalavailability andflags referenced unchanged, even if jitter moves domain. New valid96/crop_constructed are separate input diagnostics. Missingq/L retainnull; unavailable targetsUNKNOWN.',
        targets='Same frozenV180 quads/camera/registered-end classes7 and existence2support. Patchdepth once/base/native; sceneclasses once/base/variant/native; every65536row grid sampled anew. central4owner5 separately T/F/BG/AMB/INVALID rawmass, no rowrenormalization orUNKNOWNimputation; owner_targetall-1. NativeD/E/S not relabeled orqualitygold.',
        nominal_admission='All4096SELF_NOMINAL includingU must equal original saved RGB96/valid96 wholebytes/dtypes/facthashes/construction. No payload/cacheadmissionuntilallpass. Failclosed/preservefailure/noautoretry.',
        SAM='A raw8packet has noSAMmask. SELF512 oldSAM fullimage originalsceneq8prompt remains unchanged/resampleaugqL later. BEFORE512 uses originalBEFOREq8prompt/B runner; nooldpacketaugIDalias. Rootcomposer suppliesSAM2 separately.',
        memory='RGB96/valid96 memmapped scratch2.4GB; base-local8patchdepths and32classmaps; stream NPZ member hashing verifies allserialized bytes without another multiGB rawarray. Scratch retainedonfailure, deletedonlysuccess.',
        output=str(OUT),input_keys=['rgb96','valid96','row_id','original_global_index','native_slot','query_xy','L','original_query_available'],target_keys=['support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass'],
        complete_contract=dict(rows=65536,inputs='binding',targets='binding',metadata='binding',scene_metadata='binding',SAM_arrays_present=False,original_nominal4096_raw_valid_byteparity=True,original_aug_inputU=1264),
        new_pixel_or_GT_generation_requires_separate_root_entry=True,newmodel_forward_or_fit=0,selected_model=None,fully_solved=False)
    write(PROTOCOL,p);write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_source_augmentation_generation=False,source_TRAIN_only=True,code=p['code'],scope=p['scope'],protocol=bind(PROTOCOL),automatic_retry=False))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),scene_metadata=p['scenes'],row_metadata=p['rows'],checks=p['checks'],new_source_pixel_GT_header_hash_model_forward_fit_reads=0))
    print(json.dumps(dict(PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL))),flush=True)
def load_source(binding,geometry,used):
    path=check(binding);used[str(path)]=binding
    keys=['RGB_uint8','part_id','camera_depth','global_triangle_id','ambiguity','camera_K','camera_R','camera_t','native8_CF_xy','permCF_to_canonical']
    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in keys}
    require(a['RGB_uint8'].dtype==np.uint8 and a['RGB_uint8'].shape==(512,512,3),'SourceRGB schema')
    for k,dt in [('part_id',np.int32),('camera_depth',np.float64),('global_triangle_id',np.int32),('ambiguity',bool)]:require(a[k].shape==(512,512) and a[k].dtype==dt,'GTsupervision schema '+k)
    for k in ['K','R','t']:require(np.array_equal(a['camera_'+k],np.asarray(geometry['camera'][k],np.float64)),'Registered original camera')
    require(np.array_equal(a['native8_CF_xy'],np.asarray(geometry['native8_CF_xy'],np.float64)) and np.array_equal(a['permCF_to_canonical'],np.asarray(geometry['permCF_to_canonical'],np.int32)),'Declared native metadata')
    return a
def run(entry):
    e=read(entry);require(e.get('approved_source_augmentation_generation') is True and e.get('source_TRAIN_only') is True and e.get('automatic_retry') is False,'Root approved actual generationentry')
    for k,path in [('code',Path(__file__)),('scope',SCOPE),('protocol',PROTOCOL)]:require(e[k]==bind(path),'Root entry '+k)
    p=read(PROTOCOL)
    for key,path in [('operator',OP),('sampler',SAMPLER),('checks',CHECKS)]:require(p[key]==bind(path),'Frozen '+key)
    scenes=lines(check(p['scenes']));rr=lines(check(p['rows']));geometry=lines(check(p['sources']['geometry']));gm={(g['base_index'],g['native_slot']):g for g in geometry};op,crop=functions()
    require(not OUT.exists(),'Fresh output/nooverwrite orautomaticretry');OUT.mkdir();work=OUT/'_WORK_MEMMAP';work.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),code=p['code'],protocol=bind(PROTOCOL),source_TRAIN_only=True,newfit=False))
    start=time.monotonic();used={};nominal=0;computed=0;oldpath=None;oldcrop=None
    X=dict(rgb96=np.lib.format.open_memmap(work/'rgb96.npy',mode='w+',dtype=np.uint8,shape=(N,96,96,3)),valid96=np.lib.format.open_memmap(work/'valid96.npy',mode='w+',dtype=bool,shape=(N,96,96)),row_id=np.asarray([r['row_id'] for r in rr]),original_global_index=np.arange(N,dtype=np.int64),native_slot=np.asarray([r['native_slot'] for r in rr],np.int8),query_xy=np.asarray([qarray(r['query_xy']) for r in rr],np.float64),L=np.asarray([np.nan if r['L'] is None else r['L'] for r in rr],np.float64),original_query_available=np.asarray([r['original_query_available'] for r in rr],bool))
    Y=dict(support_target=np.full((N,2),-1,np.int8),owner_soft=np.full((N,3),np.nan,np.float64),owner_target=np.full(N,-1,np.int8),owner_ambiguous_mass=np.full(N,np.nan,np.float64),owner_invalid_mass=np.full(N,np.nan,np.float64))
    M=dict(end_mass7=np.full((N,7),np.nan,np.float64),projected_region_out_of_frame=np.zeros(N,bool));reasons=Counter();targets=[Counter(),Counter()]
    try:
        for base in range(128):
            base_scenes=scenes[base*4:base*4+4];before=load_source(base_scenes[0]['source_RGB_binding'],gm[base,0],used);patches={k:dict(op.expected_patch_depth(gm[base,k]['quads_world'],gm[base,k]['camera'],[512,512]),geometry_key=gm[base,k]['geometry_key']) for k in range(8)}
            for vi,var in enumerate(VARIANTS):
                group=base_scenes[vi];scene=before if vi==0 else load_source(group['source_RGB_binding'],gm[base,0],used);rgb=scene['RGB_uint8'];require(ah(rgb)==group['decoded_RGB_SHA'],'Original fullRGB fact')
                owner_cls=owner_classes(scene)
                for k in range(8):
                    g=gm[base,k];patch=patches[k];end_cls=op.classify_end_pixels(before,scene,patch,g['primary_part_id'])
                    original=group['original_input_metadata8'][k];original_i=original['original_global_index'];indices=list(range(original_i*15,original_i*15+15))+[61440+base*32+vi*8+k]
                    for j in indices:
                        r=rr[j];require(r['scene_index']==group['scene_index'] and r['native_slot']==k,'Sourceimage mapping')
                        anchor=scenes[r['anchor_scene_index']]['original_input_metadata8'][k];state=p['states15'][r['state_index']];q,L=transform(anchor['original_query_xy'],anchor['original_L'],state)
                        require(r['query_xy']==q and r['L']==L and r['original_query_available']==anchor['original_input_flags']['query_available'],'Frozen augmentation qL/U')
                        im,valid,built=crop_one(crop,rgb,q,L);X['rgb96'][j]=im;X['valid96'][j]=valid
                        if r['family']=='SELF_SCENE15' and r['state_index']==0:
                            cb=original['saved_crop_chunk'];cp=Path(cb['path'])
                            if cp!=oldpath:
                                check(cb);used[str(cp)]=cb
                                with np.load(cp,allow_pickle=False) as z:oldcrop={key:z[key].copy() for key in ['rgb96','valid96','crop_constructed','global_index']}
                                oldpath=cp
                            ci=original['saved_crop_row'];require(int(oldcrop['global_index'][ci])==original_i and bool(oldcrop['crop_constructed'][ci])==built,'Original ownnominal construction')
                            nominal_parity(im,valid,oldcrop['rgb96'][ci],oldcrop['valid96'][ci]);require(ah(im)==original['saved_rgb96_sha256_inherited'] and ah(valid)==original['saved_valid96_sha256_inherited'],'Ownnominal saved facthashes');nominal+=1
                        sup,mass,own,reason=measured(op,end_cls,owner_cls,patch,q,L,r['original_query_available']);Y['support_target'][j]=sup;M['end_mass7'][j]=mass;M['projected_region_out_of_frame'][j]=patch['projected_region_out_of_frame'];Y['owner_soft'][j]=own[:3];Y['owner_ambiguous_mass'][j]=own[3];Y['owner_invalid_mass'][j]=own[4]
                        r.update(actual_rgb96_sha256=ah(im),actual_valid96_sha256=ah(valid),crop_constructed=built,target_recomputed=True);reasons[reason]+=1
                        for c in range(2):targets[c][sup[c]]+=1
                        computed+=1
            if (base+1)%8==0:print(json.dumps(dict(bases=base+1,rows=computed,own_nominal_byteparity=nominal,seconds=time.monotonic()-start)),flush=True)
        require(nominal==4096 and computed==N and int((~X['original_query_available']).sum())==1264,'Allnominal gate/alltargets/U1264')
        require((Y['support_target'][~X['original_query_available']]==-1).all(),'UtargetUNKNOWN')
        # New training payloads admitted only after every ownnominal row passes.
        ib=savez_verified(OUT/'AUG65536_INPUTS.npz',X);tb=savez_verified(OUT/'AUG65536_TARGETS.npz',Y);mb=savez_verified(OUT/'AUG65536_MEASURES_DIAGNOSTIC_ONLY.npz',M);rp=OUT/'AUG65536_ROWS.jsonl';jlines(rp,rr)
        for b in used.values():check(b)
        for b in [p['scenes'],p['rows'],p['sources']['geometry']]:check(b)
        X['rgb96'].flush();X['valid96'].flush();del X
        for f in work.iterdir():f.unlink()
        work.rmdir()
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,source_TRAIN_only=True,scope=p['scope'],code=p['code'],protocol=bind(PROTOCOL),entry=bind(entry),rows=N,inputs=ib,targets=tb,metadata=bind(rp),scene_metadata=p['scenes'],row_input_metadata=p['rows'],measures=mb,SAM_arrays_present=False,original_nominal4096_raw_valid_byteparity=True,original_aug_inputU=1264,families=dict(SELF_SCENE15=61440,BEFORE_ANCHOR4=4096),target_rows_recomputed=N,registered_classmaps_computed=4096,fixed_base_native_patch_maps=1024,original_scenes=512,original_inputU=80,original_BEFORE_inputU=16,all_source_SHA_unchanged=True,original_source_cache_unchanged=True,original_scene_SAM_prompt_unchanged_for_SELF=True,GT_model_input=False,new_native_visibility_or_point_quality_gold=False,augmentation_training_input_admission=True,admission_basis='Root approved entry plus all4096ownnominal raw/valid exact; targets separately recomputed source-only. Does not admit originalV182diagnostic or claim physicalnativegold.',support_target_counts={'visible':dict(targets[0]),'foreign':dict(targets[1])},reasons=dict(reasons),new_model_SAM_forward_or_fit=0,selected_model=None,fully_solved=False,automatic_retry=False,seconds=time.monotonic()-start))
        print(json.dumps(dict(PASS=True,complete=bind(OUT/'COMPLETE.json'))),flush=True)
    except BaseException:
        write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False,scratch_preserved=True));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare() if a.prepare else run(a.entry)
