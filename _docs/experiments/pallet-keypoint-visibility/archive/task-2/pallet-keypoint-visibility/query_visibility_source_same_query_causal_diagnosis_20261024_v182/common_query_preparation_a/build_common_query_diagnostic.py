"""Root-entry-only COVER-original-query source diagnostic, never original-query replay or fit."""
from pathlib import Path
from collections import Counter
import argparse,ast,hashlib,importlib.util,json,sys,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
V180=P/'query_visibility_frozen_sam_source_baseline_20261024_v180'
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_COMMON_QUERY_OR_PUBLIC_PIXELS.json';SCOPE_SHA='142ba2fd62dfa16c79bda690bf90814d0a9cdd330898722b8f896edf76b396a1'
OP=V180/'source_end_supervision_preparation_a/source_end_support.py';OP_SHA='928cd3707afa56b9cdb09ad36b1cb3b6bbc8b874a75bfd1c33866a84ed135d73'
SAMPLER=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py';SAMPLER_SHA='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
GROUPS=H/'COMMON512_SAM_GROUPS_INPUT_ONLY.jsonl';SLOTS=H/'COMMON4096_SLOTS_INPUT_ONLY.jsonl';PROTOCOL=H/'PROTOCOL_BEFORE_COMMON_QUERY_PIXEL_BODIES.json';TESTS=H/'GENERATED_INVENTED_CHECKS.json';OUT=V/'common_query_diagnostic_v1'
VARIANTS=['BEFORE','COVER','NEAR_SHAM','FAR_SHAM']
def require(ok,msg):
    if not bool(ok):raise AssertionError(msg)
def permitted(p):
    p=Path(p).resolve();require(p.is_relative_to(P),'Outside source project')
    require(not any(x in str(p).lower() for x in ['_v112','_v177','_v178','_v169','current223','/private_evaluation/']),'Forbidden path');return p
def sha(p):
    h=hashlib.sha256()
    with permitted(p).open('rb') as f:
        for t in iter(lambda:f.read(1<<20),b''):h.update(t)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):p=permitted(b['path']);require(sha(p)==b['sha256'],'SHA '+str(p));return p
def read(p):return json.loads(permitted(p).read_text())
def lines(p):return [json.loads(t) for t in permitted(p).read_text().splitlines() if t]
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def jlines(p,rows):
    with Path(p).open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def qarray(q):return np.asarray([np.nan,np.nan] if q is None else q,np.float64)
def functions():
    require(sha(OP)==OP_SHA and sha(SAMPLER)==SAMPLER_SHA,'Unchanged fixed operator and exact original sampler')
    spec=importlib.util.spec_from_file_location('v182_pinned_source_end_operator',OP);op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
    nodes=[n for n in ast.parse(SAMPLER.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='crop'];require(len(nodes)==1,'Exact crop AST only')
    ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SAMPLER)+'::crop_only','exec'),ns);return op,ns['crop']

def metadata():
    paths=dict(original_TRAIN_manifest=V180/'ALL4096_TRAIN_INPUTS_METADATA_ONLY.jsonl',end_TRAIN_plan=V180/'source_end_supervision_preparation_a/TRAIN4096_END_SUPERVISION_INPUT_ONLY_PLAN.jsonl',geometry=V180/'source_end_supervision_preparation_a/TRAIN1024_FIXED_END_PATCH_GEOMETRY_METADATA.jsonl',old_TRAIN_SAM_complete=V180/'frozen_sam_allTRAIN4096_predictions_v2/COMPLETE.json',old_end_complete=V180/'source_end_supervision_train4096_v1/COMPLETE.json')
    source=lines(paths['original_TRAIN_manifest']);support=lines(paths['end_TRAIN_plan']);geometries=lines(paths['geometry']);old=read(paths['old_TRAIN_SAM_complete']);oldsam={s['scene_index']:s for s in old['scenes']}
    require(len(source)==len(support)==4096 and len(geometries)==1024,'All originalTRAIN counts')
    groups=[];slots=[]
    for base in range(128):
        anchor=source[base*32+8:base*32+16];require(all(r['variant']=='COVER' and r['split']=='TRAIN' for r in anchor),'COVER-original anchor only')
        for vi,variant in enumerate(VARIANTS):
            image=source[base*32+vi*8:base*32+(vi+1)*8];gid='V182_BASE%03d/%s'%(base,variant);rr=[]
            for k,(im,an) in enumerate(zip(image,anchor)):
                index=len(slots);require(im['native_slot']==an['native_slot']==k and im['base_index']==an['base_index']==base,'Same original base/native')
                r=dict(row_id=gid+'/native%d'%k,diagnostic_row_index=index,original_global_index=index,group_id=gid,base_index=base,scene_index=im['scene_index'],native_slot=k,variant=variant,split='TRAIN',
                    query_xy=an['original_query_xy'],L=an['original_L'],original_query_available=an['original_input_flags']['query_available'],original_input_flags=an['original_input_flags'],
                    source_image_original_global_index=im['original_global_index'],anchor_original_global_index=an['original_global_index'],source_image_original_query_id=im['original_occurrence_id'],anchor_original_query_id=an['original_occurrence_id'],
                    image_original_query_xy=im['original_query_xy'],image_original_L=im['original_L'],image_original_flags=im['original_input_flags'],anchor_original_query_xy=an['original_query_xy'],anchor_original_L=an['original_L'],anchor_original_flags=an['original_input_flags'],
                    source_RGB_binding=im['full_RGB_packet'],source_RGB_key=im['full_RGB_key'],decoded_RGB_SHA=im['decoded_RGB_sha256_inherited'],raw_hw=im['raw_hw'],
                    source_image_original_metadata=im,anchor_original_metadata=an,source_end_plan=support[im['original_global_index']],
                    anchor_saved_crop_packet=an['saved_crop_chunk'],anchor_saved_crop_row=an['saved_crop_row'],anchor_saved_rgb96_sha256=an['saved_rgb96_sha256_inherited'],anchor_saved_valid96_sha256=an['saved_valid96_sha256_inherited'],
                    common_original_query_available=an['original_input_flags']['query_available'],original_occurrence_id=gid+'/native%d'%k,original_query_xy=an['original_query_xy'],original_L=an['original_L'],
                    fit_admission=False,production_query_replay=False,GT_model_input=False,SAM_pixels_status='NOT_IN_A_PACKET_PENDING_SEPARATE_SAM')
                slots.append(r);rr.append(r)
            groups.append(dict(group_id=gid,diagnostic_group_index=len(groups),domain='COMMON_COVER_QUERY_TRAIN_DIAGNOSTIC',base_index=base,scene_index=image[0]['scene_index'],variant=variant,split='TRAIN',
                source_RGB_binding=image[0]['full_RGB_packet'],source_RGB_key=image[0]['full_RGB_key'],decoded_RGB_SHA=image[0]['decoded_RGB_sha256_inherited'],raw_hw=image[0]['raw_hw'],
                common_query_xy8=[r['original_query_xy'] for r in anchor],common_L8=[r['original_L'] for r in anchor],common_original_query_available8=[r['original_input_flags']['query_available'] for r in anchor],
                common_initial_points8=anchor[0]['original_initial_points8'],common_q8_context_box_xyxy=anchor[0]['original_q8_context_box_xyxy'],
                anchor_original_global_indices8=[r['original_global_index'] for r in anchor],anchor_original_ids8=[r['original_occurrence_id'] for r in anchor],anchor_original_flags8=[r['original_input_flags'] for r in anchor],
                image_original_global_indices8=[r['original_global_index'] for r in image],image_original_ids8=[r['original_occurrence_id'] for r in image],image_original_query_xy8=[r['original_query_xy'] for r in image],image_original_L8=[r['original_L'] for r in image],image_original_flags8=[r['original_input_flags'] for r in image],
                old_COVER_SAM_scene_binding=oldsam[anchor[0]['scene_index']]['outputs'],old_COVER_SAM_scene_metadata=oldsam[anchor[0]['scene_index']],original_input_metadata8=rr,fit_admission=False,production_query_replay=False))
    require(len(groups)==512 and len(slots)==4096 and sum(not r['original_query_available'] for r in slots)==96,'All512/4096 and cloned anchorU96')
    return groups,slots,geometries,{k:bind(p) for k,p in paths.items()}

def empty_inputs(rows):
    n=len(rows);return dict(rgb96=np.zeros((n,96,96,3),np.uint8),valid96=np.zeros((n,96,96),bool),row_id=np.array([r['row_id'] for r in rows]),original_global_index=np.array([r['original_global_index'] for r in rows],np.int64),native_slot=np.array([r['native_slot'] for r in rows],np.int8),query_xy=np.array([qarray(r['query_xy']) for r in rows],np.float64),L=np.array([np.nan if r['L'] is None else r['L'] for r in rows],np.float64),original_query_available=np.array([r['original_query_available'] for r in rows],bool))
def cover_parity(rgb,valid,original_rgb,original_valid):
    require(exact(rgb,original_rgb) and exact(valid,original_valid),'Mandatory all1024 originalCOVER rawRGB96/valid byteparity')
def owner4(op,scene,q,L):
    if q is None or L is None:return np.full(5,np.nan,np.float64)
    cls=np.where(scene['part_id']<0,2,np.where(scene['part_id']==999,1,0)).astype(np.int8);cls[scene['ambiguity']]=3
    grid=op.original_crop_positions(q,L).reshape(96,96,2)[47:49,47:49].reshape(4,2);w=op.sample_classes(cls,grid).mean(0)
    return w[[0,1,2,3,6]]
def crop_one(crop,rgb,q,L):
    if q is None or L is None:return np.zeros((96,96,3),np.uint8),np.zeros((96,96),bool),False
    im,valid=crop(rgb,qarray(q),L,96);return im.transpose(1,2,0).copy(),valid[0].copy(),True

def invented(op,crop):
    checks={};rgb=np.arange(7*9*3).reshape(7,9,3).astype(np.uint8);q=np.array([.25,2.75]);L=8.
    im,valid,built=crop_one(crop,rgb,q,L);truth=np.zeros_like(im);tv=np.zeros_like(valid)
    for y in range(96):
        for x in range(96):
            xx=q[0]+((x+.5)/96-.5)*L;yy=q[1]+((y+.5)/96-.5)*L;ix=int(np.floor(xx));iy=int(np.floor(yy));dx=xx-ix;dy=yy-iy;acc=np.zeros(3,np.float64)
            for ax,ay,w in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
                if 0<=ix+ax<9 and 0<=iy+ay<7:acc+=rgb[iy+ay,ix+ax].astype(np.float64)*w
            tv[y,x]=-.5<=xx<8.5 and -.5<=yy<6.5
            if tv[y,x]:truth[y,x]=np.floor(np.clip(acc,0,255)+.5).astype(np.uint8)
    checks['exact_original_V90_crop_independent_scalar_four_tap_rounding_and_domain']=exact(im,truth) and exact(valid,tv) and built
    cover_parity(im,valid,truth,tv);checks['mandatory_cover_parity_fullbytes_accepts_exact']=True
    bad=truth.copy();bad[0,0,0]^=1
    try:cover_parity(im,valid,bad,tv)
    except AssertionError:checks['mandatory_cover_parity_rejects_single_changed_rawbyte']=True
    else:checks['mandatory_cover_parity_rejects_single_changed_rawbyte']=False
    z,zv,built=crop_one(crop,rgb,None,None);checks['missing_anchor_preserved_explicit_zero_placeholder_not_BG']=not z.any() and not zv.any() and not built
    p={'part_id':np.zeros((7,9),np.int32),'ambiguity':np.zeros((7,9),bool)};own=owner4(op,p,[-.25,2.75],8.)
    checks['owner5_retains_invalid_mass_without_renormalization']=own[4]>0 and np.isclose(own.sum(),1,atol=2e-15,rtol=0)
    p['part_id'][:]=999;own=owner4(op,p,[4.,3.],1.)
    checks['ownerF_is_separate_literal_class_not_end_E']=own.tolist()==[0.,1.,0.,0.,0.]
    checks['missing_owner_is_allNaN_not_zero_or_BG']=np.isnan(owner4(op,p,None,None)).all()
    require(all(checks.values()),'Invented checks');return {k:bool(v) for k,v in checks.items()}

def prepare():
    require(sha(SCOPE)==SCOPE_SHA,'Root frozen scope');op,crop=functions();groups,slots,geometry,b=metadata();jlines(GROUPS,groups);jlines(SLOTS,slots)
    checks=invented(op,crop);write(TESTS,dict(completed=True,PASS=True,invented_only=True,code=bind(__file__),operator=bind(OP),sampler=bind(SAMPLER),checks=checks,number_checks=len(checks),new_diagnostic_pixels_or_GT_or_SAM_forward_reads=0))
    p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),operator=bind(OP),sampler=bind(SAMPLER),inputs=b,groups=bind(GROUPS),slots=bind(SLOTS),checks=bind(TESTS),
        counts=dict(rows4096=4096,groups512=512,bases128=128,original_COVER_anchors1024=1024,anchor_inputU24=24,common_diagnostic_inputU96=96),
        anchor='Only each base originalCOVER q8/L/alloriginalflags repeated unchanged over4existing images. No BEFOREanchor alternative, GT query, permutation or correction. Image-original querymetadata preserved separately; not productionreplay.',
        sampler_contract='Exact originalV132 savedRGB96 uses pinnedV90 crop AST. All originalCOVER1024 arrays includinginvalid/missing must byteequal saved originalCOVER RGB96/valid96 before any diagnostic payload is admitted.',
        targets='Same pinned V180 registered small-end operator; only recordq/L/inputflags are originalCOVERanchor. Ownercentral4 separateT/F/BG/AMB/INVALID, rawmass preserved; allnative/qualitygoldundefined. GTnevermodelinput; fitadmissionfalse.',
        output_schema='COMMON4096_INPUTS.npz exact8keys rgb96u8[N,96,96,3],valid96bool[N,96,96],row_idU,original_global_indexi64diagnostic0..4095,native_sloti8,query_xyf64[N,2],Lf64[N],original_query_availablebool. No SAMmask/availabilityarrays: pendingseparateB SAM composition, neverzero-substituted. Separate TARGETS exact5keys support_targeti8[N,2],owner_softf64[N,3],owner_targetall-1,owner_ambiguous_mass/owner_invalid_massf64[N]. ROWS/MEASURES JSONL preservebothidentities and diagnosticGT.',
        output=str(OUT),original_source_cache_overwrite=False,root_entry_before_new_pixels=True,new_models_or_fit_or_forward=0,selected_model=None,fully_solved=False)
    write(PROTOCOL,p);write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_common_query_generation=False,source_TRAIN_only=True,code=p['code'],protocol=bind(PROTOCOL),scope=p['scope'],automatic_retry=False))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),groups=p['groups'],slots=p['slots'],checks=p['checks'],new_common_pixels_or_models_or_forward_or_fit=0))

def load_source(binding,g,used):
    path=check(binding);used[str(path)]=binding
    keys=['RGB_uint8','part_id','camera_depth','global_triangle_id','ambiguity','camera_K','camera_R','camera_t','native8_CF_xy','permCF_to_canonical']
    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in keys}
    require(a['RGB_uint8'].dtype==np.uint8 and a['RGB_uint8'].shape==(512,512,3),'Original unmodifiedRGB')
    for key,dtype in [('part_id',np.int32),('camera_depth',np.float64),('global_triangle_id',np.int32),('ambiguity',bool)]:require(a[key].dtype==dtype and a[key].shape==(512,512),'Original GTsupervision schema')
    for key in ('K','R','t'):require(np.array_equal(a['camera_'+key],np.asarray(g['camera'][key],np.float64)),'Same registeredcamera')
    require(np.array_equal(a['native8_CF_xy'],np.asarray(g['native8_CF_xy'],np.float64)) and np.array_equal(a['permCF_to_canonical'],np.asarray(g['permCF_to_canonical'],np.int32)),'Same declarednative metadata')
    return a
def savez(path,a):
    with path.open('xb') as f:np.savez_compressed(f,**a)
    with np.load(path,allow_pickle=False) as z:require(set(z.files)==set(a) and all(exact(z[k],v) for k,v in a.items()),'Serialized payload byteparity')
    return bind(path)

def run(entry):
    e=read(entry);require(e.get('approved_common_query_generation') is True and e.get('source_TRAIN_only') is True and e.get('automatic_retry') is False,'Separate approved root generationentry')
    for k,path in [('code',Path(__file__)),('scope',SCOPE),('protocol',PROTOCOL)]:require(e[k]==bind(path),'Root entry '+k)
    p=read(PROTOCOL);require(p['code']==bind(__file__) and p['operator']==bind(OP) and p['sampler']==bind(SAMPLER) and p['checks']==bind(TESTS),'Frozen calculation')
    groups=lines(check(p['groups']));rows=lines(check(p['slots']));geometry=lines(check(p['inputs']['geometry']));gm={(g['base_index'],g['native_slot']):g for g in geometry};op,crop=functions()
    require(not OUT.exists(),'Preserve previousattempt; no automaticretry');OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),code=p['code'],protocol=bind(PROTOCOL),newfit=False))
    start=time.monotonic();used={};x=empty_inputs(rows);support=np.full((4096,2),-1,np.int8);owner5=np.full((4096,5),np.nan,np.float64);measures=[];coverparity=0;raw_path=None;raw=None
    try:
        for base in range(128):
            rr=rows[base*32:(base+1)*32];before=load_source(rr[0]['source_end_plan']['BEFORE_buffer'],gm[base,0],used);patch={k:dict(op.expected_patch_depth(gm[base,k]['quads_world'],gm[base,k]['camera'],[512,512]),geometry_key=gm[base,k]['geometry_key']) for k in range(8)}
            for vi,variant in enumerate(VARIANTS):
                group=groups[base*4+vi];scene=before if vi==0 else load_source(group['source_RGB_binding'],gm[base,0],used);rgb=scene['RGB_uint8'];require(ash(rgb)==group['decoded_RGB_SHA'],'Original decodedRGB factSHA')
                for r in rr[vi*8:(vi+1)*8]:
                    j=r['diagnostic_row_index'];k=r['native_slot'];im,valid,built=crop_one(crop,rgb,r['query_xy'],r['L']);x['rgb96'][j]=im;x['valid96'][j]=valid
                    if variant=='COVER':
                        bp=r['anchor_saved_crop_packet'];path=Path(bp['path'])
                        if raw_path!=path:
                            check(bp);used[str(path)]=bp
                            with np.load(path,allow_pickle=False) as z:raw={key:z[key].copy() for key in ['rgb96','valid96','crop_constructed','global_index']}
                            raw_path=path
                        ci=r['anchor_saved_crop_row'];require(int(raw['global_index'][ci])==r['anchor_original_global_index'] and bool(raw['crop_constructed'][ci])==built,'Original COVER construction/identity')
                        cover_parity(im,valid,raw['rgb96'][ci],raw['valid96'][ci]);require(ash(im)==r['anchor_saved_rgb96_sha256'] and ash(valid)==r['anchor_saved_valid96_sha256'],'Original COVER pixel facts');coverparity+=1
                    rec=dict(r['source_end_plan']);rec.update(original_occurrence_id=r['row_id'],query_xy=r['query_xy'],L=r['L'],original_input_flags=r['anchor_original_flags'])
                    result=op.process_one(rec,gm[base,k],before,scene,patch[k]);support[j]=[result['visible_support_target'],result['foreign_front_support_target']];owner5[j]=owner4(op,scene,r['query_xy'],r['L'])
                    measures.append(dict(row_id=r['row_id'],diagnostic_row_index=j,source_image_original_global_index=r['source_image_original_global_index'],anchor_original_global_index=r['anchor_original_global_index'],end=result,owner_central4_soft5=None if np.isnan(owner5[j]).all() else owner5[j].tolist(),diagnostic_only=True,fit_admission=False))
                    r.update(actual_rgb96_sha256=ash(im),actual_valid96_sha256=ash(valid),crop_constructed=built,COVER_original_byteparity=(variant=='COVER'),SAM_pixels_status='NOT_IN_A_PACKET_PENDING_SEPARATE_SAM')
            if (base+1)%16==0:print(json.dumps(dict(bases=base+1,rows=(base+1)*32,COVERrows_byteparity=coverparity,seconds=time.monotonic()-start)),flush=True)
        require(coverparity==1024 and len(measures)==4096 and int((~x['original_query_available']).sum())==96,'All originalanchors/U/clones retained')
        y=dict(support_target=support,owner_soft=owner5[:,:3].copy(),owner_target=np.full(4096,-1,np.int8),owner_ambiguous_mass=owner5[:,3].copy(),owner_invalid_mass=owner5[:,4].copy())
        ib=savez(OUT/'COMMON4096_INPUTS.npz',x);tb=savez(OUT/'COMMON4096_TARGETS_DIAGNOSTIC_ONLY.npz',y);mp=OUT/'COMMON4096_ROWS.jsonl';jlines(mp,rows);me=OUT/'COMMON4096_DIAGNOSTIC_MEASURES.jsonl';jlines(me,measures)
        for b in used.values():check(b)
        for b in [p['groups'],p['slots'],p['inputs']['geometry']]:check(b)
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTOCOL),scope=p['scope'],entry=bind(entry),input_groups=p['groups'],input_slots=p['slots'],
            rows4096=4096,groups512=512,bases128=128,original_COVERanchors1024_byteparity=True,anchor_inputU24_cloned_to96=True,inputs=ib,targets=tb,metadata=bind(mp),measures=bind(me),
            SAM_arrays_present=False,SAM_pixels_status='PENDING_SEPARATE_SAM_NO_SUBSTITUTION',all_source_SHA_unchanged=True,original_source_caches_unchanged=True,common_anchor_qL_flags_immutable=True,
            production_query_replay=False,diagnostic_only=True,fit_admission=False,GT_model_input=False,new_model_SAM_forward_fit_or_threshold=0,selected_model=None,fully_solved=False,automatic_retry=False,seconds=time.monotonic()-start))
    except BaseException:
        write(OUT/'FAILED_EXECUTION_PRESERVED.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);args=ap.parse_args();prepare() if args.prepare else run(args.entry)
