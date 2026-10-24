"""Exact400 material-source query freeze; depth/mask execution needs root entry."""
from pathlib import Path
import argparse,collections,hashlib,importlib.util,json,sys,time,traceback
sys.dont_write_bytecode=True
import cv2
import numpy as np
import trimesh
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_EXACT3200_MATERIAL_TEACHER_POLICY_BEFORE_LABELS.json'
POLICY_SHA='2b492b0334ec6c037205b44cc71b03f7a60f84afe9c0250f58e91fd03425499b'
OLD=P/'query_visibility_ycb_real_surface_source_20261023_v109/producer/build_surface_source.py'
OLD_SHA='a431eaa92c06b5a9a7e2d14ac7a8d6fde18c685edf988c7d099ce02411b65f38'
assert hashlib.sha256(OLD.read_bytes()).hexdigest()==OLD_SHA
spec=importlib.util.spec_from_file_location('v113_exact_v109',OLD)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha,bind,read,check,write,jsonl,rows,module=old.sha,old.bind,old.read,old.check,old.write,old.jsonl,old.rows,old.module
native_grid_function,jitter_q=old.native_grid_function,old.jitter_q
geometry_state,classify,compact_ray=old.geometry_state,old.classify,old.compact_ray
COUNT_KEYS,KERNEL,NATIVE_GRID=old.COUNT_KEYS,old.KERNEL,old.NATIVE_GRID
CENTRE,CENTRE_SHA=old.CENTRE,old.CENTRE_SHA
PREP=V/'query_preparation';OUT=V/'corpus_build_v1'
IDS=(16,21);STATES=('D','E','S','U')

def prepare():
 assert sha(POLICY)==POLICY_SHA and not PREP.exists()
 policy=read(POLICY)
 for key in('fixed_geometry','geometry_result','feasibility','input_plan','GT_lineage','cache_bindings','scene9_exposure',
            'source_policy_reused_exact','centre_policy_reused_exact','teacher_code_pure_classify_geometry_reused'):
  check(policy[key])
 assert policy['teacher_code_pure_classify_geometry_reused']==bind(OLD)
 plans=rows(policy['input_plan']['path']);lineage=rows(policy['GT_lineage']['path'])
 assert len(plans)==len(lineage)==400
 assert collections.Counter(x['original_split']for x in plans)==dict(train=300,val=100)
 geometry=read(policy['geometry_result']['path']);objects={x['object_id']:x for x in geometry['objects']}
 assert set(objects)=={16,21}
 cache={x['path']:x for x in read(policy['cache_bindings']['path'])['files']}
 native=native_grid_function();rng=np.random.Generator(np.random.PCG64(213));inputs=[];references=[];controls={}
 for i,(plan,g)in enumerate(zip(plans,lineage)):
  assert i==plan['plan_row_index']==g['plan_row_index']and plan['input_id']==g['input_id']
  assert plan['original_split']==g['original_split']and plan['image']['path']==g['image']['path']and plan['image']['sha256']==g['image']['sha256']
  assert plan['initial_points']is None and plan['roi_side']is None and plan['query_box_xyxy']is None
  for b in g['controls'].values():
   check(b);controls[b['path']]=dict(path=b['path'],sha256=b['sha256'])
  scene=Path(g['depth']['path']).parent.parent;gt=read(scene/'scene_gt.json')[str(g['frame_id'])];cam=read(scene/'scene_camera.json')[str(g['frame_id'])];target=gt[g['instance_id']]
  assert target['obj_id']==g['object_id']and target['cam_R_m2c']==g['cam_R_m2c']and target['cam_t_m2c']==g['cam_t_m2c_mm']
  assert cam['cam_K']==g['K']and cam['depth_scale']==g['depth_scale_to_mm']
  assert len(gt)==g['complete_scene_GT_owner_count']==len(g['all_visible_owners'])
  for j,owner in enumerate(g['all_visible_owners']):
   assert owner['instance_index']==j and owner['object_id']==gt[j]['obj_id']
   b=owner['visible_mask'];assert b['exists']and cache[b['path']]['sha256']==b['sha256']
  for b in(g['image'],g['depth'],g['own_full_mask'],g['own_visible_mask']):
   assert b['exists']and cache[b['path']]['sha256']==b['sha256']
  obj=objects[g['object_id']];check(obj['model'])
  assert len(obj['candidates'])==8 and [c['original_mesh_vertex_index']for c in obj['candidates']]==g['original_mesh_vertex_indices']
  xyz=np.asarray([c['xyz_mm']for c in obj['candidates']],np.float64)
  R=np.asarray(target['cam_R_m2c']).reshape(3,3);t=np.asarray(target['cam_t_m2c']);K=np.asarray(cam['cam_K']).reshape(3,3)
  cam_xyz=xyz@R.T+t;reference=(cam_xyz@K.T)[:,:2]/cam_xyz[:,2,None]
  assert np.isfinite(reference).all()and(cam_xyz[:,2]>0).all()
  query=jitter_q(reference,rng);hw=plan['raw_hw'];_,_,side=native(query,hw)
  assert np.linalg.norm(query-reference,axis=1).max()<=4
  split='TRAIN'if plan['original_split']=='train'else'VALIDATION'
  inputs.append(dict(input_id=plan['input_id'],instance_row_index=i,plan_row_index=i,image=plan['image'],raw_hw=hw,
                     initial_points=query.tolist(),roi_side=side,
                     query_box_xyxy=[float(query[:,0].min()),float(query[:,1].min()),float(query[:,0].max()),float(query[:,1].max())],
                     original_split=plan['original_split'],split=split,source='PUBLIC_YCB_MATERIAL_GT_JITTER_EMULATION',
                     source_emulation=True,actual_initializer_query=False,native_mapping='UNKNOWN',training_admission=False))
  source=dict(scene_id=g['scene_id'],frame_id=g['frame_id'],instance_id=g['instance_id'],object_id=g['object_id'],
              image=g['image']['path'],image_sha256=g['image']['sha256'],depth=g['depth']['path'],
              mask_full=g['own_full_mask']['path'],mask_visib=g['own_visible_mask']['path'],
              height=hw[0],width=hw[1],K=cam['cam_K'],cam_R_m2c=target['cam_R_m2c'],cam_t_m2c=target['cam_t_m2c'],
              depth_scale=cam['depth_scale'],split=plan['original_split'])
  references.append(dict(input_id=plan['input_id'],instance_row_index=i,plan_row_index=i,
                         original_source_row_index=g['original_source_row_index'],source_instance=source,
                         aux_surface_ids=[c['aux_surface_id']for c in obj['candidates']],
                         fixed_xyz_model_mm=xyz.tolist(),reference_xy=reference.tolist(),reference_camera_xyz_mm=cam_xyz.tolist(),
                         original_mesh_vertex_indices=g['original_mesh_vertex_indices'],
                         incident_triangle_indices=[c['incident_triangle_indices']for c in obj['candidates']],model=obj['model'],
                         complete_scene_GT_owner_count=len(gt),teacher_state='PENDING_NOT_GENERATED',source_labels_not_model_inputs=True,
                         source_VAL_is_development_exposed=(plan['original_split']=='val'),human_gold=False,native_mapping='UNKNOWN'))
 proof=H/'GENERATED_POLICY_TESTS.json';assert read(proof)['PASS']and read(proof)['code']==bind(__file__)
 PREP.mkdir();jsonl(PREP/'INPUTS400_QUERY_EMULATION_ONLY.jsonl',inputs);jsonl(PREP/'REFERENCE400_GT_ONLY.jsonl',references)
 protocol=dict(completed=True,root_policy=bind(POLICY),source_policy_reused_exact=policy['source_policy_reused_exact'],
               centre_depth_requirement=policy['centre_policy_reused_exact'],exact_V109_functions=bind(OLD),code=bind(__file__),generated=bind(proof),
               source_input_plan=policy['input_plan'],source_GT_lineage=policy['GT_lineage'],cache_bindings=policy['cache_bindings'],
               source_scene9_exposure=policy['scene9_exposure'],geometry=policy['geometry_result'],ray_kernel=bind(KERNEL),native_grid_source=bind(NATIVE_GRID),
               source_control_metadata=list(controls.values()),inputs=bind(PREP/'INPUTS400_QUERY_EMULATION_ONLY.jsonl'),reference_GT_only=bind(PREP/'REFERENCE400_GT_ONLY.jsonl'),
               instances=400,slots=3200,seed=213,instances_by_original_split=dict(train=300,val=100),
               RNG='PCG64; each8x2 draw:angle2pi*col0,radius4sqrt(col1)',
               query_freeze_before_new_depth_mask_decode=True,RGB_pixel_reads=0,PNG_header_reads=0,
               native_HW='Unchanged source input-plan raw_hw metadata; no PNG/IDAT read is needed.',
               fixed_classification='Exact V109 classify/geometry_state/compact_ray and V106 ray_hits; unchanged thresholds and centre-pixel ray.',
               output=str(OUT),original_folds_unchanged=True,source_VAL_is_development_exposed=True,source_VAL_is_fresh_holdout=False,
               training_admission=False,human_gold=False,native_mapping='UNKNOWN',physical_junction_certified=False,model_fit_private_PnP=0)
 write(PREP/'PRODUCER_PROTOCOL.json',protocol)
 write(PREP/'ROOT_BUILD_ENTRY_TEMPLATE.json',dict(approved_actual_source_depth_mask_build=False,current_allowed=False,RGB_allowed=False,model_or_fit_allowed=False,PnP_allowed=False,
       protocol=bind(PREP/'PRODUCER_PROTOCOL.json'),code=bind(__file__),output=str(OUT)))
 print(json.dumps(dict(completed=True,protocol=bind(PREP/'PRODUCER_PROTOCOL.json'),entry_template=bind(PREP/'ROOT_BUILD_ENTRY_TEMPLATE.json'),instances=400,slots=3200,
       command=f'{sys.executable} {Path(__file__).resolve()} --entry EXACT_ROOT_ENTRY.json')),flush=True)

def run(entry_path):
 e=read(entry_path)
 assert e['approved_actual_source_depth_mask_build']is True and all(e[k]is False for k in('current_allowed','RGB_allowed','model_or_fit_allowed','PnP_allowed'))
 p=read(check(e['protocol']));assert check(e['code'])==Path(__file__).resolve()and p['code']==e['code']
 assert sha(POLICY)==POLICY_SHA and sha(CENTRE)==CENTRE_SHA and not OUT.exists()and Path(e['output']).resolve()==OUT.resolve()
 for key in('root_policy','source_policy_reused_exact','centre_depth_requirement','exact_V109_functions','generated','source_input_plan','source_GT_lineage','cache_bindings',
             'source_scene9_exposure','geometry','ray_kernel','native_grid_source','inputs','reference_GT_only'):
  check(p[key])
 for b in p['source_control_metadata']:check(b)
 assert read(p['generated']['path'])['PASS']and p['instances']==400 and p['slots']==3200
 inputrows=rows(p['inputs']['path']);gtrows=rows(p['reference_GT_only']['path']);assert len(inputrows)==len(gtrows)==400
 assert collections.Counter(x['original_split']for x in inputrows)==dict(train=300,val=100)
 assert all(x['split']==('TRAIN'if x['original_split']=='train'else'VALIDATION')for x in inputrows)
 OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry_path),no_silent_restart=True));started=time.monotonic()
 _run_depth_mask_loop(e,p,inputrows,gtrows,entry_path,started)

def _run_depth_mask_loop(e,p,inputrows,gtrows,entry_path,started):
 try:
  kernel=module(p['ray_kernel']['path'],'y113_exact_pilot_ray');models={};prior={x['path']:x for x in read(p['cache_bindings']['path'])['files']};files={};frameinfo=[];current_key=None;current=None
  shape=(400,8);state=np.full(shape,'U',dtype='U1');reason=np.full(shape,'PENDING',dtype='U96');geom=np.full(shape,'UNKNOWN',dtype='U16');stable=np.zeros(shape,bool)
  first=np.full((*shape,2),np.nan);firstZ=first.copy();gap=first.copy();centerfirst=np.full((*shape,2),np.nan);raw=np.zeros((*shape,9),np.uint16);inside=np.zeros((*shape,9),bool);full=np.zeros_like(inside);ownvis=np.zeros_like(inside);ownercount=np.zeros((*shape,9),np.uint8);bits=np.zeros((*shape,9),np.uint64);counts=np.zeros((*shape,len(COUNT_KEYS)),np.uint8)
  qcont=np.zeros(shape,bool);qnearest=np.zeros(shape,bool);rcont=np.zeros(shape,bool);rnearpixel=np.zeros(shape,bool);frameidx=np.zeros(400,np.int32);scales=np.zeros(400,np.float64)
  with (OUT/'SLOT_EVIDENCE3200_GT_ONLY.jsonl').open('x')as log:
   for i,(inp,g)in enumerate(zip(inputrows,gtrows)):
    assert inp['input_id']==g['input_id']and i==inp['instance_row_index']==g['instance_row_index'];r=g['source_instance'];scene=Path(r['depth']).parent.parent;key=(str(scene),r['frame_id'])
    if key!=current_key:
     sceneGT=read(scene/'scene_gt.json')[str(r['frame_id'])];cam=read(scene/'scene_camera.json')[str(r['frame_id'])];assert len(sceneGT)<=64
     depth=cv2.imread(r['depth'],cv2.IMREAD_UNCHANGED);assert depth is not None and depth.dtype==np.uint16 and depth.shape==tuple(inp['raw_hw'])
     depthbind=bind(Path(r['depth']));assert depthbind['sha256']==prior[r['depth']]['sha256'];files[r['depth']]=depthbind;owners=[];ownerfiles=[]
     for j in range(len(sceneGT)):
      path=scene/'mask_visib'/f'{r["frame_id"]:06d}_{j:06d}.png';b=bind(path);assert b['sha256']==prior[str(path)]['sha256'];files[str(path)]=b;ownerfiles.append(b)
      mask=cv2.imread(str(path),cv2.IMREAD_UNCHANGED);assert mask is not None and mask.shape==depth.shape and set(np.unique(mask)).issubset({0,255});owners.append(mask>0)
     for name in ('scene_gt.json','scene_camera.json'):files[str(scene/name)]=bind(scene/name)
     current=dict(gt=sceneGT,cam=cam,depth=depth,owners=np.stack(owners),index=len(frameinfo));frameinfo.append(dict(frame_index=len(frameinfo),scene_id=r['scene_id'],frame_id=r['frame_id'],depth=depthbind,visible_masks=ownerfiles,instance_object_ids=[x['obj_id']for x in sceneGT],complete_scene_GT_owner_count=len(sceneGT)));current_key=key
    frameidx[i]=current['index'];own=r['instance_id'];assert current['gt'][own]['obj_id']==r['object_id']and current['gt'][own]['cam_R_m2c']==r['cam_R_m2c']and current['gt'][own]['cam_t_m2c']==r['cam_t_m2c'];assert current['cam']['cam_K']==r['K']and current['cam']['depth_scale']==r['depth_scale']
    allmask=cv2.imread(r['mask_full'],cv2.IMREAD_UNCHANGED);assert allmask is not None and allmask.shape==current['depth'].shape and set(np.unique(allmask)).issubset({0,255});files[r['mask_full']]=bind(Path(r['mask_full']));assert files[r['mask_full']]['sha256']==prior[r['mask_full']]['sha256']
    oid=r['object_id']
    if oid not in models:
     mesh=trimesh.load(check(g['model']),process=False,skip_materials=True);models[oid]=(np.asarray(mesh.vertices,np.float64),np.asarray(mesh.faces,np.int64))
    xyz,tri=models[oid];R=np.array(r['cam_R_m2c']).reshape(3,3);t=np.array(r['cam_t_m2c']);K=np.array(r['K']).reshape(3,3);invK=np.linalg.inv(K);vertices=xyz@R.T+t;query=np.array(inp['initial_points']);ref=np.array(g['reference_xy']);Hh,W=inp['raw_hw'];scales[i]=r['depth_scale']
    for k in range(8):
     target=R@np.array(g['fixed_xyz_model_mm'][k])+t;assert np.array_equal(xyz[g['original_mesh_vertex_indices'][k]],g['fixed_xyz_model_mm'][k]);projected=K@target;assert np.max(abs(projected[:2]/projected[2]-ref[k]))<1e-9
     rr=float(np.linalg.norm(target));ray=compact_ray(kernel.ray_hits(vertices,tri,target,rr,set(g['incident_triangle_indices'][k])));
     for z,a in enumerate(ray):
      if not a['no_hit']:first[i,k,z]=a['first_range_mm'];firstZ[i,k,z]=a['first_Z_mm'];gap[i,k,z]=rr-a['first_range_mm']
     gs,st,_=geometry_state(first[i,k],rr);geom[i,k]=gs;stable[i,k]=st
     qr=np.floor(query[k]+.5);rc=np.floor(ref[k]+.5).astype(int);qcont[i,k]=0<=query[k,0]<W and 0<=query[k,1]<Hh;qnearest[i,k]=0<=qr[0]<W and 0<=qr[1]<Hh;rcont[i,k]=0<=ref[k,0]<W and 0<=ref[k,1]<Hh;rnearpixel[i,k]=0<=rc[0]<W and 0<=rc[1]<Hh
     pixelray=None
     if rnearpixel[i,k]:
      pixelray=compact_ray(kernel.ray_hits(vertices,tri,invK@np.r_[rc,1.],rr,set(g['incident_triangle_indices'][k])))
      for z,a in enumerate(pixelray):
       if not a['no_hit']:centerfirst[i,k,z]=a['first_Z_mm']
     for j,(dy,dx)in enumerate((dy,dx)for dy in(-1,0,1)for dx in(-1,0,1)):
      x,y=(rc+[dx,dy]).tolist();ok=0<=x<W and 0<=y<Hh;inside[i,k,j]=ok
      if ok:
       raw[i,k,j]=current['depth'][y,x];full[i,k,j]=allmask[y,x]>0;ownvis[i,k,j]=current['owners'][own,y,x];ids=np.flatnonzero(current['owners'][:,y,x]);ownercount[i,k,j]=len(ids);bits[i,k,j]=sum(1<<int(z)for z in ids)
     c=inside[i,k];positive=c&(raw[i,k]>0);uniqueown=c&(ownercount[i,k]==1)&ownvis[i,k];otheronly=c&(ownercount[i,k]==1)&~ownvis[i,k]
     counts[i,k]=[c.sum(),positive.sum(),uniqueown.sum(),otheronly.sum(),((ownercount[i,k]>1)&c).sum(),((ownercount[i,k]==0)&c).sum(),full[i,k].sum()]
     label,why=classify(gs,qcont[i,k],rcont[i,k],inside[i,k],raw[i,k],ownercount[i,k],ownvis[i,k],bits[i,k],full[i,k],own,scales[i],target[2],centerfirst[i,k,0]);state[i,k]=label;reason[i,k]=why
     log.write(json.dumps(dict(slot_index=i*8+k,input_id=inp['input_id'],logical_aux_slot=k,aux_surface_id=g['aux_surface_ids'][k],object_id=oid,source_scene_id=r['scene_id'],source_frame_id=r['frame_id'],source_instance_id=own,raw_surface_teacher_state=label,reason=why,geometry_state=gs,geometry_stable=bool(st),reference_cameraZ_mm=float(target[2]),fixed_ray_firsthits=ray,centre_integer_ray_firsthits=pixelray,counts=dict(zip(COUNT_KEYS,map(int,counts[i,k]))),no_training_admission=True),allow_nan=False)+'\n')
    if(i+1)%100==0:print(json.dumps(dict(instances=i+1,total=400,seconds=time.monotonic()-started)),flush=True)
  reference=np.array([g['reference_xy']for g in gtrows]);query=np.array([r['initial_points']for r in inputrows]);side=np.array([r['roi_side']for r in inputrows]);external=np.where(state=='D',0,np.where(state=='E',1,-1)).astype(np.int8)
  np.savez_compressed(OUT/'RAW_SURFACE_TEACHER3200_GT_ONLY.npz',raw_surface_teacher_state=state,external_target_D0_E1_else_minus1=external,reason=reason,geometry_state=geom,geometry_stable=stable,geometry_first_hit_range_mm=first,geometry_first_hit_cameraZ_mm=firstZ,geometry_point_minus_first_range_mm=gap,centre_integer_ray_first_cameraZ_mm=centerfirst,neighbour_depth_raw_uint16=raw,neighbour_inbounds=inside,neighbour_own_full=full,neighbour_own_visible=ownvis,neighbour_visible_owner_count=ownercount,neighbour_owner_bits_uint64=bits,evidence_counts=counts,reference_xy_GT_only=reference,query_xy_input_copy=query,roi_side_input_copy=side,query_reference_error_px=np.linalg.norm(query-reference,axis=2),query_continuous_valid=qcont,query_nearest_valid=qnearest,reference_continuous_valid=rcont,reference_nearest_valid=rnearpixel,depth_scale_to_mm=scales,frame_index=frameidx,instance_row_index=np.arange(400),original_split=np.array([x['original_split']for x in inputrows]),split=np.array([x['split']for x in inputrows]),input_id=np.array([x['input_id']for x in inputrows]),logical_aux_slot=np.broadcast_to(np.arange(8),(400,8)))
  write(OUT/'SOURCE_DEPTH_MASK_BINDINGS.json',dict(files=list(files.values()),frames=frameinfo,all_visible_owner_sets_from_complete_scene_GT=True));statecounts=dict(collections.Counter(state.ravel().tolist()));reasons=dict(collections.Counter(reason.ravel().tolist()))
  write(OUT/'SUMMARY.json',dict(completed=True,all_instances=400,all_slots=3200,instances_by_original_split=dict(train=300,val=100),source_VAL_is_development_exposed=True,source_VAL_is_fresh_holdout=False,states=statecounts,reasons=reasons,by_object={str(oid):dict(collections.Counter(state[[g['source_instance']['object_id']==oid for g in gtrows]].ravel().tolist()))for oid in IDS},by_original_split={split:dict(collections.Counter(state[[g['source_instance']['split']==split for g in gtrows]].ravel().tolist()))for split in('train','val')},source_emulated_queries=True,native_mapping='UNKNOWN',human_gold=False,physical_junction_certified=False,training_admission=False,known_geometric_S_retained_outside_external_loss=True,count_keys=COUNT_KEYS,neighbour_order='dy outer/dx inner[-1,0,1],centre4',owner_bit_mapping='Bit i=scene_gt instance index; complete per-frame object list in bindings.',kernel_bary_columns=['64FP64eps','1e-10'],nonfinite_float_fields='NaN means no ray hit, not a label or imputed background.',seconds=time.monotonic()-started,RGB_pixels_model_fit_current_PnP=0))
  outputs=[bind(OUT/f)for f in ('RAW_SURFACE_TEACHER3200_GT_ONLY.npz','SLOT_EVIDENCE3200_GT_ONLY.jsonl','SOURCE_DEPTH_MASK_BINDINGS.json','SUMMARY.json')]
  write(OUT/'CORPUS_SEAL.json',dict(completed=True,entry=bind(entry_path),protocol=e['protocol'],outputs=outputs,inputs=p['inputs'],reference_GT_only=p['reference_GT_only'],code=e['code'],training_admission=False,original_folds_unchanged=True,source_VAL_is_development_exposed=True,all3200_preserved=True,RGB_model_fit_current_PnP=0));print(json.dumps(dict(completed=True,seal=bind(OUT/'CORPUS_SEAL.json'),states=statecounts,seconds=time.monotonic()-started)),flush=True)
 except BaseException:
  write(OUT/'FAILED_EXECUTION.json',dict(completed=False,error=traceback.format_exc(),preserve_no_automatic_restart=True));raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
 if args.prepare:prepare()
 elif args.entry:run(args.entry)
 else:raise SystemExit('Use metadata-only --prepare or root-approved --entry.')
