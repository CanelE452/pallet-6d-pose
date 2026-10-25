"""One root-entered public surface teacher build; no RGB/model/fit/current/PnP."""
from pathlib import Path
import argparse, ast, collections, hashlib, importlib.util, json, math, sys, time, traceback
sys.dont_write_bytecode=True
import cv2
import numpy as np
import trimesh
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_SOURCE_POLICY_V2_CORRECTED_INVENTORY_BEFORE_BULK.json'
POLICY_SHA='2817c6233634bc0238b211dc6ceca4efe9d9d0f7c24b32e5c90929b4a5c0e8fd'
CENTRE=V/'ROOT_CENTRE_DEPTH_REQUIREMENT_BEFORE_BULK.json'
CENTRE_SHA='a1434dc18fcb4d46606eebdfc839fe297d7835294a3a6744090acbe3937b95d7'
INSTANCES=P/'query_visibility_public_real_20261022_v63/ycb/instances.jsonl'
NATIVE_GRID=P/'query_visibility_public_real_20261022_v63/model/deployment_assets/head_and_roi.py'
KERNEL=P/'query_visibility_ycb_surface_cues_20261023_v106/numeric_pilot32/compute_fixed32_evidence.py'
PREP=V/'query_preparation';OUT=V/'corpus_build_v1'
IDS=(2,3,7,8);STATES=('D','E','S','U')
COUNT_KEYS=('in_bounds','positive_depth','unique_own','unique_other_only','overlap','unassigned','own_full')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def jsonl(p,rows):
 with Path(p).open('x')as f:
  for x in rows:f.write(json.dumps(x,allow_nan=False)+'\n')
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()]
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def native_grid_function():
 node=next(n for n in ast.parse(NATIVE_GRID.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='native_grid')
 namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(NATIVE_GRID),'exec'),namespace)
 return namespace['native_grid']
def jitter_q(r,rng):
 draw=rng.random((8,2));theta=2*np.pi*draw[:,0];radius=4*np.sqrt(draw[:,1])
 return r+np.stack((radius*np.cos(theta),radius*np.sin(theta)),axis=1)
def geometry_state(hit_range,point_range):
 a,b=hit_range
 if not np.isfinite([a,b]).all()or abs(a-b)>.001:return 'UNKNOWN',False,np.nan
 gap=float(point_range-a)
 if gap>2:return 'SELF',True,gap
 if abs(gap)<=2:return 'SELF_VISIBLE',True,gap
 return 'UNKNOWN',True,gap
def classify(gstate,q_cont,r_cont,cell_in,raw_depth,owner_count,own_visible,owner_bits,own_full,own_index,scale,reference_Z,centre_first_Z):
 # All criteria are fixed source rules; source state S is distinct from unknown U.
 if not q_cont:return 'U','QUERY_CONTINUOUS_OOB'
 if not r_cont:return 'U','REFERENCE_CONTINUOUS_OOB'
 if not cell_in[4]:return 'U','REFERENCE_NEAREST_PIXEL_OOB'
 if gstate=='SELF':return 'S','KNOWN_TARGET_SELF_GEOMETRY'
 if gstate!='SELF_VISIBLE':return 'U','GEOMETRY_UNKNOWN'
 positive=cell_in&(raw_depth>0);mm=raw_depth.astype(np.float64)*scale
 unique_own=cell_in&(owner_count==1)&own_visible
 if unique_own[4]and positive[4]and np.isfinite(centre_first_Z):
  if abs(mm[4]-centre_first_Z)<=15 and unique_own.sum()>=5 and positive.sum()>=5:return 'D','SELF_VISIBLE_UNIQUE_OWN_DEPTH_SUPPORTED'
 # Explicit root clarification: centre depth must itself be positive and nearer.
 same_foreign=cell_in.all()and (owner_count==1).all()and not own_visible.any()and len(set(int(x)for x in owner_bits))==1
 if same_foreign and positive[4]and mm[4]<reference_Z-15 and positive.sum()>=7 and (mm[positive]<reference_Z-15).all()and own_full.any():return 'E','SELF_VISIBLE_SAME_FOREIGN9_NEARER_DEPTH7_WITH_AMODAL_SUPPORT'
 if (owner_count>1).any():return 'U','VISIBLE_OWNER_OVERLAP'
 if not positive[4]:return 'U','CENTRE_DEPTH_MISSING'
 if unique_own[4]and not np.isfinite(centre_first_Z):return 'U','CENTRE_TARGET_PIXEL_RAY_NOHIT'
 if unique_own[4]and abs(mm[4]-centre_first_Z)>15:return 'U','CENTRE_OWN_DEPTH_DISAGREEMENT'
 if same_foreign and not own_full.any():return 'U','FOREIGN_WITHOUT_AMODAL_SUPPORT'
 if same_foreign:return 'U','FOREIGN_DEPTH_NOT_STRICTLY_NEARER_SUPPORTED'
 if owner_count[4]==0:return 'U','CENTRE_VISIBLE_OWNER_UNASSIGNED'
 return 'U','INSUFFICIENT_FIXED_3X3_SUPPORT'
def compact_ray(result):
 return [dict(no_hit=x['no_hit'],first_range_mm=x['first_hit_range_mm'],first_Z_mm=x['first_hit_camera_Z_mm'],
  first_triangle=x.get('first_triangle_index'),first_tie_count=len(x.get('bounded_first_hit_triangle_indices',[])),
  first_hit_incident_expected_vertex=x.get('first_hit_is_incident_to_expected_vertex'))for x in result['sensitivity'][1:3]]
def prepare():
 assert sha(POLICY)==POLICY_SHA and sha(CENTRE)==CENTRE_SHA and not PREP.exists()
 policy=read(POLICY);assert bind(INSTANCES)==policy['count_correction']['immutable_instance_source'];geometry=read(check(policy['source_geometry']));check(policy['pilot_numeric'])
 allrows=sorted([r for r in rows(INSTANCES)if r['object_id']in IDS],key=lambda r:(r['scene_id'],r['frame_id'],r['instance_id']))
 assert len(allrows)==1000 and all(r['split']=='train'for r in allrows)
 objects={o['object_id']:o for o in geometry['objects']};native=native_grid_function();rng=np.random.Generator(np.random.PCG64(207));inputs=[];gt=[]
 for i,r in enumerate(allrows):
  obj=objects[r['object_id']];xyz=np.array([c['xyz_mm']for c in obj['candidates']],np.float64);R=np.array(r['cam_R_m2c']).reshape(3,3);t=np.array(r['cam_t_m2c']);K=np.array(r['K']).reshape(3,3);camera=xyz@R.T+t;reference=(camera@K.T)[:,:2]/camera[:,2,None]
  assert np.isfinite(reference).all()and (camera[:,2]>0).all()
  query=jitter_q(reference,rng);grid,valid,L=native(query,[r['height'],r['width']]);assert np.linalg.norm(query-reference,axis=1).max()<=4
  identity=f'Y109_SOURCE_{i:06d}'
  inputs.append(dict(input_id=identity,instance_row_index=i,image=dict(path=r['image'],sha256=r['image_sha256']),raw_hw=[r['height'],r['width']],initial_points=query.tolist(),roi_side=L,query_box_xyxy=[float(query[:,0].min()),float(query[:,1].min()),float(query[:,0].max()),float(query[:,1].max())],split='TRAIN',source='PUBLIC_YCB_GT_JITTER_EMULATION',source_emulation=True,native_mapping='UNKNOWN'))
  gt.append(dict(input_id=identity,instance_row_index=i,source_instance=r,aux_surface_ids=[c['aux_surface_id']for c in obj['candidates']],fixed_xyz_model_mm=xyz.tolist(),reference_xy=reference.tolist(),reference_camera_xyz_mm=camera.tolist(),original_mesh_vertex_indices=[c['original_mesh_vertex_index']for c in obj['candidates']],incident_triangle_indices=[c['incident_triangle_indices']for c in obj['candidates']],model=obj['model'],teacher_state='PENDING_U_ALL8',source_labels_not_model_inputs=True))
 proof=H/'GENERATED_POLICY_TESTS.json';assert read(proof)['PASS']and read(proof)['code']==bind(__file__)
 PREP.mkdir();jsonl(PREP/'INPUTS1000_QUERY_EMULATION_ONLY.jsonl',inputs);jsonl(PREP/'REFERENCE1000_GT_ONLY.jsonl',gt)
 protocol=dict(completed=True,root_policy=bind(POLICY),centre_depth_requirement=bind(CENTRE),code=bind(__file__),generated=bind(proof),geometry=policy['source_geometry'],numeric_pilot=policy['pilot_numeric'],ray_kernel=bind(KERNEL),source_instances=bind(INSTANCES),native_grid_source=bind(NATIVE_GRID),inputs=bind(PREP/'INPUTS1000_QUERY_EMULATION_ONLY.jsonl'),reference_GT_only=bind(PREP/'REFERENCE1000_GT_ONLY.jsonl'),instances=1000,slots=8000,seed=207,RNG='PCG64; each8x2 draw:angle2pi*col0,radius4sqrt(col1)',geometry_comparison='64eps primary and1e-10; finite first-range agreement<=.001mm; primary signed gap used at fixed2mm boundary, both saved.',centre_target_pixel_ray='Literal D depth comparison uses finite64eps first-hit cameraZ at nearest integer r;1e-10 sensitivity also saved, no extra unannounced stability gate.',factual_offimage='q/r continuous bounds checked separately; nearest referencepixel OOB ->U because centre evidence unavailable. Nearest q validity stored separately.',E_root_clarification='Centre observed depth must be positive and<fixedreferenceZ-15mm, in addition to all fixed E policy conditions; prospective root instruction received before bulk pixels.',output=str(OUT),training_admission=False,RGB_or_depth_mask_pixels_read_during_prepare=0,model_fit_private_PnP=0)
 write(PREP/'PRODUCER_PROTOCOL.json',protocol);write(PREP/'ROOT_BUILD_ENTRY_TEMPLATE.json',dict(approved_actual_source_depth_mask_build=False,current_allowed=False,RGB_allowed=False,model_or_fit_allowed=False,PnP_allowed=False,protocol=bind(PREP/'PRODUCER_PROTOCOL.json'),code=bind(__file__),output=str(OUT)))
 print(json.dumps(dict(completed=True,protocol=bind(PREP/'PRODUCER_PROTOCOL.json'),entry_template=bind(PREP/'ROOT_BUILD_ENTRY_TEMPLATE.json'),command=f'{sys.executable} {Path(__file__).resolve()} --entry EXACT_ROOT_ENTRY.json')),flush=True)
def run(entry_path):
 e=read(entry_path);assert e['approved_actual_source_depth_mask_build']is True and all(e[k]is False for k in ('current_allowed','RGB_allowed','model_or_fit_allowed','PnP_allowed'))
 p=read(check(e['protocol']));assert check(e['code'])==Path(__file__).resolve()and p['code']==e['code']and sha(POLICY)==POLICY_SHA and not OUT.exists()and Path(e['output']).resolve()==OUT.resolve()
 assert sha(CENTRE)==CENTRE_SHA
 for k in ('root_policy','centre_depth_requirement','generated','geometry','numeric_pilot','ray_kernel','source_instances','native_grid_source','inputs','reference_GT_only'):check(p[k])
 assert read(p['generated']['path'])['PASS'];inputrows=rows(p['inputs']['path']);gtrows=rows(p['reference_GT_only']['path']);assert len(inputrows)==len(gtrows)==1000
 OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry_path),no_silent_restart=True));started=time.monotonic()
 try:
  kernel=module(p['ray_kernel']['path'],'y109_frozen_pilot_ray');models={};previous=read(p['geometry']['path'])['prior_depth_owner_byte_bindings'];prior={x['path']:x for x in read(check(previous))['files']};files={};frameinfo=[];current_key=None;current=None
  shape=(1000,8);state=np.full(shape,'U',dtype='U1');reason=np.full(shape,'PENDING',dtype='U96');geom=np.full(shape,'UNKNOWN',dtype='U16');stable=np.zeros(shape,bool)
  first=np.full((*shape,2),np.nan);firstZ=first.copy();gap=first.copy();centerfirst=np.full((*shape,2),np.nan);raw=np.zeros((*shape,9),np.uint16);inside=np.zeros((*shape,9),bool);full=np.zeros_like(inside);ownvis=np.zeros_like(inside);ownercount=np.zeros((*shape,9),np.uint8);bits=np.zeros((*shape,9),np.uint64);counts=np.zeros((*shape,len(COUNT_KEYS)),np.uint8)
  qcont=np.zeros(shape,bool);qnearest=np.zeros(shape,bool);rcont=np.zeros(shape,bool);rnearpixel=np.zeros(shape,bool);frameidx=np.zeros(1000,np.int32);scales=np.zeros(1000,np.float64)
  with (OUT/'SLOT_EVIDENCE8000_GT_ONLY.jsonl').open('x')as log:
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
    allmask=cv2.imread(r['mask_full'],cv2.IMREAD_UNCHANGED);assert allmask is not None and allmask.shape==current['depth'].shape and set(np.unique(allmask)).issubset({0,255});files[r['mask_full']]=bind(Path(r['mask_full']))
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
    if(i+1)%100==0:print(json.dumps(dict(instances=i+1,total=1000,seconds=time.monotonic()-started)),flush=True)
  reference=np.array([g['reference_xy']for g in gtrows]);query=np.array([r['initial_points']for r in inputrows]);side=np.array([r['roi_side']for r in inputrows]);external=np.where(state=='D',0,np.where(state=='E',1,-1)).astype(np.int8)
  np.savez_compressed(OUT/'RAW_SURFACE_TEACHER8000_GT_ONLY.npz',raw_surface_teacher_state=state,external_target_D0_E1_else_minus1=external,reason=reason,geometry_state=geom,geometry_stable=stable,geometry_first_hit_range_mm=first,geometry_first_hit_cameraZ_mm=firstZ,geometry_point_minus_first_range_mm=gap,centre_integer_ray_first_cameraZ_mm=centerfirst,neighbour_depth_raw_uint16=raw,neighbour_inbounds=inside,neighbour_own_full=full,neighbour_own_visible=ownvis,neighbour_visible_owner_count=ownercount,neighbour_owner_bits_uint64=bits,evidence_counts=counts,reference_xy_GT_only=reference,query_xy_input_copy=query,roi_side_input_copy=side,query_reference_error_px=np.linalg.norm(query-reference,axis=2),query_continuous_valid=qcont,query_nearest_valid=qnearest,reference_continuous_valid=rcont,reference_nearest_valid=rnearpixel,depth_scale_to_mm=scales,frame_index=frameidx,instance_row_index=np.arange(1000),logical_aux_slot=np.broadcast_to(np.arange(8),(1000,8)))
  write(OUT/'SOURCE_DEPTH_MASK_BINDINGS.json',dict(files=list(files.values()),frames=frameinfo,all_visible_owner_sets_from_complete_scene_GT=True));statecounts=dict(collections.Counter(state.ravel().tolist()));reasons=dict(collections.Counter(reason.ravel().tolist()))
  write(OUT/'SUMMARY.json',dict(completed=True,all_instances=1000,all_slots=8000,all_existing_TRAIN=True,VAL_instances=0,states=statecounts,reasons=reasons,by_object={str(oid):dict(collections.Counter(state[[g['source_instance']['object_id']==oid for g in gtrows]].ravel().tolist()))for oid in IDS},source_emulated_queries=True,native_mapping='UNKNOWN',human_gold=False,physical_junction_certified=False,training_admission=False,known_geometric_S_retained_outside_external_loss=True,count_keys=COUNT_KEYS,neighbour_order='dy outer/dx inner[-1,0,1],centre4',owner_bit_mapping='Bit i=scene_gt instance index; complete per-frame object list in bindings.',kernel_bary_columns=['64FP64eps','1e-10'],nonfinite_float_fields='NaN means no ray hit, not a label or imputed background.',seconds=time.monotonic()-started,RGB_pixels_model_fit_current_PnP=0))
  outputs=[bind(OUT/f)for f in ('RAW_SURFACE_TEACHER8000_GT_ONLY.npz','SLOT_EVIDENCE8000_GT_ONLY.jsonl','SOURCE_DEPTH_MASK_BINDINGS.json','SUMMARY.json')]
  write(OUT/'CORPUS_SEAL.json',dict(completed=True,entry=bind(entry_path),protocol=e['protocol'],outputs=outputs,inputs=p['inputs'],reference_GT_only=p['reference_GT_only'],code=e['code'],training_admission=False,all8000_preserved=True,RGB_model_fit_current_PnP=0));print(json.dumps(dict(completed=True,seal=bind(OUT/'CORPUS_SEAL.json'),states=statecounts,seconds=time.monotonic()-started)),flush=True)
 except BaseException:
  write(OUT/'FAILED_EXECUTION.json',dict(completed=False,error=traceback.format_exc(),preserve_no_automatic_restart=True));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
 if args.prepare:prepare()
 elif args.entry:run(args.entry)
 else:raise SystemExit('Use metadata-only --prepare or root-approved --entry.')
