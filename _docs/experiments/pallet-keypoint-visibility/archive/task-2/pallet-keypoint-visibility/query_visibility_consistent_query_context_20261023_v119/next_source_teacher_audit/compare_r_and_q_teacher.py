"""All3200 public source pixel-sampling diagnosis, no RGB/ML/fit/admission."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import collections,hashlib,importlib.util,json,time,traceback
import cv2
import numpy as np
import trimesh
H=Path(__file__).resolve().parent;POLICY=H/'POLICY_BEFORE_PAYLOADS.json'
POLICY_SHA='9424e523360f78e218f1e91dd7643a12ed0126602366c76851a0eaaeb5159d5d'
STATES=('D','E','S','U')
def bind(p):p=Path(p);return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(n,x):
 with(H/n).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def check(b):assert bind(b['path'])['sha256']==b['sha256'],b['path']
def mod(b,name):
 check(b);s=importlib.util.spec_from_file_location(name,b['path']);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def distance_bin(x):return 'LE1'if x<=1 else'GT1_LE2'if x<=2 else'GT2_LE3'if x<=3 else'GT3_LE4'if x<=4 else'GT4'

def generated(classify):
 inside=np.ones(9,bool);own=np.ones(9,bool);full=own.copy();depth=np.full(9,1000,np.uint16);counts=np.ones(9,np.uint8);bits=np.ones(9,np.uint64)
 args=['SELF_VISIBLE',True,True,inside,depth,counts,own,bits,full,0,1.,1000.,1000.]
 assert classify(*args)[0]=='D'
 e=list(args);e[4]=np.full(9,900,np.uint16);e[6]=np.zeros(9,bool);e[7]=np.full(9,2,np.uint64);assert classify(*e)[0]=='E'
 z=list(e);z[4]=np.zeros(9,np.uint16);assert classify(*z)[0]=='U'
 s=list(z);s[0]='SELF';assert classify(*s)[0]=='S'
 overlap=list(e);overlap[5]=np.full(9,2,np.uint8);assert classify(*overlap)[0]=='U'
 boundary=list(e);boundary[4]=np.full(9,985,np.uint16);assert classify(*boundary)[0]=='U'
 return dict(PASS=True,tests=['own_depth_supported_D','same_foreign_nearer_E','missing_depth_U','known_self_before_missing_depth_S','owner_overlap_U','strict15mm_E_tie_U'],source_payloads_read=0)

def run():
 assert bind(POLICY)['sha256']==POLICY_SHA and not(H/'STARTED.json').exists()
 p=read(POLICY);b=p['bindings'];old=mod(b['classify_code'],'diagnostic_exact_classify');kernel=mod(b['ray_kernel'],'diagnostic_exact_ray')
 proof=generated(old.classify);proof['code']=bind(__file__);write('GENERATED_CHECKS.json',proof)
 for val in b.values():check(val)
 inp=rows(b['input400']['path']);ref=rows(b['reference400']['path']);assert len(inp)==len(ref)==400
 with np.load(b['source_packet']['path'],allow_pickle=False)as z:
  stored={k:np.array(z[k],copy=True)for k in ('raw_surface_teacher_state','reason','geometry_state','geometry_stable','neighbour_depth_raw_uint16','neighbour_inbounds','neighbour_own_full','neighbour_own_visible','neighbour_visible_owner_count','neighbour_owner_bits_uint64','centre_integer_ray_first_cameraZ_mm','query_continuous_valid','reference_continuous_valid','depth_scale_to_mm','reference_xy_GT_only','query_xy_input_copy','roi_side_input_copy')}
 original=stored['raw_surface_teacher_state'];assert original.shape==(400,8)
 replay=np.full((400,8),'U',dtype='U1')
 for i,(a,g)in enumerate(zip(inp,ref)):
  assert a['input_id']==g['input_id']and a['instance_row_index']==g['instance_row_index']==i
  assert np.array_equal(stored['query_xy_input_copy'][i],a['initial_points'])and np.array_equal(stored['reference_xy_GT_only'][i],g['reference_xy'])and stored['roi_side_input_copy'][i]==a['roi_side']
  for k in range(8):
   label,reason=old.classify(stored['geometry_state'][i,k],bool(stored['query_continuous_valid'][i,k]),bool(stored['reference_continuous_valid'][i,k]),stored['neighbour_inbounds'][i,k],stored['neighbour_depth_raw_uint16'][i,k],stored['neighbour_visible_owner_count'][i,k],stored['neighbour_own_visible'][i,k],stored['neighbour_owner_bits_uint64'][i,k],stored['neighbour_own_full'][i,k],g['source_instance']['instance_id'],stored['depth_scale_to_mm'][i],g['reference_camera_xyz_mm'][k][2],stored['centre_integer_ray_first_cameraZ_mm'][i,k,0]);replay[i,k]=label
   assert label==original[i,k]and reason==stored['reason'][i,k],('original_replay',i,k,label,reason)
 write('ORIGINAL_R_REPLAY.json',dict(completed=True,PASS=True,labels_exact=3200,reasons_exact=3200,source_packet=b['source_packet'],code=bind(__file__),before_q_lookup=True))
 write('STARTED.json',dict(code=bind(__file__),policy=bind(POLICY),original_replay=bind(H/'ORIGINAL_R_REPLAY.json'),new_RGB_encoder_head_fit_current_PnP=0,no_automatic_retry=True));start=time.monotonic()
 prior=read(b['source_depth_mask_bindings']['path']);allowed={x['path']:x for x in prior['files']};verified=set();filefacts=[]
 def image(path,mask=False):
  path=str(path);assert path in allowed
  if path not in verified:check(allowed[path]);verified.add(path);filefacts.append(allowed[path])
  a=cv2.imread(path,cv2.IMREAD_UNCHANGED);assert a is not None
  if mask:assert a.dtype==np.uint8 and ((a==0)|(a==255)).all();return a>0
  assert a.dtype==np.uint16;return a
 shape=(400,8);qs=np.full(shape,'U',dtype='U1');reason=np.full(shape,'PENDING',dtype='U96');raw=np.zeros((*shape,9),np.uint16);inside=np.zeros((*shape,9),bool);full=np.zeros_like(inside);own=np.zeros_like(inside);count=np.zeros((*shape,9),np.uint8);bits=np.zeros((*shape,9),np.uint64);first=np.full((*shape,2),np.nan);same=np.zeros(shape,bool)
 models={};modelbindings={};framekey=None;current=None;comparison=[];qrays=0
 for i,(a,g)in enumerate(zip(inp,ref)):
  r=g['source_instance'];scene=Path(r['depth']).parent.parent;key=(str(scene),r['frame_id'])
  if key!=framekey:
   for name in ('scene_gt.json','scene_camera.json'):
    path=str(scene/name);assert path in allowed;check(allowed[path])
   gt=read(scene/'scene_gt.json')[str(r['frame_id'])];cam=read(scene/'scene_camera.json')[str(r['frame_id'])]
   dep=image(r['depth']);owners=np.stack([image(scene/'mask_visib'/f'{r["frame_id"]:06d}_{j:06d}.png',True)for j in range(len(gt))]);current=dict(gt=gt,cam=cam,depth=dep,owners=owners);framekey=key
  ownidx=r['instance_id'];assert current['gt'][ownidx]['obj_id']==r['object_id']and current['cam']['cam_K']==r['K']and current['cam']['depth_scale']==r['depth_scale']
  assert current['gt'][ownidx]['cam_R_m2c']==r['cam_R_m2c']and current['gt'][ownidx]['cam_t_m2c']==r['cam_t_m2c']
  fullmask=image(r['mask_full'],True);hh,ww=a['raw_hw'];assert current['depth'].shape==fullmask.shape==(hh,ww)and current['owners'].shape[1:]==(hh,ww)
  oid=r['object_id']
  if oid not in models:
   check(g['model']);modelbindings[oid]=g['model'];mesh=trimesh.load(g['model']['path'],process=False,skip_materials=True);models[oid]=(np.asarray(mesh.vertices,np.float64),np.asarray(mesh.faces,np.int64))
  xyz,tri=models[oid];R=np.asarray(r['cam_R_m2c']).reshape(3,3);t=np.asarray(r['cam_t_m2c']);vertices=xyz@R.T+t;invK=np.linalg.inv(np.asarray(r['K']).reshape(3,3));q=np.asarray(a['initial_points']);rr=np.asarray(g['reference_xy']);distance=np.linalg.norm(q-rr,axis=1)
  for k in range(8):
   centre=np.floor(q[k]+.5).astype(int);rc=np.floor(rr[k]+.5).astype(int);same[i,k]=np.array_equal(centre,rc)
   for j,(dy,dx)in enumerate((dy,dx)for dy in(-1,0,1)for dx in(-1,0,1)):
    x,y=centre+[dx,dy];ok=0<=x<ww and 0<=y<hh;inside[i,k,j]=ok
    if ok:
     raw[i,k,j]=current['depth'][y,x];full[i,k,j]=fullmask[y,x];own[i,k,j]=current['owners'][ownidx,y,x];ids=np.flatnonzero(current['owners'][:,y,x]);count[i,k,j]=len(ids);bits[i,k,j]=sum(1<<int(v)for v in ids)
   gs=stored['geometry_state'][i,k]
   if gs=='SELF_VISIBLE'and inside[i,k,4]and stored['query_continuous_valid'][i,k]and stored['reference_continuous_valid'][i,k]:
    target=np.asarray(g['reference_camera_xyz_mm'][k]);hits=old.compact_ray(kernel.ray_hits(vertices,tri,invK@np.r_[centre,1.],float(np.linalg.norm(target)),set(g['incident_triangle_indices'][k])));qrays+=1
    for j,h in enumerate(hits):
     if not h['no_hit']:first[i,k,j]=h['first_Z_mm']
   label,why=old.classify(gs,bool(stored['query_continuous_valid'][i,k]),bool(stored['reference_continuous_valid'][i,k]),inside[i,k],raw[i,k],count[i,k],own[i,k],bits[i,k],full[i,k],ownidx,r['depth_scale'],g['reference_camera_xyz_mm'][k][2],first[i,k,0]);qs[i,k]=label;reason[i,k]=why
   if same[i,k]:
    for actual,saved in ((raw,stored['neighbour_depth_raw_uint16']),(inside,stored['neighbour_inbounds']),(full,stored['neighbour_own_full']),(own,stored['neighbour_own_visible']),(count,stored['neighbour_visible_owner_count']),(bits,stored['neighbour_owner_bits_uint64'])):assert np.array_equal(actual[i,k],saved[i,k])
    assert label==original[i,k],('equal_sampling_centres',i,k)
   comparison.append(dict(slot_index=8*i+k,input_id=a['input_id'],instance_row_index=i,logical_aux_slot=k,split=a['split'],original_split=a['original_split'],object_id=oid,scene_id=r['scene_id'],frame_id=r['frame_id'],instance_id=ownidx,query_xy=q[k].tolist(),reference_xy=rr[k].tolist(),original_L=a['roi_side'],q_reference_distance_px=float(distance[k]),distance_bin=distance_bin(float(distance[k])),old_r_teacher_state=original[i,k],old_r_reason=stored['reason'][i,k],q_sample_diagnostic_state=label,q_sample_reason=why,original_geometry_state=gs,original_geometry_stable=bool(stored['geometry_stable'][i,k]),same_rounded_sampling_centre=bool(same[i,k]),native_mapping='UNKNOWN',human_gold=False,training_label_admitted=False))
  if(i+1)%100==0:print(json.dumps(dict(instances=i+1,total=400,seconds=time.monotonic()-start)),flush=True)
 assert len(comparison)==3200
 with(H/'ALL3200_DIAGNOSTIC_COMPARISON.jsonl').open('x')as f:
  for r in comparison:f.write(json.dumps(r,allow_nan=False)+'\n')
 np.savez_compressed(H/'DIAGNOSTIC3200_Q_SAMPLED_EVIDENCE.npz',old_r_teacher_state=original,q_sample_diagnostic_state=qs,q_sample_reason=reason,original_geometry_state=stored['geometry_state'],original_geometry_stable=stored['geometry_stable'],q_neighbour_depth_raw_uint16=raw,q_neighbour_inbounds=inside,q_neighbour_own_full=full,q_neighbour_own_visible=own,q_neighbour_visible_owner_count=count,q_neighbour_owner_bits_uint64=bits,q_centre_target_first_cameraZ_mm=first,same_rounded_sampling_centre=same,original_query_xy=stored['query_xy_input_copy'],original_reference_xy=stored['reference_xy_GT_only'],original_L=stored['roi_side_input_copy'],original_depth_scale_to_mm=stored['depth_scale_to_mm'])
 def group(data):
  matrix=[[sum(r['old_r_teacher_state']==a and r['q_sample_diagnostic_state']==b for r in data)for b in STATES]for a in STATES]
  return dict(N=len(data),rows_old_r_cols_q_states=list(STATES),matrix=matrix,old_counts=dict(collections.Counter(r['old_r_teacher_state']for r in data)),q_counts=dict(collections.Counter(r['q_sample_diagnostic_state']for r in data)),changed=sum(r['old_r_teacher_state']!=r['q_sample_diagnostic_state']for r in data),old_known_DE=sum(r['old_r_teacher_state']in'DE'for r in data),old_DE_become_U=sum(r['old_r_teacher_state']in'DE'and r['q_sample_diagnostic_state']=='U'for r in data),old_DE_flip_D_E=sum(r['old_r_teacher_state']in'DE'and r['q_sample_diagnostic_state']in'DE'and r['old_r_teacher_state']!=r['q_sample_diagnostic_state']for r in data))
 groups={'ALL':group(comparison)}
 for split in('TRAIN','VALIDATION'):
  groups[split]=group([r for r in comparison if r['split']==split])
  for oid in(16,21):
   data=[r for r in comparison if r['split']==split and r['object_id']==oid]
   groups[split+'/object'+str(oid)]=group(data)
   for name in p['distance_bins_px']:groups[split+'/object'+str(oid)+'/distance/'+name]=group([r for r in data if r['distance_bin']==name])
 for name in p['distance_bins_px']:
  data=[r for r in comparison if r['distance_bin']==name]
  groups['distance/'+name]=group(data)
 result=dict(completed=True,PASS=True,policy=bind(POLICY),code=bind(__file__),bindings=b,all400_instances_3200_queries_retained=True,original_r_replay_exact3200=True,groups=groups,q_diagnostic_reason_counts=dict(collections.Counter(reason.ravel().tolist())),original_geometric_state_counts=dict(collections.Counter(stored['geometry_state'].ravel().tolist())),same_rounded_sampling_centre_count=int(same.sum()),same_sampling_centre_masks_depth_and_labels_exact=True,q_integer_target_ray_calls=qrays,official_depth_mask_payload_files=len(verified),source_RGB_encoder_head_fit_current_V112_V117_PnP=0,original_q_r_L_labels_folds_unchanged=True,new_training_admission=False,interpretation_limits=p['limits'],surface_not_junction_evidence='V113 nearest-original-vertex construction; wood16 PCA-box→surface vertex18.29–21.39mm, one selectedpoint has one triangle and0degreeincident-normal spread. On-surface is not certified edge/junction visibility.',seconds=time.monotonic()-start)
 write('SOURCE_PAYLOAD_READ_RECEIPTS.json',dict(files=filefacts,models=list(modelbindings.values()),RGB_reads=0))
 write('RESULT.json',result);write('SEAL.json',dict(completed=True,PASS=True,files=[bind(H/n)for n in('POLICY_BEFORE_PAYLOADS.json','compare_r_and_q_teacher.py','GENERATED_CHECKS.json','ORIGINAL_R_REPLAY.json','STARTED.json','ALL3200_DIAGNOSTIC_COMPARISON.jsonl','DIAGNOSTIC3200_Q_SAMPLED_EVIDENCE.npz','SOURCE_PAYLOAD_READ_RECEIPTS.json','RESULT.json')]))
 print(json.dumps(dict(result=bind(H/'RESULT.json'),seal=bind(H/'SEAL.json'),all=groups['ALL'],seconds=result['seconds'])),flush=True)

if __name__=='__main__':
 try:run()
 except BaseException:
  if not(H/'FAILED_EXECUTION.json').exists():write('FAILED_EXECUTION.json',dict(traceback=traceback.format_exc(),no_automatic_retry=True))
  raise
