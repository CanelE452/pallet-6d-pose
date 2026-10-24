"""Independent all-ray raw-depth/instance-mask replay. No builder import/RGB."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
from PIL import Image
import io,json,hashlib,math,traceback,time
import numpy as np
R=Path(__file__).resolve().parent;V=R.parent;C=V/'corpus_build_v1'
def rd(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return {'path':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size}
def check(b):
 actual=bind(b['path']);assert all(actual[k]==b[k]for k in b if k in actual),b['path'];return actual
def save(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
SIGNS=np.array([[-1,-1,1],[1,-1,1],[1,-1,-1],[-1,-1,-1],[-1,1,1],[1,1,1],[1,1,-1],[-1,1,-1]],np.int8)
SIDES=[[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2]]
def reconstruct(row,model):
 lower=np.array([model['min_'+a]for a in 'xyz'],float);size=np.array([model['size_'+a]for a in 'xyz'],float);upper=lower+size
 corners=np.where(SIGNS>0,upper,lower)
 rot=np.asarray(row['cam_R_m2c'],float).reshape(3,3);trans=np.asarray(row['cam_t_m2c'],float);K=np.asarray(row['K'],float).reshape(3,3)
 # Independent camera model calculation and axis-aligned face selection.
 eye=np.linalg.inv(rot).dot(-trans);delta=eye-(lower+upper)/2
 front=max(range(4),key=lambda j:[-delta[1],delta[1],-delta[0],delta[0]][j])
 cam=np.array([[math.fsum(float(rot[a,b])*float(p[b])for b in range(3))+float(trans[a])for a in range(3)]for p in corners])
 assert K[0,1]==K[1,0]==K[2,0]==K[2,1]==0 and K[2,2]==1 and K[0,0]>0 and K[1,1]>0
 # Equivalently projected using divide-before-add instead of hp @ K / hp.z.
 projected=np.c_[K[0,0]*(cam[:,0]/cam[:,2])+K[0,2],K[1,1]*(cam[:,1]/cam[:,2])+K[1,2]]
 face=SIDES[front];top=sorted([j for j in face if SIGNS[j,2]>0],key=lambda j:projected[j,0])
 bottom=[j for j in face if SIGNS[j,2]<0]
 def vertical_partner(j):return next(k for k in bottom if np.array_equal(corners[k,:2],corners[j,:2]))
 order=top+[vertical_partner(top[1]),vertical_partner(top[0])]
 opp=SIDES[front^1];axis=1 if front<2 else 0
 def across(j):
  other=[a for a in range(3)if a!=axis]
  return next(k for k in opp if np.array_equal(corners[k,other],corners[j,other]))
 perm=np.array(order+[across(j)for j in order],np.int8)
 # Any adjacent AABB face camera-facing, separately from exact mesh ownership.
 selfvis=np.array([any(eye[a]>upper[a]if SIGNS[j,a]>0 else eye[a]<lower[a]for a in range(3))for j in perm],np.int8)
 return projected[perm],cam[perm],perm,selfvis,K
FIELDS=['label','reason','relation','pixel_xy','depth_raw','owner_count','unique_owner_id','owner_type','own_front_hit','observed_Z_mm','gamma','ref_range_mm','observed_range_lo_mm','observed_range_hi_mm','raster_offset_px']
def decide(reference,z,K,hw,raw=None,hits=None,target=None,scale=1.):
 h,w=hw;x,y=map(float,reference)
 rec={'label':-1,'reason':'INVALID_REFERENCE_RASTER','relation':'INVALID','pixel_xy':[-1,-1],'depth_raw':-1,'owner_count':-1,'unique_owner_id':-1,'owner_type':'INVALID','own_front_hit':False,'observed_Z_mm':float('nan'),'gamma':float('nan'),'ref_range_mm':float('nan'),'observed_range_lo_mm':float('nan'),'observed_range_hi_mm':float('nan'),'raster_offset_px':float('nan')}
 if not all(math.isfinite(t)for t in [x,y,z])or z<=0 or not(0<=x<w and 0<=y<h):return rec
 px,py=math.floor(x+.5),math.floor(y+.5)
 if not(0<=px<w and 0<=py<h):return rec
 rec['pixel_xy']=[px,py];rec['raster_offset_px']=math.hypot(px-x,py-y)
 if hits is None:return rec
 hitids=[i for i,v in hits if v];count=len(hitids);owner=hitids[0]if count==1 else-2 if count>1 else-1
 ownertype='UNASSIGNED'if count==0 else'OVERLAP'if count>1 else'OWN'if owner==target else'OTHER'
 rec.update(depth_raw=int(raw),owner_count=count,unique_owner_id=owner,owner_type=ownertype)
 if raw<=0 or not math.isfinite(scale)or scale<=0:rec['reason']='INVALID_DEPTH';return rec
 if not np.isfinite(K).all()or K[0,0]<=0 or K[1,1]<=0:rec['reason']='INVALID_CAMERA';return rec
 # K is verified canonical pinhole. hypot is independent of solve+np.linalg.norm.
 gamma=math.hypot((px-float(K[0,2]))/float(K[0,0]),(py-float(K[1,2]))/float(K[1,1]),1.)
 zl=max(0.,(raw-.5)*scale);zh=(raw+.5)*scale
 rec.update(gamma=gamma,ref_range_mm=z*gamma,observed_Z_mm=raw*scale,observed_range_lo_mm=zl*gamma,observed_range_hi_mm=zh*gamma)
 # Independent equivalent Z-domain test uses the ray-dependent range margin/gamma.
 # Do not change a result or tolerance if a label disagrees with corpus.
 if z-zh>15./gamma:
  rec['relation']='FRONT'
  if ownertype=='OTHER':rec.update(label=1,reason='EXTERNAL_OTHER_FIRST_HIT_PROXY')
  elif ownertype=='OWN':rec.update(reason='FRONT_OWN_U',own_front_hit=True)
  elif ownertype=='OVERLAP':rec['reason']='FRONT_OVERLAP_U'
  else:rec['reason']='FRONT_UNASSIGNED_U'
 elif zl-z>15./gamma:rec.update(relation='CLEAR',label=0,reason='CLEAR_TO_VIRTUAL_ENDPOINT_PROXY')
 else:rec.update(relation='BAND',reason='RANGE_BAND_U')
 return rec

def fixtures():
 K=np.eye(3);tests=[]
 def one(name,z,raw,hits,target,scale,expected,r=(0,0),k=K):
  rec=decide(r,z,k,[3,3],raw,hits,target,scale);assert(rec['label'],rec['reason'])==expected,(name,rec);tests.append(name)
 one('foreignfront',1020.,10000,[(1,True)],0,.1,(1,'EXTERNAL_OTHER_FIRST_HIT_PROXY'))
 one('ownfrontU',1020.,10000,[(0,True)],0,.1,(-1,'FRONT_OWN_U'))
 one('unassignedfrontU',1020.,10000,[],0,.1,(-1,'FRONT_UNASSIGNED_U'))
 one('overlapfrontU',1020.,10000,[(0,True),(1,True)],0,.1,(-1,'FRONT_OVERLAP_U'))
 one('invaliddepthpositiveown',1020.,0,[(0,True)],0,.1,(-1,'INVALID_DEPTH'))
 one('clearirrespectiveowner',980.,10000,[(0,True),(1,True)],0,.1,(0,'CLEAR_TO_VIRTUAL_ENDPOINT_PROXY'))
 one('strict15quantboundary',1016.,500,[(1,True)],0,2.,(-1,'RANGE_BAND_U'))
 one('above15quantboundary',1016.000001,500,[(1,True)],0,2.,(1,'EXTERNAL_OTHER_FIRST_HIT_PROXY'))
 one('nonunitrange',1010.,10000,[(1,True)],0,.1,(1,'EXTERNAL_OTHER_FIRST_HIT_PROXY'),r=(1,0),k=np.diag([.5,1.,1.]))
 one('roundOOB',1020.,10000,[(1,True)],0,.1,(-1,'INVALID_REFERENCE_RASTER'),r=(2.8,0))
 return {'PASS':True,'cases':tests,'no_builder_function_import':True,'M0_not_argument':True}

def main():
 start=time.monotonic();policy=rd(R/'INDEPENDENT_REPLAY_POLICY_BEFORE_OUTPUT_ARRAYS.json');assert policy['exact_discrete_and_label_equality_no_tolerance']
 save(R/'EXECUTION_STARTED.json',{'created_UTC':datetime.now(timezone.utc).isoformat(),'policy':bind(R/'INDEPENDENT_REPLAY_POLICY_BEFORE_OUTPUT_ARRAYS.json'),'code':bind(__file__),'new_RGB_model_current_fit_forward_PnP':0})
 save(R/'GENERATED_INDEPENDENT_ARITHMETIC_TESTS.json',fixtures())
 protocol=rd(V/'SOURCE_EXTERNAL_RAY_PROTOCOL_BEFORE_DEPTH_PIXELS.json');B=protocol['bindings']
 for item in B.values():check(item)
 assert protocol['range_margin_mm']==15
 cs=rd(C/'CORPUS_SEAL.json');assert cs['completed']and bind(C/'CORPUS_SEAL.json')['sha256']=='e806a5777d30adce18c97216beb9eb2e5680b987ba605d4e00f87b1de2608d38'
 for b in cs['outputs']:
  p=Path(b['path']);p=p if p.is_absolute()else V/p;check({**b,'path':str(p)})
 rows=[json.loads(s)for s in Path(B['manifest']['path']).read_text().splitlines()];assert len(rows)==4200
 ids=[f'YCB:{r["scene_id"]:06d}:{r["frame_id"]:06d}:{r["instance_id"]:06d}'for r in rows];assert len(set(ids))==4200
 with np.load(B['old_GT_corpus']['path'],allow_pickle=False)as z:
  chosen=np.flatnonzero(z['source']=='YCB_PUBLIC_REAL');assert z['id'][chosen].tolist()==ids
  oldref=z['native_GTref8'][chosen];oldperm=z['perm_dynamic_to_fixed8'][chosen];oldself=z['convex_self_proxy8_separate'][chosen]
 models=rd(B['models_info']['path']);groups=defaultdict(list);scenes={};official=[]
 ownxyz=[];ownproj=[];ownperm=[];ownself=[];Ks=[]
 for i,row in enumerate(rows):
  assert row['split']==('train'if row['scene_id']<8 else'val')
  scene=row['scene_id'];frame=row['frame_id'];folder=Path(row['depth']).parent.parent
  if scene not in scenes:
   scenes[scene]=(rd(folder/'scene_gt.json'),rd(folder/'scene_camera.json'));official.extend([bind(folder/'scene_gt.json'),bind(folder/'scene_camera.json')])
  gt,cam=scenes[scene];pose=gt[str(frame)][row['instance_id']];cal=cam[str(frame)]
  assert pose['obj_id']==row['object_id'];assert row['cam_R_m2c']==pose['cam_R_m2c']and row['cam_t_m2c']==pose['cam_t_m2c']
  assert row['K']==cal['cam_K']and row['depth_scale']==cal['depth_scale']and row['depth_scale']>0
  r,xyz,perm,sf,K=reconstruct(row,models[str(row['object_id'])])
  assert np.array_equal(perm,oldperm[i])and np.array_equal(sf,oldself[i]),(i,perm,sf)
  assert np.isfinite(xyz).all()and(xyz[:,2]>0).all()
  # Original frozen r is used for raster sampling; alternate projection is reported.
  altpix=np.floor(r+.5).astype(np.int64);fixedpix=np.floor(oldref[i]+.5).astype(np.int64)
  assert np.array_equal(altpix,fixedpix),('projection_rounding_pixel_disagreement',i)
  ownproj.append(r);ownxyz.append(xyz);ownperm.append(perm);ownself.append(sf);Ks.append(K);groups[scene,frame].append(i)
 assert len(groups)==1000
 result=[None]*4200;actual_target_mask=np.full((4200,8),-1,np.int8);actual_files=[];decode_count=Counter();offending=None
 try:
  for gn,((scene,frame),indices)in enumerate(groups.items(),1):
   offending={'scene':scene,'frame':frame,'indices':indices};ann=scenes[scene][0][str(frame)]
   assert sorted(rows[i]['instance_id']for i in indices)==list(range(len(ann)))
   depthpath=Path(rows[indices[0]]['depth']);assert len({rows[i]['depth']for i in indices})==1
   rawbytes=depthpath.read_bytes();assert rawbytes[:8]==b'\x89PNG\r\n\x1a\n'and list(rawbytes[24:26])==[16,0]
   depth=np.asarray(Image.open(io.BytesIO(rawbytes)));assert depth.shape==(rows[indices[0]]['height'],rows[indices[0]]['width'])
   assert np.issubdtype(depth.dtype,np.integer)and depth.min()>=0 and depth.max()<=65535
   actual_files.append({'path':str(depthpath),'sha256':hashlib.sha256(rawbytes).hexdigest(),'bytes':len(rawbytes)});decode_count['official_scene_depth_png']+=1
   masks=[]
   for i in indices:
    path=Path(rows[i]['mask_vis']);blob=path.read_bytes();a=np.asarray(Image.open(io.BytesIO(blob)));assert a.shape==depth.shape
    masks.append((rows[i]['instance_id'],a>0));actual_files.append({'path':str(path),'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)});decode_count['visible_instance_mask_png']+=1
   for i in indices:
    group=[];row=rows[i]
    for k in range(8):
     rr=oldref[i,k];z=float(ownxyz[i][k,2]);h,w=depth.shape
     if np.isfinite(rr).all()and z>0 and 0<=rr[0]<w and 0<=rr[1]<h:
      px,py=math.floor(float(rr[0])+.5),math.floor(float(rr[1])+.5)
     else:px=py=-1
     if 0<=px<w and 0<=py<h:
      hits=[(ident,bool(m[py,px]))for ident,m in masks];d=int(depth[py,px]);actual_target_mask[i,k]=int(dict(hits)[row['instance_id']])
      rec=decide(rr,z,Ks[i],depth.shape,d,hits,row['instance_id'],row['depth_scale'])
     else:rec=decide(rr,z,Ks[i],depth.shape)
     group.append(rec)
    result[i]=group
   if gn%100==0:print('INDEPENDENT_PIXEL_FRAMES',gn,'/1000',flush=True)
 except Exception as e:
  save(R/'FAILED_ACTUAL_REPLAY_PRESERVED.json',{'offending':offending,'exception':repr(e),'traceback':traceback.format_exc(),'code':bind(__file__)});raise
 assert all(x is not None for x in result)
 assert decode_count=={'official_scene_depth_png':1000,'visible_instance_mask_png':4200}
 source_bindings=rd(C/'SOURCE_DEPTH_MASK_BYTE_BINDINGS.json')['files'];assert len(source_bindings)==5200
 expected={b['path']:b for b in source_bindings};assert len(expected)==5200 and len({b['path']for b in actual_files})==5200
 for b in actual_files:assert b==expected[b['path']]
 independent={k+'8':np.asarray([[rec[k]for rec in group]for group in result])for k in FIELDS}
 independent.update(reference_xy8=oldref,reference_camera_xyz_mm8=np.asarray(ownxyz),perm_dynamic_to_fixed8=np.asarray(ownperm),convex_self_visible_proxy8_separate=np.asarray(ownself),target_M4_at_original_reference_pixel8=actual_target_mask)
 np.savez_compressed(R/'INDEPENDENT_ALL33600_REPLAY_GT_ONLY.npz',id=np.asarray(ids),**independent)
 # Only now open saved actual corpus arrays and old support diagnostics.
 with np.load(C/'NATIVE8_EXTERNAL_RAY_GT_ONLY.npz',allow_pickle=False)as z:saved={k:z[k]for k in z.files}
 assert saved['id'].tolist()==ids and saved['split'].tolist()==[r['split']for r in rows]
 assert saved['reference_xy8'].dtype==oldref.dtype and np.array_equal(saved['reference_xy8'],oldref)
 assert np.array_equal(saved['perm_dynamic_to_fixed8'],np.asarray(ownperm))and np.array_equal(saved['convex_self_visible_proxy8_separate'],np.asarray(ownself))
 assert (saved['human_semantic_gold8']==-1).all()
 float_differences={};strict_checked=[]
 floats={'observed_Z_mm8','gamma8','ref_range_mm8','observed_range_lo_mm8','observed_range_hi_mm8','raster_offset_px8','reference_camera_xyz_mm8'}
 for key,actual in independent.items():
  if key=='target_M4_at_original_reference_pixel8':continue
  expected=saved[key];assert expected.shape==actual.shape,(key,expected.shape,actual.shape)
  if key in floats:
   assert np.array_equal(np.isnan(actual),np.isnan(expected)),key
   finite=np.isfinite(actual)&np.isfinite(expected);diff=np.abs(actual[finite]-expected[finite])
   float_differences[key]={'finite_values':int(finite.sum()),'exact_different_values':int((diff!=0).sum()),'maximum_absolute_difference':float(diff.max(initial=0)),'mean_absolute_difference':float(diff.mean())if diff.size else0,'numeric_tolerance_used_to_change_or_accept_labels':False}
  else:assert np.array_equal(actual,expected),key;strict_checked.append(key)
 projection_delta=np.abs(np.asarray(ownproj)-oldref)
 float_differences['independent_analytical_projection_vs_original_frozen_r']={'values':projection_delta.size,'exact_different_values':int((projection_delta!=0).sum()),'maximum_absolute_difference_px':float(projection_delta.max()),'all_33600_floor_r_plus_half_pixels_exact':True,'frozen_r_replaced':False}
 with np.load(B['old_GT_corpus']['path'],allow_pickle=False)as z:
  for name,key in [('old_local_support_label8_DIAGNOSTIC_ONLY','label8'),('old_center_M08_DIAGNOSTIC_ONLY','center_M08'),('old_center_M48_DIAGNOSTIC_ONLY','center_M48')]:assert np.array_equal(saved[name],z[key][chosen])
  oldM4=z['center_M48'][chosen]
 assert np.array_equal(actual_target_mask,oldM4),('actualM4_at_reference_vs_old_GT_mismatch',np.count_nonzero(actual_target_mask!=oldM4))
 # Verify every JSONL field against NPZ (None is only the serialized invalid NaN).
 jsonl=[json.loads(line)for line in (C/'PER_INSTANCE_EXTERNAL_RAY_GT_ONLY.jsonl').read_text().splitlines()];assert len(jsonl)==4200
 for i,row in enumerate(jsonl):
  assert row['id']==ids[i]and row['split']==rows[i]['split']and len(row['ray8'])==8
  assert np.array_equal(np.asarray(row['reference_xy8']),saved['reference_xy8'][i])and np.array_equal(np.asarray(row['camera_xyz_mm8']),saved['reference_camera_xyz_mm8'][i])
  for k,rec in enumerate(row['ray8']):
   assert set(rec)==set(FIELDS)
   for key,value in rec.items():
    s=saved[key+'8'][i,k]
    if value is None:assert np.ndim(s)==0 and np.isnan(s)
    else:assert np.array_equal(np.asarray(value),np.asarray(s)),(i,k,key)
 ordered=[json.loads(s)for s in (C/'ORDERED_METADATA_INDEX_GT_ONLY.jsonl').read_text().splitlines()];assert len(ordered)==4200
 for i,row in enumerate(ordered):assert row['row_index']==i and row['id']==ids[i]and row['metadata']==rows[i]
 bysplit={}
 for split in ['train','val']:
  subset=[rec for i,grp in enumerate(result)if rows[i]['split']==split for rec in grp]
  bysplit[split]={'rays':len(subset),'labels':dict(Counter(str(rec['label'])for rec in subset)),'reasons':dict(Counter(rec['reason']for rec in subset)),'owners':dict(Counter(rec['owner_type']for rec in subset))}
 published=rd(C/'CORPUS_SUMMARY_KO.json')
 for split in bysplit:
  for ownname,pubname in [('labels','labels'),('reasons','reasons'),('owners','owner')]:assert bysplit[split][ownname]==published['by_split'][split][pubname]
 arrays=independent;label=arrays['label8'];reason=arrays['reason8'];owner=arrays['owner_type8'];relation=arrays['relation8'];rawdepth=arrays['depth_raw8']
 assert (label[rawdepth==0]==-1).all()
 assert (label[np.isin(reason,['FRONT_OWN_U','FRONT_UNASSIGNED_U','FRONT_OVERLAP_U','INVALID_DEPTH','INVALID_REFERENCE_RASTER','RANGE_BAND_U'])]==-1).all()
 assert ((relation[label==1]=='FRONT')&(owner[label==1]=='OTHER')&(arrays['owner_count8'][label==1]==1)).all()
 assert (relation[label==0]=='CLEAR').all()
 valid=np.isfinite(arrays['ref_range_mm8']);frontgap=arrays['ref_range_mm8'][valid]-arrays['observed_range_hi_mm8'][valid];cleargap=arrays['observed_range_lo_mm8'][valid]-arrays['ref_range_mm8'][valid]
 boundarydistance=np.minimum(abs(frontgap-15.),abs(cleargap-15.))
 report={'schema':'V87_ALL33600_INDEPENDENT_RAW_DEPTH_VISIBLE_OWNER_OUTPUT_AUDIT','completed':True,'PASS':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'rows':4200,'rays':33600,'official_frames':1000,
  'actual_decodes':dict(decode_count),'all5200_source_SHA_bytes_match_builder_inventory':True,'all_official_scene_GT_instance_ordinals_complete':True,
  'source_metadata_R_t_K_depthscale_and_objectID_exact_official':True,'independent_geometry_perm_self_exact_old_GT_all4200':True,
  'original_reference_r_exact_saved_oldGT_all33600':True,'all33600_alternate_projection_raster_pixels_exact':True,
  'independent_decision_arithmetic':'Original frozen r for NN; independent fsum cameraXYZ, canonical-pinhole hypot gamma; compare zref-Zhi >15/gamma /Zlo-zref>15/gamma, not builder refRange-observedRange arithmetic. Ownership from list of all same-frame individual mask pixel hits; no builder owner map/sample_ray/native_math imported.',
  'strict_array_equality_fields':strict_checked,'float_differences':float_differences,'minimum_actual_distance_to_15mm_relation_boundary_mm':float(boundarydistance.min()),
  'all33600_labels_reasons_relations_pixels_depth_owner_invalid_flags_exact':True,'all33600_JSONL_NPZ_fields_order_match':True,
  'all4200_ordered_metadata_rows_exact_manifest':True,'all33600_target_M4_native_pixel_equals_old_center_M4_including_invalid':True,
  'old_support_M0_M4_diagnostic_alias_arrays_exact_and_never_decision_inputs':True,'by_split':bysplit,
  'scientific_limits_KO':'이PASS는rawdepth/가시maskowner에서고정조건부rayproxy를정확히만들었다는구현·provenance검증이다. CLEAR는virtualendpoint까지의clear조건일뿐physicalcueD가아니고 OWNfront는semanticS나negative가아니다. one-layerdepth/weakM4owner와15mm고정margin은sensor/posecalibration증명아님. M0픽셀조건·human/native8gold·trainingadmission을추가하지않는다.',
  'M0_images_read':0,'source_RGB_or_ML_weights_features_read':0,'current223_read':0,'new_forward_fit_PnP':0,'training_admission':False,
  'elapsed_seconds':time.monotonic()-start,'bindings':[bind(R/'INDEPENDENT_REPLAY_POLICY_BEFORE_OUTPUT_ARRAYS.json'),bind(V/'SOURCE_EXTERNAL_RAY_PROTOCOL_BEFORE_DEPTH_PIXELS.json'),bind(V/'ROOT_BUILD_ENTRYLOCK_BEFORE_DEPTH_MASK_PIXELS.json'),bind(V/'ACTUAL_BUILD_COMPLETION_RECEIPT.json'),bind(C/'CORPUS_SEAL.json'),bind(C/'SOURCE_DEPTH_MASK_BYTE_BINDINGS.json'),bind(Path(__file__))]}
 save(R/'INDEPENDENT_ALL33600_OUTPUT_AUDIT_KO.json',report)
 save(R/'ACTUAL_SOURCE_FILES_INDEPENDENT_SHA.json',{'files':actual_files,'N':5200,'official_scene_metadata':official,'RGB_unread':True})
 save(R/'SEAL.json',{'completed':True,'PASS':True,'files':[bind(f)for f in sorted(R.iterdir())if f.is_file()],'corpus_changed':False,'new_RGB_model_current_fit_forward_PnP':0})
 print(json.dumps({'PASS':True,'report':bind(R/'INDEPENDENT_ALL33600_OUTPUT_AUDIT_KO.json'),'seal':bind(R/'SEAL.json'),'elapsed':report['elapsed_seconds'],'by_split':bysplit,'float_differences':float_differences},ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except Exception as error:
  if not(R/'FAILED_AUDIT_PRESERVED.json').exists():save(R/'FAILED_AUDIT_PRESERVED.json',{'completed':False,'error':repr(error),'traceback':traceback.format_exc(),'code':bind(__file__)})
  raise
