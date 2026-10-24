"""Independent MR6D tiny intake audit; metadata-only prepare, authorized saved run.

No producer imports, network, model/fit, pose fitting, native references or labels.
Original source bytes are read only after completed intake/download SHA guards.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,math,struct,sys
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent;M=V/'metadata';D=V/'provided_metadata';O=V/'factual_sample_v1'
POLICY_SHA='3b23bd1ef0c48819b8c84f07a8f0a53c372620457ca972878e47cfad21ebdfba'
COMMIT='3c4ed8453312655499c3d3a3cf38b4929fae77ee'
SELECTION_SHA='af217858e5d43fe63079d4bdbeccb8adb70cf4501150e9d9960b38075b20bcba'
BIN_NAMES=('LOW_LT_0.5','MID_0.5_TO_LT_0.9','HIGH_GE_0.9')
READS={}

def digest(b):return hashlib.sha256(b).hexdigest()
def h(s):return digest(s.encode())
def bind(path):
 p=Path(path).resolve();assert str(p).startswith(str(V)+'/'),p
 b=dict(path=str(p),sha256=digest(p.read_bytes()));READS[str(p)]=b;return b
def checked(b):assert bind(b['path'])==b,b;return Path(b['path'])
def read(path):bind(path);return json.loads(Path(path).read_text())
def write(path,a):
 assert not path.exists(),path
 path.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def equal(a,b):assert a==b,(a,b)
def bin_of(x):return BIN_NAMES[0] if x<.5 else BIN_NAMES[1] if x<.9 else BIN_NAMES[2]

def selection_from_metadata():
 equal(bind(V/'ROOT_PUBLIC_INTAKE_POLICY_BEFORE_REMOTE_SAMPLE_BODIES.json')['sha256'],POLICY_SHA)
 policy=read(V/'ROOT_PUBLIC_INTAKE_POLICY_BEFORE_REMOTE_SAMPLE_BODIES.json')
 clarification=read(V/'ROOT_MULTI_INSTANCE_SELECTION_CLARIFICATION_BEFORE_SAMPLES.json')
 pin=read(M/'PINNED_REMOTE_SOURCE.json');info=read(M/'HF_DATASET_INFO.json');receipt=read(M/'PROVIDED_METADATA_DOWNLOAD_RECEIPT.json')
 equal(pin['commit'],COMMIT);equal(info['sha'],COMMIT);equal(receipt['commit'],COMMIT)
 equal(bind(M/'HF_DATASET_INFO.json')['sha256'],pin['metadata_sha256'])
 assert len(info['siblings'])==92622 and len(receipt['files'])==124
 assert receipt['completed'] and receipt['PASS'] and receipt['image_depth_mask_mesh_bodies_downloaded']==0
 for r in receipt['files']:
  equal(bind(r['local_path'])['sha256'],r['sha256']);equal(Path(r['local_path']).stat().st_size,r['bytes'])
 files={r['rfilename'] for r in info['siblings']};scenes=[];frames_by_scene={};inventory=[]
 for rel in sorted(p for p in files if p.endswith('/scene_gt.json')):
  split,scene,_=rel.split('/');gt=read(D/rel);cp=read(D/split/scene/'scene_camera.json')
  gip=D/split/scene/'scene_gt_info.json';gi=read(gip) if gip.exists() else None
  fr=[];allbin=Counter();multi=0
  for key in sorted(gt,key=int):
   instances=[i for i,a in enumerate(gt[key]) if int(a['obj_id'])==1]
   if not instances:continue
   assert key in cp;multi+=len(instances)>1;fractions=[];qual=[]
   if gi is not None:
    assert len(gt[key])==len(gi[key])
    for i in instances:
     v=float(gi[key][i]['visib_fract']);assert math.isfinite(v) and 0<=v<=1
     fractions.append(dict(gt_instance_index=i,visib_fract=v));b=bin_of(v);allbin[b]+=1
     if b not in qual:qual.append(b)
   fr.append(dict(split=split,scene=scene,frame_key=key,frame_filename_stem=f'{int(key):06d}',frame_hash_input=f'{split}|{scene}|{key}',
    frame_hash=h(f'{split}|{scene}|{key}'),obj1_gt_instance_indices=instances,instance_visibility_fractions=fractions,qualifying_bins=qual))
  inventory.append(dict(split=split,scene=scene,all_frames=len(gt),obj1_frames=len(fr),obj1_instances=sum(len(r['obj1_gt_instance_indices'])for r in fr),
   multiple_obj1_frames=multi,instance_visibility_bin_counts=dict(allbin)))
  if fr:
   scenes.append(dict(split=split,scene=scene,scene_hash_input=f'MR6D_V151_SOURCE_SAMPLE_V1|{split}|{scene}',scene_hash=h(f'MR6D_V151_SOURCE_SAMPLE_V1|{split}|{scene}'),
    obj1_frames=len(fr),obj1_instances=sum(len(r['obj1_gt_instance_indices'])for r in fr),GTinfo_present=gi is not None,
    bin_eligible_frame_counts={b:sum(b in r['qualifying_bins'] for r in fr)for b in BIN_NAMES}))
   frames_by_scene[(split,scene)]=fr
 ranked=defaultdict(list)
 for s in scenes:ranked[s['split']].append(s)
 for split in ranked:ranked[split].sort(key=lambda s:(s['scene_hash'],s['scene']))
 chosen_scenes=[]
 while len(chosen_scenes)<policy['max_selected_scenes'] and any(ranked.values()):
  for split in sorted(ranked):
   if ranked[split] and len(chosen_scenes)<policy['max_selected_scenes']:chosen_scenes.append(ranked[split].pop(0))
 chosen=[];missing=[];payload=['models/obj_000001.ply']
 for s in chosen_scenes:
  pool=frames_by_scene[(s['split'],s['scene'])];dedup={}
  if s['GTinfo_present']:
   for b in BIN_NAMES:
    eligible=[r for r in pool if b in r['qualifying_bins']]
    if not eligible:missing.append(dict(split=s['split'],scene=s['scene'],bin=b,reason='NO_OBJ1_INSTANCE_QUALIFIES_METADATA_BIN'));continue
    row=min(eligible,key=lambda r:(r['frame_hash'],int(r['frame_key'])))
    if row['frame_key'] not in dedup:dedup[row['frame_key']]=dict(row,selection_reasons=[])
    dedup[row['frame_key']]['selection_reasons'].append(dict(visibility_bin=b,selection='MIN_SHA256_FRAME_AMONG_ANY_OBJ1_QUALIFYING'))
  else:
   for row in sorted(pool,key=lambda r:(r['frame_hash'],int(r['frame_key'])))[:3]:dedup[row['frame_key']]=dict(row,selection_reasons=[dict(visibility_bin=None,selection='MIN_HASH_FALLBACK_GTINFO_ABSENT')])
  for row in sorted(dedup.values(),key=lambda r:(r['frame_hash'],int(r['frame_key']))):
   prefix=f"{row['split']}/{row['scene']}";stem=row['frame_filename_stem'];rgb=[p for p in files if p.startswith(prefix+'/rgb/') and Path(p).stem==stem]
   assert len(rgb)==1;sf=dict(RGB=rgb[0],depth=prefix+'/depth/'+stem+'.png',full_masks=[],visible_masks=[])
   payload.extend([sf['RGB'],sf['depth']])
   for i in row['obj1_gt_instance_indices']:
    for folder,key in [('mask','full_masks'),('mask_visib','visible_masks')]:
     rel=f'{prefix}/{folder}/{stem}_{i:06d}.png';assert rel in files
     sf[key].append(dict(gt_instance_index=i,path=rel));payload.append(rel)
   chosen.append(dict(row,source_files=sf))
 selection=read(O/'FIXED_SOURCE_SAMPLE_SELECTION_BEFORE_PIXELS.json');equal(bind(O/'FIXED_SOURCE_SAMPLE_SELECTION_BEFORE_PIXELS.json')['sha256'],SELECTION_SHA)
 equal(selection['selected_scenes'],chosen_scenes);equal(selection['selected_frames'],chosen);equal(selection['all_obj1_scenes'],scenes)
 equal(selection['missing_bins'],missing);equal(selection['payload_paths'],payload)
 equal(len(chosen_scenes),4);equal(len(chosen),10);equal(sum(len(r['obj1_gt_instance_indices'])for r in chosen),13);equal(len(payload),47)
 equal(sum(r['obj1_frames']for r in inventory),230);equal(sum(r['obj1_instances']for r in inventory),303);equal(sum(r['multiple_obj1_frames']for r in inventory),73)
 return selection,dict(completed=True,PASS=True,HF_siblings=92622,provided_metadata_files=124,GT_released_scenes=len(inventory),
  pallet_scenes=len(scenes),pallet_frames=230,pallet_instances=303,multiple_pallet_frames=73,selected_scene_order=[(r['split'],r['scene'])for r in chosen_scenes],
  selected_frames=[dict(split=r['split'],scene=r['scene'],frame_key=r['frame_key'],all_obj1_gt_indices=r['obj1_gt_instance_indices'],selection_reasons=r['selection_reasons'])for r in chosen],
  missing_bins=missing,payload_count=47,scene_inventory=inventory,selection=bind(O/'FIXED_SOURCE_SAMPLE_SELECTION_BEFORE_PIXELS.json'))

def bbox(mask):
 yy,xx=np.nonzero(mask)
 return [int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)] if len(xx) else [-1,-1,-1,-1]

def ply(path):
 # Independent ASCII/binary scalar vertex decoder, also returning faces for static raw-axis evidence.
 data=Path(path).read_bytes();end=data.index(b'\n',data.index(b'end_header'))+1;head=data[:end].decode('ascii');lines=head.splitlines()
 elements=[];fmt=None;active=None
 for l in lines:
  p=l.split()
  if not p:continue
  if p[0]=='format':fmt=p[1]
  elif p[0]=='element':active=dict(name=p[1],N=int(p[2]),props=[]);elements.append(active)
  elif p[0]=='property':active['props'].append(p[1:])
 assert elements[0]['name']=='vertex';v=elements[0];names=[p[-1]for p in v['props']]
 assert all(len(p)==2 for p in v['props']) and all(c in names for c in ('x','y','z'))
 faces=[]
 if fmt=='ascii':
  body=data[end:].decode('ascii').splitlines();values=np.array([[float(x)for x in l.split()]for l in body[:v['N']]],np.float64)
  cursor=v['N']
  for e in elements[1:]:
   for l in body[cursor:cursor+e['N']]:
    if e['name']=='face':
     p=list(map(int,l.split()));assert p[0]==len(p)-1;faces.append(p[1:])
   cursor+=e['N']
 else:
  assert fmt in ('binary_little_endian','binary_big_endian');endian='<'if fmt=='binary_little_endian'else'>'
  types={'char':'b','uchar':'B','int8':'b','uint8':'B','short':'h','ushort':'H','int16':'h','uint16':'H','int':'i','uint':'I','int32':'i','uint32':'I','float':'f','float32':'f','double':'d','float64':'d'}
  st=struct.Struct(endian+''.join(types[p[0]]for p in v['props']));values=np.array([st.unpack_from(data,end+i*st.size)for i in range(v['N'])],np.float64);cursor=end+v['N']*st.size
  for e in elements[1:]:
   assert e['name']=='face' and len(e['props'])==1 and e['props'][0][0]=='list'
   _,ct,it,_=e['props'][0];count=struct.Struct(endian+types[ct])
   for _ in range(e['N']):
    n=count.unpack_from(data,cursor)[0];cursor+=count.size;st=struct.Struct(endian+types[it]*n);faces.append(list(st.unpack_from(data,cursor)));cursor+=st.size
 xyz=values[:,[names.index(c)for c in ('x','y','z')]];assert np.isfinite(xyz).all()
 return xyz,faces,head

def prepare():
 selection,proof=selection_from_metadata()
 write(H/'METADATA_SELECTION_RECALCULATION.json',proof)
 mask=np.zeros((5,7),bool);mask[1:4,2:6]=True;equal(bbox(mask),[2,1,4,3]);equal(int(mask.sum()),12)
 vis=mask.copy();vis[1,2]=False;equal(int((vis&~mask).sum()),0);equal(int(vis.sum()),11)
 overlap=np.zeros_like(mask);overlap[2:5,4:7]=True;equal(int((mask&overlap).sum()),4)
 write(H/'PREPARATION_READY.json',dict(completed=True,prepared_only=True,code=bind(__file__),metadata_review=bind(H/'METADATA_SELECTION_RECALCULATION.json'),
  generated_invented_mask_count_bbox_subset_overlap_tests_PASS=True,actual_RGB_depth_mask_mesh_body_or_hash_reads=0,
  run_requires_completed_intake_and_download_SHA_from_root=True,producer_helper_import_or_call=0,new_model_fit_or_native_labels=0,
  primary_BOP_source='https://raw.githubusercontent.com/thodan/bop_toolkit/master/docs/bop_datasets_format.md',
  convention_scope='BOP declares model/translation mm, scaled depth mm, right-handed model Z-up and camera looking +Z. MR actual mesh/deck/pose consistency is an additional audit; no R0-slot mapping follows.',
  uncertainty='MR paper reports nominal pallet dimensions and low-centimeter annotation accuracy; no <=3px native gold follows.'))
 print(json.dumps(dict(prepared_only=True,metadata=bind(H/'METADATA_SELECTION_RECALCULATION.json'),ready=bind(H/'PREPARATION_READY.json')),indent=2))

def run(intake_sha,download_sha):
 # Body gate is checked before loading Pillow or reading any original payload.
 equal(bind(O/'FACTUAL_SAMPLE_INTAKE_COMPLETE.json')['sha256'],intake_sha)
 equal(bind(O/'TINY_SOURCE_DOWNLOAD_COMPLETE.json')['sha256'],download_sha)
 final=read(O/'FACTUAL_SAMPLE_INTAKE_COMPLETE.json');download=read(O/'TINY_SOURCE_DOWNLOAD_COMPLETE.json')
 assert final['completed'] and final['PASS'] and download['completed'] and download['PASS']
 selection,selectionproof=selection_from_metadata();checked(final['selection']);checked(final['download_receipt'])
 plan=read(O/'PINNED_TINY_DOWNLOAD_PLAN_BEFORE_GET.json');correction=read(checked(download['manual_continuation_correction']))
 failure=read(O/'DOWNLOAD_FAILURE_PRESERVED.json');first=read(V/'factual_sample_preparation/FIRST_ATTEMPT_FAILURE_PROVENANCE.json')
 assert first and failure['successful_files']==1 and failure['failed']['relative_path']=='val/000005/rgb/000015.jpg'
 assert correction['first_session']==59891 and correction['first_terminal_exit']==1 and correction['explicit_parent_continuation_authorized']
 assert len(correction['verified_original_two_files_reused_without_GET'])==2 and correction['remaining_GET_count']==45
 equal(plan['selection'],selectionproof['selection']);equal([r['relative_path']for r in plan['payloads']],selection['payload_paths'])
 entries={}
 for r in correction['source_metadata']:
  data=read(checked(r['body']));assert r['requested_url'].startswith('https://huggingface.co/api/datasets/anas-gouda/mr6d/tree/'+COMMIT+'/')
  equal(Path(r['body']['path']).stat().st_size,r['bytes'])
  for x in data:
   if x['type']=='file':entries[x['path']]=x
 equal(len(download['files']),47);equal([r['relative_path']for r in download['files']],selection['payload_paths'])
 integrity=[];reuse=[];gets=[]
 for r,p in zip(download['files'],plan['payloads']):
  rel=r['relative_path'];entry=entries[rel];path=O/'original'/rel;raw=path.read_bytes();bind(path)
  equal(str(path),r['local_path']);equal(len(raw),entry['size']);equal(len(raw),p['expected_bytes']);equal(digest(raw),r['sha256'])
  equal(r['pinned_url'],'https://huggingface.co/datasets/anas-gouda/mr6d/resolve/'+COMMIT+'/'+rel)
  if entry.get('lfs'):
   equal(len(raw),entry['lfs']['size']);equal(digest(raw),entry['lfs']['oid']);kind='LFS_SHA256';content_ID=entry['lfs']['oid']
  else:
   equal(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),entry['oid']);kind='GIT_BLOB_SHA1';content_ID=entry['oid']
  equal(kind,r['content_identity_kind']);equal(content_ID,r['expected_source_content_ID']);assert r['source_content_ID_verified']
  (gets if r['GET_performed'] else reuse).append(rel)
  equal(r,read(O/'continuation_download_receipts'/f'{len(integrity):03d}.json'))
  integrity.append(dict(relative_path=rel,bytes=len(raw),sha256=digest(raw),source_ID_kind=kind,source_content_ID=content_ID,GET_performed=r['GET_performed']))
 equal(len(gets),45);equal(len(reuse),2);equal(reuse,selection['payload_paths'][:2]);equal(sum(r['bytes']for r in integrity),7787232)
 first_RGB=entries[failure['failed']['relative_path']]
 equal(failure['failed']['ETag'].strip('"'),first_RGB['xetHash']);assert first_RGB['xetHash']!=first_RGB['lfs']['oid']
 assert download['all_metadata_plus_payload_bytes']<=plan['cap']==100000000
 equal(download['downloaded_payload_bytes'],7787232);equal(download['all_metadata_plus_payload_bytes'],26972734)
 from PIL import Image,__version__ as PIL_VERSION
 producer=read(checked(final['validation']));records=[];overlaps=[];maskstats=[]
 for ix,r in enumerate(selection['selected_frames']):
  pref=D/r['split']/r['scene'];gt=read(pref/'scene_gt.json')[r['frame_key']];gi=read(pref/'scene_gt_info.json')[r['frame_key']];cam=read(pref/'scene_camera.json')[r['frame_key']]
  rgbpath=O/'original'/r['source_files']['RGB'];dpath=O/'original'/r['source_files']['depth']
  rgb=np.array(Image.open(rgbpath).convert('RGB'));depth=np.array(Image.open(dpath));assert rgb.dtype==np.uint8 and depth.dtype==np.uint16 and depth.shape==rgb.shape[:2]
  K=np.asarray(cam['cam_K'],np.float64).reshape(3,3);assert np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0 and cam['depth_scale']>0
  fulls={};visible={};instances=[]
  for i in r['obj1_gt_instance_indices']:
   assert gt[i]['obj_id']==1
   fullrow=next(x for x in r['source_files']['full_masks']if x['gt_instance_index']==i);visrow=next(x for x in r['source_files']['visible_masks']if x['gt_instance_index']==i)
   equal(Path(fullrow['path']).name,f"{r['frame_filename_stem']}_{i:06d}.png");equal(Path(visrow['path']).name,Path(fullrow['path']).name)
   fullraw=np.array(Image.open(O/'original'/fullrow['path']));visraw=np.array(Image.open(O/'original'/visrow['path']));assert fullraw.shape==visraw.shape==depth.shape
   full=fullraw!=0;vis=visraw!=0;fulls[i]=full;visible[i]=vis
   counts=dict(full=int(full.sum()),valid_depth_under_full=int((full&(depth>0)).sum()),visible=int(vis.sum()),visible_outside_full=int((vis&~full).sum()))
   delta={k:counts[c]-int(gi[i][k])for k,c in [('px_count_all','full'),('px_count_valid','valid_depth_under_full'),('px_count_visib','visible')]}
   R=np.asarray(gt[i]['cam_R_m2c'],np.float64).reshape(3,3);t=np.asarray(gt[i]['cam_t_m2c'],np.float64).reshape(3);assert np.isfinite(R).all() and np.isfinite(t).all()
   rec=dict(ownership_key=[r['split'],r['scene'],r['frame_key'],i,1],mask_file_index_matches_GT_list_index=True,
    full_mask=fullrow['path'],visible_mask=visrow['path'],raw_full_values=np.unique(fullraw).tolist(),raw_visible_values=np.unique(visraw).tolist(),
    counts=counts,provided_GT_info=gi[i],count_deltas_to_GTinfo=delta,mask_bbox_obj_inclusive=bbox(full),mask_bbox_visib_inclusive=bbox(vis),
    provided_fraction_matches_provided_count_ratio=math.isclose(gi[i]['visib_fract'],gi[i]['px_count_visib']/gi[i]['px_count_all'],abs_tol=1e-12),
    R_max_orthogonality_error=float(np.max(np.abs(R.T@R-np.eye(3)))),R_determinant=float(np.linalg.det(R)),cam_t_m2c_raw_mm=t.tolist(),native_label='UNASSIGNED')
   instances.append(rec);maskstats.append(rec)
  for ia,a in enumerate(r['obj1_gt_instance_indices']):
   for b in r['obj1_gt_instance_indices'][:ia]:
    overlaps.append(dict(split=r['split'],scene=r['scene'],frame_key=r['frame_key'],GT_indices=[b,a],full_overlap=int((fulls[b]&fulls[a]).sum()),visible_overlap=int((visible[b]&visible[a]).sum())))
  prior=producer['frames'][ix];equal(prior['raw_RGB_HW'],list(rgb.shape[:2]));equal(prior['decoded_RGB_sha256'],digest(rgb.tobytes()))
  equal(prior['depth_nonzero'],int((depth>0).sum()));equal(prior['depth_shape'],list(depth.shape))
  for a,b in zip(instances,prior['instances']):
   equal(a['counts']['full'],b['full_pixels']);equal(a['counts']['visible'],b['visible_pixels']);equal(a['counts']['visible_outside_full'],b['visible_outside_full_pixels'])
  records.append(dict(selection_index=ix,split=r['split'],scene=r['scene'],frame_key=r['frame_key'],RGB_HW=list(rgb.shape[:2]),
   decoded_RGB_sha256=digest(rgb.tobytes()),depth_dtype=str(depth.dtype),depth_nonzero=int((depth>0).sum()),depth_scale_to_mm=cam['depth_scale'],
   raw_K=K.tolist(),instances=instances))
 v,faces,header=ply(O/'original/models/obj_000001.ply');mi=read(D/'models/models_info.json')['1'];mn=v.min(0);mx=v.max(0)
 np.testing.assert_allclose(mn,[mi['min_x'],mi['min_y'],mi['min_z']],atol=1e-6,rtol=0);np.testing.assert_allclose(mx-mn,[mi['size_x'],mi['size_y'],mi['size_z']],atol=1e-6,rtol=0)
 equal(len(v),producer['mesh']['vertex_count']);np.testing.assert_array_equal(mn,producer['mesh']['bounds_min']);np.testing.assert_array_equal(mx,producer['mesh']['bounds_max'])
 levels=[]
 for z in np.unique(v[:,2]):
  flats=[]
  for fi,f in enumerate(faces):
   points=v[np.asarray(f)];
   if np.all(points[:,2]==z):
    area=sum(np.linalg.norm(np.cross(points[j]-points[0],points[j+1]-points[0]))*.5 for j in range(1,len(points)-1));flats.append(float(area))
  levels.append(dict(raw_model_Z_mm=float(z),vertex_count=int((v[:,2]==z).sum()),horizontal_faces=len(flats),horizontal_area_mm2=sum(flats)))
 mesh=dict(raw_bounds_min_mm=mn.tolist(),raw_bounds_max_mm=mx.tolist(),raw_extent_mm=(mx-mn).tolist(),vertex_count=len(v),face_count=len(faces),
  models_info_exact_within1e_minus6=True,raw_Z_level_evidence=levels,
  declared_BOP_model_Z_up='Specification evidence only; deck/underside interpretation and source supplied-view consistency need separate visual review.',
  canonical_AABB_corner_to_R0_native_slot_mapping='UNKNOWN_NOT_CREATED',pose_or_orientation_fit_calls=0)
 eqcounts=sum(any(v!=0 for v in r['count_deltas_to_GTinfo'].values())for r in maskstats)
 checked(final['download_receipt']);checked(final['selection'])
 result=dict(completed=True,PASS=True,scope='Independent integrity/selection/factual arithmetic, not clean-label or exclusive-owner certification.',
  selection=selectionproof,integrity=integrity,resume=dict(first_failure=bind(O/'DOWNLOAD_FAILURE_PRESERVED.json'),first_failure_provenance=bind(V/'factual_sample_preparation/FIRST_ATTEMPT_FAILURE_PROVENANCE.json'),
   correction=bind(O/'XET_ETAG_CONTENT_ID_CORRECTION_BEFORE_REMAINING_GET.json'),first_session59891_preserved=True,Xet_ETag_not_raw_LFS_SHA256=True,original_two_reused=reuse,remaining_GET_count=45),
  payload_bytes=7787232,accounted_metadata_plus_payload_bytes=26972734,frames=records,instances=13,instance_pair_overlaps=overlaps,mesh=mesh,
  observations=dict(visible_outside_full_instances=sum(r['counts']['visible_outside_full']>0 for r in maskstats),GTinfo_count_mismatch_instances=eqcounts,
   visible_mask_overlap_pairs=sum(r['visible_overlap']>0 for r in overlaps),exclusive_instance_owner_assumption_allowed=False),
  safeguards=dict(no_native_D_E_S_autofill=True,canonical_corners_are_not_R0_slot_mapping=True,nominal_model_low_cm_uncertainty_not_le3px_gold=True,
   no_source_mask_pose_or_selection_changes=True,provided_T_K_preserved=True),
  calls=dict(producer_helper_import_or_call=0,network=0,model_optimizer_forward_fit=0,current223_private_or_forbidden_phase_access=0,pose_orientation_least_fit=0,native_labels_created=0))
 write(H/'RECALCULATED_TINY_FACTUAL_SOURCE.json',result)
 for b in list(READS.values()):equal(digest(Path(b['path']).read_bytes()),b['sha256'])
 write(H/'REVIEW.json',dict(completed=True,PASS=True,code=bind(__file__),intake_seal=bind(O/'FACTUAL_SAMPLE_INTAKE_COMPLETE.json'),
  download_seal=bind(O/'TINY_SOURCE_DOWNLOAD_COMPLETE.json'),result=bind(H/'RECALCULATED_TINY_FACTUAL_SOURCE.json'),
  original_source_bindings_unchanged=True,calls=result['calls'],native_label_certification=False))
 print(json.dumps(dict(completed=True,PASS=True,receipt=bind(H/'REVIEW.json'),observations=result['observations'],overlaps=overlaps,mesh_bounds=mesh['raw_extent_mm']),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--intake-sha');p.add_argument('--download-sha');a=p.parse_args()
 assert a.prepare!=a.run
 if a.prepare:prepare()
 else:
  assert a.intake_sha and a.download_sha;run(a.intake_sha,a.download_sha)
