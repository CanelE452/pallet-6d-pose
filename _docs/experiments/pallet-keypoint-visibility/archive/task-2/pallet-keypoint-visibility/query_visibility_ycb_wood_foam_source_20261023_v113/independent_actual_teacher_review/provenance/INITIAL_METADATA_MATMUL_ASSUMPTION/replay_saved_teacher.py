"""Own saved numeric replay. No cv2/trimesh/model imports or source pixel reads."""
from pathlib import Path
import hashlib,json,collections
import numpy as np
V=Path(__file__).resolve().parents[1];O=V/'corpus_build_v1';H=Path(__file__).resolve().parent

def bind(p):p=Path(p);return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def check(d):assert bind(d['path'])['sha256']==d['sha256'];return Path(d['path'])
def own_geometry(hit,point_range):
 if not np.isfinite(hit).all()or abs(float(hit[0]-hit[1]))>.001:return'UNKNOWN',False
 gap=float(point_range-hit[0]);return('SELF'if gap>2 else'SELF_VISIBLE'if abs(gap)<=2 else'UNKNOWN'),True

def own_classify(geometry,qin,rin,inside,raw,nowner,own,bits,full,scale,z,centre_z):
 if not qin:return'U','QUERY_CONTINUOUS_OOB'
 if not rin:return'U','REFERENCE_CONTINUOUS_OOB'
 if not inside[4]:return'U','REFERENCE_NEAREST_PIXEL_OOB'
 if geometry=='SELF':return'S','KNOWN_TARGET_SELF_GEOMETRY'
 if geometry!='SELF_VISIBLE':return'U','GEOMETRY_UNKNOWN'
 good=inside&(raw>0);observed=raw.astype(np.float64)*scale;unique=inside&(nowner==1)&own
 if unique[4]and good[4]and np.isfinite(centre_z):
  if abs(observed[4]-centre_z)<=15 and np.count_nonzero(unique)>=5 and np.count_nonzero(good)>=5:return'D','SELF_VISIBLE_UNIQUE_OWN_DEPTH_SUPPORTED'
 same_foreign=bool(inside.all()and(nowner==1).all()and not own.any()and len({int(v)for v in bits})==1)
 if same_foreign and good[4]and observed[4]<z-15 and good.sum()>=7 and(observed[good]<z-15).all()and full.any():return'E','SELF_VISIBLE_SAME_FOREIGN9_NEARER_DEPTH7_WITH_AMODAL_SUPPORT'
 if(nowner>1).any():return'U','VISIBLE_OWNER_OVERLAP'
 if not good[4]:return'U','CENTRE_DEPTH_MISSING'
 if unique[4]and not np.isfinite(centre_z):return'U','CENTRE_TARGET_PIXEL_RAY_NOHIT'
 if unique[4]and abs(observed[4]-centre_z)>15:return'U','CENTRE_OWN_DEPTH_DISAGREEMENT'
 if same_foreign and not full.any():return'U','FOREIGN_WITHOUT_AMODAL_SUPPORT'
 if same_foreign:return'U','FOREIGN_DEPTH_NOT_STRICTLY_NEARER_SUPPORTED'
 if nowner[4]==0:return'U','CENTRE_VISIBLE_OWNER_UNASSIGNED'
 return'U','INSUFFICIENT_FIXED_3X3_SUPPORT'

sealpath=O/'CORPUS_SEAL.json';assert bind(sealpath)['sha256']=='573b2b3fc93869eb11f699a796bfb8ed3086e7696851f56db929a854f0ac0634'
seal=read(sealpath);assert seal['completed']and seal['all3200_preserved']and seal['training_admission']is False and seal['original_folds_unchanged']
for d in seal['outputs']:check(d)
p=read(check(seal['protocol']));check(seal['code']);check(seal['entry'])
inputs=rows(check(seal['inputs']));refs=rows(check(seal['reference_GT_only']));evidence=rows(O/'SLOT_EVIDENCE3200_GT_ONLY.jsonl');summary=read(O/'SUMMARY.json');bindings=read(O/'SOURCE_DEPTH_MASK_BINDINGS.json')
assert len(inputs)==len(refs)==400 and len(evidence)==3200 and summary['completed']and summary['all_slots']==3200
with np.load(O/'RAW_SURFACE_TEACHER3200_GT_ONLY.npz',allow_pickle=False)as f:a={k:f[k]for k in f.files}
shape=(400,8);assert a['raw_surface_teacher_state'].shape==a['external_target_D0_E1_else_minus1'].shape==shape
assert a['raw_surface_teacher_state'].dtype==np.dtype('U1')and a['external_target_D0_E1_else_minus1'].dtype==np.int8
state=a['raw_surface_teacher_state'];assert set(np.unique(state))<=set('DESU')
assert np.array_equal(a['external_target_D0_E1_else_minus1'],np.where(state=='D',0,np.where(state=='E',1,-1)).astype(np.int8))
assert np.array_equal(a['instance_row_index'],np.arange(400))and np.array_equal(a['logical_aux_slot'],np.broadcast_to(np.arange(8),(400,8)))
assert np.array_equal(a['input_id'],[r['input_id']for r in inputs])and len(set(a['input_id']))==400
assert np.array_equal(a['original_split'],[r['original_split']for r in inputs])and np.array_equal(a['split'],[r['split']for r in inputs])
assert collections.Counter(a['original_split'].tolist())==dict(train=300,val=100)
assert np.array_equal(a['query_xy_input_copy'],np.array([r['initial_points']for r in inputs],np.float64))and np.array_equal(a['roi_side_input_copy'],np.array([r['roi_side']for r in inputs],np.float64))
assert np.array_equal(a['reference_xy_GT_only'],np.array([r['reference_xy']for r in refs],np.float64))
assert np.array_equal(a['query_reference_error_px'],np.linalg.norm(a['query_xy_input_copy']-a['reference_xy_GT_only'],axis=2))
frames=bindings['frames'];assert bindings['all_visible_owner_sets_from_complete_scene_GT']
for fr in frames:assert len(fr['visible_masks'])==len(fr['instance_object_ids'])==fr['complete_scene_GT_owner_count']
reasoncounts=collections.Counter();checks=0;S_with_missing=0;overlap_saved=0;qnearest_only=0;known_qnearest_unavailable=0
for i,(inp,g)in enumerate(zip(inputs,refs)):
 assert i==inp['instance_row_index']==g['instance_row_index']and inp['input_id']==g['input_id']
 assert inp['split']==('TRAIN'if inp['original_split']=='train'else'VALIDATION')and inp['original_split']==g['source_instance']['split']
 assert inp['source_emulation']and inp['actual_initializer_query']is False and inp['native_mapping']=='UNKNOWN'and g['human_gold']is False
 fr=frames[int(a['frame_index'][i])];sr=g['source_instance'];assert fr['scene_id']==sr['scene_id']and fr['frame_id']==sr['frame_id']and fr['instance_object_ids'][sr['instance_id']]==sr['object_id']
 assert fr['complete_scene_GT_owner_count']==g['complete_scene_GT_owner_count']
 assert a['depth_scale_to_mm'][i]==sr['depth_scale']
 W=inp['raw_hw'][1];height=inp['raw_hw'][0]
 for k in range(8):
  log=evidence[i*8+k];assert log['slot_index']==i*8+k and log['input_id']==inp['input_id']and log['logical_aux_slot']==k and log['aux_surface_id']==g['aux_surface_ids'][k]
  q=a['query_xy_input_copy'][i,k];r=a['reference_xy_GT_only'][i,k];qi=bool(0<=q[0]<W and 0<=q[1]<height);ri=bool(0<=r[0]<W and 0<=r[1]<height);qn=np.floor(q+.5);rc=np.floor(r+.5).astype(int)
  assert a['query_continuous_valid'][i,k]==qi and a['reference_continuous_valid'][i,k]==ri
  qnv=0<=qn[0]<W and 0<=qn[1]<height;assert a['query_nearest_valid'][i,k]==qnv and a['reference_nearest_valid'][i,k]==(0<=rc[0]<W and 0<=rc[1]<height)
  expected_inside=np.array([0<=rc[0]+dx<W and 0<=rc[1]+dy<height for dy in(-1,0,1)for dx in(-1,0,1)])
  inside=a['neighbour_inbounds'][i,k];raw=a['neighbour_depth_raw_uint16'][i,k];nowner=a['neighbour_visible_owner_count'][i,k];own=a['neighbour_own_visible'][i,k];bits=a['neighbour_owner_bits_uint64'][i,k];full=a['neighbour_own_full'][i,k]
  assert np.array_equal(inside,expected_inside)
  assert not(raw[~inside].any()or nowner[~inside].any()or own[~inside].any()or bits[~inside].any()or full[~inside].any())
  for j in range(9):
   assert int(bits[j]).bit_count()==int(nowner[j])
   assert bool(int(bits[j])&(1<<sr['instance_id']))==bool(own[j])
   assert int(bits[j])>>(fr['complete_scene_GT_owner_count'])==0
  positive=inside&(raw>0);unique=inside&(nowner==1)&own;other=inside&(nowner==1)&~own
  cnt=np.array([inside.sum(),positive.sum(),unique.sum(),other.sum(),((nowner>1)&inside).sum(),((nowner==0)&inside).sum(),full.sum()],np.uint8)
  assert np.array_equal(cnt,a['evidence_counts'][i,k])and dict(zip(p['instances']and summary['count_keys'],map(int,cnt)))==log['counts']
  target=np.array(g['reference_camera_xyz_mm'][k]);point_range=float(np.linalg.norm(target));geom,stable=own_geometry(a['geometry_first_hit_range_mm'][i,k],point_range)
  assert geom==a['geometry_state'][i,k]==log['geometry_state']and stable==a['geometry_stable'][i,k]==log['geometry_stable']
  assert np.array_equal(a['geometry_point_minus_first_range_mm'][i,k],point_range-a['geometry_first_hit_range_mm'][i,k],equal_nan=True)
  assert target[2]==log['reference_cameraZ_mm']
  out,reason=own_classify(geom,qi,ri,inside,raw,nowner,own,bits,full,a['depth_scale_to_mm'][i],target[2],a['centre_integer_ray_first_cameraZ_mm'][i,k,0])
  assert out==state[i,k]==log['raw_surface_teacher_state']and reason==a['reason'][i,k]==log['reason'],(i,k,out,reason,state[i,k],a['reason'][i,k])
  for z,ray in enumerate(log['fixed_ray_firsthits']):
   if ray['no_hit']:assert np.isnan(a['geometry_first_hit_range_mm'][i,k,z])and np.isnan(a['geometry_first_hit_cameraZ_mm'][i,k,z])
   else:assert ray['first_range_mm']==a['geometry_first_hit_range_mm'][i,k,z]and ray['first_Z_mm']==a['geometry_first_hit_cameraZ_mm'][i,k,z]
  if log['centre_integer_ray_firsthits']is not None:
   for z,ray in enumerate(log['centre_integer_ray_firsthits']):assert (ray['no_hit']and np.isnan(a['centre_integer_ray_first_cameraZ_mm'][i,k,z]))or(not ray['no_hit']and ray['first_Z_mm']==a['centre_integer_ray_first_cameraZ_mm'][i,k,z])
  reasoncounts[reason]+=1;checks+=1;S_with_missing+=int(out=='S'and not positive[4]);overlap_saved+=int((nowner>1).any());qnearest_only+=int(qi and not qnv);known_qnearest_unavailable+=int(out in('D','E')and not qnv)
counts=dict(collections.Counter(state.ravel().tolist()));assert counts==summary['states']==dict(D=458,E=220,S=1208,U=1314)
assert dict(reasoncounts)==summary['reasons']
by_split={s:dict(collections.Counter(state[np.array([r['original_split']==s for r in inputs])].ravel().tolist()))for s in('train','val')}
byobj={str(oid):dict(collections.Counter(state[np.array([r['source_instance']['object_id']==oid for r in refs])].ravel().tolist()))for oid in(16,21)}
assert by_split==summary['by_original_split']and byobj==summary['by_object']
result=dict(completed=True,PASS=True,blockers=[],audit_code=bind(__file__),corpus_seal=bind(sealpath),protocol=seal['protocol'],builder=seal['code'],entry=seal['entry'],preparation_review=bind(V/'independent_teacher_preparation_review/REVIEW.json'),state_counts=counts,by_original_split=by_split,by_object=byobj,reason_counts=dict(reasoncounts),all3200_state_and_reason_own_math_exact=True,all3200_numeric_log_and_packet_consistent=True,all_ID_q_L_fold_and_fixed_slot_copies_exact=True,instances_by_original_split=dict(train=300,val=100),canonical_folds=dict(TRAIN=300,VALIDATION=100),all_complete_owner_sets_metadata_exact=True,owner_bits_popcount_and_own_instance_membership_all28800_exact=True,centre_depth_scale_and_S_offimage_priority_exact=True,external_target_D0_E1_else_minus1_exact=True,unknown_negative_count=0,S_known_out_of_external_task=1208,diagnostics=dict(S_missing_centre_depth_retained=S_with_missing,slots_with_any_owner_overlap=overlap_saved,query_continuous_valid_but_nearest_invalid=qnearest_only,known_D_E_with_nearest_query_unavailable=known_qnearest_unavailable,classification_calls_replayed=checks,unique_saved_frames=len(frames)),source_VAL_development_exposed=True,training_admission=False,limits=['Replay uses only recorded numeric evidence and metadata; source PNG/mesh ray correctness and geometry are not independently rerendered or decoded.','S1208 is known geometry outside learned external D/E; teacherU1314 remains unknown. Query nearest validity is a separate factual input diagnostic and does not retroactively change the pinned teacher.','Conditional surface proxy and GT-jitter queries; not human/native pallet junction gold.'],new_source_pixels_mesh_ray_model_fit_private_V112_PnP=0)
with(H/'REVIEW.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(PASS=True,review=bind(H/'REVIEW.json'),counts=counts,by_split=by_split,diagnostics=result['diagnostics'])))
