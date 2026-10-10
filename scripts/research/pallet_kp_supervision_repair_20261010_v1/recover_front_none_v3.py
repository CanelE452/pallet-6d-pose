"""Recover missing first-surface evidence for fixed existing source queries.

The no-ray preparation and immutable feature cache remain unchanged.  A
separately frozen plan bounds the necessary casts; infinity never means NONE.
"""
from pathlib import Path
from collections import Counter
import gzip,json,hashlib,time,sys,os
import numpy as np
import cv2
P=Path(__file__).parent; PREP=P/'full_source_preparation';V1=P/'depth_recovery';V2=P/'depth_recovery_v2';OUT=P/'depth_recovery_v3'
OLD=Path('/dev/shm/pallet-observation-worktree-20261009')
DOC=OLD/'_docs/experiments/pallet_kp_difficulty_20261010_v1'
MESH=Path('/dev/shm/pallet-observation-private-20261009/actual_mesh/scene.usd.npz')
sys.path.insert(0,str(OLD));os.environ.setdefault('PALLET_SOURCE_ROOT','/home/minjae/Documents/github/pallet-pose')
from scripts.research.pallet_observation_refiner_20261009_v1 import common as C,source_audit as S
from scripts.research.pallet_observation_refiner_20261009_v1.solver import project
from scripts.research.pallet_kp_difficulty_20261010_v1.source_ceiling import query_geometry
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows(p):return [json.loads(x) for x in gzip.open(p,'rt')]
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def save(p,xs):
 with gzip.open(p,'wt',compresslevel=6) as f:
  for x in xs:f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def protected_paths():
 cache=Path('/dev/shm/pallet-observation-private-20261009/learned_cache')
 mapping={}
 def add(alias,path):
  assert path.exists(),str(path);mapping[alias]=path
 for name in ['actual_wire_proposal.py','WIRE_TARGET_PROTOCOL.json','CORRECTED_TARGET_PROPOSAL_ROWS.jsonl.gz','CORRECTED_TARGET_PROPOSAL_VALIDATION.json','CORRECTED_TARGET_SCHEMA_CLARIFICATION.json','full_source_prepare.py','float64_surface_certificate.py','FLOAT64_SURFACE_PROTOCOL.json','FLOAT64_SURFACE_ROWS.jsonl.gz','FLOAT64_SURFACE_VALIDATION.json','ENCLOSURE_PROTOCOL.json','ENCLOSURE_ROWS.jsonl.gz','ENCLOSURE_VALIDATION.json','triangle_ownership_audit.py','TRIANGLE_OWNERSHIP_PROTOCOL.json','TRIANGLE_OWNERSHIP_ROWS.jsonl.gz','TRIANGLE_OWNERSHIP_AUDIT.json','WIRE_FACE_WITNESSES.json','WIRE_FACE_WITNESSES_PROTOCOL.json','export_wire_faces.py','recover_front_none.py','recover_front_none_v2.py','recover_front_none_v3.py']:
  add('PRIVATE_GATE/'+name,P/name)
 for name in ['FULL_SOURCE_PREPARATION_PROTOCOL.json','FULL_SOURCE_PREPARATION.json','FULL_SOURCE_TARGET_ROWS.jsonl.gz','PREPARED_TARGETS.npz','FULL_SOURCE_ARRAY_VERIFICATION.json']:add('NO_RAY_PREPARATION/'+name,PREP/name)
 for name in ['DEPTH_RECOVERY_PROTOCOL.json','DEPTH_RECOVERY_PLAN_ROWS.jsonl.gz']:add('V1_HISTORY/'+name,V1/name)
 add('V2_HISTORY/DEPTH_RECOVERY_V2_PROTOCOL.json',V2/'DEPTH_RECOVERY_V2_PROTOCOL.json')
 for name in ['CACHE_MANIFEST.json','features.npy','order.npy','lo.npy','hi.npy','weight.npy','valid.npy']:add('ORIGINAL_CACHE/'+name,cache/name)
 for name in ['SOURCE_CEILING_ROWS.jsonl.gz','SOURCE_RAY_PROTOCOL.json','SOURCE_RAY_VALIDATION_ROWS.jsonl.gz','SOURCE_RAY_VALIDATION.json']:add('ORIGINAL_DIAGNOSTICS/'+name,DOC/name)
 oldcode=OLD/'scripts/research/pallet_observation_refiner_20261009_v1'
 for name in ['training.py','model.py','solver.py','source_audit.py','common.py']:add('ORIGINAL_CODE/'+name,oldcode/name)
 for name in ['source_ceiling.py','source_ray_validation.py']:add('ORIGINAL_DIFFICULTY_CODE/'+name,OLD/'scripts/research/pallet_kp_difficulty_20261010_v1'/name)
 add('ORIGINAL_ACTUAL_MESH/scene.usd.npz',MESH);add('ORIGINAL_ACTUAL_MESH/scene.usd.json',MESH.with_suffix('.json'));meta=json.loads(MESH.with_suffix('.json').read_text());add('SOURCE/'+meta['asset']['path'],C.ROOT/meta['asset']['path'])
 for frame in rows(V1/'DEPTH_RECOVERY_PLAN_ROWS.jsonl.gz'):
  for name in ['source_annotation','source_visible_mask']:
   b=frame[name];path=C.ROOT/b['path'];assert sha(path)==b['sha256'];add('SOURCE/'+b['path'],path)
 return mapping
def snapshot(mapping):return {alias:dict(sha256=sha(path),bytes=path.stat().st_size) for alias,path in mapping.items()}
def freeze_v3():
 OUT.mkdir(exist_ok=True);assert not(OUT/'DEPTH_RECOVERY_V3_PROTOCOL.json').exists();assert not(V1/'RECOVERED_DEPTH_ROWS.jsonl.gz').exists(),'v1 must not have cast anything';assert not(V2/'RECOVERED_DEPTH_ROWS.jsonl.gz').exists(),'v2 must not have cast anything'
 original=json.loads((V1/'DEPTH_RECOVERY_PROTOCOL.json').read_text());assert sha(P/'recover_front_none.py')==original['inputs']['recover_front_none.py']['sha256'];plan=V1/'DEPTH_RECOVERY_PLAN_ROWS.jsonl.gz';assert sha(plan)==original['inputs'][plan.name]['sha256']
 prior=json.loads((V2/'DEPTH_RECOVERY_V2_PROTOCOL.json').read_text());mapping=protected_paths();fixed=snapshot(mapping);assert all(fixed[k]==v for k,v in prior['fixed_inputs'].items())
 value=dict(original,schema='fixed_missing_front_surface_depth_recovery_protocol_v3',v1_history=dict(code_sha256=sha(P/'recover_front_none.py'),protocol_sha256=sha(V1/'DEPTH_RECOVERY_PROTOCOL.json'),plan_sha256=sha(plan),v1_casts_executed=0),v2_history=dict(code_sha256=sha(P/'recover_front_none_v2.py'),protocol_sha256=sha(V2/'DEPTH_RECOVERY_V2_PROTOCOL.json'),v2_casts_executed=0),
  population_unchanged=True,plan_reused_without_query_refreeze=True,created_before_first_cast=True,
  correctness_guards='NONE requires finite t_hit>=0; 0<=primitive_id<len(actual triangles); finite stored float32 barycentrics converted to double with u>=0,v>=0,u+v<=1 exactly; finite nonzero primitive normal; BOTH legacy ray parameter minus sourceZ and recomputed source-R camera-Z minus sourceZ < -unchanged original tolerance. Any ambiguous witness is IGNORE.',
  ray_parameter_units='t_hit_original_ray_parameter is the original Open3D float32 ray parameter. The original inverse-camera R.T policy approximates camera Z because source R has finite orthogonality error. It is not the exact rational inverse-camera certificate used for owned-wire POSITIVE.',
  independent_camera_guard='Actual hit point = float64(stored float32 origin)+float64(t_hit)*float64(stored float32 direction). Source-camera Z = (R@hit_point+t)[2]; require it independently precedes source_camera_Z by more than original.001diagonal. Record both depths, deltas and their difference without changing geometry or tolerance.',
  fixed_inputs=fixed,input_protection='All fixed helper/code/protocol/rows/mesh/original feature/order/target and planned source annotation/mask files SHA256 match protocol before casting and again after casts; abort readiness on change.',
  array_invariants='All no-ray prepared POSITIVE target bytes, every non-recovered target byte, all source-test target bytes, source_index/partitions and every array dtype unchanged. Only witnessed original train/cal NONE currently IGNORE may become valid NONE.',
  first_surface_evidence='Valid actual-mesh finite first-surface witness must precede intended source depth in BOTH original ray-parameter policy and independently recomputed source-camera-Z, each using the unchanged .001diagonal tolerance.')
 write(OUT/'DEPTH_RECOVERY_V3_PROTOCOL.json',value);print(json.dumps({k:v for k,v in value.items() if k!='fixed_inputs'}),flush=True)
def freeze():
 cv2.setNumThreads(1);OUT.mkdir(exist_ok=True);protocol=OUT/'DEPTH_RECOVERY_PROTOCOL.json';assert not protocol.exists()
 begin=time.monotonic();source={x['index']:x for x in rows(DOC/'SOURCE_CEILING_ROWS.jsonl.gz')};prepared=rows(PREP/'FULL_SOURCE_TARGET_ROWS.jsonl.gz');families=S.selected_families();plan=[];counts=Counter();split=Counter()
 for frame in prepared[:896]:
  index=frame['index'];paths=S.locate(families[index]);g=S.geometry(families[index],C.read(paths['label']));assert g['asset']=='scene.usd'
  for k,real in [('K',g['K']),('R',g['R']),('t',g['t']),('dimensions',g['dims'])]:assert np.array_equal(np.array(frame[k]),np.array(real)),k
  centers,normals,qvalid=query_geometry(source[index]['frozen_selected_points']);uv=project(g['X'],g['R'],g['t'],g['K']);visible=cv2.imread(str(paths['visible']),0);assert visible is not None and list(visible.shape)==g['hw'];qq=[]
  for q in frame['queries']:
   if not(q['original_target']=='NONE' and q['proposed_target']=='IGNORE'):continue
   counts['residual_original_NONE']+=1;i=q['query'];edge=q['edge']
   if edge not in source[index]['physical_edge_ids'] or not qvalid[i]:counts['unsupported_or_invalid_unchanged_IGNORE']+=1;continue
   a,b=S.EDGES[edge];matrix=np.column_stack([normals[i],-(uv[b]-uv[a])])
   if abs(np.linalg.det(matrix))<1e-8:counts['degenerate_unchanged_IGNORE']+=1;continue
   offset,fraction=np.linalg.solve(matrix,uv[a]-centers[i]);position=centers[i]+offset*normals[i]
   if not(0<=fraction<=1 and abs(offset)<=32):counts['outside_segment_search_unchanged_IGNORE']+=1;continue
   x,y=position;h,w=visible.shape
   if not(0<=x<w and 0<=y<h):counts['outside_image_unchanged_IGNORE']+=1;continue
   ix,iy=int(round(x-.5)),int(round(y-.5));support=bool((visible[max(0,iy-1):min(h,iy+2),max(0,ix-1):min(w,ix+2)]>127).any())
   if not support:counts['supplied_mask_absence_unchanged_IGNORE']+=1;continue
   za=(g['X'][a]@g['R'].T+g['t'])[2];zb=(g['X'][b]@g['R'].T+g['t'])[2];u=(fraction/zb)/((1-fraction)/za+fraction/zb);X=(1-u)*g['X'][a]+u*g['X'][b];expected=float((X@g['R'].T+g['t'])[2])
   qq.append(dict(query=i,edge=edge,source_X=X.tolist(),position=position.tolist(),source_camera_Z=expected,offset_px=float(offset),fraction=float(fraction),mask_kernel=[ix,iy],physical_tolerance_evidence='Unchanged cached valid NONE plus eligible in-segment/search query implies the original closest-point tolerance passed; unsupported whole-edge IDs excluded.'))
   counts['planned_rays']+=1;split[frame['partition']]+=1
  if qq:plan.append(dict(id=frame['id'],index=index,partition=frame['partition'],K=frame['K'],R=frame['R'],t=frame['t'],dimensions=frame['dimensions'],depth_tolerance_m=float(.001*np.linalg.norm(g['dims'])),source_annotation=C.binding(paths['label']),source_visible_mask=C.binding(paths['visible']),queries=qq))
 save(OUT/'DEPTH_RECOVERY_PLAN_ROWS.jsonl.gz',plan)
 paths=[Path(__file__),PREP/'FULL_SOURCE_PREPARATION_PROTOCOL.json',PREP/'FULL_SOURCE_PREPARATION.json',PREP/'FULL_SOURCE_TARGET_ROWS.jsonl.gz',PREP/'PREPARED_TARGETS.npz',DOC/'SOURCE_CEILING_ROWS.jsonl.gz',MESH,Path(S.__file__).parent/'training.py',OUT/'DEPTH_RECOVERY_PLAN_ROWS.jsonl.gz']
 value=dict(schema='fixed_missing_front_surface_depth_recovery_protocol_v1',created_before_first_cast=True,inputs={x.name:dict(sha256=sha(x),bytes=x.stat().st_size) for x in paths},
  population='Only original train/cal valid NONE currently conservative IGNORE, physical source edge, original query/segment/search/in-image/unchanged supplied-mask support. Source-test and existing proposed POSITIVE untouched.',
  counts=dict(counts),planned_split_rays=dict(split),max_rays=counts['planned_rays'],max_scene_builds=len(plan),closest_points=0,
  exact_ray='Reproduce original float32 ray from original normal/semantic-segment intersection, origin=-R.T*t and inverse-K direction@R; one ray per planned query; no normal offsets or sweep.',
  first_surface_evidence='Actual scene.usd mesh, primitive ID and barycentric hit retained. Finite t_hit-source_camera_Z < -0.001*physical_diagonal confirms genuine front-surface NONE under the unchanged original depth policy.',
  numerical_infinity='Infinity, finite non-front, invalid primitive, and other ambiguity remain IGNORE; never infinity -> NONE. Already geometrically certified owned-wire positives take precedence and are not recast.',
  no_match_limits='Own-mesh first-surface absence at the intended physical point under the original policy; no claim that mesh/mask alone certifies every photometric boundary. Mask absence/out-of-search remain conservative IGNORE in this protocol.',
  unchanged_source_features=True,source_cache_mutated=False,family_order_unchanged=True,split_counts=dict(train=768,calibration=128,source_test=128),source_test_arrays_unchanged=True,
  outputs='Recovered raw finite/front-depth evidence plus separate new lo/hi/weight/valid target arrays and rows; no original overwrite.',
  new_RGB=0,new_heads=0,new_detector=0,new_PnP=0,new_training_updates=0,freeze_seconds=time.monotonic()-begin)
 write(protocol,value);print(json.dumps(value),flush=True)
def run():
 import open3d as o3d
 cv2.setNumThreads(1);begin=time.monotonic();protocol=OUT/'DEPTH_RECOVERY_V3_PROTOCOL.json';p=json.loads(protocol.read_text());mapping=protected_paths();before=snapshot(mapping);assert before==p['fixed_inputs'];write(OUT/'FIXED_INPUT_PROTECTION_BEFORE.json',dict(all_match_protocol=True,bindings=before,new_rays=0));planpath=V1/'DEPTH_RECOVERY_PLAN_ROWS.jsonl.gz';assert sha(planpath)==p['v1_history']['plan_sha256'];assert not(OUT/'RECOVERED_DEPTH_ROWS.jsonl.gz').exists()
 with np.load(MESH) as z:v=z['vertices'];tri=z['triangles']
 records=[];counts=Counter();front=set();scene_count=0;ray_count=0;metric_differences=[]
 for frame in rows(planpath):
  scene=S.ray_scene(v,tri,frame['dimensions']);scene_count+=1;xy=np.array([q['position'] for q in frame['queries']]);K=np.array(frame['K']);R=np.array(frame['R']);t=np.array(frame['t']);origin=-R.T@t;direction=np.column_stack([(xy[:,0]-K[0,2])/K[0,0],(xy[:,1]-K[1,2])/K[1,1],np.ones(len(xy))])@R
  ray=np.column_stack([np.broadcast_to(origin,direction.shape),direction]).astype(np.float32);hit=scene.cast_rays(o3d.core.Tensor(ray));ray_count+=len(ray)
  depth=hit['t_hit'].numpy();pid=hit['primitive_ids'].numpy();uv=hit['primitive_uvs'].numpy();normal=hit['primitive_normals'].numpy();qq=[]
  for j,q in enumerate(frame['queries']):
   finite=bool(np.isfinite(depth[j]));validprimitive=bool(0<=int(pid[j])<len(tri));bary=uv[j].astype(np.float64);nn=normal[j].astype(np.float64);norm=float(np.linalg.norm(nn));world=ray[j,:3].astype(np.float64)+float(depth[j])*ray[j,3:].astype(np.float64) if finite else None;camera_Z=float((R@world+t)[2]) if finite else None;camera_finite=bool(finite and np.isfinite(world).all() and np.isfinite(camera_Z));guards=dict(finite_nonnegative_depth=bool(finite and depth[j]>=0),valid_primitive_index=validprimitive,finite_closed_triangle_barycentrics=bool(np.isfinite(bary).all() and (bary>=0).all() and bary.sum()<=1),finite_nonzero_normal=bool(np.isfinite(nn).all() and np.isfinite(norm) and norm>0),finite_recomputed_camera_hit=bool(camera_finite));delta=float(depth[j])-q['source_camera_Z'] if finite else None;camera_delta=camera_Z-q['source_camera_Z'] if camera_finite else None;legacy_front=bool(finite and delta < -frame['depth_tolerance_m']);camera_front=bool(camera_finite and camera_delta < -frame['depth_tolerance_m']);negative=bool(all(guards.values()) and legacy_front and camera_front);state='NONE' if negative else 'IGNORE';counts[frame['partition']+'_'+state]+=1;counts['finite' if finite else 'infinite']+=1
   if camera_finite:metric_differences.append(camera_Z-float(depth[j]))
   if legacy_front!=camera_front:counts['legacy_vs_recomputed_front_disagreement']+=1
   for key,value in guards.items():
    if not value:counts['invalid_'+key]+=1
   if negative:front.add((frame['index'],q['query']))
   qq.append(dict(q,ray_float32=ray[j].tolist(),t_hit_original_ray_parameter=float(depth[j]) if finite else None,actual_hit_point_world=world.tolist() if camera_finite else None,recomputed_hit_source_camera_Z=camera_Z if camera_finite else None,recomputed_camera_Z_minus_original_ray_parameter_m=camera_Z-float(depth[j]) if camera_finite else None,legacy_original_ray_parameter_depth_policy='Original R.T approximate-camera-Z convention; not exact rational inverse-camera certificate.',primitive_id=int(pid[j]) if validprimitive else None,primitive_uv=[float(x) if np.isfinite(x) else None for x in bary] if finite else None,primitive_normal=[float(x) if np.isfinite(x) else None for x in nn] if finite else None,witness_guards=guards,original_ray_parameter_minus_target_camera_Z_m=delta,recomputed_camera_Z_minus_target_camera_Z_m=camera_delta,legacy_parameter_front=legacy_front,recomputed_camera_front=camera_front,proved_front_surface_NONE=negative,recovered_target=state))
  records.append(dict(frame,queries=qq,new_rays=len(ray),scene_build_index=scene_count))
  if scene_count%32==0:write(OUT/'DEPTH_RECOVERY_PROGRESS.json',dict(frames=scene_count,rays=ray_count,counts=dict(counts),seconds=time.monotonic()-begin));print('DEPTH_RECOVERY',scene_count,p['max_scene_builds'],ray_count,round(time.monotonic()-begin,2),flush=True)
 assert scene_count==p['max_scene_builds'] and ray_count==p['max_rays'];save(OUT/'RECOVERED_DEPTH_ROWS.jsonl.gz',records);after=snapshot(mapping);write(OUT/'FIXED_INPUT_PROTECTION_AFTER.json',dict(all_match_protocol=after==p['fixed_inputs'],all_match_before=after==before,bindings=after,new_rays=ray_count));assert after==before==p['fixed_inputs']
 arrays={k:a.copy() for k,a in np.load(PREP/'PREPARED_TARGETS.npz').items()};original=rows(PREP/'FULL_SOURCE_TARGET_ROWS.jsonl.gz');totals={s:Counter() for s in ['train','calibration','source_test']}
 for frame in original:
  for q in frame['queries']:
   q['depth_recovery_changed_proposal']=(frame['index'],q['query']) in front
   if q['depth_recovery_changed_proposal']:
    assert q['original_target']=='NONE' and q['proposed_target']=='IGNORE';q.update(proposed_target='NONE',proposal_label_changed=False,lo=65,hi=65,weight=0.,reason='Finite actual first surface precedes intended physical source point beyond unchanged original depth tolerance; new fixed recovery witness.',depth_recovery_binding=dict(frame_index=frame['index'],query=q['query']))
    i,j=frame['index'],q['query'];arrays['lo'][i,j]=65;arrays['hi'][i,j]=65;arrays['weight'][i,j]=0.;arrays['valid'][i,j]=True
   totals[frame['partition']][q['proposed_target']]+=1
 save(OUT/'READY_SOURCE_TARGET_ROWS.jsonl.gz',original);np.savez_compressed(OUT/'READY_PREPARED_TARGETS.npz',**arrays)
 with np.load(PREP/'PREPARED_TARGETS.npz') as z:old={k:z[k] for k in z.files}
 pos=old['valid']&(old['lo']<65)
 changed=np.zeros((1024,84),bool)
 for i,j in front:changed[i,j]=True
 assert all(arrays[k][pos].tobytes()==old[k][pos].tobytes() for k in ['lo','hi','weight','valid']);assert all(arrays[k][~changed].tobytes()==old[k][~changed].tobytes() for k in ['lo','hi','weight','valid']);assert all(arrays[k][896:].tobytes()==old[k][896:].tobytes() for k in ['lo','hi','weight','valid']);assert all(arrays[k].dtype==old[k].dtype for k in old);assert all(arrays[k].tobytes()==old[k].tobytes() for k in ['source_index','partitions']);ready=all(totals[s]['POSITIVE']>0 and totals[s]['NONE']>0 for s in totals)
 camera_metrics=dict(count=len(metric_differences),mean=float(np.mean(metric_differences)) if metric_differences else None,max_absolute=float(np.max(np.abs(metric_differences))) if metric_differences else None,median_absolute=float(np.median(np.abs(metric_differences))) if metric_differences else None,p90_absolute=float(np.percentile(np.abs(metric_differences),90)) if metric_differences else None)
 result=dict(schema='missing_front_surface_depth_recovery_v3',complete=True,source_supervision_ready=ready,readiness_limits='Source loss readiness requires owned physical POSITIVE and actual finite-front NONE in every split, with unsupported/ambiguous queries IGNORE. This is mixed correspondence/no-match source loss validity, not real transfer, graph sufficiency, retraining approval or success.',counts=dict(counts),target_counts={s:dict(n) for s,n in totals.items()},scene_builds=scene_count,cast_rays_API_calls=scene_count,new_rays=ray_count,new_closest_points=0,new_RGB=0,new_heads=0,new_detector=0,new_PnP=0,new_training_updates=0,source_cache_mutated=False,source_test_arrays_unchanged=True,
  inputs=p['inputs'],fixed_input_protection=dict(files=len(mapping),before_after_exact=True,before_receipt_sha256=sha(OUT/'FIXED_INPUT_PROTECTION_BEFORE.json'),after_receipt_sha256=sha(OUT/'FIXED_INPUT_PROTECTION_AFTER.json')),array_invariants=dict(all_POSITIVE_target_bytes_unchanged=True,all_non_recovered_target_bytes_unchanged=True,all_source_test_target_bytes_unchanged=True,source_index_and_partitions_bytes_unchanged=True,all_array_dtypes_unchanged=True),camera_vs_original_parameter_m=camera_metrics,protocol_sha256=sha(protocol),raw_depth_rows_sha256=sha(OUT/'RECOVERED_DEPTH_ROWS.jsonl.gz'),ready_rows_sha256=sha(OUT/'READY_SOURCE_TARGET_ROWS.jsonl.gz'),ready_arrays_sha256=sha(OUT/'READY_PREPARED_TARGETS.npz'),code_sha256=sha(Path(__file__)),wall_seconds=time.monotonic()-begin)
 write(OUT/'DEPTH_RECOVERY_VALIDATION.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 assert len(sys.argv)==2 and sys.argv[1] in ['--freeze-v3','--run'];freeze_v3() if sys.argv[1]=='--freeze-v3' else run()
