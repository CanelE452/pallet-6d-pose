"""Recover missing first-surface evidence for fixed existing source queries.

The no-ray preparation and immutable feature cache remain unchanged.  A
separately frozen plan bounds the necessary casts; infinity never means NONE.
"""
from pathlib import Path
from collections import Counter
import gzip,json,hashlib,time,sys,os
import numpy as np
import cv2
P=Path(__file__).parent; PREP=P/'full_source_preparation';OUT=P/'depth_recovery'
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
 cv2.setNumThreads(1);begin=time.monotonic();protocol=OUT/'DEPTH_RECOVERY_PROTOCOL.json';p=json.loads(protocol.read_text());assert sha(Path(__file__))==p['inputs']['recover_front_none.py']['sha256'];planpath=OUT/'DEPTH_RECOVERY_PLAN_ROWS.jsonl.gz';assert sha(planpath)==p['inputs'][planpath.name]['sha256'];assert not(OUT/'RECOVERED_DEPTH_ROWS.jsonl.gz').exists()
 with np.load(MESH) as z:v=z['vertices'];tri=z['triangles']
 records=[];counts=Counter();front=set();scene_count=0;ray_count=0
 for frame in rows(planpath):
  scene=S.ray_scene(v,tri,frame['dimensions']);scene_count+=1;xy=np.array([q['position'] for q in frame['queries']]);K=np.array(frame['K']);R=np.array(frame['R']);t=np.array(frame['t']);origin=-R.T@t;direction=np.column_stack([(xy[:,0]-K[0,2])/K[0,0],(xy[:,1]-K[1,2])/K[1,1],np.ones(len(xy))])@R
  ray=np.column_stack([np.broadcast_to(origin,direction.shape),direction]).astype(np.float32);hit=scene.cast_rays(o3d.core.Tensor(ray));ray_count+=len(ray)
  depth=hit['t_hit'].numpy();pid=hit['primitive_ids'].numpy();uv=hit['primitive_uvs'].numpy();normal=hit['primitive_normals'].numpy();qq=[]
  for j,q in enumerate(frame['queries']):
   finite=bool(np.isfinite(depth[j]));validprimitive=int(pid[j])!=4294967295;delta=float(depth[j])-q['source_camera_Z'] if finite else None;negative=bool(finite and validprimitive and delta < -frame['depth_tolerance_m']);state='NONE' if negative else 'IGNORE';counts[frame['partition']+'_'+state]+=1;counts['finite' if finite else 'infinite']+=1
   if negative:front.add((frame['index'],q['query']))
   qq.append(dict(q,ray_float32=ray[j].tolist(),t_hit_camera_Z=float(depth[j]) if finite else None,primitive_id=int(pid[j]) if validprimitive else None,primitive_uv=uv[j].tolist() if finite else None,primitive_normal=normal[j].tolist() if finite else None,hit_minus_original_target_Z_m=delta,proved_front_surface_NONE=negative,recovered_target=state))
  records.append(dict(frame,queries=qq,new_rays=len(ray),scene_build_index=scene_count))
  if scene_count%32==0:write(OUT/'DEPTH_RECOVERY_PROGRESS.json',dict(frames=scene_count,rays=ray_count,counts=dict(counts),seconds=time.monotonic()-begin));print('DEPTH_RECOVERY',scene_count,p['max_scene_builds'],ray_count,round(time.monotonic()-begin,2),flush=True)
 assert scene_count==p['max_scene_builds'] and ray_count==p['max_rays'];save(OUT/'RECOVERED_DEPTH_ROWS.jsonl.gz',records)
 arrays={k:a.copy() for k,a in np.load(PREP/'PREPARED_TARGETS.npz').items()};original=rows(PREP/'FULL_SOURCE_TARGET_ROWS.jsonl.gz');totals={s:Counter() for s in ['train','calibration','source_test']}
 for frame in original:
  for q in frame['queries']:
   q['depth_recovery_changed_proposal']=(frame['index'],q['query']) in front
   if q['depth_recovery_changed_proposal']:
    assert q['original_target']=='NONE' and q['proposed_target']=='IGNORE';q.update(proposed_target='NONE',proposal_label_changed=False,lo=65,hi=65,weight=0.,reason='Finite actual first surface precedes intended physical source point beyond unchanged original depth tolerance; new fixed recovery witness.',depth_recovery_binding=dict(frame_index=frame['index'],query=q['query']))
    i,j=frame['index'],q['query'];arrays['lo'][i,j]=65;arrays['hi'][i,j]=65;arrays['weight'][i,j]=0.;arrays['valid'][i,j]=True
   totals[frame['partition']][q['proposed_target']]+=1
 save(OUT/'READY_SOURCE_TARGET_ROWS.jsonl.gz',original);np.savez_compressed(OUT/'READY_PREPARED_TARGETS.npz',**arrays)
 old=np.load(PREP/'PREPARED_TARGETS.npz');assert all(np.array_equal(arrays[k][896:],old[k][896:]) for k in ['lo','hi','weight','valid']);ready=all(totals[s]['POSITIVE']>0 and totals[s]['NONE']>0 for s in totals)
 result=dict(schema='missing_front_surface_depth_recovery_v1',complete=True,source_supervision_ready=ready,readiness_limits='Fixed source targets contain owned physical POSITIVE and actual finite-front NONE in every split, with unsupported/ambiguous queries IGNORE. This establishes mixed correspondence/no-match source loss validity, not real transfer, graph sufficiency, retraining approval or success.',counts=dict(counts),target_counts={s:dict(n) for s,n in totals.items()},scene_builds=scene_count,cast_rays_API_calls=scene_count,new_rays=ray_count,new_closest_points=0,new_RGB=0,new_heads=0,new_detector=0,new_PnP=0,new_training_updates=0,source_cache_mutated=False,source_test_arrays_unchanged=True,
  inputs=p['inputs'],protocol_sha256=sha(protocol),raw_depth_rows_sha256=sha(OUT/'RECOVERED_DEPTH_ROWS.jsonl.gz'),ready_rows_sha256=sha(OUT/'READY_SOURCE_TARGET_ROWS.jsonl.gz'),ready_arrays_sha256=sha(OUT/'READY_PREPARED_TARGETS.npz'),code_sha256=sha(Path(__file__)),wall_seconds=time.monotonic()-begin)
 write(OUT/'DEPTH_RECOVERY_VALIDATION.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 assert len(sys.argv)==2 and sys.argv[1] in ['--freeze','--run'];freeze() if sys.argv[1]=='--freeze' else run()
