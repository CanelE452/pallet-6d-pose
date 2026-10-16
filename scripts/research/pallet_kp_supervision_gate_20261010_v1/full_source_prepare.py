"""Prepare source targets with the frozen actual-wire rule; no ray casts.

The unchanged frozen rule is extracted from the immutable128 proposal code.
Only population/status plumbing changes: unavailable negative evidence is
IGNORE. The source-test metadata and original feature cache are preserved.
"""
import ast,copy,gzip,hashlib,importlib.util,json,os,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import cv2
import open3d as o3d
P=Path(__file__).parent;OUT=P/'full_source_preparation';FIXED=P/'actual_wire_proposal.py'
OLD=Path('/dev/shm/pallet-observation-worktree-20261009');DOC=OLD/'_docs/experiments/pallet_kp_difficulty_20261010_v1'
CACHE=Path('/dev/shm/pallet-observation-private-20261009/learned_cache');MESH=Path('/dev/shm/pallet-observation-private-20261009/actual_mesh/scene.usd.npz')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def loadrows(p):return [json.loads(x) for x in gzip.open(p,'rt')]
def frozen_frame_function():
 spec=importlib.util.spec_from_file_location('frozen_actual_wire_proposal',FIXED);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
 tree=ast.parse(FIXED.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 loop=[n for n in main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='frame'][-1]
 body=copy.deepcopy(loop.body)
 for n in ast.walk(ast.Module(body=body,type_ignores=[])):
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='category':n.value=ast.parse("q.get('full_proposal_category')",mode='eval').body
 prefix=ast.parse('counts=defaultdict(Counter)\nresult=[]\nstagecounts=Counter()\nframebounds=[]').body
 suffix=ast.parse('return result,counts,stagecounts,framebounds').body
 args=['frame','sources','vertices','tri','face_edge_ids','owners','first','unique_edges']
 func=ast.FunctionDef(name='frozen_frame',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=k) for k in args],kwonlyargs=[],kw_defaults=[],defaults=[]),body=prefix+body+suffix,decorator_list=[])
 module=ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[]));env=dict(base.__dict__);exec(compile(module,str(FIXED)+':unaltered_wire_math','exec'),env)
 return base,env['frozen_frame'],hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest()
def proto():
 OUT.mkdir(exist_ok=True);p=OUT/'FULL_SOURCE_PREPARATION_PROTOCOL.json';assert not p.exists()
 paths=[FIXED,P/'WIRE_TARGET_PROTOCOL.json',P/'CORRECTED_TARGET_PROPOSAL_ROWS.jsonl.gz',DOC/'SOURCE_CEILING_ROWS.jsonl.gz',DOC/'SOURCE_RAY_VALIDATION_ROWS.jsonl.gz',MESH,CACHE/'CACHE_MANIFEST.json',CACHE/'features.npy',CACHE/'order.npy',Path(__file__)]
 write(p,dict(schema='same_frozen_actual_wire_full_source_preparation_protocol_v1',created_before_new_arithmetic=True,inputs={x.name:dict(sha256=sha(x),bytes=x.stat().st_size) for x in paths},
  fixed_families=1024,split_counts=dict(train=768,calibration=128,source_test=128),selection='identical original family ordering, original-only P0, no augmentation or scene movement',
  earlier_raw_depth='available onlysource_test128; oldtrain/cal arrays have no t_hit or primitive evidence',
  no_match_policy='retain only earlier actual front-surface proof; unknown NONE becomesIGNORE unless actual owned-wire/enclosure/mask gates positively certify a correspondence; unsupported physical edges remainIGNORE',
  source_test_preserved='reuse immutable original source-test metadata; frozen128 proposal remains untouched; check selected fields exactly against original frozenproposal',
  same_wire_rule='AST extract immutable fixed perframe wire/query/depth/mask math; only category plumbing changed, independent frozen source hash bound',
  closest_point_preparation='896singlethreadCPUscene builds, at most896*84=75264closest points; original float32 mesh/queries solely for closestprimitive candidate IDs; no cast_rays',
  target_math='float64 actual wire, rational queryintersection/perspective interpolation/enclosure, unchanged original physical/depth tolerances and suppliedmaskkernel',
  unchanged_features='originalFP16features1024x84x28x65 reused by binding; no regeneration/copy',
  outputs='targetlo/hi/weight/valid arrays +compact source rows/status/proof traces; no training',
  max_scene_builds=896,max_query_closest_points=75264,new_rays=0,new_heads=0,new_detector=0,new_PnP=0,new_RGB=0,new_training_updates=0,
  ready_for_learning=False,readiness_gate='complete correspondence/no-match loss supervision requires genuine NONE in training/calibration; unavailable earlier depths must be reported, never fabricated'))
 return p
def query_frame(index,source,family,cache,vertices,tri,common,S,stats):
 paths=S.locate(family);ann=common.read(paths['label']);g=S.geometry(family,ann);assert g['asset']=='scene.usd'
 from scripts.research.pallet_observation_refiner_20261009_v1.solver import project
 from scripts.research.pallet_kp_difficulty_20261010_v1.source_ceiling import query_geometry
 center,normal,qvalid=query_geometry(source['frozen_selected_points']);uv=project(g['X'],g['R'],g['t'],g['K']);visible=cv2.imread(str(paths['visible']),0);assert visible is not None and list(visible.shape)==g['hw']
 scene=S.ray_scene(vertices,tri,g['dims']);stats['scene_builds']+=1;qq=[];sourceids=[];xs=[]
 for i in range(84):
  edge=i//7;a,b=S.EDGES[edge];target='IGNORE' if not source['targets']['valid'][i] else 'POSITIVE' if source['targets']['lo'][i]<65 else 'NONE'
  q=dict(query=i,edge=edge,cached_target=target,predicted_role=source['predicted_role_query_ids'][i],query_valid=bool(qvalid[i]),ray_eligible=False,
   earlier_raw_depth_available=False,original_finite=None,original_hit_in_front_of_source=None,whole_edge_has_physical_sample=edge in source['physical_edge_ids'])
  qq.append(q);matrix=np.column_stack([normal[i],-(uv[b]-uv[a])])
  if not qvalid[i] or abs(np.linalg.det(matrix))<1e-8:continue
  offset,fraction=np.linalg.solve(matrix,uv[a]-center[i]);position=center[i]+offset*normal[i];q.update(offset_px=float(offset),fraction=float(fraction),position=position.tolist(),normal=normal[i].tolist())
  if not (0<=fraction<=1 and abs(offset)<=32):continue
  za=(g['X'][a]@g['R'].T+g['t'])[2];zb=(g['X'][b]@g['R'].T+g['t'])[2];u=(fraction/zb)/((1-fraction)/za+fraction/zb);X=(1-u)*g['X'][a]+u*g['X'][b]
  x,y=position;h,w=visible.shape;inside=bool(0<=x<w and 0<=y<h);ix,iy=int(round(x-.5)),int(round(y-.5));support=bool(inside and (visible[max(0,iy-1):min(h,iy+2),max(0,ix-1):min(w,ix+2)]>127).any())
  q.update(ray_eligible=True,source_X=X.tolist(),source_camera_Z=float((X@g['R'].T+g['t'])[2]),in_image=inside,supplied_visible_mask_support_3x3=support)
  sourceids.append(i);xs.append(X)
 if xs:
  xs32=np.asarray(xs,np.float32);hit=scene.compute_closest_points(o3d.core.Tensor(xs32));stats['closest_point_API_calls']+=1;stats['closest_point_count']+=len(xs)
  pos=hit['points'].numpy();dist=np.linalg.norm(pos-xs32,axis=1);pid=hit['primitive_ids'].numpy();radius=np.linalg.norm(g['dims'])
  for j,i in enumerate(sourceids):qq[i].update(actual_mesh_point_within_original_tolerance=bool(dist[j]<=1e-5*radius),closest_point_primitive_id=int(pid[j]))
 return dict(id=source['id'],index=index,partition=source['partition'],K=g['K'].tolist(),R=g['R'].tolist(),t=g['t'].tolist(),dimensions=g['dims'].tolist(),raw_hw=g['hw'],physical_diagonal_m=float(np.linalg.norm(g['dims'])),depth_tolerance_m=float(.001*np.linalg.norm(g['dims'])),queries=qq,
  earlier_raw_depth_available=False,source_annotation=common.binding(paths['label']),source_visible_mask=common.binding(paths['visible']))
def main():
 begin=time.monotonic();cv2.setNumThreads(1);protocol=proto();base,prepare,bodysha=frozen_frame_function()
 import sys
 sys.path.insert(0,str(OLD));os.environ.setdefault('PALLET_SOURCE_ROOT','/home/minjae/Documents/github/pallet-pose')
 from scripts.research.pallet_observation_refiner_20261009_v1 import common as C,source_audit as S
 sources={r['index']:r for r in loadrows(DOC/'SOURCE_CEILING_ROWS.jsonl.gz')};families=S.selected_families();held={r['index']:r for r in loadrows(DOC/'SOURCE_RAY_VALIDATION_ROWS.jsonl.gz')};frozen={(r['index'],r['query']):r for r in loadrows(P/'CORRECTED_TARGET_PROPOSAL_ROWS.jsonl.gz')}
 with np.load(MESH) as z:vertices=z['vertices'];tri=z['triangles']
 unique,first,weld=np.unique(vertices,axis=0,return_index=True,return_inverse=True);wt=weld[tri];edges=np.stack([wt[:,[0,1]],wt[:,[1,2]],wt[:,[2,0]]],axis=1).reshape(-1,2);edges.sort(axis=1);unique_edges,edge_inv=np.unique(edges,axis=0,return_inverse=True);face_edge_ids=edge_inv.reshape(-1,3)
 order=np.argsort(edge_inv,kind='stable');counts_all=np.bincount(edge_inv);starts=np.r_[0,np.cumsum(counts_all)];owners={uid:sorted(set((order[starts[uid]:starts[uid+1]]//3).tolist())) for uid in range(len(unique_edges))}
 arrays=dict(lo=np.full((1024,84),65,np.int16),hi=np.full((1024,84),65,np.int16),weight=np.zeros((1024,84),np.float32),valid=np.zeros((1024,84),bool));status={p:Counter() for p in ['train','calibration','source_test']};execute=Counter();test_parity=Counter();expected_features=np.load(CACHE/'features.npy',mmap_mode='r');assert expected_features.shape==(1024,84,28,65)
 output=OUT/'FULL_SOURCE_TARGET_ROWS.jsonl.gz';assert not output.exists()
 with gzip.open(output,'wt',compresslevel=6) as out:
  for index in range(1024):
   source=sources[index];partition=source['partition'];frame=copy.deepcopy(held[index]) if index in held else query_frame(index,source,families[index],CACHE,vertices,tri,C,S,execute)
   for q in frame['queries']:
    supported=bool(q.get('actual_mesh_point_within_original_tolerance') and q.get('in_image') and q.get('supplied_visible_mask_support_3x3') and q.get('whole_edge_has_physical_sample'))
    if q['cached_target']=='POSITIVE':category='ORIGINAL_POSITIVE'
    elif supported and q['cached_target']=='NONE':category='FRONT_SURFACE_NONE' if index in held and q.get('original_hit_in_front_of_source') else 'POTENTIAL_NONE'
    else:category=None
    q['full_proposal_category']=category
   rr,cc,sc,bounds=prepare(frame,sources,vertices,tri,face_edge_ids,owners,first,unique_edges);lookup={r['query']:r for r in rr};execute.update(sc);compact=[]
   for q in frame['queries']:
    i=q['query'];r=lookup.get(i);original=q['cached_target'];state='IGNORE' if r is None else r['proposed_target'];record=dict(query=i,edge=q['edge'],original_target=original,proposed_target=state,source_cache_mutated=False,proposal_label_changed=original!=state,
     unsupported_physical_edge=q['edge'] not in source['physical_edge_ids'],earlier_raw_depth_available=index in held,approved_for_training=False)
    if r is None:record['reason']='Unsupported, no validated physical candidate, changed/unavailable support, or uncertain negative; IGNORE instead of fabricated NONE'
    elif state=='POSITIVE':
     s=r['selected'];arrays['lo'][index,i]=s['lo'];arrays['hi'][index,i]=s['hi'];arrays['weight'][index,i]=s['weight'];arrays['valid'][index,i]=True
     record.update(lo=s['lo'],hi=s['hi'],weight=s['weight'],actual_point=s['actual_point'],actual_target_uv=s['actual_target_uv'],normal_offset_px=s['normal_offset_px'],
      actual_wire_id=s['wire_id'],actual_endpoint_vertex_ids=s['actual_endpoint_vertex_ids'],actual_face_ids=s['noncoplanar_face_witness']['triangles'],ownership=s['ownership'],normal_cross_norm=s['noncoplanar_face_witness']['normal_cross_norm'],
      source_to_actual_wire_point_m=s['source_to_actual_wire_point_m'],worst_first_surface_camera_Z_gap_m=s['worst_first_surface_camera_Z_gap_m'],original_depth_tolerance_m=frame['depth_tolerance_m'],unchanged_mask_kernel=s['original_and_corrected_mask_kernel'])
    elif state=='NONE':arrays['valid'][index,i]=True;record.update(lo=65,hi=65,weight=0.,retained_front_depth_minus_source_Z_m=r['retained_front_depth_minus_source_Z_m'])
    else:record['reason']=r['reason']
    if index in held and (index,i) in frozen:
     f=frozen[index,i];test_parity['compared']+=1;assert state==f['proposed_target']
     if state=='POSITIVE':
      for key in ['wire_id','actual_point','actual_target_uv','lo','hi','weight']:assert r['selected'][key]==f['selected'][key],key
     test_parity['exact_match']+=1
    compact.append(record);status[partition][state]+=1
    if original!=state:status[partition]['changed_'+original+'_to_'+state]+=1
   compactframe=dict(id=source['id'],index=index,family=source['family'],partition=partition,raw_hw=frame['raw_hw'],K=frame['K'],R=frame['R'],t=frame['t'],dimensions=frame['dimensions'],frozen_selected_points=source['frozen_selected_points'],queries=compact,
    source_ceiling_semantic_sha256=C.digest(source),source_cache_mutated=False,feature_cache_index=index,augmentation='existing_P0_original_only',source_features_regenerated=False)
   out.write(json.dumps(compactframe,separators=(',',':'),allow_nan=False)+'\n')
   if (index+1)%32==0:
    progress=dict(stage='actual_wire_full_source_arithmetic',families=index+1,seconds=time.monotonic()-begin,counts={p:dict(c) for p,c in status.items()},execution=dict(execute),new_rays=0,new_heads=0,new_training=0)
    write(OUT/'FULL_SOURCE_PROGRESS.json',progress);print('FULL_SOURCE_TARGETS',index+1,1024,round(time.monotonic()-begin,2),flush=True)
 np.savez_compressed(OUT/'PREPARED_TARGETS.npz',**arrays,source_index=np.arange(1024),partitions=np.array([sources[i]['partition'] for i in range(1024)]))
 assert execute['scene_builds']==896 and execute['closest_point_count']<=75264 and test_parity['exact_match']==6463
 result=dict(schema='same_actual_wire_full_source_targets_v1',complete_preparation=True,ready_for_learning=False,
  readiness_reason='No genuine earlier first-surface negative metadata in train/calibration; the dataset can train positions/existence-positive only and is NOT complete correspondence/no-match supervision. Unknown negatives correctly IGNORE. No newlearning authorized or executed.',
  family_count=1024,splits=dict(train=768,calibration=128,source_test=128),counts={p:dict(c) for p,c in status.items()},execution=dict(execute),frozen_source_test_parity=dict(test_parity),
  feature_compatibility=dict(path='learned_cache/features.npy',sha256=sha(CACHE/'features.npy'),shape=[1024,84,28,65],dtype='float16',regenerated=False,cache_manifest_sha256=sha(CACHE/'CACHE_MANIFEST.json'),batch_order_sha256=sha(CACHE/'order.npy')),
  no_real_GT_reads=True,no_calibration_or_test_selection=True,family_order_unchanged=True,augmentation='existing_original_only',source_cache_mutated=False,
  frozen_wire_math_sha256=sha(FIXED),extracted_AST_sha256=bodysha,protocol_sha256=sha(protocol),rows_sha256=sha(output),prepared_target_arrays_sha256=sha(OUT/'PREPARED_TARGETS.npz'),code_sha256=sha(Path(__file__)),
  wall_seconds=time.monotonic()-begin,new_rays=0,new_heads=0,new_detector=0,new_PnP=0,new_RGB=0,new_training_updates=0)
 write(OUT/'FULL_SOURCE_PREPARATION.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
