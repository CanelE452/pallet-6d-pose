"""Export actual finite-hit triangle evidence using retained rays only."""
from pathlib import Path
import gzip,json,hashlib,time
from collections import Counter
import numpy as np
P=Path(__file__).parent;DEPTH=P/'depth_recovery_v3';MESH=Path('/dev/shm/pallet-observation-private-20261009/actual_mesh/scene.usd.npz')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def run():
 begin=time.monotonic();protocol=P/'RECOVERED_FRONT_FACE_PROTOCOL.json';assert not protocol.exists()
 raw=DEPTH/'RECOVERED_DEPTH_ROWS.jsonl.gz';validation=DEPTH/'DEPTH_RECOVERY_VALIDATION.json'
 write(protocol,dict(schema='retained_recovered_front_face_export_protocol_v1',created_before_arithmetic=True,inputs={q.name:dict(sha256=sha(q),bytes=q.stat().st_size) for q in [raw,validation,MESH,Path(__file__)]},
  population='All finite recovered actual first-surface witnesses in the already completed fixed13,757query recovery; no case selection or new rays.',
  exported_geometry='Original primitive and vertex IDs, original normalized float64 actual-mesh triangle vertices. Reconstruct actual ray-cast vertices as float32(normalized_vertices*frame_dimensions), exactly as original scene construction.',
  arithmetic='Stored barycentric(u,v) supplies (1-u-v)*V0+u*V1+v*V2 on actual float32 cast triangle. Compare to retained float64(ray_origin)+float64(t_hit)*float64(ray_direction). Record residuals; no threshold chosen, label edited, or new validity gate.',
  new_rays=0,new_closest_points=0,new_models=0,new_training=0,new_RGB=0,new_PnP=0))
 with np.load(MESH) as mesh:vertices=mesh['vertices'];tri=mesh['triangles']
 used=set();records=[];stats=[];counts=Counter()
 for line in gzip.open(raw,'rt'):
  frame=json.loads(line);dims=np.array(frame['dimensions']);qq=[]
  for q in frame['queries']:
   if q['t_hit_original_ray_parameter'] is None:continue
   pid=q['primitive_id'];assert pid is not None and 0<=pid<len(tri);used.add(pid)
   T=(vertices[tri[pid]]*dims).astype(np.float32).astype(np.float64);u,v=np.asarray(q['primitive_uv'],np.float64);point=(1-u-v)*T[0]+u*T[1]+v*T[2];ray=np.asarray(q['ray_float32'],np.float64);onray=ray[:3]+q['t_hit_original_ray_parameter']*ray[3:];residual=float(np.linalg.norm(point-onray));stats.append(residual);counts[frame['partition']]+=1
   qq.append(dict(query=q['query'],primitive_id=pid,barycentric_point_world=point.tolist(),retained_ray_point_world=onray.tolist(),world_point_residual_m=residual,closed_triangle_barycentric=bool(u>=0 and v>=0 and u+v<=1),proved_front_surface_NONE=q['proved_front_surface_NONE']))
  records.append(dict(id=frame['id'],index=frame['index'],partition=frame['partition'],queries=qq))
 table=dict(schema='actual_recovered_front_face_witnesses_v1',mesh_cache_sha256=sha(MESH),raw_depth_rows_sha256=sha(raw),protocol_sha256=sha(protocol),normalization='Already recorded original actual mesh normalized axes [USD Y,-USD Z,-USD X]; multiply each coordinate by recorded frame dimensions before float32 cast.',
  reconstruction='float32(normalized_triangle_vertices * dimensions); barycentric order:1-u-v,u,v.',unique_faces=len(used),finite_queries=len(stats),triangles=[dict(primitive_id=pid,original_vertex_ids=tri[pid].tolist(),normalized_vertices=vertices[tri[pid]].tolist()) for pid in sorted(used)],new_rays=0,new_models=0,new_RGB=0)
 write(P/'RECOVERED_FRONT_FACE_WITNESSES.json',table)
 with gzip.open(P/'RECOVERED_FRONT_FACE_RECONSTRUCTION_ROWS.jsonl.gz','wt') as f:
  for r in records:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
 result=dict(schema='recovered_front_face_export_validation_v1',finite_queries=len(stats),unique_faces=len(used),partition_query_counts=dict(counts),world_point_residual_m=dict(mean=float(np.mean(stats)),median=float(np.median(stats)),p90=float(np.percentile(stats,90)),maximum=float(np.max(stats))),all_retained_finite_barycentrics_in_closed_triangle=all(q['closed_triangle_barycentric'] for r in records for q in r['queries']),
  raw_depth_rows_sha256=sha(raw),face_table_sha256=sha(P/'RECOVERED_FRONT_FACE_WITNESSES.json'),reconstruction_rows_sha256=sha(P/'RECOVERED_FRONT_FACE_RECONSTRUCTION_ROWS.jsonl.gz'),protocol_sha256=sha(protocol),code_sha256=sha(Path(__file__)),new_rays=0,new_closest_points=0,new_models=0,new_RGB=0,new_training=0,new_PnP=0,wall_seconds=time.monotonic()-begin)
 write(P/'RECOVERED_FRONT_FACE_VALIDATION.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':run()
