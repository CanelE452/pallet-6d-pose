"""Check stored actual triangle witness depth; no label changes or new rays."""
from pathlib import Path
import gzip,json,hashlib,time
from collections import Counter
import numpy as np
P=Path(__file__).parent;D=P/'depth_recovery_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 begin=time.monotonic();protocol=P/'FRONT_TRIANGLE_DEPTH_PROTOCOL.json';assert not protocol.exists()
 raw=D/'RECOVERED_DEPTH_ROWS.jsonl.gz';face=P/'RECOVERED_FRONT_FACE_WITNESSES.json'
 write(protocol,dict(schema='stored_actual_triangle_front_depth_check_protocol_v1',created_before_arithmetic=True,inputs={q.name:dict(sha256=sha(q),bytes=q.stat().st_size) for q in [raw,face,Path(__file__)]},population='All13,754retained finite front witnesses; no selection',
  actual_surface_point='Use exported actual normalized triangle, multiply by source dimensions and cast float32 exactly as original ray scene; then float64 barycentric point weights(1-u-v,u,v).',
  depth='Recompute source-camera Z=(source_R@actual_triangle_point+source_t)[2]. Qualification requires this Z−originalsourceZ<−original .001*diagonal, strictly unchanged.',
  margin='positive_margin=originalsourceZ−triangle_point_cameraZ−originaldepth_tolerance. Nonpositive/nonfinite is transparent ambiguity, never hidden or relabeled here.',
  source_targets_mutated=False,no_new_tolerance=True,new_rays=0,new_closest_points=0,new_models=0,new_RGB=0,new_training=0,new_PnP=0))
 table=json.loads(face.read_text());triangles={r['primitive_id']:np.array(r['normalized_vertices']) for r in table['triangles']};output=[];counts=Counter();margins=[];differences=[];ambiguous=[]
 for line in gzip.open(raw,'rt'):
  frame=json.loads(line);R=np.array(frame['R']);t=np.array(frame['t']);dims=np.array(frame['dimensions']);qq=[]
  for q in frame['queries']:
   if q['t_hit_original_ray_parameter'] is None:continue
   assert q['proved_front_surface_NONE'];T=(triangles[q['primitive_id']]*dims).astype(np.float32).astype(np.float64);u,v=q['primitive_uv'];point=(1-u-v)*T[0]+u*T[1]+v*T[2];Z=float((R@point+t)[2]);delta=Z-q['source_camera_Z'];margin=-frame['depth_tolerance_m']-delta;qualify=bool(np.isfinite(Z) and margin>0);counts['qualifying' if qualify else 'ambiguous']+=1;counts[frame['partition']+'_qualifying' if qualify else frame['partition']+'_ambiguous']+=1;margins.append(margin);differences.append(Z-q['recomputed_hit_source_camera_Z'])
   r=dict(query=q['query'],primitive_id=q['primitive_id'],actual_triangle_barycentric_point_world=point.tolist(),actual_triangle_camera_Z=Z,actual_triangle_minus_source_camera_Z_m=delta,original_depth_tolerance_m=frame['depth_tolerance_m'],strict_front_depth_margin_m=margin,qualifies_under_original_depth_tolerance=qualify,triangle_camera_Z_minus_stored_ray_camera_Z_m=Z-q['recomputed_hit_source_camera_Z'])
   qq.append(r)
   if not qualify:ambiguous.append(dict(id=frame['id'],index=frame['index'],**r))
  output.append(dict(id=frame['id'],index=frame['index'],partition=frame['partition'],queries=qq))
 with gzip.open(P/'FRONT_TRIANGLE_DEPTH_ROWS.jsonl.gz','wt') as f:
  for x in output:f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
 assert len(margins)==13754
 result=dict(schema='stored_actual_triangle_front_depth_check_v1',finite_front_queries=len(margins),counts=dict(counts),all_qualify=counts['ambiguous']==0,minimum_strict_front_margin_m=float(np.min(margins)),margin_m=dict(mean=float(np.mean(margins)),median=float(np.median(margins)),p90=float(np.percentile(margins,90))),triangle_camera_Z_minus_stored_ray_camera_Z_m=dict(max_absolute=float(np.max(np.abs(differences))),median_absolute=float(np.median(np.abs(differences))),p90_absolute=float(np.percentile(np.abs(differences),90))),ambiguous_rows=ambiguous,
  interpretation='Read-only actual closed-triangle surface depth independently agrees with finite front witness under the original tolerance if all_qualify. Actual ray first-surface ordering remains an engine execution witness, not a new exact ray certificate. Frozen labels unchanged; any ambiguity would require an explicit separate correctness revision before learning.',source_targets_mutated=False,
  protocol_sha256=sha(protocol),rows_sha256=sha(P/'FRONT_TRIANGLE_DEPTH_ROWS.jsonl.gz'),code_sha256=sha(Path(__file__)),new_rays=0,new_closest_points=0,new_models=0,new_RGB=0,new_training=0,new_PnP=0,wall_seconds=time.monotonic()-begin)
 write(P/'FRONT_TRIANGLE_DEPTH_VALIDATION.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
