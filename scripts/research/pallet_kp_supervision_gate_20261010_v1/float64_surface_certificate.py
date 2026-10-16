"""Retained source point / actual float64 triangle and enclosure arithmetic.

The protocol is sealed before new arithmetic. No rays, heads, fit, target
approval or source mutation is performed. All inputs are existing records.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter,defaultdict
import gzip,json,hashlib,time
import numpy as np
PRIVATE=Path(__file__).parent
ORIGIN=Path('/dev/shm/pallet-observation-worktree-20261009')
ROWS=ORIGIN/'_docs/experiments/pallet_kp_difficulty_20261010_v1/SOURCE_RAY_VALIDATION_ROWS.jsonl.gz'
MESH=Path('/dev/shm/pallet-observation-private-20261009/actual_mesh/scene.usd.npz')
META=MESH.with_suffix('.json')
ASSET=Path('/home/minjae/Documents/github/pallet-pose/data/pallet/raw_data/models_usd/scene.usd')
EPS=np.finfo(np.float64).eps
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def invert_camera(R,t):
 """Exact rational inverse of the retained binary-float pose matrix."""
 a=[[F(float(R[i][j])) for j in range(3)]+[-F(float(t[i]))] for i in range(3)]
 for j in range(3):
  pivot=next(k for k in range(j,3) if a[k][j]);a[j],a[pivot]=a[pivot],a[j]
  scale=a[j][j];a[j]=[x/scale for x in a[j]]
  for k in range(3):
   if k==j:continue
   scale=a[k][j];a[k]=[x-scale*y for x,y in zip(a[k],a[j])]
 return [a[i][3] for i in range(3)]
def enclosure(camera,point,lower,upper):
 near=far=None
 for o,p,lo,hi in zip(camera,point,lower,upper):
  p=F(float(p));lo=F(float(lo));hi=F(float(hi));d=p-o
  if d==0:
   if o<lo or o>hi:return None
   continue
  a,b=(lo-o)/d,(hi-o)/d;nn,ff=min(a,b),max(a,b)
  near=nn if near is None else max(near,nn);far=ff if far is None else min(far,ff)
 if near is None or near>far or far<0 or near>1:return None
 return near,far
def tri_closest(point,T):
 """Closed triangle nearest point in longdouble; result returned as floats."""
 p=np.asarray(point,np.longdouble);a,b,c=np.asarray(T,np.longdouble);u=b-a;v=c-a;n=np.cross(u,v);nn=n@n
 if nn==0:return None
 pp=p-((p-a)@n/nn)*n;uu=u@u;vv=v@v;uv=u@v;pu=(pp-a)@u;pv=(pp-a)@v;den=uu*vv-uv*uv
 if den<=0:return None
 alpha=(vv*pu-uv*pv)/den;beta=(uu*pv-uv*pu)/den;bary=np.array([1-alpha-beta,alpha,beta])
 candidates=[pp] if (bary>=0).all() else []
 for x,y in [(a,b),(b,c),(c,a)]:
  d=y-x;length=d@d
  if length>0:candidates.append(x+np.clip((p-x)@d/length,0,1)*d)
 cp=min(candidates,key=lambda x:np.linalg.norm(p-x))
 return dict(distance_m=float(np.linalg.norm(p-cp)),nearest_actual_point=[float(x) for x in cp],
  plane_signed_m=float((p-a)@n/np.sqrt(nn)),source_barycentric=[float(x) for x in bary],
  triangle_area_twice_m2=float(np.sqrt(nn)),gram_condition_proxy=float((uu*vv)/den))
def quantiles(a):
 return dict(n=len(a),min=float(np.min(a)),median=float(np.median(a)),P90=float(np.quantile(a,.9)),P99=float(np.quantile(a,.99)),max=float(np.max(a))) if a else dict(n=0)
def main():
 begin=time.monotonic();protocol=PRIVATE/'FLOAT64_SURFACE_PROTOCOL.json';assert not protocol.exists()
 write(protocol,dict(schema='retained_actual_float64_surface_and_exact_inverse_enclosure_protocol_v1',created_before_arithmetic=True,
  inputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in [ROWS,MESH,META,ASSET,Path(__file__)]},
  fixed_population='2164 physical,in-frame,mask-supported original infinite NONE +2333original POSITIVE +1966physical,in-frame,mask-supported frontsurface NONE from existing source-test128',
  expected_records=6463,mesh_definition='existing normalized vertices float64 multiplied by each retained source dimensions, no float32 rounding; original normalization metadata/asset verified',
  membership='Closed actual closest-primitive triangle; longdouble plane projection/barycentric computation with nearest segment candidates. Degenerate or unresolved conditioning returns unknown.',
  machine_guard='64 * binary64 epsilon * max(1,max(abs(actual vertices)),max(dimensions)); fixed before arithmetic, only numerical membership diagnostic',
  old_guard='1e-5 * diagonal, recorded separately, never substituted for machine membership',
  enclosure='Actual scaled vertex bounds, also outward nextafter bounds; all triangles convex combinations of enclosed actual vertices',
  camera='Exact rational Gaussian inverse solve of retained binary-float R,t; R.T origin comparison retained only as diagnostic',
  certificate='Exact rational slab interval; strict near==1 recorded separately from near-terminal within fixed1e-12 inherited prior certificate; no tolerance presented as strict proof',
  normalization_checks='all normalized vertices expected[-.5,.5] tested exactly and with fixed machine guard; actual scaled vertices enclosed by measured outward bbox',
  source_ownership_limit='Closest triangle membership proves actual surface membership, not original USDpart/material ownership or RGB boundary identity; target approval remains false',
  new_rays=0,new_models=0,new_RGB=0,new_training=0,new_fits=0,target_mutation=False))
 metadata=json.loads(META.read_text());assert sha(ASSET)==metadata['asset']['sha256']
 with np.load(MESH) as z:vertices=z['vertices'];tri=z['triangles']
 assert vertices.dtype==np.float64 and len(vertices)==metadata['vertices'] and len(tri)==metadata['triangles']
 normalized_min=vertices.min(0);normalized_max=vertices.max(0);norm_tau=64*EPS
 normcheck=dict(vertices=len(vertices),triangles=len(tri),vertex_dtype=str(vertices.dtype),
  normalized_min=normalized_min.tolist(),normalized_max=normalized_max.tolist(),
  all_vertices_inside_expected_half_exact=bool(((vertices>=-.5)&(vertices<=.5)).all()),
  all_vertices_inside_expected_half_machine_guard=bool(((vertices>=-.5-norm_tau)&(vertices<=.5+norm_tau)).all()),
  max_half_enclosure_violation=float(max(0.,float((-vertices-.5).max()),float((vertices-.5).max()))))
 frames=[json.loads(x) for x in gzip.open(ROWS,'rt')];result=[];counts=defaultdict(Counter);metrics=defaultdict(list);framechecks=[]
 for row in frames:
  dimensions=np.asarray(row['dimensions']);actual=vertices*dimensions;lower=actual.min(0);upper=actual.max(0)
  outward_lower=np.nextafter(lower,-np.inf);outward_upper=np.nextafter(upper,np.inf)
  tau=float(64*EPS*max(1.,float(np.max(np.abs(actual))),float(dimensions.max())));camera=invert_camera(row['R'],row['t']);cf=np.array([float(x) for x in camera]);transpose=-np.array(row['R']).T@np.array(row['t'])
  framechecks.append(dict(id=row['id'],index=row['index'],measured_bbox_min=lower.tolist(),measured_bbox_max=upper.tolist(),outward_bbox_min=outward_lower.tolist(),outward_bbox_max=outward_upper.tolist(),
   all_actual_vertices_enclosed=True,expected_half_dimension_max_error_m=float(max(np.abs(lower+dimensions/2).max(),np.abs(upper-dimensions/2).max())),
   exact_inverse_camera_object=cf.tolist(),exact_inverse_camera_rational=[str(x) for x in camera],
   exact_inverse_vs_transpose_camera_distance_m=float(np.linalg.norm(cf-transpose)),machine_membership_guard_m=tau))
  for q in row['queries']:
   supported=bool(q.get('actual_mesh_point_within_original_tolerance') and q.get('in_image') and q.get('supplied_visible_mask_support_3x3') and q.get('whole_edge_has_physical_sample'))
   category='ORIGINAL_POSITIVE' if q['cached_target']=='POSITIVE' else 'INFINITE_NONE' if supported and q['cached_target']=='NONE' and not q['original_finite'] else 'FRONT_SURFACE_NONE' if supported and q['cached_target']=='NONE' and q['original_hit_in_front_of_source'] else None
   if category is None:continue
   point=np.asarray(q['source_X']);pid=q['closest_point_primitive_id'];owned=tri_closest(point,actual[tri[pid]]);bbox_inside=bool(((point>=lower)&(point<=upper)).all());bbox_inside_machine=bool(((point>=lower-tau)&(point<=upper+tau)).all())
   extremes=np.minimum(np.abs(point-lower),np.abs(point-upper));extreme_axes=np.flatnonzero(extremes<=tau).tolist()
   interval=enclosure(camera,point,lower,upper);out_interval=enclosure(camera,point,outward_lower,outward_upper)
   near=None if interval is None else interval[0];out_near=None if out_interval is None else out_interval[0]
   strict=bool(near is not None and near==1 and bbox_inside);machine_certificate=bool(out_near is not None and abs(float(out_near-1))<=1e-12 and bbox_inside_machine)
   machine_member=bool(owned is not None and owned['distance_m']<=tau);old_member=bool(owned is not None and owned['distance_m']<=1e-5*row['physical_diagonal_m'])
   record=dict(id=row['id'],index=row['index'],query=q['query'],edge=q['edge'],category=category,source_X=point.tolist(),closest_primitive_id=pid,actual_triangle_vertex_ids=tri[pid].tolist(),actual_triangle=actual[tri[pid]].tolist(),
    triangle_membership=owned,machine_guard_m=tau,actual_surface_machine_membership=machine_member,actual_surface_old_tolerance_membership=old_member,
    source_inside_actual_bbox_exact=bbox_inside,source_inside_actual_bbox_machine_guard=bbox_inside_machine,source_at_bbox_extreme_axes=extreme_axes,
    exact_inverse_camera_object=cf.tolist(),actual_bbox_strict_terminal_entry=strict,outward_bbox_machine_terminal_entry=machine_certificate,
    actual_bbox_slab_interval=None if interval is None else [float(x) for x in interval],actual_bbox_slab_near_minus_one=None if near is None else float(near-1),
    actual_bbox_slab_near_rational=None if near is None else str(near),outward_bbox_slab_near_minus_one=None if out_near is None else float(out_near-1),
    retained_offset_witness=bool(q.get('fixed_normal_offset_rescues_source_depth')),candidate_actual_surface_and_strict_own_mesh_visibility=bool(machine_member and strict and len(extreme_axes)>=2),
    candidate_actual_surface_and_machine_own_mesh_visibility=bool(machine_member and machine_certificate and len(extreme_axes)>=2),
    source_target_approved=False,USDpart_ownership_unknown=True,RGB_boundary_identity_not_proven=True,original_labels_unchanged=True)
   result.append(record);c=counts[category];c['n']+=1;c['machine_triangle_membership']+=machine_member;c['old_tolerance_triangle_membership']+=old_member;c['triangle_degenerate_or_unknown']+=owned is None;c['bbox_strict_terminal_entry']+=strict;c['outward_bbox_machine_terminal_entry']+=machine_certificate;c['at_least2actual_bbox_extreme_axes']+=len(extreme_axes)>=2;c['surface_and_strict_visibility']+=record['candidate_actual_surface_and_strict_own_mesh_visibility'];c['surface_and_machine_visibility']+=record['candidate_actual_surface_and_machine_own_mesh_visibility']
   if owned is not None:metrics[category+'_source_to_actual_triangle_distance_m'].append(owned['distance_m']);metrics[category+'_triangle_gram_condition_proxy'].append(owned['gram_condition_proxy'])
 assert len(result)==6463 and counts['INFINITE_NONE']['n']==2164 and counts['ORIGINAL_POSITIVE']['n']==2333 and counts['FRONT_SURFACE_NONE']['n']==1966
 with gzip.open(PRIVATE/'FLOAT64_SURFACE_ROWS.jsonl.gz','wt') as f:
  for x in result:f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
 validation=dict(schema='actual_float64_surface_exact_camera_readonly_v1',complete=True,counts={k:dict(v) for k,v in counts.items()},metrics={k:quantiles(v) for k,v in metrics.items()},normalization=normcheck,framechecks=framechecks,
  protocol_sha256=sha(protocol),rows_sha256=sha(PRIVATE/'FLOAT64_SURFACE_ROWS.jsonl.gz'),wall_seconds=time.monotonic()-begin,
  repair_approval=False,no_point_automatically_promoted=True,all_source_targets_unchanged=True,new_rays=0,new_models=0,new_training=0,new_RGB=0,new_fits=0)
 write(PRIVATE/'FLOAT64_SURFACE_VALIDATION.json',validation);print(json.dumps({k:v for k,v in validation.items() if k not in ['framechecks']}),flush=True)
if __name__=='__main__':main()
