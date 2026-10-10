"""Read-only arithmetic over retained actual-mesh ray evidence; no casts."""
import gzip,json,hashlib,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path('/dev/shm/pallet-observation-worktree-20261009')
DOC=ROOT/'_docs/experiments/pallet_kp_difficulty_20261010_v1'
PRIVATE=Path(__file__).parent
MESH=Path('/dev/shm/pallet-observation-private-20261009/actual_mesh/scene.usd.npz')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def pdist(p,T):
 a,b,c=T;u=b-a;v=c-a;n=np.cross(u,v);nn=n@n
 if nn==0:return None
 plane_signed=(p-a)@n/np.sqrt(nn);projected=p-((p-a)@n/nn)*n
 uu=u@u;vv=v@v;uv=u@v;pu=(projected-a)@u;pv=(projected-a)@v;den=uu*vv-uv*uv
 alpha=(vv*pu-uv*pv)/den;beta=(uu*pv-uv*pu)/den;bary=np.array([1-alpha-beta,alpha,beta])
 if (bary>=0).all():dist=np.linalg.norm(p-projected)
 else:dist=min(np.linalg.norm(p-(x+np.clip((p-x)@(y-x)/((y-x)@(y-x)),0,1)*(y-x))) for x,y in [(a,b),(b,c),(c,a)] if np.linalg.norm(y-x)>0)
 return float(dist),float(plane_signed),bary,n/np.sqrt(nn)
def run():
 begin=time.monotonic();protocol=PRIVATE/'TRIANGLE_OWNERSHIP_PROTOCOL.json';assert not protocol.exists()
 write(protocol,dict(schema='stored_ray_triangle_membership_readonly_v1',inputs={p.name:sha(p) for p in [MESH,DOC/'SOURCE_RAY_VALIDATION_ROWS.jsonl.gz',Path(__file__)]},
  expected_positive_records=2333,expected_rescued_records=1202,max_records=3535,
  algorithm='Actual original float32 vertex coordinates promoted to float64; closed triangle distance via plane-projection barycentric membership or minimum point-segment distance; coplanarity signed plane distance; normals dot camera and retained semantic edge direction. No intersections/rays cast.',
  distance_gate='Original1e-5 * diagonal, diagnostic membership comparison only, not new label',
  sharing='Geometric vertices paired within same original1e-5 diagonal tolerance, independent of arbitrary duplicate primitive IDs',
  semantic_limit='Mesh cache contains positions/triangles only, not original part/face ownership; membership cannot alone establish actual RGB boundary identity',
  all_source_targets_unchanged=True,new_rays=0,new_models=0,new_fits=0,new_RGB=0))
 with np.load(MESH) as z:vertices=z['vertices'];tri=z['triangles']
 frames=[json.loads(x) for x in gzip.open(DOC/'SOURCE_RAY_VALIDATION_ROWS.jsonl.gz','rt')]
 summary=defaultdict(Counter);output=[];metrics=defaultdict(list)
 for r in frames:
  actual=(vertices*np.array(r['dimensions'])).astype(np.float32).astype(float);camera=-np.array(r['R']).T@np.array(r['t']);diag=r['physical_diagonal_m'];tol=1e-5*diag
  byedge=defaultdict(list)
  for q in r['queries']:
   if 'source_X' in q:byedge[q['edge']].append(np.array(q['source_X']))
  direction={e:(a[-1]-a[0])/np.linalg.norm(a[-1]-a[0]) for e,a in byedge.items() if len(a)>=2 and np.linalg.norm(a[-1]-a[0])>0}
  for q in r['queries']:
   if not q['ray_eligible']:continue
   category='positive' if q['cached_target']=='POSITIVE' else 'rescued' if q['cached_target']=='NONE' and q['fixed_normal_offset_rescues_source_depth'] and q['actual_mesh_point_within_original_tolerance'] and q['in_image'] and q['supplied_visible_mask_support_3x3'] else None
   if category is None:continue
   j=0 if category=='positive' else next(j for j in [1,2] if q['ray_source_depth_agreement'][j]);pid=q['ray_primitive_ids'][j];cid=q['closest_point_primitive_id'];T=actual[tri[pid]];cpT=actual[tri[cid]];p=np.array(q['source_X'])
   dist,plane,bary,n=pdist(p,T);cpdist,cpplane,cpbary,cn=pdist(p,cpT);shares=sum(any(np.linalg.norm(x-y)<=tol for y in cpT) for x in T)
   facing=float(n@(camera-p)/np.linalg.norm(camera-p));cpface=float(cn@(camera-p)/np.linalg.norm(camera-p));alignment=float(abs(n@cn));semantic=float(abs(n@direction[q['edge']])) if q['edge'] in direction else None
   record=dict(id=r['id'],index=r['index'],query=q['query'],edge=q['edge'],category=category,closest_primitive=cid,witness_primitive=pid,witness_shift_px=[0,.05,-.05][j],
    diagonal=diag,physical_tolerance_m=tol,source_to_witness_triangle_distance_m=dist,source_to_witness_plane_signed_m=plane,witness_barycentric_source=bary.tolist(),
    source_to_closest_triangle_distance_m=cpdist,source_to_closest_plane_signed_m=cpplane,closest_barycentric_source=cpbary.tolist(),shared_geometric_vertices=shares,
    witness_facing_cosine=facing,closest_facing_cosine=cpface,closest_witness_absolute_normal_dot=alignment,witness_normal_dot_semantic_edge_abs=semantic,
    source_on_witness_triangle_within_original_tolerance=dist<=tol,source_on_witness_plane_within_original_tolerance=abs(plane)<=tol,
    all_original_labels_unchanged=True)
   output.append(record);c=summary[category];c['n']+=1;c['same_primitive']+=pid==cid;c['share_geometric_edge']+=shares>=2;c['source_on_triangle']+=dist<=tol;c['source_on_plane']+=abs(plane)<=tol
   c['not_on_triangle_but_on_plane']+=dist>tol and abs(plane)<=tol;c['not_on_triangle_not_on_plane']+=dist>tol and abs(plane)>tol;c['facing_positive']+=facing>0;c['facing_negative']+=facing<0
   c['triangle_owned_facing_positive']+=dist<=tol and facing>0
   for label,value in [('triangle_distance_m',dist),('plane_abs_distance_m',abs(plane)),('normal_abs_dot',alignment),('semantic_tangent_normal_abs',semantic)]:
    if value is not None:metrics[category+'_'+label].append(value)
 assert len(output)==3535
 with gzip.open(PRIVATE/'TRIANGLE_OWNERSHIP_ROWS.jsonl.gz','wt') as f:
  for x in output:f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
 quant={k:dict(min=float(np.min(a)),median=float(np.median(a)),P90=float(np.quantile(a,.9)),P99=float(np.quantile(a,.99)),max=float(np.max(a))) for k,a in metrics.items()}
 result=dict(summary={k:dict(v) for k,v in summary.items()},metrics=quant,protocol_sha256=sha(protocol),rows_sha256=sha(PRIVATE/'TRIANGLE_OWNERSHIP_ROWS.jsonl.gz'),seconds=time.monotonic()-begin,new_rays=0,new_models=0,new_fits=0,new_RGB=0)
 write(PRIVATE/'TRIANGLE_OWNERSHIP_AUDIT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':run()
