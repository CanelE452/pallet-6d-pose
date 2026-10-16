"""Source-only actual mesh wire target proposal, without casts or models."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter,defaultdict
import gzip,json,hashlib,time
import numpy as np
PRIVATE=Path(__file__).parent
OLD=Path('/dev/shm/pallet-observation-worktree-20261009/_docs/experiments/pallet_kp_difficulty_20261010_v1')
RAYS=OLD/'SOURCE_RAY_VALIDATION_ROWS.jsonl.gz';CEILING=OLD/'SOURCE_CEILING_ROWS.jsonl.gz'
MESH=Path('/dev/shm/pallet-observation-private-20261009/actual_mesh/scene.usd.npz')
EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
EPS=np.finfo(np.float64).eps
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def loadrows(p):return [json.loads(x) for x in gzip.open(p,'rt')]
def exact_camera(R,t):
 a=[[F(float(R[i][j])) for j in range(3)]+[-F(float(t[i]))] for i in range(3)]
 for j in range(3):
  pivot=next(k for k in range(j,3) if a[k][j]);a[j],a[pivot]=a[pivot],a[j];scale=a[j][j];a[j]=[x/scale for x in a[j]]
  for k in range(3):
   if k!=j:
    scale=a[k][j];a[k]=[x-scale*y for x,y in zip(a[k],a[j])]
 return [a[i][3] for i in range(3)]
def camera_point(point,R,t):return [sum(R[i][j]*point[j] for j in range(3))+t[i] for i in range(3)]
def project(point,R,t,K):
 p=camera_point(point,R,t);return [(K[i][0]*p[0]+K[i][1]*p[1]+K[i][2]*p[2])/p[2] for i in range(2)],p[2]
def slab(camera,point,lower,upper):
 near=far=None
 for o,p,lo,hi in zip(camera,point,lower,upper):
  d=p-o
  if d==0:
   if o<lo or o>hi:return None
   continue
  a,b=(lo-o)/d,(hi-o)/d;nn,ff=min(a,b),max(a,b);near=nn if near is None else max(near,nn);far=ff if far is None else min(far,ff)
 if near is None or near>far or far<0 or near>1:return None
 return max(F(0),near),far
def face_info(T):
 a,b,c=np.asarray(T,np.longdouble);u=b-a;v=c-a;n=np.cross(u,v);nn=n@n;uu=u@u;vv=v@v;uv=u@v;den=uu*vv-uv*uv
 if nn<=0 or den<=0:return None
 return n/np.sqrt(nn),float(uu*vv/den)
def nearest(point,A,B):
 d=B-A;u=float(np.clip((point-A)@d/(d@d),0,1));return A+u*d
def proto():
 target=PRIVATE/'WIRE_TARGET_PROTOCOL.json';assert not target.exists()
 write(target,dict(schema='source_only_actual_wire_correspondence_proposal_protocol_v1',created_before_arithmetic=True,
  inputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in [RAYS,CEILING,MESH,Path(__file__)]},
  population='fixed source-test128 retained6463: original2333POSITIVE,2164physical/inframe/mask-supported infinite NONE,1966front-surface NONE; no outcome selection',
  physical_wire='actual float64 mesh triangle edges, exact-coordinate vertex welding; never invented bbox segment or mesh-face interior diagonal',
  semantic_agreement='actual wire endpoints within original1e-5*diagonal of the intended native segment line and extent; intended direction derived from retained source_X on that native edge',
  boundary_ownership='wire shared by at least two provably noncoplanar incident actual triangles, or both endpoints exactly on two actual global extremal planes; open UV/index seams alone never sufficient',
  noncoplanarity='cross(unitnormal_i,unitnormal_j) >256*eps64*max(1,Gramcondition_i,Gramcondition_j); winding sign ignored',
  candidate_limit='only3edges of retained closest actual primitive; no remote mesh search or thresholds swept; absent owner/ambiguous mapping→IGNORE proposal',
  correspondence='exact rational intersection of frozen normal query with projected actual wire; perspective-correct rational interpolation supplies actual closed-triangle point',
  visibility='exact inverse camera, actual mesh vertex bbox exact rational slab; true owned wire endpoint supplies a hit at1; worst possible preceding cameraZ bound=(1-entry)*actualtargetZ ≤ original.001*diagonal',
  mask_gate='same supplied visible-mask3x3 kernel only if round(x-.5),round(y-.5) unchanged after actualwire projection; changed kernel→unresolved without mask read',
  ambiguity='nearest actualwire point selected by source-only geometric distance; machine-level tied distinct points→IGNORE',
  original_front_surface_NONE='keptNONE, never promoted',source_target_overwrite=False,new_rays=0,new_models=0,new_training=0,new_RGB=0,new_PnP=0,
  ready_for_learning=False,reason='proposal only; fixed128audit does not prepare train/cal supervision or approve newlearning'))
 return target
def main():
 begin=time.monotonic();protocol=proto();frames=loadrows(RAYS);sources={r['index']:r for r in loadrows(CEILING)}
 with np.load(MESH) as z:vertices=z['vertices'];tri=z['triangles']
 unique,first,weld=np.unique(vertices,axis=0,return_index=True,return_inverse=True)
 wt=weld[tri];all_edges=np.stack([wt[:,[0,1]],wt[:,[1,2]],wt[:,[2,0]]],axis=1).reshape(-1,2);all_edges.sort(axis=1)
 unique_edges,edge_inv,edge_count=np.unique(all_edges,axis=0,return_inverse=True,return_counts=True);face_edge_ids=edge_inv.reshape(-1,3)
 requested=set()
 for frame in frames:
  for q in frame['queries']:
   if q.get('ray_eligible') and q.get('actual_mesh_point_within_original_tolerance') and q.get('in_image') and q.get('supplied_visible_mask_support_3x3') and q.get('whole_edge_has_physical_sample'):
    requested.update(face_edge_ids[q['closest_point_primitive_id']].tolist())
 owners=defaultdict(list)
 for pos in np.flatnonzero(np.isin(edge_inv,list(requested))):owners[int(edge_inv[pos])].append(int(pos//3))
 counts=defaultdict(Counter);result=[];stagecounts=Counter();framebounds=[]
 for frame in frames:
  source=sources[frame['index']];dims=np.array(frame['dimensions']);actual=vertices*dims;lower=actual.min(0);upper=actual.max(0);lowerF=[F(float(x)) for x in lower];upperF=[F(float(x)) for x in upper]
  camera=exact_camera(frame['R'],frame['t']);R=[[F(float(x)) for x in row] for row in frame['R']];t=[F(float(x)) for x in frame['t']];K=[[F(float(x)) for x in row] for row in frame['K']]
  tau=float(64*EPS*max(1.,float(np.max(np.abs(actual))),float(dims.max())));physical_tol=1e-5*frame['physical_diagonal_m'];depth_tolF=F(float(frame['depth_tolerance_m']));points=np.array(source['frozen_selected_points']);direction={};facecache={}
  for e in range(12):
   xx=[np.array(q['source_X']) for q in frame['queries'] if q['edge']==e and 'source_X' in q]
   if len(xx)>=2:
    v=xx[-1]-xx[0]
    if np.linalg.norm(v)>0:direction[e]=int(np.argmax(np.abs(v)))
  for q in frame['queries']:
   supported=bool(q.get('actual_mesh_point_within_original_tolerance') and q.get('in_image') and q.get('supplied_visible_mask_support_3x3') and q.get('whole_edge_has_physical_sample'))
   category='ORIGINAL_POSITIVE' if q['cached_target']=='POSITIVE' else 'INFINITE_NONE' if supported and q['cached_target']=='NONE' and not q['original_finite'] else 'FRONT_SURFACE_NONE' if supported and q['cached_target']=='NONE' and q['original_hit_in_front_of_source'] else None
   if category is None:continue
   output=dict(id=frame['id'],index=frame['index'],query=q['query'],edge=q['edge'],category=category,original_target=q['cached_target'],proposed_target='IGNORE',
    source_input_row_sha256=hashlib.sha256(json.dumps(frame,sort_keys=True,separators=(',',':')).encode()).hexdigest(),original_source_X=q['source_X'],
    original_projected_position=q['position'],physical_tolerance_m=physical_tol,original_depth_tolerance_m=frame['depth_tolerance_m'],target_changed=False,approved_for_training=False,candidates=[])
   if category=='FRONT_SURFACE_NONE':
    output.update(proposed_target='NONE',reason='original cached actual front-surface witness preserved',retained_front_depth_minus_source_Z_m=q['ray_depth_minus_source_Z_m'][0]);result.append(output);counts[category]['n']+=1;counts[category]['proposed_NONE']+=1;continue
   if q['edge'] not in direction:
    output['reason']='semantic direction unavailable';result.append(output);counts[category]['n']+=1;counts[category]['proposed_IGNORE']+=1;continue
   axis=direction[q['edge']];fixed=[j for j in range(3) if j!=axis];X=np.array(q['source_X']);a,b=EDGES[q['edge']];u=(q['query']%7+1)/8;center=(1-u)*points[a]+u*points[b];normal=np.array(q['normal']);centerF=[F(float(x)) for x in center];normalF=[F(float(x)) for x in normal];valid=[]
   for uid in sorted(set(face_edge_ids[q['closest_point_primitive_id']].tolist())):
    stagecounts['candidate_wire_checks']+=1;vertexids=first[unique_edges[uid]];A,B=actual[vertexids];c=dict(wire_id=uid,actual_endpoint_vertex_ids=vertexids.tolist(),incident_triangle_ids=sorted(set(owners[uid])),endpoint_A=A.tolist(),endpoint_B=B.tolist())
    output['candidates'].append(c)
    if np.linalg.norm(B-A)==0:c['reject']='degenerate wire';continue
    if max(np.linalg.norm(A[fixed]-X[fixed]),np.linalg.norm(B[fixed]-X[fixed]))>physical_tol or min(A[axis],B[axis])<lower[axis]-physical_tol or max(A[axis],B[axis])>upper[axis]+physical_tol:
     c['reject']='semantic line or extent mismatch';continue
    faceids=c['incident_triangle_ids'];norms=[]
    for pid in faceids:
     if pid not in facecache:facecache[pid]=face_info(actual[tri[pid]])
     if facecache[pid] is not None:norms.append((pid,*facecache[pid]))
    crease=None
    for i,(p,n,condition) in enumerate(norms):
     for p2,n2,condition2 in norms[i+1:]:
      cross=float(np.linalg.norm(np.cross(n,n2)));bound=256*EPS*max(1.,condition,condition2)
      if cross>bound:crease=dict(triangles=[p,p2],normal_cross_norm=cross,numerical_bound=bound);break
     if crease is not None:break
    extremeaxes=[j for j in range(3) if (A[j]==lower[j] and B[j]==lower[j]) or (A[j]==upper[j] and B[j]==upper[j])]
    if crease is None and len(extremeaxes)<2:c['reject']='actual noncoplanar adjacency and two exact extremal planes both unproved';continue
    c.update(ownership='NONCOPLANAR_ACTUAL_SHARED_WIRE' if crease else 'TWO_ACTUAL_GLOBAL_EXTREMAL_PLANES',noncoplanar_face_witness=crease,exact_outer_extremal_axes=extremeaxes)
    AF=[F(float(x)) for x in A];BF=[F(float(x)) for x in B];uvA,zA=project(AF,R,t,K);uvB,zB=project(BF,R,t,K)
    if zA<=0 or zB<=0:c['reject']='wire behind camera';continue
    tangent=[uvB[j]-uvA[j] for j in range(2)];m00,m10=normalF;m01,m11=-tangent[0],-tangent[1];det=m00*m11-m01*m10
    if det==0:c['reject']='normal and wire degenerate';continue
    rhs=[uvA[j]-centerF[j] for j in range(2)];offset=(rhs[0]*m11-m01*rhs[1])/det;fraction=(m00*rhs[1]-rhs[0]*m10)/det
    if not (0<=fraction<=1 and abs(offset)<=32):c['reject']='physical wire does not intersect query within search and segment';continue
    u3d=(fraction/zB)/((1-fraction)/zA+fraction/zB);P=[(1-u3d)*AF[j]+u3d*BF[j] for j in range(3)];uv,zP=project(P,R,t,K)
    bounds=slab(camera,P,lowerF,upperF)
    if bounds is None:c['reject']='actual bbox endpoint proof unavailable';continue
    bound=(1-bounds[0])*zP
    if bound<0 or bound>depth_tolF:c['reject']='worst first-surface depth bound exceeds original tolerance';continue
    kernel_old=[int(round(float(x)-.5)) for x in q['position']];kernel_new=[int(round(float(x)-.5)) for x in uv]
    if kernel_old!=kernel_new:c['reject']='supplied mask kernel changed';continue
    Pfloat=np.array([float(x) for x in P]);dist=float(np.linalg.norm(Pfloat-X))
    if dist>physical_tol:c['reject']='physical target point exceeds original semantic tolerance';continue
    delta=sum((uv[j]-centerF[j])*normalF[j] for j in range(2));binvalue=delta+32;lo=int(binvalue//1);lo=int(np.clip(lo,0,64));hi=min(64,lo+1);weight=float(binvalue-lo)
    if not (0<=weight<=1):c['reject']='target bin outside valid interpolation';continue
    c.update(validated_positive_proposal=True,actual_point=Pfloat.tolist(),actual_point_rational=[str(x) for x in P],wire_parameter=float(u3d),wire_parameter_rational=str(u3d),
     actual_target_uv=[float(x) for x in uv],normal_offset_px=float(delta),lo=lo,hi=hi,weight=weight,source_to_actual_wire_point_m=dist,
     actual_camera_Z=float(zP),enclosure_entry=float(bounds[0]),enclosure_entry_rational=str(bounds[0]),worst_first_surface_camera_Z_gap_m=float(bound),
     supplied_visible_mask_kernel_reused_unchanged=True,original_and_corrected_mask_kernel=kernel_new,
     endpoint_hit_proof='Exact rational convex combination of endpoints of an actual owned triangle wire; no mesh cast',
     visibility_limit='Within original .001diagonal depth policy for ownmesh; supplied visiblemaskkernel reused, no new externaloccluder proof')
    valid.append(c)
   if not valid:output['reason']='no owned physical wire passed semantic, query, depth and unchanged-mask gates'
   else:
    valid.sort(key=lambda c:(c['source_to_actual_wire_point_m'],c['wire_id']));selected=valid[0]
    tied=[c for c in valid[1:] if abs(c['source_to_actual_wire_point_m']-selected['source_to_actual_wire_point_m'])<=tau and np.linalg.norm(np.array(c['actual_point'])-selected['actual_point'])>tau]
    if tied:output['reason']='machine-tied distinct physical wire points; ambiguous'
    else:output.update(proposed_target='POSITIVE',reason='actual owned wire and original depth/mask policy certified',selected_wire_id=selected['wire_id'],selected=selected)
   result.append(output);counts[category]['n']+=1;counts[category]['proposed_'+output['proposed_target']]+=1
  framebounds.append(dict(id=frame['id'],index=frame['index'],actual_bbox_min=lower.tolist(),actual_bbox_max=upper.tolist(),exact_camera_rational=[str(x) for x in camera]))
 assert len(result)==6463
 with gzip.open(PRIVATE/'CORRECTED_TARGET_PROPOSAL_ROWS.jsonl.gz','wt') as f:
  for x in result:f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
 validation=dict(schema='actual_owned_wire_target_proposal_v1',proposal_only=True,source_targets_overwritten=False,ready_for_learning=False,
  reason='Only fixedsource-test128 candidates audited; no fulltrain/cal correction, no retraining, no realdeployment success established; review explicit ownership and originalmask/depthpolicy limits first',
  counts={k:dict(v) for k,v in counts.items()},stagecounts=dict(stagecounts),exact_welded_vertices=len(unique),exact_welded_edges=len(unique_edges),max_edge_incident_triangle_count=int(edge_count.max()),
  framebounds=framebounds,protocol_sha256=sha(protocol),rows_sha256=sha(PRIVATE/'CORRECTED_TARGET_PROPOSAL_ROWS.jsonl.gz'),code_sha256=sha(Path(__file__)),wall_seconds=time.monotonic()-begin,
  new_rays=0,new_models=0,new_RGB=0,new_training=0,new_PnP=0)
 write(PRIVATE/'CORRECTED_TARGET_PROPOSAL_VALIDATION.json',validation);print(json.dumps({k:v for k,v in validation.items() if k!='framebounds'}),flush=True)
if __name__=='__main__':main()
