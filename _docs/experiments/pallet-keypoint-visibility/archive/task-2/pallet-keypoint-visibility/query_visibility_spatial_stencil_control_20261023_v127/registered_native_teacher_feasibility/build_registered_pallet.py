"""Construct a NEW registered pallet and geometric corner-region teacher; NumPy only.

No legacy CAD identification, RGB, raster images, model, PnP or training is involved.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,hashlib,importlib.util,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
CANONICAL=Path('/home/minjae/Documents/github/pallet-pose/scripts/annotate/pallet_geometry.py')
CANONICAL_SHA='4ff7ea9e71473272429e26b1b065b8bb7fe0b036a4822328b58b5ad662a8a3e5'
RASTER=P/'query_visibility_synthetic_part_teacher_20261023_v93/raster_preparation/cpu_part_raster.py'
DIM=np.array([1.2,.144,1.0],np.float64) # NEW static design metres, not real-target measurements.
SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],np.int8)
NAMES=['near_top_left','near_top_right','near_bottom_right','near_bottom_left','far_top_left','far_top_right','far_bottom_right','far_bottom_left']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,v):
 with Path(p).open('x')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def pure_module(path):
 s=importlib.util.spec_from_file_location('v93_projection_only',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def native_contract():
 assert sha(CANONICAL)==CANONICAL_SHA
 node=next(n for n in ast.parse(CANONICAL.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_diagram_points')
 ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(CANONICAL)+'::_diagram_points','exec'),ns)
 anchors=ns['_diagram_points'](DIM[0],DIM[1],DIM[2])[:8];assert np.array_equal(anchors,SIGNS*DIM/2)
 quarter=np.array([[0,0,1],[0,1,0],[-1,0,0]],np.float64);perms=[];rotations=[]
 for k in range(4):
  R=np.linalg.matrix_power(quarter,k);target=ns['_diagram_points'](DIM[k%2*2],DIM[1],DIM[2-k%2*2])[:8]
  rotated=anchors@R.T;perm=[]
  for point in target:
   matches=np.flatnonzero(np.all(rotated==point,axis=1));assert len(matches)==1;perm.append(int(matches[0]))
  assert sorted(perm)==list(range(8))and np.array_equal(rotated[perm],target)
  rotations.append(R);perms.append(perm)
 return anchors,np.array(rotations),np.array(perms,np.int32)

def add_box(lo,hi,part_id,vertices,triangles,owners,face_axis,face_side):
 start=len(vertices);verts=np.array([[x,y,z]for x in[lo[0],hi[0]]for y in[lo[1],hi[1]]for z in[lo[2],hi[2]]],np.float64);vertices.extend(verts)
 for axis in range(3):
  other=[a for a in range(3)if a!=axis]
  for side in(-1,1):
   ids=np.flatnonzero(verts[:,axis]==(lo if side<0 else hi)[axis]);center=verts[ids].mean(0)
   angles=np.arctan2(verts[ids,other[1]]-center[other[1]],verts[ids,other[0]]-center[other[0]]);ids=ids[np.argsort(angles)]
   normal=np.cross(verts[ids[1]]-verts[ids[0]],verts[ids[2]]-verts[ids[0]])
   if normal[axis]*side<0:ids=ids[::-1]
   for inds in([0,1,2],[0,2,3]):triangles.append((ids[inds]+start).tolist());owners.append(part_id);face_axis.append(axis);face_side.append(side)

def make_mesh(style,anchors):
 W,Hh,D=DIM;top=.022 if style=='WOOD_BOARD_BLOCK'else .018;bottom=.020 if style=='WOOD_BOARD_BLOCK'else .018
 n=5 if style=='WOOD_BOARD_BLOCK'else 7;bw=.140 if n==5 else .110;rw=.140 if n==5 else .110
 parts=[]
 def part(name,lo,hi):parts.append(dict(part_id=len(parts),name=name,bounds=[list(lo),list(hi)]))
 deck_intervals=[]
 for j,z in enumerate(np.linspace(-D/2+bw/2,D/2-bw/2,n)):
  zl,zh=(-D/2,-D/2+bw)if j==0 else((D/2-bw,D/2)if j==n-1 else(z-bw/2,z+bw/2));deck_intervals.append((zl,zh));part('top_deck_%02d'%j,[-W/2,-Hh/2,zl],[W/2,-Hh/2+top,zh])
 x_intervals=[(-W/2,-W/2+rw),(-rw/2,rw/2),(W/2-rw,W/2)];z_intervals=[deck_intervals[0],deck_intervals[n//2],deck_intervals[-1]]
 for ix,(xl,xh)in enumerate(x_intervals):
  for iz,(zl,zh)in enumerate(z_intervals):part(('wood_block_'if n==5 else'polymer_leg_')+'%d_%d'%(ix,iz),[xl,-Hh/2+top,zl],[xh,Hh/2-bottom,zh])
 for j,(xl,xh)in enumerate(x_intervals):part('bottom_runner_%d'%j,[xl,Hh/2-bottom,-D/2],[xh,Hh/2,D/2])
 vertices=[];triangles=[];owners=[];fa=[];fs=[]
 for p in parts:add_box(*np.array(p['bounds']),p['part_id'],vertices,triangles,owners,fa,fs)
 vertices=np.array(vertices,np.float64);triangles=np.array(triangles,np.int32);owners=np.array(owners,np.int32)
 # Float construction must still have exactly the canonical outer bound, not snapped anchors.
 assert np.allclose(vertices.min(0),-DIM/2,rtol=0,atol=2e-16)and np.allclose(vertices.max(0),DIM/2,rtol=0,atol=2e-16)
 incident=[];primary=[];patches=[];edge_samples=[];probes=[]
 for i,(anchor,sign)in enumerate(zip(anchors,SIGNS)):
  # Select by DECLARED constructive role, never nearest vertex / RGB / predicted q.
  chosen=(0 if sign[2]<0 else n-1)if sign[1]<0 else len(parts)-3+(0 if sign[0]<0 else 2)
  bounds=np.array(parts[chosen]['bounds']);assert np.all((anchor==bounds[0])|(anchor==bounds[1])),(style,i,chosen)
  ids=[p['part_id']for p in parts if np.any(np.all(vertices[8*p['part_id']:8*(p['part_id']+1)]==anchor,axis=1))]
  assert chosen in ids;incident.append(ids);primary.append(chosen)
  step=min(.030,float(np.min(bounds[1]-bounds[0]))*.7);away=-sign.astype(np.float64);quads=[];grid=[]
  for axis in range(3):
   axes=[j for j in range(3)if j!=axis];u=np.eye(3)[axes[0]]*away[axes[0]]*step;v=np.eye(3)[axes[1]]*away[axes[1]]*step
   quads.append([anchor,anchor+u,anchor+u+v,anchor+v]);grid.extend([anchor+a*u+b*v for a in(.2,.5,.8)for b in(.2,.5,.8)])
  edges=np.array([anchor+np.eye(3)[axis]*away[axis]*step*t for axis in range(3)for t in(.2,.5,.8)])
  patches.append(quads);edge_samples.append(edges);probes.append(np.concatenate([anchor[None],edges,np.array(grid)]))
 return dict(vertices=vertices,triangles=triangles,part_owner=owners,face_axis=np.array(fa,np.int8),face_side=np.array(fs,np.int8),native_anchor3D=anchors,native_primary_part_id=np.array(primary,np.int32),native_incident_part_ids=np.array(incident,np.int32),native_patch_quads=np.array(patches),native_edge_samples=np.array(edge_samples),native_witness_points=np.array(probes)),parts

def point_on_triangles(point,triangles):
 a=triangles[:,0];u=triangles[:,1]-a;v=triangles[:,2]-a;w=point-a;normal=np.cross(u,v);norm=np.linalg.norm(normal,axis=1)
 dot=lambda x,y:np.einsum('ij,ij->i',x,y)
 aa=dot(u,u);bb=dot(u,v);cc=dot(v,v);dd=dot(w,u);ee=dot(w,v);denom=aa*cc-bb*bb
 b=(cc*dd-bb*ee)/denom;c=(aa*ee-bb*dd)/denom
 return (np.abs(dot(normal,w))/norm<1e-12)&(b>=-1e-12)&(c>=-1e-12)&(b+c<=1+1e-12)

def first_hit(eye,point,triangles,owners):
 direction=point-eye;distance=float(np.linalg.norm(direction));direction/=distance
 v0=triangles[:,0];e1=triangles[:,1]-v0;e2=triangles[:,2]-v0;p=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,p);good=np.abs(det)>1e-14
 inv=np.zeros_like(det);inv[good]=1/det[good];tvec=eye-v0;u=np.einsum('ij,ij->i',tvec,p)*inv;q=np.cross(tvec,e1);v=q@direction*inv;t=np.einsum('ij,ij->i',e2,q)*inv
 good&=(u>=-1e-12)&(v>=-1e-12)&(u+v<=1+1e-12)&(t>1e-10)
 if not good.any():return dict(status='U',reason='RAY_MISS',owner=-1)
 nearest=float(t[good].min());tol=64*np.finfo(np.float64).eps*max(1.,abs(nearest));which=np.flatnonzero(good&(np.abs(t-nearest)<=tol));own=np.unique(owners[which])
 if len(own)!=1:return dict(status='U',reason='DIFFERENT_PART_DEPTH_TIE',owner=-1,owners=own.tolist())
 return dict(status='HIT',owner=int(own[0]),distance=nearest,reference_distance=distance,face_ids=which.tolist())

def classify_hits(hits,expected_part):
 out=[]
 for h in hits:
  if h['status']=='U':out.append('U');continue
  tol=1e-8*max(1.,h['reference_distance']);d=h['distance']-h['reference_distance']
  if abs(d)<=tol and h['owner']==expected_part:out.append('D')
  elif d< -tol:out.append('E'if h['owner']==999 else'S') # Same-part front surface can hide a back-facing endpoint too.
  else:out.append('U')
 return np.array(out)

def region_teacher(data,camera,raster,foreign_vertices=None):
 tri=data['vertices'][data['triangles']];own=data['part_owner'];scene=tri;sceneown=own
 if foreign_vertices is not None:scene=np.concatenate([tri,foreign_vertices[BOX_TRI]]);sceneown=np.r_[own,np.full(12,999,np.int32)]
 records=[]
 for i,points in enumerate(data['native_witness_points']):
  anchor=data['native_anchor3D'][i];normal=np.diag(SIGNS[i]);front=normal@(camera['eye_world']-anchor)>1e-12
  # Faces, edge endpoints, and junction are registered before camera / foreign geometry.
  active=[0]
  for axis in range(3):
   if front[[a for a in range(3)if a!=axis]].any():active.extend(range(1+3*axis,4+3*axis))
   if front[axis]:active.extend(range(10+9*axis,19+9*axis))
  active=np.array(sorted(set(active)),np.int32);pid=int(data['native_primary_part_id'][i])
  target_hits=[first_hit(camera['eye_world'],p,tri,own)for p in points];base=classify_hits(target_hits,pid)
  combined=classify_hits([first_hit(camera['eye_world'],p,scene,sceneown)for p in points],pid)
  project=raster.project_camera(raster.world_to_camera(points,camera),camera);in_frame=bool(((project>=0)&(project<=camera['size']-1)).all())
  geometric_S=bool(np.all(base=='S'))
  if geometric_S:before='S'
  elif front.sum()>=2 and np.all(base[active]=='D'):before='D'
  else:before='U'
  if not in_frame:state='U'
  elif before=='S':state='S'
  elif before!='D':state='U'
  elif np.all(combined[active]=='E'):state='E'
  elif np.all(combined[active]=='D'):state='D'
  else:state='U'
  records.append(dict(native_canonical_index=i,name=NAMES[i],primary_part_id=pid,incident_part_ids=data['native_incident_part_ids'][i].tolist(),projected_reference_xy=project[0].tolist(),all_region_witnesses_in_frame=in_frame,front_face_axes=np.flatnonzero(front).tolist(),active_witness_indices=active.tolist(),target_alone_region_state=before,geometric_S=geometric_S,state=state,target_witness_states=base.tolist(),combined_witness_states=combined.tolist(),target_counts={k:int((base==k).sum())for k in'DSEU'},combined_counts={k:int((combined==k).sum())for k in'DSEU'}))
 return records

# A real opaque camera-aligned cuboid; the same vertices are translated for SHAM.
BOX_TRI=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],np.int32)
def fixed_foreign(data,camera,raster):
 points=data['native_patch_quads'][1].reshape(-1,3);pc=raster.world_to_camera(points,camera);xy=raster.project_camera(pc,camera);lo=xy.min(0)-3;hi=xy.max(0)+3;z0=.55*float(pc[:,2].min());z1=z0+.025
 corners=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[lo[0],hi[1]],[hi[0],hi[1]]]);front=raster.unproject_xy(corners,z0,camera);back=front.copy();back[:,2]=z1;cover_cam=np.vstack([front,back]);cover=(cover_cam-camera['t'])@camera['R']
 delta=np.array([((hi[0]-lo[0])+64)*z1/camera['K'][0,0],0,0]);sham=(cover_cam+delta-camera['t'])@camera['R']
 assert np.allclose(sham-cover,np.broadcast_to(delta@camera['R'],cover.shape),rtol=0,atol=1e-15)
 return cover,sham,dict(fixed_target_native_index=1,projected_all3face_patch_bbox_xyxy=np.r_[lo,hi].tolist(),cover_front_camera_z=z0,physical_thickness_m=.025,sham_camera_translation=delta.tolist(),selection='Native1 fixed before geometry; analytic conservative patch bbox, no raster/image/quality search.')

def main():
 assert not(H/'RESULT.json').exists(),'Preserve existing constructive result.'
 design=dict(scope='New synthetic registered geometry only; canonical native correspondence constructed explicitly, no legacy-primary/real-human equivalence.',dimensions_xyz_m=DIM.tolist(),axes='canonical X-right/Y-down/Z-far; gravity-up=-Y; proper right/down/forward CV camera',styles=['WOOD_BOARD_BLOCK','POLYMER_RIB_LEG'],camera_eye=[1.8,-1.2,-2.5],camera_size=256,camera_fov_degrees=50,pose_yaw_degrees=0,patch_side='0.7*min(0.030m, minimum chosen-part side)',region_rule='D requires >=2 camera-facing incident face patches + registered endpoint/edge witnesses all target-direct. S requires whole registered region rays first blocked by target, including same-part front surface. E requires all originally D active witnesses first hit foreign; partial/ties/misses/off-frame U. Target-alone S takes precedence. q pixel/foreground presence and in-frame alone cannot give D/E.',foreign='Fixed native1; same opaque camera-aligned cuboid cover and translated sham; no render or search.',old_assets_modified=False,RGB_models_features_fit_private_current_V112_PnP=0)
 write(H/'DESIGN_BEFORE_GEOMETRY.json',design)
 anchors,rotations,perms=native_contract();raster=pure_module(RASTER);eye=np.array(design['camera_eye'],np.float64);forward=-eye/np.linalg.norm(eye);up=np.array([0.,-1.,0.]);right=np.cross(forward,up);right/=np.linalg.norm(right);down=np.cross(forward,right);R=np.stack([right,down,forward]);f=128/np.tan(np.deg2rad(25));camera=dict(size=256,K=np.array([[f,0,127.5],[0,f,127.5],[0,0,1.]],np.float64),R=R,t=-R@eye,eye_world=eye,near=.001)
 assert np.linalg.det(R)>0 and np.allclose(R@R.T,np.eye(3),atol=1e-14)
 results=[]
 for style in design['styles']:
  data,parts=make_mesh(style,anchors);tri=data['vertices'][data['triangles']];checks=[]
  for i in range(8):
   selected=tri[data['part_owner']==data['native_primary_part_id'][i]]
   assert all(point_on_triangles(p,selected).any()for p in data['native_witness_points'][i]);assert point_on_triangles(anchors[i],selected).any();checks.append(True)
  cover,sham,foreign=fixed_foreign(data,camera,raster);before=region_teacher(data,camera,raster);covered=region_teacher(data,camera,raster,cover);shammed=region_teacher(data,camera,raster,sham)
  filename=H/(style+'_REGISTERED_MESH.npz');np.savez_compressed(filename,**data,mesh_to_canonical=np.eye(4),dimensions_xyz_m=DIM,yaw_rotations=rotations,perm_camera_facing_to_canonical=perms,camera_K=camera['K'],camera_R=R,camera_t=camera['t'],foreign_cover_vertices=cover,foreign_sham_vertices=sham,foreign_triangles=BOX_TRI)
  record=dict(style=style,mesh=bind(filename),parts=parts,vertices=len(data['vertices']),triangles=len(data['triangles']),part_count=len(parts),all8_anchors_literal_mesh_vertices=True,all8_registered_region_witnesses_on_designated_part_triangles=all(checks),all4_signed_yaw_permutations_exact=True,mesh_to_canonical='identity; no fitted transform',foreign=foreign,states={stage:[r['state']for r in rec]for stage,rec in [('BEFORE',before),('COVER',covered),('SHAM',shammed)]},counts={stage:{s:sum(r['state']==s for r in rec)for s in'DSEU'}for stage,rec in [('BEFORE',before),('COVER',covered),('SHAM',shammed)]},fixed_native1_triplet=[before[1]['state'],covered[1]['state'],shammed[1]['state']],records=dict(BEFORE=before,COVER=covered,SHAM=shammed))
  results.append(record)
 result=dict(completed=True,PASS=all(r['all8_registered_region_witnesses_on_designated_part_triangles']and r['fixed_native1_triplet']==['D','E','D']for r in results),code=bind(__file__),design=bind(H/'DESIGN_BEFORE_GEOMETRY.json'),canonical_code=bind(CANONICAL),V93_geometry_projection_and_raster_code=bind(RASTER),native_feasibility=bind(V/'native_target_feasibility/RESULT.json'),registration='Constructed exact canonical8 to named physical-part endpoint and three incident face patches. Certified only for these new procedural assets; not a legacy-primary CAD/native-human claim.',results=results,next='Root may review mesh and conservative region teacher before any RGB appearance render. V93 raster can consume vertices/triangles/part_owner directly, using declared double-sided geometry and saved CV camera; ray witnesses give cue-level D/S/E and pixel owner masks alone are insufficient.',limitations=['Rectilinear board/rib/leg design only: no rounded moulding, fasteners, wear, texture, lighting, photorealism or transfer proof.','No historical mesh/corner identity, real-image native label or legacy pose generator reconstruction is claimed.','Exact source native reference registration does not establish real-data association or coordinate correctness.','Part contacts can have different-part depth ties; these remain U, never rounded/coerced.','Region teacher is a declared conservative synthetic convention; partial faces or a visible single face remain U, rather than automatically D or E.'],RGB_raster_images_encoder_model_fit_private_current_V112_PnP=0,original_assets_modified=False)
 write(H/'RESULT.json',result);print(json.dumps({k:result[k]for k in('completed','PASS','code','design')},ensure_ascii=False),flush=True);print(json.dumps([dict(style=r['style'],mesh=r['mesh'],states=r['states'],native1=r['fixed_native1_triplet'])for r in results]),flush=True)
if __name__=='__main__':main()
