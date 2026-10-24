"""Deterministic diagnostic CPU raster; asset-local ownership, never semantic/native gold."""
import numpy as np
TIE_ULPS=64

def camera_from_bounds(bounds,direction,size=256,fov_degrees=50.):
 b=np.asarray(bounds,np.float64);direction=np.asarray(direction,np.float64)
 assert b.shape==(2,3) and np.isfinite(b).all() and np.all(b[1]>b[0])
 center=(b[0]+b[1])*.5;radius=float(np.linalg.norm((b[1]-b[0])*.5));assert radius>0
 distance=1.3*radius/np.sin(np.deg2rad(fov_degrees*.5))
 outward=direction/np.linalg.norm(direction);eye=center+outward*distance
 forward=(center-eye)/distance;up=np.array([0.,0.,1.]) if abs(outward[1])>.5 else np.array([0.,1.,0.])
 right=np.cross(forward,up);right/=np.linalg.norm(right);down=np.cross(forward,right)
 R=np.stack([right,down,forward]);t=-R@eye
 assert np.allclose(R@R.T,np.eye(3),atol=1e-14) and np.linalg.det(R)>0
 focal=(size/2.)/np.tan(np.deg2rad(fov_degrees/2.));principal=(size-1)/2.
 return dict(size=int(size),fov_degrees=float(fov_degrees),K=np.array([[focal,0,principal],[0,focal,principal],[0,0,1.]],np.float64),R=R,t=t,eye_world=eye,target_world=center,radius=radius,distance=distance,near=radius*1e-4,world_up=up,direction=outward)

def world_to_camera(vertices,camera):return np.asarray(vertices,np.float64)@camera['R'].T+camera['t']
def project_camera(vertices,camera):
 v=np.asarray(vertices,np.float64);assert np.isfinite(v).all() and np.all(v[:,2]>0)
 return (v[:,:2]/v[:,2,None])*np.array([camera['K'][0,0],camera['K'][1,1]])+camera['K'][:2,2]
def box_corners(bounds):
 b=np.asarray(bounds,np.float64)
 return np.array([[b[x,0],b[y,1],b[z,2]]for x in [0,1]for y in [0,1]for z in [0,1]],np.float64)
def unproject_xy(xy,z,camera):
 xy=np.asarray(xy,np.float64)
 return np.column_stack([(xy[:,0]-camera['K'][0,2])*z/camera['K'][0,0],(xy[:,1]-camera['K'][1,2])*z/camera['K'][1,1],np.full(len(xy),z)])
def foreign_rectangle(part_bounds,camera):
 pc=world_to_camera(box_corners(part_bounds),camera);assert np.min(pc[:,2])>camera['near']
 xy=project_camera(pc,camera);lo=xy.min(0);hi=xy.max(0);center=world_to_camera(np.mean(np.asarray(part_bounds),axis=0)[None],camera)[0]
 z=.5*float(center[2]);assert z>camera['near']
 rect=np.array([[lo[0]-2,lo[1]-2],[lo[0]+.6*(hi[0]-lo[0])+2,lo[1]-2],[lo[0]+.6*(hi[0]-lo[0])+2,hi[1]+2],[lo[0]-2,hi[1]+2]],np.float64)
 verts=unproject_xy(rect,z,camera)
 return dict(vertices_camera=verts,triangles=np.array([[0,1,2],[0,2,3]],np.int64),projected_part_AABB=np.r_[lo,hi],rectangle_xy=rect,z=z,part_center_camera=center)

def clip_near(triangle,near):
 out=[];poly=list(np.asarray(triangle,np.float64));prior=poly[-1];pin=prior[2]>=near
 for cur in poly:
  cin=cur[2]>=near
  if cin!=pin:
   a=(near-prior[2])/(cur[2]-prior[2]);v=prior+a*(cur-prior);v[2]=near;out.append(v)
  if cin:out.append(cur.copy())
  prior=cur;pin=cin
 return np.asarray(out,np.float64).reshape(-1,3)
def edge(a,b,x,y):return (b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])
def top_left(a,b):
 dy=b[1]-a[1];dx=b[0]-a[0]
 return (dy<0)or(dy==0 and dx>0)

def fragments(triangle_camera,camera):
 """Integer pixel centres; positive-area ordering only AFTER orientation culling."""
 poly=clip_near(triangle_camera,camera['near']);n=camera['size']
 for j in range(1,len(poly)-1):
  v=poly[[0,j,j+1]];s=project_camera(v,camera);area=float(edge(s[0],s[1],s[2,0],s[2,1]))
  if not np.isfinite(area) or area==0:continue
  if area<0:s=s[[0,2,1]];v=v[[0,2,1]];area=-area
  xmin=max(0,int(np.ceil(np.min(s[:,0]))));xmax=min(n-1,int(np.floor(np.max(s[:,0]))))
  ymin=max(0,int(np.ceil(np.min(s[:,1]))));ymax=min(n-1,int(np.floor(np.max(s[:,1]))))
  if xmin>xmax or ymin>ymax:continue
  yy,xx=np.mgrid[ymin:ymax+1,xmin:xmax+1];a=edge(s[1],s[2],xx,yy);b=edge(s[2],s[0],xx,yy);c=edge(s[0],s[1],xx,yy)
  valid=((a>0)|((a==0)&top_left(s[1],s[2])))&((b>0)|((b==0)&top_left(s[2],s[0])))&((c>0)|((c==0)&top_left(s[0],s[1])))
  if not np.any(valid):continue
  inverse=(a[valid]/v[0,2]+b[valid]/v[1,2]+c[valid]/v[2,2])/area
  assert np.isfinite(inverse).all() and np.all(inverse>0)
  yield yy[valid],xx[valid],1./inverse

def rasterize_parts(vertices_camera,triangles,part_ids,face_ids,double_sided,orientation,camera,num_parts):
 """One independent nearest surface layer per declared target part; no inter-part hiding yet."""
 v=np.asarray(vertices_camera,np.float64);tri=np.asarray(triangles,np.int64);pid=np.asarray(part_ids,np.int32);fid=np.asarray(face_ids,np.int32)
 ds=np.asarray(double_sided,bool);ori=np.asarray(orientation,np.int8)
 assert tri.shape==(len(pid),3) and fid.shape==ds.shape==ori.shape==pid.shape
 assert np.isfinite(v).all() and np.all(tri>=0) and np.all(tri<len(v)) and np.all((pid>=0)&(pid<num_parts)) and np.all(np.isin(ori,[-1,1]))
 n=camera['size'];depth=np.full((num_parts,n,n),np.inf,np.float64);face=np.full((num_parts,n,n),-1,np.int32)
 xyz=v[tri];normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);normal_norm=np.linalg.norm(normal,axis=1)
 # CV right/down/forward det+1: rightHanded front normal points toward origin => dot(n,p)<0.
 facing=np.einsum('ij,ij->i',normal,xyz.mean(1))*ori<0
 active=(ds|facing)&(normal_norm>0)
 stats=dict(input_faces=len(tri),culled_faces=int(np.sum(~ds&~facing)),zero_area_faces=int(np.sum(normal_norm==0)),near_straddling_faces=int(np.sum((xyz[:,:,2].min(1)<camera['near'])&(xyz[:,:,2].max(1)>=camera['near']))),behind_near_faces=int(np.sum(xyz[:,:,2].max(1)<camera['near'])),faces_rasterized=0,fragments=0)
 for i in np.flatnonzero(active):
  if np.max(xyz[i,:,2])<camera['near']:continue
  has=False
  for yy,xx,z in fragments(xyz[i],camera):
   old=depth[pid[i],yy,xx];oldface=face[pid[i],yy,xx];better=z<old;equal=z==old
   if np.any(better):depth[pid[i],yy[better],xx[better]]=z[better];face[pid[i],yy[better],xx[better]]=fid[i]
   tie=equal&((oldface<0)|(fid[i]<oldface))
   if np.any(tie):face[pid[i],yy[tie],xx[tie]]=fid[i]
   has=True;stats['fragments']+=len(z)
  stats['faces_rasterized']+=int(has)
 return depth,face,stats

def resolve_layers(depth,face,part_ids=None,foreign=None):
 """Fixed final nearest depth determines eligibility; same-part face ties are NOT ownership U."""
 assert depth.shape==face.shape and depth.ndim==3
 ids=np.arange(len(depth),dtype=np.int32) if part_ids is None else np.asarray(part_ids,np.int32)
 assert len(ids)==len(depth) and len(set(ids.tolist()))==len(ids)
 best=np.min(depth,axis=0)
 if foreign is not None:best=np.minimum(best,foreign[0])
 tol=TIE_ULPS*np.finfo(np.float64).eps*np.maximum(1.,np.where(np.isfinite(best),np.abs(best),1.))
 owner=np.full(best.shape,-1,np.int32);chosen_face=np.full(best.shape,-1,np.int32);ambiguity=np.zeros(best.shape,bool);tie_parts=np.zeros(best.shape,np.uint16)
 layers=[(int(p),depth[i],face[i]) for i,p in enumerate(ids)]
 if foreign is not None:layers.append((int(foreign[2]),foreign[0],foreign[1]))
 for p,d,f in sorted(layers,key=lambda x:x[0]):
  with np.errstate(invalid='ignore'):eligible=np.isfinite(d)&np.isfinite(best)&(np.abs(d-best)<=tol)
  fresh=eligible&(owner<0);ambiguity|=eligible&(owner>=0)&(owner!=p);tie_parts+=eligible.astype(np.uint16)
  owner[fresh]=p;chosen_face[fresh]=f[fresh]
 return dict(depth=np.where(np.isfinite(best),best,np.nan),part_id=owner,face_id=chosen_face,ambiguity=ambiguity,tie_part_count=tie_parts)

def partition_selected(PA,PS,PV,selected=152,foreign=308):
 A=(PA['part_id']==selected)&~PA['ambiguity'];S=A&(PS['part_id']==selected)&~PS['ambiguity'];V=S&(PV['part_id']==selected)&~PV['ambiguity']
 self_hidden=A&(PS['part_id']>=0)&(PS['part_id']!=selected)&~PS['ambiguity']
 external=S&(PV['part_id']==foreign)&~PV['ambiguity'];unknown=A&~(self_hidden|external|V)
 assert not np.any(S&~A) and not np.any(V&~S)
 assert np.all((self_hidden.astype(np.int8)+external+V+unknown)[A]==1)
 impossible=S&(PV['part_id']>=0)&(PV['part_id']!=selected)&(PV['part_id']!=foreign)&~PV['ambiguity'];assert not np.any(impossible)
 state=np.full(A.shape,-2,np.int8);state[unknown]=-1;state[V]=0;state[external]=1;state[self_hidden]=2
 return dict(A_selected_alone=A,S_selected_in_target=S,V_selected_with_foreign=V,self_hidden_other_target=self_hidden,foreign_blocked=external,unknown=unknown,partition_state=state)
