"""Physical axis-aligned CAD creases, distinct from virtual cuboid role lines."""
from context32 import *
import numpy as np,cv2,trimesh
from plpose_local.geometry import EDGES

def creases(mesh):
    # Weld coincident tessellation vertices, then reject coplanar triangle seams.
    m=trimesh.Trimesh(mesh.vertices.copy(),mesh.faces.copy(),process=True)
    edges=m.face_adjacency_edges[m.face_adjacency_angles>np.deg2rad(35)]
    unique,count=np.unique(np.sort(m.edges,axis=1),axis=0,return_counts=True)
    edges=np.concatenate((edges,unique[count==1]));segments=m.vertices[edges]
    delta=segments[:,1]-segments[:,0];length=np.linalg.norm(delta,axis=1);axis=np.abs(delta).argmax(-1)
    keep=(length>.002)&(np.abs(delta).max(-1)/np.maximum(length,1e-9)>.995)
    segments=segments[keep];axis=axis[keep];merged=[]
    # Merge touching collinear triangle-edge pieces; their tessellation endpoints
    # must not become physical endpoint supervision.
    groups={}
    for segment,a in zip(segments,axis):
        other=[k for k in range(3) if k!=a];key=(int(a),*np.round(segment.mean(0)[other],5))
        groups.setdefault(key,[]).append(segment)
    for key,ss in groups.items():
        a=key[0];other=[k for k in range(3) if k!=a];intervals=sorted((float(s[:,a].min()),float(s[:,a].max())) for s in ss);current=list(intervals[0])
        ranges=[]
        for lo,hi in intervals[1:]:
            if lo<=current[1]+1e-4:current[1]=max(hi,current[1])
            else:ranges.append(current);current=[lo,hi]
        ranges.append(current)
        for lo,hi in ranges:
            if hi-lo<.015:continue
            seg=np.zeros((2,3));seg[:,other]=np.array(key[1:]);seg[:,a]=[lo,hi];merged.append(seg)
    assert merged
    return np.array(merged),dict(triangles=len(m.faces),raw_crease_edges=len(edges),merged_axis_segments=len(merged),min_segment_m=.015,crease_angle_deg=35)

def assign_roles(segments,world):
    refs=world[np.array(EDGES)];rd=refs[:,1]-refs[:,0];axis=np.abs(rd).argmax(-1);sd=segments[:,1]-segments[:,0];sa=np.abs(sd).argmax(-1)
    role=[];distance=[];limit=.04*np.linalg.norm(np.ptp(world,axis=0))
    for seg,a in zip(segments,sa):
        mid=seg.mean(0);t=((mid-refs[:,0])*rd).sum(-1)/np.maximum((rd*rd).sum(-1),1e-9);projection=refs[:,0]+np.clip(t,0,1)[:,None]*rd;d=np.linalg.norm(mid-projection,axis=-1);d[axis!=a]=np.inf;j=int(d.argmin())
        role.append(j if d[j]<=limit else -1);distance.append(float(d[j]))
    return np.array(role),np.array(distance),limit

def projected(points,T,K):
    camera=(points-T[:3,3])@T[:3,:3]@np.diag([1,-1,-1]);p=camera@K.T
    return p[:,:2]/p[:,2,None],camera[:,2]

def visibility(uv,z,depth):
    h,w=depth.shape;xy=np.rint(uv).astype(int);inside=(xy[:,0]>=1)&(xy[:,0]<w-1)&(xy[:,1]>=1)&(xy[:,1]<h-1)&(z>0);ok=np.zeros(len(z),bool)
    ii=np.where(inside)[0]
    if len(ii):
        sample=np.stack([depth[xy[ii,1]+dy,xy[ii,0]+dx] for dx,dy in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]],-1)
        difference=np.abs(sample-z[ii,None]);difference[sample<=0]=np.inf
        ok[ii]=difference.min(-1)<(.003+.001*z[ii])
    return ok

def labels(segments,world,T,K,object_depth,scene_depth):
    role,distance,limit=assign_roles(segments,world);physical=[]
    for index,(seg,r) in enumerate(zip(segments,role)):
        if r<0:continue
        # Metric sampling prevents long edges receiving sparse visibility probes.
        n=max(32,int(np.ceil(np.linalg.norm(seg[1]-seg[0])/.003))+1);u=np.linspace(0,1,n);xyz=seg[0]+u[:,None]*(seg[1]-seg[0]);uv,z=projected(xyz,T,K)
        self_visible=visibility(uv,z,object_depth);visible=self_visible&visibility(uv,z,scene_depth)
        physical.append(dict(segment=index,role=int(r),world=seg.tolist(),uv=uv.tolist(),depth=z.tolist(),visible=visible.tolist(),self_visible=self_visible.tolist(),endpoint_visible=[bool(visible[0]),bool(visible[-1])],role_offset_m=float(distance[index])))
    return dict(segments=physical,role_distance_limit_m=limit,unassigned_segments=int((role<0).sum()),physical_visible_samples=sum(sum(r['visible']) for r in physical),physical_visible_endpoints=sum(sum(r['endpoint_visible']) for r in physical),definition='Actual welded CAD crease chains; only axis-aligned segments within4% cuboid diagonal of a parallel cuboid edge receive that camera-facing role. Role relation is geometric proximity, not exact coincidence or independently annotated semantic identity. Ends are physical chain ends, not crop/occluder boundaries. Unassigned/internal creases are not claimed as target outer boundaries.')

def roi_targets(label,world_uv,box,origin,hw,mask=None,size=48):
    physical=np.zeros((24,size,size),np.float32);virtual=np.zeros_like(physical);x1,y1,x2,y2=box;scale=(size-1)/np.maximum(np.array([x2-x1,y2-y1]),1);h,w=hw
    def inview(uv):
        q=uv-np.array(origin);ok=(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
        if mask is not None:
            a,u,b,v=mask;ok &= ~((uv[:,0]>=a)&(uv[:,0]<b)&(uv[:,1]>=u)&(uv[:,1]<v))
        return q,ok
    def stamp(out,role,uv,ok,endpoints):
        q,inside=inview(uv);xy=(q-[x1,y1])*scale
        for k in np.where(ok&inside)[0]:
            x,y=np.rint(xy[k]).astype(int)
            if 0<=x<size and 0<=y<size:cv2.circle(out[role],(int(x),int(y)),1,1.,-1)
        for k in endpoints:
            if ok[k] and inside[k]:
                x,y=np.rint(xy[k]).astype(int)
                if 0<=x<size and 0<=y<size:cv2.circle(out[12+role],(int(x),int(y)),2,1.,-1)
    for seg in label['segments']:
        uv=np.array(seg['uv']);stamp(physical,seg['role'],uv,np.array(seg['visible']),[0,len(uv)-1])
    for role,(a,b) in enumerate(EDGES):
        uv=np.linspace(world_uv[a],world_uv[b],300);stamp(virtual,role,uv,np.ones(300,bool),[0,299])
    return physical,virtual
