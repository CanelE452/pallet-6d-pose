"""Bounded expected structural-junction geometry teacher. GT-only, no rendering/ML.

A witness is one front-facing endpoint face patch, its two adjacent edge arms,
and the endpoint itself. Other back-facing incident faces do not veto it.
"""
import numpy as np

EPS=64*np.finfo(np.float64).eps
FACING_EPS=1e-10  # numerical grazing guard, not a selected decision threshold

def project(points,camera):
    p=np.asarray(points,np.float64);c=p@np.asarray(camera['R'],np.float64).T+np.asarray(camera['t'],np.float64)
    uv=(c@np.asarray(camera['K'],np.float64).T)[...,:2]/c[...,2:3]
    return uv,c[...,2]

def make_endpoint(local_vertices,local_to_world_row,endpoint_world,patch_radius):
    """True cuboid incident edges; never choose edges/part by visibility.

    Local source part must have exactly 8 unique min/max corner positions.
    Duplicate face vertices are collapsed. This is this AUX convention only.
    """
    v=np.unique(np.asarray(local_vertices,np.float64),axis=0);M=np.asarray(local_to_world_row,np.float64)
    if v.shape!=(8,3):raise ValueError('Eight unique cuboid corners required')
    lo=v.min(0);hi=v.max(0)
    if np.any(hi<=lo)or not np.isfinite(M).all()or patch_radius<=0:raise ValueError('Invalid cuboid geometry')
    if not np.all(np.isclose(v,lo,atol=1e-12,rtol=0)|np.isclose(v,hi,atol=1e-12,rtol=0)):raise ValueError('Not local box corners')
    world=(np.c_[v,np.ones(8)]@M)[:,:3];dist=np.linalg.norm(world-np.asarray(endpoint_world),axis=1);j=int(np.argmin(dist))
    if dist[j]>1e-9:raise ValueError('Frozen endpoint is not this source vertex')
    p=world[j];local=v[j];edges=[];normals=[];sign=[]
    for axis in range(3):
        other=local.copy();other[axis]=hi[axis]if local[axis]==lo[axis]else lo[axis]
        end=(np.r_[other,1.]@M)[:3];e=end-p;length=float(np.linalg.norm(e))
        edges.append(e/length*min(.25*length,float(patch_radius)))
        n=np.zeros(3);n[axis]=-1 if local[axis]==lo[axis]else 1
        n=np.linalg.inv(M[:3,:3])@n;n/=np.linalg.norm(n);normals.append(n);sign.append(int(-1 if local[axis]==lo[axis]else 1))
    edges=np.asarray(edges);grid=np.array([[a,b]for a in [1/6,.5,5/6]for b in [1/6,.5,5/6]],np.float64)
    quads=[];probes=[];face_edges=[]
    for axis in range(3):
        ab=[i for i in range(3)if i!=axis];u,w=edges[ab]
        quads.append([p,p+u,p+u+w,p+w]);probes.append(p+grid[:,:1]*u+grid[:,1:]*w);face_edges.append(ab)
    edge_probes=p+np.array([1/6,.5,5/6])[None,:,None]*edges[:,None,:]
    return dict(endpoint_world=p,edge_vectors=edges,edge_lengths=np.linalg.norm(edges,axis=1),face_normals=np.array(normals),
        face_patch_quads_world=np.array(quads),face_probe_points=np.array(probes),edge_probe_points=edge_probes,face_edge_axes=np.array(face_edges),local_endpoint_sign=np.array(sign))

class RayScene:
    """Opaque original winding + per-triangle two-sided rules; final-min tiesU."""
    def __init__(self,triangles,owner,orientation=None,double_sided=None,foreign_owner_min=308):
        self.xyz=np.asarray(triangles,np.float64);self.owner=np.asarray(owner,np.int32)
        if self.xyz.ndim!=3 or self.xyz.shape[1:]!=(3,3)or len(self.xyz)!=len(self.owner)or not np.isfinite(self.xyz).all():raise ValueError('Invalid triangles')
        self.a=self.xyz[:,0];self.e1=self.xyz[:,1]-self.a;self.e2=self.xyz[:,2]-self.a
        self.orientation=np.ones(len(self.xyz))if orientation is None else np.asarray(orientation,np.float64)
        self.double_sided=np.zeros(len(self.xyz),bool)if double_sided is None else np.asarray(double_sided,bool)
        self.foreign_owner_min=foreign_owner_min
    def cast(self,eye,points,expected_owner):
        eye=np.asarray(eye,np.float64);p=np.asarray(points,np.float64).reshape(-1,3);out=[]
        for target in p:
            direction=target-eye;distance=float(np.linalg.norm(direction))
            if not np.isfinite(distance)or distance<=0:out.append(dict(state='U',owners=[],margin=None));continue
            direction/=distance;h=np.cross(np.broadcast_to(direction,self.e2.shape),self.e2);det=np.einsum('ij,ij->i',self.e1,h)
            detscale=np.linalg.norm(self.e1,axis=1)*np.linalg.norm(self.e2,axis=1)
            legal=(np.abs(det)>EPS*np.maximum(1,detscale))&(self.double_sided|(det*self.orientation>0))
            safe=np.where(legal,det,1.);s=eye-self.a;u=np.einsum('ij,ij->i',s,h)/safe;q=np.cross(s,self.e1)
            v=np.einsum('j,ij->i',direction,q)/safe;depth=np.einsum('ij,ij->i',self.e2,q)/safe
            # Probe points are interior. Exact barycentric edge coverage is inclusive;
            # same-owner triangle ties do not create a different-owner ambiguity.
            hit=legal&(u>=-EPS)&(v>=-EPS)&(u+v<=1+EPS)&(depth>0)
            nearest=float(depth[hit].min())if hit.any()else None;tol=EPS*max(1,distance,nearest or 0)
            owners=[]if nearest is None else sorted(set(int(x)for x in self.owner[hit&(np.abs(depth-nearest)<=tol)]))
            state='U';margin=None if nearest is None else distance-nearest
            if len(owners)==1:
                if abs(margin)<=tol and owners[0]==expected_owner:state='DIRECT'
                elif margin>tol:state='FOREIGN'if owners[0]>=self.foreign_owner_min else'SELF'
            out.append(dict(state=state,owners=owners,margin=margin))
        return out

def evaluate_endpoint(cue,camera,target_scene,combined_scene=None):
    """All GT geometry only; q/L are retained solely to mark crop availability.

    UnderlyingD/S/E is separate from finalU for query/frame/crop miss. The
    endpoint patch definition is fixed in object geometry, not projected-mask
    selection. No actual image or learned feature is read here.
    """
    eye=np.asarray(camera['eye_world']);e=cue['geometry'];pid=int(cue['part_id']);endpoint=np.asarray(e['endpoint_world'])
    facepoints=np.asarray(e['face_probe_points']);edgepoints=np.asarray(e['edge_probe_points']);normals=np.asarray(e['face_normals'])
    centres=facepoints.mean(1);directions=eye-centres;directions/=np.linalg.norm(directions,axis=1)[:,None]
    facing=np.einsum('ij,ij->i',normals,directions);eligible=np.flatnonzero(facing>FACING_EPS).tolist()
    points=np.concatenate([endpoint[None],facepoints.reshape(-1,3),edgepoints.reshape(-1,3)])
    source=target_scene.cast(eye,points,pid);after=source if combined_scene is None else combined_scene.cast(eye,points,pid)
    witness_indices={f:[0]+list(range(1+9*f,1+9*(f+1)))+sum([list(range(28+3*a,28+3*(a+1)))for a in e['face_edge_axes'][f]],[])for f in eligible}
    def allstate(records,inds,state):return all(records[i]['state']==state for i in inds)
    direct=[f for f,inds in witness_indices.items()if allstate(source,inds,'DIRECT')]
    if direct:
        if any(allstate(after,witness_indices[f],'DIRECT')for f in direct):underlying='D';reason='COMPLETE_FRONT_FACING_JUNCTION_WITNESS_VISIBLE'
        elif all(allstate(after,witness_indices[f],'FOREIGN')for f in direct):underlying='E';reason='ALL_ORIGINAL_DIRECT_JUNCTION_WITNESSES_FOREIGN_COVERED'
        else:underlying='U';reason='PARTIAL_FOREIGN_OR_COMPETING_JUNCTION_WITNESS'
    elif witness_indices and all(allstate(source,inds,'SELF')for inds in witness_indices.values()):underlying='S';reason='ALL_FRONT_FACING_JUNCTION_WITNESSES_SELF_BLOCKED'
    elif not eligible and np.all(facing<-FACING_EPS)and all(x['state']=='SELF'for x in source):underlying='S';reason='CONVEX_ENDPOINT_BACK_FACING_AND_RAYS_SELF_BLOCKED'
    else:underlying='U';reason='PARTIAL_SELF_GRAZING_OR_COMPETING_JUNCTION_WITNESS'
    vertices=np.concatenate([np.asarray(e['face_patch_quads_world']).reshape(-1,3),endpoint[None]])
    uv,z=project(vertices,camera);q=np.asarray(cue['original_q'],np.float64);L=float(cue['L']);H,W=camera['size'],camera['size']
    positive=bool(np.isfinite(uv).all()and np.isfinite(z).all()and(z>camera.get('near',0)).all())
    inframe=bool(positive and((uv>=0)&(uv<np.array([W,H]))).all());incrop=bool(positive and((uv-q>=-.5*L)&(uv-q<.5*L)).all())
    final=underlying if inframe and incrop else'U';finalreason=reason if inframe and incrop else'QUERY_CROP_OR_FRAME_SUPPORT_MISS'
    endpoint_uv,endpoint_z=project(endpoint[None],camera);radius=float(np.linalg.norm(uv-endpoint_uv[0],axis=1).max())
    return dict(state=final,underlying_geometric_state=underlying,reason=finalreason,underlying_reason=reason,
        front_facing_faces=eligible,face_cosines=facing.tolist(),original_direct_witness_faces=direct,
        target_only_probe_states=source,combined_probe_states=after,full_patch_positive_depth=positive,full_patch_in_frame=inframe,full_patch_in_original_crop=incrop,
        projected_endpoint=endpoint_uv[0].tolist(),endpoint_camera_z=float(endpoint_z[0]),projected_support_disk=dict(center=endpoint_uv[0].tolist(),radius=radius),
        all_patch_camera_z_min=float(z.min()),all_patch_projected_vertices=uv.tolist(),native8_mapping='UNKNOWN',human_gold=False)
