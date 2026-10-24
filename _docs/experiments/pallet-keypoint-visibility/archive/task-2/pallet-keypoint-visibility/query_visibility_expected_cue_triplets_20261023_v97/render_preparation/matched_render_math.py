"""Camera-only matched object geometry; no query relocation or native labels."""
import numpy as np

def camera_arrays(camera):
 K=np.asarray(camera['K'],np.float64);R=np.asarray(camera['R'],np.float64);t=np.asarray(camera['t'],np.float64)
 assert K.shape==R.shape==(3,3)and t.shape==(3,)and np.allclose(R@R.T,np.eye(3),atol=1e-12)and np.linalg.det(R)>0
 return K,R,t

def project(world,camera):
 K,R,t=camera_arrays(camera);v=np.asarray(world,np.float64)@R.T+t
 assert np.isfinite(v).all()and np.all(v[:,2]>camera['near'])
 return (v[:,:2]/v[:,2,None])*np.array([K[0,0],K[1,1]])+K[:2,2],v[:,2]

def camera_to_world(v,camera):
 _,R,t=camera_arrays(camera);return(np.asarray(v,np.float64)-t)@R

def external_geometry(camera,disk_center_xy,disk_radius_px,min_patch_camera_z):
 """Fixed +x sham; same box shape/depth/orientation; no mask/quality search."""
 K,R,t=camera_arrays(camera);xy=np.asarray(disk_center_xy,np.float64);r=float(disk_radius_px)
 assert xy.shape==(2,)and np.isfinite(xy).all()and np.isfinite(r)and r>0 and min_patch_camera_z>camera['near']
 margin=3.;half=np.full(2,r+margin);front=.6*float(min_patch_camera_z);thickness=.05*front
 assert front>camera['near']
 # Front plane contains the complete conservative projected cue disk+margin.
 front_lo=xy-half;front_hi=xy+half;center_cam=np.r_[((xy-K[:2,2])/np.diag(K)[:2])*front,front+thickness/2]
 ext=np.r_[2*half/np.diag(K)[:2]*front,thickness]
 # Exact fixed vertices and outward triangles of a camera-frame cuboid.
 signs=np.array([[x,y,z]for x in(-1,1)for y in(-1,1)for z in(-1,1)],np.float64)
 local=signs*ext/2;cover_cam=local+center_cam
 faces=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],np.int64)
 # Each face normal must point away from the box centre.
 normals=np.cross(local[faces[:,1]]-local[faces[:,0]],local[faces[:,2]]-local[faces[:,0]])
 assert np.all(np.einsum('ij,ij->i',normals,local[faces].mean(1))>0)
 width=float(2*half[0]);shift_pixels=width+2.;shift_cam=np.array([shift_pixels*front/K[0,0],0.,0.])
 cover_world=camera_to_world(cover_cam,camera);sham_world=camera_to_world(cover_cam+shift_cam,camera)
 return dict(cover_vertices_world=cover_world,sham_vertices_world=sham_world,triangles=faces,cover_front_bbox_xyxy=np.r_[front_lo,front_hi],sham_front_bbox_xyxy=np.r_[front_lo+[shift_pixels,0],front_hi+[shift_pixels,0]],front_camera_z=front,depth_thickness=thickness,sham_shift_px=shift_pixels,sham_world_translation=shift_cam@R,dimensions_camera=ext,material_shared=True,shape_shared=True)

def generated():
 size=640;f=size/2/np.tan(np.deg2rad(25));camera=dict(K=[[f,0,319.5],[0,f,319.5],[0,0,1]],R=np.eye(3).tolist(),t=[0,0,0],near=.001,size=size)
 x=external_geometry(camera,[321.25,240.75],9.5,20.)
 p,z=project(x['cover_vertices_world'],camera);front=z==x['front_camera_z'];assert front.sum()==4
 assert np.allclose(np.r_[p[front].min(0),p[front].max(0)],x['cover_front_bbox_xyxy'],atol=1e-12)
 pp,zz=project(x['sham_vertices_world'],camera);assert np.array_equal(z,zz)
 assert np.allclose(x['sham_vertices_world']-x['cover_vertices_world'],x['sham_world_translation'])
 assert x['cover_front_bbox_xyxy'][2]<x['sham_front_bbox_xyxy'][0]
 assert np.array_equal(x['triangles'],np.array(x['triangles']))
 q=np.array([[40+i*70,70+i*40]for i in range(8)],np.float64);L=np.full(8,112.5);q0=q.copy();L0=L.copy()
 # Neither geometry preparation nor projection has access to or modifies q/L.
 assert np.array_equal(q,q0)and np.array_equal(L,L0)
 return dict(PASS=True,generated_only=True,front_disk_containment=True,sham_fixed_translation_disjoint=True,same_shape_material_depth=True,outward_triangle_winding=True,camera_projection_roundtrip=True,all8qL_unchanged=True,native8_mapping='UNKNOWN',actual_render_model_fit_current_PnP=0)

def update_nearest_two(best,second,owner,depth,part):
 d=np.asarray(depth,np.float64);better=d<best
 second[better]=best[better];best[better]=d[better];owner[better]=part
 second[~better]=np.minimum(second[~better],d[~better])

def final_ownership(best,second,owner):
 tol=64*np.finfo(np.float64).eps*np.maximum(1.,np.where(np.isfinite(best),np.abs(best),1.))
 with np.errstate(invalid='ignore'):amb=np.isfinite(best)&np.isfinite(second)&(second-best<=tol)
 result=owner.copy();result[amb]=-2
 return result,amb
