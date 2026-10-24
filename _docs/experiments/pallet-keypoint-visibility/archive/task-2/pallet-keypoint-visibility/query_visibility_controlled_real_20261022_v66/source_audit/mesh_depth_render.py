"""CPU source-mesh z-buffer from supplied object-to-camera pose, no pose solve.

OpenCV camera convention: projected native pixel centers have integer x/y.
Rasterization samples those centers, with perspective-correct inverse-z.
This renders the supplied mesh/pose; it does not establish their accuracy.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _raster(camera_vertices,faces,K,height,width):
    depth=np.full((height,width),np.inf,np.float32)
    eps=1e-8
    for ti in range(len(faces)):
        a,b,c=faces[ti]
        za,zb,zc=camera_vertices[a,2],camera_vertices[b,2],camera_vertices[c,2]
        # Source fixture objects are entirely in front of the camera. A
        # triangle crossing the near plane is explicitly skipped, not clipped.
        if min(za,zb,zc)<=1e-6:continue
        ax=K[0,0]*camera_vertices[a,0]/za+K[0,1]*camera_vertices[a,1]/za+K[0,2]
        ay=K[1,0]*camera_vertices[a,0]/za+K[1,1]*camera_vertices[a,1]/za+K[1,2]
        bx=K[0,0]*camera_vertices[b,0]/zb+K[0,1]*camera_vertices[b,1]/zb+K[0,2]
        by=K[1,0]*camera_vertices[b,0]/zb+K[1,1]*camera_vertices[b,1]/zb+K[1,2]
        cx=K[0,0]*camera_vertices[c,0]/zc+K[0,1]*camera_vertices[c,1]/zc+K[0,2]
        cy=K[1,0]*camera_vertices[c,0]/zc+K[1,1]*camera_vertices[c,1]/zc+K[1,2]
        area=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(area)<1e-12:continue
        x0=max(0,int(np.ceil(min(ax,bx,cx)-eps)));x1=min(width-1,int(np.floor(max(ax,bx,cx)+eps)))
        y0=max(0,int(np.ceil(min(ay,by,cy)-eps)));y1=min(height-1,int(np.floor(max(ay,by,cy)+eps)))
        for y in range(y0,y1+1):
            for x in range(x0,x1+1):
                wa=((by-cy)*(x-cx)+(cx-bx)*(y-cy))/area
                wb=((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/area
                wc=1-wa-wb
                if wa>=-eps and wb>=-eps and wc>=-eps:
                    iz=wa/za+wb/zb+wc/zc
                    if iz>0:
                        z=1/iz
                        if z<depth[y,x]:depth[y,x]=z
    return depth


def render_depth(vertices,faces,K,object_to_camera,hw):
    vertices=np.asarray(vertices,np.float64);faces=np.asarray(faces,np.int64)
    K=np.asarray(K,np.float64);T=np.asarray(object_to_camera,np.float64)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():raise ValueError('finite verticesNx3')
    if faces.ndim!=2 or faces.shape[1]!=3 or faces.min()<0 or faces.max()>=len(vertices):raise ValueError('triangle facesMx3')
    if K.shape!=(3,3)or T.shape!=(4,4)or not np.isfinite(K).all()or not np.isfinite(T).all():raise ValueError('finite K3x3/T4x4')
    if not np.allclose(K[2],[0,0,1],atol=1e-8)or min(K[0,0],K[1,1])<=0:raise ValueError('OpenCV pinhole intrinsics')
    if not np.allclose(T[3],[0,0,0,1],atol=1e-6):raise ValueError('SE3 last row')
    R=T[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=1e-4)or not np.isclose(np.linalg.det(R),1,atol=1e-4):raise ValueError('SE3 rotation')
    h,w=map(int,hw)
    if min(h,w)<=0:raise ValueError('positive canvas')
    camera=np.ascontiguousarray(vertices@R.T+T[:3,3])
    z=_raster(camera,np.ascontiguousarray(faces),np.ascontiguousarray(K),h,w)
    crossing=((camera[faces,2].min(1)<=1e-6)&(camera[faces,2].max(1)>1e-6)).sum()
    return z,dict(vertices=len(vertices),triangles=len(faces),positive_depth_vertices=int((camera[:,2]>1e-6).sum()),near_plane_crossing_triangles=int(crossing),
                  projected_native_pixel_centers='integer x,y; inclusive triangle edges eps1e-8',new_PnP_calls=0,pose_optimization=False)


def smoke():
    # Two triangles cover the same pixel triangle at z=2 and z=3.
    vertices=np.array([[0,0,2],[8,0,2],[0,8,2],[0,0,3],[12,0,3],[0,12,3]],float)
    K=np.eye(3);T=np.eye(4)
    z,_=render_depth(vertices,np.array([[3,4,5],[0,1,2]]),K,T,[6,6])
    yy,xx=np.mgrid[:6,:6];inside=(xx+yy)<=4
    assert np.array_equal(np.isfinite(z),inside)
    assert np.array_equal(z[inside],np.full(inside.sum(),2,np.float32))
    # At barycentric (1/3,1/3,1/3), z must be harmonic, not affine.
    p=np.array([[0,0,1],[6,0,2],[0,12,4]],float)
    z,_=render_depth(p,np.array([[0,1,2]]),K,T,[4,4])
    assert np.isclose(z[1,1],1/((1+1/2+1/4)/3),atol=1e-7)
    return dict(PASS=True,tests=['occlusion_depth_order','integer_pixel_centers','perspective_inverse_z'],new_PnP_calls=0)

if __name__=='__main__':
    import json
    print(json.dumps(smoke()))
