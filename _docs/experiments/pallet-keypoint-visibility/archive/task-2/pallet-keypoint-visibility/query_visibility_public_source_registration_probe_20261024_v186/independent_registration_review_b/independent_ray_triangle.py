"""Pure invented-data camera-ray math; no file loader, renderer, Torch or model."""
import math
import numpy as np

NEAR_MM=1e-6
EDGE_BARY_EPS=1e-9
CALC_DEPTH_ABS_MM=1e-6


def require(x,msg):
    if not x:raise AssertionError(msg)


def camera_vertices(vertices,R,t):
    v=np.asarray(vertices,np.float64);R=np.asarray(R,np.float64);t=np.asarray(t,np.float64)
    require(v.ndim==2 and v.shape[1]==3 and R.shape==(3,3) and t.shape==(3,), 'Original vertex/R/t shapes')
    require(np.isfinite(v).all() and np.isfinite(R).all() and np.isfinite(t).all(),'Original finite geometry')
    return v@R.T+t


def camera_ray(K,u,v):
    K=np.asarray(K,np.float64);require(K.shape==(3,3) and np.isfinite(K).all(),'Original finite K')
    ray=np.linalg.solve(K,np.array([float(u),float(v),1.],np.float64))
    require(np.isfinite(ray).all() and ray[2]!=0,'Valid pinhole ray')
    return ray/ray[2]


def first_hit(vertices_cam,triangles,ray):
    vertices=np.asarray(vertices_cam,np.float64);faces=np.asarray(triangles,np.int64);direction=np.asarray(ray,np.float64)
    require(vertices.ndim==2 and vertices.shape[1]==3 and faces.ndim==2 and faces.shape[1]==3 and direction.shape==(3,),'Independent ray/triangle shapes')
    require(np.isfinite(vertices).all() and np.isfinite(direction).all() and direction[2]==1,'Camera-axis Z ray convention')
    require((faces>=0).all() and (faces<len(vertices)).all(),'Original triangle indices')
    best=None;parallel=nearparallel=degenerate=boundary_candidates=0;hits=[]
    for face_index,(ia,ib,ic) in enumerate(faces):
        a,b,c=vertices[ia],vertices[ib],vertices[ic];e1=b-a;e2=c-a;normal=np.cross(e1,e2)
        if float(np.dot(normal,normal))==0:
            degenerate+=1;continue
        p=np.cross(direction,e2);det=float(np.dot(e1,p))
        if det==0:
            parallel+=1;continue
        relative_bound=64*np.finfo(np.float64).eps*float(np.linalg.norm(e1))*float(np.linalg.norm(e2))*float(np.linalg.norm(direction))
        is_nearparallel=abs(det)<=relative_bound
        nearparallel+=int(is_nearparallel)
        inv=1/det;s=-a;u=float(np.dot(s,p)*inv);q=np.cross(s,e1);v=float(np.dot(direction,q)*inv);z=float(np.dot(e2,q)*inv)
        if not all(math.isfinite(x) for x in (u,v,z)) or z<NEAR_MM:continue
        w=1-u-v
        near_boundary=(u>=-EDGE_BARY_EPS and v>=-EDGE_BARY_EPS and w>=-EDGE_BARY_EPS
                       and min(abs(u),abs(v),abs(w))<=EDGE_BARY_EPS)
        boundary_candidates+=int(near_boundary)
        if u<0 or v<0 or w<0:continue
        hit=dict(triangle_index=face_index,camera_Z_mm=z,barycentric=[w,u,v],surface_boundary=near_boundary,near_parallel=is_nearparallel)
        hits.append(hit)
        if best is None or z<best['camera_Z_mm']:best=hit
    return dict(hit=best,positive_triangle_hits=len(hits),parallel_triangles=parallel,near_parallel_triangles=nearparallel,
        degenerate_triangles=degenerate,near_boundary_candidates=boundary_candidates,
        exact_min_Z_ties=sum(x['camera_Z_mm']==best['camera_Z_mm'] for x in hits) if best else 0)


def depth_bound(dtype,independent_z):
    dtype=np.dtype(dtype)
    require(dtype in (np.dtype(np.float32),np.dtype(np.float64)),'Saved renderer depth must be float32/float64')
    if dtype==np.dtype(np.float64):return CALC_DEPTH_ABS_MM
    x=np.float32(independent_z);require(np.isfinite(x),'Representable F32 positive depth')
    above=float(np.nextafter(x,np.float32(np.inf),dtype=np.float32))-float(x)
    below=float(x)-float(np.nextafter(x,np.float32(-np.inf),dtype=np.float32))
    return CALC_DEPTH_ABS_MM+.5*max(abs(above),abs(below))


def fixed_grid_pixels(hw):
    h,w=map(int,hw);require(h>0 and w>0,'Original positive H/W')
    xs=np.floor(np.linspace(0,w-1,9,dtype=np.float64)+.5).astype(np.int64)
    ys=np.floor(np.linspace(0,h-1,7,dtype=np.float64)+.5).astype(np.int64)
    return [(int(x),int(y)) for y in ys for x in xs]


def fixed_face_indices(number_faces):
    require(number_faces>0,'Original nonempty triangle mesh')
    return np.floor(np.linspace(0,number_faces-1,17,dtype=np.float64)+.5).astype(np.int64).tolist()


def invented_checks():
    passed=[]
    verts=np.array([[-2,-2,2],[2,-2,2],[0,2,2]],np.float64);faces=np.array([[0,1,2]],np.int64)
    hit=first_hit(verts,faces,[0,0,1]);require(hit['hit']['camera_Z_mm']==2 and not hit['hit']['surface_boundary'],'Frontal interior')
    passed.append('Frontal_integer_camera_ray_Z2mm_firsthit')
    slant=np.array([[0,0,2],[4,0,4],[0,4,4]],np.float64);hit=first_hit(slant,faces,[.25,.25,1])
    require(math.isclose(hit['hit']['camera_Z_mm'],8/3,rel_tol=0,abs_tol=1e-14) and abs(hit['hit']['camera_Z_mm']-3)>.3,'Oblique perspective, not screen-linearZ')
    passed.append('Oblique_perspective_Z_matches_analytic8over3_not_screenlinear3')
    two=np.concatenate([verts*2,verts]);hit=first_hit(two,np.array([[0,1,2],[5,4,3]]),[0,0,1]);require(hit['hit']['camera_Z_mm']==2 and hit['hit']['triangle_index']==1,'Nearest bothwindings')
    passed.append('Both_face_windings_and_frontmost_of_overlapping_triangles')
    hit=first_hit(verts,np.array([[0,1,2],[2,1,0]]),[0,0,1]);require(hit['hit']['triangle_index']==0 and hit['exact_min_Z_ties']==2,'Exact ties keep original faceorder')
    passed.append('Exact_depth_tie_original_first_triangle_winding_retained')
    require(first_hit(verts,faces,[10,10,1])['hit'] is None,'Nohit')
    hit=first_hit(np.array([[-1,-1,2],[1,-1,2],[0,1,2]],np.float64),faces,[0,.5,1]);require(hit['hit'] is not None and hit['hit']['surface_boundary'],'Boundary hit preserved')
    passed.append('Outside_nohit_and_exact_vertex_boundary_separate')
    vertical=np.array([[2,-1,1],[2,1,1],[2,0,3]],np.float64);h=first_hit(vertical,faces,[0,0,1]);require(h['hit'] is None and h['parallel_triangles']==1,'Parallel')
    h=first_hit(np.array([[0,0,2],[1,0,2],[2,0,2]],np.float64),faces,[0,0,1]);require(h['hit'] is None and h['degenerate_triangles']==1,'Degenerate')
    passed.append('Parallel_and_degenerate_triangles_explicit_no_silent_repair')
    near=verts.copy();near[:,2]=NEAR_MM;require(first_hit(near,faces,[0,0,1])['hit']['camera_Z_mm']==NEAR_MM,'Nearplane inclusive')
    near[:,2]=NEAR_MM/2;require(first_hit(near,faces,[0,0,1])['hit'] is None,'Subnear nohit')
    crossing=np.array([[-1,-1,-1],[1,-1,1],[0,1,1]],np.float64);h=first_hit(crossing,faces,[0,0,1]);require(h['hit']['camera_Z_mm']==.5,'Original crossing surface beyondnear')
    passed.append('Nearplane1e6mm_fixed_and_crossing_triangle_camera_ray_equivalence')
    K=np.array([[10,0,3],[0,20,4],[0,0,1]],np.float64);require(np.array_equal(camera_ray(K,3,4),[0,0,1]) and np.array_equal(camera_ray(K,8,14),[.5,.5,1]),'K integercenter nohalfshift')
    passed.append('Original_K_integer_pixel_centers_no_halfpixel_offset')
    R=np.array([[0,-1,0],[1,0,0],[0,0,1]],np.float64);require(np.array_equal(camera_vertices([[1,0,2]],R,[3,4,5]),[[3,5,7]]),'Unmodified modeltocamera R/t')
    passed.append('R_m2c_rowvector_Rtranspose_plus_t_mm_exact')
    pix=fixed_grid_pixels([2,3]);idx=fixed_face_indices(3);require(len(pix)==63 and pix[0]==(0,0) and pix[-1]==(2,1) and len(set(pix))<63
        and len(idx)==17 and idx[0]==0 and idx[-1]==2 and len(set(idx))==3,'Fixedslots duplicates kept')
    passed.append('Fixed9x7_and17_faceorder_slots_keep_duplicates_and_endpoints')
    require(depth_bound(np.float64,1234.)==1e-6,'F64 fixedabsolute bound')
    for x in (1.,1000.00004,1024.00003,9876.54321):
        require(abs(float(np.float32(x))-x)<=depth_bound(np.float32,x),'F32 round bound')
    try:depth_bound(np.float16,1)
    except AssertionError:pass
    else:raise AssertionError('Unsupported depthdtype')
    passed.append('Prospective_F64_1e6mm_F32_halfULP_plus1e6_bound_unsupported_rejected')
    return dict(completed=True,PASS=True,invented_only=True,checks=passed,count=len(passed),asset_pose_mesh_header_body_hash_reads=0,producer_imports=0)
