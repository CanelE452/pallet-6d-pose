"""Independent ray/triangle depth reference, generated geometry only.

The reviewed raster's AST is executed without its JIT decorator; root separately
checks the Numba smoke. This review imports no model/optimizer or source images.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
REVIEWED=HERE/'mesh_depth_render.py'


def ray_reference(vertices,faces,K,T,hw):
    camera=np.asarray(vertices)@T[:3,:3].T+T[:3,3]
    h,w=hw;out=np.full((h,w),np.inf)
    inv=np.linalg.inv(K)
    for y in range(h):
        for x in range(w):
            ray=inv@np.array([x,y,1.])
            for face in faces:
                a,b,c=camera[face]
                if min(a[2],b[2],c[2])<=1e-6:continue
                ab,ac=b-a,c-a;normal=np.cross(ab,ac)
                denom=normal@ray
                if abs(denom)<1e-12:continue
                parameter=(normal@a)/denom
                if parameter<=0:continue
                point=parameter*ray
                d00,d01,d11=ab@ab,ab@ac,ac@ac
                det=d00*d11-d01*d01
                if abs(det)<1e-12:continue
                ap=point-a;d20,d21=ap@ab,ap@ac
                u=(d11*d20-d01*d21)/det;v=(d00*d21-d01*d20)/det
                if min(u,v,1-u-v)>=-1e-9:out[y,x]=min(out[y,x],point[2])
    return out


def main():
    tree=ast.parse(REVIEWED.read_text());functions=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef):node.decorator_list=[];functions.append(node)
    namespace={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(REVIEWED),'exec'),namespace)
    render=namespace['render_depth'];assert namespace['smoke']()['PASS']
    K=np.array([[30.,1.7,14.2],[.4,28.,11.3],[0,0,1]])
    uv=np.array([[-4,-3],[35,2],[5,29],[2,3],[24,4],[14,22]],float)
    depths=np.array([2.,4.,3.,1.8,2.2,1.9])
    rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T
    camera=rays*depths[:,None]
    angle=.31;R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=[.2,-.17,.6]
    vertices=(camera-T[:3,3])@R;faces=np.array([[0,1,2],[3,4,5]])
    actual,metadata=render(vertices,faces,K,T,[26,31]);reference=ray_reference(vertices,faces,K,T,[26,31])
    assert np.array_equal(np.isfinite(actual),np.isfinite(reference))
    valid=np.isfinite(reference);error=np.abs(actual[valid]-reference[valid]);assert error.max()<3e-7
    for order in [faces[::-1],faces[:,::-1]]:
        other,_=render(vertices,order,K,T,[26,31]);assert np.array_equal(actual,other)
    scaled,scale_metadata=render(vertices*1000,faces,K,np.block([[R,T[:3,3,None]*1000],[np.zeros((1,3)),np.ones((1,1))]]),[26,31])
    assert np.array_equal(np.isfinite(scaled),valid)
    assert np.allclose(scaled[valid],actual[valid]*1000,rtol=2e-7,atol=3e-4)
    crossing=np.array([[0,0,-1],[1,0,2],[0,1,2]],float)
    missing,near=render(crossing,np.array([[0,1,2]]),np.eye(3),np.eye(4),[3,3])
    assert near['near_plane_crossing_triangles']==1 and not np.isfinite(missing).any()
    reflected=np.eye(4);reflected[0,0]=-1
    try:render(vertices,faces,K,reflected,[26,31]);raise AssertionError('Reflection accepted')
    except ValueError:pass
    result=dict(schema='V66_INDEPENDENT_CPU_MESH_RASTER_MATH_AUDIT_V1',PASS=True,
                tests=['existing root smoke','independent3D ray-plane/barycentric reference','nonzero intrinsics skew',
                       'nonidentity object-to-camera rotation/translation','face-order and winding invariance',
                       'consistent metric scaling','near-plane crossing explicitly missing','reflection rejected'],
                tested_native_pixels=int(valid.sum()),max_abs_depth_error_native_units=float(error.max()),
                crossing_triangles_are_not_clipped=True,
                scope='Correct provided-mesh raster math under rectified pinhole integer centers and camera-Z depth; no accuracy/provenance validation of real mesh/pose/depth.',
                required_source_checks=['common metric units','object frame/origin and object-to-camera direction',
                                        'rectified intrinsics matched to depth/RGB native grid','camera-Z versus radial range',
                                        'no crossing-near-plane triangles for outside labels','source mesh completeness/pose quality before O certificate'],
                source_or_private_images_labels_scores_reads=0,new_models_GPU_training_PnP_calls=0,
                source_code_sha256=hashlib.sha256(REVIEWED.read_bytes()).hexdigest(),
                audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'INDEPENDENT_MESH_DEPTH_MATH_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
