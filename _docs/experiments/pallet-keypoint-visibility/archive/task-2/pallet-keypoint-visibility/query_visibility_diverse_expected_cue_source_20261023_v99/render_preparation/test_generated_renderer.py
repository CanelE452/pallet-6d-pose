"""Generated arrays/geometry only; no source RGB, mesh, teacher or learned model."""
import os, sys
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
os.environ['__GL_SHADER_DISK_CACHE'] = '0'
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json
import numpy as np
from render_math import *
H = Path(__file__).resolve().parent
CROP = H.parent.parent / 'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p): return dict(path=str(Path(p).resolve()), sha256=sha(p))
def crop_op():
    assert sha(CROP) == '6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
    fn = next(n for n in ast.parse(CROP.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'crop')
    ns = {'np': np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(CROP), 'exec'), ns)
    return ns['crop']

def tests():
    # Exact scalar bilinear oracle, native integer pixels with old96 phase.
    crop = crop_op(); rng = np.random.default_rng(20261023)
    image = rng.integers(0, 256, (17, 21, 3), dtype=np.uint8)
    cases = [([10., 8.], 12.), ([-.2, 8.], 19.), ([21., 17.], 16.), ([0., 0.], 8.)]
    scalar_N = 0
    for q, L in cases:
        actual, valid = crop(image, np.array(q), L, 96)
        for iy, ix in [(0,0), (47,47), (48,48), (95,95), (12,51), (50,6)]:
            x = q[0] + ((ix + .5) / 96 - .5) * L
            y = q[1] + ((iy + .5) / 96 - .5) * L
            vv = -.5 <= x < 20.5 and -.5 <= y < 16.5
            total = np.zeros(3)
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            for dy in (0,1):
                for dx in (0,1):
                    xx, yy = x0+dx, y0+dy
                    weight = (1-abs(x-xx))*(1-abs(y-yy))
                    if 0 <= xx < 21 and 0 <= yy < 17: total += image[yy,xx]*weight
            expected = np.floor(np.clip(total,0,255)+.5).astype(np.uint8) if vv else np.zeros(3,np.uint8)
            assert bool(valid[0,iy,ix]) == vv and np.array_equal(actual[:,iy,ix], expected)
            scalar_N += 1
    # Transparent/opaque/half-coverage compositor, no categorical interpolation.
    bg = np.full((3,4,3), 100, np.uint8); rgba = np.zeros((3,4,4),np.uint8)
    assert np.array_equal(composite_premultiplied_rgba(rgba,bg),bg)
    rgba[:] = [31,79,119,255]
    assert np.array_equal(composite_premultiplied_rgba(rgba,bg),rgba[...,:3])
    rgba[:] = [20,30,40,128]
    assert np.array_equal(composite_premultiplied_rgba(rgba,bg)[0,0], [70,80,90])
    assert len({fixed_seed('APPEARANCE',s,i) for s in ('TRAIN','VALIDATION') for i in range(96)}) == 192
    M=np.eye(4);M[2,2]=1.25;M[3,:3]=[1,2,3]
    assert np.array_equal(affine_vertices(np.array([[1.,2.,3.]]),M), [[2.,4.,6.75]])
    c=dict(R=np.eye(3).tolist(),t=[0,0,0],eye_world=[0,0,0],K=[[40,0,31.5],[0,40,31.5],[0,0,1]],near=.5,far=5.)
    assert np.array_equal(gl_pose_from_cv(c),np.diag([1.,-1.,-1.,1.]))
    check_complete_clip(c,np.array([[-1,-1,2],[1,1,3]]))
    for bad in (np.array([[0,0,.4]]),np.array([[0,0,6.]])):
        try: check_complete_clip(c,bad)
        except AssertionError: pass
        else: raise AssertionError('Clip guard failed')
    # Actual tiny EGL generated-only probe: opaque MSAA on transparent black.
    import pyrender, trimesh, PIL
    from OpenGL import GL
    assert PIL.__version__ == '12.2.0' and pyrender.__version__ == '0.1.45'
    # CV nativeinteger pixels versus GL physicalpixel centres: explicit+.5 bridge.
    glcam=pyrender.IntrinsicsCamera(**gl_intrinsics_from_integer_camera(c))
    points=rng.uniform([-1,-1,1],[1,1,4],(100,3));uv,z=project(c,points)
    pp=glcam.get_projection_matrix(64,64)
    clip=np.c_[points[:,0],-points[:,1],-points[:,2],np.ones(len(points))]@pp.T
    ndc=clip[:,:3]/clip[:,3:4]
    native=np.c_[(ndc[:,0]+1)*32-.5,(1-ndc[:,1])*32-.5]
    projection_error=float(np.max(np.abs(native-uv)))
    assert projection_error<2e-13
    # Reused UV helper + production batching, exclusively invented triangles.
    import render_corpus_once as producer
    old=producer.module(producer.OLD,'v99_generated_only_old_UV')
    xyz=np.array([[[3*i,0,0],[3*i+1,0,0],[3*i,1,0]]for i in range(308)],np.float64).reshape(-1,3)
    tri=np.arange(len(xyz)).reshape(-1,3);pid=np.arange(308,dtype=np.int32)
    inventory=dict(per_mesh=[dict(USD_orientation='rightHanded'if i%2 else 'leftHanded',double_sided=bool(i%3==0))for i in range(308)])
    batched=producer.build_target_mesh(old,xyz,tri,pid,inventory,pyrender)
    assert len(batched.primitives)==2 and sum(len(p.positions)for p in batched.primitives)==924
    assert all(p.color_0 is None and np.isfinite(p.normals).all()and np.isfinite(p.texcoord_0).all()for p in batched.primitives)
    for family in ('PROCEDURAL_WOOD','MATTE_POLYMER'):
        mf=producer.set_appearance(batched,old,pyrender,family,1501954095)
        assert all(p.material.metallicFactor==0 and np.max(p.material.baseColorFactor)>0.2 for p in batched.primitives)
    g=dict(cover_vertices_world=[[-1,-1,-2],[1,-1,-2],[1,1,-2],[-1,1,-2],[-1,-1,-3],[1,-1,-3],[1,1,-3],[-1,1,-3]],triangles=[[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[3,7,6],[3,6,2],[0,4,7],[0,7,3],[1,2,6],[1,6,5]])
    foreign,fsha=producer.external_mesh(pyrender,trimesh,g,4251810918)
    assert len(foreign.primitives)==1 and len(fsha)==64
    mesh = trimesh.Trimesh(vertices=[[-.61,-.73,-2],[.79,-.49,-2],[-.21,.81,-2]],faces=[[0,1,2]],process=False)
    mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[.5,.25,.125,1.],metallicFactor=0.,roughnessFactor=1.,doubleSided=True)
    scene=pyrender.Scene(bg_color=[0.,0.,0.,0.]);scene.add(pyrender.Mesh.from_trimesh(mesh,material=mat,smooth=False))
    scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(50),znear=.1,zfar=10.),pose=np.eye(4))
    renderer=pyrender.OffscreenRenderer(64,64);renderer._platform.make_current();GL.glDisable(GL.GL_DITHER)
    rgba, depth=renderer.render(scene,flags=pyrender.RenderFlags.RGBA|pyrender.RenderFlags.FLAT)
    renderer.delete()
    assert rgba.shape==(64,64,4) and rgba.dtype==np.uint8
    partial=(rgba[...,3]>0)&(rgba[...,3]<255);inside=rgba[...,3]==255
    assert partial.any() and inside.any() and np.all(rgba[rgba[...,3]==0,:3]==0)
    interior=np.median(rgba[inside,:3],axis=0)
    predicted=interior[None,:]*rgba[partial,3:4]/255.
    maxerr=float(np.max(np.abs(rgba[partial,:3].astype(float)-predicted)))
    assert maxerr <= 1.01, (interior,maxerr,np.unique(rgba[partial],axis=0))
    return dict(PASS=True,code=binding(__file__),producer=binding(H/'render_corpus_once.py'),math=binding(H/'render_math.py'),crop_code=binding(CROP),
        generated_scalar_crop_positions=scalar_N,deterministic_seed_fold_test=True,clip_guard_test=True,
        CV_integer_to_GL_physical_pixel_projection=dict(points=100,max_abs_error=projection_error,GL_principal_shift_only=.5),
        generated_308_triangle_material_batching=dict(primitive_count=2,vertex_count=924,part_ID_vertex_colors=False,both_materials_finite=True,foreign_mesh_valid=True),
        actual_generated_EGL_RGBA_probe=dict(canvas=[64,64],partial_alpha_pixels=int(partial.sum()),opaque_pixels=int(inside.sum()),
            opaque_RGB=interior.tolist(),max_abs_premultiplied_boundary_error=maxerr,
            framebuffer_convention='RGB coverage-premultiplied on transparent black; alpha not applied twice',
            render_source_geometry_reads=0,public_RGB_reads=0),source_mesh_RGB_weights_model_fit_current_PnP=0)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    r=tests()
    with a.output.open('x') as f: json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(r))
