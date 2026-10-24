"""Production code vs independent generated scalar oracles; actual source is never loaded."""
from pathlib import Path
import contextlib, hashlib, importlib.util, io, itertools, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parent
PRODUCTION = ROOT.parent/'raster_preparation'

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def bind(p):
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)

def main():
    math_code, runner_code = PRODUCTION/'cpu_part_raster.py', PRODUCTION/'run_raster_prototype.py'
    assert bind(math_code)['sha256']=='1ed9801add06cb9fec4c15b0078948d243a08625567c718cbf3ee01851a19d53'
    assert bind(runner_code)['sha256']=='7a92aae99d8bca34d05d335d689eff16d8a4ab6cb8014c712b469a7471cd6072'
    m=load(math_code,'cpu_part_raster')
    runner=load(runner_code,'review_raster_generated_only')
    oracle=load(ROOT/'scalar_raster_oracles.py','independent_scalar_oracle')
    captured=[]
    def capture(path,obj):
        assert path==PRODUCTION/'GENERATED_RASTER_TESTS.json'
        captured.append(obj)
    runner.write=capture
    with contextlib.redirect_stdout(io.StringIO()):
        runner.generated()
    assert len(captured)==1
    assert captured[0]==json.loads((PRODUCTION/'GENERATED_RASTER_TESTS.json').read_text())
    cam=runner.simple_camera(256)
    uv=np.array([[110.,110.],[145.,110.],[110.,145.]])
    z=np.array([2.,6.,4.])
    vertices=np.column_stack(((uv-127.5)*z[:,None]/8.,z))
    # Integer screen coordinates force exact boundary cases, not just interior samples.
    fragments={}
    for yy,xx,depth in m.fragments(vertices,cam):
        for y,x,d in zip(yy,xx,depth):
            assert (int(y),int(x)) not in fragments
            fragments[int(y),int(x)]=float(d)
    max_error=0.;checked=0
    for y in range(109,147):
        for x in range(109,147):
            expected=oracle.scalar_inside(uv,[x,y])
            assert ((y,x) in fragments)==expected
            if expected:
                direct=oracle.direct_ray_plane_depth(vertices,[x,y],8.,127.5)
                max_error=max(max_error,abs(direct-fragments[y,x]));checked+=1
    assert checked>0 and max_error<1e-12
    square=[np.array([[1.,1.],[5.,1.],[5.,5.]]),np.array([[1.,1.],[5.,5.],[1.,5.]])]
    coverage={}
    for tri in square:
        points=m.unproject_xy(tri,2.,cam)
        for yy,xx,_ in m.fragments(points,cam):
            for y,x in zip(yy,xx):coverage[int(y),int(x)]=coverage.get((int(y),int(x)),0)+1
    assert len(coverage)==16 and set(coverage.values())=={1}
    for y in range(7):
        for x in range(7):assert coverage.get((y,x),0)==int(1<=x<5 and 1<=y<5)
    bounds=np.array([[-1.,-2.,-3.],[1.,2.,3.]])
    for _,axis in runner.DIRECTIONS:
        c=m.camera_from_bounds(bounds,axis)
        assert np.array_equal(c['R'],oracle.camera(axis))
        assert c['K'][0,2]==c['K'][1,2]==127.5
        # Foreign uses only AABB/camera; formula is verified without any rendered mask.
        foreign=m.foreign_rectangle(bounds*.3,c)
        xyz=m.world_to_camera(m.box_corners(bounds*.3),c)
        projected=np.array([[c['K'][0,0]*q[0]/q[2]+127.5,c['K'][1,1]*q[1]/q[2]+127.5] for q in xyz])
        lo,hi=projected.min(0),projected.max(0)
        expected_rect=np.array([[lo[0]-2,lo[1]-2],[lo[0]+.6*(hi[0]-lo[0])+2,lo[1]-2],[lo[0]+.6*(hi[0]-lo[0])+2,hi[1]+2],[lo[0]-2,hi[1]+2]])
        assert np.array_equal(expected_rect,foreign['rectangle_xy'])
        assert foreign['z']<np.min(xyz[:,2])
    # Exact ray near-plane intersection locations, independent of polygon iteration.
    clipped=m.clip_near(np.array([[0.,0.,.5],[1.,0.,2.],[0.,1.,2.]]),1.)
    expected=np.array([[0.,1/3,1.],[1/3,0.,1.],[1.,0.,2.],[0.,1.,2.]])
    assert np.max(np.abs(clipped-expected))<1e-15
    eps=64*np.finfo(np.float64).eps
    depths=np.array([1.,1.+.75*eps,1.+2*eps])[:,None,None]
    faces=np.arange(3,dtype=np.int32)[:,None,None]
    ps=m.resolve_layers(depths,faces)
    assert ps['ambiguity'].item() and ps['tie_part_count'].item()==2
    for order in itertools.permutations(range(3)):
        out=m.resolve_layers(depths[list(order)],faces[list(order)],list(order))
        assert all(np.array_equal(out[k],ps[k],equal_nan=True) for k in ps)
    pa=m.resolve_layers(depths[2:3],faces[2:3],[2])
    pv=m.resolve_layers(depths,faces,foreign=(np.array([[1.-.75*eps]]),np.array([[99]],np.int32),308))
    partition=m.partition_selected(pa,ps,pv,selected=2,foreign=308)
    assert partition['unknown'].item() and not partition['self_hidden_other_target'].item()
    background={k:np.array(v,copy=True) for k,v in ps.items()}
    background['part_id'][:]=-1;background['ambiguity'][:]=False
    partition=m.partition_selected(pa,background,background,selected=2,foreign=308)
    assert partition['unknown'].item() and not partition['self_hidden_other_target'].item()
    protocol=json.loads((PRODUCTION/'RASTER_PROTOCOL_BEFORE_ACTUAL_PIXELS.json').read_text())
    assert protocol['buffer_bytes']==308*256*256*12==242221056
    assert protocol['buffer_bytes']<protocol['peak_RSS_goal_bytes']==1000000000
    result=dict(completed=True,PASS=True,code=bind(runner_code),math_code=bind(math_code),reviewer_code=bind(Path(__file__)),
                author_generated_rerun_exact=True,independent_integer_edge_coverage_and_ray_depth_points=checked,
                independent_ray_depth_max_abs_error=max_error,square_top_left_exactly_once_pixels=16,
                six_camera_basis_and_geometry_only_foreign_rectangle_exact=True,near_intersection_scalar=True,
                all_six_layer_orders_identical=True,selected_not_tied_ambiguity_U_not_SELF=True,
                background_U_not_SELF=True,buffer_raw_bytes=242221056,
                actual_source_mesh_pixels_RGB_models_current_fit_PnP=0)
    output=ROOT/'INDEPENDENT_PRODUCTION_GENERATED_REPLAY.json'
    assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(PASS=True,output=bind(output))))

if __name__=='__main__':main()
