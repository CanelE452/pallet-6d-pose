"""Generated opaque cube/counterexample fixtures only; no source files/renders."""
from pathlib import Path
import importlib.util,json,hashlib,itertools
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('generated_endpoint_teacher',HERE/'endpoint_cue_teacher.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
def camera(eye):
    eye=np.array(eye,np.float64);z=-eye/np.linalg.norm(eye);x=np.cross(z,[0,1,0]);x/=np.linalg.norm(x);y=np.cross(z,x);R=np.stack([x,y,z]);f=256/(2*np.tan(np.deg2rad(25)))
    return dict(R=R,t=-R@eye,K=np.array([[f,0,127.5],[0,f,127.5],[0,0,1.]]),eye_world=eye,size=256,near=1e-4)
def cube():
    triangles=[]
    for axis in range(3):
        ab=[i for i in range(3)if i!=axis]
        for sign in [-1,1]:
            q=[]
            for u,v in [(-1,-1),(1,-1),(1,1),(-1,1)]:
                p=np.zeros(3);p[axis]=sign;p[ab]=[u,v];q.append(p)
            q=np.array(q)
            if np.cross(q[1]-q[0],q[2]-q[0])[axis]*sign<0:q=q[::-1]
            triangles.extend([q[[0,1,2]],q[[0,2,3]]])
    return np.array(triangles)
def plane(camera,rect,z):
    uv=np.array([[rect[0],rect[1]],[rect[2],rect[1]],[rect[2],rect[3]],[rect[0],rect[3]]]);c=np.c_[uv,np.ones(4)]@np.linalg.inv(camera['K']).T*z
    w=(c-camera['t'])@camera['R'];return w[[[0,1,2],[0,2,3]]]
def run():
    local=np.array(list(itertools.product([-1.,1.],repeat=3)));tri=cube();target=T.RayScene(tri,np.zeros(12,int),foreign_owner_min=10);cam=camera([4,3,5])
    def cue(point,L=128):
        e=T.make_endpoint(local,np.eye(4),point,.3);q,_=T.project(np.asarray(point)[None],cam)
        return dict(part_id=0,geometry=e,original_q=q[0],L=L)
    front=cue([1.,1.,1.]);base=T.evaluate_endpoint(front,cam,target);assert base['state']=='D'and len(base['front_facing_faces'])==3
    back=T.evaluate_endpoint(cue([-1.,-1.,-1.]),cam,target);assert back['state']=='S'and not back['front_facing_faces']
    mixed=T.evaluate_endpoint(cue([-1.,1.,1.]),cam,target);assert mixed['state']=='D'and len(mixed['front_facing_faces'])==2
    # Ignored back-facing probes really are hidden; they may not forceU.
    assert any(r['state']=='SELF'for r in mixed['target_only_probe_states'])
    centre=np.array(base['projected_support_disk']['center']);r=base['projected_support_disk']['radius']+3;z=.5*base['all_patch_camera_z_min'];rect=[*(centre-r),*(centre+r)]
    cover=plane(cam,rect,z);combined=T.RayScene(np.r_[tri,cover],np.r_[np.zeros(12,int),[10,10]],double_sided=np.r_[np.zeros(12,bool),[1,1]],foreign_owner_min=10)
    covered=T.evaluate_endpoint(front,cam,target,combined);assert covered['state']=='E'
    sham_rect=np.array(rect)+np.array([2*r+2,0,2*r+2,0]);sham=plane(cam,sham_rect,z)
    aside=T.RayScene(np.r_[tri,sham],np.r_[np.zeros(12,int),[10,10]],double_sided=np.r_[np.zeros(12,bool),[1,1]],foreign_owner_min=10)
    assert T.evaluate_endpoint(front,cam,target,aside)['state']=='D'
    partial=plane(cam,[*(centre-.2),*(centre+.2)],z)
    part=T.RayScene(np.r_[tri,partial],np.r_[np.zeros(12,int),[10,10]],double_sided=np.r_[np.zeros(12,bool),[1,1]],foreign_owner_min=10)
    partial_result=T.evaluate_endpoint(front,cam,target,part);assert partial_result['state']=='U'
    ambiguous=T.RayScene(np.r_[tri,tri],np.r_[np.zeros(12,int),np.ones(12,int)],foreign_owner_min=10)
    assert T.evaluate_endpoint(front,cam,ambiguous)['state']=='U'
    sameowner=T.RayScene(np.r_[tri,tri],np.zeros(24,int),foreign_owner_min=10)
    assert T.evaluate_endpoint(front,cam,sameowner)['state']=='D'
    missed=T.evaluate_endpoint(cue([1.,1.,1.],L=1),cam,target);assert missed['state']=='U'and missed['underlying_geometric_state']=='D'and missed['reason']=='QUERY_CROP_OR_FRAME_SUPPORT_MISS'
    hidden_covered=T.evaluate_endpoint(cue([-1.,-1.,-1.]),cam,target,combined);assert hidden_covered['state']=='S' # no self->E promotion
    assert np.array_equal(front['original_q'],T.project(np.array([[1.,1.,1.]]),cam)[0][0])
    assert np.all(T.make_endpoint(local,np.eye(4),[1,1,1],.3)['edge_lengths']==.3)
    try:T.make_endpoint(local[:7],np.eye(4),[1,1,1],.3)
    except ValueError:pass
    else:raise AssertionError('Non-cuboid must reject')
    result=dict(completed=True,PASS=True,code=dict(path=str(HERE/'endpoint_cue_teacher.py'),sha256=hashlib.sha256((HERE/'endpoint_cue_teacher.py').read_bytes()).hexdigest()),fixtures=['frontD','backS','backface_ignored_mixedD','fullforeignE','shamD','junction_partialU','different_owner_tieU','sameowner_tieD','cropMISS_U_underlyingD','self_not_promoted_to_E','q_unchanged','true_edges_not_diagonals','invalid_cuboid_reject'],source_RGB_geometry_model_fit_current_reads=0)
    path=HERE/'GENERATED_ENDPOINT_CUE_TEACHER_TESTS.json'
    with path.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':run()
