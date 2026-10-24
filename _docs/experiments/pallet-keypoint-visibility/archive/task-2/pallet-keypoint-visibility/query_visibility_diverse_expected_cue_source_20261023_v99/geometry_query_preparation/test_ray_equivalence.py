"""Generated boundary/tie fixtures plus immutable V97's296 baseline rays."""
from pathlib import Path
import json,hashlib,importlib.util
import numpy as np
from conservative_ray_scene import ConservativeRayScene,load_teacher
HERE=Path(__file__).resolve().parent;P=HERE.parents[1]
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def cube():
    triangles=[]
    for axis in range(3):
        ab=[x for x in range(3)if x!=axis]
        for sign in[-1,1]:
            q=[]
            for u,v in[(-1,-1),(1,-1),(1,1),(-1,1)]:
                a=np.zeros(3);a[axis]=sign;a[ab]=[u,v];q.append(a)
            q=np.array(q)
            if np.cross(q[1]-q[0],q[2]-q[0])[axis]*sign<0:q=q[::-1]
            triangles.extend([q[[0,1,2]],q[[0,2,3]]])
    return np.array(triangles)
def main():
    T=load_teacher();tri=cube();count=0;maxdiff=0.
    fixtures=[]
    for duplicate_owner in[0,1]:
        xyz=np.r_[tri,tri];owner=np.r_[np.zeros(12,int),np.full(12,duplicate_owner)]
        slow=T.RayScene(xyz,owner);fast=ConservativeRayScene(xyz,owner)
        for eye in[[4.,3.,5.],[4.,1.,1.],[4.,2.,0.]]:
            points=np.array([[1,1,1],[-1,-1,-1],[0,0,0],[3.9,2.925,4.875],[5,2,0],[1,1+1e-13,0]],np.float64)
            a=slow.cast(eye,points,0);b=fast.cast(eye,points,0)
            for x,y in zip(a,b):
                assert x['state']==y['state']and x['owners']==y['owners']
                if x['margin']is not None:maxdiff=max(maxdiff,abs(x['margin']-y['margin']))
            count+=len(points)
        fixtures.append(dict(duplicate_owner=duplicate_owner,ray_cases=18))
    V97=P/'query_visibility_expected_cue_triplets_20261023_v97';descriptor=json.loads((V97/'cue_contract_preparation/CUE_GEOMETRY_GT_ONLY.json').read_text());contract=json.loads((V97/'cue_contract_preparation/CUE_OBSERVABILITY_CONTRACT_BEFORE_NEW_RENDER.json').read_text());inventory=json.loads(Path(contract['V93_inventory']['path']).read_text())
    with np.load(contract['V93_geometry']['path'],allow_pickle=False)as z:vertices=z['vertices_world_m'];indices=z['triangles_global'];owners=z['triangle_asset_local_part_id']
    orientation=np.array([1 if m['USD_orientation']=='rightHanded'else-1 for m in inventory['per_mesh']])[owners];two=np.array([m['double_sided']for m in inventory['per_mesh']])[owners]
    slow=T.RayScene(vertices[indices],owners,orientation,two);fast=ConservativeRayScene(vertices[indices],owners,orientation,two);eye=np.array(descriptor['camera']['eye_world']);saved=json.loads((V97/'cue_contract_preparation/ACTUAL_TARGET_ONLY_ENDPOINT_RAY_REPLAY.json').read_text())['results'];realcount=0
    for cue,previous in zip(descriptor['references'],saved):
        g=cue['geometry'];p=np.r_[np.array(g['endpoint_world'])[None],np.array(g['face_probe_points']).reshape(-1,3),np.array(g['edge_probe_points']).reshape(-1,3)]
        a=slow.cast(eye,p,cue['part_id']);b=fast.cast(eye,p,cue['part_id'])
        for x,y,z in zip(a,b,previous['target_only_probe_states']):
            assert x['state']==y['state']==z['state']and x['owners']==y['owners']==z['owners']
            if x['margin']is not None:maxdiff=max(maxdiff,abs(x['margin']-y['margin']))
        realcount+=len(p)
    assert realcount==296
    out=dict(completed=True,PASS=True,code=bind(HERE/'conservative_ray_scene.py'),immutable_teacher=contract['teacher_code'],generated_ray_cases=count,fixtures=fixtures,V97_baseline_rays=realcount,states_owners_saved_and_exact_kernel_identical=True,max_margin_difference=maxdiff,
        actual_triangle_tests=fast.triangle_tests,full_kernel_triangle_tests_without_broadphase=fast.full_triangle_tests,broadphase_only=True,source_RGB_render_model_fit_private_current_PnP=0)
    with(HERE/'GENERATED_AND_V97_RAY_EQUIVALENCE.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
