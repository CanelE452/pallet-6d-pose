"""Prospective camera/aspect/one-pass8direction placement math, no RGB."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import build_geometry96 as B
from conservative_ray_scene import load_teacher
HERE=Path(__file__).resolve().parent;V=HERE.parent;P=V.parent
def main():
    config=B.lines(V/'BASE96_SOURCE_CONFIG_INPUT_ONLY.jsonl');assert len(config)==96
    assert all(c['base_index']==(((c['azimuth_index']*2+c['elevation_index'])*2+c['aspect_index'])*2+c['appearance_index'])for c in config)
    assert sum(c['split']=='TRAIN'for c in config)==64 and sum(c['split']=='VALIDATION'for c in config)==32
    assert all(np.linalg.norm(np.asarray(c['query_jitter_xy']),axis=1).max()<=4 for c in config)
    bounds=np.array([[-2.,-1.,-1.],[2.,1.,1.]]);M=B.affine(bounds,1);trans=B.transform(np.array(list(itertools.product([-2.,2.],[-1.,1.],[-1.,1.]))),M)
    assert np.ptp(trans,axis=0)[0]==np.ptp(trans,axis=0)[2]and np.linalg.det(M[:3,:3])>0
    T=load_teacher();local=np.array(list(itertools.product([-1.,1.],repeat=3)));cue=dict(part_id=0,geometry=T.make_endpoint(local,np.eye(4),[1,1,1],.3),original_q=[30,40],L=100)
    transformed=B.transformed_cues([cue],M)[0];assert transformed['original_q']==cue['original_q']and transformed['L']==cue['L']
    assert np.array_equal(transformed['geometry']['endpoint_world'],B.transform([1,1,1],M))
    assert np.array_equal(transformed['geometry']['face_probe_points'],B.transform(cue['geometry']['face_probe_points'],M))
    for f in range(3):
        g=transformed['geometry'];u,v=g['face_edge_axes'][f];assert abs(np.dot(g['face_normals'][f],g['edge_vectors'][u]))<1e-14 and abs(np.dot(g['face_normals'][f],g['edge_vectors'][v]))<1e-14
    ref=B.read(P/'query_visibility_expected_cue_triplets_20261023_v97/cue_contract_preparation/CUE_GEOMETRY_GT_ONLY.json')['camera']
    c=B.camera_for(dict(azimuth_offset_deg=0,elevation_offset_deg=0),ref)
    assert np.allclose(c['R'],ref['R'],atol=1e-15,rtol=0)and np.allclose(c['eye_world'],ref['eye_world'],atol=4e-15,rtol=0)
    # Off-axis cube backvertices project beyond its frontbbox; all8 vertices
    # must clear the disk analytically in every compass direction.
    camera=dict(K=[[686.,0,319.5],[0,686.,319.5],[0,0,1]],R=np.eye(3),t=np.zeros(3),size=640,near=.001)
    selected=dict(projected_support_disk=dict(center=[450.,400.],radius=15.),all_patch_camera_z_min=20.,all_patch_projected_vertices=[[440,390],[460,390],[460,410],[440,410]])
    clear=[]
    for k in range(8):
        direction=np.array([np.cos(k*np.pi/4),np.sin(k*np.pi/4)]);g=B.foreign_for(camera,selected,direction)
        assert g['placement_valid']and g['full_projected_sham_clearance_px']>=2-1e-9
        assert np.allclose(np.asarray(g['sham_vertices_world'])-g['cover_vertices_world'],g['sham_world_translation'],atol=1e-14,rtol=0)
        assert np.allclose(np.asarray(g['cover_vertices_world'])[:,2],np.asarray(g['sham_vertices_world'])[:,2],atol=1e-14,rtol=0)
        p,z=B.project(g['sham_vertices_world'],camera);assert ((p-np.array([450,400]))@direction).min()>17-1e-9
        clear.append(g['full_projected_sham_clearance_px'])
    proof=dict(completed=True,PASS=True,builder=B.bind(HERE/'build_geometry96.py'),root_config=B.bind(V/'BASE96_SOURCE_CONFIG_INPUT_ONLY.jsonl'),tests=['all96orderfoldjitter','positiveaspect_equalXZ','fixedassociation_affine_only','normal_tangent_transform','camera_reference_recovery','all8_fullbox_analytic_sham_clearance','samebox_translation_depth','no_search'],compass_clearance_px=clear,new_source_RGB_renderer_model_fit_current_PnP=0)
    B.write(HERE/'GENERATED_GEOMETRY96_TESTS.json',proof);print(json.dumps(B.clean(proof),indent=2))
if __name__=='__main__':main()
