"""Constructed normal-only full-search/outline and feature contracts."""
from copy import deepcopy
import cv2
import numpy as np

from scripts.research.pallet_bi_conditioned_observation_v1.test_solver import fixture,partial_selection
from scripts.research.pallet_bi_full_execution_v1 import fullnormal_backend as B
from scripts.research.pallet_bi_full_execution_v1 import fullnormal_batch as RUN


def setup():
    frame,selection,R,t,q=fixture();selection=partial_selection(frame,selection,q)
    geometry=B.LOCAL.RT.registered_geometry(frame['raw_hw'],frame['K'],frame['native_xyz'],frame['initial_pose'])
    coarse=dict(R=frame['initial_pose']['R_cf'],t=frame['initial_pose']['centroid'],physical_registration_Q=geometry['physical_registration_Q'].tolist())
    cfg=B.configuration(dict(max_p3p_triples=0,global_rotation_starts=0,prior_weight=0.))
    return frame,selection,coarse,cfg


def solve(frame,selection,coarse,cfg):
    return B.solve_boundary_u_region_joint_ot_pnp(frame['native_xyz'],frame['K'],selection['states'],selection['full_points'],[],frame['raw_hw'],coarse,config=cfg,boundary_observations=selection['partial_lines'])


def test_archived_partial_coordinates_and_intervals_never_enter_full_search():
    frame,a,coarse,cfg=setup();baseline=solve(frame,a,coarse,cfg);b=deepcopy(a)
    k=b['partial_lines'][0]['native_corner'];record=next(p for p in b['full_points']if p['native_corner']==k)
    record['xy'][0]+=20;b['partial_lines'][0]['association_provenance']['original_observed_xy']=deepcopy(record['xy'])
    b['partial_lines'][0]['lines'][0]['support_tangent_interval_px']=[200000.,200001.]
    changed=solve(frame,b,coarse,cfg)
    assert baseline['accepted']and changed['accepted']
    np.testing.assert_array_equal(baseline['R'],changed['R']);np.testing.assert_array_equal(baseline['t'],changed['t'])
    assert k not in baseline['search']['full_point_seed_ids']
    assert baseline['search']['numeric_caps']['max_seed_nfev']==160
    assert baseline['search']['numeric_caps']['max_joint_iterations']==250
    assert baseline['all_visited_endpoints']
    assert all('R'in e and't'in e for e in baseline['all_visited_endpoints'])


def test_no_full_points_use_only_normal_and_registered_coarse_initialization():
    frame,selection,coarse,cfg=setup()
    _,fresh,_,_,q=fixture();selection=partial_selection(frame,fresh,q,all_partial=True)
    result=solve(frame,selection,coarse,cfg)
    assert result['search']['full_point_seed_ids']==[]
    assert result['search']['P3P_or_SQPNP_called']is False
    assert result['search']['uses_revoked_archive_for_initialization']is False
    assert result['accepted']


def test_u_constraint_is_kept_without_partial_tangent_or_archived_point_guards():
    frame,selection,coarse,cfg=setup();values=B._prepare(frame['native_xyz'],frame['K'],selection['states'],selection['full_points'],[],frame['raw_hw'],coarse,cfg,selection['partial_lines'],[])
    xyz,K,states,measurements,domains,hw,cfg,boundary,regions,Q,prior,_=values
    scale=np.linalg.norm(np.ptp(xyz,axis=0));p=B.V._pose_vector(np.asarray(coarse['R']),np.asarray(coarse['t']),scale)
    base=B.BASE._constraints(p,xyz,K,B._complete(measurements,boundary),domains,(),hw,scale,cfg)
    actual=B._joint_constraint_pair(p,xyz,K,measurements,domains,(),hw,scale,cfg,boundary,regions)
    np.testing.assert_array_equal(base[0],actual[0]);np.testing.assert_array_equal(base[1],actual[1])
    xy=B.V.project_native(xyz,np.asarray(coarse['R']),np.asarray(coarse['t']),K)[0]
    region=dict(native_corner=0,center_xy=xy[0]+10,precision=np.eye(2))
    with_u=B._joint_constraint_pair(p,xyz,K,measurements,domains,(),hw,scale,cfg,boundary,[region])
    assert len(with_u[0])==len(actual[0])+1
    assert with_u[0][-1]<0


def test_whole_new_hierarchy_g13_exact_collapse_and_original31_validity_features():
    frame,selection,coarse,cfg=setup();frame.update(partition='SYNTH',split='train',session='analytic',cluster_id='analytic',image_sha256='0'*64)
    observation=dict(states=selection['states'],_original_source=dict(ot_domains=dict(domains=[])),u_region_observation=dict(u_regions=[]))
    mask=np.zeros(frame['raw_hw'],np.float32);points=B.V.project_native(np.asarray(frame['native_xyz']),np.asarray(coarse['R']),np.asarray(coarse['t']),np.asarray(frame['K']))[0]
    cv2.fillConvexPoly(mask,cv2.convexHull(points.astype(np.int32)),1.)
    result=RUN.solve_frame(frame,selection,observation,mask,cfg)
    assert result['proof']['G13_exact_reuses_G10']
    assert result['stage_results']['G13']==result['stage_results']['G10']
    assert result['proof']['revoked_2D_seed_or_guard_terms']==0
    assert result['all_visited_endpoints']
    features=RUN.feature_frame(frame,result,dict(none_prior=.05))
    assert len(features['feature_names'])==31
    assert features['candidates']
    for candidate in features['candidates']:
        assert len(candidate['features'])==62
        for i in features['B_profile_features']['energy_indices_masked']:assert candidate['features'][i+31]==0.
