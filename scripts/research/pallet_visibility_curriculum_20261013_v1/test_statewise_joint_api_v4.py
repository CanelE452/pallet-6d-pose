"""Full API contracts for physical registration and erased S image carriers."""
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from . import common as C
from .joint_ot_pnp import project_native
from .state_visibility_v2 import StateVisibilityModel
from .state_visibility_v4 import infer_state_visibility_v4
from .statewise_joint_api_v4 import StatewiseJointV4


def fixture():
    xyz=C.cuboid(1.3,.11,1.1)
    R=Rotation.from_euler('xyz',[.1,.23,.06]).as_matrix()
    Q=Rotation.from_euler('y',90,degrees=True).as_matrix()
    t=np.array([.03,.02,4.]);K=np.array([[400.,3.,320.],[0.,410.,240.],[0.,0.,1.]])
    q,_=project_native(xyz,R,t,K);q=np.vstack([q,[320.,240.]])
    coarse=dict(R_cf=R.tolist(),R_physical=(R@Q).tolist(),centroid=t.tolist(),
                cf_extents=[1.3,.11,1.1])
    image=np.zeros((480,640,3),np.uint8)
    model=dict(schema='STATEWISE_UNCERTAINTY_CAL_V4_1',fit_partition='CAL',
        reference_scale_px=200.,covariance_scale=1.,groups={},
        state_BI_total_error_floor_px2={'B':4.,'I':9.},
        state_loss_weights={'B':.9,'I':.8},prior_covariance6=(np.eye(6)*.01).tolist())
    classifier=StateVisibilityModel(dict(feature_family='TEST_ONLY',visible_threshold=.8,
        abstention_margin=.05,external_O_enabled=True),{},None)
    def producer(image,raw,hidden,roles,projected,model,**kwargs):
        assert np.array_equal(raw[:8][hidden],np.tile([-1.,-1.],(sum(hidden),1)))
        return infer_state_visibility_v4(image,raw,hidden,roles,projected,classifier,
            feature_extractor=lambda image,raw,ids,radius,backbone,**kwargs:
                (np.zeros((len(ids),1)),np.zeros((len(ids),1))),
            probability_predictor=lambda local,spatial,model:np.full(len(local),.99),**kwargs)
    api=StatewiseJointV4(uncertainty_model=model,state_producer=producer,
        config=dict(solver=dict(prior_weight=1.,global_rotation_starts=0,
            max_p3p_triples=1,max_seed_nfev=40,max_joint_iterations=80)))
    return api,image,q,K,xyz,coarse,model


def test_end_to_end_keeps_observations_scale_and_nonidentity_physical_registration():
    api,image,q,K,xyz,coarse,model=fixture()
    raw_before=q.copy();pose_before=deepcopy(coarse)
    result=api.refine(image,q,K,xyz,coarse,pallet_mask=np.ones(image.shape[:2]))
    assert result['accepted'] and result['provenance']['S_image_queries']==0
    assert np.array_equal(q,raw_before) and coarse==pose_before
    assert np.allclose(result['actual_pose']['R_physical'],result['solution']['R_physical'])
    assert result['keypoint_readout']['reported_points'][8]==q[8].tolist()
    ratio=np.clip(result['ot_domains']['nonself_prediction_scale_px']/200.,.5,2.)
    for original,fit in zip(result['frozen_visible_measurements'],result['calibrated_visible_measurements']):
        assert original['xy']==fit['xy'] and original['covariance_px2']==fit['covariance_px2']
        assert np.allclose(fit['fit_covariance_px2'],np.asarray(original['covariance_px2'])+
            np.eye(2)*model['state_BI_total_error_floor_px2'][original['state']]*ratio**2)
        k=original['native_corner']
        assert result['keypoint_readout']['reported_points'][k]==original['xy']
    for k,state in enumerate(result['states']):
        if state=='S':
            assert np.allclose(result['keypoint_readout']['reported_points'][k],
                               result['keypoint_readout']['rigid_model_corners'][k])
            assert k not in result['state_observation']['rgb_queried_ids']


def test_full_API_S_raw_perturbation_cannot_change_calibration_pose_or_prompts():
    api,image,q,K,xyz,coarse,_=fixture()
    mask=np.ones(image.shape[:2])
    first=api.refine(image,q,K,xyz,coarse,pallet_mask=mask)
    changed=q.copy()
    for k,state in enumerate(first['states']):
        if state=='S':changed[k]=[1e8,-1e9]
    assert not np.array_equal(changed,q)
    second=api.refine(image,changed,K,xyz,coarse,pallet_mask=mask)
    for key in ['states','frozen_visible_measurements','calibrated_visible_measurements',
                'actual_pose','mask_audit','keypoint_readout']:
        assert first[key]==second[key],key


def test_no_prior_no_OT_control_has_truthful_roles_and_model_fingerprint():
    api,image,q,K,xyz,coarse,model=fixture()
    api.config['solver']['prior_weight']=0.
    result=api.refine(image,q,K,xyz,coarse,pallet_mask=np.ones(image.shape[:2]))
    assert result['accepted'] and not result['ot_domains']['domains']
    assert result['provenance']['initial_pose_is_initialization_only']
    assert not result['provenance']['initial_pose_is_soft_regularizer']
    assert not result['keypoint_readout']['OT_solved']
    assert not result['fallback_audit']['O_T_reported_as_solved']
    assert result['uncertainty_model_binding']['schema']==model['schema']
    assert len(result['uncertainty_model_binding']['canonical_model_sha256'])==64
