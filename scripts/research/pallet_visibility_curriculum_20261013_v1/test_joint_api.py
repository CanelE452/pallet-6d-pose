"""Optional joint API routing, geometric exclusions and physical-basis checks."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from . import common as C
from . import joint_api as J


def fixture():
    dims=np.array([1.1,.11,1.3]);R=cv2.Rodrigues(np.array([.35,-.4,.1]))[0]
    t=np.array([.05,-.1,3.]);K=np.array([[700.,0.,320.],[0.,700.,240.],[0.,0.,1.]])
    q=np.vstack([C.project(C.cuboid(*dims),R,t,K),[321.25,238.75]])
    initial=dict(available=True,R_cf=R.tolist(),R_physical=R.tolist(),centroid=t.tolist(),
                 cf_extents=dims.tolist(),selected_hypothesis='supplied_fixture')
    image=np.full((480,640,3),100,np.uint8)
    legacy=dict(schema='local_rgb_logistic_visibility_v1',mean=[0.]*26,scale=[1.]*26,
                coefficients=[0.]*26,intercept=2.,visible_threshold=.35,boundary_threshold=.35)
    return image,q,K,initial,dims,legacy


def install_stubs(monkeypatch,unknown_corner=None,broken_center=False):
    calls=[]
    def predict(image,q,H,initial,K,model):
        calls.append((image,q.copy(),H.copy(),deepcopy(initial)))
        h,w=image.shape[:2];valid=np.isfinite(q[:8]).all(1)&~np.all(q[:8]==-1,axis=1)
        eligible=valid&~H&(q[:8,0]>=0)&(q[:8,0]<w)&(q[:8,1]>=0)&(q[:8,1]<h)
        probability=np.where(eligible,.95,0.)
        if unknown_corner is not None:probability[unknown_corner]=.1
        corrected=q.copy()
        if broken_center:corrected[8]+=[1.,0.]
        return dict(P_usable=probability.tolist(),points=corrected.tolist(),RGB_delta_px=np.zeros((8,2)).tolist(),
                    eligible=eligible.tolist(),observed_refined_ids=[],threshold=.45,diagnostics={})
    def appearance(image,q,H,prob):
        h,w=image.shape[:2];states=[]
        for k,xy in enumerate(q):
            if not np.isfinite(xy).all() or np.all(xy==-1):s='UNKNOWN'
            elif not(0<=xy[0]<w and 0<=xy[1]<h):s='OUT_OF_FRAME'
            elif H[k]:s='SELF_OCCLUDED'
            else:s='UNKNOWN'
            states.append(s)
        return dict(states=states,visibility_score=[.05]*8,boundary_score=[.5]*8,
            local_accuracy_score=[.02]*8,diagnostics=dict(patch_radius_px=8,
                corners=[dict(reason='frozen_source_CNN_mask_and_raw_RGB_features') for _ in range(8)]))
    monkeypatch.setattr(J,'_predict_joint',predict)
    monkeypatch.setattr(J,'_segment',lambda image,weights,device:np.ones(image.shape[:2],np.float32))
    monkeypatch.setattr(J,'_segmented_appearance',appearance)
    return calls


def test_single_forward_immutable_arrays_and_retained_physical_basis(monkeypatch):
    image,q,K,initial,dims,legacy=fixture();calls=install_stubs(monkeypatch)
    before=[a.tobytes() for a in (image,q,K)];initial_before=deepcopy(initial)
    for a in (image,q,K):a.flags.writeable=False
    result=J.JointSpatialBranch(object(),'stub_weights',legacy).refine(image,q,K,initial,dims)
    assert len(calls)==1
    assert [a.tobytes() for a in (image,q,K)]==before and initial==initial_before
    np.testing.assert_array_equal(result['observed_points'],q)
    np.testing.assert_array_equal(result['corrected_measurements'][8],q[8])
    np.testing.assert_array_equal(result['completed_points'][8],q[8])
    assert not result['branch_result']['branch_flip']
    np.testing.assert_allclose(result['actual_pose']['R_physical'],initial['R_physical'],atol=1e-9)
    assert result['provenance']['projected_points_used_for_fit'] is False
    assert result['provenance']['default_V1_changed'] is False


def test_legacy_unknown_accuracy_floor_is_preserved(monkeypatch):
    image,q,K,initial,dims,legacy=fixture();install_stubs(monkeypatch,unknown_corner=0)
    result=J.JointSpatialBranch(object(),'stub_weights',legacy).refine(image,q,K,initial,dims)
    assert result['appearance']['states'][0]=='UNKNOWN'
    assert result['appearance']['local_accuracy_score'][0]==.12
    assert result['appearance']['visibility_score'][0]==.1
    assert result['visibility_assessment']['states'][0]=='VISIBLE_BOUNDARY'
    assert result['keypoint_status'][0]['rgb_visibility_state']=='VISIBLE_BOUNDARY'
    assert result['keypoint_status'][0]['observation_usability_state']=='LOW_USABILITY'
    assert result['keypoint_status'][0]['low_RGB_support_hypothesis'] is None


def test_missing_outside_self_priorities_and_fallback(monkeypatch):
    image,q,K,initial,dims,legacy=fixture();q[0]=[np.nan,np.nan];q[1]=[-5.,80.]
    # Disallow every RGB observation so neither branch can establish a fit.
    calls=install_stubs(monkeypatch)
    original=J._predict_joint
    def empty(*args,**kwargs):
        value=original(*args,**kwargs);value['P_usable']=[0.]*8;return value
    monkeypatch.setattr(J,'_predict_joint',empty)
    result=J.JointSpatialBranch(object(),'stub_weights',legacy).refine(image,q,K,initial,dims)
    assert result['appearance']['states'][0]=='UNKNOWN'
    assert result['appearance']['states'][1]=='OUT_OF_FRAME'
    hidden=np.flatnonzero(result['self_hidden']);assert len(hidden)>0
    assert all(result['appearance']['states'][k]=='SELF_OCCLUDED' for k in hidden)
    assert not result['branch_result']['accepted']
    np.testing.assert_allclose(result['actual_pose']['R_cf'],initial['R_cf'])
    np.testing.assert_allclose(result['actual_pose']['centroid'],initial['centroid'])
    assert result['completed_points'][0]==[None,None]
    assert all(not result['self_hidden'][k] for c in result['branch_result']['candidates']
               if c['available'] for stage in c['curriculum']['stages'] for k in stage['used_ids'])


def test_rejects_center_mutation_and_unregistered_model(monkeypatch):
    image,q,K,initial,dims,legacy=fixture();install_stubs(monkeypatch,broken_center=True)
    api=J.JointSpatialBranch(object(),'stub_weights',legacy)
    with pytest.raises(ValueError,match='center'):api.refine(image,q,K,initial,dims)
    with pytest.raises(ValueError,match='registered W/D'):api.refine(image,q,K,initial,[1.2,.11,1.3])


def test_requires_explicit_physical_rotation_and_legacy_support(monkeypatch):
    image,q,K,initial,dims,legacy=fixture();install_stubs(monkeypatch)
    with pytest.raises(ValueError,match='legacy'):J.JointSpatialBranch(object(),'stub_weights',None)
    del initial['R_physical']
    with pytest.raises(ValueError,match='R_physical'):
        J.JointSpatialBranch(object(),'stub_weights',legacy).refine(image,q,K,initial,dims)


def test_config_rejects_replaced_training_checkpoint_before_loading(tmp_path):
    for name in ('joint.json','source.pt','legacy.json'):
        (tmp_path/name).write_bytes(b'initial frozen bytes')
    config=dict(joint_model='joint.json',segmentation_weights='source.pt',
                legacy_visibility_model='legacy.json',usable_threshold=.45,
                model_sha256=dict(segmentation_weights=C.sha(tmp_path/'source.pt')))
    C.write(tmp_path/'config.json',config)
    (tmp_path/'source.pt').write_bytes(b'different training population')
    with pytest.raises(ValueError,match='model binding changed: segmentation_weights'):
        J.JointSpatialBranch.from_config(tmp_path/'config.json')


def test_fixed_family_api_matches_original_candidate_and_preserves_basis(monkeypatch):
    image,q,K,initial,dims,legacy=fixture();install_stubs(monkeypatch)
    compared=J.JointSpatialBranch(object(),'stub_weights',legacy).refine(image,q,K,initial,dims)
    fixed=J.JointSpatialBranch(object(),'stub_weights',legacy,
                              branch_policy='preserve_original_family').refine(image,q,K,initial,dims)
    original=compared['branch_result']['candidates'][0]
    assert len(fixed['branch_result']['candidates'])==1 and not fixed['branch_result']['branch_flip']
    assert fixed['actual_pose']['cf_extents']==initial['cf_extents']
    np.testing.assert_array_equal(fixed['actual_pose']['R_cf'],original['R_cf'])
    np.testing.assert_array_equal(fixed['actual_pose']['R_physical'],original['R_physical'])
    np.testing.assert_array_equal(fixed['actual_pose']['centroid'],original['centroid'])
    assert fixed['provenance']['registered_family_policy']=='preserve_original_family'
