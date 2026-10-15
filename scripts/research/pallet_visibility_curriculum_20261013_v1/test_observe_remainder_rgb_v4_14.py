"""Focused carrier/recovery contracts; no real forwards, fits, or labels."""
from copy import deepcopy
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from . import observe_remainder_rgb_v4_14 as O


def correction():
    states=['B','I','S','U','O','T','B','S']
    raw=[[10.+k,20.] for k in range(9)]
    raw[2]=raw[7]=[-1.,-1.]
    measurements=[dict(native_corner=k,state=states[k],xy=raw[k],covariance_px2=[[1.,0.],[0.,1.]])
                  for k in (0,1,6)]
    return dict(raw_points=raw, measured_points=deepcopy(raw),visible_measurements=measurements,
                records=[dict(native_corner=k,state=s,corrected=k==0) for k,s in enumerate(states)],
                recovery_queried_native_ids=[])


def source():
    c=correction()
    return dict(visible_correction=c,states=[r['state'] for r in c['records']],
        initial_geometry=dict(initial_self_hidden=[False,False,True,False,False,False,False,True],
            K=np.eye(3).tolist(),R_cf=np.eye(3).tolist(),centroid=[0.,0.,2.],cf_extents=[1.,.1,1.],
            native_xyz=np.zeros((8,3)).tolist()))


def test_recovery_stage_preserves_successes_and_uses_exact_order():
    s=source(); snapshots=[]
    def recover(image,raw,K,initial,states,*,pallet_mask,base_correction,native_xyz):
        result=deepcopy(base_correction); snapshots.append((pallet_mask[0,0],deepcopy(result)))
        if len(snapshots)==1:
            result['visible_measurements'][-1]['xy']=[16.,21.]
            result['records'][6]['corrected']=True
            result['recovery_queried_native_ids']=[6]
        return result
    with patch.object(O.RECOVER,'recover_visible_boundary',side_effect=recover), \
         patch.object(O.REACH,'recover_visible_boundary',side_effect=recover):
        result,audit=O.recover_stages(np.zeros((2,2,3),np.uint8),s,
                                    np.zeros((2,2)),np.ones((2,2)))
    assert [v[0] for v in snapshots]==[0,1,1]
    assert snapshots[1][1]['visible_measurements'][-1]['xy']==[16.,21.]
    assert snapshots[2][1]['visible_measurements'][-1]['xy']==[16.,21.]
    assert result['visible_measurements'][:2]==s['visible_correction']['visible_measurements'][:2]
    assert result['measured_points'][8]==s['visible_correction']['measured_points'][8]
    assert [v['stage'] for v in audit]==list(O.STAGE_NAMES)


@pytest.mark.parametrize('violation',['successful_B','I','S_query'])
def test_bad_recovery_cannot_change_existing_BI_or_query_S(violation):
    s=source()
    def bad(*args,**kwargs):
        result=deepcopy(kwargs['base_correction'])
        if violation=='S_query': result['recovery_queried_native_ids']=[2]
        else: result['visible_measurements'][0 if violation=='successful_B' else 1]['xy']=[99.,99.]
        return result
    with patch.object(O.RECOVER,'recover_visible_boundary',side_effect=bad),pytest.raises(ValueError):
        O.recover_stages(np.zeros((2,2,3),np.uint8),s,np.zeros((2,2)),np.zeros((2,2)))


def test_recording_provider_returns_unmodified_bundle():
    returned={'pallet_mask':np.ones((2,2)), 'state_pallet_mask':np.zeros((2,2))}
    calls=[]
    def provider(image,**kwargs): calls.append(kwargs);return returned
    wrapped=O.RecordingMaskProvider(provider)
    assert wrapped(np.zeros((2,2,3),np.uint8),prompt_native_ids=[1]) is returned
    assert wrapped.last is returned and calls==[{'prompt_native_ids':[1]}]


def test_observer_never_accesses_original_S_carrier_or_calls_pose(tmp_path):
    class Poison:
        def __array__(self,*args): raise AssertionError('Original raw carrier read')
        def __iter__(self): raise AssertionError('Original raw carrier read')
    image=np.zeros((4,4,3),np.uint8); path=tmp_path/'image.png';cv2.imwrite(str(path),image)
    s=source(); g=s['initial_geometry']
    g.update(initial_projection_xy=np.zeros((8,2)).tolist(),initial_roles=['B']*8,
             image_hw=[4,4],physical_registration_Q=np.eye(3).tolist())
    s.update(original_raw_points=deepcopy(s['visible_correction']['raw_points']),
             frozen_visible_measurements=deepcopy(s['visible_correction']['visible_measurements']),
             ot_domains={'nonself_prediction_scale_px':20.},
             mask_audit={'prompt_native_ids':[0,1,6]},state_observation={'rgb_queried_ids':[0,1,6]})
    frame=dict(id='fixture:1',partition='REMAINDER',image=str(path),image_sha256=O.C.sha(path),
        raw_hw=[4,4],raw_points=Poison(),K=g['K'],native_xyz=g['native_xyz'],
        initial_pose={},crop_transform=None,session='fixture',cohort='REMAINDER',
        known_physical_dimensions_xyz_m=[1.,.1,1.],prediction_object_type='fixture')
    carrier=dict(frame=frame,initial_geometry=g,raw_points_initial_S_gated=s['original_raw_points'])
    tight=SimpleNamespace(last=None,delegate=SimpleNamespace(predictor=None))
    class API:
        def __init__(self,*args,**kwargs): pass
        def observe(self,image,raw,*args,**kwargs):
            assert raw is carrier['raw_points_initial_S_gated']
            tight.last={'pallet_mask':np.zeros((4,4)), 'state_pallet_mask':np.zeros((4,4)), 'provenance':{}}
            return deepcopy(s)
        def solve_observations(self,*args): raise AssertionError('Pose fit forbidden')
    class Full:
        predictor=None
        def __call__(self,image,*,prompt,state_pallet_mask):
            return {'state_pallet_mask':state_pallet_mask.copy(),'pallet_mask_clean':np.zeros((4,4)), 'provenance':{}}
    fit=[dict(m,fit_covariance_px2=[[8.,0.],[0.,8.]],fit_reliability_weight=.8)
         for m in s['frozen_visible_measurements']]
    bundle=dict(head=None,ot=None,tight=tight,full=Full(),model={},interval={},config={},device='cpu')
    with patch.object(O,'StatewiseIntervalV4_2',API), \
         patch.object(O.MASK,'build_geometry_prompt',return_value={}), \
         patch.object(O,'apply_calibrated_measurements',return_value=fit), \
         patch.object(O,'recover_stages',return_value=(s['visible_correction'],[])), \
         patch.object(O,'build_u_regions',return_value={'u_regions':[]}), \
         patch.object(O,'observe_shared_supported_b_lines',return_value={'boundary_observations':[]}):
        row,_=O.observe_remainder_native_frame_v4_14(carrier,bundle)
    assert row['_original_source']['original_raw_points'][2]==[-1.,-1.]
    assert row['frozen_BI_fit'][0]['fit_covariance_px2']==[[8.,0.],[0.,8.]]
    assert row['contract']['pose_solver_calls']==0
