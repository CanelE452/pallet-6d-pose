from copy import deepcopy
import numpy as np
from . import common as C
from .visible_i_posefix_alternatives_v4_12 import apply_posefix_fallback_I,MODES


def _inputs():
    g=dict(cf_extents=[1.,.2,1.],native_xyz=C.cuboid(1.,.2,1.).tolist(),physical_registration_Q=np.eye(3).tolist(),
        initial_self_hidden=[False]*7+[True],image_hw=[480,640])
    states=['I','B','I','U','O','T','B','S']
    def measurement(k,source):return dict(native_corner=k,state=states[k],source=source,xy=[100.,200.],
        covariance_px2=[[4.,0.],[0.,4.]],fit_covariance_px2=[[50.,2.],[2.,30.]],fit_reliability_weight=.8)
    visible=[measurement(0,'VISIBLE_BACKBONE_FALLBACK'),measurement(1,'B_TARGET_BACKGROUND_LINES'),measurement(2,'I_INTERNAL_RGB_JUNCTION')]
    pf=dict(method='POSEFIX_RAW',native_points=[[110.,220.]]*9,actual_pose=dict(available=True,cf_extents=g['cf_extents'],R_cf=np.eye(3).tolist(),R_physical=np.eye(3).tolist()))
    return g,states,visible,pf


def test_only_fallback_I_coordinate_can_be_read_with_all_other_slots_poisoned():
    g,states,visible,pf=_inputs()
    class Points:
        def __len__(self):return 9
        def __getitem__(self,k):
            assert k==0,'Excluded PF coordinate read'
            return [110.,220.]
    pf['native_points']=Points()
    result=apply_posefix_fallback_I(visible,g,states,pf,MODES[0])
    assert result['PF_coordinate_native_ids_read']==[0]
    assert result['visible_measurements'][1:]==visible[1:]
    assert result['S_PF_coordinate_reads']==0


def test_template_Q_mismatch_rejects_before_any_PF_coordinate_index():
    g,states,visible,pf=_inputs()
    class Points:
        def __len__(self):raise AssertionError('Coordinate list accessed before Q gate')
        def __getitem__(self,k):raise AssertionError('Coordinate indexed before Q gate')
    pf['native_points']=Points();pf['actual_pose']['R_physical']=[[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]]
    result=apply_posefix_fallback_I(visible,g,states,pf,MODES[0])
    assert result['visible_measurements']==visible and result['PF_coordinate_native_ids_read']==[]


def test_RAW_and_HALF_are_fixed_correlated_controls_with_noise_archive_unchanged():
    g,states,visible,pf=_inputs();original=deepcopy(visible)
    raw=apply_posefix_fallback_I(visible,g,states,pf,MODES[0])['visible_measurements'][0]
    half=apply_posefix_fallback_I(visible,g,states,pf,MODES[1])['visible_measurements'][0]
    assert raw['xy']==[110.,220.] and half['xy']==[105.,210.]
    for field in ('covariance_px2','fit_covariance_px2','fit_reliability_weight'):
        assert raw[field]==half[field]==original[0][field]
    assert visible==original and not half['fusion_assumes_independent_measurements']


def test_missing_and_outside_PF_preserve_original_reference():
    for point in ([-1.,-1.],[640.,10.],[np.nan,20.]):
        g,states,visible,pf=_inputs();pf['native_points'][0]=point
        result=apply_posefix_fallback_I(visible,g,states,pf,MODES[0])
        assert result['visible_measurements']==visible and result['changed_native_ids']==[]
