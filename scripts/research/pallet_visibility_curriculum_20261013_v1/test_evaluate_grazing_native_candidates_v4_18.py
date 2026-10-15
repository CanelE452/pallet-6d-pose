"""Essential no-fit caller contracts for the two explicit cached controls."""
from copy import deepcopy
import numpy as np
import pytest
from . import common as C
from . import evaluate_grazing_native_candidates_v4_18 as V
from .test_geometric_visibility_uncertainty_v4_9 import fixture


def candidate(name,native,mask,accepted=True):
    return dict(name=name,family=name,accepted=accepted,all_joint_constraints_satisfied=accepted,
        native_costs=[dict(kind='BI_COMPLETE',native_corner=0,weighted_robust_cost=native)],global_mask_cost=mask)


def test_declared_native_plus_mask_is_an_actual_separate_control():
    rows=[candidate('native',.2,4.),candidate('mask',.7,.1)];before=deepcopy(rows)
    assert V.select_native_plus_mask(rows,0.)['selected_name']=='native'
    assert V.select_native_plus_mask(rows,1.)['selected_name']=='mask'
    assert rows==before


def test_rejected_candidate_gates_energy_fields_and_undefined_mask_does_not_become_zero():
    rows=[dict(name='bad',accepted=False,all_joint_constraints_satisfied=False,native_costs=object()),
        candidate('undefined_mask',.1,None),candidate('defined',1.,2.)]
    assert V.select_native_plus_mask(rows,1.)['selected_name']=='defined'
    assert V.select_native_plus_mask(rows,0.)['selected_name']=='undefined_mask'


@pytest.mark.parametrize('weight',[-1.,float('nan')])
def test_invalid_new_mask_weights_reject(weight):
    with pytest.raises(ValueError):V.select_native_plus_mask([],weight)


def test_adapter_drops_new_U_payload_before_reading_it_and_preserves_OT_numeric_inputs():
    g,s=fixture();s[7]='I';s[1]='O'
    class Raw:
        def __len__(self):return 9
        def __getitem__(self,k):
            assert k==7,'Forbidden raw coordinate read'
            return[-20.,30.]
    source=dict(initial_geometry=g,original_raw_points=Raw(),ot_domains=dict(nonself_prediction_scale_px=100.))
    olddomain=dict(native_corner=1,state='O',shape='ellipse',center_xy=[5,6],half_axes_px=[8,8])
    visible=[dict(native_corner=7,xy=object()),dict(native_corner=0,xy=[10,20],source='old',covariance_px2=[[4,0],[0,4]])]
    task=dict(states=s,spec=dict(states=s,visible=visible,domains=[olddomain],config=dict(geometry_tolerance_deg=2.)),
        initial_self_hidden=g['initial_self_hidden'],families=dict(ORIGINAL_WD=dict(native_xyz=g['native_xyz'],coarse_pose=dict(R=g['R_cf'],t=g['centroid']))),
        u_regions=[],boundary_merged=[dict(native_corner=7,lines=object())],boundary_original=object(),baseline={})
    out=V.adapt_cached_task(task,source,C.read(V.REGION_MODEL))
    assert out['states'][7]=='U'and out['states'][6]=='S'
    assert [m['native_corner']for m in out['spec']['visible']]==[0]
    assert out['boundary_merged']==[]and out['spec']['domains']==[olddomain]
    assert out['u_regions'][0]['center_xy']==[-20.,30.]
    assert out['u_regions'][0]['halfplanes']==[]
    assert out['grazing_adaptation']['original_OT_numeric_domains_sha256']==out['grazing_adaptation']['adapted_OT_numeric_domains_sha256']


def test_unaffected_measurements_remain_archived_not_recalibrated():
    g,s=fixture();s[7]='U'
    source=dict(initial_geometry=g,original_raw_points=None,ot_domains=dict(nonself_prediction_scale_px=100.))
    m=dict(native_corner=0,xy=[10,20],source='RGB_CORRECTED',covariance_px2=[[2,0],[0,3]],
        fit_covariance_px2=[[9,1],[1,10]],fit_reliability_weight=.8)
    task=dict(states=s,spec=dict(states=s,visible=[m],domains=[],config=dict(geometry_tolerance_deg=2.)),
        initial_self_hidden=g['initial_self_hidden'],families=dict(ORIGINAL_WD=dict(native_xyz=g['native_xyz'],coarse_pose=dict(R=g['R_cf'],t=g['centroid']))),
        u_regions=[],boundary_merged=[],baseline={})
    out=V.adapt_cached_task(task,source,C.read(V.REGION_MODEL))
    assert out['states']==s and out['spec']['visible']==[m]
    assert out['grazing_adaptation']['geometry_gate']['new_U_native_ids']==[]
