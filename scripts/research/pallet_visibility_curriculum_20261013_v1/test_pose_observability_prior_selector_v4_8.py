"""Structural choice checks, not an accuracy/coverage benchmark."""
from copy import deepcopy

import pytest

from . import common as C
from .pose_observability_prior_selector_v4_8 import choose_prior_by_observability as choose


def inputs(ids=(0,1,3,5), partial=None, rank=6):
    states=['U']*8
    for k in ids:states[k]='B'
    visible=[{'native_corner':k,'xy':object(),'covariance_px2':object()} for k in ids]
    solution=dict(states=states,accepted=True,rank={'visible':{'rank':rank,'condition':5.}},
        ambiguity={'detected':False},OT_constraints_satisfied=True,
        objective_components={'soft_BI_robust_loss':1.},joint_constraint_audit=[])
    prior={**solution,'objective_components':{'soft_BI_robust_loss':2.}}
    return C.cuboid(1.1,.11,1.3),states,visible,partial or [],prior,solution


def test_non_coplanar_redundant_full_2D_admits_no_prior_without_reading_xy():
    a=inputs();before=deepcopy(a[1]);result=choose(*a)
    assert result['no_prior_admissible'] and result['chosen_arm']=='NO_PRIOR_FROZEN_WEIGHTS'
    assert result['diagnostics']['complete_native_affine_rank']==3
    assert a[1]==before and result['GT_or_frame_ID_inputs'] is False


@pytest.mark.parametrize('ids',[(0,4,7),(0,3,4,7),(1,2,4,7)])
def test_minimal_P3P_or_coplanar_complete_geometry_preserves_prior(ids):
    r=choose(*inputs(ids))
    assert r['chosen_arm']=='SAME_V47' and r['no_prior_admissible'] is False
    assert r['checks']['complete_native_geometry_non_coplanar'] is False


def test_partial_normals_do_not_fake_full_points_or_jointguard_image_rank():
    a=inputs((1,2,4),partial=[{'native_corner':1,'lines':[object()]},{'native_corner':2,'lines':[object()]}],rank=2)
    a[-1]['rank'].update(BI_and_partial_boundary={'rank':4,'condition':None},visible_and_active_joint={'rank':6})
    r=choose(*a)
    assert r['chosen_arm']=='SAME_V47' and r['diagnostics']['image_only_rank']==4
    assert r['diagnostics']['image_residual_rows']==4 and r['diagnostics']['complete_2D_observations']==1


def test_ambiguity_or_failed_individual_region_guard_rejects_candidate():
    a=inputs();a[-1]['ambiguity']={'detected':True}
    assert choose(*a)['chosen_arm']=='SAME_V47'
    a=inputs();a[-1]['U_region_constraint_audit']=[{'feasible':False}]
    assert choose(*a)['chosen_arm']=='SAME_V47'


def test_S_gate_before_coordinate_or_line_fields():
    a=inputs();a[1][0]='S'
    with pytest.raises(ValueError,match='Only fixed B/I'):
        choose(*a)
    a=list(inputs());a[1][0]='S';a[2]=[r for r in a[2] if r['native_corner']!=0]
    a[3].append({'native_corner':0,'lines':object()})
    with pytest.raises(ValueError,match='Only B partial'):
        choose(*a)
