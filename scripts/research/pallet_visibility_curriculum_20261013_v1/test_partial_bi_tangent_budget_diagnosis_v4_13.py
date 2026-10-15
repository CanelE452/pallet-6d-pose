from copy import deepcopy
import numpy as np
import pytest
from .partial_bi_tangent_budget_diagnosis_v4_13 import typed_tangent_supplement as budget


class Poison:
    def __array__(self,*args,**kwargs):raise AssertionError('Excluded coordinate read')


def fixture():
    states=['I']+['U']*6+['S'];q=[[15.,30.]]+[Poison() for _ in range(7)]
    record=dict(native_corner=0,state='I',explicit_complete_BI_likelihood_revocation=True,
        lines=[dict(normal_unit_xy=[0.,1.])],association_provenance=dict(
        original_observed_xy=[10.,10.],original_effective_fit_covariance_px2=[[4.,0.],[0.,9.]],
        original_fit_reliability_weight=.8,original_measurement_source='I_OLD_RGB_CORRECTION'))
    return q,states,[record]


def test_only_tangent_cost_original_normal_and_excluded_points_not_consumed():
    q,s,r=fixture();a=budget(q,s,r)
    q[0][1]=1000000.;b=budget(q,s,r)
    assert a['cost']==b['cost'] and a['records'][0]['tangent_residual_px']==-5.
    assert a['records'][0]['old_measurement_source']=='I_OLD_RGB_CORRECTION'
    assert not a['normal_duplication'] and a['U_S_or_OT_observation_terms']==0


def test_quarter_weight_means_covariance_times_four_inside_same_Huber():
    args=fixture();a=budget(*args,inverse_variance_weight=.25)
    assert a['records'][0]['effective_tangent_variance_px2']==16.
    z=-5/4
    assert a['cost']==pytest.approx(.8*18*(np.sqrt(1+(z/3)**2)-1))
    assert a['cost']!=pytest.approx(.25*budget(*args)['cost'])


def test_zero_is_exact_zero_without_archive_or_projection_lookup():
    q,s,r=fixture();r[0]['association_provenance']=Poison();r[0]['lines']=Poison()
    assert budget(Poison(),s,r,inverse_variance_weight=0)['cost']==0.


def test_S_U_gates_precede_poisoned_archive_and_line_access():
    q,s,r=fixture()
    for state in ('S','U','O','T'):
        s[0]=state;r[0]['state']=state;r[0]['lines']=Poison();r[0]['association_provenance']=Poison()
        with pytest.raises(ValueError,match='B/I'):budget(q,s,r)


def test_input_covariance_archive_and_record_are_unchanged():
    q,s,r=fixture();before=deepcopy(r)
    budget(q,s,r)
    assert r==before
    r[0]['association_provenance']['original_effective_fit_covariance_px2']=[[1.,2.],[2.,1.]]
    with pytest.raises(np.linalg.LinAlgError):budget(q,s,r)
