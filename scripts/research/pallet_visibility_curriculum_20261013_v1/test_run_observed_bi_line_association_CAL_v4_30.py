import pytest
from . import run_observed_bi_line_association_CAL_v4_30 as R


def test_fixed_three_cases_cannot_be_reselected_or_reordered():
    assert R.fixed_cases(list(R.CASES))==R.CASES
    with pytest.raises(ValueError):R.fixed_cases(list(reversed(R.CASES)))
    with pytest.raises(ValueError):R.fixed_cases([*R.CASES[:2],'TEST:other'])


def test_actual_rerank_GO_precedes_all_IO(monkeypatch):
    monkeypatch.setattr(R.C,'read',lambda path:pytest.fail('No IO before actual rerank GO'))
    with pytest.raises(PermissionError):R.run()


def test_NEW_population_profile_explicitly_forbids_old_V29_noise_transfer():
    assert R.F.RECIPE['old_V29_noise_validated_for_new_population']is False
    assert R.F.RECIPE['Euclidean_corner_cap_or_along_line_cap']is False
    assert R.F.RECIPE['both_BI_endpoints_checked']is True
    assert R.F.OBSERVATION_GENERATION!=R.F.P.OBSERVATION_GENERATION


def test_only_saved_named_reference_projection_is_used_after_selection():
    node=dict(native_edge=[0,1],normal=[0.,1.],offset=-19.,actual_inliers=[[1.,19.],[3.,19.]])
    saved=dict(native_edge=[1,0],projected_CAL_reference_edge=[[10.,0.],[0.,0.]])
    result=R.compare_saved_reference_edge(node,saved)
    assert result['signed_normal_residual_px']==-19. and result['angle_deg']==0.
    node['native_edge']=[0,4]
    assert R.compare_saved_reference_edge(node,saved)is None
