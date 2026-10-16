import pytest

from scripts.research.pallet_selection_score_diagnosis_v1.pair_analysis import (
    component_delta, lineage_link, observation_deltas,
)


def cost(stage="G13",family="ORIGINAL_WD",parts=None,observations=None):
    parts=parts or dict(full_BI=2.,partial_normal=1.,outline=3.,prior=0.)
    return dict(id="frame",stage=stage,family=family,objective=sum(parts.values()),
                weighted_components=parts,per_observation=observations or [])


def test_pair_difference_is_same_objective_only_and_additive():
    selected=cost(); other=cost(parts=dict(full_BI=3.,partial_normal=.5,outline=4.,prior=0.))
    delta,total,error=component_delta(selected,other)
    assert delta==dict(full_BI=1.,partial_normal=-.5,outline=1.,prior=0.)
    assert total==1.5 and error==0.
    with pytest.raises(ValueError,match="Never subtract"):
        component_delta(selected,cost(stage="G10"))
    other["objective"]+=.01
    with pytest.raises(ValueError,match="does not equal"):
        component_delta(selected,other)


def test_cross_family_outline_never_matches_projected_sample_indices():
    observed=[dict(loss_family="outline",observation_id="outline:model_to_observed",weighted_contribution=3.,count=30,normalizer=75.)]
    selected=cost(observations=observed)
    other=cost(family="SWAPPED_WD",observations=[dict(observed[0],weighted_contribution=4.,count=45,normalizer=90.)])
    delta=observation_deltas(selected,other,[])[0]
    assert delta["delta"]==1.
    assert delta["observation_alignment"]=="DIRECTION_TOTALS_ONLY_DIFFERENT_PROJECTED_SAMPLES_NO_SAMPLE_INDEX_MATCH"
    assert delta["observation_s"]["count"]==30 and delta["observation_other"]["count"]==45


def test_fallback_rays_are_only_attempted_sources_and_never_independent():
    obs=dict(loss_family="full_BI",native_corner=2,inherited_predictor_fallback=True)
    ray=dict(native_corner=2,origin="original_pre_e_BI_recovery",exact_sample_source_group="shared",
             ray=dict(edge=4,supported=True),semantic_owner_status="SEMANTIC_OWNER_UNVERIFIED")
    result=lineage_link(obs,[ray],"G13")
    assert result["matches"][0]["exact_sample_source_group"]=="shared"
    assert result["status"]=="ATTEMPTED_RAYS_NOT_A_DEPENDENCY_OF_PREDICTOR_FALLBACK"
    assert result["source_group_is_independence_proof"] is False


def test_partial_source_requires_stored_line_normal_and_offset():
    obs=dict(loss_family="partial_normal",native_corner=2,association_provenance={},
             lines=[dict(line_index=0,normal_unit_xy=[1.,0.],offset_px=-4.)])
    base=dict(native_corner=2,origin="original_pre_e_BI_recovery",exact_sample_source_group="shared",
              semantic_owner_status="SEMANTIC_OWNER_UNVERIFIED")
    rays=[dict(base,ray=dict(normal=[1.,0.],offset=-4.,edge=1)),
          dict(base,ray=dict(normal=[0.,1.],offset=-4.,edge=2))]
    result=lineage_link(obs,rays,"G10")
    assert len(result["matches"])==1 and result["matches"][0]["original_edge"]==1
    result=lineage_link(obs,rays[1:],"G10")
    assert result["matches"]==[]
    assert result["status"]=="NO_RECORDED_RAY_MATCH_AT_NATIVE_CORNER_AND_ORIGIN"
