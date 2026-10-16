"""Meaningful comparator, eligibility, and role invariants for fixed records."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.research.pallet_selection_score_diagnosis_v1.bank import group_for_oracle
from scripts.research.pallet_selection_score_diagnosis_v1.selection import (
    discrete_family_choice, rank_retained, same_pose,
)


def candidate(cid, score, name, order, schema="statewise_partial_BI_outline_solver_v4_13"):
    return dict(id="frame", stage="G13", family="ORIGINAL_WD", solver_schema=schema,
        retained=True, candidate_id=cid, original_outcome_order=order,
        original_record=dict(objective=score, initialization=name))


def test_exact_family_tie_and_rejected_cheaper_family():
    families = [dict(accepted=True, original_objective=1.), dict(accepted=True, original_objective=1.)]
    assert discrete_family_choice(families) == 0
    families[0] = dict(accepted=False, original_objective=-100.)
    assert discrete_family_choice(families) == 1
    families[1]["accepted"] = False
    assert discrete_family_choice(families) == 0  # Explicit own-family fallback.


def test_comparator_name_tie_and_stable_delegated_parent():
    first, second = candidate("a", 2., "Z", 0), candidate("b", 2., "A", 1)
    assert [c["candidate_id"] for c in rank_retained([first, second])] == ["b", "a"]
    for c in (first, second):
        c["solver_schema"] = "statewise_u_region_solver_v4_5"
    assert [c["candidate_id"] for c in rank_retained([second, first])] == ["a", "b"]


def test_fixed_score_diagnosis_preserves_pool_and_rejects_cross_package():
    values = [candidate("a", 1., "A", 0), candidate("b", 2., "B", 1)]
    before = deepcopy(values)
    assert [c["candidate_id"] for c in rank_retained(values, dict(a=2., b=1.))] == ["b", "a"]
    assert values == before
    with pytest.raises(ValueError):
        rank_retained(values, dict(a=1.))
    values[1]["family"] = "SWAPPED_WD"
    with pytest.raises(ValueError):
        rank_retained(values)


def test_dedup_is_original_fixed_pose_predicate():
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_ot_pnp as source
    origin = dict(R=np.eye(3), t=np.zeros(3))
    for distance in (0., 0.999e-5, 1e-5, 1.001e-5):
        other = dict(R=np.eye(3), t=np.array([distance, 0., 0.]))
        assert same_pose(origin, other, 1.) == source._same_pose(
            (origin["R"], origin["t"]), (other["R"], other["t"]), 1.)
    assert same_pose(origin, dict(R=np.eye(3), t=np.zeros(3)), 1.)


def test_group_mapping_requires_original_complete_consumer_scope():
    oracle = dict(final_joint_success=False, earliest_observed_failure_stage="family_selection",
                  original_consumer_generated_bank_complete=True)
    assert group_for_oracle(oracle) == "FAMILY_SELECTION_LOSS"
    oracle["original_consumer_generated_bank_complete"] = False
    with pytest.raises(ValueError):
        group_for_oracle(oracle)
