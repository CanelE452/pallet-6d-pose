"""Metadata admission tests; no RGB, annotation, model fit or pose solve."""
from copy import deepcopy

from . import evaluate_two_stage_top_normal_v4_16b as D


def _fixture():
    arm = dict(ambiguity_audit=dict(material_ambiguity=False),
               solution=dict(ambiguity=dict(detected=False), OT_solved=True),
               keypoint_readout=dict(OT_solved=True))
    baseline = dict(hierarchical_correspondence_audit=dict(
        initial_family_ambiguity=dict(detected=False)))
    stage1 = dict(families=[dict(accepted=False), dict(accepted=False)],
                  family_ambiguity=dict(detected=False), selected_family='ORIGINAL_WD')
    return arm, baseline, stage1


def test_no_accepted_initial_family_remains_unresolved_after_accepted_refinement():
    arm, baseline, stage1 = _fixture()
    before = deepcopy((baseline, stage1))
    result = D._preserve_ambiguity(arm, baseline, stage1)
    audit = result['hierarchical_correspondence_audit']
    assert audit['initial_joint_feasible_family_count'] == 0
    assert audit['initial_correspondence_unresolved']
    assert 'deterministic fallback' in audit['initial_family_selection_rule']
    assert result['ambiguity_audit']['material_ambiguity']
    assert result['solution']['ambiguity']['initial_correspondence_unresolved']
    assert not result['solution']['OT_solved']
    assert not result['keypoint_readout']['OT_solved']
    assert (baseline, stage1) == before


def test_accepted_family_does_not_create_unresolved_flag():
    arm, baseline, stage1 = _fixture()
    stage1['families'][0]['accepted'] = True
    result = D._preserve_ambiguity(arm, baseline, stage1)
    assert not result['hierarchical_correspondence_audit']['initial_correspondence_unresolved']
    assert not result['ambiguity_audit']['material_ambiguity']
    assert result['solution']['OT_solved']


def test_historical_or_current_family_ambiguity_survives_refinement():
    for historical in (True, False):
        arm, baseline, stage1 = _fixture()
        stage1['families'][0]['accepted'] = True
        baseline['hierarchical_correspondence_audit']['initial_family_ambiguity']['detected'] = historical
        stage1['family_ambiguity']['detected'] = not historical
        result = D._preserve_ambiguity(arm, baseline, stage1)
        assert result['ambiguity_audit']['material_ambiguity']
        assert not result['solution']['OT_solved']
