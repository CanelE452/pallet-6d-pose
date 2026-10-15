"""Metadata-only fixed correspondence uncertainty check; no pose fitting."""
from copy import deepcopy
import pytest
from .evaluate_fixed_wd_archived_tangent_v4_15 import _preserve_frozen_family_uncertainty


@pytest.mark.parametrize('initial,within',[(True,False),(False,True),(False,False)])
def test_old_cross_family_uncertainty_survives_unsearched_family_refinement(initial,within):
    baseline=dict(hierarchical_correspondence_audit=dict(initial_family_ambiguity={'detected':initial},
        refined_within_family_ambiguity={'material_ambiguity':False},frozen_family='ORIGINAL_WD'))
    before=deepcopy(baseline)
    arm=dict(states=['O','B','S','U','T','B','S','B'],ambiguity_audit={'material_ambiguity':within},
        solution={'ambiguity':{'detected':within},'OT_solved':True,'OT_unique_solution_claimed':False},
        keypoint_readout={'reported_points':[[10.,20.],None,object(),None,[12.,22.],None,None,None],
            'point_sources':['joint']*8,'OT_solved':True,'OT_unique_solution_claimed':False})
    hidden=arm['keypoint_readout']['reported_points'][2]
    _preserve_frozen_family_uncertainty(arm,baseline)
    assert baseline==before and arm['keypoint_readout']['reported_points'][2] is hidden
    assert arm['hierarchical_correspondence_audit']['initial_family_ambiguity']['detected']==initial
    if initial or within:
        assert arm['ambiguity_audit']['material_ambiguity'] and not arm['solution']['OT_solved']
        assert arm['keypoint_readout']['point_sources'][0]=='AMBIGUOUS_FROZEN_FAMILY_JOINT_PROJECTION'
        assert arm['keypoint_readout']['point_sources'][4]=='AMBIGUOUS_FROZEN_FAMILY_JOINT_PROJECTION'
