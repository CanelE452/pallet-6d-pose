"""Synthetic source/visibility contracts; no actual images, labels or fits."""
from copy import deepcopy

import numpy as np
import pytest

from . import edge_visibility_v4_28 as V


class Poison:
    def __array__(self, *args, **kwargs): raise AssertionError('Forbidden coordinate conversion')
    def __iter__(self): raise AssertionError('Forbidden coordinate iteration')


class ModelProjection:
    def __init__(self, points): self.points = points
    def __getitem__(self, k):
        if k not in self.points: raise AssertionError('S model coordinate indexed')
        return self.points[k]


def inputs(mate_state='B', *, target=True):
    states = ['B',mate_state,'S','S','S','S','S','S']
    geometry = dict(image_hw=[96,96],initial_self_hidden=[s=='S' for s in states],
                    initial_projection_xy=ModelProjection({0:[16.,32.],1:[80.,32.]}))
    records = [dict(native_corner=0,xy=[16.,32.]),dict(native_corner=1,xy=[80.,32.] if mate_state in ('B','I') else Poison()),
               dict(native_corner=2,xy=Poison())]
    mask = np.zeros((96,96))
    if target: mask[32:,:] = 1.
    binding = dict(sha256='a'*64,image_hw=[96,96])
    return mask,geometry,states,records,binding


def context(mate_state='B', *, target=True, occluder=None):
    mask,geometry,states,records,binding = inputs(mate_state,target=target)
    occ_binding = None if occluder is None else dict(image_sha256='a'*64,image_hw=[96,96],
        sha256='b'*64,mask_pixel_sha256=V.mask_pixel_sha256(occluder),independent_of_target_mask=True,mask_role='NON_TARGET_OCCLUDER')
    return V.prepare_edge_visibility(mask,geometry,states,records,image_binding=binding,
        occluder_mask=occluder,occluder_binding=occ_binding)


def chain(): return np.column_stack([np.arange(16.,81.),np.full(65,32.)])


def test_missing_target_is_unknown_not_occlusion_even_when_line_points_are_given():
    result=context(target=False).classify_expected_edge(0,1,observed_chain_points=chain())
    assert result['counts'][V.OBSERVED_SUPPORT]==0
    assert result['counts'][V.OCCLUDER_SUPPORTED_GAP]==0
    assert result['counts'][V.UNSUPPORTED_UNKNOWN]==len(result['samples'])


def test_target_mask_without_actual_chain_does_not_manufacture_observed_edge():
    result=context().classify_expected_edge(0,1)
    assert result['counts'][V.OBSERVED_SUPPORT]==result['counts'][V.OCCLUDER_SUPPORTED_GAP]==0


def test_actual_chain_and_actual_target_background_context_support_observed_segment():
    result=context().classify_expected_edge(0,1,observed_chain_points=chain(),mask_context_kind='TARGET_BACKGROUND_OUTLINE')
    assert result['counts'][V.OBSERVED_SUPPORT]>30
    assert result['counts'][V.OCCLUDER_SUPPORTED_GAP]==0
    assert result['actual_chain_inlier_count']==65
    assert not result['full_KP_or_U_intersection_created']


def test_internal_requires_target_on_both_sides():
    result=context().classify_expected_edge(0,1,observed_chain_points=chain(),mask_context_kind='TARGET_TARGET_INTERNAL')
    assert result['counts'][V.OBSERVED_SUPPORT]==0


def test_independent_occluder_has_priority_and_removes_boundary_context_from_support():
    occ=np.zeros((96,96));occ[:,38:45]=1
    c=context(occluder=occ)
    result=c.classify_expected_edge(0,1,observed_chain_points=chain(),mask_context_kind='TARGET_BACKGROUND_OUTLINE')
    assert result['counts'][V.OCCLUDER_SUPPORTED_GAP]==7
    assert result['occluder_supported_gap_is_conditional_on_model_segment']
    assert all(s['classification']==V.OCCLUDER_SUPPORTED_GAP for s in result['samples'] if 38<=s['xy'][0]<=44)
    assert all(not s['fitting_query_allowed'] for s in result['samples'] if 31<=s['xy'][0]<=51)
    assert result['actual_chain_inlier_count']<result['actual_chain_inlier_input_count']


def test_occluder_context_gate_prevents_sampling_its_boundary_but_target_is_not_cut():
    occ=np.zeros((96,96));occ[:,40:43]=1
    c=context(occluder=occ)
    expected=np.zeros((96,96));expected[32:,:]=1
    np.testing.assert_array_equal(c.target_mask,expected)
    assert c.query_allowed(0,1,np.array([[34.,32.],[23.,32.]]),context_radius_px=7).tolist()==[False,True]
    fit=c.gated_sampling_inputs()
    assert fit['target_mask_not_cut_or_complemented'] and fit['apply_query_allowed_before_all_RGB_and_mask_remaps']
    with pytest.raises(ValueError):fit['target_mask'][40,40]=0


@pytest.mark.parametrize('mate_state',['U','O','T'])
def test_unknown_mate_endpoint_and_entire_probe_footprint_are_protected(mate_state):
    c=context(mate_state)
    allowed=c.query_allowed(0,1,np.array([[65.,32.],[55.,32.]]),context_radius_px=7)
    assert allowed.tolist()==[False,True] # distances15 and25 vs fixed14+probe7.
    result=c.classify_expected_edge(0,1,observed_chain_points=chain())
    assert not result['full_KP_or_U_intersection_created']
    assert all(np.linalg.norm(np.asarray(s['xy'])-[80,32])>=21 for s in result['samples'] if s['fitting_query_allowed'])


def test_U_anchor_or_S_mate_returns_before_candidate_or_chain_fields():
    c=context('U')
    assert c.query_allowed(1,0,Poison()) is False
    assert not c.classify_expected_edge(1,0,observed_chain_points=Poison())['eligible']
    assert c.query_allowed(0,3,Poison()) is False
    assert not c.classify_expected_edge(0,3,observed_chain_points=Poison())['eligible']


def test_nonincident_edge_is_rejected_before_model_or_chain_access():
    c=context()
    assert c.query_allowed(0,2,Poison()) is False


@pytest.mark.parametrize('change',[dict(independent_of_target_mask=False),dict(mask_role='TARGET_COMPLEMENT'),dict(image_sha256='c'*64)])
def test_occluder_must_have_independent_non_target_source_binding(change):
    mask,geometry,states,records,binding=inputs()
    ob=dict(image_sha256='a'*64,image_hw=[96,96],sha256='b'*64,independent_of_target_mask=True,mask_role='NON_TARGET_OCCLUDER')
    ob['mask_pixel_sha256']=V.mask_pixel_sha256(np.zeros((96,96)))
    ob.update(change)
    with pytest.raises(ValueError,match='Independently'):
        V.prepare_edge_visibility(mask,geometry,states,records,image_binding=binding,occluder_mask=np.zeros((96,96)),occluder_binding=ob)


def test_occluder_file_binding_without_exact_array_digest_cannot_certify_actual_pixels():
    mask,geometry,states,records,binding=inputs()
    occ=np.zeros((96,96));occ[20:30,20:30]=1
    ob=dict(image_sha256='a'*64,image_hw=[96,96],sha256='b'*64,independent_of_target_mask=True,
            mask_role='NON_TARGET_OCCLUDER',mask_pixel_sha256=V.mask_pixel_sha256(np.zeros((96,96))))
    with pytest.raises(ValueError,match='occupancy-array digest'):
        V.prepare_edge_visibility(mask,geometry,states,records,image_binding=binding,occluder_mask=occ,occluder_binding=ob)


def test_None_and_separately_bound_empty_occluder_have_same_support_and_query_mask():
    a,b=context(),context(occluder=np.zeros((96,96)))
    aa=a.classify_expected_edge(0,1,observed_chain_points=chain())
    bb=b.classify_expected_edge(0,1,observed_chain_points=chain())
    assert aa['counts']==bb['counts'] and aa['samples']==bb['samples']
    np.testing.assert_array_equal(a.query_allowed(0,1,chain()),b.query_allowed(0,1,chain()))


def test_large_outside_model_segment_is_clipped_before_sampling_not_allocated_unbounded():
    mask,geometry,states,records,binding=inputs()
    geometry['initial_projection_xy']=ModelProjection({0:[-1e9,32],1:[1e9,32]})
    c=V.prepare_edge_visibility(mask,geometry,states,records,image_binding=binding)
    result=c.classify_expected_edge(0,1)
    assert 0<len(result['samples'])<=98
    assert all(0<=s['xy'][0]<=95 for s in result['samples'])


def test_input_arrays_and_records_are_unchanged_and_mask_snapshot_is_stable():
    mask,geometry,states,records,binding=inputs()
    original=mask.copy();before=[r['native_corner']for r in records]
    c=V.prepare_edge_visibility(mask,geometry,states,records,image_binding=binding)
    c.classify_expected_edge(0,1,observed_chain_points=chain())
    np.testing.assert_array_equal(mask,original)
    assert before==[r['native_corner']for r in records]
    mask[:]=0
    np.testing.assert_array_equal(c.target_mask,original)


def test_all_canvas_probe_sites_are_safe_before_occupancy_lookup():
    c=context()
    sites=np.array([[0.,32.],[95.,32.],[48.,0.],[48.,95.],[48.,32.]])
    assert c.query_allowed(0,1,sites,context_radius_px=7).tolist()==[False,False,False,False,True]
