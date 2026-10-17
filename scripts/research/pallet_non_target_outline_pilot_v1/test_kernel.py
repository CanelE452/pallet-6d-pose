"""API fixtures for missing observations, parent lineage, and safe selection.

These small invented geometries test the implementation contract. They are
not research data, physical labels, segmentation accuracy or pose performance.
"""
from copy import deepcopy
import subprocess
import sys

import numpy as np
import pytest

from scripts.research.pallet_non_target_outline_pilot_v1.kernel import (
    evaluate_outline, lineage_for_outline, nearest_original_segment, parent_support)
from scripts.research.pallet_non_target_outline_pilot_v1.replay import (
    DENIED_KEYS, ReferenceFreeIO, _deny_fields, choose)


def test_preclip_endpoint_is_not_canvas_intersection():
    outline = dict(supported=True, image_hw=[6, 6],
        configuration=dict(frame_margin_px=1, observed_sample_spacing_px=1),
        polygon_xy=[[0, 2], [4, 2], [4, 4], [0, 4]],
        observed_boundary_segments_xy=[[[1, 2], [4, 2]], [[4, 2], [4, 4]], [[4, 4], [1, 4]]],
        observed_boundary_samples_xy=[[1.5, 2], [2.5, 2], [3.5, 2], [4, 2.5], [4, 3.5],
                                     [3.5, 4], [2.5, 4], [1.5, 4]])
    lineage = lineage_for_outline(outline)
    mask = np.zeros((6, 6), dtype=bool)
    mask[2, 1] = True  # canvas intersection, not original endpoint
    assert not any(row['censor'] for row in parent_support(lineage, mask))
    mask[2, 0] = True  # actual original endpoint outside the margin
    support = parent_support(lineage, mask)
    assert [row['censor'] for row in support] == [True, False, False]
    assert support[0]['endpoint_pixels_xy'] == [[0, 2], [4, 2]]
    assert lineage['sample_parent_segment_ids'] == [0, 0, 0, 1, 1, 2, 2, 2]


def test_nearest_original_ties_and_no_re_nearest_to_survivor():
    segments = [[[0, 0], [0, 4]], [[4, 0], [4, 4]]]
    parents, residual = nearest_original_segment([[1, 2], [3, 2], [2, 2]], segments)
    assert parents.tolist() == [0, 1, 0]  # exact tie retains original first
    assert residual.tolist() == [[1, 0], [-1, 0], [2, 0]]
    keep = ~np.asarray([True, False])[parents]
    assert keep.tolist() == [False, True, False]
    # Deleting parent0 and re-querying parent1 would keep all three and create
    # a new residual -3 for the first sample. The contract explicitly forbids it.
    assert not keep[0]


def cost_fixture():
    candidate = dict(candidate_id='fixture', stage='G13', family='ORIGINAL_WD',
                     original_record=dict(objective=20.))
    outline = dict(observed_boundary_samples_xy=[[0, 2], [4, 2]],
                   observed_boundary_segments_xy=[[[0, 0], [0, 4]], [[4, 0], [4, 4]]])
    lineage = dict(used_segments=[{}, {}], sample_parent_segment_ids=[0, 1])
    dense = dict(objective=20., weighted_components=dict(full_BI=8., partial_normal=0., outline=12., prior=0.),
        outline_active=True, per_observation=[
            dict(loss_family='outline', direction='observed_to_model', normalizer=2.,
                 sample_xy=[[0, 2], [4, 2]], arc_or_sample_weights=[1., 1.],
                 weighted_sample_contributions=[1., 3.], weighted_contribution=4.),
            dict(loss_family='outline', direction='model_to_observed', normalizer=4.,
                 sample_xy=[[1, 2], [3, 2]], residual_xy_px=[[1., 0.], [-1., 0.]],
                 arc_or_sample_weights=[1., 3.], weighted_sample_contributions=[2., 6.],
                 weighted_contribution=8.)])
    return candidate, dense, outline, lineage


def test_support_and_mass_control_have_same_mass_but_different_loss():
    candidate, dense, outline, lineage = cost_fixture()
    original = deepcopy((candidate, dense, outline, lineage))
    result = evaluate_outline(candidate, dense, outline, lineage, [dict(censor=True), dict(censor=False)])
    assert result['support_outline_cost'] == 9.
    assert result['mass_control_outline_cost'] == 8.
    assert result['scores'] == dict(BASELINE_E=20., SUPPORT_AWARE_RERANK=17., MASS_MATCHED_CONTROL=16.)
    assert [d['retained_mass_ratio'] for d in result['directions']] == [.5, .75]
    assert all(d['original_normalizer_preserved'] for d in result['directions'])
    assert (candidate, dense, outline, lineage) == original
    # Missing inputs copy the exact original scalar instead of replacing a
    # nearly equal recomputed sum and perturbing exact comparator ties.
    identity = evaluate_outline(candidate, dense, outline, lineage, [dict(censor=False), dict(censor=False)])
    assert identity['identity_shortcut']
    assert set(identity['scores'].values()) == {20.}


class ForbiddenDiagnosticFields(dict):
    """A poison field must never be consulted by the original comparator path."""
    def __getitem__(self, key):
        if key in DENIED_KEYS:
            raise AssertionError('A reference-only diagnostic was accessed')
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in DENIED_KEYS:
            raise AssertionError('A reference-only diagnostic was accessed')
        return super().get(key, default)


def selection_fixture():
    pose = dict(R_cf=np.eye(3).tolist(), centroid=[0., 0., 2.], R_physical=np.eye(3).tolist(),
                physical_registration_Q=np.eye(3).tolist(), cf_extents=[1., 1., 1.],
                available=True, joint_feasible=True, fallback=False)
    frame = ForbiddenDiagnosticFields(id='api_fixture', stages={})
    candidates, results = [], {}
    for stage in ('G10', 'G13'):
        frame['stages'][stage] = dict(families=[])
        for family in ('ORIGINAL_WD', 'SWAPPED_WD'):
            frame['stages'][stage]['families'].append(dict(family=family, accepted=True,
                published_actual_pose=pose, status='fixture'))
            for index, name in enumerate(('z', 'a')):
                cid = stage+'/'+family+'/'+name
                c = ForbiddenDiagnosticFields(id='api_fixture', stage=stage, family=family, candidate_id=cid,
                    solver_schema='statewise_global_outline_solver_v4_10', retained=True,
                    original_outcome_order=index, original_record=dict(objective=1., initialization=name),
                    pose=dict(R_cf=pose['R_cf'], R_physical=pose['R_physical'], centroid=pose['centroid'],
                              native_t=pose['centroid']),
                    R=pose['R_cf'], t=pose['centroid'], native_xyz=[[0., 0., 0.]],
                    physical_registration_Q=pose['physical_registration_Q'], cf_extents=pose['cf_extents'],
                    dense=dict(original_joint_feasible=True, recomputed_joint_feasible=True, minimum_constraint_slack=0.))
                candidates.append(c)
                results[(stage, family, cid)] = dict(scores=dict(BASELINE_E=1., SUPPORT_AWARE_RERANK=1., MASS_MATCHED_CONTROL=1.))
    return frame, candidates, results


def test_gt_sentinel_ties_and_original_hierarchy():
    frame, candidates, results = selection_fixture()
    value = choose(frame, candidates, results, 'BASELINE_E')
    assert value['selected_family'] == 'ORIGINAL_WD'  # family exact tie
    assert value['candidate_id'] == 'G13/ORIGINAL_WD/a'  # initialization tie-break
    # G10 selects SWAPPED, while the other family's G13 values are much lower.
    # The hierarchy must nevertheless read G13 in the G10-selected family.
    for key in results:
        if key[:2] == ('G10', 'SWAPPED_WD'):
            results[key]['scores']['SUPPORT_AWARE_RERANK'] = .5
        if key[:2] == ('G13', 'ORIGINAL_WD'):
            results[key]['scores']['SUPPORT_AWARE_RERANK'] = -100.
    value = choose(frame, candidates, results, 'SUPPORT_AWARE_RERANK')
    assert value['selected_family'] == 'SWAPPED_WD'
    assert value['candidate_id'] == 'G13/SWAPPED_WD/a'
    assert value['compares_G10_with_G13_objectives'] is False


def test_poisoned_neutral_export_is_rejected_not_silently_read():
    with pytest.raises(ValueError, match='Non-neutral field'):
        _deny_fields(dict(candidate=dict(reference_success=True)))


def test_fallback_is_original_own_family_not_a_fake_candidate():
    frame, candidates, results = selection_fixture()
    for stage in ('G10', 'G13'):
        for meta in frame['stages'][stage]['families']:
            meta['accepted'] = False
            meta['published_actual_pose']['fallback'] = True
            meta['published_actual_pose']['joint_feasible'] = False
    value = choose(frame, [], {}, 'MASS_MATCHED_CONTROL')
    assert value['selected_family'] == 'ORIGINAL_WD'
    assert value['candidate_id'] is None and value['fallback'] is True
    assert value['original_constraint_pass'] is None
    assert value['pose']['fallback'] is True


def test_process_read_boundary_denies_reference_and_unbound_file(tmp_path):
    package = tmp_path/'neutral.json'
    package.write_text('{}')
    reference = tmp_path/'FIXED_REFERENCES.json'
    reference.write_text('{"must_not_read":true}')
    unknown = tmp_path/'unbound.json'
    unknown.write_text('{}')
    guard = ReferenceFreeIO(package, None, tmp_path/'new_output')
    guard.allow(reference)  # Explicit forbidden artifact beats a corrupt binding.
    with guard:
        assert package.read_text() == '{}'
        with pytest.raises(PermissionError, match='forbidden_scientific_artifact'):
            reference.read_text()
        with pytest.raises(PermissionError, match='unbound_file'):
            unknown.read_text()
        with pytest.raises(PermissionError, match='network_or_subprocess'):
            subprocess.run([sys.executable, '-c', 'raise SystemExit("must never start")'], check=True)
    assert len(guard.receipt()['denied_events']) == 3
