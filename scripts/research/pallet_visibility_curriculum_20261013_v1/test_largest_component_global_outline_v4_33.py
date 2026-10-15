"""Toy masks only: no saved images, masks, rows, models or references."""
from copy import deepcopy
import ast
import inspect
import hashlib

import cv2
import numpy as np
import pytest

from . import largest_component_global_outline_v4_33 as M


def disconnected():
    mask = np.zeros((64, 72), dtype=np.float32)
    mask[30:42, 25:45] = 1.
    mask[10:12, 15:18] = 1.
    return mask


def test_small_disconnected_component_is_removed_before_global_envelope():
    mask = disconnected()
    result = M.observe_union_and_largest_component(mask, mask.shape)
    union, lcc = result['original_union'], result['largest_component']
    assert result['row_retained'] and not result['observation_chosen_for_runtime']
    assert union['mask_area_px'] == 246
    assert lcc['mask_area_px'] == 240
    assert min(np.asarray(union['polygon_xy'])[:, 1]) == 10
    assert min(np.asarray(lcc['polygon_xy'])[:, 1]) == 30
    assert lcc['foreground_hull_area_px'] == 209.
    assert len(lcc['observed_boundary_samples_xy']) == 60
    assert lcc['foreground_component_selection']['discarded_foreground_area_px'] == 6


def test_diagonally_touching_pixels_are_one_eight_connected_component():
    mask = np.zeros((24, 24), np.float32)
    mask[8, 8] = mask[9, 9] = mask[10, 10] = 1.
    mask[17, 17] = mask[17, 18] = 1.
    selected, audit = M.largest_component_mask(mask, mask.shape)
    assert audit['foreground_component_count'] == 2
    assert selected.sum() == 3
    assert selected[8, 8] == selected[9, 9] == selected[10, 10] == 1.


def test_exact_area_tie_uses_smallest_returned_foreground_label():
    mask = np.zeros((32, 32), np.float32)
    mask[8:11, 8:12] = mask[20:23, 19:23] = 1.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask >= .5).astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    candidates = [k for k in range(1, len(stats)) if stats[k, cv2.CC_STAT_AREA] == 12]
    selected, audit = M.largest_component_mask(mask, mask.shape)
    assert audit['selected_label'] == min(candidates)
    assert audit['maximum_area_tied_labels'] == candidates
    np.testing.assert_array_equal(selected, (labels == min(candidates)).astype(np.float32))


def test_single_component_and_holes_are_preserved_without_morphology():
    mask = np.zeros((40, 40), np.float32)
    mask[10:29, 10:29] = 1.
    mask[15:24, 15:24] = 0.
    selected, audit = M.largest_component_mask(mask, mask.shape)
    np.testing.assert_array_equal(selected, mask)
    assert selected[18, 18] == 0
    assert audit['discarded_foreground_area_px'] == 0
    assert not audit['holes_filled']
    # The unchanged V13 observer still uses an envelope; LCC does not make it a hole model.
    direct = M.FROZEN_OUTLINE.observe_global_outline(mask, mask.shape)
    actual = M.observe_largest_component_outline(mask, mask.shape)
    for key in direct:
        assert actual[key] == direct[key]


def test_empty_mask_retains_two_unsupported_observations():
    mask = np.zeros((24, 24), np.float32)
    result = M.observe_union_and_largest_component(mask, mask.shape)
    assert result['row_retained']
    assert not result['original_union']['supported']
    assert not result['largest_component']['supported']
    assert result['largest_component']['foreground_component_selection']['selected_label'] is None
    assert result['largest_component']['foreground_component_selection']['retained_foreground_area_px'] == 0


def test_threshold_membership_is_fixed_to_frozen_half_and_not_tuned():
    mask = np.zeros((24, 24), np.float64)
    mask[8:11, 8:11] = .5
    mask[14:20, 14:20] = np.nextafter(.5, 0.)
    selected, audit = M.largest_component_mask(mask, mask.shape)
    assert selected.sum() == 9
    assert audit['original_foreground_area_px'] == 9
    assert audit['mask_threshold'] == .5


def test_input_readonly_array_and_binding_configuration_are_unchanged():
    mask = disconnected(); before = mask.tobytes(); mask.setflags(write=False)
    binding = {'image_sha256': 'a'*64, 'mask_file_sha256': 'b'*64}
    cfg = {'observed_sample_spacing_px': 2.}
    saved_binding, saved_cfg = deepcopy(binding), deepcopy(cfg)
    M.observe_union_and_largest_component(mask, mask.shape, image_binding=binding, configuration=cfg)
    assert mask.tobytes() == before
    assert binding == saved_binding and cfg == saved_cfg


def test_variant_sampler_result_exactly_matches_frozen_observer_of_selected_mask():
    mask = disconnected(); selected, _ = M.largest_component_mask(mask, mask.shape)
    binding = {'source': 'synthetic'}
    configuration = {'observed_sample_spacing_px': 1.7, 'frame_margin_px': 3.}
    expected = M.FROZEN_OUTLINE.observe_global_outline(selected, mask.shape,
        image_binding=binding, configuration=configuration)
    actual = M.observe_largest_component_outline(mask, mask.shape,
        image_binding=binding, configuration=configuration)
    assert {key: value for key, value in actual.items()
            if key not in ('foreground_component_selection', 'selection_profile')} == expected


def test_union_baseline_is_exact_frozen_call_not_a_cleaned_or_lcc_baseline():
    mask = disconnected()
    expected = M.FROZEN_OUTLINE.observe_global_outline(mask, mask.shape)
    actual = M.observe_union_and_largest_component(mask, mask.shape)['original_union']
    assert {key: value for key, value in actual.items() if key != 'selection_profile'} == expected


def test_sampler_and_C_kernel_are_exact_function_objects_with_unchanged_AST():
    assert M.frozen_union_observer is M.FROZEN_OUTLINE.observe_global_outline
    assert M.outline_value_and_gradient is M.FROZEN_CONTAINMENT.outline_value_and_gradient
    assert M.outline_value_and_gradient.__code__ is M.FROZEN_CONTAINMENT.outline_value_and_gradient.__code__
    assert ast.dump(ast.parse(inspect.getsource(M.frozen_union_observer))) == ast.dump(ast.parse(inspect.getsource(M.FROZEN_OUTLINE.observe_global_outline)))


def test_C_function_uses_same_geometry_and_interior_zero_for_toy_observation():
    xyz = np.array([[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]])
    K = np.array([[20.,0.,32.],[0.,20.,32.],[0.,0.,1.]])
    p = np.array([0.,0.,0.,0.,0.,5.])
    observation = {'supported': True, 'observed_boundary_samples_xy': [[32.,32.], [42.,32.]]}
    actual = M.outline_value_and_gradient(p, xyz, K, 1., observation)
    expected = M.FROZEN_CONTAINMENT.outline_value_and_gradient(p, xyz, K, 1., observation)
    assert actual[0] == expected[0] > 0
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]
    assert actual[2]['inside_or_boundary_sample_count'] == 1


def test_uniform_empty_and_fragment_rows_have_no_frame_or_pose_selection():
    masks = [np.zeros((64,72),np.float32), disconnected(), np.ones((64,72),np.float32)]
    results = [M.observe_union_and_largest_component(mask, mask.shape) for mask in masks]
    assert len(results) == 3 and all(row['row_retained'] for row in results)
    assert all(row['new_NN_or_RGB_or_prompt_or_point_queries'] == 0 for row in results)
    assert 'states' not in inspect.signature(M.observe_largest_component_outline).parameters
    assert 'pose' not in inspect.signature(M.observe_largest_component_outline).parameters
    assert 'id' not in inspect.signature(M.observe_largest_component_outline).parameters


@pytest.mark.parametrize('kind', ['nan','negative','above_one','wrong_shape','empty_canvas'])
def test_invalid_masks_fail_explicitly_instead_of_silently_modifying(kind):
    mask = np.zeros((16,16),np.float32); hw = mask.shape
    if kind=='nan': mask[0,0]=np.nan
    if kind=='negative': mask[0,0]=-.1
    if kind=='above_one': mask[0,0]=1.1
    if kind=='wrong_shape': hw=(15,16)
    if kind=='empty_canvas': mask=np.zeros((0,16));hw=mask.shape
    with pytest.raises(ValueError): M.largest_component_mask(mask, hw)


def test_frozen_modules_and_functions_are_not_globally_monkeypatched():
    modules = (M.FROZEN_OUTLINE, M.FROZEN_CONTAINMENT)
    before = {module.__file__: hashlib.sha256(open(module.__file__,'rb').read()).hexdigest() for module in modules}
    observer, kernel = M.FROZEN_OUTLINE.observe_global_outline, M.FROZEN_CONTAINMENT.outline_value_and_gradient
    mask=disconnected();M.observe_union_and_largest_component(mask,mask.shape)
    assert M.FROZEN_OUTLINE.observe_global_outline is observer
    assert M.FROZEN_CONTAINMENT.outline_value_and_gradient is kernel
    assert before == {module.__file__: hashlib.sha256(open(module.__file__,'rb').read()).hexdigest() for module in modules}
