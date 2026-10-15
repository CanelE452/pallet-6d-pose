"""Synthetic global-grid/provenance/privacy contracts; no actual image/model."""
from copy import deepcopy
import inspect
from pathlib import Path

import numpy as np
import pytest

from . import capture_class_agnostic_AMG_case_v1 as M


def fixture():
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    image[:, :, 0] = 7
    image[:, :, 2] = 19
    target = np.zeros((16, 32), dtype=np.uint8)
    target[5:10, 5:10] = 1
    mask = np.zeros_like(target)
    mask[5:10, 5:10] = 1
    ann = dict(segmentation=mask, area=25, bbox=[5., 5., 5., 5.], predicted_iou=.92,
        point_coords=[M.global_grid(target.shape)[0].tolist()], stability_score=.97,
        crop_box=[0., 0., 32., 16.])
    return image, target, ann


class Generator:
    def __init__(self, annotations):
        self.annotations = annotations
        self.calls = []

    def generate(self, rgb):
        self.calls.append(rgb.copy())
        rgb[:] = 0  # Only the independent copy can be changed.
        return deepcopy(self.annotations)


def invoke(annotations):
    image, target, _ = fixture()
    saved = []
    def save(index, mask):
        saved.append((index, mask.copy()))
        mask[:] = 0  # A storage sink cannot change the record's original mask.
        return dict(path=f'synthetic/{index}.npz', sha256='a' * 64)
    generator = Generator(annotations)
    before = image.copy(), target.copy()
    result = M.infer_case(image, target, generator, save_mask=save)
    assert np.array_equal(image, before[0]) and np.array_equal(target, before[1])
    assert len(generator.calls) == 1
    assert np.array_equal(generator.calls[0], before[0][:, :, ::-1])
    return result, saved


def test_only_budget_change_from_source_verified_vendor_defaults():
    defaults = M.verify_vendor_defaults()
    assert defaults == {**M.CONFIGURATION, 'points_per_side': 32}
    assert M.CONFIGURATION['points_per_side'] == 16 and M.CONFIGURATION['crop_n_layers'] == 0
    assert M.CONFIGURATION['points_per_batch'] == 64
    assert M.CONFIGURATION['min_mask_region_area'] == 0


def test_256_global_cell_centered_points_do_not_read_local_metadata():
    grid = M.global_grid((480, 640))
    assert grid.shape == (256, 2)
    assert np.array_equal(grid[0], [20., 15.])
    assert np.array_equal(grid[-1], [620., 465.])
    assert len(np.unique(grid[:, 0])) == len(np.unique(grid[:, 1])) == 16
    assert np.array_equal(M.global_grid((16, 32))[0], [1., .5])


def test_generator_only_receives_full_RGB_and_target_overlap_is_postprediction():
    _, _, ann = fixture()
    result, saved = invoke([ann])
    item = result['candidates'][0]
    assert result['returned_candidates'] == 1 and result['global_grid_prompts'] == 256
    assert item['target_overlap']['IoU'] == 1 and item['status'] == 'suspected_target'
    assert item['target_identity'] == 'UNKNOWN'
    assert item['mask_pixel_sha256'] == M.P.mask_pixel_sha256(saved[0][1])
    assert not item['other_object_confirmed'] and not item['foreground_occlusion_confirmed']
    assert result['current_pipeline_mask_supply'] == result['KP_S_U_O_local_prompt_queries'] == 0


def test_vendor_order_and_all_returned_masks_preserved_without_quality_winner():
    _, _, ann = fixture()
    other = deepcopy(ann)
    other['predicted_iou'] = .99
    other['segmentation'] = np.roll(ann['segmentation'], 15, axis=1)
    result, saved = invoke([ann, other])
    assert [x['candidate_index'] for x in result['candidates']] == [0, 1]
    assert [x['predicted_iou'] for x in result['candidates']] == [.92, .99]
    assert [x['status'] for x in result['candidates']] == ['suspected_target', 'CANDIDATE']
    assert len(saved) == 2 and result['all_returned_masks_preserved']
    assert all(not x['scores_are_semantic_or_depth_probabilities'] for x in result['candidates'])


def test_no_returned_mask_is_explicit_uniform_unavailable_row():
    result, saved = invoke([])
    assert result['status'] == 'UNAVAILABLE' and result['returned_candidates'] == 0
    assert result['unavailable_reason'] == 'no_masks_survived_fixed_vendor_filters'
    assert saved == [] and result['other_object_or_depth_certificates'] == 0


@pytest.mark.parametrize('change', ['offgrid', 'nonfinite_score', 'crop', 'area', 'canvas', 'nonbinary'])
def test_actual_grid_crop_pixels_and_area_not_fabricated_or_resized(change):
    _, _, ann = fixture()
    if change == 'offgrid':
        ann['point_coords'] = [[0., 0.]]
    elif change == 'nonfinite_score':
        ann['stability_score'] = np.nan
    elif change == 'crop':
        ann['crop_box'] = [0., 0., 16., 16.]
    elif change == 'area':
        ann['area'] = 24
    elif change == 'canvas':
        ann['segmentation'] = np.zeros((8, 16))
    elif change == 'nonbinary':
        ann['segmentation'] = ann['segmentation'].astype(float) * .5
    with pytest.raises(ValueError):
        invoke([ann])


def test_zero_target_does_not_turn_complement_into_other_object():
    image, target, ann = fixture()
    result = M.infer_case(image, target * 0, Generator([ann]), save_mask=lambda *a: {})
    assert result['candidates'][0]['status'] == 'UNKNOWN'
    assert not result['target_complement_used_as_occluder']
    assert result['other_object_or_depth_certificates'] == 0


@pytest.mark.parametrize('kind', ['image_type', 'target_canvas', 'target_nan'])
def test_invalid_inputs_fail_before_generator(kind):
    image, target, _ = fixture()
    if kind == 'image_type': image = image.astype(float)
    if kind == 'target_canvas': target = target[:8]
    if kind == 'target_nan': target = target.astype(float); target[0, 0] = np.nan
    generator = Generator([])
    with pytest.raises(ValueError):
        M.infer_case(image, target, generator, save_mask=lambda *a: {})
    assert generator.calls == []


def test_actual_and_metadata_GOs_precede_any_IO(monkeypatch):
    def poison(*args):
        raise AssertionError('IO before GO')
    monkeypatch.setattr(M.C, 'sha', poison)
    with pytest.raises(PermissionError):
        M.run(expected_plan_sha256='a' * 64)
    with pytest.raises(PermissionError):
        M.prepare()


def test_no_state_pose_box_or_precise_point_prompt_input_in_pure_API():
    assert tuple(inspect.signature(M.infer_case).parameters) == ('image_bgr', 'target_mask', 'generator', 'save_mask')
    source = inspect.getsource(M.load_generator)
    assert "sam_model_registry['vit_t']" in source
    assert "device='cpu'" in source and 'download' not in source.replace('no downloads', '')
    assert M.CONFIGURATION['point_grids'] is None


def test_parent_configuration_and_source_globals_unchanged():
    original = dict(vars(M.P))
    parent_source = Path(M.P.__file__).read_bytes()
    _, _, ann = fixture()
    invoke([ann])
    assert all(vars(M.P)[key] is value for key, value in original.items())
    assert Path(M.P.__file__).read_bytes() == parent_source
