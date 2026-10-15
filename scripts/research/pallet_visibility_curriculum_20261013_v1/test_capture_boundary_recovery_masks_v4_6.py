"""Small semantic contracts for box guidance and conservative mask cleanup."""
from copy import deepcopy

import numpy as np
import pytest

from . import common as C
from . import capture_boundary_recovery_masks_v4_6 as M


def geometry():
    xyz = C.cuboid(1., .2, 1.)
    R = np.eye(3)
    t = [0., 0., 3.]
    K = [[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]]
    return dict(native_xyz=xyz.tolist(),R_cf=R.tolist(),centroid=t,K=K,
        initial_projection_xy=C.project(xyz,R,t,K).tolist(),
        initial_self_hidden=[False]*7+[True],image_hw=[100,100])


class PoisonS:
    def __getitem__(self, k):
        if k == 7:
            raise AssertionError("S raw coordinate was accessed")
        return [48.+k, 49.]

    def __array__(self, *args, **kwargs):
        raise AssertionError("Raw coordinates were converted before S gating")


def test_full_geometry_box_can_use_hidden_model_projection_without_S_raw_access():
    g = geometry()
    p = M.build_geometry_prompt(g,PoisonS(),["B"]*7+["S"])
    assert p["S_raw_coordinate_reads"] == 0 and p["prompt_native_ids"] == list(range(7))
    assert p["geometry_projection_native_ids"] == list(range(8))
    lo,hi = np.asarray(g["initial_projection_xy"]).min(0),np.asarray(g["initial_projection_xy"]).max(0)
    np.testing.assert_allclose(p["prompt_box_xyxy"],np.r_[lo-19,hi+19])
    assert not p["projection_used_as_measurement"] and not p["point_prompts_used"]


def test_outside_missing_queries_do_not_make_or_clip_point_prompts():
    g = geometry()
    q = [[-1.,-1.],[150.,40.],[10.,20.],[np.nan,40.],[45.,55.],[30.,45.],[40.,50.],[1e8,1e8]]
    before = deepcopy(g)
    p = M.build_geometry_prompt(g,q,["U","U","B","U","O","B","I","S"])
    assert p["prompt_native_ids"] == [2,4,5,6]
    assert p["box_was_canvas_clipped"] and p["prompt_box_xyxy"][0] == 0.
    assert g == before and q[7] == [1e8,1e8]


def test_cleaning_removes_small_components_preserves_large_detached_parts_and_holes():
    mask = np.zeros((80,100),np.float32)
    mask[5:35,5:35] = 1  # largest area 899; hole is kept.
    mask[10,10] = 0
    mask[50:55,50:55] = 1  # Real separated 25px part survives relative1%.
    mask[70,80] = 1
    before = mask.copy()
    clean,audit = M.clean_boundary_components(mask)
    np.testing.assert_array_equal(mask,before)
    assert clean[70,80] == 0 and clean[52,52] == 1 and clean[10,10] == 0
    assert audit["clean_component_count"] == 2 and audit["removed_foreground_pixels"] == 1
    assert np.all(clean <= mask) and not audit["holes_filled"]


def test_empty_mask_abstains_without_synthetic_foreground():
    clean,audit = M.clean_boundary_components(np.zeros((10,10),np.uint8))
    assert clean.sum() == 0 and audit["clean_component_count"] == 0


def test_unknown_configuration_and_mismatched_projection_reject():
    with pytest.raises(ValueError,match="Unknown"):
        M.configuration({"target_tuned_padding":30.})
    g = geometry()
    g["initial_projection_xy"][0][0] += .1
    with pytest.raises(ValueError,match="binding mismatch"):
        M.build_geometry_prompt(g,PoisonS(),["B"]*7+["S"])


def test_cleanup_validates_mask_range():
    with pytest.raises(ValueError,match="mask required"):
        M.clean_boundary_components(np.array([[0.,np.nan]]))
