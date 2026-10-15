"""Integration checks for explicit registered model basis and fallback readout."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from .common import cuboid, project
from .global_pallet_silhouette_v4_9 import observe_mask_outline
from .evaluate_outline_wd_candidates_v4_9 import select_explicit_candidates


def fixture():
    x = cuboid(1.3, .11, 1.1)
    swapped = cuboid(1.1, .11, 1.3)
    K = np.array([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]])
    a = .6
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Q = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
    t = np.array([0., .3, 2.])
    g = dict(native_xyz=x, cf_extents=[1.3, .11, 1.1], physical_registration_Q=np.eye(3), K=K)
    sg = dict(native_xyz=swapped, cf_extents=[1.1, .11, 1.3], physical_registration_Q=Q)
    def result(geometry, translation, accepted=True):
        return dict(accepted=accepted, actual_pose=dict(R_cf=R,
            centroid=translation, R_physical=R @ np.asarray(geometry['physical_registration_Q']),
            cf_extents=geometry['cf_extents']))
    prior = result(g, t + [.1, 0, 0])
    original = result(g, t + [0, 0, .2])
    swap = result(sg, t)
    mask = np.zeros((480, 640), np.uint8)
    xy = project(swapped, R, t, K)
    cv2.fillConvexPoly(mask, np.rint(cv2.convexHull(xy.astype(np.float32))).astype(np.int32), 1)
    return observe_mask_outline(mask, mask.shape), g, prior, original, sg, swap


def test_selected_pose_retains_swapped_vertices_and_physical_rotation():
    args = fixture()
    out = select_explicit_candidates(*args, original_no_prior_admissible=False)
    assert out['selected_arm'] == 'SWAPPED_WD_NO_PRIOR'
    np.testing.assert_equal(out['selected_geometry']['native_xyz'], args[4]['native_xyz'])
    np.testing.assert_allclose(out['selected_result']['actual_pose']['R_physical'],
        args[5]['actual_pose']['R_physical'])
    assert {c['name'] for c in out['outline_selection']['candidates']} == {'INITIAL_REGULARIZED', 'SWAPPED_WD_NO_PRIOR'}


def test_registration_mismatch_rejects_before_selecting_a_wrong_readout():
    args = list(fixture())
    args[-1] = deepcopy(args[-1])
    args[-1]['actual_pose']['R_physical'] = args[-1]['actual_pose']['R_cf']
    with pytest.raises(ValueError, match='registration'):
        select_explicit_candidates(*args, original_no_prior_admissible=True)


def test_no_accepted_evidence_retains_explicit_original_fallback():
    args = list(fixture())
    for i in (2, 3, 5):
        args[i]['accepted'] = False
    out = select_explicit_candidates(*args, original_no_prior_admissible=True)
    assert out['explicit_original_fallback']
    assert out['selected_arm'] == 'INITIAL_REGULARIZED'
    assert out['selected_result']['accepted'] is False
    np.testing.assert_equal(out['selected_geometry']['native_xyz'], args[1]['native_xyz'])
