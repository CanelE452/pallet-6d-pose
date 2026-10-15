"""Analytic row-coordinate physical basis checks; no dataset labels needed."""
from pathlib import Path
import runpy

import cv2
import numpy as np

from . import common as C


def test_phase_compatibility_handles_general_proper_physical_basis():
    compatible = runpy.run_path(str(C.DOC / "nonsevere_v1/anchored_completion_v1/experiment.py"))["compatible"]
    Q = cv2.Rodrigues(np.array([.41, -.22, .31]))[0]
    S = cv2.Rodrigues(np.array([0., np.pi/2, 0.]))[0]
    X = C.cuboid(1.1, .15, 1.3)
    target_X = C.cuboid(1.3, .15, 1.1)
    mapping = np.linalg.norm((X @ S)[:, None]-target_X[None], axis=-1).argmin(1)
    target_Q = S.T @ Q
    initial = dict(R_cf=np.eye(3).tolist(), R_physical=Q.tolist(), cf_extents=[1.1, .15, 1.3])
    truth = dict(body_R=S.tolist(), R=Q.tolist(), body_xyz=[1.3, .15, 1.1], order=2)
    np.testing.assert_allclose(X @ Q, target_X[mapping] @ target_Q, atol=1e-12)
    assert compatible(initial, truth, mapping)
    # Inverting both transforms incorrectly does not commute with the signed
    # cuboid basis permutation when the physical basis is a general rotation.
    assert np.max(np.linalg.norm(X @ Q.T-target_X[mapping] @ target_Q.T, axis=1)) > .1


def test_phase_compatibility_rejects_different_registered_shape():
    compatible = runpy.run_path(str(C.DOC / "nonsevere_v1/anchored_completion_v1/experiment.py"))["compatible"]
    initial = dict(R_cf=np.eye(3).tolist(), R_physical=np.eye(3).tolist(), cf_extents=[1.1, .15, 1.3])
    truth = dict(body_R=np.eye(3).tolist(), R=np.eye(3).tolist(), body_xyz=[1.8, .17, .6], order=2)
    assert not compatible(initial, truth, np.arange(8))
