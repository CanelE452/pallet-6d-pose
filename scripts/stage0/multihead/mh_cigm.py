"""Hough lines -> CIGM corners -> PnP, and the same PnP for the corner head.

Section 7 asked for this explicitly: CIGM exists and is differentiable
(`Deep_Object_Pose/common/corner_incident_geometry.py:38`) and its oracle passes
at 0.003 cell, but nothing has ever fed it a *learned* Hough output.  The whole
line stage instead writes `CIGM: BLOCKED` into its verdicts, and five test files
assert that string.  Those runners and tests are left alone -- the adapter lives
here, so the historical record keeps its lock and this screen gets its number.

The 3D model each frame's PnP uses is not guessed.  `objects[0].cuboid` is in
world coordinates and `pose_transform` maps object to camera, so the two are one
fixed signed axis permutation apart; a search over the 24 proper ones found
exactly one that reprojects through the frame's own intrinsics, and it does so to
5.5e-06 px median over 200 frames.  `OBJECT_FROM_WORLD` is that matrix.
"""
from __future__ import annotations

import numpy as np
import torch

import mh_arms as ARMS_MOD
from mh_arms import DH, V2

import corner_incident_geometry as CIGM
import instance_edge_topology as IET

# cuboid(world, centred) -> object frame.  Verified by reprojection, not assumed.
OBJECT_FROM_WORLD = np.array([[1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0],
                              [0.0, -1.0, 0.0]])

EDGES = [tuple(e) for e in IET.build_topology()["edges"]]
INCIDENCE = CIGM.incidence_table()


# --------------------------------------------------------------------------
# the adapter


def lines_to_segments(theta_deg_centred: torch.Tensor,
                      rho_centred: torch.Tensor):
    """(B,12) decoded lattice coordinates -> (centre, direction) in 50-grid.

    `solve_corners` reads its lines through `lines_from_segments`, which takes
    normal = [-dy, dx] and rho = normal . centre.  Choosing
    direction = (sin t, -cos t) makes that normal exactly (cos t, sin t) and that
    rho exactly the canonical rho, so the adapter introduces no sign convention
    of its own -- `test_mh_cigm` pins the round trip at 1e-6.

    A line has no distinguished point, so the centre used is the foot of the
    perpendicular from the origin.  CIGM only ever consumes the line through its
    normal form, so which point is passed cannot affect the intersection.
    """
    theta, rho = DH.canonical_from_centred(theta_deg_centred, rho_centred)
    normal = torch.stack([theta.cos(), theta.sin()], -1)
    direction = torch.stack([theta.sin(), -theta.cos()], -1)
    centre = normal * rho[..., None]
    return centre, direction


def cigm_corners(theta_deg_centred: torch.Tensor, rho_centred: torch.Tensor):
    """(B,12) lines -> (B,8,2) corners in canonical 50-grid, plus diagnostics."""
    centre, direction = lines_to_segments(theta_deg_centred, rho_centred)
    corners, residual, condition = CIGM.solve_corners(centre, direction, INCIDENCE)
    return corners, residual, condition


# --------------------------------------------------------------------------
# PnP


def object_points(label: dict) -> np.ndarray:
    world = np.asarray(label["objects"][0]["cuboid"], float)
    return (OBJECT_FROM_WORLD @ (world - world.mean(0)).T).T


def intrinsics(label: dict) -> np.ndarray:
    k = label["camera_data"]["intrinsics"]
    return np.array([[k["fx"], 0.0, k["cx"]],
                     [0.0, k["fy"], k["cy"]],
                     [0.0, 0.0, 1.0]], float)


def gt_pose(label: dict):
    transform = np.asarray(label["objects"][0]["pose_transform"], float)
    return transform[:3, :3], transform[:3, 3]


def grid_to_pixels(corners: np.ndarray, width: float, height: float,
                   grid: int = 50) -> np.ndarray:
    return np.stack([corners[:, 0] * width / grid,
                     corners[:, 1] * height / grid], 1)


def solve(model: np.ndarray, pixels: np.ndarray, camera: np.ndarray):
    """EPnP then an iterative refinement, matching the project's PnP recipe.

    RANSAC is deliberately not used here.  With eight labelled correspondences and
    no outlier process -- both paths emit exactly eight points, always the same
    eight -- RANSAC would silently drop the corners a method is worst at and turn
    a pose metric into a subset-selection metric that differs between arms.
    """
    import cv2
    if not np.isfinite(pixels).all():
        return None
    ok, rvec, tvec = cv2.solvePnP(model.astype(np.float64),
                                  pixels.astype(np.float64), camera, None,
                                  flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    ok, rvec, tvec = cv2.solvePnP(model.astype(np.float64),
                                  pixels.astype(np.float64), camera, None,
                                  rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation, tvec.reshape(3)


def pose_error(predicted, rotation_gt, translation_gt):
    """Geodesic rotation error in degrees and translation error in metres.

    No symmetry group is quotiented out.  The corner labels are camera-facing, so
    corner 0 is a specific corner in this view and the correspondence pins the
    pose; folding in the cuboid's 180-degree symmetry would hide exactly the
    front/rear confusion this project keeps finding.
    """
    if predicted is None:
        return None
    rotation, translation = predicted
    trace = np.clip((np.trace(rotation.T @ rotation_gt) - 1.0) / 2.0, -1.0, 1.0)
    return (float(np.degrees(np.arccos(trace))),
            float(np.linalg.norm(translation - translation_gt)))


def reprojection(model, predicted, camera, pixels_gt):
    if predicted is None:
        return None
    import cv2
    rotation, translation = predicted
    rvec, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(model.astype(np.float64), rvec,
                                     translation.astype(np.float64), camera, None)
    return float(np.median(np.linalg.norm(
        projected.reshape(-1, 2) - pixels_gt, axis=1)))
