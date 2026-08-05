"""Three decoders over one model output.

Every recent eval56/wood number came from D0, the mechanism decoder: raw
argmax, `peak >= 0.30`, a 7x7 local softargmax, nine indexed points.  That path
has no Gaussian, no NMS, no 11x11 weighted centroid, no +0.4395 offset and no
affinity grouping.  The project's live code has all of them.  So "corner
improvements do not reach the pose" has only ever been tested on a decoder the
forklift never runs.

This module exposes the same cached belief/affinity to three paths:

  P0  the D0 mechanism path, unchanged, for parity with the existing results
  P1  the real D2 coordinate extractor called directly, then the same canonical
      PnP wrapper P0 uses -- so P0 vs P1 isolates coordinate extraction
  P2  the deployment path: ObjectDetector.find_object_poses and
      CuboidPNPSolver, imported and called as they are, with only a wrapper
      that hands them cached tensors instead of running the network

Nothing here reimplements a decoder.  P1 calls
`filter_pr_camfacing.extract_keypoints_from_belief`; P2 calls
`detector.ObjectDetector.find_object_poses`.  The wrapper's whole job is
tensors in, results out.

Two conventions have to be reconciled for P2, and both are constants of the
deployment code rather than choices made here:

* `Cuboid3d` numbers its corners as the yaw-180 partner of the camera-facing
  0123 convention the network was trained on.  `M_cuboid = M_camfacing @ Ry^T`
  with `Ry = diag(-1, 1, -1)` holds to 2.8e-17, and the index map is exactly
  the `swap_map = [5, 4, 7, 6, 1, 0, 3, 2, 8]` that `run_live.enforce_camera_facing`
  already applies.  A pose solved against the Cuboid3d model is therefore the
  camera-facing pose right-multiplied by Ry; `to_camfacing_pose` undoes that so
  every path reports in one frame.  run_live only swaps the displayed indices
  and leaves R alone -- its own comment flags the mapping as a TODO.
* the live loop feeds an aspect-preserving resize to height 400 and scales K by
  the same factor, while this checkpoint is a squash model and every evaluation
  in this programme squashes to 400x400.  Holding the squash fixed and swapping
  only the decoder is the point of the audit, so P2 receives the squash
  tensors and a squash-space K.  That K is exact, not an approximation:
  scaling an image by (400/W, 400/H) scales fx, cx by 400/W and fy, cy by
  400/H, which leaves the recovered pose unchanged.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _path in (ROOT / "Deep_Object_Pose/common", ROOT / "challenge/scripts",
              ROOT / "scripts/data_prep/eval"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cuboid import Cuboid3d                                    # noqa: E402
from cuboid_pnp_solver import CuboidPNPSolver                  # noqa: E402
from detector import ObjectDetector                            # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
import run_live_gates as GATES                                 # noqa: E402

INPUT_SIZE = 400
BELIEF_SIZE = 50
SCALE_FACTOR = INPUT_SIZE // BELIEF_SIZE
# run_live.enforce_camera_facing:78 -- the Cuboid3d order is the yaw-180 partner
CUBOID_SWAP_MAP = (5, 4, 7, 6, 1, 0, 3, 2, 8)
RY180 = np.diag([-1.0, 1.0, -1.0])


class DeploymentConfig:
    """challenge/config/task.yaml inference.belief, read not invented."""

    def __init__(self, belief: dict) -> None:
        self.threshold = float(belief["threshold"])
        self.thresh_map = float(belief["thresh_map"])
        self.thresh_points = float(belief["thresh_points"])
        self.thresh_angle = float(belief["thresh_angle"])
        self.sigma = int(belief["sigma"])
        self.mask_edges = 1
        self.mask_faces = 1
        self.vertex = 1
        self.softmax = 1000


def squash_intrinsics(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """K for the squashed frame.  Exact: pose is invariant under this pair."""
    scale = np.diag([INPUT_SIZE / float(width), INPUT_SIZE / float(height), 1.0])
    return scale @ np.asarray(K, dtype=np.float64)


def to_camfacing_pose(rotation: np.ndarray) -> np.ndarray:
    """Cuboid3d-frame rotation -> camera-facing-frame rotation."""
    return np.asarray(rotation, dtype=np.float64) @ RY180


# ---------------------------------------------------------------------------
# P1 -- coordinate extraction only
# ---------------------------------------------------------------------------
def decode_p1(belief: np.ndarray, scale_x: float, scale_y: float):
    """The repository's own D2 extractor, called directly.

    Returns nine image-space points (None where the channel was rejected) and
    the raw peak the extractor reports, so the caller can book-keep membership
    exactly as it does for P0.
    """
    keypoints = extract_keypoints_from_belief(belief)
    points, peaks = [], []
    for x, y, peak in keypoints:
        peaks.append(float(peak))
        points.append(None if (x < 0 and y < 0)
                      else [float(x) * scale_x, float(y) * scale_y])
    return points, peaks


# ---------------------------------------------------------------------------
# P2 -- deployment path
# ---------------------------------------------------------------------------
def build_solver(dims_m, K: np.ndarray, width: int, height: int):
    """CuboidPNPSolver exactly as run_live builds it.

    Dimensions go in centimetres because the deployment gate reads
    `z_m = location[2] / 100.0` (run_live_gates.py:60); handing it metres would
    silently fail every z check.
    """
    w, d, h = (float(v) for v in dims_m)
    solver = CuboidPNPSolver("pallet",
                             cuboid3d=Cuboid3d([w * 100.0, h * 100.0, d * 100.0]))
    solver.set_camera_intrinsic_matrix(squash_intrinsics(K, width, height))
    solver.set_dist_coeffs(np.zeros((4, 1), dtype=np.float64))
    return solver


def run_p2(belief: np.ndarray, affinity: np.ndarray, dims_m, K: np.ndarray,
           width: int, height: int, config: DeploymentConfig):
    """Hand cached tensors to the deployment decoder and return its objects."""
    vertex2 = torch.from_numpy(np.asarray(belief, dtype=np.float32))
    aff = torch.from_numpy(np.asarray(affinity, dtype=np.float32))
    solver = build_solver(dims_m, K, width, height)
    results = ObjectDetector.find_object_poses(vertex2, aff, solver, config,
                                               scale_factor=SCALE_FACTOR)
    return results, solver


def enforce_camera_facing(result, solver):
    """run_live's index convention step, applied through the same swap map.

    Imported behaviour would be preferable, but run_live.py opens a GUI at
    import time, so the swap is reproduced here from run_live.py:49-89 and
    pinned by a test that reads that function's source.
    """
    from pyrr import Quaternion, matrix33

    location = result.get("location")
    quaternion = result.get("quaternion")
    if location is None or quaternion is None or solver._cuboid3d is None:
        return result
    rotation = matrix33.create_from_quaternion(Quaternion(quaternion))
    translation = np.asarray(location, dtype=np.float64)
    vertices = np.asarray(solver._cuboid3d.get_vertices()[:8], dtype=np.float64)
    z_cam = ((rotation @ vertices.T).T + translation)[:, 2]
    if z_cam[:4].mean() <= z_cam[4:].mean():
        return result
    swapped = dict(result)
    for key in ("raw_points", "projected_points"):
        value = result.get(key)
        if value is None:
            continue
        swapped[key] = [value[CUBOID_SWAP_MAP[i]]
                        if CUBOID_SWAP_MAP[i] < len(value) else None
                        for i in range(9)]
    return swapped


def production_selection(results, gates: dict, K_proc: np.ndarray, solver):
    """run_live.py:440-455 -- first hypothesis clearing every gate wins.

    No GT is consulted.  Depth is unavailable offline, so the depth agreement
    check is skipped exactly as it is when run_live has no depth frame.
    """
    reason = "no_result"
    for index, result in enumerate(results):
        candidate = enforce_camera_facing(result, solver)
        ok, reason, info = GATES.evaluate_result(candidate, gates, None, K_proc)
        if ok:
            return index, candidate, info, "ok"
    return None, None, {}, reason


def config_with_sigma(base: "DeploymentConfig", sigma: float) -> "DeploymentConfig":
    """The one field this calibration may vary.

    `find_objects` reads `config.sigma` and passes it straight to
    `gaussian_filter` (detector.py:684).  Every other field -- thresh_map,
    thresh_points, thresh_angle, threshold -- is copied unchanged, so no
    threshold moves and the decoder code is untouched.
    """
    clone = DeploymentConfig.__new__(DeploymentConfig)
    clone.__dict__.update(base.__dict__)
    clone.sigma = float(sigma)
    return clone
