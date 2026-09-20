"""Small image-conditioned utility predictor for frozen N2/Replay candidates.

``features`` and ``UtilitySelector`` cannot access target coordinates.  The
separate ``utility_targets`` function uses synthetic supervision to learn which
candidate lowers error, without inventing corrected coordinates or moving GT.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from scripts.research.pallet_sensors_submission_v1.posefix_contract_math import axis_aligned_crop_matrix


PAIRS = ((0, 3), (1, 2), (4, 7), (5, 6))
PARTNER = (3, 2, 1, 0, 7, 6, 5, 4)
NUMERIC_DIM = 26
GLOBAL_DIM = 64
PATCH_SIDE_FRACTION = .12
NUMERIC_FIELDS = (
    "r0_x", "r0_y", "n2_x", "n2_y", "replay_x", "replay_y",
    "n2_minus_r0_x", "n2_minus_r0_y", "replay_minus_r0_x", "replay_minus_r0_y",
    "replay_minus_n2_x", "replay_minus_n2_y", "mode_minus_replay_x", "mode_minus_replay_y",
    "entropy_normalized", "peak_mass7x7", "peak_probability", "valid",
    *tuple(f"native_{i}" for i in range(8)),
)
GLOBAL_FIELDS = tuple(f"{arm}_p{i}_{axis}" for arm in ("r0", "n2", "replay")
                      for i in range(9) for axis in ("x", "y")) + tuple(f"valid_p{i}" for i in range(9)) + ("log_box_aspect",)


def _points(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (9, 2):
        raise ValueError(f"{name} must have shape (9, 2)")
    return result


def _valid(value, name="valid"):
    result = np.asarray(value, dtype=bool)
    if result.shape != (9,):
        raise ValueError(f"{name} must have shape (9,)")
    return result


def _finite(points):
    return np.isfinite(points).all(-1) & ~(points == -1).all(-1)


def _heat_vector(heatmap, key):
    result = np.asarray(heatmap[key], dtype=np.float64)
    if result.shape != (9,) or not np.isfinite(result).all():
        raise ValueError(f"heatmap.{key} must be finite shape (9,)")
    return result


def _patch_matrix(center, side):
    """Map original-image center exactly to pixel (12, 12) of a 24px patch."""
    scale = 24. / side
    return np.array([[scale, 0., 12. - scale * center[0]],
                     [0., scale, 12. - scale * center[1]]], dtype=np.float64)


def features(bgr, r0, n2, replay, box, valid, heatmap):
    """Extract GT-free visual/candidate features in ONE shared pixel frame.

    Source prepared RGB and all source points may retain the reflected border;
    real RGB uses raw-image pixels.  No padding or hidden +/-100 translation is
    performed here.  The caller must provide matching coordinates for the
    image, all candidates, the predicted box, and heatmap mode positions.
    """
    bgr = np.asarray(bgr)
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8 or min(bgr.shape[:2]) <= 0:
        raise ValueError("bgr must be a nonempty uint8 HWC image")
    r0, n2, replay = (_points(q, name) for q, name in ((r0, "r0"), (n2, "n2"), (replay, "replay")))
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or not (box[2:] > box[:2]).all():
        raise ValueError("box must be a finite, positive-size predicted xyxy box")
    valid = _valid(valid) & _finite(r0) & _finite(n2) & _finite(replay)
    size = box[2:] - box[:2]
    diagonal = float(np.linalg.norm(size))
    center = (box[2:] + box[:2]) * .5
    mode = _points(heatmap["mode_original_xy9"], "heatmap.mode_original_xy9")
    if not np.isfinite(mode).all():
        raise ValueError("Heatmap mode coordinates must be finite")
    entropy = (_heat_vector(heatmap, "entropy_normalized9") if "entropy_normalized9" in heatmap
               else _heat_vector(heatmap, "entropy_nats9") / np.log(96 * 72))
    mass = _heat_vector(heatmap, "peak_mass7x7_9")
    probability = _heat_vector(heatmap, "peak_probability9")
    if np.any(entropy < -1e-6) or np.any(entropy > 1 + 1e-6) or np.any(mass < 0) or np.any(mass > 1 + 1e-6) or np.any(probability < 0) or np.any(probability > 1 + 1e-6):
        raise ValueError("Invalid heatmap uncertainty values")
    rgb_image = np.ascontiguousarray(bgr[:, :, ::-1])
    matrix = axis_aligned_crop_matrix(box, input_shape=(96, 72), expansion=1.25)
    global_rgb = cv2.warpAffine(rgb_image, matrix[:2], (72, 96), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    patches = np.zeros((8, 6, 24, 24), dtype=np.uint8)
    side = PATCH_SIDE_FRACTION * diagonal
    for i in range(8):
        if not valid[i]:
            continue
        for candidate_index, candidate in enumerate((n2, replay)):
            patch = cv2.warpAffine(rgb_image, _patch_matrix(candidate[i], side), (24, 24),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            patches[i, candidate_index * 3:(candidate_index + 1) * 3] = patch.transpose(2, 0, 1)
    normalized = [np.where(valid[:, None], (q - center) / diagonal, 0.) for q in (r0, n2, replay)]
    differences = [normalized[1] - normalized[0], normalized[2] - normalized[0], normalized[2] - normalized[1]]
    mode_delta = np.where(valid[:, None], (mode - replay) / diagonal, 0.)
    numeric = np.concatenate([
        *[q[:8] for q in normalized], *[q[:8] for q in differences], mode_delta[:8],
        entropy[:8, None], mass[:8, None], probability[:8, None], valid[:8, None].astype(float), np.eye(8),
    ], axis=1).astype(np.float32)
    global_numeric = np.concatenate([*[q.reshape(-1) for q in normalized], valid.astype(float),
                                      [np.log(size[0] / size[1])]]).astype(np.float32)
    assert numeric.shape == (8, NUMERIC_DIM) and global_numeric.shape == (GLOBAL_DIM,)
    assert np.isfinite(numeric).all() and np.isfinite(global_numeric).all()
    return dict(rgb=np.ascontiguousarray(global_rgb.transpose(2, 0, 1)),
                patches=np.ascontiguousarray(patches), numeric=numeric, global_numeric=global_numeric)


class UtilitySelector(nn.Module):
    """Shared corner CNN plus partner and full-object context; signed output.

    No sigmoid or post-hoc learned threshold: predicted positive utility means
    expected error reduction in units of one percent of predicted box diagonal.
    """
    def __init__(self, numeric_dim=NUMERIC_DIM, global_dim=GLOBAL_DIM):
        super().__init__()
        self.numeric_dim, self.global_dim = int(numeric_dim), int(global_dim)
        if self.numeric_dim <= 0 or self.global_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        self.local = nn.Sequential(
            nn.Conv2d(6, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.global_image = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.numeric = nn.Sequential(nn.Linear(self.numeric_dim, 32), nn.SiLU(), nn.Linear(32, 24), nn.SiLU())
        self.global_numeric = nn.Sequential(nn.Linear(self.global_dim, 32), nn.SiLU(), nn.Linear(32, 24), nn.SiLU())
        self.predictor = nn.Sequential(nn.Linear(224, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        self.register_buffer("partner", torch.tensor(PARTNER, dtype=torch.long))

    def forward(self, batch):
        rgb, patches = batch["rgb"], batch["patches"]
        numeric, global_numeric = batch["numeric"], batch["global_numeric"]
        b = len(rgb)
        if rgb.shape != (b, 3, 96, 72) or patches.shape != (b, 8, 6, 24, 24):
            raise ValueError("Unexpected RGB or candidate-patch shape")
        if numeric.shape != (b, 8, self.numeric_dim) or global_numeric.shape != (b, self.global_dim):
            raise ValueError("Unexpected numeric feature shape")
        if rgb.dtype != torch.uint8 or patches.dtype != torch.uint8:
            raise ValueError("RGB and patches must be uint8; normalization belongs inside the model")
        if not torch.isfinite(numeric).all() or not torch.isfinite(global_numeric).all():
            raise ValueError("Numeric features must be finite")
        dtype = self.predictor[0].weight.dtype
        local_image = self.local((patches.to(dtype) / 255. - .5).reshape(b * 8, 6, 24, 24)).reshape(b, 8, 32)
        own = torch.cat([local_image, self.numeric(numeric.to(dtype))], -1)
        partner = own[:, self.partner]
        global_features = torch.cat([self.global_image(rgb.to(dtype) / 255. - .5),
                                     self.global_numeric(global_numeric.to(dtype))], -1)
        joined = torch.cat([own, partner, own - partner, global_features[:, None].expand(-1, 8, -1)], -1)
        return self.predictor(joined).squeeze(-1)


def utility_targets(n2, replay, gt, gt_valid, permutations, valid, diagonal):
    """Supervision ONLY: choose one N2 branch and hold it for both candidates.

    GT outside the image/crop remains eligible when its source validity says it
    is valid.  The centroid is never supervised.  Missing N2 predictions receive
    a fixed diagonal penalty during whole-object branch selection, rather than
    quietly changing the GT denominator.
    """
    n2, replay, gt = (_points(q, name) for q, name in ((n2, "n2"), (replay, "replay"), (gt, "gt")))
    gt_valid = _valid(gt_valid, "gt_valid") & _finite(gt)
    valid = _valid(valid)
    gt_valid[8] = False
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("diagonal must be finite and positive")
    permutations = np.asarray(permutations, dtype=np.int64)
    if permutations.ndim != 2 or permutations.shape[1] != 9 or len(permutations) == 0:
        raise ValueError("permutations must be nonempty (groups, 9)")
    if any(sorted(p.tolist()) != list(range(9)) or p[8] != 8 for p in permutations):
        raise ValueError("Each approved whole-object permutation must fix centroid 8")
    n2_valid = valid & _finite(n2)
    both_valid = n2_valid & _finite(replay)
    costs = []
    aligned_targets = []
    for permutation in permutations:
        aligned_gt, aligned_valid = gt[permutation], gt_valid[permutation]
        # Invalid source coordinates are not evaluated; invalid predictions at
        # valid targets retain a fixed penalty in this branch's denominator.
        distances = np.full(9, float(diagonal))
        comparable = n2_valid & aligned_valid
        distances[comparable] = np.linalg.norm(n2[comparable] - aligned_gt[comparable], axis=1)
        costs.append(float(distances[:8][aligned_valid[:8]].mean()) if aligned_valid[:8].any() else float("inf"))
        aligned_targets.append((aligned_gt, aligned_valid))
    branch = int(np.argmin(costs))
    aligned_gt, aligned_valid = aligned_targets[branch]
    mask = aligned_valid[:8] & both_valid[:8]
    target = np.zeros(8, dtype=np.float32)
    target[mask] = (100. * (np.linalg.norm(n2[:8][mask] - aligned_gt[:8][mask], axis=1)
                           - np.linalg.norm(replay[:8][mask] - aligned_gt[:8][mask], axis=1)) / diagonal).astype(np.float32)
    assert np.isfinite(target).all()
    return dict(target=target, mask=mask.copy(), branch=branch,
                gt_aligned=aligned_gt.copy(), gt_valid=aligned_valid.copy())


def choose_pairs(utility, valid):
    utility = np.asarray(utility, dtype=np.float64)
    valid = _valid(valid)
    if utility.shape != (8,):
        raise ValueError("utility must have shape (8,)")
    return np.array([bool(valid[a] and valid[b] and np.isfinite(utility[[a, b]]).all()
                          and utility[a] > 0 and utility[b] > 0) for a, b in PAIRS])


def apply_pairs(n2, replay, pairs):
    base = np.asarray(n2)
    replay = _points(replay, "replay")
    pairs = np.asarray(pairs, dtype=bool)
    if base.shape != (9, 2) or not np.issubdtype(base.dtype, np.floating) or pairs.shape != (4,):
        raise ValueError("Expected floating N2[9,2] and pair mask[4]")
    result = base.copy()
    for accepted, pair in zip(pairs, PAIRS):
        if accepted:
            if not _finite(replay)[list(pair)].all():
                raise ValueError("Selected Replay corners must be finite and valid")
            result[list(pair)] = replay[list(pair)]
    return result
