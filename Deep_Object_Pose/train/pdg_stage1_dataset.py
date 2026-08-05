"""Source-space dataset for Stage 1.

The canonical loader hands back a 400x400 squashed tensor, so TACA cannot run on
its output: the acceptance envelope is stated in 640x480 source pixels and the
squash is anisotropic.  It has to run where the legacy augmentation already
runs -- on the native frame, before albumentations -- so this subclass
substitutes a TACA adapter at exactly that call and inherits the rest of the
geometry unchanged.  That is what makes the no-transform parity check meaningful
rather than decorative.

Targets are then rebuilt from the parent's own transformed keypoints:

* corners at sigma 2.0 and the centroid at 2.5, which the parent cannot do
  because CreateBeliefMap takes one sigma for all nine channels;
* a corner whose transformed centre left the frame gets no Gaussian and a zero
  loss mask on its belief channel and on both of its affinity channels;
* three-state visibility, decided from the transformed coordinate for
  off-screen and from the loader's own visibility metadata otherwise.
"""
from __future__ import annotations

import contextlib
import pathlib
import random
import sys

import numpy as np
import torch

_COMMON = pathlib.Path(__file__).resolve().parents[1] / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

import pdg_taca as TACA                     # noqa: E402
import utils_dataset as UD                  # noqa: E402
from utils_dataset import CleanVisiiDopeLoader   # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pdg_targets as TARGETS               # noqa: E402

BELIEF = TARGETS.BELIEF
N_KP = TARGETS.N_KP


@contextlib.contextmanager
def _taca_seam(adapter):
    """Swap the one call the parent makes on the native frame."""
    original = UD.apply_truncation_aug
    UD.apply_truncation_aug = adapter
    try:
        yield
    finally:
        UD.apply_truncation_aug = original


class PDGStage1Dataset(CleanVisiiDopeLoader):
    """arm='A1' (no TACA), 'A2' (TACA), or 'PARITY' (diagnostic, shared sigma)."""

    def __init__(self, *args, arm: str = "A1", taca_seed: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        assert arm in ("A1", "A2", "PARITY"), arm
        self.arm = arm
        self.taca_seed = int(taca_seed)
        self.taca_records: list[dict] = []
        # the parent's own augmentation is never used here: it pads its
        # truncation back inside the frame, which is the distribution Stage 1
        # exists to stop producing
        self.truncation_aug_prob = 1.0 if arm == "A2" else 0.0

    # -- TACA adapter ------------------------------------------------------
    def _taca_adapter(self, index: int):
        def adapter(image, keypoints, rng):
            out, moved, record = TACA.apply(image, np.asarray(keypoints, float),
                                            rng)
            record["index"] = int(index)
            self.taca_records.append(record)
            if record["class"] == TACA.CLASS_LEGACY and not record["fallback"]:
                return None            # parent keeps the untouched frame
            return out, moved
        return adapter

    # -- target rebuild ----------------------------------------------------
    def _rebuild(self, sample: dict) -> dict:
        points = sample["refine_keypoints"].numpy().astype(np.float64)
        source_valid = sample["refine_keypoints_valid"].numpy() > 0
        loader_visibility = sample.get("visibility")
        occluded = np.zeros(N_KP, dtype=bool)
        visibility_known = np.ones(N_KP, dtype=bool)
        if loader_visibility is not None:
            values = loader_visibility.numpy()
            occluded = values < 0.5
            visibility_known = np.isfinite(values)

        corner_sigma = TARGETS.CORNER_SIGMA
        centroid_sigma = (TARGETS.CORNER_SIGMA if self.arm == "PARITY"
                          else TARGETS.CENTROID_SIGMA)
        inside = ((points[:, 0] >= 0) & (points[:, 0] < BELIEF)
                  & (points[:, 1] >= 0) & (points[:, 1] < BELIEF))

        # the repository generator, called once per sigma group, so PARITY mode
        # (both groups at 2.0) is bit-identical to the parent and the only
        # difference in A1/A2 is the centroid width and the off-screen mask
        from utils_belief import CreateBeliefMap
        listed = [points.tolist()]
        clip = bool(getattr(self, "clip_belief_border", False))
        corner_maps = CreateBeliefMap(size=BELIEF, pointsBelief=listed,
                                      nbpoints=N_KP, sigma=corner_sigma,
                                      clip_at_border=clip)
        centroid_maps = (corner_maps if centroid_sigma == corner_sigma else
                         CreateBeliefMap(size=BELIEF, pointsBelief=listed,
                                         nbpoints=N_KP, sigma=centroid_sigma,
                                         clip_at_border=clip))
        belief = np.zeros((N_KP, BELIEF, BELIEF), dtype=np.float32)
        belief_mask = np.zeros(N_KP, dtype=np.float32)
        visibility = np.full(N_KP, TARGETS.VIS_OFF_SCREEN, dtype=np.int64)
        visibility_mask = np.zeros(N_KP, dtype=np.float32)
        for channel in range(N_KP):
            if not source_valid[channel]:
                continue
            visibility_mask[channel] = 1.0 if visibility_known[channel] else 0.0
            if not inside[channel]:
                visibility[channel] = TARGETS.VIS_OFF_SCREEN
                continue
            source = centroid_maps if channel == N_KP - 1 else corner_maps
            belief[channel] = np.asarray(source[channel], dtype=np.float32)
            belief_mask[channel] = 1.0
            visibility[channel] = (TARGETS.VIS_OCCLUDED if occluded[channel]
                                   else TARGETS.VIS_VISIBLE)

        affinity_mask = sample["affinity_channel_mask"].numpy().copy()
        for corner in range(8):
            if belief_mask[corner] == 0.0:
                affinity_mask[2 * corner] = 0.0
                affinity_mask[2 * corner + 1] = 0.0

        hull = np.zeros((BELIEF, BELIEF), dtype=np.float32)
        usable = points[:8][inside[:8] & source_valid[:8]]
        if len(usable) >= 3:
            import cv2
            canvas = np.zeros((BELIEF, BELIEF), dtype=np.uint8)
            polygon = cv2.convexHull(usable.astype(np.float32).reshape(-1, 1, 2))
            cv2.fillConvexPoly(canvas, np.round(polygon).astype(np.int32), 1)
            hull = canvas.astype(np.float32)

        truncated = bool(source_valid[:8].any()
                         and (~inside[:8] & source_valid[:8]).any())
        all_in_frame = bool(source_valid.all() and inside.all())

        sample["beliefs"] = torch.from_numpy(belief).double()
        sample["belief_channel_mask"] = torch.from_numpy(
            sample["belief_channel_mask"].numpy() * belief_mask)
        sample["affinity_channel_mask"] = torch.from_numpy(affinity_mask)
        sample["pdg_visibility"] = torch.from_numpy(visibility)
        sample["pdg_visibility_mask"] = torch.from_numpy(visibility_mask)
        sample["pdg_palletness"] = torch.from_numpy(hull)[None]
        sample["pdg_truncated"] = torch.tensor([1.0 if truncated else 0.0])
        sample["pdg_in_frame"] = torch.from_numpy(inside.astype(np.float32))
        if "diffpnp_valid" in sample:
            sample["diffpnp_valid"] = sample["diffpnp_valid"] * (
                1.0 if all_in_frame else 0.0)
        return sample

    def __getitem__(self, index):
        if self.arm == "A2":
            rng = random.Random((self.taca_seed * 2654435761) ^ int(index))
            adapter = self._taca_adapter(index)
            with _taca_seam(lambda img, kps, _r: adapter(img, kps, rng)):
                sample = super().__getitem__(index)
        else:
            sample = super().__getitem__(index)
        return self._rebuild(sample)


def build(arm: str, options, taca_seed: int = 1):
    """Mirror the canonical loader construction, swapping in this subclass."""
    import importlib.util
    root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "screen", root / "scripts/stage0/paper_s2_corner_replacement_screen.py")
    screen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(screen)
    class _Factory(PDGStage1Dataset):
        def __init__(self, *a, **k):
            super().__init__(*a, arm=arm, taca_seed=taca_seed, **k)

    # train.build_training_loader resolves the class from its own module
    # namespace (train.py:804), so that is the binding to swap.
    import train as TRAIN
    originals = [(TRAIN, getattr(TRAIN, "CleanVisiiDopeLoader", None)),
                 (UD, UD.CleanVisiiDopeLoader)]
    for module, _ in originals:
        module.CleanVisiiDopeLoader = _Factory
    try:
        dataset, loader, extra_a, extra_b = screen.build_loader(options)
    finally:
        for module, previous in originals:
            if previous is not None:
                module.CleanVisiiDopeLoader = previous
    return dataset, loader, extra_a, extra_b
