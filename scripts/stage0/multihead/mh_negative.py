"""Negative (no-pallet) suppression for SPLIT_LATE_2HEAD.  No new head, no new
architecture, no new geometric loss.

The only addition is a dense zero target on the corner belief maps for frames
whose label says the pallet is absent.

## Why a separate branch is required

`mh_arms.corner_loss` masks channels by `valid`, and for a negative frame every
channel is invalid, so:

    valid all False -> corner_loss = 0.000000e+00   (measured)
    valid all True  -> 9.967461e-01

Feeding negatives through the positive path would therefore compute exactly
nothing.  `object_present == False` is read explicitly and routed here instead.

## Stage contract

The positive supervision is active on `TRAINABLE_BELIEF_STAGES = (4, 5, 6)`, so
the zero suppression uses the same stages -- not "all stages", not "the last
stage".  Affinity is computed by `heads_from_f50` but discarded by
`corner_forward`, i.e. it is inactive, so it is not resurrected here.

## One backward, one step

The negative term is added to the same scalar loss as the positive terms:

    L = L_line(B_L) + lambda_corner * L_corner(B_C) + lambda_neg * L_neg(B_N)

There is no negative-only `optimizer.step()`.  N0 and N1 therefore take an
identical number of optimizer steps on an identical scheduler trajectory, and
the only difference is the extra term in the sum.
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_arms as MH                                             # noqa: E402
import mh_curriculum as CU                                       # noqa: E402
import mh_data as MD                                             # noqa: E402

OUT = MD.OUT
NEG_ROOT = (MD.ROOT / "data/pallet/training_data/paper_release/negative"
            / "extracted")
BELIEF_CHANNELS = 9          # 8 corners + centroid, the channels corner_loss supervises
PRESENCE_CHANNELS = 8        # centroid excluded from the presence score by design


def negative_pool(split, seed=None):
    """[(root, stem), ...] from the filtered manifest.

    The stem alone is ambiguous -- both splits number from f00000 -- so the root
    is carried with it rather than reconstructed later.
    """
    manifest = json.loads(
        (OUT / f"negative_filtered_manifest_{split}.json").read_text())
    root = MD.ROOT / manifest["root"]
    items = [(root, row["stem"]) for row in manifest["items"]]
    if seed is not None:
        import random
        items = list(items)
        random.Random(seed).shuffle(items)
    return items


def load_negative_pack(items):
    """Images only.  A negative frame has no keypoints to build a target from."""
    frames = []
    for root, stem in items:
        image = cv2.imread(str(root / "rgb" / f"{stem}_rgb.png"))
        if image is None:
            raise FileNotFoundError(f"{root.name}/{stem}: no rgb")
        rgb = cv2.cvtColor(cv2.resize(image, (MD.IMAGE, MD.IMAGE)),
                           cv2.COLOR_BGR2RGB)
        frames.append(((rgb.astype(np.float32) / 255.0 - MD.MEAN) / MD.STD)
                      .transpose(2, 0, 1))
    return {"images": torch.from_numpy(np.stack(frames)).to(MD.DEV),
            "chunk": [s for _, s in items]}


def assert_absent(items):
    """Read the contract rather than trusting the folder name."""
    bad = []
    for root, stem in items:
        payload = CU.read_label_from(root, stem)
        if (payload.get("object_present") is not False
                or payload.get("pose_valid") is not False
                or payload.get("objects")):
            bad.append(stem)
    return bad


def negative_belief_loss(beliefs, stages=MH.TRAINABLE_BELIEF_STAGES):
    """Dense zero target on the supervised stages.  No validity mask.

    Every belief channel of a no-pallet frame should be empty everywhere, so the
    mean is taken over the whole map -- this is the one place a dense mean is
    correct, because there is no corner whose channel deserves to be dropped.
    """
    total = 0.0
    for stage in stages:
        predicted = beliefs[stage - 1][:, :BELIEF_CHANNELS]
        total = total + (predicted ** 2).mean()
    return total / len(stages)


def presence_score(beliefs):
    """score_4kp: the 4th highest corner peak, centroid excluded.

    Four correspondences is the minimum a pose needs, so the 4th peak is the
    scalar that says "there is enough corner evidence here to attempt a pose".
    """
    final = beliefs[-1][:, :PRESENCE_CHANNELS]
    peaks = final.amax(dim=(2, 3))
    ordered, _ = peaks.sort(dim=1, descending=True)
    return ordered[:, 3]


def corner_features(model, images):
    """Belief maps from the corner branch only -- the line branch is untouched."""
    return CU.corner_forward(model, images)
