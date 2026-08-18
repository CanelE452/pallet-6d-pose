"""A0 / A1 / A2: one shared backbone, one line head, and the heads already in it.

The three arms differ by exactly one thing each.  Nothing is invented: the corner
and mask heads are the modules that are already inside the A1 checkpoint
(`weights/paper_s2/paper_s2_pdg/A1/epoch_003.pth` is a `DopeNetwork(numSeg=1)`),
so all three arms load byte-identical weights and no head is randomly
initialised.  That matters twice over -- the RNG stream is the same in every arm,
which is what makes the zero-weight parity test in `mh_wiring` exact, and the
gradient calibration in section 12 is not measuring a random head.

    A0_LINE_ONLY          vgg[19:27] + DirectHoughModel
    A1_CORNER_LINE        + belief stages 4-6 trained, L_corner added
    A2_CORNER_LINE_MASK   + seg stages trained, L_mask added

Two departures from the stock DOPE recipe, stated because they are choices:

  * the affinity head is frozen and unsupervised.  Affinity exists in DOPE to
    associate corners with instances, there is one pallet per frame here, and
    this project already measured that swapping affinity association in at
    inference is indistinguishable from reading the belief peak.  Supervising it
    would put a second uncontrolled factor inside the A0 -> A1 contrast.  Its
    outputs still feed the stage inputs, deterministically, so the stage
    architecture is unchanged.
  * belief stages 1-3 are frozen and stages 4-6 trained, which is the boundary
    `PDGStage1Model` already uses for this checkpoint.  Supervising a frozen
    stage would inflate the reported loss without producing a gradient and would
    corrupt the lambda measurement.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
_S0 = str(ROOT / "scripts/stage0")
sys.path[:0] = [_S0] + [os.path.join(_S0, d) for d in sorted(os.listdir(_S0))
                        if os.path.isdir(os.path.join(_S0, d)) and not d.startswith(".")]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LATE = _load("MH_LATE", "scripts/stage0/adaptation/late_a1_adaptation_screen.py")
DH, CAP, V2 = LATE.DH, LATE.CAP, LATE.V2
DEV = LATE.DEV

ARMS = ("A0_LINE_ONLY", "A1_CORNER_LINE", "A2_CORNER_LINE_MASK")
TRAINABLE_BELIEF_STAGES = (4, 5, 6)
FIRST_TRAINABLE_VGG = LATE.FIRST_TRAINABLE_INDEX          # 19
EXPECTED_VGG_TRAINABLE = 5_014_912                        # HARD_BLOCK reference


# --------------------------------------------------------------------------


def heads_from_f50(net, f50):
    """Replay `DopeNetwork.forward` from the shared feature instead of the image.

    A faithful re-composition, not a reimplementation: the stage order, the
    concatenation order and the stage inputs are copied from `models.py:167-215`.
    `mh_wiring.T1` pins this against `net(images)` at 0.0 absolute difference, so
    if the upstream forward ever changes, the test fails rather than this file
    drifting quietly.
    """
    beliefs, affinities = [], []
    previous = f50
    for stage in range(1, net.stop_at_stage + 1):
        if stage == 1:
            source = f50
        else:
            source = torch.cat([beliefs[-1], affinities[-1], f50], 1)
        beliefs.append(getattr(net, f"m{stage}_2")(source))
        affinities.append(getattr(net, f"m{stage}_1")(source))
        previous = source
    del previous
    segments = None
    if net.numSeg > 0:
        seg1 = net.m_seg1(f50)
        seg2 = net.m_seg2(torch.cat([seg1, f50], 1))
        segments = [seg1, seg2]
    return beliefs, affinities, segments


class MultiHeadModel(nn.Module):
    """The shared trunk plus whichever heads this arm is allowed to train."""

    def __init__(self, arm: str):
        super().__init__()
        assert arm in ARMS, arm
        self.arm = arm
        # Construction order is fixed across arms so the RNG stream is shared.
        # Neither the corner head nor the mask head is constructed here at all --
        # both arrive with the checkpoint through AdaptableA1.
        self.a1 = LATE.AdaptableA1(trainable_from=FIRST_TRAINABLE_VGG)
        self.line = DH.DirectHoughModel().to(DEV)
        self.net = self.a1.inner.model.net
        self._set_trainable()

    def _set_trainable(self):
        for parameter in self.net.parameters():
            if parameter.requires_grad and not self._is_vgg(parameter):
                parameter.requires_grad_(False)
        if self.arm in ("A1_CORNER_LINE", "A2_CORNER_LINE_MASK"):
            for stage in TRAINABLE_BELIEF_STAGES:
                for parameter in getattr(self.net, f"m{stage}_2").parameters():
                    parameter.requires_grad_(True)
        if self.arm == "A2_CORNER_LINE_MASK":
            for module in (self.net.m_seg1, self.net.m_seg2):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def _is_vgg(self, parameter) -> bool:
        return any(parameter is p for p in self.a1.vgg.parameters())

    # -- groups -----------------------------------------------------------

    def shared_parameters(self):
        """The late-A1 block every arm trains -- where lambda is calibrated."""
        return [p for p in self.a1.vgg.parameters() if p.requires_grad]

    def line_parameters(self):
        return list(self.line.parameters())

    def corner_parameters(self):
        return [p for stage in TRAINABLE_BELIEF_STAGES
                for p in getattr(self.net, f"m{stage}_2").parameters()
                if p.requires_grad]

    def mask_parameters(self):
        return [p for m in (self.net.m_seg1, self.net.m_seg2)
                for p in m.parameters() if p.requires_grad]

    def trainable_parameters(self):
        seen, out = set(), []
        for parameter in self.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                out.append(parameter)
        return out

    def report(self) -> dict:
        def count(params):
            return sum(p.numel() for p in params)
        return {"arm": self.arm,
                "vgg_trainable": count(self.shared_parameters()),
                "line_trainable": count(self.line_parameters()),
                "corner_trainable": count(self.corner_parameters()),
                "mask_trainable": count(self.mask_parameters()),
                "total_trainable": count(self.trainable_parameters()),
                "first_trainable_vgg": FIRST_TRAINABLE_VGG,
                "numSeg": self.net.numSeg,
                "stop_at_stage": self.net.stop_at_stage}

    # -- forward ----------------------------------------------------------

    def forward(self, images, hypothesis_features):
        """One pass; every arm computes every head, only the losses differ.

        Computing the corner head in A0 too would change A0's runtime but not its
        gradient, so A0 skips it -- the arms must differ in the loss, and a head
        that receives no gradient and is read by nothing is not part of A0.
        """
        f50, _, _ = self.a1(images)
        out = {"f50": f50, "line_scores": self.line(f50, hypothesis_features)}
        if self.arm == "A0_LINE_ONLY":
            return out
        want_mask = self.arm == "A2_CORNER_LINE_MASK"
        beliefs, affinities, segments = heads_from_f50(self.net, f50)
        out["beliefs"] = beliefs
        out["affinities"] = affinities
        if want_mask:
            out["segments"] = segments
        return out


# --------------------------------------------------------------------------
# losses


def corner_loss(beliefs, target, valid, stages=TRAINABLE_BELIEF_STAGES):
    """Channel-masked MSE over the supervised stages.

    The mask is the point.  A cuboid corner projected outside the grid has no
    Gaussian to regress, and the legacy DOPE path supervises that channel as
    empty -- teaching the network that a truncated corner is "no corner
    anywhere".  Here the channel is dropped from the mean instead, so an
    off-frame corner contributes exactly zero and is never clamped to a border.
    """
    weight = valid.float()[:, :, None, None]
    denominator = weight.sum().clamp_min(1.0) * target.shape[-1] * target.shape[-2]
    total = 0.0
    for stage in stages:
        predicted = beliefs[stage - 1]
        total = total + ((predicted - target) ** 2 * weight).sum() / denominator
    return total / len(stages)


def mask_loss(segments, target, valid=None):
    """BCE-with-logits summed over the two seg stages, per-frame valid weighted.

    Same shape as `Deep_Object_Pose/train/train.py:284-303`, kept deliberately --
    section 8 forbids inventing a segmentation loss for the first ablation.  Every
    frame in this dataset carries a visible mask, so `valid` is all ones here and
    exists only so a source without masks can be mixed in later without changing
    the formula.
    """
    if valid is None:
        valid = torch.ones(target.shape[0], device=target.device)
    denominator = valid.sum().clamp_min(1.0)
    total = 0.0
    for stage in segments:
        per_pixel = F.binary_cross_entropy_with_logits(stage, target,
                                                       reduction="none")
        total = total + (per_pixel.mean(dim=(1, 2, 3)) * valid).sum() / denominator
    return total


def gradient_norm(loss, parameters):
    """||d loss / d shared||, without touching any .grad already accumulated."""
    grads = torch.autograd.grad(loss, parameters, retain_graph=True,
                                allow_unused=True)
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float((grad.detach() ** 2).sum())
    return total ** 0.5
