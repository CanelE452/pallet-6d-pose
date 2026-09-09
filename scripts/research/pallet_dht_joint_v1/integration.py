"""Trainable global Hough feedback inside the unfused YOLO26 pose network.

No detection selection, GT crop, external corner correction, or pseudo-labels.
The stock one2one detach policy is retained. Auxiliary semantic-line supervision
is added once per batch after the unchanged stock E2ELoss(PoseLoss26).
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from ultralytics.nn.modules.head import Pose26
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.loss import E2ELoss, PoseLoss26

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ARMS = ("point_only", "hough_features", "hough_joint")
R0_PATH = ROOT / "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt"
R0_SHA256 = "970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7"


class DHTPose26(Pose26):
    """Preserve all stock head state keys and insert trainable neck feedback."""
    @classmethod
    def from_stock(cls, stock: Pose26, arm: str):
        from hough_block import HoughFeatureFusion
        if type(stock) is not Pose26 or stock.cv4 is None or stock.flow_model is None:
            raise ValueError("An unfused stock Pose26 with both training branches is required")
        # This is a new in-memory module; no checkpoint/source object is modified.
        head = cls.__new__(cls)
        head.__dict__ = copy.deepcopy(stock.__dict__)
        head.hough = HoughFeatureFusion(arm=arm)
        head.joint_arm = arm
        return head

    def forward(self, x):
        hw = tuple(int(v) for v in x[1].shape[-2:])
        fused = self.hough(x)
        logits = self.hough.line_logits
        lattice = self.hough.lattice
        # Keep checkpoint/EMA deepcopy safe: the live graph travels in preds.
        self.hough.clear_transient()
        output = super().forward(fused)
        auxiliary = dict(line_logits=logits, line_lattice=lattice,
                         line_input_hw=tuple(int(v * 16) for v in hw))
        if self.training:
            output.update(auxiliary)
        elif isinstance(output, tuple):
            output[1].update(auxiliary)
        return output


class JointCriterion:
    def __init__(self, model):
        self.stock = E2ELoss(model, PoseLoss26)
        self.weight = float(model.line_weight) if model.joint_arm == "hough_joint" else 0.0
        self.latest_line_diagnostics = {}

    @property
    def updates(self):
        return self.stock.updates

    @updates.setter
    def updates(self, value):
        self.stock.updates = value

    def update(self):
        self.stock.update()

    def __call__(self, predictions, batch):
        raw = predictions[1] if isinstance(predictions, (tuple, list)) else predictions
        base, items = self.stock(predictions, batch)
        logits = raw["line_logits"]
        if self.weight:
            from line_targets import auxiliary_line_loss
            aux, diagnostics = auxiliary_line_loss(
                logits, batch["keypoints"], batch["batch_idx"],
                tuple(int(v) for v in batch["img"].shape[-2:]), raw["line_lattice"],
            )
            self.latest_line_diagnostics = diagnostics
        else:
            aux = logits.float().sum() * 0.0
            self.latest_line_diagnostics = {"auxiliary_weight": 0.0}
        weighted = aux * self.weight
        # Stock returns a per-component vector multiplied by batch size.
        return (torch.cat((base, (weighted * batch["img"].shape[0]).reshape(1))),
                torch.cat((items, weighted.detach().reshape(1))))


class DHTPoseModel(PoseModel):
    def init_criterion(self):
        return JointCriterion(self)


def build_model(arm: str, weights: str | Path = R0_PATH, line_weight: float = .1,
                verbose: bool = False) -> PoseModel:
    """Warm-start stock weights, then add only the new Hough module.

    A zero-initialized feedback projection makes initial point/box/conf outputs
    identical to R0. The caller controls device, mode, and training hyperparameters.
    Loading does not fuse, predict, change the source checkpoint, or access GT.
    """
    if arm not in ARMS or line_weight < 0:
        raise ValueError("Unsupported arm or negative auxiliary loss weight")
    checkpoint = torch.load(str(weights), map_location="cpu")
    source = checkpoint.get("ema") or checkpoint.get("model")
    if source is None or not isinstance(source, PoseModel):
        raise ValueError("Expected an unfused Ultralytics PoseModel checkpoint")
    model = copy.deepcopy(source).float()
    if type(model.model[-1]) is not Pose26 or model.model[-1].cv4 is None:
        raise ValueError("Training must begin from the unfused stock R0 architecture")
    if list(model.model[-1].kpt_shape) != [9, 3]:
        raise ValueError("Pallet keypoint contract is nine xyz/visibility triplets")
    model.criterion = None
    model.joint_arm, model.line_weight = arm, float(line_weight)
    model.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, **checkpoint.get("train_args", {})})
    if arm != "point_only":
        model.model[-1] = DHTPose26.from_stock(model.model[-1], arm)
        model.__class__ = DHTPoseModel
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    if verbose:
        print(f"{arm}: {sum(p.numel() for p in model.parameters()):,} trainable parameters")
    return model


def snapshot(model):
    """Small CPU parameter/BN snapshot used for completion provenance."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if isinstance(v, torch.Tensor) and (v.is_floating_point() or "num_batches_tracked" in k)}


def changes_since(model, initial):
    state = model.state_dict()
    groups = {}
    for name, before in initial.items():
        if name not in state:
            raise ValueError(f"Missing initial tensor: {name}")
        after = state[name].detach().cpu()
        if not torch.isfinite(after).all():
            raise ValueError(f"Nonfinite trained tensor: {name}")
        group = ("hough" if ".hough." in name else "pose_head" if name.startswith("model.23.")
                 else "backbone_neck")
        g = groups.setdefault(group, dict(tensors=0, changed_tensors=0, changed_values=0,
                                         max_abs_change=0., bn_changed=0))
        n = int(torch.count_nonzero(after != before))
        g["tensors"] += 1; g["changed_tensors"] += int(n > 0); g["changed_values"] += n
        g["max_abs_change"] = max(g["max_abs_change"], float((after.double()-before.double()).abs().max()))
        g["bn_changed"] += int(n > 0 and any(s in name for s in ("running_mean", "running_var", "num_batches_tracked")))
    return groups


def load_trained_model(checkpoint_path, use_ema=True):
    """Load a complete unfused model without inference or state mutation."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not checkpoint.get("complete"):
        raise ValueError("Incomplete checkpoint cannot be used for final evaluation")
    model = checkpoint.get("ema") if use_ema else checkpoint.get("model")
    if model is None:
        raise ValueError("Requested model state is absent")
    result = copy.deepcopy(model).float().eval()
    result.criterion = None
    return result, checkpoint
