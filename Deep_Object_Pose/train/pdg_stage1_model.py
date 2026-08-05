"""A1 and A2 models.  A1 is ep57 with stages 4-6 trainable and nothing added;
A2 is the same local path plus the three Stage-1 heads, which never write into
the belief or affinity output."""
from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn

_COMMON = pathlib.Path(__file__).resolve().parents[1] / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

import pdg_heads as HEADS                 # noqa: E402
from models import DopeNetwork             # noqa: E402

EP57 = (pathlib.Path(__file__).resolve().parents[2]
        / "weights/paper_s2_stageB/net_epoch_0057.pth")
EP57_SHA = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"


class PDGStage1Model(nn.Module):
    def __init__(self, arm: str, seed: int = 1) -> None:
        super().__init__()
        assert arm in ("A1", "A2"), arm
        self.arm = arm
        self.net = DopeNetwork(numSeg=1)
        self.net.load_state_dict(torch.load(str(EP57), map_location="cpu",
                                            weights_only=True), strict=True)
        for parameter in self.net.vgg.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.net.named_parameters():
            if any(f"m{stage}_" in name for stage in (1, 2, 3)):
                parameter.requires_grad_(False)
        if arm == "A2":
            generator = torch.Generator().manual_seed(seed)
            torch.manual_seed(seed)
            self.prh = HEADS.PalletnessResponseHead()
            self.kvh = HEADS.KeypointVisibilityHead()
            self.trunc = HEADS.TruncationHead()
            del generator

    def shared_feature(self, images: torch.Tensor) -> torch.Tensor:
        return self.net.vgg(images)

    def forward(self, images: torch.Tensor) -> dict:
        outputs = self.net(images)
        beliefs, affinities = outputs[0], outputs[1]
        result = {"beliefs": beliefs, "affinities": affinities,
                  "h6": beliefs[-1], "a6": affinities[-1]}
        if self.arm == "A2":
            feature = self.shared_feature(images)
            result["palletness"] = self.prh(feature)
            result["visibility"] = self.kvh(feature, beliefs[-1][:, :9])
            result["truncation"] = self.trunc(feature)
        return result

    def trainable_groups(self):
        local = [p for n, p in self.net.named_parameters() if p.requires_grad]
        heads = []
        if self.arm == "A2":
            for module in (self.prh, self.kvh, self.trunc):
                heads.extend(list(module.parameters()))
        return local, heads

    def frozen_checksum(self) -> str:
        import hashlib
        digest = hashlib.sha256()
        for name, parameter in sorted(self.net.named_parameters()):
            if not parameter.requires_grad:
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
