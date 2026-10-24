"""V181 prospective raw96 support head. No file I/O, crop generation, or labels.

The two support logits are independent and nonexclusive. Owner logits describe
literal point ownership only; they do not redefine native visibility.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class LocalInputSpec:
    size: int = 96
    channels: int = 8
    support_order: tuple = ("visible_end_support", "foreign_front_support")
    owner_order: tuple = ("TARGET", "FOREIGN", "BACKGROUND")


SPEC = LocalInputSpec()


def relative_xy_grid(device=None):
    """Constant pixel-center XY, q at (47.5,47.5); independent of q/L/native."""
    centers = (torch.arange(96, device=device, dtype=torch.float32) + 0.5) / 96.0
    axis = centers * 2.0 - 1.0
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=0)


def assemble_local_input(rgb96, valid96, *, use_sam, sam_mask96=None, sam_available=None):
    """Assemble same eight channels for both arms from already aligned arrays.

    rgb96: uint8[N,96,96,3], valid96: bool[N,96,96]. Original raw RGB and
    validity are unchanged. This helper does not decide original input
    availability or supervision admission. SAM channels are zero in RAW;
    missing SAM is explicitly represented by mask=0 and availability=0.
    SAM masks must already come from the separately frozen same-grid producer.
    """
    if rgb96.dtype != torch.uint8 or rgb96.ndim != 4 or tuple(rgb96.shape[1:]) != (96,96,3):
        raise ValueError("Original rgb96 must be uint8[N,96,96,3]")
    n = rgb96.shape[0]
    if valid96.dtype != torch.bool or tuple(valid96.shape) != (n,96,96):
        raise ValueError("Original valid96 must be bool[N,96,96]")
    if rgb96.device != valid96.device:
        raise ValueError("Input devices must match")
    if not isinstance(use_sam, bool):
        raise ValueError("Arm flag must be boolean")
    if use_sam:
        if sam_mask96 is None or sam_available is None:
            raise ValueError("SAM arm requires explicit same-grid mask and availability")
        if sam_mask96.dtype != torch.bool or tuple(sam_mask96.shape) != (n,96,96):
            raise ValueError("SAM mask must be bool[N,96,96]")
        if sam_available.dtype != torch.bool or tuple(sam_available.shape) != (n,96,96):
            raise ValueError("SAM availability must be bool[N,96,96]")
        if sam_mask96.device != rgb96.device or sam_available.device != rgb96.device:
            raise ValueError("SAM devices must match original input")
        sm = (sam_mask96 & sam_available).to(torch.float32)[:,None]
        sa = sam_available.to(torch.float32)[:,None]
    else:
        if sam_mask96 is not None or sam_available is not None:
            raise ValueError("RAW arm must not consume SAM tensors")
        sm = torch.zeros((n,1,96,96),dtype=torch.float32,device=rgb96.device)
        sa = torch.zeros_like(sm)
    rgb = rgb96.permute(0,3,1,2).contiguous().to(torch.float32) / 255.0
    valid = valid96.to(torch.float32)[:,None]
    xy = relative_xy_grid(rgb96.device)[None].expand(n,-1,-1,-1)
    return torch.cat((rgb,valid,xy,sm,sa),dim=1).contiguous()


class Raw96SupportHead(torch.nn.Module):
    """Same prospective 115157-parameter GN network for RAW and SAM arms."""
    def __init__(self):
        super().__init__()
        layers = []
        incoming = 8
        for outgoing,stride in [(16,1),(32,2),(32,1),(48,2),(48,1),(64,2)]:
            layers.extend((torch.nn.Conv2d(incoming,outgoing,3,stride=stride,padding=1),
                           torch.nn.GroupNorm(4 if outgoing == 16 else 8,outgoing),
                           torch.nn.SiLU()))
            incoming = outgoing
        self.encoder = torch.nn.Sequential(*layers)
        # Fixed96 input and three stride2 blocks give12x12. Four-by-four means
        # are the same12-to3 average partition; this has strict CUDA backward.
        self.pool = torch.nn.AvgPool2d(kernel_size=4, stride=4)
        self.shared = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(64*3*3,64),torch.nn.SiLU())
        self.support_head = torch.nn.Linear(64,2)
        self.owner_head = torch.nn.Linear(64,3)

    def forward(self, inputs):
        if inputs.dtype != torch.float32 or inputs.ndim != 4 or tuple(inputs.shape[1:]) != (8,96,96):
            raise ValueError("Expected FP32[N,8,96,96] only")
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("Nonfinite local input")
        z = self.shared(self.pool(self.encoder(inputs)))
        return {"support_logits":self.support_head(z),
                "owner_logits":self.owner_head(z)}

