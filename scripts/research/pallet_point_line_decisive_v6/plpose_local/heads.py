"""Point / direct-line / Hough-line heads over a fixed ROI feature crop.

Deviation from the external v5 export contract: features carry the native 192
channels (stock R0 P3 64 + P4 128) rather than a pre-reduced 32, so no arbitrary
lossy projection sits between the backbone and the trunk. The trunk's first
convolution does the reduction and is trained.

The three arms share the trunk, the dimension and camera conditioning, the
point head and the seed head. They differ only in how -- or whether -- a line
distribution over the (theta, rho) lattice is produced. Nothing here reads
ground truth.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from .geometry import EDGES

ARMS = ('point', 'direct', 'hough')


def hough_matrix(size: int, thetas: int, rhos: int, dtype=torch.float32) -> Tensor:
    """[T*R, size*size] accumulation weights over ROI-normalised coordinates.

    Cell centres live in [-1, 1]^2. A pixel contributes to a (theta, rho) bin
    with a triangular kernel in rho, so the transform stays differentiable and
    preserves a constant input's total mass up to the kernel width.
    """
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, size, dtype=dtype),
                            torch.linspace(-1, 1, size, dtype=dtype), indexing='ij')
    th = torch.arange(thetas, dtype=dtype) * math.pi / thetas
    rho_max = math.sqrt(2.)
    grid = torch.linspace(-rho_max, rho_max, rhos, dtype=dtype)
    step = grid[1] - grid[0]
    proj = (xs.reshape(1, -1) * torch.cos(th)[:, None]
            + ys.reshape(1, -1) * torch.sin(th)[:, None])          # [T, HW]
    w = (1 - (proj[:, None] - grid[None, :, None]).abs() / step).clamp_min(0)
    return w.reshape(thetas * rhos, -1)


def lattice_lines(thetas: int, rhos: int, dtype=torch.float32) -> Tensor:
    """[T*R, 3] homogeneous lines in ROI-normalised coordinates."""
    th = torch.arange(thetas, dtype=dtype) * math.pi / thetas
    rho_max = math.sqrt(2.)
    grid = torch.linspace(-rho_max, rho_max, rhos, dtype=dtype)
    c, s = torch.cos(th)[:, None].expand(thetas, rhos), torch.sin(th)[:, None].expand(thetas, rhos)
    r = grid[None].expand(thetas, rhos)
    return torch.stack((c, s, -r), -1).reshape(-1, 3)


def decode_modes(logits: Tensor, lines: Tensor, thetas: int, rhos: int,
                 modes: int = 3, radius: int = 2) -> tuple[Tensor, Tensor]:
    """Keep `modes` separated peaks per role and average locally around each.

    A single global soft-argmax over a multi-modal distribution would invent a
    line between two real ones, so peaks are found first and the continuous
    average is taken only inside a window. The returned mass is the conditional
    mass of the kept modes, not a probability that the line is correct.
    """
    b, role, n = logits.shape
    prob = torch.softmax(logits.reshape(b, role, -1), -1)
    work = prob.clone()
    out_l, out_m = [], []
    for _ in range(modes):
        peak = work.argmax(-1)                                        # [B,role]
        t_idx, r_idx = peak // rhos, peak % rhos
        dt = (torch.arange(thetas, device=logits.device)[None, None] - t_idx[..., None]).abs()
        rg = torch.arange(rhos, device=logits.device)[None, None]
        dr = (rg - r_idx[..., None]).abs()
        dr_flip = (rg - (rhos - 1 - r_idx)[..., None]).abs()
        # Unoriented-line parameter space has (theta+pi,rho) == (theta,-rho).
        near = ((dt[..., None] <= radius) & (dr[..., None, :] <= radius)) | (
            ((thetas-dt)[..., None] <= radius) & (dr_flip[..., None, :] <= radius))
        near = near.reshape(b, role, -1)
        w = prob * near
        mass = w.sum(-1)
        wn = w / mass.clamp_min(1e-9)[..., None]
        reference = lines[peak]
        dot = (lines[None, None, :, :2] * reference[..., None, :2]).sum(-1)
        sign = torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))
        mean = (wn[..., None] * sign[..., None] * lines[None, None]).sum(-2)
        mean = mean / torch.linalg.vector_norm(mean[..., :2], dim=-1, keepdim=True).clamp_min(1e-9)
        out_l.append(mean)
        out_m.append(mass)
        work = work * (~near)
    return torch.stack(out_l, 2), torch.stack(out_m, 2)


class Trunk(nn.Module):
    def __init__(self, channels=192, width=48, condition_dims=True):
        super().__init__()
        self.condition_dims = condition_dims
        self.stem = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1), nn.GroupNorm(4, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(4, width), nn.SiLU())
        self.cond = nn.Sequential(nn.Linear(9, width), nn.SiLU(), nn.Linear(width, 2 * width))
        self.mix = nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                 nn.GroupNorm(4, width), nn.SiLU())

    def context(self, K, dims, box, image_hw):
        wh = (box[:, 2:] - box[:, :2]).clamp_min(1e-6)
        diag = torch.linalg.vector_norm(image_hw.flip(-1), dim=-1, keepdim=True)
        feats = [wh / diag, (box[:, :2] + box[:, 2:]) * .5 / diag,
                 K[:, 0, 0:1] / diag, K[:, 1, 1:2] / diag]
        d = dims if self.condition_dims else torch.zeros_like(dims)
        return torch.cat(feats + [d], -1)

    def forward(self, features, content, K, dims, box, image_hw):
        x = self.stem(torch.where(content, features, torch.zeros_like(features)))
        g, b = self.cond(self.context(K, dims, box, image_hw)).chunk(2, -1)
        x = x * (1 + g[..., None, None]) + b[..., None, None]
        return self.mix(x) * content.to(x.dtype)


class PointHead(nn.Module):
    """Heatmap mean plus an unbounded residual, so amodal corners may fall outside."""
    def __init__(self, width=48, points=9):
        super().__init__()
        self.points = points
        self.heat = nn.Conv2d(width, points, 1)
        self.res = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, points * 2))

    def forward(self, x, box):
        b, _, h, w = x.shape
        prob = torch.softmax(self.heat(x).reshape(b, self.points, -1), -1)
        ys, xs = torch.meshgrid(torch.linspace(0, 1, h, device=x.device),
                                torch.linspace(0, 1, w, device=x.device), indexing='ij')
        grid = torch.stack((xs.reshape(-1), ys.reshape(-1)), -1)
        uv01 = prob @ grid
        res = self.res(x.mean((-2, -1))).reshape(b, self.points, 2)
        size = (box[:, 2:] - box[:, :2])[:, None]
        return box[:, None, :2] + (uv01 + res) * size


class LineHead(nn.Module):
    def __init__(self, arm, width=48, thetas=36, rhos=65, roles=12, size=24):
        super().__init__()
        self.arm, self.thetas, self.rhos, self.roles = arm, thetas, rhos, roles
        if arm == 'direct':
            self.query = nn.Parameter(torch.randn(roles, width) * .02)
            self.score = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(),
                                       nn.Linear(width, thetas * rhos))
        elif arm == 'hough':
            self.reduce = nn.Conv2d(width, 16, 1)
            self.register_buffer('H', hough_matrix(size, thetas, rhos))
            self.conv = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
                                      nn.Conv2d(16, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU())
            self.out = nn.Conv2d(16, roles, 1)
        else:
            raise ValueError(arm)

    def forward(self, x):
        b = x.shape[0]
        if self.arm == 'direct':
            pooled = x.mean((-2, -1))
            q = self.query[None].expand(b, -1, -1)
            pair = torch.cat((pooled[:, None].expand_as(q), q), -1)
            return self.score(pair).reshape(b, self.roles, self.thetas * self.rhos)
        z = self.reduce(x).flatten(2)                                  # [B,16,HW]
        acc = (z @ self.H.T).reshape(b, 16, self.thetas, self.rhos)    # feature voting
        acc = acc / (self.H.sum(-1).reshape(1, 1, self.thetas, self.rhos) + 1e-6)
        return self.out(self.conv(acc)).reshape(b, self.roles, self.thetas * self.rhos)


class SeedHead(nn.Module):
    """Image-derived multiple starts; no point PnP is required to begin."""
    def __init__(self, width=48, starts=4):
        super().__init__()
        self.starts = starts
        self.net = nn.Sequential(nn.Linear(width + 9, width), nn.SiLU(), nn.Linear(width, starts * 9))

    def forward(self, x, ctx, dims, K, box):
        b = x.shape[0]
        raw = self.net(torch.cat((x.mean((-2, -1)), ctx), -1)).reshape(b, self.starts, 9)
        a, bvec = raw[..., :3], raw[..., 3:6]
        # Gram-Schmidt 6D -> rotation, so every start is a valid SO(3) element.
        e1 = a / torch.linalg.vector_norm(a, dim=-1, keepdim=True).clamp_min(1e-6)
        e2 = bvec - (e1 * bvec).sum(-1, keepdim=True) * e1
        e2 = e2 / torch.linalg.vector_norm(e2, dim=-1, keepdim=True).clamp_min(1e-6)
        R = torch.stack((e1, e2, torch.cross(e1, e2, dim=-1)), -1)
        # Metric depth seed from the object diagonal and the box size (dims are
        # used here even when neural conditioning on dims is switched off).
        obj = torch.linalg.vector_norm(dims, dim=-1)[:, None]
        pix = torch.linalg.vector_norm((box[:, 2:] - box[:, :2]), dim=-1)[:, None].clamp_min(1.)
        z = (K[:, 0, 0:1] * obj / pix).clamp(.2, 50.)
        centre = (box[:, :2] + box[:, 2:]) * .5
        xy = (centre - torch.stack((K[:, 0, 2], K[:, 1, 2]), -1)) / K[:, 0, 0:1]
        t0 = torch.cat((xy * z, z), -1)[:, None].expand(b, self.starts, 3)
        return R, t0 * (1 + .15 * torch.tanh(raw[..., 6:9]))


class PoseNet(nn.Module):
    def __init__(self, arm='hough', channels=192, width=48, starts=4, thetas=36, rhos=65,
                 size=24, modes=3, condition_dims=True):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm, self.modes, self.thetas, self.rhos = arm, modes, thetas, rhos
        self.trunk = Trunk(channels, width, condition_dims)
        self.point = PointHead(width)
        self.seed = SeedHead(width, starts)
        self.line = None if arm == 'point' else LineHead(arm, width, thetas, rhos, len(EDGES), size)
        self.register_buffer('lattice', lattice_lines(thetas, rhos))

    def forward(self, obs: dict):
        x = self.trunk(obs['features'], obs['content'], obs['K'], obs['dims'],
                       obs['box'], obs['image_hw'])
        ctx = self.trunk.context(obs['K'], obs['dims'], obs['box'], obs['image_hw'])
        points = self.point(x, obs['box'])
        R0, t0 = self.seed(x, ctx, obs['dims'], obs['K'], obs['box'])
        out = dict(points=points, R0=R0, t0=t0, line_logits=None, lines=None, mass=None)
        if self.line is not None:
            logits = self.line(x)
            lines_roi, mass = decode_modes(logits, self.lattice, self.thetas, self.rhos, self.modes)
            out.update(line_logits=logits, lines=roi_lines_to_raw(lines_roi, obs['box']), mass=mass)
        return out


def roi_lines_to_raw(lines: Tensor, box: Tensor) -> Tensor:
    """ROI-normalised [-1,1] homogeneous lines -> raw-pixel homogeneous lines."""
    centre = (box[:, :2] + box[:, 2:]) * .5
    half = (box[:, 2:] - box[:, :2]).clamp_min(1e-6) * .5
    a, b, c = lines[..., 0], lines[..., 1], lines[..., 2]
    ax = a / half[:, None, None, 0]
    by = b / half[:, None, None, 1]
    cc = c - ax * centre[:, None, None, 0] - by * centre[:, None, None, 1]
    return torch.stack((ax, by, cc), -1)
