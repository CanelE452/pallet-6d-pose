"""Joint full-ROI residual correction; inference never accepts targets."""
import torch
from torch import nn
from torch.nn import functional as F

KEYS = ('p3', 'p4', 'points', 'boxes', 'point_valid', 'input_shape', 'context')
VERTICAL = ((0, 3), (1, 2), (4, 7), (5, 6))


class WideRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.p3 = nn.Conv2d(64, 16, 1)
        self.p4 = nn.Conv2d(128, 16, 1)
        self.visual = nn.Sequential(nn.Conv2d(42, 32, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
                                    nn.AdaptiveAvgPool2d(4), nn.Flatten())
        self.output = nn.Sequential(nn.Linear(535, 128), nn.SiLU(), nn.Linear(128, 16))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        t = (torch.arange(32, dtype=torch.float32) + .5) / 32
        y, x = torch.meshgrid(t, t, indexing='ij')
        self.register_buffer('uv', torch.stack((x, y), -1))

    def forward(self, batch):
        p = batch['points'].float()
        valid = batch['point_valid'].bool() & torch.isfinite(p).all(-1)
        box = batch['boxes'].float()
        wh = (box[:, 2:] - box[:, :2]).clamp_min(1)
        diagonal = wh.norm(dim=-1)
        lo = box[:, :2] - .2 * wh
        span = 1.4 * wh
        xy = lo[:, None, None] + self.uv[None] * span[:, None, None]
        # Both feature maps are right/bottom padded to a 640px canvas.
        # align_corners=False gives feature centres at (i+.5)*stride.
        grid = 2 * xy / 640 - 1
        shape = batch['input_shape'].float()[:, [1, 0]]
        support = ((xy >= 0) & (xy < shape[:, None, None])).all(-1)[:, None]
        features = []
        for key, project, stride in [('p3', self.p3, 8), ('p4', self.p4, 16)]:
            feature = project(batch[key].float())
            yy = (torch.arange(feature.shape[-2], device=p.device) + .5) * stride
            xx = (torch.arange(feature.shape[-1], device=p.device) + .5) * stride
            native_support = ((yy[None, :, None] < shape[:, 1, None, None])
                              & (xx[None, None, :] < shape[:, 0, None, None]))
            feature = feature * native_support[:, None]
            features.append(F.grid_sample(feature, grid, align_corners=False) * support)
        safe = torch.where(valid[..., None], p, torch.zeros_like(p))
        relative = (safe[:, :8] - lo[:, None]) / span[:, None]
        dist = (self.uv[None, None] - relative[:, :, None, None]).square().sum(-1)
        priors = torch.exp(-dist / (2 * .04 ** 2)) * valid[:, :8, None, None]
        coords = self.uv.permute(2, 0, 1)[None].expand(len(p), -1, -1, -1) * 2 - 1
        visual = self.visual(torch.cat([*features, priors, coords], 1))
        meta = torch.cat([relative.flatten(1), wh / diagonal[:, None], batch['context'].float()], 1)
        delta = self.output(torch.cat([visual, meta], 1)).view(-1, 8, 2).tanh()
        delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1)
        delta = delta * (.20 * diagonal[:, None, None])
        corners = torch.where(valid[:, :8, None], p[:, :8] + delta, p[:, :8])
        return torch.cat([corners, p[:, 8:]], 1)


def corrupt(batch, generator):
    """50% unchanged; 25% one corner; 25% a shared vertical-edge offset."""
    out = dict(batch)
    p = batch['points'].clone()
    n, device = len(p), p.device
    mode = torch.rand(n, generator=generator, device=device)
    pick = torch.randint(8, (n,), generator=generator, device=device)
    edge = torch.randint(4, (n,), generator=generator, device=device)
    angle = torch.rand(n, generator=generator, device=device) * (2 * torch.pi)
    radius = .05 + .10 * torch.rand(n, generator=generator, device=device)
    diag = (batch['boxes'][:, 2:] - batch['boxes'][:, :2]).float().norm(dim=-1)
    delta = torch.stack([angle.cos(), angle.sin()], -1) * (radius * diag)[:, None]
    mask = torch.zeros((n, 9), device=device, dtype=torch.bool)
    rows = torch.arange(n, device=device)
    mask[rows, pick] = (mode >= .5) & (mode < .75)
    pairs = torch.tensor(VERTICAL, device=device)[edge]
    mask[rows[:, None], pairs] |= (mode >= .75)[:, None]
    mask &= batch['point_valid'].bool()
    p = torch.where(mask[..., None], p + delta[:, None], p)
    out['points'] = p
    return out


def objective(q, batch):
    gt = batch['gt_points'].float()[:, :8]
    mask = (batch['gt_valid'][:, :8].bool() & batch['point_valid'][:, :8].bool()
            & torch.isfinite(gt).all(-1) & torch.isfinite(q[:, :8]).all(-1))
    gt = torch.where(mask[..., None], gt, q[:, :8].detach())
    diagonal = (batch['boxes'][:, 2:] - batch['boxes'][:, :2]).float().norm(dim=-1).clamp_min(1)
    diff = (q[:, :8] - gt) / diagonal[:, None, None]
    loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=.01, reduction='none').sum(-1)
    return ((loss * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
