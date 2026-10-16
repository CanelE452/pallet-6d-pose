"""Object-frame geometry for the local point/line joint-pose pilot.

Object frame: origin = geometric centre, +x = width, +y = up (height),
+z = depth. Dimensions are supplied as (W, D, H) in metres, matching the
external v5 contract. Camera is OpenCV right/down/forward and
``X_camera = R @ X_object + t``.

This is an independent implementation. It is not the author's plpose_v5 core.
"""
from __future__ import annotations
import torch
from torch import Tensor

# Corner sign order over (x, y, z) = (width, up, depth). Index 8 is the centre.
SIGNS = torch.tensor([
    [-1., -1., -1.], [+1., -1., -1.], [+1., -1., +1.], [-1., -1., +1.],
    [-1., +1., -1.], [+1., +1., -1.], [+1., +1., +1.], [-1., +1., +1.],
])
# 12 structural edges: 4 bottom, 4 top, 4 vertical.
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7))


def corners(dims: Tensor) -> Tensor:
    """dims [B,3] = (W, D, H) metres -> [B,9,3] object-frame points."""
    if dims.ndim != 2 or dims.shape[1] != 3:
        raise ValueError('dims must be [B,3] as (W, D, H)')
    if (dims <= 0).any():
        raise ValueError('dimensions must be positive metres')
    w, d, h = dims[:, 0], dims[:, 1], dims[:, 2]
    half = torch.stack((w, h, d), -1) * .5              # (x=W, y=H, z=D)
    pts = SIGNS.to(dims) * half[:, None]                 # [B,8,3]
    return torch.cat((pts, torch.zeros_like(pts[:, :1])), 1)


def skew(v: Tensor) -> Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack((
        torch.stack((z, -v[..., 2], v[..., 1]), -1),
        torch.stack((v[..., 2], z, -v[..., 0]), -1),
        torch.stack((-v[..., 1], v[..., 0], z), -1)), -2)


def exp_so3(r: Tensor) -> Tensor:
    """Rodrigues on [...,3] -> [...,3,3]; stable near zero."""
    theta = torch.linalg.vector_norm(r, dim=-1, keepdim=True)
    small = theta < 1e-8
    safe = torch.where(small, torch.ones_like(theta), theta)
    a = torch.where(small, torch.ones_like(theta), torch.sin(safe) / safe)
    b = torch.where(small, .5 * torch.ones_like(theta), (1 - torch.cos(safe)) / safe ** 2)
    S = skew(r)
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand(*r.shape[:-1], 3, 3)
    return eye + a[..., None] * S + b[..., None] * (S @ S)


def project(K: Tensor, R: Tensor, t: Tensor, X: Tensor) -> tuple[Tensor, Tensor]:
    """X [B,N,3] object -> (uv [B,N,2], depth [B,N]). Depth is not clamped."""
    cam = X @ R.transpose(-1, -2) + t[:, None]
    z = cam[..., 2]
    uv = (cam @ K.transpose(-1, -2))[..., :2] / z.clamp(min=1e-6)[..., None]
    return uv, z


def project_jacobian(K: Tensor, R: Tensor, t: Tensor, X: Tensor) -> Tensor:
    """d(uv)/d(dr, dt) at identity increment: [B,N,2,6].

    For ``X_cam = Exp(dr) @ R @ X + t + dt`` the rotation derivative is
    ``-skew(R @ X)`` -- not ``-skew(R @ X + t)``. The finite-difference test
    in tests/ fails if that distinction is broken.
    """
    RX = X @ R.transpose(-1, -2)
    cam = RX + t[:, None]
    x, y, z = cam[..., 0], cam[..., 1], cam[..., 2]
    zi = 1. / z.clamp(min=1e-6)
    fx, fy = K[:, 0, 0][:, None], K[:, 1, 1][:, None]
    s = K[:, 0, 1][:, None]
    # d uv / d cam
    duv = torch.zeros(*cam.shape[:-1], 2, 3, dtype=cam.dtype, device=cam.device)
    duv[..., 0, 0] = fx * zi
    duv[..., 0, 1] = s * zi
    duv[..., 0, 2] = -(fx * x + s * y) * zi ** 2
    duv[..., 1, 1] = fy * zi
    duv[..., 1, 2] = -fy * y * zi ** 2
    dcam = torch.cat((-skew(RX), torch.eye(3, dtype=cam.dtype, device=cam.device)
                      .expand(*cam.shape[:-1], 3, 3)), -1)          # [B,N,3,6]
    return duv @ dcam


def yaw(k: Tensor | int, dtype=torch.float32, device=None) -> Tensor:
    """Rotation by k*90 degrees about the +y (up) axis."""
    k = int(k) % 4
    c, s = [(1., 0.), (0., 1.), (-1., 0.), (0., -1.)][k]
    return torch.tensor([[c, 0., s], [0., 1., 0.], [-s, 0., c]], dtype=dtype, device=device)


def symmetry_permutations(order: int) -> list[list[int]]:
    """Corner index permutations for upright yaw symmetry of the given order.

    ``corners[perm_g]`` equals ``yaw(g) @ corners`` for a body whose footprint
    admits that rotation. Order 4 additionally requires W == D; the caller is
    responsible for that check -- shape alone never promotes an order.
    """
    if order not in (1, 2, 4):
        raise ValueError('supported upright yaw orders are 1, 2 and 4')
    step = 4 // order
    out = []
    for g in range(order):
        Rg = yaw(g * step, dtype=torch.float64)
        rotated = SIGNS.double() @ Rg.T
        perm = []
        for row in rotated:
            hit = ((SIGNS.double() - row).abs().max(-1).values < 1e-9).nonzero()
            if len(hit) != 1:
                raise ValueError('yaw symmetry does not permute the corner set')
            perm.append(int(hit[0]))
        out.append(perm + [8])
    return out


def reframe_pose(R: Tensor, A: Tensor) -> Tensor:
    """Same-centre rotation-only reframe: X_old = A @ X_new  ->  R_new = R_old @ A.

    Translation is unchanged and must not be touched. This function refuses a
    non-rotation A rather than silently absorbing a reflection or a scale.
    """
    if A.shape[-2:] != (3, 3):
        raise ValueError('A must be 3x3')
    eye = torch.eye(3, dtype=A.dtype, device=A.device)
    if not torch.allclose(A @ A.transpose(-1, -2), eye, atol=1e-6):
        raise ValueError('A is not orthonormal; a reframe must be a rotation')
    if not torch.allclose(torch.linalg.det(A), torch.ones_like(torch.linalg.det(A)), atol=1e-6):
        raise ValueError('A must be a proper rotation (det = +1)')
    return R @ A
