"""Corner-Incident Geometry Module: three edge-role lines to one corner.

The module has no parameters.  Each corner is the least-squares intersection of
the three lines whose roles are incident to it, and the incidence comes from the
existing topology rather than from anything learned -- a learned incidence would
let the network choose which edges explain a corner, which is the assumption
under test.
"""
from __future__ import annotations

import pathlib
import sys

import torch

_COMMON = pathlib.Path(__file__).resolve().parent
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

import instance_edge_topology as IET  # noqa: E402

EPSILON = 1e-4
N_CORNERS = 8


def incidence_table(topology: dict | None = None) -> list[list[int]]:
    topology = topology or IET.build_topology()
    return IET.incidence_lists(topology, "O12")


def lines_from_segments(centre: torch.Tensor, direction: torch.Tensor):
    """Normal form: n . x = rho, with n perpendicular to the segment."""
    normal = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)
    rho = (normal * centre).sum(dim=-1)
    return normal, rho


def solve_corners(centre: torch.Tensor, direction: torch.Tensor,
                  incidence: list[list[int]], epsilon: float = EPSILON):
    """(B, 12, 2) segments -> (B, 8, 2) corners, differentiable.

    Ridge term epsilon keeps a near-parallel triple from exploding instead of
    dropping it, so a bad frame degrades rather than producing NaN.
    """
    normal, rho = lines_from_segments(centre, direction)
    corners, residuals, conditions = [], [], []
    eye = torch.eye(2, device=centre.device, dtype=centre.dtype)
    for corner in range(len(incidence)):
        index = incidence[corner]
        n = normal[:, index]                      # (B, 3, 2)
        r = rho[:, index]                         # (B, 3)
        ata = torch.einsum("bij,bik->bjk", n, n) + epsilon * eye
        atb = torch.einsum("bij,bi->bj", n, r)
        point = torch.linalg.solve(ata, atb.unsqueeze(-1)).squeeze(-1)
        corners.append(point)
        residuals.append((torch.einsum("bij,bj->bi", n, point) - r).norm(dim=-1))
        eigenvalues = torch.linalg.eigvalsh(ata)
        conditions.append((eigenvalues[:, -1] / eigenvalues[:, 0].clamp_min(1e-12)).sqrt())
    return (torch.stack(corners, 1), torch.stack(residuals, 1),
            torch.stack(conditions, 1))


def render_proposals(corners: torch.Tensor, grid: int = 50, sigma: float = 2.0):
    """Corners -> (B, 8, grid, grid) Gaussians at A1's own corner sigma."""
    device = corners.device
    axis = torch.arange(grid, device=device, dtype=corners.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    dx = xx[None, None] - corners[..., 0][..., None, None]
    dy = yy[None, None] - corners[..., 1][..., None, None]
    return torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))
