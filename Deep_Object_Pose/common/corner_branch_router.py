"""Raw-logit proposal decoders and the coordinate-level branch router.

The first screen decoded the proposal with sigmoid(Q) and the belief decoder's
absolute 0.3 threshold, but Q was trained with a spatial softmax objective, so
its absolute level carries no meaning at all — only its spatial ranking does.
Everything here therefore reads raw Q: no sigmoid, no threshold, no "not
detected" branch.

The router works on coordinates, not maps.  The map-level gate collapsed to
g ~ 1e-9 and was never exercised, and blending two coordinates that disagree by
tens of pixels would land between two peaks rather than on either, so the choice
is hard.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

TEMPERATURE = 0.1
WINDOW = 7
N_CORNERS = 8
ROUTER_MARGIN_PX = 3.0


# ============================================================================
# raw-Q coordinate decoders
# ============================================================================
def _scaled(point: np.ndarray, scale: tuple[float, float]) -> list[float]:
    return [float(point[0] * scale[0]), float(point[1] * scale[1])]


def decode_argmax(logits: np.ndarray, scale: tuple[float, float]
                  ) -> list[list[float]]:
    """Top-1 cell of the raw logits.  Always returns eight coordinates."""
    points = []
    for corner in range(N_CORNERS):
        y, x = np.unravel_index(int(np.argmax(logits[corner])),
                                logits[corner].shape)
        points.append(_scaled(np.array([x, y], float), scale))
    return points


def decode_local(logits: np.ndarray, scale: tuple[float, float]
                 ) -> list[list[float]]:
    """Softmax over a 7x7 window of raw logits centred on the argmax."""
    height, width = logits.shape[-2:]
    radius = WINDOW // 2
    points = []
    for corner in range(N_CORNERS):
        heat = logits[corner].astype(np.float64)
        cy, cx = np.unravel_index(int(np.argmax(heat)), heat.shape)
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        window = heat[y0:y1, x0:x1] / TEMPERATURE
        weights = np.exp(window - window.max())
        weights /= weights.sum()
        ys, xs = np.mgrid[y0:y1, x0:x1]
        points.append(_scaled(
            np.array([(weights * xs).sum(), (weights * ys).sum()]), scale))
    return points


def decode_dsnt(logits: np.ndarray, scale: tuple[float, float]
                ) -> list[list[float]]:
    """Full-map softmax expectation — the training objective's own read-out."""
    points = []
    for corner in range(N_CORNERS):
        flat = logits[corner].astype(np.float64).reshape(-1) / TEMPERATURE
        weights = np.exp(flat - flat.max())
        weights /= weights.sum()
        ys, xs = np.mgrid[0:logits.shape[-2], 0:logits.shape[-1]]
        points.append(_scaled(
            np.array([(weights * xs.reshape(-1)).sum(),
                      (weights * ys.reshape(-1)).sum()]), scale))
    return points


DECODERS = {"argmax": decode_argmax, "local": decode_local, "dsnt": decode_dsnt}
PRIMARY_DECODER = "local"


# ============================================================================
# oracle routing
# ============================================================================
def route_oracle(error_base: np.ndarray, error_proposal: np.ndarray,
                 margin: float = 0.0) -> np.ndarray:
    """True where the proposal is taken.  With a margin, ties keep the base."""
    if margin <= 0.0:
        return error_proposal < error_base
    return (error_proposal + margin) < error_base


# ============================================================================
# learned router
# ============================================================================
def spatial_entropy(heat: np.ndarray, temperature: float = TEMPERATURE) -> float:
    flat = heat.astype(np.float64).reshape(-1) / temperature
    weights = np.exp(flat - flat.max())
    weights /= weights.sum()
    return float(-(weights * np.log(weights + 1e-12)).sum())


def top_two_gap(heat: np.ndarray) -> float:
    flat = np.sort(heat.astype(np.float64).reshape(-1))
    return float(flat[-1] - flat[-2])


def local_sharpness(heat: np.ndarray, radius: int = 1) -> float:
    """Peak height relative to the ring just outside the peak neighbourhood."""
    cy, cx = np.unravel_index(int(np.argmax(heat)), heat.shape)
    peak = float(heat[cy, cx])
    y0, y1 = max(0, cy - radius - 1), min(heat.shape[0], cy + radius + 2)
    x0, x1 = max(0, cx - radius - 1), min(heat.shape[1], cx + radius + 2)
    patch = heat[y0:y1, x0:x1].astype(np.float64)
    ring = patch.sum() - heat[max(0, cy - radius):cy + radius + 1,
                              max(0, cx - radius):cx + radius + 1].sum()
    count = max(patch.size - (2 * radius + 1) ** 2, 1)
    return float(peak - ring / count)


class CoordinateRouter(nn.Module):
    """Tiny MLP choosing base or proposal for one corner.  Hard choice only."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def router_labels(error_base: np.ndarray, error_proposal: np.ndarray,
                  margin: float = ROUTER_MARGIN_PX
                  ) -> tuple[np.ndarray, np.ndarray]:
    """(label, usable).  Ties are dropped from the loss and default to base."""
    take_proposal = (error_proposal + margin) < error_base
    take_base = (error_base + margin) < error_proposal
    usable = take_proposal | take_base
    return take_proposal.astype(np.float32), usable
