"""Frozen scientific and geometry constants for the v1 pilot."""

from __future__ import annotations

EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
IDENTITY = (0, 1, 2, 3, 4, 5, 6, 7, 8)
C2_YAW = (5, 4, 7, 6, 1, 0, 3, 2, 8)
R0_SHA256 = "970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7"
GRID_SIZE = 24
THETA_BINS = 36
RHO_BINS = 65
N_MODES = 3
MAX_SHIFT_DIAGONAL_FRACTION = 0.01
POINT_SIGMA_DIAGONAL_FRACTION = 0.005
LINE_SIGMA_DIAGONAL_FRACTION = 0.005


def permutations_for_order(order: int) -> list[list[int]]:
    if order == 1:
        return [list(IDENTITY)]
    if order == 2:
        return [list(IDENTITY), list(C2_YAW)]
    raise ValueError("v1 data contract permits only confirmed C1/C2 assets")
