"""S4 게이트 — square crop 역매핑이 정확한지 검증한다.

    python3 -m pytest scripts/paper/fast6d_screen_v1/test_square_roi_inverse_map.py -q

original -> square crop -> resize -> inverse map 왕복 오차가 1e-4 px 를 넘으면
S4 를 실행하지 않는다(lock §16).
"""

from __future__ import annotations

import numpy as np
import pytest

from square_roi import forward_map, inverse_map, square_context


CASES = [
    # (image w, h, bbox, 설명)
    (640, 480, (100.0, 120.0, 300.0, 260.0), "중앙 정상 박스"),
    (640, 480, (0.0, 0.0, 120.0, 90.0), "좌상단 경계"),
    (640, 480, (520.0, 400.0, 640.0, 480.0), "우하단 경계"),
    (640, 480, (10.0, 200.0, 630.0, 250.0), "극단 wide, edge-on 팔레트"),
    (640, 480, (300.0, 10.0, 340.0, 470.0), "극단 tall"),
    (1280, 720, (400.0, 300.0, 900.0, 600.0), "다른 해상도"),
]


@pytest.mark.parametrize("width,height,bbox,label", CASES)
def test_round_trip_is_exact(width, height, bbox, label):
    context = square_context(bbox, width, height, ratio=1.25, out_size=400)
    rng = np.random.default_rng(0)
    points = rng.uniform([0, 0], [width, height], size=(200, 2))
    mapped = forward_map(points, context)
    back = inverse_map(mapped, context)
    error = float(np.abs(back - points).max())
    assert error < 1e-4, f"{label}: round-trip error {error:.3e} px"


def test_square_stays_square():
    """정사각이라 종횡비가 변하지 않는다 — 기각된 strip crop 과의 차이."""

    context = square_context((10.0, 200.0, 630.0, 250.0), 640, 480,
                             ratio=1.25, out_size=400)
    assert abs(context["side"] - context["side"]) < 1e-9
    assert context["scale_x"] == pytest.approx(context["scale_y"])


def test_context_is_larger_than_the_box():
    bbox = (100.0, 120.0, 300.0, 260.0)
    context = square_context(bbox, 640, 480, ratio=1.25, out_size=400)
    longest = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    assert context["side"] == pytest.approx(1.25 * longest)
