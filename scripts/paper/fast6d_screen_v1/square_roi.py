"""정사각 context crop 과 정확한 역매핑.

기각된 방식과 다른 점: tight bbox 를 short-side 400 으로 늘리면 edge-on 팔레트가
극단 wide strip 이 되어 rear/front localization 이 무너졌다.  여기서는 **정사각** 을
자르므로 물체의 종횡비가 바뀌지 않는다.

이미지 밖은 잘라내지 않고 padding 으로 채운다 — 그래야 crop 크기가 항상 정사각이고
역매핑이 단순한 affine 으로 남는다.
"""

from __future__ import annotations

import cv2
import numpy as np


def square_context(bbox, image_width: int, image_height: int,
                   ratio: float, out_size: int) -> dict:
    """bbox 중심의 정사각 context.  이미지 밖으로 나가도 자르지 않는다."""

    left, top, right, bottom = [float(v) for v in bbox]
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    side = ratio * max(right - left, bottom - top)
    side = max(side, 1.0)
    origin_x = center_x - side / 2.0
    origin_y = center_y - side / 2.0
    scale = out_size / side
    return {
        "origin_x": origin_x, "origin_y": origin_y, "side": side,
        "out_size": int(out_size), "scale_x": scale, "scale_y": scale,
        "image_width": int(image_width), "image_height": int(image_height),
    }


def forward_map(points: np.ndarray, context: dict) -> np.ndarray:
    """원본 좌표 -> crop 좌표."""

    points = np.asarray(points, np.float64)
    out = np.empty_like(points)
    out[..., 0] = (points[..., 0] - context["origin_x"]) * context["scale_x"]
    out[..., 1] = (points[..., 1] - context["origin_y"]) * context["scale_y"]
    return out


def inverse_map(points: np.ndarray, context: dict) -> np.ndarray:
    """crop 좌표 -> 원본 좌표."""

    points = np.asarray(points, np.float64)
    out = np.empty_like(points)
    out[..., 0] = points[..., 0] / context["scale_x"] + context["origin_x"]
    out[..., 1] = points[..., 1] / context["scale_y"] + context["origin_y"]
    return out


def crop_square(image: np.ndarray, context: dict, border) -> np.ndarray:
    """context 를 실제로 잘라 out_size x out_size 로 만든다.

    이미지 밖 영역은 `border` 방식으로 채운다.  crop 이 이미지를 벗어날 수 있으므로
    먼저 필요한 만큼 패딩한 뒤 정수 좌표로 잘라내고, 소수점 오차 없이 resize 한다.
    """

    height, width = image.shape[:2]
    x0 = int(np.floor(context["origin_x"]))
    y0 = int(np.floor(context["origin_y"]))
    side = int(np.ceil(context["side"])) + 1
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x0 + side - width)
    pad_bottom = max(0, y0 + side - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                                   border, value=(127, 127, 127))
    x0 += pad_left
    y0 += pad_top
    patch = image[y0:y0 + side, x0:x0 + side]
    if patch.shape[0] != side or patch.shape[1] != side:
        return None
    # 정수 격자로 잘랐으므로 origin 이 소수점만큼 어긋난다.  역매핑은 origin 을
    # 그대로 쓰므로, resize 는 side(정수) 기준이 아니라 context["side"] 기준으로
    # 스케일이 맞아야 한다.  아래 warpAffine 이 그 소수점을 정확히 처리한다.
    matrix = np.array([[context["scale_x"], 0.0,
                        -context["origin_x"] * context["scale_x"]],
                       [0.0, context["scale_y"],
                        -context["origin_y"] * context["scale_y"]]], np.float64)
    matrix[0, 2] += pad_left * context["scale_x"]
    matrix[1, 2] += pad_top * context["scale_y"]
    return cv2.warpAffine(image, matrix,
                          (context["out_size"], context["out_size"]),
                          flags=cv2.INTER_LINEAR, borderMode=border,
                          borderValue=(127, 127, 127))
