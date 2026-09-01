"""YOLO teacher 예측에 대한 무차원 geometry consistency score 3종.

전부 projected cuboid diagonal 로 나눈 **무차원** 값이라 해상도·거리·시야에
무관하다.  YOLO 가 640 이라는 이유로 새 절대 px threshold 를 만들지 않는다.

    reprojection_score        모든 valid keypoint 로 PnP → 전체 재투영 오차
    keypoint_removal_score    한 점씩 빼고 PnP → 빠진 점만 재투영해 비교
    flip_score                좌우반전 추론을 되돌린 뒤 원본과 대응 거리

논문 표기 (reader-facing).  "LOO" 는 코드 내부 식별자로만 남기고 문서에는 쓰지 않는다.

    reprojection_score      -> Reprojection consistency
    keypoint_removal_score  -> single-keypoint-removal reprojection consistency
    flip_score              -> horizontal-flip keypoint consistency

GT 누수 방지 (§10):
  * 3D geometry 는 OBJECT_GEOMETRY_REGISTRY 에서만 온다.
  * GT axis assignment / GT pose / per-frame dimensions_m 를 받지 않는다.
    이 모듈의 어떤 함수도 그런 인자를 갖지 않는다 (테스트로 강제).
  * W/D 가 모호하면 registry 가 허용하는 두 hypothesis 를 각각 풀고
    **score 의 최소값**만 취한다.  이긴 hypothesis 의 pose 는 반환은 하되
    pseudo 6D GT 로 저장하지 않는다 (호출부 계약).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training"))

from pnp_solver import make_pallet_keypoints_3d_isaac  # noqa: E402

N_CORNERS = 8
MIN_PNP_POINTS = 4
# canonical_filters.filter_C 와 같은 하한.  4 점으로 PnP + 1 점 검증.
MIN_REMOVAL_POINTS = 5


def projected_diagonal(points_2d: np.ndarray) -> float:
    """8 점의 최대 pairwise 거리.  canonical_filters 와 같은 정의."""

    points = np.asarray(points_2d, dtype=np.float64)[:N_CORNERS]
    if len(points) < 2:
        return 0.0
    differences = points[:, None, :] - points[None, :, :]
    return float(np.linalg.norm(differences, axis=-1).max())


def cuboid_keypoints_3d(width: float, depth: float, height: float) -> np.ndarray:
    """camera-facing 0123 규약의 9 점.  프로젝트 정본 함수를 그대로 쓴다."""

    return make_pallet_keypoints_3d_isaac(width=width, depth=depth, height=height)


def registry_hypotheses(dimensions_xyz: dict) -> tuple[tuple[str, np.ndarray], ...]:
    """registry 의 X/Y/Z 로부터 W/D 배정 hypothesis 를 만든다.

    Y 는 height(top-to-bottom) 로 고정이고, 남은 두 footprint 축 중 어느 쪽이
    카메라를 향한 width 인지가 미결이다.  그래서 hypothesis 는 정확히 둘이다.
    GT 를 보고 고르지 않는다 — 호출부가 score 최소값만 쓴다.
    """

    x = float(dimensions_xyz["x"])
    y = float(dimensions_xyz["y"])
    z = float(dimensions_xyz["z"])
    hypotheses = [("XZ", cuboid_keypoints_3d(width=x, depth=z, height=y))]
    if not np.isclose(x, z):
        hypotheses.append(("ZX", cuboid_keypoints_3d(width=z, depth=x, height=y)))
    return tuple(hypotheses)


def _solve(points_3d: np.ndarray, points_2d: np.ndarray, camera_matrix: np.ndarray):
    if len(points_3d) < MIN_PNP_POINTS:
        return None
    flags = cv2.SOLVEPNP_EPNP if len(points_3d) >= 5 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(
        np.ascontiguousarray(points_3d, dtype=np.float64),
        np.ascontiguousarray(points_2d, dtype=np.float64),
        np.asarray(camera_matrix, dtype=np.float64),
        None,
        flags=flags,
    )
    if not ok or not np.isfinite(tvec).all():
        return None
    return rvec, tvec


def _project(points_3d: np.ndarray, rvec, tvec, camera_matrix: np.ndarray) -> np.ndarray:
    projected, _ = cv2.projectPoints(
        np.ascontiguousarray(points_3d, dtype=np.float64),
        rvec, tvec, np.asarray(camera_matrix, dtype=np.float64), None,
    )
    return projected.reshape(-1, 2)


@dataclass(frozen=True)
class HypothesisScores:
    """한 W/D hypothesis 에서 나온 세 score.  pose 는 진단용이다."""

    name: str
    projected_diagonal_px: float
    reprojection: float
    keypoint_removal: float
    flip: float | None
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None


def _hypothesis_scores(
    name: str,
    keypoints_3d: np.ndarray,
    keypoints_2d: np.ndarray,
    valid: np.ndarray,
    camera_matrix: np.ndarray,
    flip_keypoints_2d: np.ndarray | None,
    flip_valid: np.ndarray | None,
) -> HypothesisScores:
    infinite = HypothesisScores(name, 0.0, float("inf"), float("inf"), None)
    indices = np.flatnonzero(valid[:N_CORNERS])
    if len(indices) < MIN_PNP_POINTS:
        return infinite

    solved = _solve(keypoints_3d[indices], keypoints_2d[indices], camera_matrix)
    if solved is None:
        return infinite
    rvec, tvec = solved

    reprojected = _project(keypoints_3d[:N_CORNERS], rvec, tvec, camera_matrix)
    diagonal = projected_diagonal(reprojected)
    if not np.isfinite(diagonal) or diagonal < 1e-6:
        return infinite

    # ── Reprojection consistency ────────────────────────────────────────
    residuals = np.linalg.norm(
        reprojected[indices] - keypoints_2d[indices], axis=1
    )
    s_reproj = float(np.median(residuals)) / diagonal

    # ── single-keypoint-removal reprojection consistency ────────────────
    # canonical_filters.filter_C 의 수식 그대로다: 한 점을 빼고 나머지로 PnP 를
    # 풀고, **빠진 점** 을 다시 투영해 원 예측과 비교한다.  전체 재투영과 달리
    # 자기 자신이 해에 기여하지 않으므로 구조 붕괴에 민감하다.
    if len(indices) >= MIN_REMOVAL_POINTS:
        removal_errors: list[float] = []
        for position, left_out in enumerate(indices):
            remaining = np.delete(indices, position)
            if len(remaining) < MIN_PNP_POINTS:
                continue
            partial = _solve(
                keypoints_3d[remaining], keypoints_2d[remaining], camera_matrix
            )
            if partial is None:
                continue
            predicted = _project(
                keypoints_3d[left_out].reshape(1, 3), partial[0], partial[1], camera_matrix
            )[0]
            removal_errors.append(
                float(np.linalg.norm(predicted - keypoints_2d[left_out]))
            )
        s_remove = (
            float(np.median(removal_errors)) / diagonal if removal_errors else float("inf")
        )
    else:
        s_remove = float("inf")

    # ── horizontal-flip keypoint consistency ────────────────────────────
    s_flip: float | None = None
    if flip_keypoints_2d is not None and flip_valid is not None:
        both = valid[:N_CORNERS] & flip_valid[:N_CORNERS]
        matched = np.flatnonzero(both)
        if len(matched) >= 3:
            distances = np.linalg.norm(
                keypoints_2d[matched] - flip_keypoints_2d[matched], axis=1
            )
            s_flip = float(np.median(distances)) / diagonal
        else:
            s_flip = float("inf")

    return HypothesisScores(
        name=name,
        projected_diagonal_px=diagonal,
        reprojection=s_reproj,
        keypoint_removal=s_remove,
        flip=s_flip,
        rvec=rvec,
        tvec=tvec,
    )


def geometry_scores(
    keypoints_2d: np.ndarray,
    valid: np.ndarray,
    camera_matrix: np.ndarray,
    dimensions_xyz: dict,
    flip_keypoints_2d: np.ndarray | None = None,
    flip_valid: np.ndarray | None = None,
) -> dict:
    """registry hypothesis 전부를 풀고 score 별 최소값을 돌려준다.

    인자에 GT pose 도 GT axis assignment 도 없다.  들어오는 것은 teacher 의 2D
    예측, 그 유효성, 카메라 행렬, 그리고 registry 치수뿐이다.
    """

    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if flip_keypoints_2d is not None:
        flip_keypoints_2d = np.asarray(flip_keypoints_2d, dtype=np.float64)
    if flip_valid is not None:
        flip_valid = np.asarray(flip_valid, dtype=bool)

    scored = [
        _hypothesis_scores(
            name, keypoints_3d, keypoints_2d, valid, camera_matrix,
            flip_keypoints_2d, flip_valid,
        )
        for name, keypoints_3d in registry_hypotheses(dimensions_xyz)
    ]

    def best(attribute: str) -> float:
        values = [
            getattr(item, attribute) for item in scored
            if getattr(item, attribute) is not None
        ]
        return min(values) if values else float("inf")

    flip_values = [item.flip for item in scored if item.flip is not None]
    return {
        "s_reproj": best("reprojection"),
        "s_remove": best("keypoint_removal"),
        "s_flip": min(flip_values) if flip_values else None,
        "projected_diagonal_px": max(
            (item.projected_diagonal_px for item in scored), default=0.0
        ),
        "hypotheses": [
            {
                "name": item.name,
                "s_reproj": item.reprojection,
                "s_remove": item.keypoint_removal,
                "s_flip": item.flip,
                "projected_diagonal_px": item.projected_diagonal_px,
            }
            for item in scored
        ],
    }
