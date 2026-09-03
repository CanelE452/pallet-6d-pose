"""V2 — keypoint 단위 신뢰도와 시점 모호성.  GT 를 읽지 않는다.

V1 은 frame 하나에 score 세 개를 매겨 통째로 ACCEPT/REJECT 했다.  그 결과 pool
272 장 중 13 장(4.8%)만 바뀌었고, 정작 near-square 시점은 학습셋에서 사라졌다
(`_docs/archive/paper_pre_final_20260903/diagnostics/CORNER_REGRESSION_CAUSES.md`).

V2 는 같은 기하 신호를 **코너마다** 매긴다.

    r_remove_i   코너 i 를 빼고 나머지로 PnP 를 푼 뒤 i 를 재투영한 잔차
    r_flip_i     수평 반전 추론을 되돌려 맞춘 뒤의 코너 i 잔차
    KEEP_i       kp_conf_i >= 0.5  AND  r_remove_i <= tau  AND  r_flip_i <= tau

둘 다 projected cuboid diagonal 로 나눈 **무차원** 값이라 해상도·거리에 불변이다.

centroid(index 8)는 removal-PnP 코너가 아니므로 같은 규칙을 억지로 적용하지 않는다
(§7).  `kp_conf_8` 과 flip 잔차만 본다.

시점 모호성:

    q = min(projected_width, projected_depth) / max(projected_width, projected_depth)

`q -> 1` 이면 투영이 정사각이라 수직축 90 도 회전이 시각적 대칭이 된다.  min/max 이므로
**90 도 순열에 불변**이다 (unit test 로 강제).

이 모듈은 teacher 의 2D 예측 · 유효성 · 카메라 행렬 · registry 치수만 받는다.
GT pose 도 GT axis assignment 도 인자에 없다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo"))

# PnP 풀이·투영·3D 규약은 V1 모듈에서 **그대로 가져온다**.  복사하면 갈라진다 —
# 실제로 한 번 `_solve` 의 flag 조건을 뒤집어 옮겨 5 점에서 DLT 오류가 났다.
# V1 파일은 읽기만 한다 (수정 금지).
from pseudo_label_filters import (  # noqa: E402
    N_CORNERS,
    _project,
    _solve,
    projected_diagonal,
    registry_hypotheses,
)

# OpenCV 의 non-planar PnP 는 4 점에서 DLT 오류를 낸다 (EPNP 는 5 점부터).
# V1 의 MIN_PNP_POINTS=4 는 removal 경로가 항상 5 점 이상을 넘겨서 드러나지 않았다.
# V2 는 2 pass 라 신뢰 집합이 작아질 수 있으므로 여기서 명시한다.
MIN_REMOVAL_FIT_POINTS = 5

CENTROID = 8
FLIP_PAIRS = ((0, 1), (3, 2), (4, 5), (7, 6))
# gt[i] <- pred[YAW90[i]].  AXIS_FAILURES.md 에서 Hungarian 이 독립적으로 낸 순열.
YAW90 = (1, 5, 6, 2, 0, 4, 7, 3, 8)


def ambiguity_q(keypoints_2d) -> float:
    """투영된 footprint 의 정사각성.  1.0 이면 90 도 회전이 시각적 대칭이다.

    teacher 예측만으로 계산된다.  GT 를 쓰지 않는다.
    """

    points = np.asarray(keypoints_2d, dtype=np.float64)
    if points.shape[0] < 6 or not np.isfinite(points[[0, 1, 4, 5]]).all():
        return float("nan")
    # 상단면 네 코너: 0 근좌 · 1 근우 · 4 원좌 · 5 원우.
    # **마주보는 두 변의 평균**을 쓴다.  한쪽 변만 쓰면 원근 때문에 90 도 순열에서
    # 값이 달라진다 — unit test 가 그걸 잡았다 (0.450 vs 0.358).
    width = 0.5 * (float(np.linalg.norm(points[1] - points[0]))
                   + float(np.linalg.norm(points[5] - points[4])))
    depth = 0.5 * (float(np.linalg.norm(points[4] - points[0]))
                   + float(np.linalg.norm(points[5] - points[1])))
    larger = max(width, depth)
    if larger <= 1e-9:
        return float("nan")
    return min(width, depth) / larger


def _removal_residuals(
    keypoints_3d: np.ndarray,
    keypoints_2d: np.ndarray,
    valid: np.ndarray,
    camera_matrix: np.ndarray,
    diagonal: float,
) -> np.ndarray:
    """leave-two-out.  한 코너가 망가지면 그 코너가 다른 코너의 해까지 오염시킨다.

    코너 하나만 빼고 재면(leave-one-out) 한 점을 89 px 흔들었을 때 8 개 중 5 개가
    같이 탈락한다 — 오염된 점이 나머지 모두의 PnP 적합에 들어가기 때문이다
    (unit test 로 확인).

    그래서 코너 i 의 잔차를 "i 를 뺀 나머지 중 **하나를 더 빼도 되는** 적합 가운데
    가장 관대한 것" 으로 정의한다.  6 점이 6-DoF 를 여전히 구속하므로 느슨해지지
    않으면서, 다른 한 점의 오염에는 견딘다.
    """

    residuals = np.full(N_CORNERS, np.inf)
    indices = np.flatnonzero(valid[:N_CORNERS])
    for left_out in indices:
        others = indices[indices != left_out]
        if len(others) < MIN_REMOVAL_FIT_POINTS:
            continue
        best = np.inf
        # 하나도 더 빼지 않는 경우 + 다른 한 점을 더 빼는 경우 전부.
        subsets = [others]
        if len(others) > MIN_REMOVAL_FIT_POINTS:
            subsets += [others[others != drop] for drop in others]
        for subset in subsets:
            solved = _solve(keypoints_3d[subset], keypoints_2d[subset], camera_matrix)
            if solved is None:
                continue
            predicted = _project(
                keypoints_3d[left_out].reshape(1, 3), solved[0], solved[1], camera_matrix
            )[0]
            best = min(
                best,
                float(np.linalg.norm(predicted - keypoints_2d[left_out])) / diagonal,
            )
        residuals[left_out] = best
    return residuals


def per_keypoint_scores(
    keypoints_2d,
    keypoint_conf,
    camera_matrix,
    dimensions_xyz: dict,
    flip_keypoints_2d=None,
    flip_conf=None,
    kp_conf_threshold: float = 0.5,
    remove_threshold: float = 0.05,
    flip_threshold: float = 0.05,
    ambiguity_threshold: float = 0.75,
) -> dict:
    """코너별 KEEP 판정과 시점 모호성.  GT 인자는 존재하지 않는다.

    registry hypothesis 를 모두 풀고, 코너마다 **가장 관대한** hypothesis 의 잔차를
    쓴다 — W/D 배정이 미해결이므로 어느 한쪽을 임의로 고르지 않는다.
    """

    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float64)
    keypoint_conf = np.asarray(keypoint_conf, dtype=np.float64)
    valid_conf = np.nan_to_num(keypoint_conf, nan=0.0) >= kp_conf_threshold

    diagonal = projected_diagonal(keypoints_2d[:N_CORNERS])
    remove = np.full(N_CORNERS, np.inf)
    if np.isfinite(diagonal) and diagonal > 1e-6:
        for _, keypoints_3d in registry_hypotheses(dimensions_xyz):
            candidate = _removal_residuals(
                keypoints_3d, keypoints_2d, valid_conf, camera_matrix, diagonal
            )
            remove = np.minimum(remove, candidate)

    flip = np.full(len(keypoints_2d), np.inf)
    if flip_keypoints_2d is not None and np.isfinite(diagonal) and diagonal > 1e-6:
        flipped = np.asarray(flip_keypoints_2d, dtype=np.float64)
        flip_valid = (
            np.nan_to_num(np.asarray(flip_conf, dtype=np.float64), nan=0.0)
            >= kp_conf_threshold
            if flip_conf is not None
            else np.ones(len(flipped), dtype=bool)
        )
        limit = min(len(flipped), len(keypoints_2d))
        for index in range(limit):
            if not flip_valid[index]:
                continue
            flip[index] = (
                float(np.linalg.norm(keypoints_2d[index] - flipped[index])) / diagonal
            )

    keep = (
        valid_conf[:N_CORNERS]
        & (remove <= remove_threshold)
        & (flip[:N_CORNERS] <= flip_threshold)
    )

    # centroid 는 removal-PnP 코너가 아니다.  conf 와 flip 만 본다 (§7).
    centroid_keep = bool(
        len(keypoint_conf) > CENTROID
        and valid_conf[CENTROID]
        and flip[CENTROID] <= flip_threshold
    )

    q = ambiguity_q(keypoints_2d)
    ambiguous = bool(np.isfinite(q) and q >= ambiguity_threshold)

    return {
        "r_remove": remove.tolist(),
        "r_flip": flip[:N_CORNERS].tolist(),
        "r_flip_centroid": float(flip[CENTROID]) if len(flip) > CENTROID else None,
        "valid_conf": valid_conf.tolist(),
        "keep_corner": keep.tolist(),
        "keep_centroid": centroid_keep,
        "n_keep_corner": int(np.count_nonzero(keep)),
        "projected_diagonal_px": float(diagonal),
        "q": float(q) if np.isfinite(q) else None,
        "ambiguous_view": ambiguous,
    }


def visibility_vector(scores: dict, ambiguity_aware: bool) -> list[int]:
    """KEEP 판정을 Ultralytics pose label 의 visibility 로 옮긴다.

    `ambiguity_aware` 면 모호한 시점의 semantic corner 를 통째로 끈다.  프레임은
    남는다 — box 와 centroid supervision 은 그대로다 (§9).

    주의: visibility 0 은 순수한 ignore 가 아니다.  `kpt_shape[-1] == 3` 이라
    keypoint objectness 가 '보이지 않음' 을 학습한다
    (`KEYPOINT_MASK_CONTRACT.json`).
    """

    keep = list(scores["keep_corner"])
    if ambiguity_aware and scores["ambiguous_view"]:
        keep = [False] * N_CORNERS
    visibility = [2 if flag else 0 for flag in keep]
    visibility.append(2 if scores["keep_centroid"] else 0)
    return visibility
