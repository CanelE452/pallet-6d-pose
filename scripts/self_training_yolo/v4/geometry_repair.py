"""V4 — 중간 신뢰도 pseudo-keypoint 를 버리지 않고 기하로 **복원**한다.

V1(frame-level 거부) · V2(per-keypoint mask) · V3(true-ignore)는 전부 나쁜 keypoint 를
**없앨** 수만 있었다.  없애는 것으로는 R0 보다 나은 좌표가 생기지 않는다.

V4 의 주장: 신뢰할 수 있는 고신뢰 코너들이 이미 물체의 투영 기하를 결정한다.  그
합의로부터 불확실한 코너의 2D 위치를 **재구성**하면 teacher 원본보다 정확할 수 있다.

## 계층 (§4)

    c_i >= 0.95            HIGH-CONFIDENCE   기하 검사도 통과해야 raw 좌표를 쓴다
    0.5 <= c_i < 0.95      REPAIR-CANDIDATE  복원 대상
    c_i < 0.5              UNRELIABLE        true-ignore

`c_i` 를 calibrated probability 라고 부르지 않는다 — teacher keypoint confidence 다.

## Anchor (§8)

candidate i 의 anchor 는 **자기 자신을 뺀** 다른 코너 중

    c_j >= 0.95  AND  removal consistency PASS  AND  flip consistency PASS

anchor 가 6 개 미만이면 복원하지 않는다.  6 은 기존 `min_valid_corners` 를 그대로
재사용한 값이지 새 tuning parameter 가 아니다.

## Hypothesis 안전성 (§10)

registry 의 deployment-valid W/D hypothesis 를 **전부** 푼다.  어느 것이 맞는지 GT 로
고르지 않는다.

    valid hypothesis 0 개           -> true-ignore
    1 개                            -> 그 투영을 복원 좌표로
    2 개 이상이고 서로 동의         -> 평균을 복원 좌표로
    2 개 이상이고 불일치            -> true-ignore

동의 판정은 후보 투영들의 최대 pairwise 거리 / projected diagonal <= tau_reproj.
즉 **"어느 hypothesis 가 맞는가" 를 고르지 않고, "어느 것을 써도 이 2D 위치에는
동의하는가" 만 쓴다.**

## 저장하지 않는 것 (§11)

pseudo 6D pose · 승리 hypothesis · GT 축 · rotation/translation.  PnP 는 2D target 을
다듬기 위한 latent operator 일 뿐이다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo" / "v2"))

# V1/V2 모듈은 읽기만 한다.  복사하면 갈라진다.
from pseudo_label_filters import (  # noqa: E402
    N_CORNERS,
    _project,
    _solve,
    projected_diagonal,
    registry_hypotheses,
)
from keypoint_scores import per_keypoint_scores  # noqa: E402

CENTROID = 8
MIN_ANCHORS = 6          # 기존 min_valid_corners 재사용

HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
REPAIR_CANDIDATE = "REPAIR_CANDIDATE"
UNRELIABLE = "UNRELIABLE"

REPAIR_OK = "REPAIRED"
REPAIR_NO_ANCHORS = "NO_ANCHORS"
REPAIR_NO_VALID_HYPOTHESIS = "NO_VALID_HYPOTHESIS"
REPAIR_HYPOTHESIS_DISAGREE = "HYPOTHESIS_DISAGREE"
REPAIR_AMBIGUOUS_VIEW = "AMBIGUOUS_VIEW"
REPAIR_OUT_OF_IMAGE = "OUT_OF_IMAGE"


def tier(confidence: float, kp_high_conf: float, kp_floor: float) -> str:
    if confidence >= kp_high_conf:
        return HIGH_CONFIDENCE
    if confidence >= kp_floor:
        return REPAIR_CANDIDATE
    return UNRELIABLE


def repair_keypoints(
    keypoints_2d,
    keypoint_conf,
    camera_matrix,
    dimensions_xyz: dict,
    scores: dict,
    image_size: tuple[float, float],
    kp_high_conf: float = 0.95,
    kp_floor: float = 0.5,
    tau_reproj: float = 0.05,
) -> dict:
    """코너별 계층·복원 결과.  GT 를 인자로 받지 않는다.

    `scores` 는 V2/V3 의 `per_keypoint_scores` 출력이다 (removal/flip 신뢰도와 q).
    """

    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float64)
    confidence = np.nan_to_num(np.asarray(keypoint_conf, dtype=np.float64), nan=0.0)
    width, height = image_size

    tiers = [tier(float(confidence[i]), kp_high_conf, kp_floor)
             for i in range(len(confidence))]
    geometry_ok = list(scores["keep_corner"])
    ambiguous = bool(scores["ambiguous_view"])

    # anchor 는 고신뢰 + 기하 통과 코너다.
    anchor_flags = np.array(
        [tiers[i] == HIGH_CONFIDENCE and geometry_ok[i] for i in range(N_CORNERS)])

    diagonal = projected_diagonal(keypoints_2d[:N_CORNERS])
    repaired = np.full((N_CORNERS, 2), np.nan)
    status = [None] * N_CORNERS
    displacement = np.full(N_CORNERS, np.nan)

    for index in range(N_CORNERS):
        if tiers[index] != REPAIR_CANDIDATE:
            continue
        if ambiguous:
            status[index] = REPAIR_AMBIGUOUS_VIEW
            continue
        anchors = np.flatnonzero(anchor_flags & (np.arange(N_CORNERS) != index))
        if len(anchors) < MIN_ANCHORS:
            status[index] = REPAIR_NO_ANCHORS
            continue
        if not np.isfinite(diagonal) or diagonal <= 1e-6:
            status[index] = REPAIR_NO_VALID_HYPOTHESIS
            continue

        projections = []
        for _, keypoints_3d in registry_hypotheses(dimensions_xyz):
            solved = _solve(keypoints_3d[anchors], keypoints_2d[anchors], camera_matrix)
            if solved is None:
                continue
            reprojected = _project(keypoints_3d[anchors], solved[0], solved[1],
                                   camera_matrix)
            residual = float(np.median(
                np.linalg.norm(reprojected - keypoints_2d[anchors], axis=1))) / diagonal
            if residual > tau_reproj:
                continue
            point = _project(keypoints_3d[index].reshape(1, 3), solved[0], solved[1],
                             camera_matrix)[0]
            if np.isfinite(point).all():
                projections.append(point)

        if not projections:
            status[index] = REPAIR_NO_VALID_HYPOTHESIS
            continue
        if len(projections) > 1:
            stacked = np.asarray(projections)
            spread = float(np.linalg.norm(
                stacked[:, None, :] - stacked[None, :, :], axis=-1).max()) / diagonal
            if spread > tau_reproj:
                status[index] = REPAIR_HYPOTHESIS_DISAGREE
                continue
        target = np.mean(np.asarray(projections), axis=0)
        # 화면 밖으로 나간 복원은 clipping 으로 감추지 않는다 — 버린다 (§20).
        if not (0.0 <= target[0] < width and 0.0 <= target[1] < height):
            status[index] = REPAIR_OUT_OF_IMAGE
            continue
        repaired[index] = target
        displacement[index] = float(np.linalg.norm(target - keypoints_2d[index]))
        status[index] = REPAIR_OK

    return {
        "tier": tiers,
        "anchor": anchor_flags.tolist(),
        "n_anchor": int(np.count_nonzero(anchor_flags)),
        "geometry_ok": geometry_ok,
        "ambiguous_view": ambiguous,
        "repair_status": status,
        "repaired_xy": repaired.tolist(),
        "displacement_px": displacement.tolist(),
        "displacement_normalised": (displacement / diagonal).tolist()
        if np.isfinite(diagonal) and diagonal > 1e-6 else [None] * N_CORNERS,
        "projected_diagonal_px": float(diagonal),
        "centroid_high_confidence": bool(
            len(confidence) > CENTROID
            and confidence[CENTROID] >= kp_high_conf
            and (scores.get("r_flip_centroid") is not None
                 and scores["r_flip_centroid"] <= tau_reproj)),
    }


def supervision_plan(scores: dict, repair: dict, arm: str) -> tuple[list[str], list]:
    """arm 별로 코너마다 무엇을 줄지 정한다.

    돌려주는 것: (per-corner action, per-corner target xy 또는 None)
    action ∈ {RAW, REPAIRED, IGNORE}
    """

    actions: list[str] = []
    targets: list = []
    for index in range(N_CORNERS):
        if repair["ambiguous_view"]:
            actions.append("IGNORE")
            targets.append(None)
            continue
        high = (repair["tier"][index] == HIGH_CONFIDENCE
                and repair["geometry_ok"][index])
        if high:
            actions.append("RAW")
            targets.append(None)          # raw teacher 좌표를 쓴다
            continue
        if arm == "V4C_GEOMETRY_REPAIR" and repair["repair_status"][index] == REPAIR_OK:
            actions.append("REPAIRED")
            targets.append(repair["repaired_xy"][index])
            continue
        actions.append("IGNORE")
        targets.append(None)
    if arm == "V4A_BOX_ONLY":
        actions = ["IGNORE"] * N_CORNERS
        targets = [None] * N_CORNERS
    return actions, targets
