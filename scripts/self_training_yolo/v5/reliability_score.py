"""V5 — GT 없이 pseudo frame 의 신뢰도를 하나의 점수로 묶는다.

FILTER_SEPARABILITY 가 보여준 것: 신호 하나하나는 약하고(AUC 0.60~0.74), 임계로
자르면 정보를 거의 못 쓴다.  그렇다고 임계를 세게 하면 멀쩡한 라벨을 수백 개 버린다.

그래서 V5 는 **자르지 않는다.**  모든 accepted frame 을 남기되 신뢰도가 높은 프레임을
더 자주 노출한다.

## 점수는 학습된 것이 아니다

FIXED · UNSUPERVISED · MONOTONIC · RANK-FUSION.

    1  Day / Night 안에서 각 신호를 mid-rank 백분위로 바꾼다
           u = (r - 0.5) / N,   0 < u < 1,   동점은 average rank
       good-high 신호는 u, good-low 신호는 1 - u 를 quality 로 쓴다.
    2  코너 신뢰도    R_corner_i = (q_kp * q_remove * q_flip)^(1/3)
    3  keypoint 블록  R_kp_frame  = geometric mean over usable corners
    4  frame 블록     R_frame_geom = (q_box * q_reproj * q_remove * q_flip)^(1/4)
    5  최종           R_total = sqrt(R_frame_geom * R_kp_frame)

가중치는 전부 동일하다.  0.5 / 0.5 는 PAPER_EVAL 최적화값이 아니라 **두 정보원을
동등하게 묶는 구조적 정의**다.  AUC 를 보고 0.4 / 0.3 같은 값을 만들지 않는다.

Day 와 Night 를 따로 정규화하는 이유는 조명에 따라 raw confidence·residual 의 스케일이
달라서다.  절대값을 곱하지 않는다.

centroid(kp8)는 frame 신뢰도에서 제외한다 — cuboid semantic corner 와 성격이 다르다.

이 모듈은 GT · PAPER_EVAL · gross 라벨 · 20 px 임계를 **읽지 않는다** (테스트로 강제).
`probability` 라고 부르지 않는다 — pseudo-label reliability score 다.
"""

from __future__ import annotations

import numpy as np

N_CORNERS = 8
EPSILON = 1e-9

# 신호 방향.  True = 값이 클수록 좋다.
FRAME_SIGNALS = {
    "box_conf": True,
    "s_reproj": False,
    "s_remove": False,
    "s_flip": False,
}
CORNER_SIGNALS = {
    "kp_conf": True,
    "r_remove": False,
    "r_flip": False,
}


def mid_rank_quality(values, higher_is_better: bool) -> np.ndarray:
    """mid-rank 경험 백분위.  0 < u < 1 이라 곱해도 0 이 되지 않는다.

    NaN/inf 는 **가장 나쁜 쪽**으로 몰아 넣는다 — 신호가 없다고 좋게 봐줄 이유가 없다.
    """

    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    worst = -np.inf if higher_is_better else np.inf
    filled = np.where(finite, array, worst)

    order = np.argsort(filled, kind="mergesort")
    ranks = np.empty(len(filled), dtype=float)
    ranks[order] = np.arange(1, len(filled) + 1, dtype=float)
    # 동점은 average rank
    unique, inverse = np.unique(filled, return_inverse=True)
    sums = np.zeros(len(unique))
    counts = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    np.add.at(counts, inverse, 1.0)
    ranks = (sums / counts)[inverse]

    u = (ranks - 0.5) / len(filled)
    return u if higher_is_better else 1.0 - u


def geometric_mean(values) -> float:
    array = np.asarray([v for v in values
                        if v is not None and np.isfinite(v) and v > 0], dtype=float)
    if array.size == 0:
        return float("nan")
    return float(np.exp(np.log(array).mean()))


def score_condition(records: list[dict]) -> list[dict]:
    """한 condition(Day 또는 Night) 안에서 점수를 낸다.

    `records` 의 각 항목은 frame 신호 4 개와 코너 신호 3 개 × 8 을 갖는다.
    """

    if not records:
        return []

    # ── frame 신호를 이 condition 안에서 정규화 ──────────────────────
    frame_quality: dict[str, np.ndarray] = {}
    for name, higher in FRAME_SIGNALS.items():
        frame_quality[name] = mid_rank_quality([r[name] for r in records], higher)

    # ── 코너 신호는 (frame, corner) 를 모두 펼쳐 한 번에 정규화 ──────
    #    코너끼리 같은 척도가 되어야 프레임 간 비교가 성립한다.
    corner_quality: dict[str, np.ndarray] = {}
    for name, higher in CORNER_SIGNALS.items():
        flat = [r[name][i] for r in records for i in range(N_CORNERS)]
        corner_quality[name] = mid_rank_quality(flat, higher).reshape(
            len(records), N_CORNERS)

    scored = []
    for index, record in enumerate(records):
        q_frame = {name: float(frame_quality[name][index]) for name in FRAME_SIGNALS}
        r_frame_geom = geometric_mean(q_frame.values())

        corners = []
        for corner in range(N_CORNERS):
            available = []
            for name in CORNER_SIGNALS:
                raw = record[name][corner]
                if raw is not None and np.isfinite(raw):
                    available.append(float(corner_quality[name][index, corner]))
            # 쓸 수 있는 신호가 1 개뿐인 코너는 frame 집계에서 뺀다 (§7).
            corners.append(geometric_mean(available) if len(available) >= 2
                           else float("nan"))
        r_kp_frame = geometric_mean(corners)

        r_total = (float(np.sqrt(r_frame_geom * r_kp_frame))
                   if np.isfinite(r_frame_geom) and np.isfinite(r_kp_frame)
                   else float("nan"))
        scored.append({
            **record,
            **{f"q_{name}": q_frame[name] for name in FRAME_SIGNALS},
            "R_corner": corners,
            "R_kp_frame": r_kp_frame,
            "R_frame_geom": r_frame_geom,
            "R_total": r_total,
        })
    return scored


def score_pool(records: list[dict]) -> list[dict]:
    """condition 별로 나눠 점수를 내고 다시 합친다."""

    out: list[dict] = []
    conditions = sorted({r["condition"] for r in records})
    for condition in conditions:
        out += score_condition([r for r in records if r["condition"] == condition])
    return out


def largest_remainder_allocation(weights, total: int) -> list[int]:
    """결정론적 배분.  무작위 추첨을 쓰지 않는다 (§12).

    각 항목에 먼저 1 을 주고, 남은 몫만 weight 에 비례해 나눈다.  소수부는
    largest remainder 로, 동점은 인덱스 순서로 깬다 — 실행마다 같은 답이 나온다.
    """

    count = len(weights)
    if total < count:
        raise ValueError(f"TOTAL_BELOW_UNIQUE_COUNT: {total} < {count}")
    remaining = total - count
    array = np.asarray(weights, dtype=float)
    array = np.where(np.isfinite(array) & (array > 0), array, EPSILON)
    share = array / array.sum() * remaining
    base = np.floor(share).astype(int)
    leftover = remaining - int(base.sum())
    if leftover:
        fractional = share - base
        order = sorted(range(count), key=lambda i: (-fractional[i], i))
        for i in order[:leftover]:
            base[i] += 1
    return (base + 1).tolist()
