"""모든 계열을 한 표에 — YOLO / DOPE / Hough arm / detection AP.

세 산출물을 합친다.  따로 보면 "어느 방법이 낫다" 로 읽히지만, 한 줄로 세우면
경계가 방법이 아니라 **타깃 파렛트를 학습에서 봤는가**에 있다는 게 바로 보인다.
정본 161 이 전부 사용자 파렛트라 그렇다.

    MODEL_COMPARE_AUC.json              pose 지표 + detection AP
    HYBRID_POINT_LINE_PER_FRAME.csv     Hough arm (프레임별 -> 여기서 재집계)

빈 칸은 `-` 다.  0 이 아니라 **그 자로 재지 않았다**는 뜻이다 — Hough arm 은 yaw/AP
가 없고, DOPE 는 score_4kp 가 box conf 와 달라 AP 곡선에서 제외된다.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts/stage0/real_eval"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json                      # noqa: E402
import re_metrics as RM          # noqa: E402
import mc_geom as MG             # noqa: E402
import mc_frames as MF           # noqa: E402

OUT = os.path.join(ROOT, "data/pallet/results/model_compare")

# 맨 위에 고정하는 기준선.  논문 트랙이 따라잡아야 할 줄이라 정렬에서 빼고
# "목표" 로 박는다.  단 이 모델은 타깃 파렛트를 35,914 장 봤다 — 타깃을 못 본
# 모델에게 공정한 목표가 아니라는 단서를 표 안에 같이 남긴다.
TARGET_KEY = "yolo26n_synth"

# real FT 한 계열은 이 표에서 뺀다.  기준선이 FT 없는 모델이라 같은 조건끼리
# 세우는 것이 목적이고, FT 행은 감독량이 한 단계 더 얹혀 있어 줄을 흐린다.
# 파생인 `+Hough` 도 base 가 FT 면 같이 뺀다.
EXCLUDE_FT = True

# (내부키, 표시명, 타깃을 봤나)
MODELS = [
    ("yolo26m_ft",               "m-SYN74K + realFT",    True),
    ("yolo26n_ft",               "n-SYN74K + realFT",    True),
    ("yolo26n_synth",            "n-SYN74K  타깃+G38",   True),
    ("Y0E",                      "n-G38 +반복9K",        False),
    ("yolo26n_paper_generic_v1", "n-GEN40K  generic만",  False),
    ("YN",                       "n-G38 +neg9K",         False),
    ("yolo26n_broad40k_5ep",     "n-GEN40K  5ep만",      False),
    ("FINAL40K_seed1_step18000", "DOPE-GEN40K 18k step", False),
    ("FINAL40K_seed1",           "DOPE-GEN40K 25k step", False),
    ("FINAL40K_seed1_step12000", "DOPE-GEN40K 12k step", False),
    ("FINAL40K_seed1_step6000",  "DOPE-GEN40K  6k step", False),
]
# Hough arm — HYBRID CSV 의 열 prefix.  네 번째 항목은 detection 지표를 물려받을
# base 모델이다: Direct-Hough 는 **pose 만 바꾸고 검출은 건드리지 않는다**(같은
# 프레임의 같은 점에서 출발한다).  그래서 AP / AUROC / FPR95 는 base 와 같은 값이고,
# 다시 재는 것이 아니라 물려받는 것이 맞다.  오라클은 base 가 없다.
HOUGH = [
    ("P3", "n-SYN74K + realFT  +Hough", True,  "yolo26n_ft"),
    ("P2", "n-SYN74K  타깃+G38 +Hough", True,  "yolo26n_synth"),
    ("P1", "n-GEN40K  5ep만    +Hough", False, "yolo26n_broad40k_5ep"),
    ("P0", "DOPE-GEN40K        +Hough", False, "FINAL40K_seed1"),
    ("O2", "[오라클] GT theta B",       None,  None),
    ("O1", "[오라클] GT theta A",       None,  None),
]


def hough_rows():
    """Hough arm 을 pose 표와 같은 자로 재집계한다."""
    dia = {}
    for key, _sealed, jp, _ip, label in MF.frames():
        fid = os.path.splitext(os.path.basename(jp))[0]
        dia[(key, fid)] = RM.model_diameter(MG.gt_of(label)["model"])
    rows = list(csv.DictReader(
        open(os.path.join(OUT, "HYBRID_POINT_LINE_PER_FRAME.csv"))))

    def agg(sel, arm):
        add_n, adds_n, R, t, iou, corner, ok = [], [], [], [], [], [], 0
        for r in sel:
            d = dia.get((r["set"], r["fid"]))
            if d is None:
                continue

            def f(k):
                try:
                    return float(r[f"{arm}_{k}"])
                except (ValueError, KeyError):
                    return np.nan
            if r[f"{arm}_ok"] in ("1", "1.0", "True") and np.isfinite(f("add")):
                ok += 1
                add_n.append(f("add") / d)
                adds_n.append(f("adds") / d)
                R.append(f("R"))
                t.append(f("t"))
                iou.append(f("iou"))
            else:
                add_n.append(np.inf)
                adds_n.append(np.inf)
            if np.isfinite(f("corner")):
                corner.append(f("corner"))
        med = lambda v: round(float(np.median(v)), 4) if v else None  # noqa: E731
        return {"pnp_rate": round(ok / max(len(sel), 1), 4),
                "corner_px_med": med(corner), "R_deg_med": med(R),
                "yaw_deg_med": None, "t_m_med": med(t), "IoU3D_med": med(iou),
                "ADD_S_AUC": round(RM.pose_auc(adds_n, 1.0, 0.1), 4)}

    open_rows = [r for r in rows if r["sealed"] in ("False", "0", "")]
    seal_rows = [r for r in rows if r["sealed"] in ("True", "1")]
    return {arm: {"OPEN_56": agg(open_rows, arm),
                  "SEALED_105": agg(seal_rows, arm),
                  "ALL_161": agg(rows, arm)} for arm, _n, _t, _b in HOUGH}


HEAD = [("타깃", 5), ("model", 27), ("pnp↑", 7), ("corner↓", 8), ("R med↓", 8),
        ("yaw med↓", 9), ("t med↓", 8), ("IoU3D↑", 8), ("AUC-open↑", 10),
        ("AUC-seal↑", 10), ("AUC-all↑", 9), ("AP↑", 8), ("AUROC↑", 8),
        ("FPR95↓", 8)]


def line(cells):
    return "".join(f"{c:>{w}}" if i else f"{c:<{w}}"
                   for i, ((_h, w), c) in enumerate(zip(HEAD, cells)))


def fmt(v, dec):
    return "-" if v is None else f"{v:.{dec}f}"


def main():
    report = json.load(open(os.path.join(OUT, "MODEL_COMPARE_AUC.json")))
    models, ap = report["models"], report["detection"]["per_model"]
    hough = hough_rows()

    def is_ft(label):
        return "realFT" in label

    rows, target = [], None
    for key, label, seen in MODELS:
        if key not in models:
            continue
        if EXCLUDE_FT and is_ft(label):
            continue
        entry = (models[key], ap.get(key), label, seen)
        if key == TARGET_KEY:
            target = entry
        else:
            rows.append(entry)
    for arm, label, seen, base in HOUGH:
        if EXCLUDE_FT and is_ft(label):
            continue
        rows.append((hough[arm], ap.get(base) if base else None, label, seen))
    rows.sort(key=lambda r: -(r[0]["ALL_161"]["ADD_S_AUC"]))

    print("↑ 클수록 좋음   ↓ 작을수록 좋음   `-` = 그 자로 재지 않음(0 아님)")
    print("타깃: ✔ = 평가셋 파렛트를 학습에서 봤다   ✗ = 못 봤다   ORC = 오라클(배포불가)\n")
    header = line([h for h, _w in HEAD])
    print(header)
    print("─" * len(header))

    def emit(block, det, label, seen):
        a, o, s_ = block["ALL_161"], block["OPEN_56"], block["SEALED_105"]
        mark = "ORC" if seen is None else ("✔" if seen else "✗")
        print(line([mark, label,
                    fmt(a["pnp_rate"], 3), fmt(a["corner_px_med"], 2),
                    fmt(a["R_deg_med"], 2), fmt(a["yaw_deg_med"], 2),
                    fmt(a["t_m_med"], 3), fmt(a["IoU3D_med"], 3),
                    fmt(o["ADD_S_AUC"], 4), fmt(s_["ADD_S_AUC"], 4),
                    fmt(a["ADD_S_AUC"], 4),
                    fmt(det["AP"] if det else None, 4),
                    fmt(det["AUROC"] if det else None, 4),
                    fmt(det["FPR_at_TPR95"] if det else None, 4)]))

    if target is not None:
        print(f"{'[목표]':<5}{'← FT 없이 여기까지 가는 것이 기준선':<27}")
        emit(*target)
        print("═" * len(header))
    for block, det, label, seen in rows:
        emit(block, det, label, seen)


if __name__ == "__main__":
    sys.exit(main())
