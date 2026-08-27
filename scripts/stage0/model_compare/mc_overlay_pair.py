"""목표 vs 도전자 나란히 오버레이 — 정본 161 에서.

왼쪽 = 목표(타깃 파렛트를 학습에서 본 모델), 오른쪽 = 도전자(한 장도 안 본 모델).
두 모델 다 YOLO 라 전처리가 같다(PAD=100 reflect, imgsz 640) — DOPE 비교와 달리
비대칭이 없어 같은 그림 위에서 그대로 비교된다.

그리기 규약은 `wood_gt_eval.save_overlay` 를 그대로 따른다.  따로 그리면 두 그림이
다른 것을 말하게 된다.

    green   GT projected_cuboid
    blue    예측 keypoint (번호 = camera-facing 0123 인덱스)
    red     예측 pose 로 되쏜 cuboid

패널에 찍는 글자는 ASCII 로만 쓴다 — `cv2.putText` 는 한글을 `???` 로 그린다.

★ 프레임은 고르지 않는다.  "잘 나온 것" 을 뽑으면 그림이 표와 다른 말을 한다.
층(stratum)마다 정해진 규칙으로 자동 선정하고, 각 패널에 **왜 뽑혔는지**를 적는다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0/real_eval", "scripts/annotate", "challenge"):
    sys.path.insert(0, os.path.join(ROOT, sub))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2                       # noqa: E402
import re_metrics as RM          # noqa: E402
import mc_geom as MG             # noqa: E402
import mc_frames as MF           # noqa: E402

# camera-facing 0123 cuboid edges — 0~3 앞면 / 4~7 뒷면 / 앞뒤 연결.
# `internet_pallet_infer` 에서 가져오려 했으나 그 모듈이 지금 없는 의존을 물고 있다.
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]

OUT = os.path.join(ROOT, "data/pallet/results/model_compare/overlay_target_vs_challenger")
TARGET = ("yolo26n_synth", "TARGET  n-SYN74K  (saw target pallet 35,914 imgs)")
CHALLENGER = ("yolo26n_paper_generic_v1", "CHALLENGER  n-GEN40K  (saw target 0 imgs)")
GREEN, BLUE, RED, WHITE = (0, 255, 0), (255, 80, 0), (0, 0, 255), (255, 255, 255)


def load(name):
    payload = json.load(open(os.path.join(
        ROOT, "data/pallet/results/model_compare", f"kps_{name}.json")))
    return {(e["set"], e["fid"]): e for e in payload["frames"]}


def measure(entry, label, name):
    """프레임 하나에 대한 예측 8점 + pose + 지표."""
    truth = MG.gt_of(label)
    points = MG.points_of(entry, name)
    row = MG.metrics(points, truth)
    pose = MG.solve(points, truth) if row["pnp_ok"] else None
    proj = None
    if pose is not None:
        row["yaw"] = RM.yaw_error(pose[0], truth["R"])
        rvec, _ = cv2.Rodrigues(pose[0])
        proj, _ = cv2.projectPoints(truth["model"].astype(np.float64), rvec,
                                    pose[1].astype(np.float64), truth["K"], None)
        proj = proj.reshape(-1, 2)
    else:
        row["yaw"] = np.nan
    return row, points, proj, truth


def draw_edges(img, pts, col, th):
    ok = np.isfinite(pts[:, 0])
    for a, b in EDGES:
        if a < len(pts) and b < len(pts) and ok[a] and ok[b]:
            cv2.line(img, (int(pts[a, 0]), int(pts[a, 1])),
                     (int(pts[b, 0]), int(pts[b, 1])), col, th, cv2.LINE_AA)


def panel(image_path, row, points, proj, truth, header, note):
    img = cv2.imread(image_path)
    draw_edges(img, truth["gt8"], GREEN, 2)
    if proj is not None:
        draw_edges(img, proj, RED, 2)
    for i in range(8):
        if np.isfinite(points[i, 0]):
            p = (int(points[i, 0]), int(points[i, 1]))
            cv2.circle(img, p, 5, BLUE, -1, cv2.LINE_AA)
            cv2.putText(img, str(i), (p[0] + 5, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1, cv2.LINE_AA)
    cm = "n/a" if not np.isfinite(row["corner_med"]) else f"{row['corner_med']:.1f}px"
    if row["pnp_ok"]:
        detail = f"R={row['R']:.1f}deg  yaw={row['yaw']:.1f}deg  t={row['t']:.3f}m"
    else:
        detail = "PnP FAILED"
    cv2.rectangle(img, (0, 0), (img.shape[1], 62), (0, 0, 0), -1)
    for i, text in enumerate([header, f"corner_med={cm}  {detail}", note]):
        cv2.putText(img, text, (6, 17 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, WHITE, 1, cv2.LINE_AA)
    return img


def contact_sheet(index, cols=2, panel_w=760):
    """낱장을 한 장으로 — 층 6 개를 한눈에 훑을 수 있어야 판단이 된다.

    좌우쌍이 이미 가로로 붙어 있어 원본 폭이 크다.  `panel_w` 로 줄이되 헤더는
    축소 후에도 읽혀야 하므로, 줄인 그림 위에 **다시** 한 줄을 얹는다.
    """
    if not index:
        return
    tiles = []
    for item in index:
        img = cv2.imread(os.path.join(OUT, item["file"]))
        scale = panel_w / img.shape[1]
        img = cv2.resize(img, (panel_w, int(img.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
        t, c = item["target"], item["challenger"]

        def brief(v):
            if not v["pnp_ok"]:
                return "PnP FAILED"
            return f"{v['corner_med']:.1f}px R{v['R']:.0f} yaw{v['yaw']:.0f}"
        bar = np.zeros((26, panel_w, 3), np.uint8)
        cv2.putText(bar, f"{item['stratum']}  |  TARGET {brief(t)}"
                         f"   vs   CHALLENGER {brief(c)}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, img]))

    height = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((height - t.shape[0], t.shape[1], 3), np.uint8)])
             for t in tiles]
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)

    caption = np.zeros((30, sheet.shape[1], 3), np.uint8)
    cv2.putText(caption, "LEFT half of each tile = TARGET n-SYN74K (saw target) | "
                         "RIGHT half = CHALLENGER n-GEN40K (saw target 0) | "
                         "green=GT  blue=pred kp  red=pred pose reproj",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    sheet = np.vstack([caption, sheet])
    path = os.path.join(OUT, "CONTACT_SHEET.png")
    cv2.imwrite(path, sheet)
    print(f"  contact sheet {sheet.shape[1]}x{sheet.shape[0]} -> "
          f"{os.path.basename(path)}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    dumps = {TARGET[0]: load(TARGET[0]), CHALLENGER[0]: load(CHALLENGER[0])}

    records = []
    for key, sealed, jp, ip, label in MF.frames():
        fid = os.path.splitext(os.path.basename(jp))[0]
        e_t = dumps[TARGET[0]].get((key, fid))
        e_c = dumps[CHALLENGER[0]].get((key, fid))
        if e_t is None or e_c is None:
            continue
        t = measure(e_t, label, TARGET[0])
        c = measure(e_c, label, CHALLENGER[0])
        records.append({"key": key, "sealed": sealed, "fid": fid, "ip": ip,
                        "t": t, "c": c})

    # 층화 선정 — 각 층에서 규칙으로 하나씩. 고르지 않는다.
    def gap(r):
        if not r["t"][0]["pnp_ok"] or not r["c"][0]["pnp_ok"]:
            return -1
        return r["c"][0]["corner_med"] - r["t"][0]["corner_med"]

    both = [r for r in records if r["t"][0]["pnp_ok"] and r["c"][0]["pnp_ok"]]
    op = [r for r in both if not r["sealed"]]      # OPEN 56  — 표에서 91% 로 붙는 구간
    se = [r for r in both if r["sealed"]]          # SEALED 105 — 53% 로 벌어지는 구간
    only_t = [r for r in records if r["t"][0]["pnp_ok"] and not r["c"][0]["pnp_ok"]]
    mid = lambda v: sorted(v, key=gap)[len(v) // 2] if v else None  # noqa: E731
    strata = [
        ("A_open_typical", "OPEN: median-gap frame (the regime where they tie)",
         mid(op)),
        ("B_open_worst", "OPEN: largest gap (worst case in the close regime)",
         max(op, key=gap) if op else None),
        ("C_sealed_typical", "SEALED: median-gap frame (the regime that breaks)",
         mid(se)),
        ("D_sealed_worst", "SEALED: largest gap (tail collapse)",
         max(se, key=gap) if se else None),
        ("E_challenger_wins", "COUNTER-EXAMPLE: challenger beats target",
         min(both, key=gap) if both else None),
        ("F_only_target", "DETECTION GAP: target solves PnP, challenger fails",
         only_t[len(only_t) // 2] if only_t else None),
    ]

    index = []
    for tag, why, rec in strata:
        if rec is None:
            print(f"  {tag}: 해당 프레임 없음 -> 건너뜀", flush=True)
            continue
        left = panel(rec["ip"], *rec["t"], f"{TARGET[1]}", why)
        right = panel(rec["ip"], *rec["c"], f"{CHALLENGER[1]}", why)
        pair = np.hstack([left, np.full((left.shape[0], 4, 3), 255, np.uint8),
                          right])
        name = f"{tag}__{rec['key']}__{rec['fid']}.png"
        cv2.imwrite(os.path.join(OUT, name), pair)
        index.append({"file": name, "stratum": tag, "why": why,
                      "set": rec["key"], "sealed": rec["sealed"],
                      "fid": rec["fid"],
                      "target": {k: rec["t"][0].get(k) for k in
                                 ("pnp_ok", "corner_med", "R", "yaw", "t")},
                      "challenger": {k: rec["c"][0].get(k) for k in
                                     ("pnp_ok", "corner_med", "R", "yaw", "t")}})
        print(f"  {tag:20} {rec['key']}/{rec['fid']} -> {name}", flush=True)

    contact_sheet(index)
    json.dump({"target": TARGET, "challenger": CHALLENGER,
               "selection": "층화 자동 선정 — 사람이 고르지 않았다",
               "legend": "green=GT  blue=pred keypoint  red=pred pose 재투영",
               "panels": index},
              open(os.path.join(OUT, "INDEX.json"), "w"), indent=1, default=str)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
