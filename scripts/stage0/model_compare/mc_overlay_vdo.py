"""목표 vs 도전자 — 실내 리프터 영상(vdoframes)에서 나란히.

`mc_overlay_pair` 와 같은 질문이지만 셋이 다르다.  vdoframes 에는 **GT 가 없다**
(json 0 개).  그래서 여기서 낼 수 있는 것과 없는 것을 먼저 못박는다.

    낼 수 있다   검출 여부 · box conf · cuboid 형상 · 두 모델이 같은 곳을 잡았는가
    낼 수 없다   corner px · R / yaw 오차 · ADD · IoU3D   (전부 GT 가 필요하다)

정본 161 은 야외·야간 중심인데 이 영상은 실내 창고 · 리프터 시점이다.  도메인이
다르므로 여기 결과를 정본 순위의 확인이나 반박으로 읽지 말 것 — 별개의 관찰이다.

★ 프레임은 고르지 않는다.  전수를 훑어 층으로 나눈 뒤 규칙으로 뽑는다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2                       # noqa: E402
import mc_dump_yolo as MDY       # noqa: E402

SRC = os.path.join(ROOT, "data/pallet/raw_data/vdoframes")
OUT = os.path.join(ROOT, "data/pallet/results/model_compare/overlay_vdo_target_vs_challenger")
PAD, IMGSZ, CONF = 100, 640, 0.25
TARGET = ("yolo26n_synth", "TARGET  n-SYN74K  (saw target pallet 35,914)")
CHALLENGER = ("yolo26n_paper_generic_v1", "CHALLENGER  n-GEN40K  (saw target 0)")
EXTRA = {"yolo26n_paper_generic_v1":
         "challenge/yolo_pose_one_model/runs_paper/"
         "yolo26n_paper_generic_v1_seed42/weights/best.pt"}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
YELLOW, BLUE, WHITE = (0, 255, 255), (255, 80, 0), (255, 255, 255)


def frames(step=8):
    """전수를 다 돌리면 오래 걸린다 — 일정 간격으로 훑되 간격을 기록한다."""
    names = sorted(n for n in os.listdir(SRC) if n.endswith(".png"))
    return [(n, os.path.join(SRC, n)) for n in names[::step]]


def predict(model, path):
    image = cv2.imread(path)
    padded = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
    result = model.predict(padded, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    if result.boxes is None or not len(result.boxes):
        return None, 0.0, image
    i = int(np.argmax(result.boxes.conf.cpu().numpy()))
    kps = result.keypoints.xy.cpu().numpy()[i] - PAD
    return kps, float(result.boxes.conf.cpu().numpy()[i]), image


def panel(image, kps, conf, header, note):
    img = image.copy()
    if kps is not None:
        ok = np.isfinite(kps[:, 0])
        for a, b in EDGES:
            if ok[a] and ok[b]:
                cv2.line(img, (int(kps[a, 0]), int(kps[a, 1])),
                         (int(kps[b, 0]), int(kps[b, 1])), YELLOW, 2, cv2.LINE_AA)
        for i in range(9):
            if ok[i]:
                p = (int(kps[i, 0]), int(kps[i, 1]))
                cv2.circle(img, p, 4, BLUE, -1, cv2.LINE_AA)
                cv2.putText(img, str(i), (p[0] + 5, p[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLUE, 1, cv2.LINE_AA)
    state = f"DETECTED  conf={conf:.2f}" if kps is not None else "NO DETECTION"
    cv2.rectangle(img, (0, 0), (img.shape[1], 62), (0, 0, 0), -1)
    for i, text in enumerate([header, state, note]):
        cv2.putText(img, text, (6, 17 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, WHITE, 1, cv2.LINE_AA)
    return img


def sheet(items, cols=2, panel_w=760):
    tiles = []
    for img, cap in items:
        scale = panel_w / img.shape[1]
        img = cv2.resize(img, (panel_w, int(img.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
        bar = np.zeros((26, panel_w, 3), np.uint8)
        cv2.putText(bar, cap, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    WHITE, 1, cv2.LINE_AA)
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
    out = np.vstack(rows)
    cap = np.zeros((30, out.shape[1], 3), np.uint8)
    cv2.putText(cap, "LEFT of each tile = TARGET n-SYN74K | RIGHT = CHALLENGER "
                     "n-GEN40K | yellow=pred cuboid  blue=pred kp | NO GT in this "
                     "set: detection/shape only",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return np.vstack([cap, out])


def main(step=8):
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    models = dict(MDY.MODELS)
    models.update(EXTRA)
    net_t = YOLO(os.path.join(ROOT, models[TARGET[0]]), task="pose")
    net_c = YOLO(os.path.join(ROOT, models[CHALLENGER[0]]), task="pose")

    rows = frames(step)
    print(f"프레임 {len(rows)}장 (전체 2,362 중 {step} 간격)", flush=True)
    scan = []
    for name, path in rows:
        k_t, c_t, image = predict(net_t, path)
        k_c, c_c, _ = predict(net_c, path)
        centre = None
        if k_t is not None and k_c is not None:
            centre = float(np.linalg.norm(np.nanmean(k_t[:8], 0)
                                          - np.nanmean(k_c[:8], 0)))
        scan.append({"name": name, "path": path,
                     "t_det": k_t is not None, "t_conf": c_t,
                     "c_det": k_c is not None, "c_conf": c_c,
                     "centre_gap": centre,
                     "k_t": k_t, "k_c": k_c, "image": image})

    n = len(scan)
    both = [s for s in scan if s["t_det"] and s["c_det"]]
    only_t = [s for s in scan if s["t_det"] and not s["c_det"]]
    only_c = [s for s in scan if s["c_det"] and not s["t_det"]]
    agree = [s for s in both if s["centre_gap"] is not None
             and s["centre_gap"] < 30]
    print(f"  검출  목표 {sum(s['t_det'] for s in scan)}/{n}   "
          f"도전자 {sum(s['c_det'] for s in scan)}/{n}", flush=True)
    print(f"  둘 다 {len(both)}   목표만 {len(only_t)}   도전자만 {len(only_c)}   "
          f"같은 곳(<30px) {len(agree)}", flush=True)

    def pick(pool, key, take_max=True):
        if not pool:
            return None
        return (max if take_max else min)(pool, key=key)

    strata = [
        ("A_agree", "BOTH DETECT, same object (centre gap smallest)",
         pick(both, lambda s: -(s["centre_gap"] or 1e9))),
        ("B_disagree", "BOTH DETECT but different object (centre gap largest)",
         pick(both, lambda s: s["centre_gap"] or -1)),
        ("C_only_target", "TARGET detects, CHALLENGER misses",
         only_t[len(only_t) // 2] if only_t else None),
        ("D_only_challenger", "CHALLENGER detects, TARGET misses",
         only_c[len(only_c) // 2] if only_c else None),
    ]

    items, index = [], []
    for tag, why, s in strata:
        if s is None:
            print(f"  {tag}: 해당 프레임 없음 -> 건너뜀", flush=True)
            continue
        left = panel(s["image"], s["k_t"], s["t_conf"], TARGET[1], why)
        right = panel(s["image"], s["k_c"], s["c_conf"], CHALLENGER[1], why)
        pair = np.hstack([left, np.full((left.shape[0], 4, 3), 255, np.uint8),
                          right])
        fname = f"{tag}__{s['name']}"
        cv2.imwrite(os.path.join(OUT, fname), pair)
        gapstr = ("n/a" if s["centre_gap"] is None
                  else f"{s['centre_gap']:.0f}px")
        items.append((pair, f"{tag}  {s['name']}  |  TARGET conf "
                            f"{s['t_conf']:.2f}  vs  CHALLENGER conf "
                            f"{s['c_conf']:.2f}  |  centre gap {gapstr}"))
        index.append({"file": fname, "stratum": tag, "why": why,
                      "frame": s["name"], "target_conf": s["t_conf"],
                      "challenger_conf": s["c_conf"], "centre_gap": s["centre_gap"]})
        print(f"  {tag:20} {s['name']} -> {fname}", flush=True)

    if items:
        cv2.imwrite(os.path.join(OUT, "CONTACT_SHEET.png"), sheet(items))
        print("  contact sheet -> CONTACT_SHEET.png", flush=True)
    json.dump({"set": "vdoframes (indoor lifter, NO GT)", "step": step,
               "n_scanned": n,
               "detect": {"target": sum(s["t_det"] for s in scan),
                          "challenger": sum(s["c_det"] for s in scan),
                          "both": len(both), "only_target": len(only_t),
                          "only_challenger": len(only_c),
                          "same_object_lt30px": len(agree)},
               "caveat": "GT 없음 — corner/R/yaw/ADD/IoU3D 산출 불가. "
                         "정본 161 과 도메인이 다르므로 순위 확인/반박으로 쓰지 말 것",
               "panels": index},
              open(os.path.join(OUT, "INDEX.json"), "w"), indent=1, default=str)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
