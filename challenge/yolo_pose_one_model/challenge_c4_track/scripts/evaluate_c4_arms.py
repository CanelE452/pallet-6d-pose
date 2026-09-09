#!/usr/bin/env python3
"""F0 / F1 을 같은 val 로 재고, fixed-index 와 C4-equivalent 를 **분리해서** 보고한다.

Ultralytics pose mAP 하나로 성공을 판정하지 않는다.  정사각 팔레트에서는 "위치는
맞는데 번호가 90도 돌았다" 와 "정말 못 찾았다" 가 완전히 다른 실패인데, indexed
metric 은 둘을 같은 크기로 벌한다.

산출:
    per_frame.csv          프레임별 fixed/C4 오차 + 선택된 permutation
    COMPARISON.json/.md    집계
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

TRACK = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
ONE = REPO / "challenge/yolo_pose_one_model"
DS = ONE / "datasets/live_gt_v4"

PERMS = json.loads((TRACK / "C4_PERMUTATIONS.json").read_text(encoding="utf-8"))
P = {int(k): list(v) for k, v in PERMS["permutations"].items()}
DEGREES = [0, 90, 180, 270]


def per_frame(weights: Path) -> list[dict]:
    """val 프레임마다 fixed-index / C4-equivalent 오차와 선택된 permutation 을 낸다."""
    model = YOLO(str(weights))
    rows = []
    for ip in sorted(glob.glob(str(DS / "images/val/*.png"))):
        lp = ip.replace("/images/", "/labels/").replace(".png", ".txt")
        im = cv2.imread(ip)
        h, w = im.shape[:2]
        kp = np.array(open(lp).read().split()[5:], float).reshape(-1, 3)
        gt = kp[:, :2] * [w, h]
        vis = kp[:, 2] > 0
        r = model.predict(im, imgsz=640, conf=0.25, verbose=False)[0]
        stem = Path(ip).stem
        if not len(r.boxes):
            rows.append(dict(stem=stem, detected=0, fixed_px=np.nan, c4_px=np.nan,
                             best_deg=-1, n_vis=int(vis.sum())))
            continue
        pr = r.keypoints.data[0][:, :2].cpu().numpy()
        errs = {}
        for d in DEGREES:
            g = gt[P[d]]
            errs[d] = float(np.linalg.norm(g[vis] - pr[vis], axis=1).mean())
        best = min(errs, key=errs.get)
        rows.append(dict(stem=stem, detected=1, fixed_px=errs[0], c4_px=errs[best],
                         best_deg=best, n_vis=int(vis.sum()),
                         **{f"err_{d}": errs[d] for d in DEGREES}))
    return rows


def summarize(rows: list[dict]) -> dict:
    det = [r for r in rows if r["detected"]]
    fx = np.array([r["fixed_px"] for r in det])
    c4 = np.array([r["c4_px"] for r in det])

    def stats(a):
        return {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                "lt5_rate": float((a < 5).mean()), "gt20_rate": float((a > 20).mean()),
                "mean": float(a.mean()), "max": float(a.max())}

    rescued = int(((fx > 20) & (c4 < 5)).sum())
    collapse = int((c4 > 20).sum())
    hist = {str(d): sum(1 for r in det if r["best_deg"] == d) for d in DEGREES}
    return {
        "n_frames": len(rows), "n_detected": len(det),
        "n_missed": len(rows) - len(det),
        "fixed_index": stats(fx), "c4_equivalent": stats(c4),
        "symmetry_rescued": rescued, "true_collapse": collapse,
        "selected_permutation_hist": hist,
    }


def detection_metrics(weights: Path) -> dict:
    m = YOLO(str(weights)).val(data=str(DS / "data.yaml"), imgsz=640, batch=32,
                               split="val", verbose=False,
                               project=str(TRACK / "_eval"),
                               name=weights.parent.parent.name + "__val",
                               exist_ok=True)
    return {"box_mAP50": float(m.box.map50), "box_mAP50_95": float(m.box.map),
            "box_precision": float(m.box.mp), "box_recall": float(m.box.mr),
            "pose_mAP50": float(m.pose.map50), "pose_mAP50_95": float(m.pose.map)}


ARMS = {
    "F0": TRACK / "F0/weights/best.pt",
    "F1": TRACK / "F1/weights/best.pt",
    "F2": TRACK / "F2/weights/best.pt",
    "v4": ONE / "runs_live_gt/ft_live_gt_v4/weights/best.pt",
}


def main() -> int:
    # stale 캐시는 pose mAP 를 통째로 뒤집은 이력이 있다.
    for c in (DS / "labels").glob("*.cache"):
        c.unlink()
        print(f"캐시 제거: {c.name}")

    out = {}
    all_rows = []
    for arm, w in ARMS.items():
        if not w.is_file():
            print(f"[STOP] {arm} 가중치 없음: {w}")
            return 1
        print(f"\n=== {arm} === {w}")
        rows = per_frame(w)
        s = summarize(rows)
        s["detection"] = detection_metrics(w)
        out[arm] = s
        for r in rows:
            all_rows.append({"arm": arm, **r})
        print(f"  검출 {s['n_detected']}/{s['n_frames']}  미검출 {s['n_missed']}")
        print(f"  fixed  median {s['fixed_index']['median']:.2f}px  "
              f"p90 {s['fixed_index']['p90']:.2f}  "
              f"<5px {s['fixed_index']['lt5_rate']*100:.1f}%  "
              f">20px {s['fixed_index']['gt20_rate']*100:.1f}%")
        print(f"  C4     median {s['c4_equivalent']['median']:.2f}px  "
              f"p90 {s['c4_equivalent']['p90']:.2f}  "
              f"<5px {s['c4_equivalent']['lt5_rate']*100:.1f}%  "
              f">20px {s['c4_equivalent']['gt20_rate']*100:.1f}%")
        print(f"  rescued {s['symmetry_rescued']}  true collapse {s['true_collapse']}")
        print(f"  perm hist {s['selected_permutation_hist']}")

    cols = ["arm", "stem", "detected", "fixed_px", "c4_px", "best_deg", "n_vis"] + \
           [f"err_{d}" for d in DEGREES]
    with open(TRACK / "per_frame.csv", "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wcsv.writeheader()
        wcsv.writerows(all_rows)

    (TRACK / "COMPARISON.json").write_text(json.dumps({
        "init_sha256": "6a40a4d430fd205a427e38a1927aad2a0a0bef20b984484e0b77318e7bd355ea",
        "dataset": str(DS.relative_to(REPO)),
        "val_frames": out["F0"]["n_frames"],
        "note": "F2 는 crop 을 뺀 live_gt_v5_nocrop 으로 학습했으나 val 155 는 전부 동일하다",
        "permutations": PERMS["permutations"],
        "arms": out,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n기록: {TRACK/'COMPARISON.json'} · {TRACK/'per_frame.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
