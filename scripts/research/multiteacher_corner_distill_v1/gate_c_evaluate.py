"""GATE C 평가 — R0 coarse vs C0 refined vs C1 refined.

불확실성 gate 는 SOURCE_DEV(합성 val)에서만 정한다.

    규칙: "u <= g 인 표본들에서 refined p90 <= 같은 표본의 R0 p90" 을 만족하는 가장 큰 g.

DEV_EVAL 로 gate 를 고르지 않는다.  gate 를 통과하지 못한 keypoint 는 R0 좌표를 유지한다.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M
from mtcd_specialist import CROP, LocalCornerSpecialist, extract_crop

SYN = M.REPO_ROOT / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"
REF = "T0_R0_YOLO26N_G38LEGACY"
N_SOURCE_GATE_FRAMES = 250


def load_specialist(arm, device="cuda:0"):
    path = M.REPO_ROOT / f"weights/multiteacher_corner_distill_v1/specialist_{arm}_last.pt"
    blob = torch.load(str(path), map_location=device)
    model = LocalCornerSpecialist().to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, path, blob


@torch.no_grad()
def refine(model, image, coarse_pts, indices, device="cuda:0"):
    """coarse 좌표 목록을 한 번에 정제. 반환: refined xy, uncertainty."""
    patches, origins = [], []
    for i in indices:
        patch, origin = extract_crop(image, coarse_pts[i])
        patches.append(patch)
        origins.append(origin)
    x = torch.from_numpy((np.stack(patches).astype(np.float32) / 255.0)
                         .transpose(0, 3, 1, 2)).to(device)
    ids = torch.tensor(indices, device=device)
    out = model(x, ids)
    idx = out["heat_logits"].argmax(1).cpu().numpy()
    unc = torch.nn.functional.softplus(out["log_uncertainty"]).cpu().numpy()
    local = np.stack([(idx % CROP).astype(np.float64), (idx // CROP).astype(np.float64)], 1)
    return local + np.stack(origins), unc


def source_gate(model, device="cuda:0"):
    """합성 val 에서 불확실성 gate 를 정한다. real GT 미사용."""
    stems = sorted(p.stem for p in (SYN / "images/val").glob("*.png"))
    random.Random(20260907).shuffle(stems)
    residuals = np.load(M.AUDIT / "R0_SOURCE_COARSE_RESIDUALS.npy")
    per_corner = {k: residuals[residuals[:, 0] == k][:, 1:] for k in range(8)}
    rng = np.random.default_rng(20260907)
    rows = []
    for stem in stems[:N_SOURCE_GATE_FRAMES]:
        img = cv2.imread(str(SYN / "images/val" / f"{stem}.png"))
        if img is None:
            continue
        h, w = img.shape[:2]
        v = list(map(float, (SYN / "labels/val" / f"{stem}.txt").read_text()
                     .split("\n")[0].split()[5:]))
        gt = np.array([[v[3 * i] * w, v[3 * i + 1] * h] for i in range(9)])
        vis = np.array([v[3 * i + 2] for i in range(9)])
        idxs = [k for k in range(8) if vis[k] > 0]
        if not idxs:
            continue
        coarse = {k: gt[k] + per_corner[k][rng.integers(len(per_corner[k]))] for k in idxs}
        pts = np.zeros((9, 2))
        for k, p in coarse.items():
            pts[k] = p
        ref, unc = refine(model, img, pts, idxs, device)
        for j, k in enumerate(idxs):
            rows.append({"u": float(unc[j]),
                         "refined_err": float(np.linalg.norm(ref[j] - gt[k])),
                         "coarse_err": float(np.linalg.norm(coarse[k] - gt[k]))})
    u = np.array([r["u"] for r in rows])
    re = np.array([r["refined_err"] for r in rows])
    ce = np.array([r["coarse_err"] for r in rows])
    curve, best = [], None
    for g in np.percentile(u, np.arange(5, 101, 5)):
        m = u <= g
        if m.sum() < 50:
            continue
        entry = {"gate": float(g), "coverage": float(m.mean()),
                 "refined_p90": float(np.percentile(re[m], 90)),
                 "coarse_p90": float(np.percentile(ce[m], 90)),
                 "refined_median": float(np.median(re[m])),
                 "coarse_median": float(np.median(ce[m]))}
        curve.append(entry)
        if entry["refined_p90"] <= entry["coarse_p90"]:
            best = entry
    return {"n_source_keypoints": len(rows), "curve": curve, "chosen": best,
            "rule": "u <= g 인 표본에서 refined p90 <= coarse p90 을 만족하는 가장 큰 g",
            "population": "SOURCE_DEV synthetic val — real GT 미사용"}


def evaluate_arm(arm, gate_value, gts, r0_pred, device="cuda:0"):
    model, path, blob = load_specialist(arm, device)
    frames, stats = {}, {"n_refined": 0, "n_kept": 0}
    for gt in gts:
        entry = r0_pred.get(gt["frame_id"])
        pts = M.prediction_keypoints(entry)
        if pts is None:
            frames[gt["frame_id"]] = entry or {"status": "NO_DETECTION"}
            continue
        image = cv2.imread(str(M.REPO_ROOT / gt["image"]))
        if image is None:
            frames[gt["frame_id"]] = {"status": "IMAGE_MISSING"}
            continue
        idxs = [k for k in range(8) if np.isfinite(pts[k]).all()]
        new = pts.copy()
        if idxs:
            ref, unc = refine(model, image, pts, idxs, device)
            for j, k in enumerate(idxs):
                if unc[j] <= gate_value:
                    new[k] = ref[j]
                    stats["n_refined"] += 1
                else:
                    stats["n_kept"] += 1
        out = dict(entry)
        out["keypoints_xy"] = new.tolist()
        out["refined_by"] = arm
        frames[gt["frame_id"]] = out
    del model
    payload = {"schema_version": "frozen_arm_prediction_v1",
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "arm": arm, "specialist_checkpoint": str(path.relative_to(M.REPO_ROOT)),
               "specialist_sha256": M.sha256_file(path),
               "uncertainty_gate": gate_value,
               "base_arm": REF, "new_training": 0,
               "refine_stats": stats, "frames": frames}
    (M.PREDICTIONS / f"{arm}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["C0", "C1"])
    args = parser.parse_args()

    gts = [M.load_gt(f) for f in M.dev_eval_frames()]
    r0 = M.load_prediction_file(M.PREDICTIONS / f"{REF}.json")

    report = {"schema_version": "mtcd_gate_c_eval_v1",
              "method_lock_sha256": M.sha256_file(M.METHOD_LOCK_PATH),
              "arms": {}, "gates": {}}
    for arm in args.arms:
        model, _, _ = load_specialist(arm)
        gate = source_gate(model)
        del model
        if gate["chosen"] is None:
            print(f"{arm}: SOURCE_DEV 에서 refined p90 <= coarse p90 인 gate 가 없다 "
                  f"-> 전부 R0 유지 (gate=0)")
            value = 0.0
        else:
            value = gate["chosen"]["gate"]
        report["gates"][arm] = gate
        report["gates"][arm]["applied_gate"] = value
        stats = evaluate_arm(arm, value, gts, r0)
        report["arms"][arm] = {"refine_stats": stats, "gate": value}
        print(f"{arm}: gate {value:.3f}  refined {stats['n_refined']}  kept {stats['n_kept']}")

    blocks = {}
    for name, path in [("R0", M.PREDICTIONS / f"{REF}.json")] + \
                      [(a, M.PREDICTIONS / f"{a}.json") for a in args.arms]:
        blocks[name] = M.arm_2d_report(M.load_prediction_file(path), gts)
    report["two_d"] = blocks

    # rescue / harm — R0 대비
    detail = {}
    for arm in args.arms:
        pa = M.load_prediction_file(M.PREDICTIONS / f"{arm}.json")
        resc = harm = n_gross = n_good = 0
        for gt in gts:
            p0 = M.prediction_keypoints(r0.get(gt["frame_id"]))
            p1 = M.prediction_keypoints(pa.get(gt["frame_id"]))
            if p0 is None or p1 is None:
                continue
            for k in range(8):
                if not gt["supervised"][k]:
                    continue
                e0 = float(np.linalg.norm(p0[k] - gt["xy"][k]))
                e1 = float(np.linalg.norm(p1[k] - gt["xy"][k]))
                if e0 > 20:
                    n_gross += 1
                    resc += e1 <= 10
                if e0 <= 10:
                    n_good += 1
                    harm += e1 > 10
        detail[arm] = {"n_r0_gross20": n_gross,
                       "rescue_rate": resc / n_gross if n_gross else None,
                       "n_r0_good10": n_good,
                       "harm_rate": harm / n_good if n_good else None}
    report["rescue_and_harm"] = detail

    out = M.GATE_C / "GATE_C_2D.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float) + "\n")
    print(f"\n{'arm':6}{'med':>8}{'p90':>9}{'g20':>8}{'g40':>8}   (visible, POOLED_ALL)")
    for name in ["R0"] + args.arms:
        v = blocks[name]["POOLED_ALL"]["visible"]
        print(f"{name:6}{v['median_px']:8.2f}{v['p90_px']:9.2f}{v['gross20']:8.3f}{v['gross40']:8.3f}")
    print(f"-> {out.relative_to(M.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
