"""§20 — lambda calibration.  real PAPER_EVAL 을 쓰지 않는다.

synthetic TRAIN 에서 층화 512 instance 를 뽑아 멤버십을 **먼저** 저장하고,
frozen R0 init 에서 optimizer step 없이 forward 해 base 와 LC 의 gradient 비를 잰다.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "challenge/yolo_pose_one_model/pallet_translation_loss_v1"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = REPO / "data/pallet/results/pallet_translation_loss_v1"
MANIFEST_OUT = PKG / "LC_CALIBRATION_MANIFEST.txt"
N_CALIB_IMAGES = 512
LAMBDA_TARGET = 0.05


def strata(table, train_stems):
    """asset x elevation 사분위 로 층화한다 (projected size / aspect 는 보고만)."""
    stems = table["stems"]
    keep = [i for i, s in enumerate(stems) if s in train_stems]
    R, t, X = table["R"], table["t"], table["Xcf"]
    up = R[keep][:, :, 1]                                   # 물체 up 축 (fixed y = height)
    tt = t[keep]
    d = tt / np.linalg.norm(tt, axis=1, keepdims=True)
    elev = np.degrees(np.arcsin(np.clip(np.abs((up * d).sum(1)), 0, 1)))
    ext = X[keep].max(1) - X[keep].min(1)
    ratio = ext[:, [0, 2]].max(1) / np.maximum(ext[:, [0, 2]].min(1), 1e-9)
    asset = np.array([s.split("__")[0] for s in np.array(stems)[keep]])
    q = np.quantile(elev, [.25, .5, .75])
    ebin = np.digitize(elev, q)
    cells = {}
    for k, (a, e) in enumerate(zip(asset, ebin)):
        cells.setdefault((a, int(e)), []).append(keep[k])
    rng = np.random.default_rng(20260906)
    per = max(1, N_CALIB_IMAGES // max(1, len(cells)))
    chosen = []
    for c in sorted(cells):
        pool = cells[c]
        chosen.extend(rng.choice(pool, min(per, len(pool)), replace=False).tolist())
    if len(chosen) < N_CALIB_IMAGES:
        rest = [i for i in keep if i not in set(chosen)]
        chosen.extend(rng.choice(rest, N_CALIB_IMAGES - len(chosen), replace=False).tolist())
    chosen = chosen[:N_CALIB_IMAGES]
    info = {"cells": len(cells), "per_cell": per,
            "elevation_quartiles_deg": [float(x) for x in q],
            "elev_p10_p50_p90": [float(np.percentile(elev, p)) for p in (10, 50, 90)],
            "aspect_p10_p50_p90": [float(np.percentile(ratio, p)) for p in (10, 50, 90)]}
    return chosen, info


def main() -> int:
    from a2b_preflight import screen_args, DS, DATA_YAML, IMGSZ, R0_CKPT
    import a2b_loss as A

    table = dict(np.load(PKG / "GEOMETRY_SIDETABLE.npz", allow_pickle=True))
    table["stems"] = table["stems"].tolist()
    train_stems = set((PKG / "TRAIN_STEMS.txt").read_text().split())

    chosen, info = strata(table, train_stems)
    stems = [table["stems"][i] for i in chosen]
    MANIFEST_OUT.write_text("\n".join(stems) + "\n", encoding="utf-8")
    print(f"calibration manifest: {len(stems)} images -> {MANIFEST_OUT.name}")

    cfg = {"enabled": True, "lambda_geo": 1.0, "calibration": True,
           "train_stem_list": str(PKG / "TRAIN_STEMS.txt")}
    cfg_path = PKG / "_a2b_calib_config.json"
    cfg_path.write_text(json.dumps(cfg))
    os.environ["A2B_CONFIG"] = str(cfg_path)

    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.loss import E2ELoss

    args = screen_args()
    data = check_det_dataset(str(DATA_YAML))
    # get_img_files 는 리스트 원소를 "디렉터리 또는 목록 txt" 로 본다 -> 목록 파일로 넘긴다
    listing = PKG / "_a2b_calib_images.txt"
    listing.write_text("\n".join(str(DS / "images/train" / f"{s}.png") for s in stems) + "\n")
    ds = build_yolo_dataset(args, str(listing), 16, data, mode="train", rect=False, stride=32)
    loader = build_dataloader(ds, batch=16, workers=0, shuffle=False, rank=-1)

    model, _ = load_checkpoint(str(R0_CKPT), device="cuda")
    model = model.float().to("cuda").train()
    model.args = args
    crit = (E2ELoss(model, A.A2BPoseLoss26) if getattr(model, "end2end", False)
            else A.A2BPoseLoss26(model))
    model.criterion = crit
    inners = [getattr(crit, n) for n in ("one2many", "one2one") if getattr(crit, n, None)] or [crit]

    for b in loader:
        b["img"] = b["img"].to("cuda", non_blocking=True).float() / 255
        for k in ("cls", "bboxes", "batch_idx", "keypoints"):
            if k in b:
                b[k] = b[k].to("cuda")
        model.loss(b)

    rows = []
    for m in inners:
        rows.extend(m.calib_rows)
    gb = np.array([r[1] for r in rows], float)
    gl = np.array([r[2] for r in rows], float)
    finite = np.isfinite(gb) & np.isfinite(gl)
    nan_n = int(np.isnan(gb).sum() + np.isnan(gl).sum())
    inf_n = int(np.isinf(gb).sum() + np.isinf(gl).sum())
    zero_n = int((gl[finite] <= 0).sum())
    valid = finite & (gl > 0)
    ratio = gb[valid] / (gl[valid] + 1e-12)
    lam = float(LAMBDA_TARGET * np.median(ratio))
    scaled = lam * gl[valid] / np.maximum(gb[valid], 1e-12)

    def pct(a, ps=(10, 50, 90)):
        return {f"p{p}": float(np.percentile(a, p)) for p in ps}

    rep = {
        "schema_version": "a2b_lambda_calibration_v1",
        "manifest": str(MANIFEST_OUT.relative_to(REPO)),
        "manifest_images": len(stems), "strata": info,
        "N_total": int(gb.size), "N_valid": int(valid.sum()),
        "valid_fraction": float(valid.sum() / max(gb.size, 1)),
        "base_grad_norm": {**pct(gb[finite]), "max": float(gb[finite].max())},
        "raw_lc_grad_norm": {**pct(gl[valid]), "max": float(gl[valid].max())},
        "ratio_base_over_lc": pct(ratio),
        "lambda_target_fraction": LAMBDA_TARGET,
        "lambda_lc": lam,
        "scaled_lc_over_base": pct(scaled),
        "NaN_count": nan_n, "Inf_count": inf_n, "zero_gradient_count": zero_n,
        "LAMBDA_FROZEN_BEFORE_REAL_EVAL": "YES",
    }
    gate_fail = []
    if nan_n > 0:
        gate_fail.append("NaN>0")
    if inf_n > 0:
        gate_fail.append("Inf>0")
    if rep["valid_fraction"] < 0.90:
        gate_fail.append("valid<90%")
    if rep["scaled_lc_over_base"]["p90"] > 0.5:
        gate_fail.append("scaled p90>0.5")
    rep["sanity_gate"] = "PASS" if not gate_fail else "FAIL: " + ", ".join(gate_fail)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "A2B_LAMBDA_CALIBRATION.json").write_text(
        json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if not gate_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
