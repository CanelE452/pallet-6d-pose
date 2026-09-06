"""Q0 — DiffPnP 항이 keypoint 예측을 미는 힘을 재서 lambda 를 고른다 (판정 아님).

loss **값**의 비율로 lambda 를 잡으면 안 된다 (memory: 집계 통계는 pose 로 전이 안 된다).
두 항이 ``pred_kpts`` 에 만드는 **기울기 노름**의 비율로 잡는다. 목표 대역은 DOPE
트랙에서 쓴 것과 같은 5% 다.

    conda run -n pallet-yolo26 python -u \
        scripts/research/diffpnp_yolo_v1/q0_grad_band.py --batches 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "data/pallet/results/diffpnp_yolo_v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--target-band", type=float, default=0.05)
    ap.add_argument("--fraction", type=float, default=0.02)
    args = ap.parse_args()

    cfg_path = OUT / "diffpnp_config_Q0.json"
    cfg_path.write_text(json.dumps({
        "enabled": True, "lambda_dp": 1.0, "gn_steps": 5,
        "huber_delta_norm": 0.10, "damping": 1e-3, "delta_clip": 0.5,
        "min_visible": 6, "affine_residual_max_px": 0.5, "index_dir": str(OUT),
    }, indent=1))
    os.environ["DIFFPNP_CONFIG"] = str(cfg_path)
    os.environ["DIFFPNP_PROBE"] = "1"

    from pallet_yolo_loss.trainer import DiffPnPTrainer

    rows = []

    def probe(trainer):
        crit = getattr(trainer.model, "criterion", None)
        inner = [x for x in (getattr(crit, "one2many", None),
                             getattr(crit, "one2one", None))
                 if x is not None and hasattr(x, "dp_stats")]
        if not inner and hasattr(crit, "dp_stats"):
            inner = [crit]
        for x in inner:
            s = x.dp_stats
            if "grad_base" in s and s.get("n_valid", 0) > 0:
                rows.append({"grad_base": s["grad_base"],
                             "grad_dp": s["grad_dp_at_lambda1"],
                             "lambda_5pct": s["lambda_for_5pct"],
                             "dp": s["last_dp"], "n_valid": s["n_valid"],
                             "corner_norm": s["mean_corner_norm"]})

    overrides = dict(
        model=str(REPO / "challenge/weights/pretrained_yolo/yolo26n-pose.pt"),
        data=str(REPO / "challenge/yolo_pose_one_model/datasets/"
                        "g38_legacy_v1v2_p0_tex20k/data.yaml"),
        epochs=1, batch=args.batch, imgsz=640, device="0", workers=2,
        seed=42, deterministic=True, val=False, plots=False, save=False,
        fraction=args.fraction, patience=0, optimizer="SGD", amp=False,
        project=str(OUT / "smoke_runs"), name="Q0", exist_ok=True, verbose=False,
    )
    trainer = DiffPnPTrainer(overrides=overrides)
    trainer.add_callback("on_train_batch_end", probe)
    trainer.train()

    if not rows:
        print("측정된 배치가 없다 — 항이 돌지 않았다")
        return 1
    lam = np.array([r["lambda_5pct"] for r in rows])
    ratio_at_1 = np.array([r["grad_dp"] / max(r["grad_base"], 1e-12) for r in rows])
    report = {
        "batches_measured": len(rows),
        "target_band": args.target_band,
        "grad_ratio_at_lambda1_median": float(np.median(ratio_at_1)),
        "lambda_for_5pct_median": float(np.median(lam)),
        "lambda_for_5pct_p25": float(np.percentile(lam, 25)),
        "lambda_for_5pct_p75": float(np.percentile(lam, 75)),
        "median_supervised_per_head": float(np.median([r["n_valid"] for r in rows])),
        "median_dp_value": float(np.median([r["dp"] for r in rows])),
        "median_corner_norm": float(np.median([r["corner_norm"] for r in rows])),
        "note": "lambda 는 값이 아니라 pred_kpts 기울기 노름 비율로 정했다",
    }
    (OUT / "Q0_GRAD_BAND.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
