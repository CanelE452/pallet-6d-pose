"""DiffPnP 학습 스모크 — 실제 ultralytics 루프에서 항이 도는지, 얼마나 감독되는지.

특히 mosaic 은 이미지 4장을 합치면서 ``im_file`` 을 대표 1장만 남긴다. 그러면
인스턴스와 사이드카가 어긋날 수 있는데, affine 잔차 게이트가 그걸 걸러내는지를
**수치로** 확인한다 (추정하지 않는다).

    conda run -n pallet-yolo26 python -u \
        scripts/research/diffpnp_yolo_v1/smoke_train.py --lambda-dp 0.01 --fraction 0.01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

OUT = REPO / "data/pallet/results/diffpnp_yolo_v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-dp", type=float, default=0.01)
    ap.add_argument("--fraction", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mosaic", type=float, default=None,
                    help="지정하면 mosaic 확률을 덮어쓴다 (진단용)")
    ap.add_argument("--name", default="SMOKE")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = OUT / f"diffpnp_config_{args.name}.json"
    cfg_path.write_text(json.dumps({
        "enabled": True, "lambda_dp": args.lambda_dp, "gn_steps": 5,
        "huber_delta_norm": 0.10, "damping": 1e-3, "delta_clip": 0.5,
        "min_visible": 6, "affine_residual_max_px": 0.5,
        "index_dir": str(OUT),
    }, indent=1))
    os.environ["DIFFPNP_CONFIG"] = str(cfg_path)

    from pallet_yolo_loss.trainer import DiffPnPTrainer

    seen = {"instances": 0, "valid": 0, "miss": 0, "affine_reject": 0,
            "batches": 0, "last_dp": 0.0, "corner_norm": 0.0, "heads": 0}

    def _inner_losses(criterion):
        """end2end 면 criterion 은 E2ELoss 래퍼다 — 안쪽 두 벌을 꺼낸다."""
        if criterion is None:
            return []
        if hasattr(criterion, "dp_stats"):
            return [criterion]
        return [x for x in (getattr(criterion, "one2many", None),
                            getattr(criterion, "one2one", None))
                if x is not None and hasattr(x, "dp_stats")]

    def on_batch_end(trainer):
        inner = _inner_losses(getattr(trainer.model, "criterion", None))
        if not inner:
            return
        stats = {"n_valid": sum(x.dp_stats["n_valid"] for x in inner),
                 "n_lookup_miss": sum(x.dp_stats["n_lookup_miss"] for x in inner),
                 "n_affine_reject": sum(x.dp_stats["n_affine_reject"] for x in inner),
                 "last_dp": inner[0].dp_stats["last_dp"],
                 "mean_corner_norm": inner[0].dp_stats["mean_corner_norm"]}
        seen["heads"] = len(inner)
        seen["batches"] += 1
        seen["valid"] += stats["n_valid"]
        seen["miss"] += stats["n_lookup_miss"]
        seen["affine_reject"] += stats["n_affine_reject"]
        seen["last_dp"] = stats["last_dp"]
        seen["corner_norm"] = stats["mean_corner_norm"]
        if seen["batches"] % 5 == 0:
            print(f"  batch {seen['batches']:4d}  valid={stats['n_valid']:4d} "
                  f"miss={stats['n_lookup_miss']:4d} "
                  f"affine_reject={stats['n_affine_reject']:4d} "
                  f"dp={stats['last_dp']:.6f} "
                  f"corner_norm={stats['mean_corner_norm']:.4f}", flush=True)

    overrides = dict(
        model=str(REPO / "challenge/weights/pretrained_yolo/yolo26n-pose.pt"),
        data=str(REPO / "challenge/yolo_pose_one_model/datasets/"
                        "g38_legacy_v1v2_p0_tex20k/data.yaml"),
        epochs=args.epochs, batch=args.batch, imgsz=640, device="0", workers=2,
        seed=42, deterministic=True, val=False, plots=False, save=False,
        fraction=args.fraction, patience=0, optimizer="SGD", cos_lr=True,
        project=str(OUT / "smoke_runs"), name=args.name, exist_ok=True,
        verbose=False,
    )
    if args.mosaic is not None:
        overrides["mosaic"] = args.mosaic

    trainer = DiffPnPTrainer(overrides=overrides)
    trainer.add_callback("on_train_batch_end", on_batch_end)
    trainer.train()

    supervised = seen["valid"]
    report = {"lambda_dp": args.lambda_dp, "fraction": args.fraction,
              "mosaic_override": args.mosaic, "batches": seen["batches"],
              "instances_supervised": supervised,
              "lookup_miss": seen["miss"],
              "affine_rejected": seen["affine_reject"],
              "loss_heads_with_term": seen["heads"],
              "last_dp": seen["last_dp"],
              "mean_corner_norm": seen["corner_norm"]}
    (OUT / f"SMOKE_{args.name}.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
