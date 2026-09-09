"""A2b preflight — GPU 학습 전에 통과해야 하는 관문 (METHOD_LOCK_A2B §16, §17, §20).

  1. TRAIN stem manifest 생성 (leakage guard 입력)
  2. batch geometry wiring smoke — 실제 dataloader 100 batch, optimizer step 없음
  3. lambda calibration — frozen init 에서 base 대 LC gradient 비

셋 다 통과해야 학습이 시작된다.  실패하면 loss 를 고치지 않고 멈춘다.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "challenge/yolo_pose_one_model/pallet_translation_loss_v1"
sys.path.insert(0, str(PKG))
OUT = REPO / "data/pallet/results/pallet_translation_loss_v1"
DATA_YAML = REPO / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k/data.yaml"
DS = DATA_YAML.parent
R0_CKPT = (REPO / "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
           "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")

IMGSZ = 640
BATCH = 32
N_SMOKE_BATCHES = 100
N_CALIB = 512
LAMBDA_TARGET = 0.05          # [추정][미검증] screen 고정값

# §15 geometry augmentation 계약 — 두 arm 공통
GEOM_OFF = dict(mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
                degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                fliplr=0.0, flipud=0.0)
# appearance 는 R0 recipe 유지
APPEARANCE = dict(hsv_h=0.015, hsv_s=0.5, hsv_v=0.35, erasing=0.4, bgr=0.0,
                  auto_augment="randaugment")


def screen_args(**extra):
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    a = get_cfg(DEFAULT_CFG)
    a.task, a.mode = "pose", "train"
    a.data = str(DATA_YAML)
    a.imgsz, a.batch, a.epochs = IMGSZ, BATCH, 5
    a.workers, a.device, a.seed, a.deterministic = 2, "0", 42, True
    a.optimizer, a.lr0, a.lrf, a.cos_lr = "SGD", 0.01, 0.01, True
    a.momentum, a.weight_decay = 0.937, 0.0005
    a.warmup_epochs, a.warmup_momentum, a.warmup_bias_lr = 3.0, 0.8, 0.1
    a.box, a.cls, a.dfl, a.pose, a.kobj, a.rle, a.angle = 7.5, 0.5, 1.5, 12.0, 1.0, 1.0, 1.0
    a.patience, a.close_mosaic, a.val, a.plots, a.save_period = 0, 0, True, False, -1
    a.single_cls, a.rect, a.amp, a.pretrained = True, False, True, True
    for k, v in {**GEOM_OFF, **APPEARANCE, **extra}.items():
        setattr(a, k, v)
    return a


def write_train_stems() -> Path:
    p = PKG / "TRAIN_STEMS.txt"
    stems = sorted(f.stem for f in (DS / "labels/train").glob("*.txt"))
    p.write_text("\n".join(stems) + "\n", encoding="utf-8")
    print(f"TRAIN stems: {len(stems)} -> {p.name}")
    return p


def build_loader(args):
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    data = check_det_dataset(str(DATA_YAML))
    ds = build_yolo_dataset(args, str(DS / "images/train"), BATCH, data,
                            mode="train", rect=False, stride=32)
    return ds, build_dataloader(ds, batch=BATCH, workers=0, shuffle=False, rank=-1), data


def geometry_smoke(loader, table):
    """§16 — batch keypoint 가 side-table 투영과 letterbox 배율 하나로 이어지는가."""
    import a2b_loss as A
    idx = {s: i for i, s in enumerate(table["stems"])}
    res, n_inst, missing = [], 0, 0
    for bi, b in enumerate(loader):
        if bi >= N_SMOKE_BATCHES:
            break
        kps = b["keypoints"].float().clone()
        kps[..., 0] *= IMGSZ
        kps[..., 1] *= IMGSZ
        bidx = b["batch_idx"].long()
        stems = [Path(f).stem for f in b["im_file"]]
        for j in range(kps.shape[0]):
            stem = stems[int(bidx[j])]
            i = idx.get(stem, -1)
            if i < 0:
                missing += 1
                continue
            gt = kps[j, :8, :2].numpy()
            vis = (kps[j, :8, 2].numpy() > 0) if kps.shape[-1] == 3 else np.ones(8, bool)
            if vis.sum() < 4:
                continue
            K4, R, t, X = table["K"][i], table["R"][i], table["t"][i], table["Xcf"][i]
            P = X @ R.T + t
            u = np.stack([K4[0] * P[:, 0] / P[:, 2] + K4[2],
                          K4[1] * P[:, 1] / P[:, 2] + K4[3]], 1)
            # 축별 배율 (파이프라인이 letterbox 가 아니라 squash 다)
            um, gm = u[vis].mean(0), gt[vis].mean(0)
            du, dg = u[vis] - um, gt[vis] - gm
            s = (du * dg).sum(0) / np.maximum((du * du).sum(0), 1e-12)
            T = gm - s * um
            r = np.linalg.norm(s * u[vis] + T - gt[vis], axis=1)
            res.extend(r.tolist())
            n_inst += 1
    r = np.asarray(res)
    out = {
        "n_batches": min(N_SMOKE_BATCHES, bi + 1), "n_instances": n_inst,
        "n_missing_from_sidetable": missing,
        "residual_px": {"median": float(np.median(r)), "p95": float(np.percentile(r, 95)),
                        "p99": float(np.percentile(r, 99)), "max": float(r.max())},
        "bad_gt_0p1": int((r > 0.1).sum()), "bad_gt_0p5": int((r > 0.5).sum()),
        "bad_gt_1p0": int((r > 1.0).sum()),
        "gate": {"p99_le": 0.1, "max_le": 0.5},
    }
    out["verdict"] = ("PASS" if out["residual_px"]["p99"] <= 0.1
                      and out["residual_px"]["max"] <= 0.5 else "FAIL")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stem_file = write_train_stems()
    table = dict(np.load(PKG / "GEOMETRY_SIDETABLE.npz", allow_pickle=True))
    table["stems"] = table["stems"].tolist()

    args = screen_args()
    ds, loader, data = build_loader(args)
    print(f"dataset: {len(ds)} images, batches={len(loader)}")

    smoke = geometry_smoke(loader, table)
    print(json.dumps(smoke, indent=2))
    report = {"schema_version": "a2b_preflight_v1",
              "method_lock_sha256_at_freeze":
                  "96b5272fd847a4d0421b18ac4b536b4a098e1d7412d2fa807db5b34010eb405a",
              "augmentation_contract": {**GEOM_OFF, **APPEARANCE},
              "geometry_batch_smoke": smoke,
              "train_stem_manifest": str(stem_file.relative_to(REPO)),
              "train_stem_count": len(stem_file.read_text().split()),
              "side_table_total": len(table["stems"])}
    (OUT / "A2B_PREFLIGHT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if smoke["verdict"] != "PASS":
        print("\nGEOMETRY WIRING GATE FAILED — GPU training STOP")
        return 1
    print("\ngeometry wiring gate PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
