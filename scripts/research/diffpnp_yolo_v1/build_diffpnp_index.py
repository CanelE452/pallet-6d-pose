"""DiffPnP loss 가 쓸 프레임별 기하 사이드카를 만든다.

YOLO 학습셋의 이미지 stem 하나마다 다음을 저장한다.

    K       (3,3)  소스 카메라 내부 파라미터 (pad·resize 이전)
    X       (8,3)  world frame cuboid 코너 — projected_cuboid 와 같은 순서
    uv_src  (8,2)  소스 projected_cuboid — 학습 시 affine 복원의 기준
    R_ref   (3,3)  world->camera 참조 회전   (GT 2D 로 푼 PnP)
    t_ref   (3,)   world->camera 참조 이동

참조 pose 는 GT 2D 로 푼 PnP 다. 평가가 pose 를 읽는 방식과 같은 연산이므로
규약 오프셋(perm_v4 180도 문제)이 예측·참조 양쪽에서 상쇄된다.

    conda run -n pallet-pose python -u \
        scripts/research/diffpnp_yolo_v1/build_diffpnp_index.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
YOLO = REPO / "challenge/yolo_pose_one_model"
DATASET = YOLO / "datasets/g38_legacy_v1v2_p0_tex20k"
OUT_DIR = REPO / "data/pallet/results/diffpnp_yolo_v1"

G38_LABELS = REPO / "data/pallet/training_data/paper_release/v2_prod40k_clean_merged/labels"
P0_RAW = YOLO / "datasets/_raw_legacy_v1v2_p0_10k"
TEX_RAW = YOLO / "datasets/_raw_legacy_v1v2_p0_tex10k"

STEM_RE = {
    "G38": re.compile(r"^G38__G__(f\d+)$"),
    "P0": re.compile(r"^P0__(shard_\d+)_(f\d+)$"),
    "TEX": re.compile(r"^TEX__(shard_\d+)_(f\d+)$"),
}


def source_label(stem: str) -> Path | None:
    m = STEM_RE["G38"].match(stem)
    if m:
        return G38_LABELS / f"{m.group(1)}_label.json"
    m = STEM_RE["P0"].match(stem)
    if m:
        return P0_RAW / m.group(1) / "labels" / f"{m.group(2)}_label.json"
    m = STEM_RE["TEX"].match(stem)
    if m:
        return TEX_RAW / m.group(1) / "labels" / f"{m.group(2)}_label.json"
    return None


def entry_from_label(path: Path):
    d = json.loads(path.read_text())
    objects = d.get("objects") or []
    if len(objects) != 1:
        return None, "not_single_object"
    o = objects[0]
    ci = d["camera_data"]["intrinsics"]
    K = np.array([[ci["fx"], 0.0, ci["cx"]],
                  [0.0, ci["fy"], ci["cy"]], [0.0, 0.0, 1.0]], np.float64)
    X = np.asarray(o["cuboid"], np.float64)
    uv = np.asarray(o["projected_cuboid"], np.float64)
    if X.shape != (8, 3) or uv.shape != (8, 2):
        return None, "bad_shape"
    if not (np.isfinite(X).all() and np.isfinite(uv).all()):
        return None, "non_finite"

    ok, rvec, tvec = cv2.solvePnP(X, uv, K, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None, "pnp_failed"
    rvec, tvec = cv2.solvePnPRefineLM(X, uv, K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    projected, _ = cv2.projectPoints(X, rvec, tvec, K, None)
    residual = float(np.linalg.norm(projected.reshape(-1, 2) - uv, axis=1).mean())
    if not np.isfinite(residual) or residual > 1e-3:
        return None, "reference_reproj_too_large"
    if float((X @ R.T + t)[:, 2].min()) <= 0.0:
        return None, "behind_camera"
    return {"K": K, "X": X, "uv": uv, "R": R, "t": t}, "ok"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stems, rows, reasons = [], [], {}
    for split in ("train", "val"):
        images = sorted((DATASET / "images" / split).glob("*.png"))
        print(f"{split}: {len(images)} images", flush=True)
        for i, img in enumerate(images):
            stem = img.stem
            label = source_label(stem)
            if label is None or not label.exists():
                reasons["no_source_label"] = reasons.get("no_source_label", 0) + 1
                continue
            entry, why = entry_from_label(label)
            reasons[why] = reasons.get(why, 0) + 1
            if entry is None:
                continue
            stems.append(f"{split}/{stem}")
            rows.append(entry)
            if (i + 1) % 10000 == 0:
                print(f"  {split} {i + 1}/{len(images)}", flush=True)

    index = {s: i for i, s in enumerate(stems)}
    np.savez_compressed(
        OUT_DIR / "diffpnp_index.npz",
        K=np.stack([r["K"] for r in rows]).astype(np.float32),
        X=np.stack([r["X"] for r in rows]).astype(np.float32),
        uv=np.stack([r["uv"] for r in rows]).astype(np.float32),
        R=np.stack([r["R"] for r in rows]).astype(np.float32),
        t=np.stack([r["t"] for r in rows]).astype(np.float32),
    )
    (OUT_DIR / "diffpnp_index_stems.json").write_text(json.dumps(index))
    meta = {"schema_version": "diffpnp_index_v1",
            "dataset": str(DATASET.relative_to(REPO)),
            "n_indexed": len(stems), "reasons": reasons,
            "reference_pose": "cv2.SOLVEPNP_SQPNP + RefineLM on GT projected_cuboid",
            "frames_3d": "world-frame cuboid, same order as projected_cuboid",
            "note": "perm_v4 는 적용하지 않는다 — 이 학습셋 라벨이 원본 순서다"}
    (OUT_DIR / "DIFFPNP_INDEX_META.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nindexed {len(stems)}")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {k:28s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
