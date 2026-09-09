"""§8 · §12 — DIVERSE_SWAP 데이터셋을 심볼릭 링크로 만든다 (이미지 복제 없음).

CONTROL 은 기존 R0 데이터셋을 **그대로** 쓴다.  treatment 만 새로 만든다.
총 장수는 CONTROL 과 정확히 같아야 한다.
"""
from __future__ import annotations
import json, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DSROOT = REPO / "challenge/yolo_pose_one_model/datasets"
BASE = DSROOT / "g38_legacy_v1v2_p0_tex20k"
POOLDS = DSROOT / "low_angle_diversity_v1_pool"
DST = DSROOT / "low_angle_diversity_v1_swap"
OUT = REPO / "data/pallet/results/low_angle_diversity_v1"


def link(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def main():
    removed = set((OUT / "SWAP_REMOVED.txt").read_text().split())
    added = set((OUT / "SWAP_ADDED.txt").read_text().split())
    for sp in ("train", "val"):
        (DST / "images" / sp).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / sp).mkdir(parents=True, exist_ok=True)

    n_base = n_pool = 0
    for f in sorted((BASE / "labels/train").glob("*.txt")):
        if f.stem in removed:
            continue
        link(BASE / "images/train" / f"{f.stem}.png", DST / "images/train" / f"{f.stem}.png")
        link(f, DST / "labels/train" / f"{f.stem}.txt")
        n_base += 1
    for stem in sorted(added):
        link(POOLDS / "images/train" / f"{stem}.png", DST / "images/train" / f"{stem}.png")
        link(POOLDS / "labels/train" / f"{stem}.txt", DST / "labels/train" / f"{stem}.txt")
        n_pool += 1
    # val 은 CONTROL 과 동일 (§12)
    n_val = 0
    for f in sorted((BASE / "labels/val").glob("*.txt")):
        link(BASE / "images/val" / f"{f.stem}.png", DST / "images/val" / f"{f.stem}.png")
        link(f, DST / "labels/val" / f"{f.stem}.txt")
        n_val += 1

    yaml = (BASE / "data.yaml").read_text().replace(str(BASE), str(DST))
    (DST / "data.yaml").write_text(yaml)

    n_ctrl = len(list((BASE / "labels/train").glob("*.txt")))
    rep = {"schema_version": "low_angle_diversity_v1_treatment_dataset_v1",
           "control_dataset": str(BASE.relative_to(REPO)),
           "treatment_dataset": str(DST.relative_to(REPO)),
           "control_train_n": n_ctrl,
           "treatment_train_n": n_base + n_pool,
           "kept_from_base": n_base, "added_from_pool": n_pool,
           "removed": len(removed), "val_n": n_val,
           "total_N_equal": (n_base + n_pool) == n_ctrl,
           "images_are_symlinks": True,
           "stale_label_cache_present": (DST / "labels/train.cache").exists()}
    if not rep["total_N_equal"]:
        raise RuntimeError(f"총 N 불일치: {n_base + n_pool} != {n_ctrl}")
    (OUT / "TREATMENT_DATASET.json").write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
