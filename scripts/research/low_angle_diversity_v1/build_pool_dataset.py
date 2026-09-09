"""§4 · §5 — pool 을 **정본 converter** 로 새로 변환하고 불변식·누수를 검사한다.

과거 생성된 YOLO txt 를 믿지 않는다.  `prepare_yolo_pose.py`(f2b2739 이후 정본)의
`one()` 을 그대로 써서 source JSON -> pad100 PNG + YOLO txt 를 만든다.
"""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import cv2

cv2.setNumThreads(1)                     # OpenMP 스핀 회피 (memory: openmp-fork-spin)

REPO = Path(__file__).resolve().parents[3]
YOLO_ROOT = REPO / "challenge/yolo_pose_one_model"
POOL = REPO / "data/pallet/training_data/paper_release/oblique/extracted"
DST = YOLO_ROOT / "datasets/low_angle_diversity_v1_pool"
OUT = REPO / "data/pallet/results/low_angle_diversity_v1"
PKGS = ["corner_la_oblique_v1_y15_30", "corner_la_oblique_v1_y30_plus"]

_spec = importlib.util.spec_from_file_location(
    "prepare_yolo_pose", YOLO_ROOT / "scripts/prepare_yolo_pose.py")
prep = importlib.util.module_from_spec(_spec)
sys.modules["prepare_yolo_pose"] = prep
_spec.loader.exec_module(prep)


def violates_camera_facing_0123(kp):
    """규약: 0 왼쪽 / 1 오른쪽,  0·1 위 / 3·2 아래  (앞면 0123)."""
    k = np.asarray(kp, float)
    return bool(k[0, 0] > k[1, 0] or k[0, 1] > k[3, 1] or k[1, 1] > k[2, 1])


def main():
    (DST / "images/train").mkdir(parents=True, exist_ok=True)
    (DST / "labels/train").mkdir(parents=True, exist_ok=True)
    stats = {"ok": 0}
    viol, stems = 0, []
    for pkg in PKGS:
        labs = sorted((POOL / pkg / "labels").glob("*_label.json"))
        for i, lab in enumerate(labs):
            if i % 500 == 0:
                print(f"  {pkg} {i}/{len(labs)}", flush=True)
            fid = lab.name.replace("_label.json", "")
            img = POOL / pkg / "rgb" / f"{fid}_rgb.png"
            if not img.is_file():
                stats["missing_rgb"] = stats.get("missing_rgb", 0) + 1
                continue
            stem = f"OBL__{pkg.replace('corner_la_oblique_v1_', '')}__{fid}"
            obj = json.load(open(lab))["objects"][0]
            if violates_camera_facing_0123(np.asarray(obj["projected_cuboid"], float)[:8]):
                viol += 1
            res = prep.one((stem, str(img.relative_to(REPO)), str(lab.relative_to(REPO)),
                            str(DST / "images/train" / f"{stem}.png"),
                            str(DST / "labels/train" / f"{stem}.txt")))
            stats[res] = stats.get(res, 0) + 1
            if res == "ok":
                stems.append(stem)

    # ---- §4 불변식
    lbls = sorted((DST / "labels/train").glob("*.txt"))
    imgs = sorted((DST / "images/train").glob("*.png"))
    nan = unknown_sup = struct_bad = 0
    for p in lbls:
        v = p.read_text().split()
        if len(v) != 5 + 27:
            struct_bad += 1
            continue
        a = np.array(v[1:], float)
        if not np.isfinite(a).all():
            nan += 1
        kp = a[4:].reshape(9, 3)
        # "모른다"(v=0)인데 좌표가 0 이 아니면 계약 위반
        if ((kp[:, 2] == 0) & ((kp[:, 0] != 0) | (kp[:, 1] != 0))).any():
            unknown_sup += 1

    # ---- §5 누수: 합성 렌더 대 실사 평가셋
    man = json.loads((REPO / "data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json").read_text())
    eval_imgs = [REPO / f["image"] for f in man["frames_list"]]
    eval_h = set()
    for p in eval_imgs:
        if p.is_file():
            eval_h.add(hashlib.sha256(p.read_bytes()).hexdigest())
    rng = np.random.default_rng(0)
    sample = [imgs[i] for i in rng.choice(len(imgs), min(400, len(imgs)), replace=False)]
    overlap = sum(hashlib.sha256(p.read_bytes()).hexdigest() in eval_h for p in sample)
    eval_ids = {f["frame_id"] for f in man["frames_list"]}

    rep = {
        "schema_version": "low_angle_diversity_v1_pool_dataset_v1",
        "converter": "challenge/yolo_pose_one_model/scripts/prepare_yolo_pose.py",
        "converter_sha256": hashlib.sha256(
            (YOLO_ROOT / "scripts/prepare_yolo_pose.py").read_bytes()).hexdigest(),
        "convert_stats": stats,
        "n_images": len(imgs), "n_labels": len(lbls),
        "image_label_count_equal": len(imgs) == len(lbls),
        "duplicate_stem": len(stems) - len(set(stems)),
        "camera_facing_0123_violations": viol,
        "unknown_promoted_to_supervised": unknown_sup,
        "nan_or_inf": nan,
        "structure_invalid": struct_bad,
        "eval_leakage": {
            "eval_images_hashed": len(eval_h),
            "pool_sampled": len(sample),
            "image_hash_overlap": overlap,
            "frame_id_overlap": len(eval_ids & set(stems)),
            "note": "pool 은 합성 렌더, PAPER_EVAL 319 는 실사 촬영이라 구조적으로 분리된다.",
        },
    }
    gates = {
        "camera_facing_0123_violations": viol == 0,
        "unknown_promoted_to_supervised": unknown_sup == 0,
        "nan_or_inf": nan == 0,
        "structure_invalid": struct_bad == 0,
        "image_label_count_equal": len(imgs) == len(lbls),
        "duplicate_stem": len(stems) == len(set(stems)),
        "TRAIN_EVAL_IMAGE_OVERLAP_zero": overlap == 0 and len(eval_ids & set(stems)) == 0,
    }
    rep["gates"] = gates
    rep["verdict"] = "PASS" if all(gates.values()) else "FAIL"
    (OUT / "POOL_LABEL_CONTRACT.json").write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    (OUT / "POOL_STEMS.txt").write_text("\n".join(sorted(stems)) + "\n")
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
