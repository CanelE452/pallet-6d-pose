"""좌표 mapping parity 검증 — SOURCE_DEV(합성 val) 에서만.

무엇을 검증하나
  1. 각 teacher 의 9 keypoint 가 **같은 index 순서**를 쓴다
  2. belief 격자 / 640 격자 -> 원본 픽셀 역매핑이 맞다
  3. 어떤 teacher 도 좌우·상하가 뒤집혀 있지 않다

무엇을 검증하지 않나
  real 도메인 성능. 이 스크립트는 배선 검사이지 연구 질문의 측정이 아니다.
  그래서 DEV_EVAL 을 건드리지 않는다.

판정
  identity 배정이 모든 순열 대안보다 확실히 낫고, 합성 오차가 작아야 PASS.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M
import mtcd_teachers as T

SYN = M.REPO_ROOT / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"
DOPE_VAL = M.REPO_ROOT / "data/pallet/training_data/dope_final_g38_p0_tex/val"

# 검사할 순열 대안 — 규약 사고에서 실제로 나온 것들
ALT_PERMS = {
    "identity":      list(range(9)),
    "flip_lr":       [1, 0, 3, 2, 5, 4, 7, 6, 8],     # dataset flip_idx
    "near_far_swap": [4, 5, 6, 7, 0, 1, 2, 3, 8],
    "top_bottom":    [3, 2, 1, 0, 7, 6, 5, 4, 8],
    "c4_rot90":      [1, 5, 6, 2, 0, 4, 7, 3, 8],     # challenge_c4_track
}


def yolo_gt(stem: str):
    """합성 val 라벨 -> (9,2) 픽셀 좌표. 이미지는 이미 PAD 100 이 구워져 있다."""
    label = SYN / "labels/val" / f"{stem}.txt"
    image = SYN / "images/val" / f"{stem}.png"
    img = cv2.imread(str(image))
    if img is None:
        return None, None
    h, w = img.shape[:2]
    parts = label.read_text().split("\n")[0].split()
    values = list(map(float, parts[5:]))
    xy = np.array([[values[3 * i] * w, values[3 * i + 1] * h] for i in range(9)])
    vis = np.array([values[3 * i + 2] for i in range(9)])
    return img, (xy, vis)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--registry", type=Path,
                        default=M.TRACK / "TEACHER_REGISTRY.json")
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text())["teachers"]
    stems = sorted(p.stem for p in (SYN / "images/val").glob("*.png"))
    random.Random(20260905).shuffle(stems)
    stems = stems[:args.n]

    report = {"population": "SOURCE_DEV synthetic val (already padded canvas)",
              "n_frames": len(stems), "teachers": {}}
    for tid, spec in registry.items():
        weights = M.REPO_ROOT / spec["checkpoint"]
        actual = M.sha256_file(weights)
        if actual != spec["sha256"]:
            raise SystemExit(f"{tid}: sha mismatch {actual} != {spec['sha256']}")
        kind = spec["kind"]
        model = T.load_dope(weights) if kind == "dope" else T.load_yolo(weights)
        errs = {name: [] for name in ALT_PERMS}
        n_ok = 0
        for stem in stems:
            img, gt = yolo_gt(stem)
            if img is None:
                continue
            gt_xy, gt_vis = gt
            out = (T.infer_dope(model, img, already_padded=True) if kind == "dope"
                   else T.infer_yolo(model, img, already_padded=True))
            if out.get("status") != "OK" or not out.get("keypoints_xy"):
                continue
            pred = np.asarray(out["keypoints_xy"], dtype=np.float64)
            n_ok += 1
            for name, perm in ALT_PERMS.items():
                d = np.linalg.norm(pred[perm] - gt_xy, axis=1)
                m = (gt_vis > 0) & np.isfinite(d)
                errs[name] += list(d[m])
        block = {"n_frames_with_prediction": n_ok,
                 "detection_rate": n_ok / len(stems)}
        for name in ALT_PERMS:
            block[name] = M.error_stats(errs[name])
        medians = {n: block[n]["median_px"] for n in ALT_PERMS
                   if block[n]["median_px"] is not None}
        best = min(medians, key=medians.get) if medians else None
        block["best_assignment"] = best
        block["identity_is_best"] = best == "identity"
        runner_up = sorted(v for k, v in medians.items() if k != "identity")
        block["margin_vs_next_px"] = (runner_up[0] - medians["identity"]
                                      if runner_up and "identity" in medians else None)
        block["verdict"] = ("PASS" if block["identity_is_best"]
                            and medians.get("identity", 1e9) < 15.0 else "FAIL")
        report["teachers"][tid] = block
        print(f"{tid:22} det {block['detection_rate']:.3f}  "
              f"identity med {medians.get('identity', float('nan')):7.3f}  "
              f"best={best:14} margin {block['margin_vs_next_px']}  {block['verdict']}")
        del model

    report["all_pass"] = all(b["verdict"] == "PASS" for b in report["teachers"].values())
    out = M.AUDIT / "TEACHER_WIRING_PARITY.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nall_pass = {report['all_pass']}   -> {out.relative_to(M.REPO_ROOT)}")
    return 0 if report["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
