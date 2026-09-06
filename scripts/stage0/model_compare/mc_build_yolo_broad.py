"""실험 B — BROAD 40K 를 YOLO-pose 포맷으로 변환한다.

목적: YOLO 를 FINAL40K 와 **똑같은 데이터·똑같은 노출량**으로 학습시켜,
dev105 붕괴의 원인이 데이터인지 아키텍처인지 가른다.

계약은 `prepare_yolo_pose` 의 함수를 그대로 import 해서 쓴다.  복제하면 반드시
어긋난다 — PAD=100 REFLECT_101, v = 패딩 캔버스 안이면 2, bbox 는 v==2 점들의
axis-aligned, 9 키포인트(8 코너 + centroid).

★ v 규칙이 FINAL 의 `n_supervised`(화면 안 채널)와 같은 계약이라는 것이 이 비교의
전제다 (2026-08-20 §5 감사에서 확인).
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "challenge/yolo_pose_one_model/scripts"))
import prepare_yolo_pose as PY  # noqa: E402  PAD / load_kps / to_line

BROAD = os.path.join(ROOT, "data/pallet/training_data/paper_release/"
                           "v2_prod40k_clean_merged")
OUT = os.path.join(ROOT, "challenge/yolo_pose_one_model/datasets/broad40k")
MANIFEST = os.path.join(ROOT, "data/pallet/results/paper_s2_multihead/"
                              "dataset_release/FINAL_SYNTH_TRAIN_V1.json")
VAL_N = 500          # 로깅 전용. in-train 이므로 checkpoint 선택에 쓰지 않는다.


def job(args):
    stem, split = args
    ann = os.path.join(BROAD, "labels", f"{stem}_label.json")
    img = os.path.join(BROAD, "rgb", f"{stem}_rgb.png")
    kps = PY.load_kps(ann)
    if kps is None:
        return "no_annotation"
    image = cv2.imread(img)
    if image is None:
        return "unreadable_image"
    padded = cv2.copyMakeBorder(image, PY.PAD, PY.PAD, PY.PAD, PY.PAD,
                                cv2.BORDER_REFLECT_101)
    ph, pw = padded.shape[:2]
    line = PY.to_line(pw, ph, [(x + PY.PAD, y + PY.PAD, k) for x, y, k in kps])
    if line is None:
        return "all_kp_outside"
    cv2.imwrite(os.path.join(OUT, "images", split, f"{stem}.png"), padded)
    with open(os.path.join(OUT, "labels", split, f"{stem}.txt"), "w") as fh:
        fh.write(line + "\n")
    return "ok"


def main():
    stems = [i["frame_id"] for i in json.load(open(MANIFEST))["items"]]
    stems.sort()
    val = set(stems[::len(stems) // VAL_N][:VAL_N])
    jobs = [(s, "val" if s in val else "train") for s in stems]
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

    from collections import Counter
    counts = Counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, result in enumerate(pool.map(job, jobs, chunksize=64), 1):
            counts[result] += 1
            if i % 5000 == 0:
                print(f"    {i}/{len(jobs)}  {dict(counts)}", flush=True)
    with open(os.path.join(OUT, "data.yaml"), "w") as fh:
        fh.write(f"path: {OUT}\ntrain: images/train\nval: images/val\n"
                 "nc: 1\nkpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n"
                 "names:\n  0: pallet\n")
    json.dump({"source": "FINAL_SYNTH_TRAIN_V1 (BROAD 40,000)",
               "contract": "prepare_yolo_pose (PAD=100 REFLECT_101, v=in-canvas)",
               "counts": dict(counts),
               "val_note": f"{VAL_N} frames held out for logging ONLY. They are "
                           f"in the same pool FINAL40K trained on, so no "
                           f"checkpoint selection may use them — take last.pt.",
               }, open(os.path.join(OUT, "_build.json"), "w"), indent=1)
    print(f"  {dict(counts)}  -> {OUT}")


if __name__ == "__main__":
    main()
