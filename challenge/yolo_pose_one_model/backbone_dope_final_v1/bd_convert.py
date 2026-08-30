"""최종 학습셋(YOLO 포맷) -> DOPE 포맷.

프레임 집합을 새로 고르지 않는다.  `g38_legacy_v1v2_p0_tex20k` 의 파일 목록을
그대로 읽어서 같은 프레임만 변환한다 — 그래야 YOLO26n CLEANSTART 와의 비교가
"같은 데이터" 라고 말할 수 있다.

이미지는 재인코딩하지 않고 **이미 pad 된 데이터셋 PNG 를 symlink** 한다.
라벨은 원본 JSON 의 projected_cuboid 를 PAD 만큼 옮긴다 (실측 확인: 정확히 100px).

    720x480 원본  --pad 100-->  920x680  (데이터셋 PNG 가 이미 이 상태)
    projected_cuboid[i] + 100  ==  YOLO 라벨의 픽셀 좌표   (실측 대조 완료)
"""
from __future__ import annotations

import json
import os
import sys

ROOT = "/home/minjae/Documents/github/pallet-pose"
Y = f"{ROOT}/challenge/yolo_pose_one_model"
SRC_DS = f"{Y}/datasets/g38_legacy_v1v2_p0_tex20k"
OUT = f"{ROOT}/data/pallet/training_data/dope_final_g38_p0_tex"
PAD = 100

# 접두어 -> (rgb 루트, label 루트, 파일명 규칙)
SOURCES = {
    "G38": (f"{ROOT}/data/pallet/training_data/paper_release/v2_prod40k_clean_merged/rgb",
            f"{ROOT}/data/pallet/training_data/paper_release/v2_prod40k_clean_merged/labels",
            "flat"),
    "P0":  (None, f"{Y}/datasets/_raw_legacy_v1v2_p0_10k", "shard"),
    "TEX": (None, f"{Y}/datasets/_raw_legacy_v1v2_p0_tex10k", "shard"),
}


def label_path(prefix, stem):
    """데이터셋 파일명 -> 원본 라벨 JSON 경로."""
    _, lab_root, kind = SOURCES[prefix]
    if kind == "flat":
        # G38__G__f0000.png -> stem 'G__f0000' -> f0000_label.json
        return f"{lab_root}/{stem.split('__')[-1]}_label.json"
    # P0__shard_00_f0000.png -> stem 'shard_00_f0000' -> shard_00/labels/f0000_label.json
    shard, frame = stem.rsplit("_", 1)
    return f"{lab_root}/{shard}/labels/{frame}_label.json"


def convert_one(src_json, src_png, dst_json, dst_png):
    d = json.load(open(src_json, encoding="utf-8"))
    obj = d["objects"][0]

    kps = obj.get("projected_cuboid")
    cen = obj.get("projected_cuboid_centroid")
    if not kps or len(kps) != 8 or cen is None:
        return "bad_keypoints"

    obj["projected_cuboid"] = [[x + PAD, y + PAD] for x, y in kps]
    obj["projected_cuboid_centroid"] = [cen[0] + PAD, cen[1] + PAD]

    cam = d.get("camera_data", {})
    intr = cam.get("intrinsics")
    if isinstance(intr, dict):
        for k in ("cx", "cy"):
            if k in intr:
                intr[k] = intr[k] + PAD
    if "width" in cam:
        cam["width"] = cam["width"] + 2 * PAD
    if "height" in cam:
        cam["height"] = cam["height"] + 2 * PAD
    cam["_pad_applied_px"] = PAD

    json.dump(d, open(dst_json, "w", encoding="utf-8"))
    if os.path.lexists(dst_png):
        os.remove(dst_png)
    os.symlink(src_png, dst_png)
    return "ok"


def run_split(split):
    img_dir = f"{SRC_DS}/images/{split}"
    out_dir = f"{OUT}/{split}"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(os.listdir(img_dir))

    stats, index, i = {}, [], 0
    for name in names:
        prefix, stem = name.split("__", 1)
        stem = stem[:-4]                                   # .png 제거
        src_png = os.path.realpath(f"{img_dir}/{name}")
        src_json = label_path(prefix, stem)
        if not os.path.exists(src_json):
            stats["missing_json"] = stats.get("missing_json", 0) + 1
            continue
        r = convert_one(src_json, src_png,
                        f"{out_dir}/{i:06d}.json", f"{out_dir}/{i:06d}.png")
        stats[r] = stats.get(r, 0) + 1
        if r == "ok":
            index.append({"idx": i, "src": name, "json": os.path.relpath(src_json, ROOT)})
            i += 1
        if i and i % 5000 == 0:
            print(f"  {split} {i}/{len(names)}", flush=True)

    json.dump({"split": split, "n_input": len(names), "n_written": i,
               "stats": stats, "pad_px": PAD,
               "source_dataset": os.path.relpath(SRC_DS, ROOT),
               "index": index},
              open(f"{OUT}/_convert_{split}.json", "w"), indent=1)
    print(f"{split}: 입력 {len(names)} -> 변환 {i}   {stats}", flush=True)
    return i, stats


def main():
    os.makedirs(OUT, exist_ok=True)
    total = {}
    for split in ("train", "val"):
        n, st = run_split(split)
        total[split] = {"n": n, "stats": st}
        if st.get("ok", 0) != n or n == 0:
            print(f"FAIL {split} 불일치 {st}", flush=True)
            return 1
    # 계약 검사 — 프레임 수가 원본 데이터셋과 같아야 한다
    assert total["train"]["n"] == 55980, total["train"]["n"]
    assert total["val"]["n"] == 4020, total["val"]["n"]
    print(f"OK -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
