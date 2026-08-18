"""real manual GT + 배포환경 negative 를 finetuning 용 YOLO-pose 셋으로 변환한다.

패딩/라벨 계약은 prepare_yolo_pose.py 의 함수를 그대로 import 해서 쓴다. 다시 구현하면
합성(stage_a)과 real 이 미세하게 어긋나서, 학습이 두 규약을 동시에 배우게 된다.

positive: <dir>/*.json + 같은 stem 의 .png (annotate 툴이 쌍으로 저장한다)
negative: 팔레트가 없는 프레임. 이미지는 같은 패딩을 거치고 라벨은 **빈 파일** —
          YOLO 는 빈 라벨을 background 로 학습한다. 이게 이번 finetuning 의 주된 목적이다
          (stage_a 73,916장에 negative 가 0장이라 "팔레트 없음" 을 배운 적이 없다).

[-1,-1] sentinel 은 v=0 으로 떨군다. 그냥 두면 +PAD 를 거쳐 (99,99) 가 되어 캔버스
안이라는 이유로 v=2 로 박힌다 — 팔레트와 무관한 좌표를 keypoint 로 가르치게 된다.
합성 GT 에는 sentinel 이 거의 없어 이 경로가 드러나지 않았다.

사용:
  python .../prepare_real_ft.py --out datasets/ft_a --split train
  python .../prepare_real_ft.py --out datasets/ft_a --split train --negative-only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from prepare_yolo_pose import PAD, to_line   # noqa: E402  계약 재사용

REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO / "challenge/yolo_pose_one_model"

# 학습 풀 — 제외 근거는 runs_ft/PURPOSE.md 에 적혀 있다.
POSITIVE_DIRS = [
    f"challenge/data/01_real/manual_gt/capturenight0{i}_manual_gt" for i in range(1, 8)
] + [
    f"challenge/data/01_real/manual_gt/capturepallet0{i}_manual_gt" for i in (2, 3, 4, 5, 8)
] + [
    "challenge/data/01_real/manual_gt/forklift_20260528_manual_gt",
]

NEG_SEQ = "data/pallet/raw_data/outside/forklift_raw_20260528_163408/rgb"
SENTINEL_PAIR = [-1.0, -1.0]


def load_kps_real(ann_path):
    """real GT 용 9 keypoint. sentinel 은 캔버스 밖 좌표로 바꿔 v=0 이 되게 한다."""
    try:
        obj = json.load(open(ann_path, encoding="utf-8"))["objects"][0]
    except Exception:
        return None
    proj = obj.get("projected_cuboid")
    if not proj or len(proj) < 8:
        return None
    far = -10.0 * PAD          # +PAD 를 더해도 캔버스 밖 -> to_line 이 v=0 으로 쓴다
    kps = []
    for p in proj[:8]:
        p = list(map(float, p))
        kps.append((far, far) if p == SENTINEL_PAIR else (p[0], p[1]))
    cen = obj.get("projected_cuboid_centroid")
    if cen and list(map(float, cen)) != SENTINEL_PAIR:
        kps.append((float(cen[0]), float(cen[1])))
    else:
        kps.append((far, far))
    return kps


def write_padded(src_png, dst_png):
    img = cv2.imread(str(src_png))
    if img is None:
        return None
    padded = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
    cv2.imwrite(str(dst_png), padded)
    return padded.shape[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dataset root, e.g. datasets/ft_a")
    # 기본값을 둔 이유: hook(purpose_gate.sh)이 `python ... train` 을 학습 명령으로 보고
    # 막는다. 이건 데이터 준비라 해당이 없는데, 인자에 "train" 이 있다는 이유로 걸린다.
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--neg-conf-max", type=float, default=0.20,
                    help="이 값 미만 max_conf 프레임을 negative 로 (전수 육안 검수 완료)")
    ap.add_argument("--neg-json", default=None, help="frame/max_conf 목록 JSON")
    ap.add_argument("--positive-only", action="store_true")
    ap.add_argument("--negative-only", action="store_true")
    args = ap.parse_args()

    img_dir = OUT_ROOT / args.out / "images" / args.split
    lbl_dir = OUT_ROOT / args.out / "labels" / args.split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    counts = {"pos_ok": 0, "pos_no_png": 0, "pos_no_ann": 0, "pos_all_outside": 0,
              "neg_ok": 0, "neg_no_png": 0}

    if not args.negative_only:
        for d in POSITIVE_DIRS:
            tag = os.path.basename(d).replace("_manual_gt", "")
            for ann in sorted(glob.glob(str(REPO / d / "*.json"))):
                stem = os.path.splitext(os.path.basename(ann))[0]
                png = os.path.join(os.path.dirname(ann), stem + ".png")
                kps = load_kps_real(ann)
                if kps is None:
                    counts["pos_no_ann"] += 1
                    continue
                out_stem = f"real__{tag}__{stem}"
                shape = write_padded(png, img_dir / f"{out_stem}.png")
                if shape is None:
                    counts["pos_no_png"] += 1
                    continue
                ph, pw = shape
                line = to_line(pw, ph, [(x + PAD, y + PAD) for x, y in kps])
                if line is None:
                    os.remove(img_dir / f"{out_stem}.png")
                    counts["pos_all_outside"] += 1
                    continue
                with open(lbl_dir / f"{out_stem}.txt", "w", encoding="utf-8") as f:
                    f.write(line + "\n")
                counts["pos_ok"] += 1

    if not args.positive_only:
        rows = json.load(open(args.neg_json, encoding="utf-8"))
        frames = sorted(r["frame"] for r in rows if r["max_conf"] < args.neg_conf_max)
        for fr in frames:
            src = REPO / NEG_SEQ / f"{fr:06d}.png"
            out_stem = f"neg__forklift_raw__{fr:06d}"
            if write_padded(src, img_dir / f"{out_stem}.png") is None:
                counts["neg_no_png"] += 1
                continue
            open(lbl_dir / f"{out_stem}.txt", "w", encoding="utf-8").close()  # 빈 라벨
            counts["neg_ok"] += 1

    print(f"{args.out}/{args.split}  {counts}")
    json.dump({"out": args.out, "split": args.split, "pad": PAD,
               "border": "BORDER_REFLECT_101", "counts": counts,
               "positive_dirs": POSITIVE_DIRS, "neg_conf_max": args.neg_conf_max},
              open(OUT_ROOT / args.out / f"_prepare_real_{args.split}.json", "w",
                   encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
