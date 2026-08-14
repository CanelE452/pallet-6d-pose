"""
challenge/scripts/merge_dataset.py

여러 GT 디렉토리(manual + pseudo) 를 합쳐 train / val NDDS 데이터셋으로 만든다.

사용:
  python challenge/scripts/merge_dataset.py \
      --inputs challenge/data/01_real/manual_gt/capturepallet07_manual_gt \
               challenge/data/01_real/pseudo_gt/capturepallet09_pseudo_gt \
      --out challenge/data/03_derived/_train_v1 --val_fraction 0.15

출력 구조:
  challenge/data/03_derived/_train_v1/
    train/000000.png + 000000.json + ...   # 순차 rename (CleanVisiiDopeLoader 호환)
    val/000000.png + 000000.json + ...
    _manifest.json                           # source 추적
"""

import argparse
import glob
import json
import os
import random
import shutil


def gather(inputs):
    pairs = []
    for d in inputs:
        for j in sorted(glob.glob(os.path.join(d, "*.json"))):
            png = j[:-5] + ".png"
            if os.path.exists(png):
                pairs.append((j, png, os.path.basename(d.rstrip("/\\"))))
    return pairs


def emit(pairs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i, (j, png, src) in enumerate(pairs):
        stem = f"{i:06d}"
        dst_j = os.path.join(out_dir, f"{stem}.json")
        dst_p = os.path.join(out_dir, f"{stem}.png")
        # NDDS loader 호환: stem 기준 매칭. json 내 일부 필드 보존.
        shutil.copy2(j, dst_j)
        try: os.link(png, dst_p)
        except (OSError, NotImplementedError): shutil.copy2(png, dst_p)
        manifest.append({"stem": stem, "source": src, "src_json": j})
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="GT 디렉토리들 (manual_gt, pseudo_gt 등)")
    ap.add_argument("--out", required=True, help="출력 train/val 루트")
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manual_to_val_only", action="store_true",
                    help="manual 만 val 에. pseudo 는 모두 train (권장 — manual 을 held-out 평가용으로)")
    args = ap.parse_args()

    pairs = gather(args.inputs)
    if not pairs:
        print(f"[ERROR] no JSON/PNG pairs found in {args.inputs}")
        return

    if args.manual_to_val_only:
        manuals = [p for p in pairs if "manual" in p[2]]
        pseudos = [p for p in pairs if "pseudo" in p[2]]
        random.Random(args.seed).shuffle(manuals)
        n_val = max(1, int(len(manuals) * args.val_fraction))
        val_pairs = manuals[:n_val]
        train_pairs = manuals[n_val:] + pseudos
    else:
        random.Random(args.seed).shuffle(pairs)
        n_val = max(1, int(len(pairs) * args.val_fraction))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]

    train_dir = os.path.join(args.out, "train")
    val_dir = os.path.join(args.out, "val")
    print(f"[Merge] {len(pairs)} pairs from {len(args.inputs)} dirs")
    print(f"  train → {train_dir}  ({len(train_pairs)})")
    print(f"  val   → {val_dir}    ({len(val_pairs)})")

    train_manifest = emit(train_pairs, train_dir)
    val_manifest = emit(val_pairs, val_dir)

    with open(os.path.join(args.out, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"train": train_manifest, "val": val_manifest,
                   "inputs": args.inputs, "manual_to_val_only": args.manual_to_val_only},
                  f, indent=2, ensure_ascii=False)
    print(f"\n[Done] _manifest.json saved")


if __name__ == "__main__":
    main()
