"""finetuning 셋 조립 — real positive / negative 를 oversample 하고 합성을 섞는다.

real 157장은 합성 73,916장의 0.2% 다. 그대로 이어붙이면 매 epoch 거의 등장하지 않아
학습에 반영되지 않고, 반대로 real 만 쓰면 합성에서 배운 keypoint 구조를 잊는다
(catastrophic forgetting). 그래서 real/negative 를 반복 노출시키고 합성은 subsample 한다.

반복은 유일한 이름의 symlink 다 — 같은 경로를 두 번 나열하면 로더가 중복 제거해 버린다
(build_stage_dataset.py 와 같은 방식).

val 은 합성 val 에서만 뽑는다. real 을 val 로 쓰면 그게 model selection 에 들어가는데,
정본 161장은 평가 전용이고 final-test 4세션은 model selection 자체가 금지다.
최종 판정은 학습 후 runs_ft/PURPOSE.md 의 지표 3종으로 따로 한다.

사용:
  python .../build_ft_dataset.py --out datasets/ft_a \
      --real-repeat 20 --neg-repeat 6 --synth 12000 --synth-val 1000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO / "challenge/yolo_pose_one_model"
STAGE_A = OUT_ROOT / "datasets/stage_a"


def link(src, dst):
    """symlink -> hardlink -> copy 순으로 시도."""
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
        return
    except OSError:
        pass
    try:
        os.link(src, dst)
        return
    except OSError:
        shutil.copy2(src, dst)


def add_repeats(img_dir, lbl_dir, prefix, repeat):
    """이미 있는 prefix 파일들을 repeat-1 번 더 복제(유일한 이름)."""
    base = sorted(p for p in glob.glob(str(img_dir / f"{prefix}*.png"))
                  if "__rep" not in os.path.basename(p))
    n = 0
    for p in base:
        stem = os.path.basename(p)[:-4]
        lbl = lbl_dir / f"{stem}.txt"
        for r in range(1, repeat):
            link(p, str(img_dir / f"{stem}__rep{r}.png"))
            link(str(lbl), str(lbl_dir / f"{stem}__rep{r}.txt"))
            n += 1
    return len(base), n


def add_synth(img_dir, lbl_dir, split, k, seed):
    src_i = STAGE_A / "images" / split
    src_l = STAGE_A / "labels" / split
    files = sorted(os.path.basename(p) for p in glob.glob(str(src_i / "*.png")))
    if k and k < len(files):
        files = random.Random(seed).sample(files, k)
    for f in files:
        stem = f[:-4]
        link(str(src_i / f), str(img_dir / f"synth__{f}"))
        link(str(src_l / f"{stem}.txt"), str(lbl_dir / f"synth__{stem}.txt"))
    return len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--real-repeat", type=int, default=20)
    ap.add_argument("--neg-repeat", type=int, default=6)
    ap.add_argument("--synth", type=int, default=12000)
    ap.add_argument("--synth-val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = OUT_ROOT / args.out
    ti, tl = root / "images/train", root / "labels/train"
    vi, vl = root / "images/val", root / "labels/val"
    for d in (ti, tl, vi, vl):
        d.mkdir(parents=True, exist_ok=True)

    n_real, r_real = add_repeats(ti, tl, "real__", args.real_repeat)
    n_neg, r_neg = add_repeats(ti, tl, "neg__", args.neg_repeat)
    n_syn = add_synth(ti, tl, "train", args.synth, args.seed)
    n_val = add_synth(vi, vl, "val", args.synth_val, args.seed)

    n_train = len(glob.glob(str(ti / "*.png")))
    yaml = root / "data.yaml"
    yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\n"
        f"kpt_shape: [9, 3]\n"
        # contract/pallet_pose_contract.yaml 과 stage_a 가 쓰는 값. 좌우 반전 시
        # 0<->1, 2<->3, 4<->5, 6<->7. fliplr=0.0 이라 실제로 안 쓰이지만 계약이
        # 어긋나 있으면 나중에 누가 flip 을 켤 때 조용히 라벨이 깨진다.
        f"flip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n"
        f"names:\n  0: pallet\n", encoding="utf-8")

    stat = {"real_base": n_real, "real_repeat": args.real_repeat,
            "real_total": n_real * args.real_repeat,
            "neg_base": n_neg, "neg_repeat": args.neg_repeat,
            "neg_total": n_neg * args.neg_repeat,
            "synth_train": n_syn, "synth_val": n_val,
            "train_total": n_train, "seed": args.seed}
    json.dump(stat, open(root / "_build_ft.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(stat, indent=2, ensure_ascii=False))
    print(f"\ndata.yaml -> {yaml}")
    print(f"train {n_train}장  (real {n_real*args.real_repeat} / "
          f"neg {n_neg*args.neg_repeat} / synth {n_syn}) | val {n_val}")
    print(f"negative 비중 {n_neg*args.neg_repeat/max(1,n_train)*100:.1f}%")


if __name__ == "__main__":
    main()
