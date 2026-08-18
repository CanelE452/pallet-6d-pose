"""Rewrite the dataset's alias symlinks from absolute to relative targets.

The aliases were first created with absolute paths, so copying the dataset to another
machine (or just moving the folder) leaves every alias dangling. Since an alias and its
target sit in the same directory, a bare filename is enough and survives any move.

Only symlinks whose target resolves inside the same directory are touched. Anything else
is left alone and reported.

Usage:
  python .../relink_relative.py --dataset datasets/stage_a --dry-run
  python .../relink_relative.py --dataset datasets/stage_a
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "challenge/yolo_pose_one_model"


def fix_dir(d: Path, dry: bool):
    stats = {"already_relative": 0, "rewritten": 0, "outside": 0, "broken": 0}
    for p in sorted(d.iterdir()):
        if not p.is_symlink():
            continue
        tgt = os.readlink(p)
        if not os.path.isabs(tgt):
            stats["already_relative"] += 1
            continue
        tgt_path = Path(tgt)
        if tgt_path.parent != p.parent:
            stats["outside"] += 1
            continue
        if not tgt_path.exists():
            stats["broken"] += 1
            continue
        if not dry:
            tmp = p.with_name(p.name + ".relinking")
            os.symlink(tgt_path.name, tmp)
            os.replace(tmp, p)
        stats["rewritten"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = OUT / args.dataset
    total = {}
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        d = root / sub
        if not d.exists():
            continue
        s = fix_dir(d, args.dry_run)
        print(f"{sub:<14} {s}")
        for k, v in s.items():
            total[k] = total.get(k, 0) + v
    print(f"{'TOTAL':<14} {total}   {'(dry run)' if args.dry_run else ''}")

    # verify: every symlink must resolve
    bad = 0
    for sub in ("images/train", "labels/train"):
        d = root / sub
        for p in d.iterdir():
            if p.is_symlink() and not p.resolve().exists():
                bad += 1
    print(f"dangling symlinks after pass: {bad}")


if __name__ == "__main__":
    main()
