"""restore_orig.py — .orig 백업으로 JSON 복원."""
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import argparse
import glob
import os
import shutil
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3]))
from challenge.data_paths import get as _dp  # 경로 단일 출처


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "data/pallet/training_data/mixed_v8_train",
        _dp("synth.v1"),
        _dp("synth.v2"),
    ])
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    n_restored = 0
    n_no_orig = 0
    for r in args.roots:
        root = r if os.path.isabs(r) else os.path.join(repo_root, r)
        for orig in glob.glob(os.path.join(root, "**", "*.json.orig"), recursive=True):
            target = orig[:-5]   # strip .orig
            if args.dry_run:
                print(f"  [DRY] would restore: {orig} → {target}")
            else:
                shutil.copy2(orig, target)
                os.remove(orig)
            n_restored += 1
    print(f"\n[Done] restored {n_restored} files {'(DRY)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
