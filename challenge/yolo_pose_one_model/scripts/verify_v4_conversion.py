"""Verify that the target-synth sets (v1/v2) are fully converted to camera-facing 0123 (v4).

The conversion script (scripts/annotate/convert_to_camera_facing_v4.py) rewrites
projected_cuboid and only writes a .orig backup when the permutation is NOT identity.
So a missing .orig is ambiguous: either "already in v4 order" or "never converted".

This resolves the ambiguity by re-running the same permutation solver on the current
JSON. If every frame now yields the identity permutation, the set is provably v4.

Usage:
  python challenge/yolo_pose_one_model/scripts/verify_v4_conversion.py --n 500
  python challenge/yolo_pose_one_model/scripts/verify_v4_conversion.py --all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "challenge" / "scripts"))

# 어노테이션 툴은 2026-08-15 에 scripts/annotate/ 로 옮겼다.
sys.path.insert(0, str(REPO / "scripts" / "annotate"))
sys.path[:0] = [str(REPO / "challenge" / "scripts" / _s) for _s in ("infer", "live")]
from convert_to_camera_facing_v4 import compute_perm_v4, get_origin_3d  # noqa: E402
from challenge.data_paths import get as dp  # noqa: E402

IDENTITY = [0, 1, 2, 3, 4, 5, 6, 7]


def check(path):
    """Return (verdict, detail). verdict in {identity, needs_perm, skip}."""
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        return "skip", f"read_fail:{type(e).__name__}"
    objs = data.get("objects") or []
    if not objs:
        return "skip", "no_objects"
    obj = objs[0]
    proj = obj.get("projected_cuboid")
    if not proj or len(proj) < 8:
        return "skip", "no_proj"
    origin = get_origin_3d(obj)
    if origin is None:
        return "skip", "no_3d_corner"
    perm = compute_perm_v4(origin, proj)
    if perm is None:
        return "skip", "degenerate"
    return ("identity" if perm[:8] == IDENTITY else "needs_perm"), str(perm[:8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="samples per set (0/--all = every file)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rc = 0
    for key in ("synth.v1", "synth.v2"):
        root = dp(key, absolute=True)
        files = [p for p in glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
                 if not p.endswith(".orig")]
        files.sort()
        n_orig = len(glob.glob(os.path.join(root, "**", "*.json.orig"), recursive=True))
        picked = files if (args.all or args.n <= 0) else random.Random(args.seed).sample(
            files, min(args.n, len(files)))

        verdicts, details = Counter(), Counter()
        offenders = []
        for f in picked:
            v, d = check(f)
            verdicts[v] += 1
            if v != "identity":
                details[d] += 1
                if len(offenders) < 5:
                    offenders.append((os.path.relpath(f, REPO), d))

        print(f"\n=== {key}  root={os.path.relpath(root, REPO)}")
        print(f"    json={len(files)}  json.orig={n_orig}  checked={len(picked)}")
        for k, v in verdicts.most_common():
            print(f"    {k:<12} {v}")
        if details:
            print("    non-identity detail:", dict(details))
            for o in offenders:
                print("      ", o)
        if verdicts["needs_perm"]:
            rc = 1
            print("    VERDICT: NOT fully v4  <-- BLOCKER")
        elif verdicts["identity"] == 0:
            rc = 1
            print("    VERDICT: could not verify (no checkable frame)  <-- BLOCKER")
        else:
            print(f"    VERDICT: v4 confirmed on {verdicts['identity']}/{len(picked)} checked"
                  f" ({verdicts['skip']} unverifiable)")
    sys.exit(rc)


if __name__ == "__main__":
    main()
