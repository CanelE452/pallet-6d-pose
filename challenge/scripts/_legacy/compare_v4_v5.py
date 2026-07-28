"""compare_v4_v5.py — 같은 sample 에 대한 v4 / v5 perm 비교 (2026-05-22).

각 file 에 대해:
  - 원본 JSON 로딩 (.orig 가 있으면 .orig, 없으면 현재 file)
  - v4 perm 계산 (compute_perm_v4)
  - v5 perm 계산 (compute_perm_v5)
  - 두 perm 일치 / 불일치 통계

출력:
  per-root 통계 + 차이가 큰 (perm 다른) sample 의 file path list 일부
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_camera_facing_v4 import compute_perm_v4, get_origin_3d
from convert_to_camera_facing_v5 import compute_perm_v5, get_pts_cam


def load_original(json_path):
    """Load .orig if exists else current file."""
    bak = json_path + ".orig"
    src = bak if os.path.exists(bak) else json_path
    with open(src, "r", encoding="utf-8") as f:
        return json.load(f), src


def gather(roots):
    files = []
    for r in roots:
        for p in glob.glob(os.path.join(r, "**", "*.json"), recursive=True):
            if p.endswith(".orig"):
                continue
            files.append(p)
    return sorted(files)


def perm_compare(json_path):
    """Compute (v4_perm, v5_perm, status) on first object of file."""
    try:
        data, src = load_original(json_path)
    except Exception as e:
        return None, None, f"read_fail: {e}"
    objs = data.get("objects", [])
    if not objs:
        return None, None, "no_obj"
    obj = objs[0]
    proj = obj.get("projected_cuboid")
    if not proj or len(proj) < 8:
        return None, None, "no_proj"
    origin = get_origin_3d(obj)
    if origin is None:
        return None, None, "no_origin"
    pts_cam = get_pts_cam(data, obj)
    if pts_cam is None:
        return None, None, "no_cam"
    p4 = compute_perm_v4(origin, proj)
    p5 = compute_perm_v5(origin, pts_cam, proj)
    if p4 is None and p5 is None:
        return None, None, "both_degenerate"
    if p4 is None:
        return None, p5, "v4_degenerate"
    if p5 is None:
        return p4, None, "v5_degenerate"
    return p4, p5, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "data/pallet/training_data/mixed_v8_train",
        "challenge/data/training/v1",
        "challenge/data/training/v2",
    ])
    ap.add_argument("--n_diff_show", type=int, default=20,
                    help="how many differing-perm samples to print per root")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    roots = [r if os.path.isabs(r) else os.path.join(repo_root, r) for r in args.roots]
    files = gather(roots)
    print(f"[Found] {len(files)} JSON files")

    def which_root(fp):
        for r in roots:
            if fp.startswith(r):
                return r
        return roots[0]

    stats = {}
    diff_examples = {}
    diff_patterns = {}   # perm-diff signature → count

    for r in roots:
        stats[r] = {"total": 0, "agree_identity": 0, "agree_perm": 0,
                    "disagree": 0, "fail": 0,
                    "v4_id_v5_perm": 0, "v4_perm_v5_id": 0, "both_perm_diff": 0}
        diff_examples[r] = []
        diff_patterns[r] = {}

    for i, fp in enumerate(files):
        rk = which_root(fp)
        stats[rk]["total"] += 1
        p4, p5, status = perm_compare(fp)
        if status != "ok":
            stats[rk]["fail"] += 1
            continue
        p4_8, p5_8 = p4[:8], p5[:8]
        id_perm = [0, 1, 2, 3, 4, 5, 6, 7]
        v4_id = (p4_8 == id_perm)
        v5_id = (p5_8 == id_perm)
        if p4_8 == p5_8:
            if v4_id:
                stats[rk]["agree_identity"] += 1
            else:
                stats[rk]["agree_perm"] += 1
        else:
            stats[rk]["disagree"] += 1
            if v4_id and not v5_id:
                stats[rk]["v4_id_v5_perm"] += 1
            elif v5_id and not v4_id:
                stats[rk]["v4_perm_v5_id"] += 1
            else:
                stats[rk]["both_perm_diff"] += 1
            # pattern: which corners differ
            differing = tuple(j for j in range(8) if p4_8[j] != p5_8[j])
            diff_patterns[rk][differing] = diff_patterns[rk].get(differing, 0) + 1
            if len(diff_examples[rk]) < args.n_diff_show:
                rel = os.path.relpath(fp, repo_root)
                diff_examples[rk].append((rel, p4_8, p5_8))
        if i % 3000 == 0 and i > 0:
            print(f"  [{i}/{len(files)}]")

    print()
    print("=" * 72)
    grand = {k: 0 for k in next(iter(stats.values())).keys()}
    for r, s in stats.items():
        if s["total"] == 0:
            continue
        rel = os.path.basename(r.rstrip("/").rstrip("\\"))
        print(f"\n[{rel}]")
        print(f"  Total                : {s['total']}")
        print(f"  Agree identity       : {s['agree_identity']:5d}  ({100*s['agree_identity']/s['total']:.1f}%)")
        print(f"  Agree permutation    : {s['agree_perm']:5d}  ({100*s['agree_perm']/s['total']:.1f}%)")
        print(f"  Disagree             : {s['disagree']:5d}  ({100*s['disagree']/s['total']:.1f}%)")
        print(f"    v4 id, v5 perm     : {s['v4_id_v5_perm']:5d}")
        print(f"    v4 perm, v5 id     : {s['v4_perm_v5_id']:5d}")
        print(f"    both perm but diff : {s['both_perm_diff']:5d}")
        print(f"  Fail                 : {s['fail']:5d}")
        for k in grand:
            grand[k] += s[k]
        # top diff patterns
        if diff_patterns[r]:
            top_pat = sorted(diff_patterns[r].items(), key=lambda x: -x[1])[:5]
            print(f"  Top diff signatures (differing indices):")
            for pat, cnt in top_pat:
                print(f"    {pat}  → {cnt}")

    print()
    print("-" * 72)
    print(f"[GRAND TOTAL]")
    print(f"  Total            : {grand['total']}")
    print(f"  Agree identity   : {grand['agree_identity']:5d}  ({100*grand['agree_identity']/grand['total']:.1f}%)")
    print(f"  Agree permutation: {grand['agree_perm']:5d}  ({100*grand['agree_perm']/grand['total']:.1f}%)")
    print(f"  Agree (TOTAL)    : {grand['agree_identity']+grand['agree_perm']:5d}  ({100*(grand['agree_identity']+grand['agree_perm'])/grand['total']:.1f}%)")
    print(f"  Disagree         : {grand['disagree']:5d}  ({100*grand['disagree']/grand['total']:.1f}%)")
    print(f"    v4 id, v5 perm : {grand['v4_id_v5_perm']:5d}")
    print(f"    v4 perm, v5 id : {grand['v4_perm_v5_id']:5d}")
    print(f"    both diff perm : {grand['both_perm_diff']:5d}")
    print(f"  Fail             : {grand['fail']:5d}")

    print()
    print("=" * 72)
    print("Diff examples (first N per root):")
    for r, ex_list in diff_examples.items():
        if not ex_list:
            continue
        rel = os.path.basename(r.rstrip("/").rstrip("\\"))
        print(f"\n[{rel}] {len(ex_list)} shown")
        for path, p4, p5 in ex_list:
            print(f"  {path}")
            print(f"    v4: {p4}")
            print(f"    v5: {p5}")


if __name__ == "__main__":
    main()
