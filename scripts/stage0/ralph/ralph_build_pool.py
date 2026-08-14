"""ralph_build_pool.py — self_train.py 용 real unlabeled pool 구축 (leak-guarded).

INF.DOMAINS(wood 제외)의 rgb/*.png 를 symlink. eval GT 프레임은 홀드아웃(누수0).
강한 base 실패 실험과 동일 pool 이라 직접 비교 가능.

출력: data/pallet/real_unlabeled_ralph/{session}__{fid}.png
Usage: python -u scripts/stage0/ralph/ralph_build_pool.py
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import glob
import os
import sys

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

import s2_fullpool_infer as INF          # noqa: E402
import s2_fullpool_build_pl as B         # noqa: E402

EXCLUDE = {"wood_indoor", "wood_outdoor"}
OUT = os.path.join(ROOT, "data/pallet/real_unlabeled_ralph")


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*.png")):
        os.unlink(f)
    eval_fids = B.collect_eval_fids()
    print(f"[leak-guard] eval GT held-out: {len(eval_fids)}")
    n, skip = 0, 0
    per = {}
    for dom, seqs in INF.DOMAINS.items():
        if dom in EXCLUDE:
            continue
        for seq in seqs:
            sess = os.path.basename(seq)
            for ip in sorted(glob.glob(os.path.join(seq, "rgb", "*.png"))):
                fid = os.path.splitext(os.path.basename(ip))[0]
                if fid in eval_fids:
                    skip += 1
                    continue
                link = os.path.join(OUT, f"{sess}__{fid}.png")
                if not os.path.exists(link):
                    os.symlink(os.path.abspath(ip), link)
                n += 1
                per[dom] = per.get(dom, 0) + 1
    print(f"[pool] {n} real unlabeled -> {OUT}  (eval-leak skipped: {skip})")
    print("  per-domain:", per)


if __name__ == "__main__":
    main()
