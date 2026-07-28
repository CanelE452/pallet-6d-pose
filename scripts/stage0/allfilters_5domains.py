"""allfilters_5domains.py — 5개 도메인(outside/night/manual = filterval, cad/noapril =
testset17 지정 평가셋) 전부에서, GT-free 필터 전체 AND(f1~f7, ★f3 flip 포함) 통과 수를
Stage B(squash)로 집계. f3 제외(배포) 대비도 병기해 f3(고장) 영향 표시.

재사용: F(filterval, T.flip_infer_squash=(W-1)-x 패치 적용) / T(build_rec/M/TAU) /
        S.frames_filterval / T.read_manifest(testset17 manifest).
FULL7 = f1&f2&f3&f4&f5&f6&f7 (GT-free 전체). DEPLOY6 = f3 제외. f8/f9=GT필요 제외.
purity = 통과 중 GT-good(corner_med<10px) 비율.

Usage: conda activate pallet-pose; python scripts/stage0/allfilters_5domains.py
"""
from __future__ import annotations
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

import paper_s2_filterval_9filters as F   # noqa: E402,F401 (applies (W-1)-x flip to T)
import paper_s2_testset17_9filters as T   # noqa: E402
import stage25_paperbase_eval as S        # noqa: E402
import annotate_pnp as APNP               # noqa: E402
import torch                              # noqa: E402

M, TAU, E = T.M, T.TAU, T.E
N_DET_MIN, GOOD_PX = T.N_DET_MIN, T.GOOD_PX
MANIFEST = os.path.join(
    ROOT, "data/pallet/eval_results/stage22_myannot_eval/testset_full8_manifest.txt")
FULL7 = ["f1_peak", "f2_peak_ratio", "f3_flip", "f4_tta_stab",
         "f5_rear_conf", "f6_frsep", "f7_posdepth"]
DEPLOY6 = [f for f in FULL7 if f != "f3_flip"]
ENV = {"size_lo": 0.0, "size_hi": 1.0, "asp_lo": 0.0, "asp_hi": 10.0}
DOMAINS = ["outside", "cad", "manual", "noapril", "night"]


def passes(r, filters):
    if r["n_det"] < N_DET_MIN:
        return False
    row = {"n_det": r["n_det"], "scores": r["scores"], "f7_posdepth": r["f7_posdepth"],
           "pred_sr": r["pred_sr"], "pred_asp": r["pred_asp"]}
    return all(M.apply_filter(f, row, TAU.get(f), ENV) for f in filters)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    APNP.PALLET_DIMS = (1.1, 1.3, 0.12)
    model = E.load_model(T.WEIGHTS, device)

    frames = list(S.frames_filterval())                    # outside/night/manual
    frames += [(dom, str(fid), jp, ip) for dom, fid, jp, ip in T.read_manifest(MANIFEST)]  # cad/noapril
    print(f"[all] {len(frames)} frames across 5 domains; Stage B squash; flip=(W-1)-x")

    recs = []
    for dom, fid, jp, ip in frames:
        r = T.build_rec(model, dom, fid, jp, ip, device)
        if r is not None:
            r["dom"] = dom
            recs.append(r)

    L = []
    L.append("# 5개 도메인 필터 통과 수 — GT-free 전체 AND (f1~f7, f3 flip 포함) — Stage B")
    L.append("")
    L.append("- weights: paper_s2_stageB net_epoch_0057, squash-parity, flip=(W-1)-x")
    L.append("- FULL7 = f1&f2&f3&f4&f5&f6&f7 (GT-free 전체). DEPLOY6 = f3 제외(배포).")
    L.append("- f8/f9 = GT 필요라 배포 불가로 제외. purity = 통과 중 GT-good(corner_med<10px).")
    L.append("- ★f3(flip) = 모델 좌우 비대칭으로 정상도 대량 탈락(broken) → FULL7 급감 예상.")
    L.append("")
    L.append("```")
    L.append(f"{'domain':<9}{'N':>4}{'det>=6':>8}{'GTgood':>8}"
             f"{'FULL7 pass':>12}{'purity':>8}   {'DEPLOY6 pass':>13}{'purity':>8}")
    L.append("-" * 80)

    def line(name, rs):
        det = [r for r in rs if r["n_det"] >= N_DET_MIN]
        gtg = sum(1 for r in rs if r["good"])
        p7 = [r for r in det if passes(r, FULL7)]
        p6 = [r for r in det if passes(r, DEPLOY6)]
        pur7 = f"{100*sum(x['good'] for x in p7)/len(p7):.0f}%" if p7 else "  -"
        pur6 = f"{100*sum(x['good'] for x in p6)/len(p6):.0f}%" if p6 else "  -"
        return (f"{name:<9}{len(rs):>4}{len(det):>8}{gtg:>8}"
                f"{len(p7):>12}{pur7:>8}   {len(p6):>13}{pur6:>8}")

    for dom in DOMAINS:
        L.append(line(dom, [r for r in recs if r["dom"] == dom]))
    L.append("-" * 80)
    L.append(line("ALL", recs))
    L.append("```")
    L.append("")
    L.append("해석: FULL7(f3 포함)은 f3 하나 때문에 통과가 급감(정상 프레임도 탈락) = f3 broken 확인. "
             "purity(통과 중 실제 정확 비율)는 도메인 base 품질 종속: outside/manual↑ night↓.")

    out = os.path.join(ROOT, "data/pallet/eval_results/paper_s2_scratch_diffpnp",
                       "allfilters_5domains.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
