"""stage0/pl_pass_count.py — ★ probe go/no-go counter.

CPU-only. Reads the _records.json produced by extract_pl_v1.py (which already
carries n_detected, diag_pass, flip_score per pool frame) and reports, PER DOMAIN
(outside / night):
    detectable  (n_detected >= 6)
    diag-pass   (detectable AND diag_pass)
    diag∧flip   (diag-pass AND flip_score <= flip_tau)

These counts are the go/no-go signal for the self-training probe:
    수십 장 이상 통과  -> probe go (run a self-training round)
    한 자릿수          -> base 보강 직행 (필터 천장이 너무 낮음)

detectable def (n_detected>=6) matches filter_pr_camfacing's PnP min_pts / the
hungarian "valid.sum()>=6" gate — same convention, no new geometry.

Usage:
  python scripts/stage0/pl_pass_count.py \
      --records data/pallet/pl/stage0_paper_base/_records.json
  (sweep flip_tau)  --flip_taus 5 8 10 15
"""
from __future__ import annotations
import argparse
import json
import os

DOMAINS = ["outside", "night"]
DET_MIN = 6


def count(records, flip_tau):
    out = {}
    for dom in DOMAINS + ["ALL"]:
        rows = (records if dom == "ALL"
                else [r for r in records if r["domain"] == dom])
        det = [r for r in rows if r["n_detected"] >= DET_MIN]
        diag = [r for r in det if r["diag_pass"]]
        diagflip = [r for r in diag
                    if r["flip_score"] is not None and r["flip_score"] <= flip_tau]
        out[dom] = {
            "total": len(rows), "detectable": len(det),
            "diag": len(diag), "diag_and_flip": len(diagflip),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True,
                    help="_records.json from extract_pl_v1.py")
    ap.add_argument("--flip_taus", type=float, nargs="+", default=[5, 8, 10, 15],
                    help="flip_score thresholds for diag∧flip count (px)")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    records = json.load(open(args.records))
    print(f"[pl_pass_count] {len(records)} frames from {args.records}\n")

    report = {"records": args.records, "det_min": DET_MIN, "sweeps": {}}
    for tau in args.flip_taus:
        c = count(records, tau)
        report["sweeps"][str(tau)] = c
        print(f"=== flip_tau = {tau:g}px ===")
        hdr = f"{'domain':<9}{'total':>7}{'detectable':>12}{'diag':>7}{'diag∧flip':>11}"
        print(hdr)
        print("-" * 46)
        for dom in DOMAINS + ["ALL"]:
            d = c[dom]
            print(f"{dom:<9}{d['total']:>7}{d['detectable']:>12}"
                  f"{d['diag']:>7}{d['diag_and_flip']:>11}")
        print()

    # go/no-go hint at the middle tau
    mid = args.flip_taus[len(args.flip_taus) // 2]
    cm = report["sweeps"][str(mid)]
    print(f"[go/no-go @ flip_tau={mid:g}]  per-domain diag∧flip:")
    for dom in DOMAINS:
        n = cm[dom]["diag_and_flip"]
        verdict = "probe GO" if n >= 10 else "few -> base 보강 검토"
        print(f"   {dom:<8} diag∧flip={n:<4} -> {verdict}")

    out_json = args.out_json or os.path.join(
        os.path.dirname(args.records), "_pass_count.json")
    json.dump(report, open(out_json, "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] {out_json}")


if __name__ == "__main__":
    main()
