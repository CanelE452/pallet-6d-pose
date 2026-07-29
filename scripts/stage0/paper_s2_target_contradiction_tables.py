"""paper_s2_target_contradiction_tables.py — Step 3 (Gate A / H1).

target_semantics_keypoints.parquet 를 받아 C0~C4 를 집계하고 Table 1~5 +
핵심수치 A/B/C/D 를 낸다.  판정은 별도 — 여기서는 관찰만 출력한다.

카테고리 (belief 50x50, sigma=2 -> w=int(2*sigma)=4)
  C0 center_inside=F, nonzero=F                      정상 outside/missing
  C1 center_inside=T, full_support=T, nonzero=T      정상 interior
  C2 center_inside=T, full_support=F, nonzero=F, mask=1
        -> border-positive 가 background-negative 로 supervise (의심 대상)
  C3 center_inside=T, nonzero=F, mask=0              partial supervision 으로 제외
  C4 center_inside=F, nonzero=T                      비정상 target
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "pallet", "results",
                   "paper_s2_target_semantics_audit")
PQ = os.path.join(OUT, "target_semantics_keypoints.parquet")

NEAR, FAR = [0, 1, 2, 3], [4, 5, 6, 7]
TOP, BOTTOM = [0, 1, 4, 5], [2, 3, 6, 7]


def categorize(df):
    ci = df.center_inside_belief
    fs = df.full_gaussian_support_inside
    nz = df.belief_target_nonzero
    m1 = df.belief_channel_mask == 1
    cat = pd.Series(index=df.index, dtype="object")
    cat[:] = "other"
    cat[(~ci) & (~nz)] = "C0"
    cat[ci & fs & nz] = "C1"
    cat[ci & (~fs) & (~nz) & m1] = "C2"
    cat[ci & (~nz) & (~m1)] = "C3"
    cat[(~ci) & nz] = "C4"
    return cat


def pct(a, b):
    return round(100.0 * a / b, 3) if b else float("nan")


def main():
    df = pd.read_parquet(PQ)
    df["cat"] = categorize(df)
    n = len(df)
    print(f"[rows] {n} keypoints  ({df.frame_id.nunique()} unique frame_id, "
          f"{df.dataset.nunique()} datasets)")
    print(f"[sanity] formula-vs-actual mismatch rows = "
          f"{int(df.formula_actual_mismatch.sum())}\n")

    # ---------------- Table 1 : dataset x C0..C4 -------------------------
    t1 = (df.groupby(["dataset", "cat"]).size().unstack(fill_value=0))
    for c in ["C0", "C1", "C2", "C3", "C4", "other"]:
        if c not in t1.columns:
            t1[c] = 0
    t1 = t1[["C0", "C1", "C2", "C3", "C4", "other"]]
    t1["n_kp"] = t1.sum(axis=1)
    t1["n_frames"] = df.groupby("dataset").frame_id.nunique()
    for c in ["C0", "C1", "C2", "C3", "C4"]:
        t1[c + "_pct"] = (100 * t1[c] / t1["n_kp"]).round(3)
    print("=== Table 1. dataset별 C0~C4 ===")
    print(t1.to_string())
    t1.to_csv(os.path.join(OUT, "table1_dataset_categories.csv"))

    # ---------------- 핵심수치 A/B/C/D -----------------------------------
    ci, nz = df.center_inside_belief, df.belief_target_nonzero
    m1 = df.belief_channel_mask == 1
    A = int((ci & ~nz).sum())
    B = int((ci & ~nz & m1).sum())
    far_mask = df.keypoint_id.isin(FAR)
    k56 = df.keypoint_id.isin([5, 6])
    Bfar = int((ci & ~nz & m1 & far_mask).sum())
    B56 = int((ci & ~nz & m1 & k56).sum())
    print("\n=== 핵심수치 ===")
    print(f"  A. center_inside=T & target all-zero            = {A} "
          f"({pct(A, n)}% of {n})")
    print(f"  B. A 중 belief_channel_mask=1                   = {B} "
          f"({pct(B, n)}%)")
    print(f"  C. kp4-7 에서 B                                 = {Bfar} "
          f"({pct(Bfar, int(far_mask.sum()))}% of far kp)")
    print(f"  D. kp5/kp6 에서 B                               = {B56} "
          f"({pct(B56, int(k56.sum()))}% of kp5/6)")
    print(f"  C4 (center outside 인데 target nonzero)         = "
          f"{int(((~ci) & nz).sum())}")
    print(f"  mask==0 인 keypoint 총수                        = "
          f"{int((~m1).sum())}")

    # ---------------- Table 2 : keypoint id별 C2 -------------------------
    g = df.groupby("keypoint_id")
    t2 = pd.DataFrame({
        "n_kp": g.size(),
        "C2": g.apply(lambda x: int((x.cat == "C2").sum())),
        "center_inside_pct": (100 * g.center_inside_belief.mean()).round(2),
        "outside_pct": (100 * (1 - g.center_inside_belief.mean())).round(2),
    })
    t2["C2_pct"] = (100 * t2.C2 / t2.n_kp).round(3)
    print("\n=== Table 2. keypoint id별 C2 ===")
    print(t2.to_string())
    t2.to_csv(os.path.join(OUT, "table2_keypoint_C2.csv"))

    # ---------------- Table 3 : near vs far ------------------------------
    df["kp_group"] = np.where(df.keypoint_id.isin(NEAR), "near(0-3)",
                              np.where(df.keypoint_id.isin(FAR), "far(4-7)",
                                       "centroid(8)"))
    t3 = df.groupby("kp_group").apply(lambda x: pd.Series({
        "n_kp": len(x),
        "C2": int((x.cat == "C2").sum()),
        "C2_pct": pct(int((x.cat == "C2").sum()), len(x)),
        "outside_pct": round(100 * (~x.center_inside_belief).mean(), 2),
    }))
    print("\n=== Table 3. near vs far ===")
    print(t3.to_string())
    t3.to_csv(os.path.join(OUT, "table3_near_far.csv"))

    # ---------------- Table 4 : trunc vs non-trunc -----------------------
    df["trunc_grp"] = np.where(df.dataset == "aug_trunc_v2",
                               "aug_trunc_v2", "non-trunc datasets")
    t4 = df.groupby("trunc_grp").apply(lambda x: pd.Series({
        "n_kp": len(x),
        "C2": int((x.cat == "C2").sum()),
        "C2_pct": pct(int((x.cat == "C2").sum()), len(x)),
        "outside_pct": round(100 * (~x.center_inside_belief).mean(), 2),
    }))
    print("\n=== Table 4. trunc vs non-trunc ===")
    print(t4.to_string())
    t4.to_csv(os.path.join(OUT, "table4_trunc_vs_not.csv"))

    # ---------------- Table 5 : mask=1 & all-zero ------------------------
    t5 = df[ci & ~nz & m1].groupby(["dataset", "keypoint_id"]).size()
    print("\n=== Table 5. mask=1 인데 target all-zero (dataset x kp) ===")
    if len(t5):
        print(t5.unstack(fill_value=0).to_string())
        t5.unstack(fill_value=0).to_csv(
            os.path.join(OUT, "table5_mask1_allzero.csv"))
    else:
        print("  (없음)")

    # ---------------- aug_kind 분해 --------------------------------------
    t6 = df.groupby("aug_kind").apply(lambda x: pd.Series({
        "n_kp": len(x),
        "C2_pct": pct(int((x.cat == "C2").sum()), len(x)),
        "outside_pct": round(100 * (~x.center_inside_belief).mean(), 2),
        "C0_pct": pct(int((x.cat == "C0").sum()), len(x)),
    }))
    print("\n=== aug_kind 분해 ===")
    print(t6.to_string())

    # ---------------- border-distance bin --------------------------------
    inside = df[df.center_inside_belief].copy()
    inside["bin"] = pd.cut(inside.dist_to_border,
                           [-0.001, 1, 2, 3, 4, 6, 10, 100],
                           labels=["0-1", "1-2", "2-3", "3-4", "4-6",
                                   "6-10", "10+"])
    t7 = inside.groupby("bin").apply(lambda x: pd.Series({
        "n_kp": len(x),
        "C2": int((x.cat == "C2").sum()),
        "C2_pct": pct(int((x.cat == "C2").sum()), len(x)),
    }))
    print("\n=== center_inside keypoint 의 border-distance bin (belief px) ===")
    print(t7.to_string())
    t7.to_csv(os.path.join(OUT, "table7_border_bins.csv"))

    # dataset summary csv
    summ = t1.reset_index()
    summ.to_csv(os.path.join(OUT, "dataset_summary.csv"), index=False)
    t2.reset_index().to_csv(os.path.join(OUT, "keypoint_summary.csv"),
                            index=False)
    print(f"\n[saved] tables -> {OUT}")


if __name__ == "__main__":
    main()
