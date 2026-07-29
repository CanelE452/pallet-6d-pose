"""paper_s2_failure_association.py — Step 8.

strict filter-val N87 의 ep57 frozen prediction 을 target-semantics audit 와
**연관 분석으로만** 연결한다.  frame 대응이 아니라 group/통계 수준이며,
여기서 나오는 인과 주장은 전부 [추정] 이다 ([확인]으로 승격하려면 matched
retraining 필요).

입력
  target_semantics_keypoints.parquet  (학습셋 C2 빈도)
  decoder_parity.csv                  (ep57 on N87: peak/err/missing/border)
  truncation_populations.parquet      (real truncated 분류)

주의 (표본/방법 한계)
  * N87 = outside44 + night43.  truncated frame 은 17 개뿐 = 소표본.
  * decoder_parity 의 err 는 index-wise (channel i <-> GT corner i) 이며
    공식 eval 의 order-free/hungarian 과 다르다.  절대값이 아니라 group 간
    상대 비교에만 쓴다.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "pallet", "results",
                   "paper_s2_target_semantics_audit")

NEAR, FAR = [0, 1, 2, 3], [4, 5, 6, 7]


def main():
    tgt = pd.read_parquet(os.path.join(OUT,
                                       "target_semantics_keypoints.parquet"))
    dec = pd.read_csv(os.path.join(OUT, "decoder_parity.csv"))

    ci = tgt.center_inside_belief
    nz = tgt.belief_target_nonzero
    m1 = tgt.belief_channel_mask == 1
    tgt["is_C2"] = ci & ~nz & m1

    # ---- keypoint group 단위 C2 빈도 (학습셋) --------------------------
    c2_by_kp = tgt.groupby("keypoint_id").is_C2.mean().mul(100).round(3)
    print("=== 학습셋 C2 빈도 (%) by keypoint ===")
    print(c2_by_kp.to_string())

    # ---- ep57 on N87: keypoint 단위 실패 지표 --------------------------
    ep = dec.groupby("keypoint_id").agg(
        n=("err_d2", "size"),
        d2_missing_pct=("d2_missing", lambda s: round(100 * s.mean(), 2)),
        peak_med=("peak", lambda s: round(float(s.median()), 4)),
        err_d2_med=("err_d2", lambda s: round(float(np.nanmedian(s)), 2)),
        err_d0_med=("err_d0", lambda s: round(float(np.nanmedian(s)), 2)),
        entropy_med=("entropy", lambda s: round(float(np.nanmedian(s)), 3)),
    )
    ep["C2_train_pct"] = c2_by_kp
    print("\n=== ep57 on strict N87 (index-wise err) + 학습셋 C2 빈도 ===")
    print(ep.to_string())
    ep.to_csv(os.path.join(OUT, "failure_association_by_kp.csv"))

    # ---- 상관 (9 point 뿐 = 매우 약한 증거) ----------------------------
    print("\n=== C2 빈도 vs ep57 실패 지표 상관 (n=9 keypoints, 매우 약함) ===")
    for col in ["d2_missing_pct", "err_d2_med", "err_d0_med", "peak_med"]:
        r, p = spearmanr(ep.C2_train_pct, ep[col])
        print(f"  C2_train_pct vs {col:16s} rho={r:+.3f} p={p:.3f}")

    # ---- near vs far ---------------------------------------------------
    print("\n=== near(0-3) vs far(4-7) ===")
    tgt_g = tgt.assign(grp=np.where(tgt.keypoint_id.isin(NEAR), "near",
                                    np.where(tgt.keypoint_id.isin(FAR),
                                             "far", "ctr")))
    dec_g = dec.assign(grp=np.where(dec.keypoint_id.isin(NEAR), "near",
                                    np.where(dec.keypoint_id.isin(FAR),
                                             "far", "ctr")))
    a = tgt_g.groupby("grp").is_C2.mean().mul(100).round(3).rename("C2_train_pct")
    b = dec_g.groupby("grp").agg(
        ep57_err_d2_med=("err_d2", lambda s: round(float(np.nanmedian(s)), 2)),
        ep57_missing_pct=("d2_missing", lambda s: round(100 * s.mean(), 2)),
        ep57_peak_med=("peak", lambda s: round(float(s.median()), 4)))
    print(pd.concat([a, b], axis=1).to_string())

    # ---- truncated vs non-truncated (real) -----------------------------
    print("\n=== real N87: truncated(17) vs non-truncated(70) ===")
    t = dec.groupby("frame_is_truncated").agg(
        n_kp=("err_d2", "size"),
        n_frames=("frame_id", "nunique"),
        missing_pct=("d2_missing", lambda s: round(100 * s.mean(), 2)),
        peak_med=("peak", lambda s: round(float(s.median()), 4)),
        err_d2_med=("err_d2", lambda s: round(float(np.nanmedian(s)), 2)),
        err_d0_med=("err_d0", lambda s: round(float(np.nanmedian(s)), 2)))
    print(t.to_string())

    # ---- 화면 밖 keypoint 여부 ------------------------------------------
    print("\n=== real N87: keypoint 가 원본 화면 밖인가 ===")
    o = dec.groupby("kp_outside_image").agg(
        n_kp=("err_d2", "size"),
        missing_pct=("d2_missing", lambda s: round(100 * s.mean(), 2)),
        peak_med=("peak", lambda s: round(float(s.median()), 4)),
        err_d2_med=("err_d2", lambda s: round(float(np.nanmedian(s)), 2)))
    print(o.to_string())

    # ---- peak 위치가 경계 근처인지 --------------------------------------
    dec2 = dec.copy()
    dec2["border_grp"] = np.where(dec2.dist_to_border_peak <= 4,
                                  "peak<=4px(deadzone폭)", "interior")
    print("\n=== ep57 peak 가 belief 경계 4px(=w) 이내인가 ===")
    print(dec2.groupby("border_grp").agg(
        n_kp=("err_d2", "size"),
        missing_pct=("d2_missing", lambda s: round(100 * s.mean(), 2)),
        peak_med=("peak", lambda s: round(float(s.median()), 4)),
        err_d2_med=("err_d2", lambda s: round(float(np.nanmedian(s)), 2))
    ).to_string())

    # ---- domain -------------------------------------------------------
    print("\n=== domain별 ===")
    print(dec.groupby("domain").agg(
        n_frames=("frame_id", "nunique"),
        missing_pct=("d2_missing", lambda s: round(100 * s.mean(), 2)),
        peak_med=("peak", lambda s: round(float(s.median()), 4)),
        err_d2_med=("err_d2", lambda s: round(float(np.nanmedian(s)), 2))
    ).to_string())

    print("\n[caveat] err 는 index-wise(order-free 아님) — group 간 상대비교 전용.")
    print("[caveat] truncated frame 17개 = 소표본. 인과 주장 금지 ([추정]).")


if __name__ == "__main__":
    main()
