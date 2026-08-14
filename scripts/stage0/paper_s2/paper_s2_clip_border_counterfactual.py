"""paper_s2_clip_border_counterfactual.py — Step 4 (T0 vs T1).

학습 없이, 동일 frame / 동일 keypoint 에서 두 target 을 모두 만들어 비교한다.
  T0 : clip_at_border=False  (ep57 이 실제로 쓴 legacy 경로)
  T1 : clip_at_border=True   (중심이 map 안이면 tail 만 잘라 그린다)

C2 keypoint 에서 T1 이 실제로 peak 를 복구하는지 확인하고, 예시 이미지를 남긴다.
clip=True 는 center 가 belief map 안인 경우에만 적용된다 — 화면 밖 keypoint 를
경계에 강제로 찍지 않는다 (CreateBeliefMap 의 centre_inside 조건).

읽기 전용: 기존 데이터/체크포인트 수정 없음.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
from utils_belief import CreateBeliefMap  # noqa: E402

OUT = os.path.join(ROOT, "data", "pallet", "results",
                   "paper_s2_target_semantics_audit")
FIG = os.path.join(OUT, "figures")
PQ = os.path.join(OUT, "target_semantics_keypoints.parquet")
BELIEF, SIGMA = 50, 2.0


def make_targets(kp9_belief):
    """(9,2) belief 좌표 -> (T0, T1) 각각 (9,50,50)."""
    pts = [[list(map(float, p)) for p in kp9_belief]]
    t0 = np.array(CreateBeliefMap(BELIEF, pts, 9, sigma=SIGMA,
                                  clip_at_border=False))
    t1 = np.array(CreateBeliefMap(BELIEF, pts, 9, sigma=SIGMA,
                                  clip_at_border=True))
    return t0, t1


def main():
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_parquet(PQ)

    ci = df.center_inside_belief
    nz = df.belief_target_nonzero
    m1 = df.belief_channel_mask == 1
    df["is_C2"] = ci & ~nz & m1

    # frame 단위로 9 keypoint 를 모아 target 을 다시 만든다
    c2_frames = df[df.is_C2][["dataset", "frame_id"]].drop_duplicates()
    print(f"[C2] keypoints={int(df.is_C2.sum())}  "
          f"frames={len(c2_frames)} / {df.frame_id.nunique()}")

    rng = np.random.default_rng(42)
    take = c2_frames.sample(min(400, len(c2_frames)), random_state=42)

    recs = []
    for _, row in take.iterrows():
        g = df[(df.dataset == row.dataset)
               & (df.frame_id == row.frame_id)].sort_values("keypoint_id")
        if len(g) != 9:
            continue
        kp = g[["x_belief", "y_belief"]].to_numpy()
        t0, t1 = make_targets(kp)
        for k in range(9):
            recs.append(dict(
                dataset=row.dataset, frame_id=row.frame_id, keypoint_id=k,
                x_belief=float(kp[k][0]), y_belief=float(kp[k][1]),
                dist_to_border=float(g.iloc[k].dist_to_border),
                center_inside=bool(g.iloc[k].center_inside_belief),
                full_support=bool(g.iloc[k].full_gaussian_support_inside),
                is_C2=bool(g.iloc[k].is_C2),
                t0_max=float(t0[k].max()), t0_sum=float(t0[k].sum()),
                t1_max=float(t1[k].max()), t1_sum=float(t1[k].sum()),
                t0_nonzero=bool(t0[k].max() > 0),
                t1_nonzero=bool(t1[k].max() > 0),
            ))
    cf = pd.DataFrame(recs)
    cf.to_csv(os.path.join(OUT, "clip_counterfactual.csv"), index=False)

    print(f"\n[frames sampled] {take.shape[0]}  rows={len(cf)}")
    c2 = cf[cf.is_C2]
    print("\n=== C2 keypoint 에서 T1(clip=True) 이 peak 를 복구하는가 ===")
    print(f"  C2 keypoints            : {len(c2)}")
    print(f"  T0 nonzero              : {int(c2.t0_nonzero.sum())} "
          f"({100*c2.t0_nonzero.mean():.1f}%)")
    print(f"  T1 nonzero (복구)        : {int(c2.t1_nonzero.sum())} "
          f"({100*c2.t1_nonzero.mean():.1f}%)")
    if len(c2):
        print(f"  T1 peak  median         : {c2.t1_max.median():.4f}")
        print(f"  T1 mass(sum) median     : {c2.t1_sum.median():.3f}")

    ctrl = cf[(~cf.is_C2) & cf.full_support]
    print("\n=== C1 control (full support) — T0/T1 동일해야 정상 ===")
    print(f"  n={len(ctrl)}  T0 nonzero={100*ctrl.t0_nonzero.mean():.1f}%  "
          f"T1 nonzero={100*ctrl.t1_nonzero.mean():.1f}%")
    same = np.allclose(ctrl.t0_max, ctrl.t1_max) and np.allclose(
        ctrl.t0_sum, ctrl.t1_sum)
    print(f"  T0==T1 (peak & mass)    : {same}")

    outside = cf[~cf.center_inside]
    print("\n=== center 가 map 밖인 keypoint — clip=True 여도 안 그려야 정상 ===")
    print(f"  n={len(outside)}  T1 nonzero={int(outside.t1_nonzero.sum())} "
          f"(0 이어야 정상)")

    # 복구된 mass 를 border distance 별로
    if len(c2):
        c2 = c2.copy()
        c2["bin"] = pd.cut(c2.dist_to_border, [-0.001, 1, 2, 3, 4],
                           labels=["0-1", "1-2", "2-3", "3-4"])
        print("\n=== border-distance 별 T1 복구 mass ===")
        print(c2.groupby("bin", observed=True).agg(
            n=("t1_max", "size"),
            t1_peak_med=("t1_max", lambda s: round(float(s.median()), 4)),
            t1_mass_med=("t1_sum", lambda s: round(float(s.median()), 3)),
            full_mass_ref=("t1_sum", lambda s: "")).to_string())
        # 참고: 완전 interior Gaussian 의 mass
        ref = np.array(CreateBeliefMap(BELIEF, [[[25.0, 25.0]] * 9], 9,
                                       sigma=SIGMA, clip_at_border=False))
        print(f"  (참고) 완전 interior Gaussian mass = {ref[0].sum():.3f}, "
              f"peak = {ref[0].max():.4f}")

    # ---- 예시 이미지 -------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[fig] matplotlib 없음 — 이미지 생략 ({e})")
        return

    def draw(sel, fname, title):
        if not len(sel):
            return
        n = min(5, len(sel))
        fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
        axes = np.atleast_2d(axes)
        for r in range(n):
            row = sel.iloc[r]
            g = df[(df.dataset == row.dataset)
                   & (df.frame_id == row.frame_id)].sort_values("keypoint_id")
            kp = g[["x_belief", "y_belief"]].to_numpy()
            t0, t1 = make_targets(kp)
            k = int(row.keypoint_id)
            for c, (t, lbl) in enumerate([(t0, "T0 clip=False"),
                                          (t1, "T1 clip=True")]):
                ax = axes[r, c]
                ax.imshow(t[k], vmin=0, vmax=1, cmap="viridis")
                ax.plot(row.x_belief, row.y_belief, "rx", ms=8, mew=2)
                w = int(2 * SIGMA)
                ax.add_patch(plt.Rectangle(
                    (row.x_belief - w, row.y_belief - w), 2 * w, 2 * w,
                    fill=False, ec="r", ls="--", lw=1))
                ax.set_title(f"{lbl}\nkp{k} d={row.dist_to_border:.1f} "
                             f"max={t[k].max():.3f}", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
            axes[r, 0].set_ylabel(f"{row.dataset[:14]}\n{str(row.frame_id)[:14]}",
                                  fontsize=6)
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        p = os.path.join(FIG, fname)
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print(f"[fig] {p}")

    draw(c2[c2.keypoint_id == 5].head(5), "clip_false_vs_true_kp5.png",
         "C2 examples — kp5 (T0 all-zero vs T1 clipped)")
    draw(c2[c2.keypoint_id == 6].head(5), "clip_false_vs_true_kp6.png",
         "C2 examples — kp6")
    draw(c2[~c2.keypoint_id.isin([5, 6])].head(5),
         "clip_false_vs_true_other.png", "C2 examples — other keypoints")
    draw(ctrl.head(5), "clip_false_vs_true_control_C1.png",
         "C1 control (full support) — T0 == T1")


if __name__ == "__main__":
    main()
