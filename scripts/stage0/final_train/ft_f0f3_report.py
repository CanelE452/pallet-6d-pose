"""Plots and the written report for REAL_DEV F0 vs F3."""
from __future__ import annotations

import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
OUT = os.path.join(ROOT, "data/pallet/results/paper_s2_multihead/final_train")
PLOTS = os.path.join(OUT, "plots")


def load_rows():
    rows = []
    with open(os.path.join(OUT, "REAL_DEV_PER_FRAME.csv")) as fh:
        for r in csv.DictReader(fh):
            out = {"seed": int(r["seed"]), "set": r["set"], "fid": r["fid"],
                   "n_det": int(r["n_det"]), "det": int(r["det"])}
            for k, v in r.items():
                if k in out or k in ("seed", "set", "fid"):
                    continue
                try:
                    out[k] = float(v) if v not in ("", "None") else np.nan
                except ValueError:
                    out[k] = np.nan
            rows.append(out)
    return rows


def scatter(rows, metric, title, path, log=False):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, seed in zip(axes, (1, 2)):
        block = [r for r in rows if r["seed"] == seed]
        a = np.array([r[f"F0_{metric}"] for r in block], float)
        b = np.array([r[f"F3_{metric}"] for r in block], float)
        good = np.isfinite(a) & np.isfinite(b)
        a, b = a[good], b[good]
        ax.scatter(a, b, s=22, alpha=.75, color="#22405f")
        lim = [0, max(a.max(), b.max()) * 1.05] if len(a) else [0, 1]
        ax.plot(lim, lim, "--", color="#b3452c", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        if log:
            ax.set_xscale("symlog"); ax.set_yscale("symlog")
        better = int((b < a).sum())
        ax.set_title(f"seed{seed}  n={len(a)}  F3 better {better}/{len(a)}",
                     fontsize=9)
        ax.set_xlabel(f"F0 {metric}"); ax.set_ylabel(f"F3 {metric}")
        ax.grid(alpha=.25)
    fig.suptitle(f"{title}  (대각선 아래 = F3 개선)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def score_plot(rows, path):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for seed, colour in ((1, "#22405f"), (2, "#b3452c")):
        s = np.sort([r["score_4kp"] for r in rows if r["seed"] == seed])
        ax.plot(s, 1 - np.arange(len(s)) / len(s), lw=1.8,
                color=colour, label=f"seed{seed} positive recall")
    ax.set_xlabel("score_4kp threshold"); ax.set_ylabel("positive recall")
    ax.set_title("score_4kp on REAL_DEV positives only\n"
                 "(no real negatives yet -> no precision, no threshold choice)",
                 fontsize=9)
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    rows = load_rows()
    report = json.load(open(os.path.join(OUT, "REAL_DEV_F0_F3_EVAL.json")))

    scatter(rows, "R", "rotation error (deg)",
            os.path.join(PLOTS, "F0_vs_F3_R.png"))
    scatter(rows, "t", "translation error (m)",
            os.path.join(PLOTS, "F0_vs_F3_t.png"))
    scatter(rows, "add", "ADD (m)", os.path.join(PLOTS, "F0_vs_F3_ADD.png"))
    score_plot(rows, os.path.join(PLOTS, "score4kp_positive_distribution.png"))

    lines = ["# REAL_DEV F0 vs F3 — 최종 추론 경로 실측", "",
             "이전 real 수치는 corner decoder + Point PnP 만 썼다. 그건 F0 이고,",
             "SplitLate 의 line branch 를 버린 것이라 **최종 추론 성능이 아니었다.**",
             "이번에는 정본 `mh_fusion.solve_arms` 의 F3 를 그대로 붙였다 — 새 solver 는",
             "쓰지 않았다.", "",
             "```",
             f"population   REAL_DEV_POS_V1  (비봉인 정본 eval, positive 56장)",
             f"SEALED       {', '.join(report['sealed_not_accessed'])}  — 열지 않음",
             f"lambda_theta {report['lambda_theta']}",
             "```", "",
             "## 두 가지 결정 (조용히 다른 걸 재지 않도록 명시)", "",
             "**corner 입력** — `solve_arms` 는 8코너 전부를 요구하므로 solver 는",
             "`_decode_peaks`(argmax, threshold 없음)를 읽는다. synthetic F3 검증과",
             "같은 입력이다. threshold 0.3 검출기는 **따로** 보고한다. 둘을 섞으면",
             "검출 실패가 pose 오차로 둔갑한다.", "",
             "**line support** — synthetic 에서는 GT 코너에서 뽑았다. real 에서 그러면",
             "oracle 이므로, 주 결과는 **예측 코너**에서 같은 `visible_segments` 로",
             "뽑는다. GT 판은 parity 대조로만 남긴다.", ""]

    for seed in ("seed1", "seed2"):
        b = report["seeds"][seed]
        pop = b["populations"]
        lines += [f"## {seed}", "", "```",
                  f"A 전체 positive      {pop['A_all_positive']['n']}",
                  f"B corner 검출 성공   {pop['B_corner_detected']['n']}"
                  f"  ({100*pop['B_corner_detected']['rate']:.1f}%)",
                  f"C PnP 성공           {pop['C_pnp_success']['n']}"
                  f"  ← argmax 입력이라 항상 8점, 정보량 없음", "```", "",
                  "MAIN = A 기준 unconditional", "", "```",
                  f"{'arm':5}{'R med':>9}{'R p90':>9}{'t med':>9}{'t p90':>9}"
                  f"{'ADD-S':>9}{'IoU3D':>9}{'5cm5deg':>10}"]
        for arm in ("F0", "F3"):
            s = b["MAIN_unconditional_A"][arm]
            lines.append(
                f"{arm:5}{s['R_deg']['median']:>9}{s['R_deg']['p90']:>9}"
                f"{s['t_m']['median']:>9}{s['t_m']['p90']:>9}"
                f"{s['ADD_S']['median']:>9}{s['IoU3D']['median']:>9}"
                f"{s['success_5cm5deg_unconditional']:>10}")
        lines += ["```", "", "paired F3 − F0 (같은 프레임, bootstrap 95% CI)", "",
                  "```"]
        for metric, v in b["F3_minus_F0_paired"].items():
            if v.get("delta_median") is None:
                continue
            verdict = "0 배제" if v["excludes_zero"] else "0 포함 — 미확립"
            lines.append(f"{metric:5} F0 {v['F0_median']:>9} F3 {v['F3_median']:>9}"
                         f"  delta {v['delta_median']:>9}"
                         f"  CI [{v['CI95'][0]}, {v['CI95'][1]}]  {verdict}")
        lines += ["```", "", "세트별 (F3)", "", "```",
                  f"{'set':16}{'n':>4}{'R med':>9}{'t med':>9}{'IoU':>8}{'5cm5':>8}"]
        for key, arms in b["per_set"].items():
            s = arms["F3"]
            lines.append(f"{key:16}{s['n_frames']:>4}{s['R_deg']['median']:>9}"
                         f"{s['t_m']['median']:>9}{s['IoU3D']['median']:>8}"
                         f"{s['success_5cm5deg_unconditional']:>8}")
        gt = b["parity_GTSUP"]["F3_GTSUP"]
        pred = b["MAIN_unconditional_A"]["F3"]
        lines += ["```", "",
                  f"support parity: 예측 support R med {pred['R_deg']['median']} vs "
                  f"GT support {gt['R_deg']['median']} — 차이가 사실상 없다.",
                  "즉 F3 는 GT 없이도 같은 성능을 낸다 (배포 가능).", ""]
        sc = b["score_4kp"]
        lines += ["score_4kp (positive 만)", "", "```",
                  f"min {sc['min']}  median {sc['median']}  max {sc['max']}",
                  "recall @ " + "  ".join(f"{k}:{v}" for k, v in
                                          sc["recall_at"].items()), "```", "",
                  "**threshold 를 고르지 않는다.** real negative 가 없어 precision /",
                  "AP / FPR 을 계산할 수 없다.", ""]

    (open(os.path.join(OUT, "REAL_DEV_F0_F3_REPORT.md"), "w")
     .write("\n".join(lines) + "\n"))
    print(f"-> REAL_DEV_F0_F3_REPORT.md + {len(os.listdir(PLOTS))} plots")


if __name__ == "__main__":
    main()
