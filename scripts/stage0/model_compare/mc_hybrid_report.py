"""hybrid 판정 그림 + 보고서."""
from __future__ import annotations
import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
OUT = os.path.join(ROOT, "data/pallet/results/model_compare")
PLOTS = os.path.join(OUT, "plots")
import glob
for pat in ("/usr/share/fonts/**/NanumGothic.ttf",
            "/usr/share/fonts/**/NotoSansCJK*.ttc"):
    hit = glob.glob(pat, recursive=True)
    if hit:
        fm.fontManager.addfont(hit[0])
        plt.rcParams["font.family"] = fm.FontProperties(fname=hit[0]).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def rows():
    out = []
    with open(os.path.join(OUT, "HYBRID_POINT_LINE_PER_FRAME.csv")) as fh:
        for r in csv.DictReader(fh):
            d = {"fid": r["fid"], "set": r["set"],
                 "sealed": r["sealed"] == "True"}
            for k, v in r.items():
                if k in d:
                    continue
                try:
                    d[k] = float(v) if v not in ("", "None") else np.nan
                except ValueError:
                    d[k] = np.nan
            out.append(d)
    return out


def scatter(data, field, title, path, unit):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (lbl, sub) in zip(axes, (("REAL_DEV_OPEN_56",
                                      [r for r in data if not r["sealed"]]),
                                     ("REAL_CHALLENGE_DEV_105",
                                      [r for r in data if r["sealed"]]))):
        a = np.array([r[f"B1_{field}"] for r in sub], float)
        b = np.array([r[f"P1_{field}"] for r in sub], float)
        g = np.isfinite(a) & np.isfinite(b)
        a, b = a[g], b[g]
        ax.scatter(a, b, s=24, alpha=.75, color="#22405f")
        lim = [0, max(a.max(), b.max()) * 1.05] if len(a) else [0, 1]
        ax.plot(lim, lim, "--", color="#b3452c", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        win = int((b < a).sum())
        ax.set_title(f"{lbl}\nn={len(a)}   F3 개선 {win}/{len(a)}", fontsize=9)
        ax.set_xlabel(f"B1  point-only ({unit})")
        ax.set_ylabel(f"P1  +Hough/F3 ({unit})")
        ax.grid(alpha=.25)
    fig.suptitle(f"{title}   [대각선 아래 = hybrid 개선]")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    data = rows()
    rep = json.load(open(os.path.join(OUT, "HYBRID_POINT_LINE_MATRIX.json")))
    scatter(data, "R", "회전 오차 (deg)",
            os.path.join(PLOTS, "B1_vs_P1_R.png"), "deg")
    scatter(data, "t", "이동 오차 (m)",
            os.path.join(PLOTS, "B1_vs_P1_t.png"), "m")
    scatter(data, "add", "ADD (m)",
            os.path.join(PLOTS, "B1_vs_P1_ADD.png"), "m")

    # point source 별 base vs hybrid
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    tags = ["B0/P0", "B1/P1", "B2/P2", "B3/P3"]
    names = ["FINAL40K", "y_BROAD40K", "y_synth", "y_ft"]
    for ax, pop in zip(axes, ("REAL_DEV_OPEN_56", "REAL_CHALLENGE_DEV_105")):
        arms = rep["populations"][pop]["arms"]
        base = [arms[t.split("/")[0]]["R_deg"]["median"] for t in tags]
        hyb = [arms[t.split("/")[1]]["R_deg"]["median"] for t in tags]
        x = np.arange(len(tags))
        ax.bar(x - .19, base, .38, label="point-only", color="#22405f")
        ax.bar(x + .19, hyb, .38, label="+Hough/F3", color="#b3452c")
        for i, (u, v) in enumerate(zip(base, hyb)):
            ax.text(i - .19, u, f"{u:.2f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + .19, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, names, fontsize=8)
        ax.set_title(f"{pop}   [↓ 낮을수록 좋음]", fontsize=9)
        ax.set_ylabel("R median (deg)"); ax.grid(alpha=.25, axis="y")
        ax.legend(fontsize=8); ax.margins(y=.18)
    fig.suptitle("point source 별 — Hough/F3 를 얹으면 좋아지나")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "point_source_comparison.png"), dpi=130)
    plt.close(fig)

    lines = ["# HYBRID POINT x LINE — 판정", "",
             f"**VERDICT: {rep['VERDICT']}**", "",
             "질문: robust 한 point estimator 위에서도 Direct-Hough + F3 가 독립적",
             "가치를 가지는가. 학습 0, 기존 추론 결과만 사용.", "",
             "## 공정성", "", "```",
             "코너 순서   index-wise / order-free 비율 1.02~1.17 -> 순열 일치 [확인]",
             "3D 점       annotate_pnp, top={0,1,4,5} = mh_fusion.TOP 과 일치",
             "K / dims    프레임 라벨 동일",
             "solver      base = SQPnP->refineLM,  hybrid = mh_fusion F3 정본",
             "신뢰도      필터 없음 (visibility 와 belief peak 은 다른 양)",
             "support     예측 점에서 생성. GT 는 O1/O2 에만", "```", ""]
    for pop in ("REAL_DEV_OPEN_56", "REAL_CHALLENGE_DEV_105"):
        blk = rep["populations"][pop]
        lines += [f"## {pop} (n={blk['n']})", "", "```",
                  f"{'arm':6}{'avail':>7}{'R med ↓':>10}{'t med ↓':>10}"
                  f"{'ADD-S ↓':>10}{'IoU ↑':>8}{'5cm5 ↑':>9}"]
        for a in ("B0", "P0", "B1", "P1", "B2", "P2", "B3", "P3", "O1", "O2"):
            s = blk["arms"][a]
            lines.append(
                f"{a:6}{s['available']:>7.3f}{s['R_deg']['median']:>10.2f}"
                f"{s['t_m']['median']:>10.4f}{s['ADD_S']['median']:>10.4f}"
                f"{s['IoU3D']['median']:>8.3f}{s['success_5cm5deg']:>9.3f}")
        lines += ["```", ""]
        p = blk["paired"]["P1 vs B1"]
        lines += ["P1 vs B1 paired (bootstrap 95% CI)", "", "```"]
        for f in ("R", "t", "adds", "iou"):
            e = p[f]
            lines.append(f"{f:5} base {e['base_median']:>9} -> hybrid "
                         f"{e['hybrid_median']:>9}  delta {e['delta_median']:>9}"
                         f"  CI {e['CI95']}  win {e['win_fraction']}")
        lines += ["```", ""]
    lines += ["## 사전등록 판정 입력", "", "```"]
    for k, v in rep["verdict_inputs"].items():
        lines.append(f"{k:28} {v}")
    lines += ["```", ""]
    open(os.path.join(OUT, "HYBRID_POINT_LINE_REPORT.md"), "w").write(
        "\n".join(lines) + "\n")
    json.dump({"rule": rep["prelocked_rule"], "verdict": rep["VERDICT"],
               "inputs": rep["verdict_inputs"]},
              open(os.path.join(OUT, "HYBRID_PREREG_DECISION.json"), "w"), indent=1)
    print("-> HYBRID_POINT_LINE_REPORT.md / HYBRID_PREREG_DECISION.json / plots 4")


if __name__ == "__main__":
    main()
