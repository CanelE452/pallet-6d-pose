"""비교 보고서 + 그림.  숫자만큼이나 '무엇을 비교한 것인가' 를 적는 게 중요하다."""
from __future__ import annotations

import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def _korean_font():
    """한글이 두부(네모)로 나오지 않게 실제 설치된 폰트를 찾아 지정한다.

    matplotlib 의 기본 DejaVu Sans 에는 한글 글리프가 없어 U+FFFD 네모가 찍힌다.
    fontManager 캐시에 없을 수 있으므로 fc-list 경로로 직접 등록한다.
    """
    import glob
    for pattern in ("/usr/share/fonts/**/NanumGothic.ttf",
                    "/usr/share/fonts/**/NanumBarunGothic.ttf",
                    "/usr/share/fonts/**/NotoSansCJK*.ttc"):
        for path in glob.glob(pattern, recursive=True):
            try:
                fm.fontManager.addfont(path)
                return fm.FontProperties(fname=path).get_name()
            except Exception:
                continue
    for name in ("Noto Sans CJK JP", "NanumGothic"):
        if any(f.name == name for f in fm.fontManager.ttflist):
            return name
    return None


_FONT = _korean_font()
if _FONT:
    plt.rcParams["font.family"] = _FONT
plt.rcParams["axes.unicode_minus"] = False   # 마이너스 기호가 두부로 안 나오게

ROOT = "/home/minjae/Documents/github/pallet-pose"
OUT = os.path.join(ROOT, "data/pallet/results/model_compare")
ORDER = ["yolo26n_synth", "yolo26n_ft", "yolo26m_ft", "FINAL40K_seed1"]
LABEL = {"yolo26n_synth": "yolo26n\n(합성만)", "yolo26n_ft": "yolo26n\nFT",
         "yolo26m_ft": "yolo26m\nFT", "FINAL40K_seed1": "FINAL40K\n(논문)"}
COLOUR = ["#9aa7b1", "#22405f", "#3f7d4e", "#b3452c"]


def bar(ax, data, title, ylabel, higher_better, fmt="{:.3f}"):
    """화살표는 장식이 아니라 방향 표시다 — 어느 쪽이 좋은지 표에서 바로 읽히게."""
    arrow = "\u2191 높을수록 좋음" if higher_better else "\u2193 낮을수록 좋음"
    x = np.arange(len(ORDER))
    ax.bar(x, data, color=COLOUR, width=.68)
    finite = [v for v in data if np.isfinite(v)]
    best = (max(finite) if higher_better else min(finite)) if finite else None
    for i, v in enumerate(data):
        if not np.isfinite(v):
            continue
        star = "  \u2605" if best is not None and v == best else ""
        ax.text(i, v, fmt.format(v) + star, ha="center", va="bottom",
                fontsize=8, fontweight="bold" if star else "normal")
    ax.set_xticks(x, [LABEL[m] for m in ORDER], fontsize=8)
    ax.set_title(f"{title}   [{arrow}]", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.margins(y=0.16)
    ax.grid(alpha=.25, axis="y")


def main():
    report = json.load(open(os.path.join(OUT, "MODEL_COMPARE.json")))
    models = report["models"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, pop, lbl in ((0, "OPEN_56", "OPEN 56"), (1, "SEALED_105", "SEALED 105")):
        bar(axes[row][0], [models[m][pop]["det_rate"] for m in ORDER],
            f"{lbl} — 검출률", "det rate", True)
        bar(axes[row][1],
            [models[m][pop]["success_5cm5deg_unconditional"] for m in ORDER],
            f"{lbl} — 5cm5deg (unconditional)", "success", True)
        bar(axes[row][2],
            [models[m][pop]["R_deg"]["median"] if models[m][pop]["R_deg"] else np.nan
             for m in ORDER], f"{lbl} — 회전 오차 median", "deg", False, "{:.2f}")
    fig.suptitle("4-model comparison on manual GT 161 (open 56 / sealed 105 분리)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "model_compare.png"), dpi=130)
    plt.close(fig)

    def line(m, pop):
        s = models[m][pop]
        def g(k):
            v = s.get(k)
            return f"{v['median']:.2f}" if isinstance(v, dict) and v.get("median") is not None else "-"
        return (f"{m:17}{s['det_rate']:>7.3f}{s['pnp_rate']:>7.3f}"
                f"{g('corner_px'):>9}{g('R_deg'):>9}{g('t_m'):>9}"
                f"{g('ADD_S'):>9}{g('IoU3D'):>7}"
                f"{s['success_5cm5deg_unconditional']:>8.3f}")

    # 화살표 = 어느 방향이 좋은지.  표를 볼 때마다 다시 생각하지 않게 헤더에 박는다.
    hdr = (f"{'model':17}{'det ↑':>7}{'pnp ↑':>7}{'corner ↓':>9}{'R med ↓':>9}"
           f"{'t med ↓':>9}{'ADD-S ↓':>9}{'IoU ↑':>7}{'5cm5 ↑':>8}")
    lines = ["# 4-MODEL COMPARISON — manual GT 161", "",
             "네 모델을 **한 downstream** 으로 채점했다. 추론 환경(ultralytics 8.4.60 vs",
             "8.0.120)만 다르고 GT·order-free 대응·PnP·pose 지표는 완전히 같다.", "",
             "## ⚠ 봉인 소진", "",
             "```", "SEALED 105 (pallet07 27 / pallet09 36 / night08 17 / night09 25)",
             "사용자 승인 2026-08-20. 재봉인 불가 — 이후 어떤 최종 주장에도",
             "held-out 으로 쓸 수 없다.", "```", "",
             "open 56 과 sealed 105 는 **합산하지 않는다.**", "",
             "## OPEN 56", "", "```", hdr, "─" * len(hdr)]
    lines += [line(m, "OPEN_56") for m in ORDER]
    lines += ["```", "", "## SEALED 105", "", "```", hdr, "─" * len(hdr)]
    lines += [line(m, "SEALED_105") for m in ORDER]
    lines += ["```", "", "## 세트별 5cm5deg  [↑ 높을수록 좋음]", "", "```"]
    sets = list(models[ORDER[0]]["per_set"].keys())
    lines.append(f"{'model':17}" + "".join(
        f"{s.replace('eval_',''):>11}" for s in sets))
    lines.append("─" * (17 + 11 * len(sets)))
    for m in ORDER:
        lines.append(f"{m:17}" + "".join(
            f"{models[m]['per_set'][s]['success_5cm5deg_unconditional']:>11.3f}"
            for s in sets))
    lines += ["```", "",
              "## 이 표가 비교하는 것 — 아키텍처가 아니다", "",
              "```",
              "yolo26n_synth   합성만.  단 그 합성에 **사용자 팔레트(target) 36k** 포함",
              "yolo26n_ft      + real 157장 파인튜닝",
              "yolo26m_ft      + real 157장 파인튜닝 (medium 백본)",
              "FINAL40K        합성만.  real 0장 + **사용자 팔레트 v1/v2 의도적 제외**",
              "```", "",
              "누수는 없다 — FT 의 real 157장은 night01~07 · pallet02/03/04/05/08 이고",
              "평가셋의 night08/09 · pallet07/09 는 `runs_ft/PURPOSE.md` 가 명시적으로",
              "제외했다. 인접 프레임(non-eval 53장)까지 뺐다.", "",
              "그러나 **감독량이 다르다.** 이건 '논문 모델이 약하다' 가 아니라",
              "'대상 팔레트를 본 적 없는 모델 vs 본 모델' 의 비교다. 논문 트랙은",
              "처음 본 팔레트 일반화를 목표로 v1/v2 를 뺐고, 그 대가가 여기 그대로 나온다.", ""]
    open(os.path.join(OUT, "MODEL_COMPARE_REPORT.md"), "w").write("\n".join(lines) + "\n")
    print("-> MODEL_COMPARE_REPORT.md / model_compare.png")


if __name__ == "__main__":
    main()
