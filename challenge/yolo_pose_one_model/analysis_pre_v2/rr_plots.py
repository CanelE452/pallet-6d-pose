import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm, glob
for p in ("/usr/share/fonts/**/NanumGothic.ttf",):
    h = glob.glob(p, recursive=True)
    if h:
        fm.fontManager.addfont(h[0])
        plt.rcParams["font.family"] = fm.FontProperties(fname=h[0]).get_name()
plt.rcParams["axes.unicode_minus"] = False
ROOT = "/home/minjae/Documents/github/pallet-pose"
A2 = os.path.join(ROOT, "challenge/yolo_pose_one_model/analysis_pre_v2")
PL = os.path.join(A2, "plots")
o = json.load(open(f"{A2}/RERANK_ORACLE.json"))
f4 = json.load(open(f"{A2}/RERANK_FEATURES.json"))
cp = json.load(open(f"{A2}/RERANK_DEPLOY_COUPLING.json"))

K = [s["K"] for s in o["phase1_topk_oracle"]]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].plot(K, [s["correct_recall"] for s in o["phase1_topk_oracle"]], "o-",
           c="#2c3e50", label="correct box recall")
ax[0].plot(K, [s["success_5cm5_any"] for s in o["phase1_topk_oracle"]], "s-",
           c="#c0392b", label="5cm5 (top-K 중 아무거나)")
ax[0].set_xlabel("K"); ax[0].set_ylabel("비율"); ax[0].set_ylim(0, 1)
ax[0].set_title("top-K oracle 상한 — box 는 오르는데 pose 는 안 오른다")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
c = o["phase2_classification"]
keys = ["TOP1_ALREADY_GOOD", "A_GOOD_CANDIDATE_MISRANKED",
        "B_CORRECT_BOX_BAD_KP", "C_NO_CORRECT_CANDIDATE"]
lbl = ["top1 이미 정상", "A 오정렬\n(rerank 상금)", "B 박스 정상\nKP 불량",
       "C 정답 후보 없음"]
col = ["#95a5a6", "#27ae60", "#c0392b", "#7f8c8d"]
ax[1].bar(range(4), [c[k]["n"] for k in keys], color=col)
for i, k in enumerate(keys):
    ax[1].text(i, c[k]["n"] + 1, f"{c[k]['n']}\n{c[k]['frac']:.1%}",
               ha="center", fontsize=8)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(lbl, fontsize=8)
ax[1].set_ylabel("프레임 수 (n=161)")
ax[1].set_title("PHASE 2 — rerank 로 회수 가능한 건 A 뿐")
ax[1].grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(f"{PL}/TOPK_ORACLE.png", dpi=130)

fig, ax = plt.subplots(figsize=(8, 4.4))
ph3 = f4["phase3_single_feature"]
names = sorted(ph3, key=lambda k: -(ph3[k]["usable"]["AUC"] or 0))
x = np.arange(len(names))
ax.bar(x - 0.2, [ph3[k]["box"]["AUC"] for k in names], 0.38, label="AUC (correct box)")
ax.bar(x + 0.2, [ph3[k]["usable"]["AUC"] for k in names], 0.38, label="AUC (usable pose)")
ax.axhline(0.5, ls="--", c="#c0392b", lw=1, label="무작위 0.5")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("frame 별 ranking AUC")
ax.set_title("PHASE 3 — box 크기가 confidence 를 크게 앞선다")
ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(f"{PL}/FEATURE_RANKING.png", dpi=130)

fig, ax = plt.subplots(figsize=(7.5, 4.4))
cs = [r["conf"] for r in cp["by_conf"]]
ax.plot(cs, [r["recall_gain_frames"] for r in cp["by_conf"]], "o-",
        c="#2c3e50", label="correct box 회수 (장)")
ax.plot(cs, [r["usable_gain_frames"] for r in cp["by_conf"]], "s-",
        c="#c0392b", label="usable pose 회수 (장)")
ax2 = ax.twinx()
ax2.plot(cs, [r["frac_multi_cand"] for r in cp["by_conf"]], ":", c="#7f8c8d",
         label="후보 2개 이상 비율")
ax.set_xscale("log"); ax.set_xlabel("conf threshold")
ax.set_ylabel("rerank 로 늘어난 프레임 수"); ax2.set_ylabel("후보 2개 이상 비율")
ax.set_title("rerank 는 낮은 conf 에서만 작동하고, 그때도 pose 는 거의 안 는다")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=8, loc="upper right"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{PL}/RERANK_COUPLING.png", dpi=130)
print("plots ok")
