"""PHASE 6 — 왜 oracle headroom 이 예측 가능한 gate 로 회수되지 않는가.

oracle 이 15% 를 약속했는데 CV gate 는 -6.4% 를 냈다.  그 간극이 어디서 오는지
본다.  결론을 바꾸지 않는다 — 판정은 이미 사전등록 gate 로 났다.
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import glob as _glob
for _pat in ("/usr/share/fonts/**/NanumGothic.ttf",
             "/usr/share/fonts/**/NotoSansCJK*.ttc"):
    _hit = _glob.glob(_pat, recursive=True)
    if _hit:
        fm.fontManager.addfont(_hit[0])
        plt.rcParams["font.family"] = fm.FontProperties(fname=_hit[0]).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/home/minjae/Documents/github/pallet-pose"
OUT = os.path.join(ROOT, "challenge/yolo_pose_one_model/analysis_pre_v2")

rows = []
with open(os.path.join(OUT, "CONDITIONAL_HOUGH_PER_FRAME.csv")) as fh:
    for r in csv.DictReader(fh):
        d = {k: r[k] for k in ("fid", "set", "population")}
        for k, v in r.items():
            if k not in d:
                try:
                    d[k] = float(v) if v not in ("", "None") else np.nan
                except ValueError:
                    d[k] = np.nan
        rows.append(d)

oracle = json.load(open(os.path.join(OUT, "CONDITIONAL_HOUGH_ORACLE.json")))
cv = json.load(open(os.path.join(OUT, "CONDITIONAL_HOUGH_CV.json")))

win = np.array([1 if (np.isfinite(r["YH_R"]) and r["YH_R"] < r["Y0_R"]) else 0
                for r in rows])
rep = np.array([r["reproj"] for r in rows], float)
y0R = np.array([r["Y0_R"] for r in rows], float)
yhR = np.array([r["YH_R"] for r in rows], float)

# 1. gate feature 의 판별력 — reproj 로 Hough 승리를 예측할 수 있는가 (AUC)
def auc(score, label):
    o = np.argsort(score)
    lab = label[o]
    pos, neg = lab.sum(), len(lab) - lab.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = np.arange(1, len(lab) + 1)
    return float((ranks[lab == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))

good = np.isfinite(rep) & np.isfinite(y0R) & np.isfinite(yhR)
auc_rep = auc(rep[good], win[good])

# 2. oracle 이득의 편중 — 상위 몇 프레임이 이득을 독점하는가
gain = np.where(win == 1, y0R - yhR, 0.0)[good]
order = np.sort(gain)[::-1]
tot = order.sum()
share_top5 = float(order[:5].sum() / max(tot, 1e-9))

# 3. 활성화될 때의 실제 손익 (CV gate 가 켠 프레임에서)
tau_by = {f["held_out"]: (f["tau"] if f["tau"] is not None else np.inf)
          for f in cv["folds"]}
act = np.array([1 if (np.isfinite(r["reproj"]) and r["reproj"] > tau_by[r["set"]])
                else 0 for r in rows])
sel = (act == 1) & good
helped = int((yhR[sel] < y0R[sel]).sum())
hurt = int((yhR[sel] > y0R[sel]).sum())
med_help = float(np.median((y0R - yhR)[sel][(yhR < y0R)[sel]])) if helped else np.nan
med_hurt = float(np.median((yhR - y0R)[sel][(yhR > y0R)[sel]])) if hurt else np.nan

phase6 = {
 "question": "oracle 15% 가 예측 gate 에서 -6.4% 가 된 이유",
 "gate_feature_discrimination": {
   "auc_reproj_predicts_hough_win": round(auc_rep, 4),
   "reading": "0.5 = 무작위. reproj 는 Hough 가 이길 프레임을 거의 못 가른다."},
 "oracle_gain_concentration": {
   "n_hough_wins": int(win[good].sum()),
   "top5_frames_share_of_total_gain": round(share_top5, 4),
   "reading": "이득이 소수 프레임에 몰려 있으면, 그 프레임을 정확히 집지 못하는 "
              "gate 는 이득을 못 얻고 손해만 얻는다."},
 "activated_frame_outcome": {
   "n_activated": int(sel.sum()),
   "helped": helped, "hurt": hurt,
   "median_gain_when_helped_deg": None if not helped else round(med_help, 3),
   "median_loss_when_hurt_deg": None if not hurt else round(med_hurt, 3),
   "reading": "켠 프레임 중 손해 본 쪽이 많거나, 손해 크기가 이득보다 크면 "
              "median 이 악화된다."},
 "asymmetry": {
   "note": "5cm5 는 challenge +3.0pp / open +2.0pp 로 올랐으나 R median 은 "
           "악화됐다. Hough 는 '이미 거의 맞은' 프레임을 5cm5 문턱 너머로 "
           "밀어주는 대신 중간 구간을 흐트러뜨린다.",
   "challenge_5cm5_gain_pp": cv["verdict_inputs"]["challenge_5cm5_gain"],
   "challenge_R_gain": cv["verdict_inputs"]["challenge_R_gain"]},
 "conclusion": "oracle headroom 은 실재하나 GT 없이는 접근 불가. "
               "gate feature(pnp_reproj)의 판별력이 무작위에 가깝다. "
               "HOUGH_TRACK_CLOSED — 사전등록 판정 유지.",
}
json.dump(phase6, open(os.path.join(OUT, "CONDITIONAL_HOUGH_PHASE6.json"), "w"),
          indent=1, ensure_ascii=False)

# --- plots ---
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].scatter(rep[good][win[good] == 0], y0R[good][win[good] == 0], s=18,
              c="#888", label="point-only 우세")
ax[0].scatter(rep[good][win[good] == 1], y0R[good][win[good] == 1], s=22,
              c="#c0392b", label="Hough 우세")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("pnp_reproj (px)"); ax[0].set_ylabel("Y0 rotation error (deg)")
ax[0].set_title(f"gate feature vs Hough win  (AUC {auc_rep:.3f})")
ax[0].legend(fontsize=8)

ax[1].plot(np.cumsum(order) / max(tot, 1e-9), marker="o", ms=3, c="#2c3e50")
ax[1].axhline(0.8, ls="--", c="#c0392b", lw=1)
ax[1].set_xlabel("frames (gain 내림차순)")
ax[1].set_ylabel("누적 oracle gain 비율")
ax[1].set_title(f"oracle 이득 편중 (top5 = {share_top5:.0%})")

arms = ["Y0", "YH", "YG"]
for i, pop in enumerate(("REAL_DEV_OPEN_56", "REAL_CHALLENGE_DEV_105")):
    vals = [cv["populations"][pop][a]["R_median"] for a in arms]
    ax[2].bar(np.arange(3) + i * 0.38, vals, width=0.36,
              label=pop.replace("REAL_", ""))
ax[2].set_xticks(np.arange(3) + 0.19); ax[2].set_xticklabels(arms)
ax[2].set_ylabel("rotation error median (deg)")
ax[2].set_title("point-only vs 항상-Hough vs CV gate")
ax[2].legend(fontsize=8)
for a in ax:
    a.grid(alpha=.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "CONDITIONAL_HOUGH_PLOTS.png"), dpi=130)
print(json.dumps(phase6, ensure_ascii=False, indent=1))
