"""PHASE 4 부록 — rerank 가 배포에서 실제로 무엇을 사는가.

세 가지를 확인한다.
 1. conf threshold 별 rerank 이득 — rerank 는 후보가 2개 이상일 때만 작동한다.
    conf=0.4 에서 후보가 사실상 1개면 rerank 는 배포에서 아무 것도 안 한다.
 2. 회수되는 것이 box 인가 pose 인가 — 프레임 수로.
 3. negative 부작용 (secondary — negative DEV 가 편향 표본이므로 참고만).
"""
from __future__ import annotations
import json, os
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
A2 = os.path.join(ROOT, "challenge/yolo_pose_one_model/analysis_pre_v2")
IOU, UR, UT = 0.5, 10.0, 0.10
CONFS = [0.001, 0.01, 0.05, 0.10, 0.20, 0.40]

frames = json.load(open(os.path.join(A2, "_rr_cands.json")))
folds = json.load(open(os.path.join(A2, "RERANK_FEATURES.json")))["phase4_folds"]
feat = {f["held_out"]: f["feature"] for f in folds}
FN = {"box_area": lambda c: c["box_area"], "box_diag": lambda c: c["box_diag"]}


def usable(c):
    return (c["iou"] >= IOU and np.isfinite(c["R"]) and c["R"] <= UR
            and c["t"] <= UT)


rows = []
for tau in CONFS:
    nat_r = rr_r = nat_u = rr_u = 0
    ncand = []
    for f in frames:
        cs = [c for c in f["cands"] if c["conf"] >= tau]
        ncand.append(len(cs))
        if not cs:
            continue
        a = cs[0]
        b = max(cs, key=FN[feat[f["set"]]])
        nat_r += int(a["iou"] >= IOU); rr_r += int(b["iou"] >= IOU)
        nat_u += int(usable(a));       rr_u += int(usable(b))
    n = len(frames)
    rows.append({"conf": tau, "median_n_cand": float(np.median(ncand)),
                 "frac_multi_cand": float(np.mean([x >= 2 for x in ncand])),
                 "native_recall": nat_r/n, "rerank_recall": rr_r/n,
                 "native_usable": nat_u/n, "rerank_usable": rr_u/n,
                 "recall_gain_frames": rr_r - nat_r,
                 "usable_gain_frames": rr_u - nat_u})

print("rerank 이득 vs conf threshold  (n=161)")
print(f"{'conf':>7}{'n_cand':>8}{'multi':>8}{'nat rec':>9}{'rr rec':>8}"
      f"{'+장':>5}{'nat use':>9}{'rr use':>8}{'+장':>5}")
print("─"*67)
for r in rows:
    print(f"{r['conf']:>7}{r['median_n_cand']:>8.0f}{r['frac_multi_cand']:>8.2f}"
          f"{r['native_recall']:>9.3f}{r['rerank_recall']:>8.3f}"
          f"{r['recall_gain_frames']:>5}{r['native_usable']:>9.3f}"
          f"{r['rerank_usable']:>8.3f}{r['usable_gain_frames']:>5}")

# negative 부작용 (secondary)
# ★ threshold 를 **먼저** 적용한 뒤 고른다.  전체에서 고른 뒤 conf 를 보면
#   positive 쪽 계산과 계약이 달라져 비교가 무의미해진다 (첫 판에서 그랬다).
raw = json.load(open(os.path.join(A2, "_cc_raw_dump.json")))
print("\nnegative 부작용 (secondary — 편향 표본이라 참고만)")
print(f"{'conf':>8}{'FP 프레임':>10}{'FP/image':>10}{'선택박스 FP':>12}  비고")
print("─"*58)
neg_out = []
for tau in (0.001, 0.05, 0.40):
    frac, fpi = [], []
    for e in raw["negative"]:
        surv = [b for b in e["boxes"] if b["conf"] >= tau]
        frac.append(len(surv) > 0)
        fpi.append(len(surv))
    a = float(np.mean(frac)); b = float(np.mean(fpi))
    neg_out.append({"conf": tau, "fp_frame_fraction": round(a, 4),
                    "fp_per_image": round(b, 4),
                    "selected_box_is_fp_fraction": round(a, 4)})
    print(f"{tau:>8}{a:>10.3f}{b:>10.3f}{a:>12.3f}  rerank 무관")
print("  -> threshold 를 먼저 적용하므로 **rerank 는 negative FP 를 바꾸지 않는다**.")
print("     살아남는 박스 개수가 같고, 어느 것을 보고할지만 달라진다.")

json.dump({"note": "rerank 는 후보가 2개 이상일 때만 작동한다. "
                   "threshold 를 먼저 적용하므로 negative FP 는 바뀌지 않는다.",
           "by_conf": rows, "negative_secondary": neg_out},
          open(os.path.join(A2, "RERANK_DEPLOY_COUPLING.json"), "w"),
          indent=1, ensure_ascii=False)
