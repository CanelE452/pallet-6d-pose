"""S7 — 아침에 한 화면만 보면 되도록 최종 표를 만든다."""
import glob, json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
Y = f"{ROOT}/challenge/yolo_pose_one_model"
R = f"{Y}/runs_pallet_loss"
O = f"{R}/overnight_20260823"
EP = ["epoch0", "epoch10", "epoch20", "epoch30", "epoch40", "epoch50", "last"]
ARMS = [("A0", "A0"), ("ASC42", "ASC"), ("ASC43", "ASC43"),
        ("TAKL42", "TAKL"), ("NRL42", "NRL"), ("ASC+TAKL", "COMB")]


def L(tag):
    p = f"{R}/A1_MEASURE_{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None


state = json.load(open(f"{O}/OVERNIGHT_STATE.json"))
md = ["# OVERNIGHT LOSS QUEUE — 2026-08-23", "",
      "> screen dataset = V1_FIXED_MATCHED10K (A0 대비 변화를 loss 에 귀속시키기 위한 것).",
      "> 모든 판정은 **engineering screen** 이다. real 평가 전 METHOD_SUPPORTED 선언 금지.", ""]

md += ["## 1. RUN STATUS", "", "```",
       f"{'stage':16} {'status':10} {'verdict':34} reason", "-" * 96]
for s, v in state.items():
    if s.startswith("_"):
        continue
    ver = v.get("verdict") or v.get("TAKL_SYNTH_SIGNAL") or v.get("NRL_SYNTH_SIGNAL") \
        or v.get("ASC_SEED_ROBUST") or v.get("COMBINED_SIGNAL") or "-"
    md.append(f"{s:16} {v.get('status','?'):10} {str(ver):34} {v.get('reason','')[:40]}")
md += ["```", ""]

md += ["## 2. SYNTHETIC (v1 val 133, 동일 evaluator)", "", "```",
       f"{'model':10} {'mAP50-95':>9} {'corner med':>11} {'p90':>8} {'gross20':>8} "
       f"{'flip':>7} {'ep20 mAP':>9}", "-" * 70]
conv = {}
for nm, tg in ARMS:
    last = L(f"{tg}_last")
    if not last:
        continue
    e20 = L(f"{tg}_epoch20")
    md.append(f"{nm:10} {last['pose_map50_95']:9.4f} "
              f"{last['corner_px_identity_median']:11.2f} {last['corner_px_identity_p90']:8.2f} "
              f"{last.get('gross20',float('nan')):8.4f} {100*last['flip_rate']:6.2f}% "
              f"{(e20['pose_map50_95'] if e20 else float('nan')):9.4f}")
    cur = []
    for e in EP:
        d = L(f"{tg}_{e}")
        cur.append((e, d["pose_map50_95"] if d else None))
    conv[nm] = cur
md += ["```", ""]

md += ["## 3. REAL (EXPLORATORY — canonical semantic accuracy 아님)", "", "```",
       f"{'arm':8} {'N':>4} {'det':>7} {'corner med':>11} {'p90':>8} {'gross20':>8}  verdict", "-" * 72]
for p in sorted(glob.glob(f"{O}/REAL_*_VERDICT.json")):
    d = json.load(open(p))
    a = d["arm"]
    md.append(f"{a:8} {d['n_paired']:4d} {d['detection_rate'][a]:7.3f} "
              f"{d[a]['corner_median']:11.2f} {d[a]['corner_p90']:8.2f} "
              f"{d[a]['gross20']:8.4f}  {d['verdict']}")
    md.append(f"{'  (A0)':8} {d['n_paired']:4d} {d['detection_rate']['A0']:7.3f} "
              f"{d['A0']['corner_median']:11.2f} {d['A0']['corner_p90']:8.2f} "
              f"{d['A0']['gross20']:8.4f}")
md += ["```", ""]
dg = f"{O}/REAL_DIAGNOSTIC_A0.json"
if os.path.exists(dg):
    g = json.load(open(dg))
    md += ["★ real 평가기 건전성 진단 (A0):", "```",
           f"  G(4) 최소   median {g['G4_min']['median']:.2f}",
           f"  전체 8! 최소 median {g['all8_min']['median']:.2f}",
           f"  Hungarian   median {g['hungarian']['median']:.2f}",
           f"  box IoU     median {g['box_iou']['median']:.3f}", "```",
           "세 값이 같다 = 순열/규약 문제가 아니라 **모델이 real 에서 실제로 실패**한다.",
           "즉 real 평가는 현재 loss 후보를 구분할 해상도가 없다.", ""]

md += ["## 4. CONVERGENCE (mAP50-95 가 처음 넘는 epoch)", "", "```",
       f"{'model':10} " + " ".join(f"{t:>7}" for t in (0.3, 0.4, 0.5, 0.6, 0.7)) + "     AULC", "-" * 62]
for nm, cur in conv.items():
    vals = [(int(e.replace("epoch", "")) if e != "last" else 59, v)
            for e, v in cur if v is not None]
    row = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        hit = next((ep for ep, v in vals if v >= t), None)
        row.append(f"{hit if hit is not None else '-':>7}")
    xs = [x for x, _ in vals]
    ys = [v for _, v in vals]
    aulc = float(np.trapz(ys, xs) / max(xs[-1] - xs[0], 1)) if len(xs) > 1 else float("nan")
    md.append(f"{nm:10} " + " ".join(row) + f"  {aulc:8.4f}")
md += ["```", ""]

# ---- 5. BEST CANDIDATE -------------------------------------------------------
real_pos = []
for p in glob.glob(f"{O}/REAL_*_VERDICT.json"):
    d = json.load(open(p))
    if d["verdict"] in ("POSITIVE", "POINT_ESTIMATE_POSITIVE_UNCERTAIN"):
        real_pos.append((d["arm"], d["verdict"]))
synth_pass = [s for s, v in state.items()
              if v.get("TAKL_SYNTH_SIGNAL") == "PASS" or v.get("NRL_SYNTH_SIGNAL") == "PASS"
              or v.get("COMBINED_SIGNAL") == "PASS"]
if real_pos:
    best, lvl = real_pos[0][0], f"REAL_{real_pos[0][1]}"
    why = "real 에서 양의 신호"
elif synth_pass:
    best, lvl = synth_pass[0], "SYNTHETIC_ONLY_PROVISIONAL"
    why = "synthetic gate 통과, real 미확인"
else:
    best, lvl = "A0", "NONE — A0 유지"
    why = "어떤 후보도 gate 를 넘지 못함"
failed = [s for s, v in state.items() if v.get("status") in ("FAIL", "SKIPPED")]
md += ["## 5. BEST CANDIDATE", "", "```",
       f"BEST_OVERNIGHT_CANDIDATE = {best}",
       f"EVIDENCE_LEVEL           = {lvl}",
       f"WHY                      = {why}",
       f"WHAT_FAILED              = {', '.join(failed) if failed else '없음'}", "```", "",
       "## 6. 다음날 자동 실행하지 않는 것", "",
       "추가 hyperparameter sweep / seed 44,45 / dataset 변경 / 새 architecture /",
       "self-training / paper final claim — 전부 사용자 확인 후.", "",
       "## 근거 태그", "",
       "- [확인] 위 모든 수치는 disk artifact 에서 읽었다.",
       "- [추정] 메커니즘 해석은 추정이다.",
       "- [미검증] real 전이·seed 일반화는 확립되지 않았다. ASC 는 현재 **convergence "
       "acceleration** 만 확립됐고 final accuracy 우위는 주장하지 않는다."]
open(f"{O}/OVERNIGHT_SUMMARY.md", "w").write("\n".join(md))
json.dump({"state": state, "best": best, "evidence": lvl, "why": why,
           "failed": failed, "convergence": conv},
          open(f"{O}/OVERNIGHT_SUMMARY.json", "w"), indent=2, ensure_ascii=False)
print("\n".join(md))
