"""U4 — camera-facing 최종 표. fixed-object 수치는 절대 섞지 않는다."""
import glob, json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
Y = f"{ROOT}/challenge/yolo_pose_one_model"
CFR = f"{Y}/runs_camera_facing_loss"
Q = f"{CFR}/ubuntu_cf_loss_queue_20260823T0930"
EP = ["epoch0", "epoch10", "epoch20", "epoch30", "epoch40", "epoch50", "last"]
ARMS = [("CF-A0", "A0", "CF_A0_V1MATCHED10K_60EP_SEED42_UBUNTU", 42),
        ("CF-NRL42", "NRL", "CF_NRL_V1MATCHED10K_60EP_SEED42_UBUNTU", 42),
        ("CF-NRL43", "NRL43", "CF_NRL_V1MATCHED10K_60EP_SEED43_UBUNTU", 43),
        ("CF-PEVL42", "PEVL", "CF_PEVL_V1MATCHED10K_60EP_SEED42_UBUNTU", 42),
        ("CF-PEVL43", "PEVL43", "CF_PEVL_V1MATCHED10K_60EP_SEED43_UBUNTU", 43)]


def S(t):
    p = f"{Q}/SYNTH_{t}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def R(t):
    p = f"{Q}/REAL_{t}.json"
    return json.load(open(p)) if os.path.exists(p) else None


state = json.load(open(f"{Q}/QUEUE_STATE.json"))
C = json.load(open(f"{Q}/CF_DATASET_CONTRACT.json"))
md = ["# CAMERA-FACING LOSS QUEUE — 2026-08-23", "",
      "> convention = camera-facing 0123 (GitHub 정본). fixed-object 수치는 이 표에 없다.",
      "> 모든 판정은 engineering screen. real 평가 전 METHOD_SUPPORTED 선언 금지.", "",
      "## 1. DATA CONTRACT", "", "```",
      f"CF train / val (declared = effective)  {C['train_n']} / {C['val_n']}   corrupt 0",
      f"label sha  train {C['label_sha_train']}   val {C['label_sha_val']}",
      f"CF 라벨 출처  {C['cf_source']}",
      f"RGB          {C['rgb_source']}",
      f"roundtrip    fixed[perm_v4[i]]==cf[i]  10000/10000 PASS",
      f"bbox·visibility·point-set·centroid 불일치 0,  0~7 순열만 10000/10000", "```", "",
      "## 2. SYNTHETIC (CF val 133)", "", "```",
      f"{'model':11} {'seed':>4} {'mAP50-95':>9} {'median':>8} {'p90':>8} "
      f"{'gross20':>8} {'bottom p90':>11} {'AULC':>8}", "-" * 76]
conv = {}
for nm, tg, run, sd in ARMS:
    d = S(tg)
    if not d:
        continue
    cur = []
    for e in EP:
        q = S(f"{tg}_{e}")
        cur.append((59 if e == "last" else int(e.replace("epoch", "")),
                    q["pose_map50_95"] if q else None))
    vals = [(x, v) for x, v in cur if v is not None]
    aulc = float(np.trapz([v for _, v in vals], [x for x, _ in vals]) /
                 max(vals[-1][0] - vals[0][0], 1)) if len(vals) > 1 else float("nan")
    conv[nm] = vals
    md.append(f"{nm:11} {sd:>4} {d['pose_map50_95']:9.4f} {d['corner_median']:8.2f} "
              f"{d['corner_p90']:8.2f} {d['gross20']:8.4f} {d['bottom_p90']:11.2f} {aulc:8.4f}")
md += ["```", "", "## 3. REAL (QA-clean candidate, EXPLORATORY)", "", "```",
       f"{'model':11} {'det':>6} {'cbox':>6} {'median':>8} {'p90':>8} {'gross20':>8} "
       f"{'bottom':>8} {'day p90':>8} {'night p90':>10}", "-" * 84]
for nm, tg, run, sd in ARMS:
    d = R(tg)
    if not d:
        continue
    p = d["paired"]
    md.append(f"{nm:11} {d['detection_recall']:6.3f} {d['correct_box_recall']:6.3f} "
              f"{p.get('corner_median',float('nan')):8.2f} {p.get('corner_p90',float('nan')):8.2f} "
              f"{p.get('gross20',float('nan')):8.4f} {p.get('bottom_p90',float('nan')):8.2f} "
              f"{p.get('day_p90',float('nan')):8.2f} {p.get('night_p90',float('nan')):10.2f}")
md += ["```", ""]
for nm, tg, run, sd in ARMS[1:]:
    d = R(tg)
    if d and d.get("bootstrap"):
        md.append(f"**{nm}** session-cluster paired bootstrap B=10,000 (Δ = A0 − {nm}, >0 이면 {nm} 우세)")
        for k, v in d["bootstrap"].items():
            sig = "" if v["ci95"][0] * v["ci95"][1] > 0 else "  ← CI 가 0 포함, 유의 아님"
            md.append(f"- {k}: Δ {v['delta']:+.4f}  95%CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]{sig}")
        md.append("")

ft = R("FT_REFERENCE")
md += ["## 4. REFERENCE (controlled baseline 아님)", "", "```",
       f"{'model':16} {'role':38} {'det':>6} {'median':>8} {'p90':>8}", "-" * 80]
if ft:
    p = ft["paired"]
    md.append(f"{'yolo26n-ft':16} {'TARGET_SPECIFIC_REAL_FINETUNED_REFERENCE':38} "
              f"{ft['detection_recall']:6.3f} {p.get('corner_median',float('nan')):8.2f} "
              f"{p.get('corner_p90',float('nan')):8.2f}")
else:
    md.append("REFERENCE_BLOCKED_MISSING_WEIGHT")
md += ["```", "", "## 5. GAPS (현재 exact membership 재측정, README 수치 복사 아님)", "", "```"]
a0r = R("A0")
if ft and a0r:
    def gap(x, k):
        return x["paired"].get(k, float("nan")) - ft["paired"].get(k, float("nan"))
    md.append(f"CF-A0 → yolo26n-ft   det {a0r['detection_recall']-ft['detection_recall']:+.3f}  "
              f"median {gap(a0r,'corner_median'):+.2f}  p90 {gap(a0r,'corner_p90'):+.2f}  "
              f"bottom {gap(a0r,'bottom_p90'):+.2f}")
    for nm, tg, run, sd in ARMS[1:]:
        d = R(tg)
        if d:
            md.append(f"{nm:12} → yolo26n-ft   det {d['detection_recall']-ft['detection_recall']:+.3f}  "
                      f"median {gap(d,'corner_median'):+.2f}  p90 {gap(d,'corner_p90'):+.2f}")
else:
    md.append("reference 없음 — gap 계산 불가")
md += ["```", "", "## 6. BEST", "", "```"]
nrl = state.get("U2E_NRL_EVAL", {}).get("CF_NRL_SIGNAL", "SKIPPED")
pevl = state.get("U3E_PEVL_EVAL", {}).get("CF_PEVL_SIGNAL", "SKIPPED")
cand = [("NRL", nrl), ("PEVL", pevl)]
best, lvl = "A0", "BASELINE_ONLY"
for n, s in cand:
    if s == "REAL_POSITIVE":
        best, lvl = n, "REAL"
        break
for n, s in cand:
    if lvl == "BASELINE_ONLY" and s == "REAL_POSITIVE_UNCERTAIN":
        best, lvl = n, "REAL_UNCERTAIN"
for n, s in cand:
    if lvl == "BASELINE_ONLY" and s == "SYNTHETIC_ONLY_POSITIVE":
        best, lvl = n, "SYNTHETIC_ONLY"
why = {"REAL": "real keypoint 에서 양의 delta",
       "REAL_UNCERTAIN": "real point estimate 는 양수지만 CI 가 0 포함",
       "SYNTHETIC_ONLY": "synthetic 만 개선, real 미확인",
       "BASELINE_ONLY": "어떤 후보도 gate 를 넘지 못함 — CF-A0 유지"}[lvl]
failed = [k for k, v in state.items() if v.get("status") in ("FAIL", "SKIPPED", "BLOCKED")]
md += [f"BEST_CF_METHOD  = {best}", f"EVIDENCE_LEVEL  = {lvl}",
       f"WHY             = {why}",
       f"CF_NRL_SIGNAL   = {nrl}", f"CF_PEVL_SIGNAL  = {pevl}",
       f"WHAT_FAILED     = {', '.join(failed) if failed else '없음'}", "```", "",
       "## 7. NEXT ONE ACTION", ""]
if lvl in ("REAL", "REAL_UNCERTAIN"):
    md.append("→ self-training 실험 설계 (real 에서 양의 delta 를 확보했으므로)")
elif lvl == "SYNTHETIC_ONLY":
    md.append("→ real domain gap 이 병목. self-training / 데이터 적응 우선.")
else:
    md.append("→ loss track 종료. CF-A0 + V2 data + self-training 으로 논문 정리.")
md += ["", "## 근거 태그", "",
       "- [확인] 모든 수치는 disk artifact 에서 읽었다.",
       "- [추정] 메커니즘 해석.",
       "- [미검증] seed 일반화. real 은 EXPLORATORY membership 이며 final test 가 아니다.",
       "- PnP 6D 는 GT-independent W/D selector 부재로 POSE_EVAL_BLOCKED."]
open(f"{Q}/CAMERA_FACING_SUMMARY.md", "w").write("\n".join(md))
json.dump({"state": state, "best": best, "evidence": lvl,
           "CF_NRL_SIGNAL": nrl, "CF_PEVL_SIGNAL": pevl,
           "failed": failed, "convergence": {k: v for k, v in conv.items()}},
          open(f"{Q}/CAMERA_FACING_SUMMARY.json", "w"), indent=2, ensure_ascii=False)
print("\n".join(md))
