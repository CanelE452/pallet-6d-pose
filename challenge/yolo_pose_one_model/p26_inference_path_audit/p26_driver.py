"""P26 INFERENCE PATH ABLATION — training-0.  M0/M1/M2 평가 -> 전이분석 -> gate -> 보고."""
from __future__ import annotations
import csv, json, os, subprocess, sys, time

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
NS = f"{Y}/p26_inference_path_audit"
QY = f"{Y}/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
AB = f"{Y}/runs_arch_baseline"
DATA = f"{Y}/datasets/g38_generic_only/data.yaml"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
LOG, LOCK = f"{NS}/DRIVER_LOG.txt", f"{NS}/DRIVER.lock"
W = os.path.join(ROOT, json.load(open(f"{AB}/RESULT_Y0.json"))["weights"])
LEAK = set(json.load(open(f"{QY}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
SPEC = json.load(open(f"{NS}/METHOD_SPEC.json"))
MODES = [("M0", "E2E", "P26_M0"), ("M1", "O2M_RAW", "P26_M1"), ("M2", "O2M_NMS", "P26_M2")]
sys.path.insert(0, f"{Y}/runs_arch_baseline")
import ab_common as AC                                              # noqa: E402


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); open(LOG, "a").write(line + "\n")


def notify(m):
    try: subprocess.run([NOTIFY, m], timeout=60)
    except Exception as e: log(f"discord 실패(무시): {e}")


def die(m):
    log("FAIL " + m); notify(f"❌ P26 PATH: {m}")
    if os.path.exists(LOCK): os.remove(LOCK)
    sys.exit(1)


if os.path.exists(LOCK):
    L = json.load(open(LOCK))
    if os.path.exists(f"/proc/{L['pid']}"): log("이미 실행 중"); sys.exit(0)
json.dump({"pid": os.getpid(), "start": time.strftime("%F %T")}, open(LOCK, "w"))

CKPT_SHA_BEFORE = AC.sha256(W)
CKPT_MTIME_BEFORE = os.path.getmtime(W)


def wrapped(mode, script_dir, script, tag, out_path):
    if os.path.exists(out_path):
        return json.load(open(out_path))
    r = subprocess.run([sys.executable, f"{NS}/p26_run_eval.py", mode,
                        f"{script_dir}/{script}", "--weights", W, "--tag", tag],
                       capture_output=True, text=True, cwd=script_dir)
    if not os.path.exists(out_path):
        die(f"{script} {tag} 실패: {(r.stderr or r.stdout)[-900:]}")
    return json.load(open(out_path))


def agg(rs):
    cb = [r for r in rs if r.get("correct_box")]
    e = np.concatenate([r["err"] for r in cb]) if cb else np.array([])
    return {"n_frames": len(rs), "n_correct_box": len(cb),
            "cbox": len(cb) / max(len(rs), 1),
            "median": float(np.median(e)) if e.size else None,
            "p90": float(np.percentile(e, 90)) if e.size else None,
            "gross20": float((e > 20).mean()) if e.size else None,
            "n_used_kpt": int(e.size),
            "top1_conf_median": float(np.median([r["conf"] for r in rs if r.get("conf") is not None]))
            if any(r.get("conf") is not None for r in rs) else None}


def negmetrics(rows):
    y = np.array([r["label"] for r in rows]); s = np.array([r["max_conf"] for r in rows])
    o = np.argsort(-s); ys = y[o]
    P, N = ys.sum(), (1 - ys).sum()
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    rec, fpr = tp / max(P, 1), fp / max(N, 1)
    prec = tp / np.maximum(tp + fp, 1)
    i95 = int(np.searchsorted(rec, 0.95))
    neg = np.array([r["max_conf"] for r in rows if r["label"] == 0])
    negrows = [r for r in rows if r["label"] == 0]
    return {"n_neg": int(neg.size),
            "AP_AUPRC": float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec)),
            "AUROC": float(np.trapz(rec, fpr)),
            "FPR@TPR95": float(fpr[min(i95, len(fpr) - 1)]) if len(fpr) else None,
            "FP_per_image_040": float(np.mean(
                [sum(1 for c in r["confs"] if c >= 0.40) for r in negrows])),
            "neg_detect_rate_040": float((neg >= 0.40).mean()),
            "neg_p90": float(np.percentile(neg, 90)),
            "neg_p95": float(np.percentile(neg, 95)),
            "neg_p99": float(np.percentile(neg, 99))}


R = {}
for key, mode, tag in MODES:
    log(f"평가 {key} ({mode})")
    rl = wrapped(mode, QY, "cf_real_eval.py", tag, f"{QY}/REAL_{tag}.json")
    nc = wrapped(mode, QY, "night_cand_one.py", tag, f"{QY}/NIGHT_CAND_{tag}.json")
    ns_ = wrapped(mode, QY, "neg_eval_one.py", tag, f"{QY}/NEGSCORE_{tag}.json")
    fw = wrapped(mode, NS, "p26_night_framewise.py", tag, f"{NS}/results/NIGHT_FW_{tag}.json")
    pf = [r for r in rl["per_frame"] if r["frame"] not in LEAK]
    real = {"ALL": agg(pf), "DAY": agg([r for r in pf if r["domain"] == "DAY"]),
            "NIGHT": agg([r for r in pf if r["domain"] == "NIGHT"]), "night_candidate": nc}
    s9 = AC.synth_9kp(W, f"P26_{key}") if False else None      # 아래에서 mode 별로 따로
    R[key] = {"mode": mode, "tag": tag, "real128": real,
              "negative": negmetrics(ns_["rows"]), "night_framewise": fw}
    json.dump(R[key], open(f"{NS}/RESULT_{key}_{mode}.json", "w"), indent=2, ensure_ascii=False)
    log(f"  {key} cbox {real['ALL']['cbox']:.3f} med {real['ALL']['median']:.2f} "
        f"NIGHT any {nc['any_cbox']:.3f} top1 {nc['top1_cbox']:.3f} "
        f"negAP {R[key]['negative']['AP_AUPRC']:.4f}")

# --------------------------------------------------- evaluator 수준 M0 parity
y0 = json.load(open(f"{QY}/REAL_PC_Y0.json"))["per_frame"]
m0 = json.load(open(f"{QY}/REAL_P26_M0.json"))["per_frame"]
a = {r["frame"]: r for r in y0}; b = {r["frame"]: r for r in m0}
dif = {"conf": 0.0, "iou": 0.0, "err": 0.0, "correct_box_mismatch": 0}
for f in a:
    if f not in b: dif["correct_box_mismatch"] += 1; continue
    x, z = a[f], b[f]
    if x.get("correct_box") != z.get("correct_box"): dif["correct_box_mismatch"] += 1
    for k in ("conf", "iou"):
        if x.get(k) is not None and z.get(k) is not None:
            dif[k] = max(dif[k], abs(x[k] - z[k]))
    if x.get("err") and z.get("err"):
        dif["err"] = max(dif["err"], float(np.abs(np.array(x["err"]) - np.array(z["err"])).max()))
ab_y0 = json.load(open(f"{AB}/RESULT_Y0.json"))["real128"]
lvl = {"ALL_cbox": [ab_y0["ALL"]["cbox"], R["M0"]["real128"]["ALL"]["cbox"]],
       "ALL_median": [ab_y0["ALL"]["median"], R["M0"]["real128"]["ALL"]["median"]],
       "ALL_p90": [ab_y0["ALL"]["p90"], R["M0"]["real128"]["ALL"]["p90"]],
       "NIGHT_top1": [ab_y0["night_candidate"]["top1_cbox"],
                      R["M0"]["real128"]["night_candidate"]["top1_cbox"]]}
raw = json.load(open(f"{NS}/tests/P26_M0_PARITY_RAW.json"))
par = {"raw_tensor": raw, "evaluator_level_max_abs_diff": dif,
       "evaluator_level_values": lvl,
       "evaluator_identical": all(abs(v[0] - v[1]) <= 1e-9 for v in lvl.values()),
       "PASS": bool(raw["RAW_PARITY_PASS"] and dif["correct_box_mismatch"] == 0
                    and dif["conf"] <= 1e-6 and dif["iou"] <= 1e-6 and dif["err"] <= 1e-6)}
json.dump(par, open(f"{NS}/M0_PARITY.json", "w"), indent=2, ensure_ascii=False)
json.dump(par, open(f"{NS}/tests/P26_M0_PARITY.json", "w"), indent=2, ensure_ascii=False)
if not par["PASS"]:
    die(f"M0_PARITY_FAIL {dif} / {lvl}")
log(f"M0 PARITY PASS  raw {raw['max_abs_diff']}  evaluator {dif}")

# --------------------------------------------------- synthetic (secondary)
for key, mode, tag in MODES:
    o = f"{NS}/results/SYNTH_{tag}.json"
    if not os.path.exists(o):
        r = subprocess.run([sys.executable, f"{NS}/p26_run_eval.py", mode,
                            f"{NS}/p26_synth.py", "--weights", W, "--tag", tag],
                           capture_output=True, text=True, cwd=NS)
        if not os.path.exists(o):
            log(f"  synth {key} 실패(기록만): {(r.stderr or r.stdout)[-400:]}")
    R[key]["synth"] = json.load(open(o)) if os.path.exists(o) else None
    if R[key]["synth"]:
        log(f"  {key} synth 9kp med {R[key]['synth']['kp9']['kp_median']}")
json.dump({k: R[k].get("synth") for k in R},
          open(f"{NS}/P26_SYNTH_COMPARISON.json", "w"), indent=2, ensure_ascii=False)

# --------------------------------------------------- NIGHT framewise + 전이
FW = {k: {r["frame"]: r for r in R[k]["night_framewise"]["rows"]} for k in R}
frames = sorted(FW["M0"])
with open(f"{NS}/results/P26_NIGHT_FRAMEWISE.csv", "w", newline="") as fh:
    w_ = csv.writer(fh)
    w_.writerow(["frame_id"] + [f"{m}_{c}" for m in ("M0", "M1", "M2")
                                for c in ("top1", "any", "n", "top1_conf", "top1_iou",
                                          "correct_conf", "wrong_conf", "kp_median")])
    for f in frames:
        row = [f]
        for m in ("M0", "M1", "M2"):
            d = FW[m].get(f, {})
            row += [d.get("top1"), d.get("any"), d.get("n"), d.get("top1_conf"),
                    d.get("top1_iou"), d.get("correct_conf"), d.get("wrong_conf"),
                    d.get("kp_median")]
        w_.writerow(row)

buckets = {"A_M0wrong_M2correct": [], "B_M0correct_M2wrong": [],
           "C_both_correct": [], "D_both_wrong": []}
for f in frames:
    a_, b_ = FW["M0"].get(f, {}), FW["M2"].get(f, {})
    ta, tb = bool(a_.get("top1")), bool(b_.get("top1"))
    k = ("C_both_correct" if (ta and tb) else "D_both_wrong" if (not ta and not tb)
         else "A_M0wrong_M2correct" if tb else "B_M0correct_M2wrong")
    buckets[k].append({"frame": f,
                       "M0_top1_score": a_.get("top1_conf"), "M2_top1_score": b_.get("top1_conf"),
                       "M0_correct_score": a_.get("correct_conf"), "M2_correct_score": b_.get("correct_conf"),
                       "M0_wrong_score": a_.get("wrong_conf"), "M2_wrong_score": b_.get("wrong_conf"),
                       "M0_top1_iou": a_.get("top1_iou"), "M2_top1_iou": b_.get("top1_iou"),
                       "M0_kp_median": a_.get("kp_median"), "M2_kp_median": b_.get("kp_median")})
TR = {"n_night": len(frames), "counts": {k: len(v) for k, v in buckets.items()},
      "detail_A": buckets["A_M0wrong_M2correct"], "detail_B": buckets["B_M0correct_M2wrong"],
      "M0_vs_M1_top1": {"M0": sum(FW["M0"][f].get("top1", False) for f in frames),
                        "M1": sum(FW["M1"][f].get("top1", False) for f in frames),
                        "M2": sum(FW["M2"][f].get("top1", False) for f in frames)},
      "M0_vs_M1_any": {"M0": sum(FW["M0"][f].get("any", False) for f in frames),
                       "M1": sum(FW["M1"][f].get("any", False) for f in frames),
                       "M2": sum(FW["M2"][f].get("any", False) for f in frames)},
      "★diagnostic_only": "후보 정의가 mode 마다 달라 cand 수 절대비교는 근거로 쓰지 않는다."}
json.dump(TR, open(f"{NS}/P26_TRANSITION_ANALYSIS.json", "w"), indent=2, ensure_ascii=False)
json.dump({k: R[k]["negative"] for k in R},
          open(f"{NS}/P26_REALNEG_COMPARISON.json", "w"), indent=2, ensure_ascii=False)

# --------------------------------------------------- GATE (사전등록)
m0, m2 = R["M0"], R["M2"]
G = SPEC["gate"]
rel = lambda a, b: (a - b) / max(abs(a), 1e-12)
D = {"ALL_cbox_pp": m2["real128"]["ALL"]["cbox"] - m0["real128"]["ALL"]["cbox"],
     "NIGHT_any_pp": (m2["real128"]["night_candidate"]["any_cbox"]
                      - m0["real128"]["night_candidate"]["any_cbox"]),
     "ALL_median_degrade_rel": ((m2["real128"]["ALL"]["median"] - m0["real128"]["ALL"]["median"])
                               / max(m0["real128"]["ALL"]["median"], 1e-12)),
     "NIGHT_top1_pp": (m2["real128"]["night_candidate"]["top1_cbox"]
                       - m0["real128"]["night_candidate"]["top1_cbox"]),
     "neg_AP_gain": m2["negative"]["AP_AUPRC"] - m0["negative"]["AP_AUPRC"],
     "FPR95_rel_drop": rel(m0["negative"]["FPR@TPR95"], m2["negative"]["FPR@TPR95"]),
     "detect040_rel_drop": rel(m0["negative"]["neg_detect_rate_040"],
                               m2["negative"]["neg_detect_rate_040"])}
S = G["safety_all"]
safety = {"ALL_cbox": D["ALL_cbox_pp"] >= -S["ALL_cbox_drop_pp_max"],
          "NIGHT_any": D["NIGHT_any_pp"] >= -S["NIGHT_any_cbox_drop_pp_max"],
          "ALL_median": D["ALL_median_degrade_rel"] <= S["ALL_median_degrade_rel_max"]}
B = G["benefits"]
ben = {"A_NIGHT_top1": D["NIGHT_top1_pp"] >= B["A_NIGHT_top1_gain_pp"],
       "B_neg_AP": D["neg_AP_gain"] >= B["B_neg_AP_gain"],
       "C_FPR95": D["FPR95_rel_drop"] >= B["C_FPR95_rel_drop"],
       "D_detect040": D["detect040_rel_drop"] >= B["D_neg_detect040_rel_drop"]}
nben = sum(ben.values())
loc_harm = (not safety["ALL_cbox"]) or (not safety["NIGHT_any"])
if loc_harm:
    VERD = "NMS_HARMS_LOCALIZATION"
elif all(safety.values()) and nben >= G["need"]:
    VERD = "INFERENCE_PATH_IS_MAJOR_FACTOR"
elif nben >= 1:
    VERD = "INFERENCE_PATH_PARTIAL_FACTOR"
else:
    VERD = "INFERENCE_PATH_NOT_FACTOR"

CKPT_SHA_AFTER = AC.sha256(W)
integ = {"sha_before": CKPT_SHA_BEFORE, "sha_after": CKPT_SHA_AFTER,
         "sha_unchanged": CKPT_SHA_BEFORE == CKPT_SHA_AFTER,
         "mtime_before": CKPT_MTIME_BEFORE, "mtime_after": os.path.getmtime(W),
         "mtime_unchanged": CKPT_MTIME_BEFORE == os.path.getmtime(W),
         "training_runs": 0}
OUT = {"method_spec": SPEC, "M0_parity": par, "results": R, "delta_M2_minus_M0": D,
       "safety": safety, "benefits": ben, "n_benefits": nben, "VERDICT": VERD,
       "transition": TR, "checkpoint_integrity": integ,
       "★scope": ("training-0. NIGHT n=28, seed 1. 후보 수 기반 지표(cand/frame, "
                  "wrong-candidate)는 mode 간 정의가 달라 primary evidence 아님. "
                  "FP/image@0.40 도 NMS 영향을 받으므로 AP/AUROC/FPR95/detect-rate 와 함께 읽는다.")}
json.dump(OUT, open(f"{NS}/P26_FINAL.json", "w"), indent=2, ensure_ascii=False)

f3 = lambda x: "   n/a" if x is None else f"{x:7.3f}"
L = ["# P26 INFERENCE PATH ABLATION (training-0)", "", "## real positive n=128", "```",
     f"{'mode':14} {'ALLcbox':>8} {'med':>7} {'p90':>8} {'n_kpt':>7} {'NIGHTany':>9} {'NIGHTtop1':>10}",
     "-" * 62]
for k, mode, _ in MODES:
    r = R[k]["real128"]; nc = r["night_candidate"]
    L.append(f"{k+' '+mode:14} {r['ALL']['cbox']:8.3f} {f3(r['ALL']['median'])} "
             f"{f3(r['ALL']['p90'])[-8:]:>8} {r['ALL']['n_used_kpt']:7d} "
             f"{nc['any_cbox']:9.3f} {nc['top1_cbox']:10.3f}")
L += ["```", "", "## real positive DAY / NIGHT (n_correct_box 동반)", "```",
      f"{'mode':14} {'scope':6} {'cbox':>7} {'n_cb':>5} {'med':>7} {'p90':>8} {'gross20':>8}", "-" * 58]
for k, mode, _ in MODES:
    for sc in ("ALL", "DAY", "NIGHT"):
        v = R[k]["real128"][sc]
        L.append(f"{k:14} {sc:6} {v['cbox']:7.3f} {v['n_correct_box']:5d} {f3(v['median'])} "
                 f"{f3(v['p90'])[-8:]:>8} "
                 f"{(v['gross20'] if v['gross20'] is not None else float('nan')):8.3f}")
L += ["```", "", "## real negative n=2,689", "```",
      f"{'mode':14} {'AP':>8} {'AUROC':>8} {'FPR@95':>8} {'det@.4':>8} {'FP/img@.4':>10} "
      f"{'p90':>7} {'p95':>7} {'p99':>7}", "-" * 82]
for k, mode, _ in MODES:
    n = R[k]["negative"]
    L.append(f"{k+' '+mode:14} {n['AP_AUPRC']:8.4f} {n['AUROC']:8.4f} {n['FPR@TPR95']:8.4f} "
             f"{n['neg_detect_rate_040']:8.4f} {n['FP_per_image_040']:10.4f} "
             f"{n['neg_p90']:7.3f} {n['neg_p95']:7.3f} {n['neg_p99']:7.3f}")
L += ["```", "", "## NIGHT 전이 (M0 -> M2, n=28)", "```"]
for k, v in TR["counts"].items():
    L.append(f"{k:24} {v}")
L += [f"top1 correct  M0 {TR['M0_vs_M1_top1']['M0']}  M1 {TR['M0_vs_M1_top1']['M1']}  M2 {TR['M0_vs_M1_top1']['M2']}  / 28",
      f"any  correct  M0 {TR['M0_vs_M1_any']['M0']}  M1 {TR['M0_vs_M1_any']['M1']}  M2 {TR['M0_vs_M1_any']['M2']}  / 28",
      "```", "", "## synthetic G38 val 1,998 (secondary)", "```",
      f"{'mode':14} {'boxmAP50':>9} {'poseMAP':>8} {'9kp med':>8} {'9kp p90':>8} {'cbox':>7}", "-" * 60]
for k, mode, tag in MODES:
    s = R[k].get("synth")
    if not s:
        L.append(f"{k:14} (실패)"); continue
    mp = s.get("map") or {}
    L.append(f"{k+' '+mode:14} {mp.get('box_map50', float('nan')):9.4f} "
             f"{mp.get('pose_map', float('nan')):8.4f} "
             f"{(s['kp9']['kp_median'] or float('nan')):8.2f} "
             f"{(s['kp9']['kp_p90'] or float('nan')):8.2f} {s['kp9']['cbox']:7.3f}")
L += ["```", "", "## delta M2 - M0", "```"]
for k, v in D.items():
    L.append(f"{k:26} {v:+.4f}")
L += ["```", "", f"safety {safety}", f"benefits {ben}  ({nben}/4, 필요 {G['need']})", "",
      f"**VERDICT = {VERD}**", "",
      f"M0 PARITY  raw max diff {par['raw_tensor']['max_abs_diff']}  "
      f"evaluator {par['evaluator_level_max_abs_diff']}",
      f"checkpoint sha 불변 {integ['sha_unchanged']} · mtime 불변 {integ['mtime_unchanged']} · 학습 {integ['training_runs']} 회",
      "", "★ 후보 수 기반 지표는 mode 간 정의가 달라 primary evidence 아님.",
      "★ FP/image@0.40 단독 해석 금지 — AP/AUROC/FPR95/detect-rate 와 함께 본다.",
      "★ NIGHT n=28, seed 1, training-0."]
txt = "\n".join(L)
open(f"{NS}/FINAL_P26_INFERENCE_PATH_REPORT.md", "w").write(txt)
print(txt)
notify(txt[:1800])
log(f"=== 완료 {VERD} ===")
if os.path.exists(LOCK): os.remove(LOCK)
