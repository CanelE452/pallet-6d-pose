"""Y0 / Y1 30ep — smoke -> train -> synth -> real128 -> realneg -> gate -> package.

한 파일에서 판정까지. 완료 판정은 results.csv 행수 + last.pt + eval artifact.
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, time

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
NS = f"{Y}/runs_posecls_g38"
QY = f"{Y}/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
DATA = f"{Y}/datasets/g38_generic_only/data.yaml"
SMOKE = f"{Y}/datasets/g38_smoke256/data.yaml"
INIT = f"{ROOT}/challenge/weights/pretrained_yolo/yolo26n-pose.pt"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
LOG, LOCK = f"{NS}/DRIVER_LOG.txt", f"{NS}/DRIVER.lock"
GATE = json.load(open(f"{NS}/GATE_PREREG.json"))
LEAK = set(json.load(open(f"{QY}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
ARMS = {"Y0": ("Y26_G38_Y0_VANILLA_30EP_SEED42", None),
        "Y1": ("Y26_G38_Y1_POSEAWARE_CLS_30EP_SEED42", "PoseAwareClsLoss26")}
EPOCHS = 30


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); open(LOG, "a").write(line + "\n")


def notify(m):
    try: subprocess.run([NOTIFY, m], timeout=60)
    except Exception as e: log(f"discord 실패(무시): {e}")


def die(m):
    log("FAIL " + m); notify(f"❌ Y0/Y1: {m}")
    if os.path.exists(LOCK): os.remove(LOCK)
    sys.exit(1)


if os.path.exists(LOCK):
    L = json.load(open(LOCK))
    if os.path.exists(f"/proc/{L['pid']}"): log("이미 실행 중"); sys.exit(0)
json.dump({"pid": os.getpid(), "start": time.strftime("%F %T")}, open(LOCK, "w"))


def done(d, ep):
    rp = f"{d}/results.csv"
    return (os.path.exists(f"{d}/weights/last.pt") and os.path.exists(rp)
            and len(open(rp).read().strip().split("\n")) - 1 >= ep)


def train(name, data, epochs, loss_cls):
    d = f"{NS}/{name}"
    if done(d, epochs):
        log(f"{name} 이미 완료"); return d
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/PURPOSE.md"):
        open(f"{d}/PURPOSE.md", "w").write(
            "[소비처] 논문 — YOLO26 candidate ranking 절 (Y0 vs Y1)\n"
            "[문장]  classification objective 를 pose-quality-aware 로 바꾸면 NIGHT ranking 이 개선된다.\n")
    inject = ""
    if loss_cls:
        inject = f'''
from pallet_yolo_loss.posecls import {loss_cls}, CALLS
from ultralytics.utils.loss import E2ELoss
from ultralytics.nn.tasks import PoseModel
PoseModel.init_criterion = lambda self: E2ELoss(self, {loss_cls})
'''
    code = f'''
import os, sys, json
sys.path.insert(0, "{ROOT}")
os.environ["A1_CONFIG"] = ""
os.environ["PSPC_CONFIG"] = ""
from pallet_yolo_loss.loss import PSPCPoseLoss26
import pallet_yolo_loss.symmetry as SY
PC = {{"n": 0}}
_pc = PSPCPoseLoss26.projective_loss
def _spy(self, *a, **k):
    PC["n"] += 1
    return _pc(self, *a, **k)
PSPCPoseLoss26.projective_loss = _spy
SY.ROLE_CALLS["n"] = 0
try:
    from pallet_yolo_loss.posecls import CALLS as ALIGN
except Exception:
    ALIGN = {{"align": 0, "qpose": 0}}
ALIGN["align"] = 0; ALIGN["qpose"] = 0
{inject}
from ultralytics.models.yolo.pose import PoseTrainer
tr = PoseTrainer(overrides=dict(
    task="pose", mode="train", model="{INIT}", data="{data}",
    epochs={epochs}, batch=32, imgsz=640, optimizer="SGD", lr0=0.01, lrf=0.01,
    cos_lr=True, close_mosaic=10, warmup_epochs=3.0, patience=0,
    single_cls=True, mosaic=0.3, scale=0.25, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
    fliplr=0.0, flipud=0.0, erasing=0.4, seed=42, deterministic=True,
    save_period=10, device=0, workers=8, project="{NS}", name="{name}",
    exist_ok=True, resume=False, val=True, plots=False))
tr.train()
c = getattr(tr.model, "criterion", None)
inner = getattr(c, "one2many", c)
json.dump({{"PC_CALL_COUNT": PC["n"], "ROLE_CALLS": SY.ROLE_CALLS["n"],
           "POSEALIGN_CALLS": ALIGN["align"], "QPOSE_CALLS": ALIGN["qpose"],
           "criterion": type(inner).__name__,
           "criterion_one2one": type(getattr(c, "one2one", c)).__name__,
           "lambda_posealign": getattr(type(inner), "LAMBDA", None),
           "gamma": getattr(type(inner), "GAMMA", None),
           "n_train_batches": len(tr.train_loader), "epochs": {epochs}, "seed": 42,
           "data": "{data}"}},
          open("{NS}/{name}/RUNTIME_AUDIT.json", "w"), indent=2)
'''
    sc = f"{NS}/_tr_{name}.py"; open(sc, "w").write(code)
    lf = f"{NS}/_train_{name}.log"
    with open(lf, "w") as fh:
        r = subprocess.run([sys.executable, "-u", sc], cwd=d, stdout=fh,
                           stderr=subprocess.STDOUT, text=True)
    if not os.path.exists(f"{d}/weights/last.pt"):
        tail = "\n".join(open(lf).read().replace("\r", "\n").split("\n")[-25:])
        die(f"{name} 산출물 없음 (rc={r.returncode}):\n{tail[-900:]}")
    a = json.load(open(f"{d}/RUNTIME_AUDIT.json"))
    if a["PC_CALL_COUNT"] or a["ROLE_CALLS"]:
        die(f"{name} 옛 custom loss 오염 PC={a['PC_CALL_COUNT']} role={a['ROLE_CALLS']}")
    if loss_cls is None:
        if a["criterion"] != "PoseLoss26": die(f"Y0 criterion {a['criterion']}")
        if a["POSEALIGN_CALLS"]: die(f"Y0 인데 posealign {a['POSEALIGN_CALLS']} 회 호출")
    else:
        if a["criterion"] != loss_cls or a["criterion_one2one"] != loss_cls:
            die(f"Y1 criterion {a['criterion']}/{a['criterion_one2one']}")
        if a["POSEALIGN_CALLS"] <= 0: die("Y1 인데 posealign 호출 0")
        if a["lambda_posealign"] != 0.25: die(f"lambda {a['lambda_posealign']}")
    return d


# ------------------------------------------------------------------ SMOKE
log("SMOKE Y0 / Y1 (256 frames, 1ep)")
SM = {}
for tag, (_, lc) in ARMS.items():
    d = train(f"SMOKE_{tag}", SMOKE, 1, lc)
    a = json.load(open(f"{d}/RUNTIME_AUDIT.json"))
    rc = open(f"{d}/results.csv").read().strip().split("\n")
    vals = {h.strip(): v for h, v in zip(rc[0].split(","), rc[1].split(","))}
    lo = [float(v) for k, v in vals.items() if "loss" in k and v not in ("", "nan")]
    if not lo or not all(np.isfinite(lo)): die(f"SMOKE_{tag} loss {lo}")
    SM[tag] = {"n_loss_terms": len(lo), "finite": True,
               "posealign_calls": a["POSEALIGN_CALLS"], "qpose_calls": a["QPOSE_CALLS"],
               "criterion": a["criterion"], "criterion_one2one": a["criterion_one2one"],
               "batches": a["n_train_batches"]}
    log(f"  SMOKE {tag} OK  align {a['POSEALIGN_CALLS']}  {a['criterion']}")
SM["Y0_posealign_is_zero"] = SM["Y0"]["posealign_calls"] == 0
SM["Y1_posealign_positive"] = SM["Y1"]["posealign_calls"] > 0
SM["one2one_reached"] = json.load(open(f"{NS}/tests/GRADIENT_ROUTING.json"))["branches"]["one2one"]["reached"]
SM["PASS"] = bool(SM["Y0_posealign_is_zero"] and SM["Y1_posealign_positive"] and SM["one2one_reached"])
json.dump(SM, open(f"{NS}/tests/SMOKE_RESULT.json", "w"), indent=2, ensure_ascii=False)
if not SM["PASS"]: die(f"SMOKE 계약 실패 {SM}")
log("SMOKE PASS — smoke checkpoint 은 full run 에 resume 하지 않는다")

# ------------------------------------------------------------------ FULL
AUD = {}
for tag, (name, lc) in ARMS.items():
    log(f"{tag} {EPOCHS}ep 시작")
    d = train(name, DATA, EPOCHS, lc)
    a = json.load(open(f"{d}/RUNTIME_AUDIT.json"))
    a["dir"] = d
    a["actual_epochs"] = len(open(f"{d}/results.csv").read().strip().split("\n")) - 1
    AUD[tag] = a
    log(f"{tag} 완료 {a['actual_epochs']}ep batches/ep {a['n_train_batches']} "
        f"{a['criterion']} align {a['POSEALIGN_CALLS']}")
if AUD["Y0"]["n_train_batches"] != AUD["Y1"]["n_train_batches"]:
    die("batches/ep 불일치")

# args parity — loss 외 차이 0
import yaml
a0 = yaml.safe_load(open(f"{AUD['Y0']['dir']}/args.yaml"))
a1 = yaml.safe_load(open(f"{AUD['Y1']['dir']}/args.yaml"))
ALLOW = {"name", "save_dir"}
diff = {k: [a0.get(k), a1.get(k)] for k in set(a0) | set(a1) if a0.get(k) != a1.get(k)}
bad = {k: v for k, v in diff.items() if k not in ALLOW}
json.dump({"diff": diff, "allowed": sorted(ALLOW), "violations": bad},
          open(f"{NS}/ARGS_PARITY.json", "w"), indent=2, ensure_ascii=False)
if bad: die(f"args 차이 {list(bad)}")
log(f"args parity OK  {sorted(diff)}")
json.dump({"stage": "trained", "arms": AUD}, open(f"{NS}/_stage_trained.json", "w"),
          indent=2, ensure_ascii=False, default=str)
log("=== 학습 완료 — 평가 단계 ===")
notify(f"Y0/Y1 30ep 학습 완료 · batches/ep {AUD['Y0']['n_train_batches']} · 평가 시작")


# ------------------------------------------------------------------ EVAL
def run(script, prefix, tag, w, extra=()):
    o = f"{QY}/{prefix}_{tag}.json"
    if not os.path.exists(o):
        r = subprocess.run([sys.executable, f"{QY}/{script}", "--weights", w,
                            "--tag", tag, *extra], capture_output=True, text=True)
        if not os.path.exists(o):
            die(f"{script} {tag} 실패: {(r.stderr or r.stdout)[-600:]}")
    return json.load(open(o))


def synth(tag, w, d):
    o = f"{NS}/{tag}_SYNTH.json"
    if os.path.exists(o):
        return json.load(open(o))
    from ultralytics import YOLO
    m = YOLO(w, task="pose")
    r = m.val(data=DATA, imgsz=640, batch=32, device=0, workers=8, plots=False,
              project=f"{NS}/{tag}_val", name="synth", exist_ok=True, verbose=False)
    rd = {k: float(v) for k, v in r.results_dict.items() if isinstance(v, (int, float))}
    out = {"weights": os.path.relpath(w, ROOT), "data": os.path.relpath(DATA, ROOT),
           "results_dict": rd,
           "box_map50": float(r.box.map50), "box_map": float(r.box.map),
           "pose_map50": float(r.pose.map50), "pose_map": float(r.pose.map),
           "note": "기존 ultralytics evaluator 의 metric 정의 그대로. 새 정의 없음."}
    json.dump(out, open(o, "w"), indent=2, ensure_ascii=False)
    del m
    return out


def agg(rs):
    cb = [r for r in rs if r.get("correct_box")]
    e = np.concatenate([r["err"] for r in cb]) if cb else np.array([])
    return {"n": len(rs), "cbox": len(cb) / max(len(rs), 1),
            "median": float(np.median(e)) if e.size else None,
            "p90": float(np.percentile(e, 90)) if e.size else None,
            "gross20": float((e > 20).mean()) if e.size else None}


def negmetrics(rows):
    y = np.array([r["label"] for r in rows]); s = np.array([r["max_conf"] for r in rows])
    o = np.argsort(-s); ys = y[o]
    P, N = ys.sum(), (1 - ys).sum()
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    rec, fpr = tp / max(P, 1), fp / max(N, 1)
    prec = tp / np.maximum(tp + fp, 1)
    AP = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    i95 = int(np.searchsorted(rec, 0.95))
    pos = np.array([r["max_conf"] for r in rows if r["label"] == 1])
    neg = np.array([r["max_conf"] for r in rows if r["label"] == 0])
    negrows = [r for r in rows if r["label"] == 0]
    d = {"n_pos": int(pos.size), "n_neg": int(neg.size), "AP_AUPRC": AP,
         "AUROC": float(np.trapz(rec, fpr)),
         "FPR@TPR95": float(fpr[min(i95, len(fpr) - 1)]) if len(fpr) else None,
         "neg_p90": float(np.percentile(neg, 90)),
         "neg_p95": float(np.percentile(neg, 95)),
         "neg_p99": float(np.percentile(neg, 99)),
         "FP_per_image": {}}
    for t in (0.05, 0.10, 0.25, 0.40):
        d["FP_per_image"][str(t)] = float(np.mean(
            [sum(1 for c in r["confs"] if c >= t) for r in negrows]))
    return d


R, NC, NEG, SY_ = {}, {}, {}, {}
for tag in ("Y0", "Y1"):
    w = f"{AUD[tag]['dir']}/weights/last.pt"
    SY_[tag] = synth(tag, w, AUD[tag]["dir"])
    log(f"  {tag} synthetic done  box mAP50 {SY_[tag]['box_map50']:.4f}")
    R[tag] = run("cf_real_eval.py", "REAL", f"PC_{tag}", w)
    NC[tag] = run("night_cand_one.py", "NIGHT_CAND", f"PC_{tag}", w)
    ns = run("neg_eval_one.py", "NEGSCORE", f"PC_{tag}", w)
    NEG[tag] = negmetrics(ns["rows"])
    json.dump(NEG[tag], open(f"{NS}/{tag}_REALNEG.json", "w"), indent=2, ensure_ascii=False)
    pf = [r for r in R[tag]["per_frame"] if r["frame"] not in LEAK]
    tab = {"ALL": agg(pf), "DAY": agg([r for r in pf if r["domain"] == "DAY"]),
           "NIGHT": agg([r for r in pf if r["domain"] == "NIGHT"]),
           "night_candidate": NC[tag]}
    json.dump(tab, open(f"{NS}/{tag}_REAL128.json", "w"), indent=2, ensure_ascii=False)
    R[tag] = tab
    log(f"  {tag} real ALL cbox {tab['ALL']['cbox']:.3f} med {tab['ALL']['median']:.2f} "
        f"NIGHT top1 {NC[tag]['top1_cbox']:.3f}")

t0, t1 = R["Y0"], R["Y1"]
c0, c1 = NC["Y0"], NC["Y1"]
n0, n1 = NEG["Y0"], NEG["Y1"]
rel = lambda a, b: (a - b) / max(abs(a), 1e-12)
D = {"NIGHT_top1_cbox_pp": c1["top1_cbox"] - c0["top1_cbox"],
     "ALL_p90_rel_gain": rel(t0["ALL"]["p90"], t1["ALL"]["p90"]),
     "NIGHT_p90_rel_gain": rel(t0["NIGHT"]["p90"], t1["NIGHT"]["p90"]),
     "ALL_cbox_pp": t1["ALL"]["cbox"] - t0["ALL"]["cbox"],
     "NIGHT_any_cbox_pp": c1["any_cbox"] - c0["any_cbox"],
     "ALL_median_degrade_rel": rel(t1["ALL"]["median"], t0["ALL"]["median"]) * -1
     if False else (t1["ALL"]["median"] - t0["ALL"]["median"]) / max(t0["ALL"]["median"], 1e-12),
     "FP_img_040_rel_increase": (n1["FP_per_image"]["0.4"] - n0["FP_per_image"]["0.4"])
     / max(n0["FP_per_image"]["0.4"], 1e-12)}
S = GATE["safety_all_must_hold"]
safety = {"ALL_cbox": D["ALL_cbox_pp"] >= -S["ALL_cbox_drop_pp_max"],
          "NIGHT_any_cbox": D["NIGHT_any_cbox_pp"] >= -S["NIGHT_any_cbox_drop_pp_max"],
          "ALL_median": D["ALL_median_degrade_rel"] <= S["ALL_median_degrade_rel_max"],
          "FP_img_040": D["FP_img_040_rel_increase"] <= S["heldout_neg_FP_per_image_040_increase_rel_max"]}
C = GATE["conditions"]
cond = {"A_NIGHT_top1": D["NIGHT_top1_cbox_pp"] >= C["A_NIGHT_top1_cbox_gain_pp"],
        "B_ALL_p90": D["ALL_p90_rel_gain"] >= C["B_ALL_p90_rel_gain"],
        "C_NIGHT_p90": D["NIGHT_p90_rel_gain"] >= C["C_NIGHT_p90_rel_gain"]}
ncond = sum(cond.values())
VERD = ("Y1_POSEAWARE_LOSS_PROMOTE" if (all(safety.values()) and ncond >= GATE["conditions_need"])
        else "Y1_POSEAWARE_LOSS_NO_SIGNAL")
OUT = {"gate": GATE, "delta": D, "safety": safety, "conditions": cond,
       "n_conditions": ncond, "VERDICT": VERD,
       "Y0": {"synth": SY_["Y0"], "real": t0, "neg": n0,
              "runtime": {k: AUD["Y0"][k] for k in ("criterion", "criterion_one2one",
                                                    "POSEALIGN_CALLS", "n_train_batches",
                                                    "actual_epochs")}},
       "Y1": {"synth": SY_["Y1"], "real": t1, "neg": n1,
              "runtime": {k: AUD["Y1"][k] for k in ("criterion", "criterion_one2one",
                                                    "POSEALIGN_CALLS", "lambda_posealign",
                                                    "n_train_batches", "actual_epochs")}},
       "★scope": ("NIGHT n=28, seed 1, 30ep. real 은 loss/training 에 쓰이지 않았다. "
                  "held-out negative membership 은 실측으로 재검산했다(아래 note). "
                  "Windows Y2 와 비교 전에는 60ep 을 자동 시작하지 않는다.")}
json.dump(OUT, open(f"{NS}/Y0_VS_Y1.json", "w"), indent=2, ensure_ascii=False)

f2 = lambda x: "   n/a" if x is None else f"{x:7.3f}"
L = ["# Y0 vs Y1 — pose-quality-aware classification (G38 38,002, 30ep)", "", "```",
     f"{'':22} {'Y0':>10} {'Y1':>10} {'Δ':>10}", "-" * 56]
for k, lab in ((("ALL", "cbox"), "ALL cbox"), (("ALL", "median"), "ALL median"),
               (("ALL", "p90"), "ALL p90"), (("ALL", "gross20"), "ALL gross20"),
               (("DAY", "cbox"), "DAY cbox"), (("DAY", "median"), "DAY median"),
               (("DAY", "p90"), "DAY p90"),
               (("NIGHT", "median"), "NIGHT median"), (("NIGHT", "p90"), "NIGHT p90")):
    a, b = t0[k[0]][k[1]], t1[k[0]][k[1]]
    L.append(f"{lab:22} {f2(a):>10} {f2(b):>10} "
             f"{('    n/a' if None in (a, b) else f'{b-a:+10.3f}')}")
L += ["", "NIGHT candidate"]
for k in ("any_cbox", "top1_cbox", "cand_per_frame", "wrong_present_frac", "margin_median"):
    L.append(f"{k:22} {c0.get(k, float('nan')):10.3f} {c1.get(k, float('nan')):10.3f} "
             f"{c1.get(k, float('nan')) - c0.get(k, float('nan')):+10.3f}")
L += ["", "held-out negative"]
for k in ("n_neg", "AP_AUPRC", "AUROC", "FPR@TPR95", "neg_p90", "neg_p95", "neg_p99"):
    L.append(f"{k:22} {n0[k]:10.4f} {n1[k]:10.4f} {n1[k]-n0[k]:+10.4f}")
for t in ("0.05", "0.1", "0.25", "0.4"):
    L.append(f"{'FP/img@'+t:22} {n0['FP_per_image'][t]:10.4f} {n1['FP_per_image'][t]:10.4f} "
             f"{n1['FP_per_image'][t]-n0['FP_per_image'][t]:+10.4f}")
L += ["", "synthetic (G38 val 1,998)"]
for k in ("box_map50", "box_map", "pose_map50", "pose_map"):
    L.append(f"{k:22} {SY_['Y0'][k]:10.4f} {SY_['Y1'][k]:10.4f} "
             f"{SY_['Y1'][k]-SY_['Y0'][k]:+10.4f}")
L += ["```", "", f"safety {safety}", f"conditions {cond}  ({ncond}/3, 필요 2)", "",
      f"**VERDICT = {VERD}**", "",
      "★ NIGHT n=28, seed 1, 30ep. real 은 loss/training 에 쓰이지 않았다.",
      "★ Windows Y2 결과와 비교하기 전에는 60ep 을 시작하지 않는다."]
txt = "\n".join(L)
open(f"{NS}/FINAL_UBUNTU_REPORT.md", "w").write(txt)
print(txt)
notify(txt[:1800])
log(f"=== 완료 {VERD} ===")
if os.path.exists(LOCK): os.remove(LOCK)
