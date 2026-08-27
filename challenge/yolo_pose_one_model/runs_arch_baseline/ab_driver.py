"""ARCHITECTURE BASELINE — B11 / B8 30ep + Y0 재사용 -> 한 표.

smoke -> train -> synth(mAP + 9kp) -> real128 -> night -> negative -> efficiency -> table.
새 loss 0, 새 solver 0, 새 threshold 0. 기존 결과 overwrite 0.
"""
from __future__ import annotations
import json, os, subprocess, sys, time

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
NS = f"{Y}/runs_arch_baseline"
sys.path.insert(0, NS)
import ab_common as AC                                                # noqa: E402

QY = f"{Y}/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
PC = f"{Y}/runs_posecls_g38"
DATA = f"{Y}/datasets/g38_generic_only/data.yaml"
SMOKE = f"{Y}/datasets/g38_smoke256/data.yaml"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
LOG, LOCK = f"{NS}/DRIVER_LOG.txt", f"{NS}/DRIVER.lock"
LEAK = set(json.load(open(f"{QY}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
EPOCHS = 30
ORDER = ["B11", "B8"]                                   # 지시된 실행 순서


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); open(LOG, "a").write(line + "\n")


def notify(m):
    try: subprocess.run([NOTIFY, m], timeout=60)
    except Exception as e: log(f"discord 실패(무시): {e}")


def die(m):
    log("FAIL " + m); notify(f"❌ ARCH BASELINE: {m}")
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


def train(name, init, data, epochs):
    d = f"{NS}/{name}"
    if done(d, epochs):
        log(f"{name} 이미 완료"); return d
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/PURPOSE.md"):
        open(f"{d}/PURPOSE.md", "w").write(
            "[소비처] 논문 — architecture baseline 표\n"
            "[문장]  동일 G38 조건에서 이 backbone 의 sim-to-real 일반화를 측정한다.\n")
    code = f'''
import os, sys, json
sys.path.insert(0, "{ROOT}")
os.environ["A1_CONFIG"] = ""
os.environ["PSPC_CONFIG"] = ""
# 옛/신 custom loss 가 끼어들지 않는지 런타임에서 감시만 한다 (주입은 하지 않는다)
from pallet_yolo_loss.loss import PSPCPoseLoss26
import pallet_yolo_loss.symmetry as SY
from pallet_yolo_loss import posecls as PCL
CNT = {{"pspc": 0}}
_pc = PSPCPoseLoss26.projective_loss
def _spy(self, *a, **k):
    CNT["pspc"] += 1
    return _pc(self, *a, **k)
PSPCPoseLoss26.projective_loss = _spy
SY.ROLE_CALLS["n"] = 0
PCL.CALLS["align"] = 0
from ultralytics.models.yolo.pose import PoseTrainer
tr = PoseTrainer(overrides=dict(
    task="pose", mode="train", model="{init}", data="{data}",
    epochs={epochs}, batch=32, imgsz=640, optimizer="SGD", lr0=0.01, lrf=0.01,
    cos_lr=True, close_mosaic=10, warmup_epochs=3.0, patience=0,
    single_cls=True, mosaic=0.3, scale=0.25, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
    fliplr=0.0, flipud=0.0, erasing=0.4, seed=42, deterministic=True,
    save_period=10, device=0, workers=8, project="{NS}", name="{name}",
    exist_ok=True, resume=False, val=True, plots=False))
tr.train()
c = getattr(tr.model, "criterion", None)
inner = getattr(c, "one2many", c)
m = tr.model.model[-1]
json.dump({{"PSPC_CALLS": CNT["pspc"], "ROLE_CALLS": SY.ROLE_CALLS["n"],
           "POSEALIGN_CALLS": PCL.CALLS["align"],
           "criterion": type(inner).__name__, "criterion_top": type(c).__name__,
           "head": type(m).__name__, "kpt_shape": list(getattr(m, "kpt_shape", [])),
           "end2end": bool(getattr(tr.model, "end2end", False)),
           "n_train_batches": len(tr.train_loader), "epochs": {epochs}, "seed": 42,
           "init": "{init}", "data": "{data}",
           "fliplr": tr.args.fliplr, "flipud": tr.args.flipud,
           "params": int(sum(p.numel() for p in tr.model.parameters()))}},
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
    if a["PSPC_CALLS"] or a["ROLE_CALLS"] or a["POSEALIGN_CALLS"]:
        die(f"{name} custom criterion 오염 {a['PSPC_CALLS']}/{a['ROLE_CALLS']}/{a['POSEALIGN_CALLS']}")
    if a["kpt_shape"] != [9, 3]: die(f"{name} kpt_shape {a['kpt_shape']}")
    if a["fliplr"] != 0.0 or a["flipud"] != 0.0: die(f"{name} flip {a['fliplr']}/{a['flipud']}")
    return d


# ------------------------------------------------------- PRETRAINED AUDIT
import ultralytics                                                    # noqa: E402
PA = {"ultralytics_version": ultralytics.__version__, "models": {}}
for k, v in AC.MODELS.items():
    p = v["init"]
    params, gflops = AC.model_stats(p)
    PA["models"][k] = {"label": v["label"], "path": os.path.relpath(p, ROOT),
                       "sha256": AC.sha256(p), "bytes": os.path.getsize(p),
                       "params": params, "GFLOPs": gflops}
json.dump(PA, open(f"{NS}/PRETRAINED_AUDIT.json", "w"), indent=2, ensure_ascii=False)
log("PRETRAINED AUDIT: " + ", ".join(
    f"{k} {PA['models'][k]['params']:,}p {PA['models'][k]['GFLOPs']}G" for k in PA["models"]))

# ------------------------------------------------------- SMOKE
SM = {}
for k in ORDER:
    d = train(f"SMOKE_{k}", AC.MODELS[k]["init"], SMOKE, 1)
    a = json.load(open(f"{d}/RUNTIME_AUDIT.json"))
    rc = open(f"{d}/results.csv").read().strip().split("\n")
    vals = {h.strip(): v for h, v in zip(rc[0].split(","), rc[1].split(","))}
    lo = [float(v) for kk, v in vals.items() if "loss" in kk and v not in ("", "nan")]
    if not lo or not all(np.isfinite(lo)): die(f"SMOKE_{k} loss {lo}")
    SM[k] = {"criterion": a["criterion"], "head": a["head"], "kpt_shape": a["kpt_shape"],
             "end2end": a["end2end"], "fliplr": a["fliplr"], "flipud": a["flipud"],
             "n_loss_terms": len(lo), "loss_finite": True,
             "custom_criterion_calls": {"pspc": a["PSPC_CALLS"], "role": a["ROLE_CALLS"],
                                        "posealign": a["POSEALIGN_CALLS"]},
             "batches": a["n_train_batches"]}
    log(f"  SMOKE {k} OK  {a['criterion']} / {a['head']} kpt {a['kpt_shape']} "
        f"end2end {a['end2end']} loss {len(lo)}항")
SM["dataset"] = json.load(open(f"{Y}/datasets/g38_smoke256/_build.json"))
SM["PASS"] = all(SM[k]["kpt_shape"] == [9, 3] and SM[k]["loss_finite"]
                 and SM[k]["fliplr"] == 0.0
                 and sum(SM[k]["custom_criterion_calls"].values()) == 0 for k in ORDER)
json.dump(SM, open(f"{NS}/SMOKE_RESULT.json", "w"), indent=2, ensure_ascii=False)
if not SM["PASS"]: die(f"SMOKE 계약 실패 {SM}")
log("SMOKE PASS")

# ------------------------------------------------------- FULL TRAIN
RUN = {}
for k in ORDER:
    log(f"{k} ({AC.MODELS[k]['label']}) {EPOCHS}ep 시작")
    d = train(AC.MODELS[k]["run"], AC.MODELS[k]["init"], DATA, EPOCHS)
    a = json.load(open(f"{d}/RUNTIME_AUDIT.json"))
    a["dir"] = d
    a["actual_epochs"] = len(open(f"{d}/results.csv").read().strip().split("\n")) - 1
    RUN[k] = a
    log(f"{k} 완료 {a['actual_epochs']}ep batches/ep {a['n_train_batches']} {a['criterion']}")
    AC.MODELS[k]["weights"] = f"{d}/weights/last.pt"
json.dump({"stage": "trained", "runs": RUN}, open(f"{NS}/_stage_trained.json", "w"),
          indent=2, ensure_ascii=False, default=str)
notify("ARCH BASELINE 학습 완료 (B11, B8) — 평가 시작")


# ------------------------------------------------------- EVAL
def qrun(script, prefix, tag, w):
    o = f"{QY}/{prefix}_{tag}.json"
    if not os.path.exists(o):
        r = subprocess.run([sys.executable, f"{QY}/{script}", "--weights", w, "--tag", tag],
                           capture_output=True, text=True)
        if not os.path.exists(o):
            die(f"{script} {tag} 실패: {(r.stderr or r.stdout)[-600:]}")
    return json.load(open(o))


def synth_map(tag, w):
    o = f"{NS}/SYNTH_{tag}.json"
    if os.path.exists(o):
        return json.load(open(o))
    from ultralytics import YOLO
    import torch
    m = YOLO(w, task="pose")
    r = m.val(data=DATA, imgsz=640, batch=32, device=0, workers=8, plots=False,
              project=f"{NS}/_val", name=tag, exist_ok=True, verbose=False)
    out = {"weights": os.path.relpath(w, ROOT),
           "box_map50": float(r.box.map50), "box_map": float(r.box.map),
           "pose_map50": float(r.pose.map50), "pose_map": float(r.pose.map),
           "note": "ultralytics val 정의 그대로 (Y0 와 동일)"}
    json.dump(out, open(o, "w"), indent=2, ensure_ascii=False)
    del m
    torch.cuda.empty_cache()
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
            "neg_p90": float(np.percentile(neg, 90))}


TAGS = {"B8": "AB_B8", "B11": "AB_B11", "Y0": "PC_Y0"}
RES = {}
for k in ("B8", "B11", "Y0"):
    w = AC.MODELS[k].get("weights")
    if not w or not os.path.exists(w):
        die(f"{k} weights 없음: {w}")
    tag = TAGS[k]
    log(f"평가 {k} ({AC.MODELS[k]['label']})")
    if k == "Y0":
        sm = json.load(open(f"{PC}/Y0_SYNTH.json"))          # 기존 정본에서 읽는다
        sm = {"box_map50": sm["box_map50"], "box_map": sm["box_map"],
              "pose_map50": sm["pose_map50"], "pose_map": sm["pose_map"],
              "source": "runs_posecls_g38/Y0_SYNTH.json (기존 정본, 재계산 안 함)"}
    else:
        sm = synth_map(tag, w)
    s9 = AC.synth_9kp(w, tag)
    rl = qrun("cf_real_eval.py", "REAL", tag, w)
    nc = qrun("night_cand_one.py", "NIGHT_CAND", tag, w)
    ns_ = qrun("neg_eval_one.py", "NEGSCORE", tag, w)
    pf = [r for r in rl["per_frame"] if r["frame"] not in LEAK]
    real = {"ALL": agg(pf), "DAY": agg([r for r in pf if r["domain"] == "DAY"]),
            "NIGHT": agg([r for r in pf if r["domain"] == "NIGHT"]), "night_candidate": nc}
    eff = AC.latency(w)
    params, gflops = AC.model_stats(w)
    RES[k] = {"label": AC.MODELS[k]["label"], "weights": os.path.relpath(w, ROOT),
              "checkpoint_sha256": AC.sha256(w),
              "pretrained": PA["models"][k], "synth_map": sm, "synth_9kp": s9,
              "real128": real, "negative": negmetrics(ns_["rows"]),
              "efficiency": {**eff, "params": params, "GFLOPs": gflops}}
    json.dump(RES[k], open(f"{NS}/RESULT_{k}.json", "w"), indent=2, ensure_ascii=False)
    log(f"  {k} cbox {real['ALL']['cbox']:.3f} med {real['ALL']['median']:.2f} "
        f"NIGHT top1 {nc['top1_cbox']:.3f} FPS {eff['FPS_from_median']:.1f}")

CONTRACT = {
    "data": {"train": 38002, "val": 1998, "target_specific_positive": 0,
             "real_positive_training": 0, "convention": "camera_dynamic_0123_v4",
             "kpt_shape": [9, 3], "single_cls": True},
    "recipe": {"epochs": 30, "batch": 32, "imgsz": 640, "optimizer": "SGD", "lr0": 0.01,
               "lrf": 0.01, "cos_lr": True, "close_mosaic": 10, "warmup_epochs": 3.0,
               "patience": 0, "mosaic": 0.3, "scale": 0.25, "hsv": [0.015, 0.5, 0.35],
               "fliplr": 0.0, "flipud": 0.0, "erasing": 0.4, "seed": 42,
               "deterministic": True, "resume": False},
    "evaluator": {"real": "cf_real_eval.py (pad100 REFLECT_101, imgsz640, conf 0.001, "
                          "top-1 by box conf, correct_box IoU>=0.5)",
                  "night": "night_cand_one.py", "negative": "neg_eval_one.py (n=2,689)",
                  "synth_map": "ultralytics val", "synth_9kp": "real 과 동일 정의, 추가 pad 없음"},
    "no_new_solver_or_threshold": True,
    "6D_PnP": "SKIPPED — Y0 정본 real evaluator 는 2D keypoint 전용이라 동일 PnP evaluator 가 없다",
}
OUT = {"contract": CONTRACT, "pretrained_audit": PA, "results": RES,
       "★comparability": {
           "directly_comparable": ["cbox", "keypoint median/p90", "gross20",
                                   "NIGHT top1-cbox", "NIGHT any-cbox",
                                   "synthetic mAP", "params", "GFLOPs"],
           "NOT_directly_comparable": ["candidates/frame", "wrong-candidate presence"],
           "why": ("v8n/11n 은 Detect + NMS, 26n 은 end2end one2one — 후보 집합의 정의가 "
                   "구조적으로 다르다. 후보 수에 의존하는 지표는 같은 자가 아니다."),
           "caution": ("negative FP/image@0.40 은 박스 개수를 세므로 NMS 유무의 영향을 받는다. "
                       "절대값보다 순서·자릿수로 읽고, 프레임 단위 neg_detect_rate 를 함께 볼 것."),
           "latency": "predict() 왕복이라 v8/11 은 NMS 비용이 포함된다."},
       "VERDICT": "ARCHITECTURE_BASELINE_TABLE_COMPLETE",
       "★scope": ("NIGHT n=28, seed 1, 30ep, synthetic-only. 승자 선정 아님. "
                  "60ep·s/m 확대는 하지 않는다.")}
json.dump(OUT, open(f"{NS}/ARCHITECTURE_BASELINE_TABLE.json", "w"), indent=2, ensure_ascii=False)

ORD = ["B8", "B11", "Y0"]
L = ["# ARCHITECTURE BASELINE — G38 synthetic-only, 30ep, 동일 recipe/evaluator", "", "```",
     f"{'model':16} {'params':>10} {'GFLOPs':>7} {'synPose':>8} {'cbox':>6} {'9kp med':>8} "
     f"{'9kp p90':>8} {'NIGHTtop1':>10} {'FP/img@.4':>10} {'FPS':>6}", "-" * 96]
for k in ORD:
    r = RES[k]
    L.append(f"{r['label']:16} {r['efficiency']['params']:10,} "
             f"{(r['efficiency']['GFLOPs'] or 0):7.1f} {r['synth_map']['pose_map']:8.4f} "
             f"{r['real128']['ALL']['cbox']:6.3f} {r['real128']['ALL']['median']:8.2f} "
             f"{r['real128']['ALL']['p90']:8.2f} "
             f"{r['real128']['night_candidate']['top1_cbox']:10.3f} "
             f"{r['negative']['FP_per_image_040']:10.4f} "
             f"{r['efficiency']['FPS_from_median']:6.1f}")
L += ["```", "", "## synthetic (G38 val 1,998)", "```",
      f"{'model':16} {'boxmAP50':>9} {'boxmAP':>8} {'poseMAP50':>10} {'poseMAP':>8} "
      f"{'9kp med':>8} {'9kp p90':>8} {'cbox':>6}", "-" * 78]
for k in ORD:
    r = RES[k]; s = r["synth_map"]; n = r["synth_9kp"]
    L.append(f"{r['label']:16} {s['box_map50']:9.4f} {s['box_map']:8.4f} "
             f"{s['pose_map50']:10.4f} {s['pose_map']:8.4f} "
             f"{(n['kp_median'] or float('nan')):8.2f} {(n['kp_p90'] or float('nan')):8.2f} "
             f"{n['cbox']:6.3f}")
L += ["```", "", "## real n=128", "```",
      f"{'model':16} {'scope':6} {'cbox':>6} {'9kp med':>8} {'9kp p90':>8} {'gross20':>8}", "-" * 60]
for k in ORD:
    for sc in ("ALL", "DAY", "NIGHT"):
        v = RES[k]["real128"][sc]
        f = lambda x: "     n/a" if x is None else f"{x:8.2f}"
        L.append(f"{RES[k]['label']:16} {sc:6} {v['cbox']:6.3f} {f(v['median'])} "
                 f"{f(v['p90'])} {(v['gross20'] if v['gross20'] is not None else float('nan')):8.3f}")
L += ["```", "", "## NIGHT candidate  (★ cand/frame·wrong% 는 모델 간 직접 비교 금지)", "```",
      f"{'model':16} {'any':>7} {'top1':>7} {'cand/f':>8} {'wrong%':>8} {'margin':>8}", "-" * 60]
for k in ORD:
    n = RES[k]["real128"]["night_candidate"]
    L.append(f"{RES[k]['label']:16} {n['any_cbox']:7.3f} {n['top1_cbox']:7.3f} "
             f"{n['cand_per_frame']:8.2f} {n['wrong_present_frac']:8.3f} {n['margin_median']:+8.3f}")
L += ["```", "", "## real negative n=2,689", "```",
      f"{'model':16} {'AP':>8} {'AUROC':>8} {'FPR@95':>8} {'FP/img@.4':>10} {'detrate@.4':>11} "
      f"{'neg p90':>8}", "-" * 78]
for k in ORD:
    n = RES[k]["negative"]
    L.append(f"{RES[k]['label']:16} {n['AP_AUPRC']:8.4f} {n['AUROC']:8.4f} "
             f"{n['FPR@TPR95']:8.4f} {n['FP_per_image_040']:10.4f} "
             f"{n['neg_detect_rate_040']:11.4f} {n['neg_p90']:8.4f}")
L += ["```", "", "## efficiency (RTX 3080, batch1, imgsz640, warmup 30 / run 200)", "```",
      f"{'model':16} {'lat med ms':>11} {'lat p90 ms':>11} {'FPS':>7} {'params':>10} {'GFLOPs':>7}",
      "-" * 68]
for k in ORD:
    e = RES[k]["efficiency"]
    L.append(f"{RES[k]['label']:16} {e['latency_ms_median']:11.2f} {e['latency_ms_p90']:11.2f} "
             f"{e['FPS_from_median']:7.1f} {e['params']:10,} {(e['GFLOPs'] or 0):7.1f}")
L += ["```", "", "## checkpoint SHA256", "```"]
for k in ORD:
    L.append(f"{RES[k]['label']:16} {RES[k]['checkpoint_sha256']}")
    L.append(f"{'  pretrained':16} {RES[k]['pretrained']['sha256']}")
L += ["```", "", "**VERDICT = ARCHITECTURE_BASELINE_TABLE_COMPLETE**", "",
      "★ 승자 선정 아님. 60ep·s/m 확대 없음.",
      "★ cand/frame·wrong% 는 v8/11(NMS) 과 26(end2end one2one) 의 후보 정의가 달라 직접 비교 금지.",
      "★ negative FP/image 는 박스 개수 기반이라 NMS 영향을 받는다 — 프레임 단위 detrate 를 같이 볼 것.",
      "★ 6D/PnP 는 Y0 정본 evaluator 가 2D 전용이라 제외(새 solver 금지).",
      "★ NIGHT n=28, seed 1."]
txt = "\n".join(L)
open(f"{NS}/ARCHITECTURE_BASELINE_TABLE.md", "w").write(txt)
print(txt)
notify(txt[:1800])
log("=== ARCHITECTURE_BASELINE_TABLE_COMPLETE ===")
if os.path.exists(LOCK): os.remove(LOCK)
