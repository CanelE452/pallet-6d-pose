"""G38_PLUS_SUPPORT — target-free BEST-PERFORMANCE CANDIDATE (causal ablation 아님).

smoke -> 60ep -> args diff -> same-real -> gap closure -> gate -> notify. 한 파일.
recipe 는 G38/OLD args 를 정본으로 복제 (patience 15, save_period 5).
"""
import json, os, subprocess, sys, time
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
CFR = f"{Y}/runs_camera_facing_loss"
Q = f"{CFR}/ubuntu_cf_loss_queue_20260823T0930"
DATA = f"{Y}/datasets/g38_plus_support/data.yaml"
INIT = f"{ROOT}/challenge/weights/pretrained_yolo/yolo26n-pose.pt"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
ARM = "G38_PLUS_SUPPORT_60EP_SEED42"
SMOKE = "G38_PLUS_SUPPORT_SMOKE_SEED42"
LOG = f"{Q}/COMBO_LOG.txt"
LOCK = f"{Q}/COMBO.lock"
LEAK = set(json.load(open(f"{Q}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
GATE = json.load(open(f"{Q}/G38_PLUS_SUPPORT_GATE_PREREG.json"))


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def notify(m):
    try:
        subprocess.run([NOTIFY, m], timeout=60)
    except Exception as e:
        log(f"discord 실패(무시): {e}")


def die(m):
    log("FAIL " + m)
    notify(f"❌ G38+SUPPORT: {m}")
    if os.path.exists(LOCK):
        os.remove(LOCK)
    sys.exit(1)


if os.path.exists(LOCK):
    L = json.load(open(LOCK))
    if os.path.exists(f"/proc/{L['pid']}"):
        log("이미 실행 중")
        sys.exit(0)
json.dump({"pid": os.getpid(), "start": time.strftime("%F %T")}, open(LOCK, "w"))


def done(d, ep):
    rp = f"{d}/results.csv"
    return (os.path.exists(f"{d}/weights/last.pt") and os.path.exists(rp)
            and len(open(rp).read().strip().split("\n")) - 1 >= ep)


def train(name, epochs):
    d = f"{CFR}/{name}"
    if done(d, epochs):
        log(f"{name} 이미 완료")
        return d
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/PURPOSE.md"):
        open(f"{d}/PURPOSE.md", "w").write(
            "[소비처] 논문 — target-free 최종 성능 후보 (adaptation base 선정)\n"
            "[문장]  G38 broad scale 에 target-free thin/mid-elevation support 를 더하면 "
            "OLD synthetic-only 에 더 근접한다.\n")
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
from ultralytics.models.yolo.pose import PoseTrainer
tr = PoseTrainer(overrides=dict(
    task="pose", mode="train", model="{INIT}", data="{DATA}",
    epochs={epochs}, batch=32, imgsz=640, optimizer="SGD", lr0=0.01, lrf=0.01,
    cos_lr=True, close_mosaic=10, warmup_epochs=3.0, patience=15,
    single_cls=True, mosaic=0.3, scale=0.25, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
    fliplr=0.0, flipud=0.0, erasing=0.4, seed=42, deterministic=True,
    save_period=5, device=0, workers=8, project="{CFR}", name="{name}",
    exist_ok=True, resume=False, val=True, plots=False))
tr.train()
c = getattr(tr.model, "criterion", None)
json.dump({{"PC_CALL_COUNT": PC["n"], "ROLE_CALLS": SY.ROLE_CALLS["n"],
           "criterion": type(getattr(c, "one2many", c)).__name__,
           "n_train_batches": len(tr.train_loader), "epochs": {epochs}, "seed": 42}},
          open("{CFR}/{name}/RUNTIME_AUDIT.json", "w"), indent=2)
'''
    sc = f"{Q}/_tr_{name}.py"
    open(sc, "w").write(code)
    lf = f"{Q}/_train_{name}.log"
    with open(lf, "w") as fh:
        r = subprocess.run([sys.executable, "-u", sc], cwd=d, stdout=fh,
                           stderr=subprocess.STDOUT, text=True)
    if not os.path.exists(f"{d}/weights/last.pt"):
        tail = "\n".join(open(lf).read().replace("\r", "\n").split("\n")[-20:])
        die(f"{name} 산출물 없음 (rc={r.returncode}):\n{tail[-900:]}")
    a = json.load(open(f"{d}/RUNTIME_AUDIT.json"))
    if a["PC_CALL_COUNT"] or a["ROLE_CALLS"]:
        die(f"{name} 커스텀 loss 오염")
    if "PoseLoss" not in a["criterion"]:
        die(f"{name} criterion standard 아님 {a['criterion']}")
    return d


log("PHASE 9 smoke 1ep")
sd = train(SMOKE, 1)
rc = open(f"{sd}/results.csv").read().strip().split("\n")
vals = {h.strip(): v for h, v in zip(rc[0].split(","), rc[1].split(","))}
lo = [float(v) for k, v in vals.items() if "loss" in k and v not in ("", "nan")]
if not lo or not all(np.isfinite(lo)):
    die(f"smoke loss 비정상 {lo}")
sa = json.load(open(f"{sd}/RUNTIME_AUDIT.json"))
log(f"PHASE 9 PASS  batches/ep {sa['n_train_batches']}  PC 0 role 0 {sa['criterion']}")

log("PHASE 10 60ep")
cd_ = train(ARM, 60)
nep = len(open(f"{cd_}/results.csv").read().strip().split("\n")) - 1
ca = json.load(open(f"{cd_}/RUNTIME_AUDIT.json"))
log(f"PHASE 10 완료  {nep} epoch  batches/ep {ca['n_train_batches']}")

import yaml
ga = yaml.safe_load(open(f"{CFR}/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/args.yaml"))
na = yaml.safe_load(open(f"{cd_}/args.yaml"))
ALLOW = {"data", "name", "save_dir", "project"}
diff = {k: [ga.get(k), na.get(k)] for k in set(ga) | set(na) if ga.get(k) != na.get(k)}
bad = {k: v for k, v in diff.items() if k not in ALLOW}
json.dump({"diff": diff, "allowed": sorted(ALLOW), "violations": bad},
          open(f"{Q}/G38_vs_G38SUPPORT_ARGS_DIFF.json", "w"), indent=2, ensure_ascii=False)
if bad:
    die(f"args 허용 외 차이 {list(bad)}")
log(f"args diff {sorted(diff)} (전부 허용)")


def real(tag, w):
    o = f"{Q}/REAL_{tag}.json"
    if not os.path.exists(o) and os.path.exists(w):
        subprocess.run([sys.executable, f"{Q}/cf_real_eval.py", "--weights", w, "--tag", tag],
                       capture_output=True, text=True)
    return json.load(open(o)) if os.path.exists(o) else None


def nightc(tag, w):
    o = f"{Q}/NIGHT_CAND_{tag}.json"
    if not os.path.exists(o) and os.path.exists(w):
        subprocess.run([sys.executable, f"{Q}/night_cand_one.py", "--weights", w, "--tag", tag],
                       capture_output=True, text=True)
    return json.load(open(o)) if os.path.exists(o) else None


# G38 이 어떤 checkpoint 로 평가됐는지 artifact 에서 확인
g38w = json.load(open(f"{Q}/REAL_G38.json"))["weights"] if os.path.exists(f"{Q}/REAL_G38.json") else ""
POLICY = "last.pt" if g38w.endswith("last.pt") else ("best.pt" if g38w.endswith("best.pt") else "last.pt")
log(f"checkpoint policy = {POLICY} (G38 artifact 에서 확인: {os.path.basename(g38w)})")

MODELS = [("A42", f"{CFR}/CF_A0_V1MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt", "A0"),
          ("C43", f"{CFR}/CF_DATA_C_V2_EARLY10K_STD_60EP_SEED43_UBUNTU/weights/last.pt", "DATA_C43"),
          ("G38", f"{CFR}/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/{POLICY}", "G38"),
          ("COMBO", f"{cd_}/weights/{POLICY}", "G38SUP"),
          ("OLD", f"{Y}/runs/stage_a_synth_640_b32_seed42/weights/best.pt", "OLD_STAGE_A"),
          ("FT", "/home/minjae/Documents/github/25y_automatic_lifter-master/pallet_yolo26n_pose_ft.pt",
           "FT_REFERENCE")]
log("PHASE 12 same-real")
R, NC = {}, {}
for nm, w, tag in MODELS:
    R[nm] = real(tag, w)
for nm in ("G38", "COMBO", "OLD", "FT"):
    w = dict((a, b) for a, b, _ in MODELS)[nm]
    t = dict((a, c) for a, _, c in MODELS)[nm]
    NC[nm] = nightc(t if nm != "OLD" else "OLD_S", w)
if NC.get("OLD") is None and os.path.exists(f"{Q}/OLD_STAGE_A_NIGHT_CANDIDATES.json"):
    NC["OLD"] = json.load(open(f"{Q}/OLD_STAGE_A_NIGHT_CANDIDATES.json"))


def agg(rs):
    cb = [r for r in rs if r.get("correct_box")]
    e = np.concatenate([r["err"] for r in cb]) if cb else np.array([])
    return {"n": len(rs), "cbox": len(cb) / max(len(rs), 1),
            "med": float(np.median(e)) if e.size else None,
            "p90": float(np.percentile(e, 90)) if e.size else None,
            "gross": float((e > 20).mean()) if e.size else None}


TAB = {}
for nm, d in R.items():
    if not d:
        continue
    pf = [r for r in d["per_frame"] if r["frame"] not in LEAK]
    TAB[nm] = {"ALL": agg(pf), "DAY": agg([r for r in pf if r["domain"] == "DAY"]),
               "NIGHT": agg([r for r in pf if r["domain"] == "NIGHT"])}

g, c, o = TAB.get("G38"), TAB.get("COMBO"), TAB.get("OLD")
ng, nc_, no = NC.get("G38"), NC.get("COMBO"), NC.get("OLD")


def rc_(a, b, ref):
    return None if None in (a, b, ref) or abs(ref - a) < 1e-12 else (b - a) / (ref - a)


GC = {}
if g and c and o:
    GC = {"R_cbox": rc_(g["ALL"]["cbox"], c["ALL"]["cbox"], o["ALL"]["cbox"]),
          "R_med": rc_(g["ALL"]["med"], c["ALL"]["med"], o["ALL"]["med"]),
          "R_p90": rc_(g["ALL"]["p90"], c["ALL"]["p90"], o["ALL"]["p90"]),
          "R_night_cbox": rc_(g["NIGHT"]["cbox"], c["NIGHT"]["cbox"], o["NIGHT"]["cbox"]),
          "R_night_p90": rc_(g["NIGHT"]["p90"], c["NIGHT"]["p90"], o["NIGHT"]["p90"])}
if ng and nc_ and no:
    GC["R_margin"] = rc_(ng["margin_median"], nc_["margin_median"], no["margin_median"])

hits, guards = {}, {}
if g and c:
    rel = lambda x, y: (x - y) / max(abs(x), 1e-12)
    H = GATE["hits"]
    N = ng["n"] if ng else 28
    hits = {"all_cbox_+2pp": (c["ALL"]["cbox"] - g["ALL"]["cbox"]) >= H["all_cbox_pp"],
            "all_median_-8%": rel(g["ALL"]["med"], c["ALL"]["med"]) >= H["all_median_rel"],
            "all_p90_-10%": rel(g["ALL"]["p90"], c["ALL"]["p90"]) >= H["all_p90_rel"],
            "night_top1_+3f": ((nc_["top1_cbox"] - ng["top1_cbox"]) * N >= H["night_top1_cbox_frames"]
                               if (ng and nc_) else False),
            "night_p90_-15%": rel(g["NIGHT"]["p90"], c["NIGHT"]["p90"]) >= H["night_p90_rel"],
            "night_margin_+0.10": ((nc_["margin_median"] - ng["margin_median"]) >= H["night_margin_abs"]
                                   if (ng and nc_) else False)}
    G_ = GATE["guards"]
    guards = {"all_cbox": (c["ALL"]["cbox"] - g["ALL"]["cbox"]) >= G_["all_cbox_pp"],
              "night_any_cbox": ((nc_["any_cbox"] - ng["any_cbox"]) * N >= G_["night_any_cbox_frames"]
                                 if (ng and nc_) else True),
              "all_p90": rel(g["ALL"]["p90"], c["ALL"]["p90"]) >= -G_["all_p90_rel_worse"]}
nhit = sum(1 for v in hits.values() if v)
SIG = "POSITIVE" if (nhit >= GATE["hits_need"] and all(guards.values())) else \
      ("HARM" if guards and not all(guards.values()) else "NULL")
BEST = "G38_PLUS_SUPPORT" if SIG == "POSITIVE" else "G38"

out = {"gate": GATE, "hits": hits, "n_hits": nhit, "guards": guards,
       "TARGET_FREE_COMBO_SIGNAL": SIG, "CURRENT_TARGET_FREE_BEST": BEST,
       "RENDER_TRACK": "CLOSED", "RENDER_RESUME": False,
       "table": TAB, "night_candidate": NC, "gap_closure": GC,
       "checkpoint_policy": POLICY, "actual_epochs": nep,
       "scanner": json.load(open(f"{Q}/G38_PLUS_SUPPORT_SCANNER.json")),
       "★scope": ("CAUSAL ABLATION 아님. support 1,933 만큼 exposure/steps +5.1%. "
                  "TARGET_FREE_BEST_PERFORMANCE_CANDIDATE 로만 해석한다. "
                  "OLD/FT 는 REFERENCE/UPPER DIAGNOSTIC — 같은 training data 조건 아님.")}
json.dump(out, open(f"{Q}/G38_PLUS_SUPPORT_RESULT.json", "w"), indent=2, ensure_ascii=False)

L = ["# G38 + SUPPORT — target-free BEST-PERFORMANCE CANDIDATE", "", "```",
     f"{'model':14} {'cbox':>7} {'median':>8} {'p90':>8} {'gross20':>8}", "-" * 50]
for nm in ("A42", "C43", "G38", "COMBO", "OLD", "FT"):
    if nm not in TAB:
        continue
    v = TAB[nm]["ALL"]
    f_ = lambda z: "     n/a" if z is None else f"{z:8.2f}"
    L.append(f"{nm:14} {v['cbox']:7.3f} {f_(v['med'])} {f_(v['p90'])} "
             f"{(v['gross'] if v['gross'] is not None else float('nan')):8.3f}")
L += ["```", "", "## NIGHT", "```",
      f"{'model':8} {'any':>7} {'top1':>7} {'med':>8} {'p90':>8} {'cand/fr':>8} {'wrong%':>7} {'margin':>8}",
      "-" * 62]
for nm in ("G38", "COMBO", "OLD", "FT"):
    if nm not in TAB:
        continue
    t = TAB[nm]["NIGHT"]
    n = NC.get(nm) or {}
    f_ = lambda z: "     n/a" if z is None else f"{z:8.2f}"
    L.append(f"{nm:8} {n.get('any_cbox',float('nan')):7.3f} {n.get('top1_cbox',float('nan')):7.3f} "
             f"{f_(t['med'])} {f_(t['p90'])} {n.get('cand_per_frame',float('nan')):8.2f} "
             f"{n.get('wrong_present_frac',float('nan')):7.3f} {n.get('margin_median',float('nan')):+8.3f}")
L += ["```", "", "## GAP CLOSURE (G38 → OLD 구간에서 COMBO 위치)", "```"]
for k, v in GC.items():
    L.append(f"{k:16} {'n/a' if v is None else f'{100*v:+7.1f}%'}")
L += ["```", "", f"hits {nhit}/6 → {hits}", f"guards {guards}", "",
      f"**TARGET_FREE_COMBO_SIGNAL = {SIG}**",
      f"**CURRENT_TARGET_FREE_BEST = {BEST}**", "",
      f"RENDER_TRACK = CLOSED   RENDER_RESUME = FALSE",
      f"epochs {nep}/60 · batches/ep {ca['n_train_batches']} · checkpoint {POLICY}", "",
      "★ CAUSAL ABLATION 아님 — exposure/steps +5.1% 동반. OLD/FT 는 REFERENCE/UPPER",
      "  DIAGNOSTIC 이며 같은 training data 조건이 아니다. NIGHT n=28, seed 1."]
txt = "\n".join(L)
open(f"{Q}/G38_PLUS_SUPPORT_RESULT.md", "w").write(txt)
print(txt)
notify(txt[:1800])
log(f"=== 완료 {SIG} / BEST {BEST} ===")
if os.path.exists(LOCK):
    os.remove(LOCK)
