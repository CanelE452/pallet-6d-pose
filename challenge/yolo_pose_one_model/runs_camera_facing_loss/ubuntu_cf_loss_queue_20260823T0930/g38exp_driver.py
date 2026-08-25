"""G38_EXP73916 — exposure-matched generic-only control.

OLD 와 total exposure(73,916)·epochs(60)·recipe 를 맞추고 target content 만 0 으로 둔다.
smoke -> 60ep -> step parity -> same-real -> night -> recovery -> notify. 한 파일.
"""
import collections, json, os, re, subprocess, sys, time
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
CFR = f"{Y}/runs_camera_facing_loss"
Q = f"{CFR}/ubuntu_cf_loss_queue_20260823T0930"
DATA = f"{Y}/datasets/g38_exp73916/data.yaml"
INIT = f"{ROOT}/challenge/weights/pretrained_yolo/yolo26n-pose.pt"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
ARM = "OLD_ROOT_G38_EXP73916_60EP_SEED42"
SMOKE = "OLD_ROOT_G38_EXP73916_SMOKE_SEED42"
LOG = f"{Q}/G38EXP_LOG.txt"
LOCK = f"{Q}/G38EXP.lock"
LEAK = set(json.load(open(f"{Q}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])


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
    notify(f"❌ G38EXP: {m}")
    if os.path.exists(LOCK):
        os.remove(LOCK)
    sys.exit(1)


if os.path.exists(LOCK):
    L = json.load(open(LOCK))
    if os.path.exists(f"/proc/{L['pid']}"):
        log("이미 실행 중")
        sys.exit(0)
json.dump({"pid": os.getpid(), "start": time.strftime("%F %T")}, open(LOCK, "w"))


def train(name, epochs):
    d = f"{CFR}/{name}"
    if os.path.exists(f"{d}/weights/last.pt"):
        log(f"{name} 이미 있음 — 건너뜀")
        return d
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/PURPOSE.md"):
        open(f"{d}/PURPOSE.md", "w").write(
            "[소비처] 논문 — OLD Stage-A NIGHT ranking 우위의 원인 분해\n"
            "[문장]  target content 없이 exposure 만 OLD 와 맞추면 NIGHT ranking 이 회복되는가.\n")
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
           "data": "{DATA}", "epochs": {epochs}, "seed": 42,
           "n_train_batches": len(tr.train_loader), "batch": 32}},
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
        die(f"{name} 산출물 없음 (rc={r.returncode}, 로그 {lf}):\n{tail[-900:]}")
    a = json.load(open(f"{d}/RUNTIME_AUDIT.json"))
    if a["PC_CALL_COUNT"] or a["ROLE_CALLS"]:
        die(f"{name} 커스텀 loss 오염")
    if "PoseLoss" not in a["criterion"]:
        die(f"{name} criterion standard 아님 {a['criterion']}")
    return d


MAN = json.load(open(f"{Q}/G38_EXP73916_MANIFEST.json"))
log(f"PHASE 5 smoke 1ep  (exposure {MAN['exposure']} / unique {MAN['unique']} / target 0)")
sd = train(SMOKE, 1)
rc = open(f"{sd}/results.csv").read().strip().split("\n")
vals = {h.strip(): v for h, v in zip(rc[0].split(","), rc[1].split(","))}
lo = [float(v) for k, v in vals.items() if "loss" in k and v not in ("", "nan")]
if not lo or not all(np.isfinite(lo)):
    die(f"smoke loss 비정상 {lo}")
ar = open(f"{sd}/args.yaml").read()
for need in ("batch: 32", "fliplr: 0.0", "seed: 42"):
    if need not in ar:
        die(f"smoke args 위반 {need}")
sa = json.load(open(f"{sd}/RUNTIME_AUDIT.json"))
log(f"PHASE 5 PASS  batches/epoch {sa.get('n_train_batches')}")

log("PHASE 6 G38EXP 60ep seed42")
ed = train(ARM, 60)
nep = len(open(f"{ed}/results.csv").read().strip().split("\n")) - 1
ea = json.load(open(f"{ed}/RUNTIME_AUDIT.json"))
log(f"PHASE 6 완료  {nep} epoch  batches/epoch {ea.get('n_train_batches')}")


# ---- step parity -------------------------------------------------------------
def steps_of(run, n_train):
    p = f"{run}/results.csv"
    ep = len(open(p).read().strip().split("\n")) - 1 if os.path.exists(p) else None
    bpe = int(np.ceil(n_train / 32))
    return {"epochs": ep, "n_train_exposure": n_train, "batches_per_epoch": bpe,
            "total_steps": (bpe * ep) if ep else None}


PAR = {"OLD": steps_of(f"{Y}/runs/stage_a_synth_640_b32_seed42", 73916),
       "G38": steps_of(f"{CFR}/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42", 38002),
       "G38EXP": steps_of(ed, 73916)}
PAR["parity_OLD_vs_G38EXP"] = (PAR["OLD"]["batches_per_epoch"] == PAR["G38EXP"]["batches_per_epoch"]
                               and PAR["OLD"]["total_steps"] == PAR["G38EXP"]["total_steps"])
PAR["reported_batches_per_epoch_from_trainer"] = ea.get("n_train_batches")
json.dump(PAR, open(f"{Q}/OLD_vs_G38EXP_STEP_PARITY.json", "w"), indent=2, ensure_ascii=False)
log(f"step parity {PAR['parity_OLD_vs_G38EXP']}  OLD {PAR['OLD']['total_steps']} / "
    f"G38EXP {PAR['G38EXP']['total_steps']} / G38 {PAR['G38']['total_steps']}")


# ---- PHASE 7 same real --------------------------------------------------------
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


MODELS = [("A42", f"{CFR}/CF_A0_V1MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt", "A0"),
          ("G38", f"{CFR}/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt", "G38"),
          ("G38EXP", f"{ed}/weights/last.pt", "G38EXP"),
          ("OLD", f"{Y}/runs/stage_a_synth_640_b32_seed42/weights/best.pt", "OLD_STAGE_A"),
          ("C43", f"{CFR}/CF_DATA_C_V2_EARLY10K_STD_60EP_SEED43_UBUNTU/weights/last.pt", "DATA_C43"),
          ("FT", "/home/minjae/Documents/github/25y_automatic_lifter-master/pallet_yolo26n_pose_ft.pt",
           "FT_REFERENCE")]
log("PHASE 7 same-real")
R, NC = {}, {}
for nm, w, tag in MODELS:
    d = real(tag, w)
    if d:
        R[nm] = d
for nm in ("G38", "G38EXP", "OLD"):
    w = dict((a, b) for a, b, _ in MODELS)[nm]
    NC[nm] = nightc(nm if nm != "OLD" else "OLD_S", w)
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
    pf = [r for r in d["per_frame"] if r["frame"] not in LEAK]
    TAB[nm] = {"ALL": agg(pf), "DAY": agg([r for r in pf if r["domain"] == "DAY"]),
               "NIGHT": agg([r for r in pf if r["domain"] == "NIGHT"])}

# ---- PHASE 8 recovery ----------------------------------------------------------
g, x, o = NC.get("G38"), NC.get("G38EXP"), NC.get("OLD")


def rec(a, b, c):
    return None if (a is None or b is None or c is None or abs(c - a) < 1e-12) else (b - a) / (c - a)


REC = {}
if g and x and o:
    REC = {"R_cbox": rec(g["top1_cbox"], x["top1_cbox"], o["top1_cbox"]),
           "R_margin": rec(g["margin_median"], x["margin_median"], o["margin_median"]),
           "R_candidate": rec(g["cand_per_frame"], x["cand_per_frame"], o["cand_per_frame"])}
vals = [v for v in REC.values() if v is not None]
hi = sum(1 for v in vals if v >= 0.70)
loo = sum(1 for v in vals if v <= 0.30)
VER = ("EXTRA_EXPOSURE_DOMINANT" if hi >= 2 else
       "TARGET_CONTENT_REQUIRED" if loo >= 2 else "MIXED")

# ---- guards + bootstrap --------------------------------------------------------
GUARD = {}
if "G38" in TAB and "G38EXP" in TAB:
    GUARD = {"any_cbox_drop": (NC["G38EXP"]["any_cbox"] - NC["G38"]["any_cbox"]) if (g and x) else None,
             "DAY_cbox_delta": TAB["G38EXP"]["DAY"]["cbox"] - TAB["G38"]["DAY"]["cbox"],
             "DAY_med_delta": ((TAB["G38EXP"]["DAY"]["med"] or 0) - (TAB["G38"]["DAY"]["med"] or 0))}
BOOT = None
if R.get("G38") and R.get("G38EXP"):
    a = {r["frame"]: r for r in R["G38"]["per_frame"] if r["domain"] == "NIGHT" and r.get("correct_box")}
    b = {r["frame"]: r for r in R["G38EXP"]["per_frame"] if r["domain"] == "NIGHT" and r.get("correct_box")}
    k = sorted(set(a) & set(b))
    if len(k) >= 5:
        d = np.array([np.median(a[f]["err"]) - np.median(b[f]["err"]) for f in k])
        rng = np.random.default_rng(0)
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        BOOT = {"n_paired_night": len(k), "delta_median_G38_minus_G38EXP": float(d.mean()),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

out = {"manifest": MAN, "step_parity": PAR, "table": TAB, "night_candidate": NC,
       "recovery": REC, "verdict": VER, "guards": GUARD, "bootstrap_night": BOOT,
       "prereg_gate": "2/3 recovery >= 0.70 -> EXTRA_EXPOSURE_DOMINANT, <= 0.30 -> TARGET_CONTENT_REQUIRED",
       "★caveat": ("NIGHT n=28. bootstrap 유의성만으로 mechanism 결정하지 않는다. "
                   "alias 는 동일 RGB 반복이라 unique content 다양성은 늘지 않는다 — "
                   "'exposure/update 를 맞춘 통제' 이지 'generic 을 늘린 것' 이 아니다.")}
json.dump(out, open(f"{Q}/G38EXP_ROOT_STEP2.json", "w"), indent=2, ensure_ascii=False)

L = ["# G38_EXP73916 — ROOT CAUSE STEP 2 (exposure-matched generic-only)", "", "```",
     f"{'':10} {'unique':>8} {'exposure':>9} {'batch/ep':>9} {'total steps':>12} {'target':>7}",
     "-" * 60,
     f"{'G38':10} {38002:8d} {38002:9d} {PAR['G38']['batches_per_epoch']:9d} "
     f"{str(PAR['G38']['total_steps']):>12} {0:7d}",
     f"{'G38EXP':10} {38002:8d} {73916:9d} {PAR['G38EXP']['batches_per_epoch']:9d} "
     f"{str(PAR['G38EXP']['total_steps']):>12} {0:7d}",
     f"{'OLD':10} {55959:8d} {73916:9d} {PAR['OLD']['batches_per_epoch']:9d} "
     f"{str(PAR['OLD']['total_steps']):>12} {17957:7d}", "```", "",
     "## SAME REAL n=128", "```",
     f"{'model':8} {'scope':6} {'cbox':>6} {'med':>7} {'p90':>7} {'gross':>6}", "-" * 46]
for nm in ("A42", "G38", "G38EXP", "OLD", "C43", "FT"):
    if nm not in TAB:
        continue
    for sc in ("ALL", "DAY", "NIGHT"):
        v = TAB[nm][sc]
        f_ = lambda z: "    n/a" if z is None else f"{z:7.2f}"
        g_ = lambda z: "   n/a" if z is None else f"{z:6.3f}"
        L.append(f"{nm:8} {sc:6} {g_(v['cbox'])} {f_(v['med'])} {f_(v['p90'])} {g_(v['gross'])}")
    L.append("")
L += ["```", "", "## NIGHT candidate", "```",
      f"{'model':8} {'any-cbox':>9} {'top1':>7} {'cand/fr':>8} {'wrong%':>7} {'margin':>8}", "-" * 50]
for nm in ("G38", "G38EXP", "OLD"):
    v = NC.get(nm)
    if v:
        L.append(f"{nm:8} {v['any_cbox']:9.3f} {v['top1_cbox']:7.3f} {v['cand_per_frame']:8.2f} "
                 f"{100*v['wrong_present_frac']:6.0f}% {v['margin_median']:+8.4f}")
L += ["```", "", "## RECOVERY (G38 → OLD 구간에서 G38EXP 위치)", "```"]
for k, v in REC.items():
    L.append(f"{k:14} {'n/a' if v is None else f'{100*v:+7.1f}%'}")
L += ["```", "", f"**{VER}**", "",
      f"guards: {GUARD}",
      f"bootstrap(night paired): {BOOT}", "",
      "★ alias 는 동일 RGB 반복 — unique 다양성은 그대로다. exposure/update 통제이지",
      "  'generic 을 늘린 것' 이 아니다. NIGHT n=28."]
txt = "\n".join(L)
open(f"{Q}/G38EXP_ROOT_STEP2.md", "w").write(txt)
print(txt)
notify(txt[:1800])
log(f"=== G38EXP 완료  {VER} ===")
if os.path.exists(LOCK):
    os.remove(LOCK)
