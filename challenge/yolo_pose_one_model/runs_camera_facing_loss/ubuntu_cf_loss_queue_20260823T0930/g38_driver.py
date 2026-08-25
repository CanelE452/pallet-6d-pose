"""G38 GENERIC-ONLY — OLD Stage-A 에서 target 만 뺐을 때 어디에 위치하는가.

smoke(1ep) -> 60ep -> 동일 real 재평가 -> night candidate -> 판정 -> notify. 한 파일.
recipe 는 OLD Stage-A args.yaml 을 source of truth 로 복제 (data/name 만 다름).
"""
import collections, json, os, subprocess, sys, time
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
CFR = f"{Y}/runs_camera_facing_loss"
Q = f"{CFR}/ubuntu_cf_loss_queue_20260823T0930"
DATA = f"{Y}/datasets/g38_generic_only/data.yaml"
INIT = f"{ROOT}/challenge/weights/pretrained_yolo/yolo26n-pose.pt"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
ARM = "OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42"
SMOKE = "OLD_ROOT_G38_GENERIC_ONLY_SMOKE_SEED42"
LOG = f"{Q}/G38_LOG.txt"
LOCK = f"{Q}/G38.lock"
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
    notify(f"❌ G38: {m}")
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
            "[소비처] 논문 — OLD Stage-A 성공 원인 분해 (generic scale vs target 추가)\n"
            "[문장]  OLD Stage-A 에서 target 을 제거해도 real 성능이 유지되면 "
            "성공 원인은 generic scale 이다.\n")
    # ★ OLD Stage-A args 복제. patience/save_period 도 원본값(15/5) 그대로.
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
           "data": "{DATA}", "epochs": {epochs}, "seed": 42}},
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
        die(f"{name} criterion 이 standard 가 아님: {a['criterion']}")
    return d


# ---- PHASE 3 smoke ----------------------------------------------------------
log("PHASE 3 smoke 1ep (G38 generic-only)")
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
log("PHASE 3 PASS")

# ---- PHASE 4 full -----------------------------------------------------------
log("PHASE 4 G38 60ep seed42")
gd = train(ARM, 60)
nr = len(open(f"{gd}/results.csv").read().strip().split("\n")) - 1
log(f"PHASE 4 완료  results.csv {nr} epoch  (patience=15 이라 조기종료 가능 — 원본 recipe)")

# ---- PHASE 5 same real ------------------------------------------------------
log("PHASE 5 same-real 평가")


def real(tag, w):
    o = f"{Q}/REAL_{tag}.json"
    if not os.path.exists(o) and os.path.exists(w):
        subprocess.run([sys.executable, f"{Q}/cf_real_eval.py", "--weights", w, "--tag", tag],
                       capture_output=True, text=True)
    return json.load(open(o)) if os.path.exists(o) else None


MODELS = [("A42", f"{CFR}/CF_A0_V1MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt", "A0",
           "generic 10K"),
          ("G38", f"{gd}/weights/last.pt", "G38", "generic 38K"),
          ("OLD", f"{Y}/runs/stage_a_synth_640_b32_seed42/weights/best.pt", "OLD_STAGE_A",
           "generic38K + target17.9K x2"),
          ("C43", f"{CFR}/CF_DATA_C_V2_EARLY10K_STD_60EP_SEED43_UBUNTU/weights/last.pt",
           "DATA_C43", "V2 10K"),
          ("FT", "/home/minjae/Documents/github/25y_automatic_lifter-master/pallet_yolo26n_pose_ft.pt",
           "FT_REFERENCE", "OLD + real FT")]
R = {}
for nm, w, tag, comp in MODELS:
    d = real(tag, w)
    if d:
        R[nm] = (d, comp)


def agg(rs):
    cb = [r for r in rs if r.get("correct_box")]
    e = np.concatenate([r["err"] for r in cb]) if cb else np.array([])
    return {"n": len(rs), "cbox": len(cb) / max(len(rs), 1),
            "med": float(np.median(e)) if e.size else None,
            "p90": float(np.percentile(e, 90)) if e.size else None,
            "gross": float((e > 20).mean()) if e.size else None}


TAB = {}
for nm, (d, comp) in R.items():
    pf = [r for r in d["per_frame"] if r["frame"] not in LEAK]
    TAB[nm] = {"composition": comp,
               "ALL": agg(pf),
               "DAY": agg([r for r in pf if r["domain"] == "DAY"]),
               "NIGHT": agg([r for r in pf if r["domain"] == "NIGHT"])}

# ---- night candidate for G38 -------------------------------------------------
log("PHASE 5 night candidate (G38)")
r = subprocess.run([sys.executable, f"{Q}/night_cand_one.py", "--weights", f"{gd}/weights/last.pt",
                    "--tag", "G38"], capture_output=True, text=True)
NC = (json.load(open(f"{Q}/NIGHT_CAND_G38.json"))
      if os.path.exists(f"{Q}/NIGHT_CAND_G38.json") else None)

# ---- PHASE 6 판정 --------------------------------------------------------------
a, g, o = TAB.get("A42", {}).get("ALL", {}), TAB.get("G38", {}).get("ALL", {}), TAB.get("OLD", {}).get("ALL", {})
frac = None
if all(x and x.get("med") for x in (a, g, o)) and abs(a["med"] - o["med"]) > 1e-9:
    frac = (a["med"] - g["med"]) / (a["med"] - o["med"])     # A42→OLD 구간에서 G38 위치
verdict = {"generic_scale_recovery_fraction_median": frac,
           "GENERIC_SCALE_EFFECT": None, "TARGET_ADDITION_REQUIRED": None}
if frac is not None:
    verdict["GENERIC_SCALE_EFFECT"] = "STRONG" if frac >= 0.5 else ("PARTIAL" if frac >= 0.2 else "WEAK")
    verdict["TARGET_ADDITION_REQUIRED"] = bool(frac < 0.8)
out = {"table": TAB, "night_candidate_G38": NC, "verdict": verdict,
       "real_membership": {"n": 128, "leak_excluded": sorted(LEAK)},
       "★caveat": ("60ep 고정이므로 G38 은 OLD 대비 update 수도 함께 줄었다. "
                   "target 제거의 순수 인과효과로 과장하지 않는다. "
                   "'generic-only full 60ep recipe' 의 결과다."),
       "recipe_source": "runs/stage_a_synth_640_b32_seed42/args.yaml (patience 15, save_period 5)"}
json.dump(out, open(f"{Q}/G38_ROOT_STEP1.json", "w"), indent=2, ensure_ascii=False)

L = ["# G38 GENERIC-ONLY — ROOT CAUSE STEP 1", "", "```",
     f"{'model':6} {'composition':28} {'scope':6} {'n':>4} {'cbox':>6} {'med':>7} {'p90':>7} {'gross':>6}",
     "-" * 76]
for nm in ("A42", "G38", "OLD", "C43", "FT"):
    if nm not in TAB:
        continue
    for sc in ("ALL", "DAY", "NIGHT"):
        v = TAB[nm][sc]
        f_ = lambda x: "    n/a" if x is None else f"{x:7.2f}"
        g_ = lambda x: "   n/a" if x is None else f"{x:6.3f}"
        L.append(f"{nm:6} {TAB[nm]['composition'][:28]:28} {sc:6} {v['n']:4d} "
                 f"{g_(v['cbox'])} {f_(v['med'])} {f_(v['p90'])} {g_(v['gross'])}")
    L.append("")
L += ["```", ""]
if NC:
    L += ["## G38 NIGHT candidate", "```",
          f"any-cbox {NC['any_cbox']:.3f}  top1-cbox {NC['top1_cbox']:.3f}  "
          f"cand/frame {NC['cand_per_frame']:.2f}  wrong 존재 {100*NC['wrong_present_frac']:.0f}%  "
          f"margin med {NC['margin_median']:+.4f}", "```", ""]
L += [f"**GENERIC_SCALE_EFFECT = {verdict['GENERIC_SCALE_EFFECT']}**",
      f"**TARGET_ADDITION_REQUIRED = {verdict['TARGET_ADDITION_REQUIRED']}**",
      f"median 회수율 (A42→OLD 구간에서 G38 위치) = "
      f"{'n/a' if frac is None else f'{100*frac:.1f}%'}", "",
      "★ 60ep 고정 = update 수도 함께 감소. target 제거의 순수 인과효과로 읽지 않는다."]
txt = "\n".join(L)
open(f"{Q}/G38_ROOT_STEP1.md", "w").write(txt)
print(txt)
notify(txt[:1800])
log("=== G38 완료 ===")
if os.path.exists(LOCK):
    os.remove(LOCK)
