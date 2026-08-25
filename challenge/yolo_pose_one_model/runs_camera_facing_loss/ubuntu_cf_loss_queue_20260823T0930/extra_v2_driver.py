"""EXTRA_V2 — V2-10K(C) vs V2-12.5K(E), 같은 recipe·같은 val, train 만 +2,500.

U5 smoke -> U6 E 60ep -> U7 공통 evaluator -> U8 gate -> U9 winner seed43
-> U10 replication.  CPU 병렬로 night decomposition.
완료 판정은 results.csv 60행 + last.pt + VERDICT json 으로만.
"""
import json, os, subprocess, sys, time
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
CFR = f"{Y}/runs_camera_facing_loss"
Q = f"{CFR}/ubuntu_cf_loss_queue_20260823T0930"
V1 = f"{Y}/datasets/v1_cf_matched10k/data.yaml"
V2C = f"{Y}/datasets/v2_cf_early10k/data.yaml"
V2E = f"{Y}/datasets/v2_cf_complete/data.yaml"
INIT = f"{ROOT}/challenge/weights/pretrained_yolo/yolo26n-pose.pt"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
A_ARM = "CF_A0_V1MATCHED10K_60EP_SEED42_UBUNTU"
C_ARM = "CF_DATA_C_V2_EARLY10K_STD_60EP_SEED42_UBUNTU"
E_ARM = "CF_DATA_E_V2_COMPLETE12K5_STD_60EP_SEED42_UBUNTU"
SMOKE = "CF_DATA_E_V2_COMPLETE12K5_STD_SMOKE_SEED42"
LOG = f"{Q}/EXTRA_V2_LOG.txt"
LOCK = f"{Q}/EXTRA_V2.lock"


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def notify(m):
    try:
        subprocess.run([NOTIFY, m], timeout=60)
    except Exception as e:
        log(f"discord 실패(무시): {e}")


def die(m):
    log(f"FAIL {m}")
    notify(f"❌ EXTRA_V2: {m}")
    if os.path.exists(LOCK):
        os.remove(LOCK)
    sys.exit(1)


if os.path.exists(LOCK):
    L = json.load(open(LOCK))
    if os.path.exists(f"/proc/{L['pid']}"):
        log("이미 실행 중")
        sys.exit(0)
json.dump({"pid": os.getpid(), "start": time.strftime("%F %T")}, open(LOCK, "w"))


def train(name, data, epochs, seed):
    d = f"{CFR}/{name}"
    if os.path.exists(f"{d}/weights/last.pt"):
        log(f"{name} 이미 있음 — 건너뜀")
        return d
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/PURPOSE.md"):
        open(f"{d}/PURPOSE.md", "w").write(
            f"[소비처] 논문 — EXTRA_V2 (V2-10K vs V2-12.5K, 같은 recipe·같은 val)\n"
            f"[문장]  V2 안에서 추가 노출 2,500장이 real keypoint 성능을 더 개선한다.\n")
    code = f'''
import os, sys, json
sys.path.insert(0, "{ROOT}")
os.environ["A1_CONFIG"] = ""          # STANDARD only
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
    task="pose", mode="train", model="{INIT}", data="{data}",
    epochs={epochs}, batch=32, imgsz=640, optimizer="SGD", lr0=0.01, lrf=0.01,
    cos_lr=True, close_mosaic=10, pose=12.0, kobj=1.0, warmup_epochs=3.0,
    patience=0, single_cls=True, mosaic=0.3, scale=0.25, hsv_h=0.015,
    hsv_s=0.5, hsv_v=0.35, fliplr=0.0, flipud=0.0, erasing=0.4,
    seed={seed}, deterministic=True, save_period=10, device=0, workers=8,
    project="{CFR}", name="{name}", exist_ok=True, resume=False, val=True, plots=False))
tr.train()
c = getattr(tr.model, "criterion", None)
json.dump({{"PC_CALL_COUNT": PC["n"], "ROLE_CALLS": SY.ROLE_CALLS["n"],
           "criterion": type(getattr(c, "one2many", c)).__name__,
           "data": "{data}", "epochs": {epochs}, "seed": {seed}}},
          open("{CFR}/{name}/RUNTIME_AUDIT.json", "w"), indent=2)
'''
    sc = f"{Q}/_tr_{name}.py"
    open(sc, "w").write(code)
    lf = f"{Q}/_train_{name}.log"
    with open(lf, "w") as fh:
        r = subprocess.run([sys.executable, "-u", sc], cwd=d, stdout=fh,
                           stderr=subprocess.STDOUT, text=True)
    if not os.path.exists(f"{d}/weights/last.pt"):
        tail = ""
        try:
            tail = "\n".join(open(lf).read().replace("\r", "\n").split("\n")[-20:])
        except Exception:
            pass
        die(f"{name} 산출물 없음 (rc={r.returncode}, 로그 {lf}):\n{tail[-1000:]}")
    a = json.load(open(f"{d}/RUNTIME_AUDIT.json"))
    if a["PC_CALL_COUNT"] or a["ROLE_CALLS"]:
        die(f"{name} 커스텀 loss 오염 PC={a['PC_CALL_COUNT']} role={a['ROLE_CALLS']}")
    if "PoseLoss" not in a["criterion"]:
        die(f"{name} criterion 이 standard 가 아님: {a['criterion']}")
    nr = len(open(f"{d}/results.csv").read().strip().split("\n")) - 1
    if nr != epochs:
        die(f"{name} results.csv {nr}행 != {epochs}")
    if not os.path.exists(f"{d}/weights/best.pt"):
        die(f"{name} best.pt 없음")
    return d


def synth(tag, w, data):
    o = f"{Q}/SYNTH_{tag}.json"
    if not os.path.exists(o) and os.path.exists(w):
        subprocess.run([sys.executable, f"{Q}/cf_synth_eval.py", "--weights", w,
                        "--tag", tag, "--data", data], capture_output=True, text=True)
    return json.load(open(o)) if os.path.exists(o) else None


def real(tag, w):
    o = f"{Q}/REAL_{tag}.json"
    if not os.path.exists(o) and os.path.exists(w):
        subprocess.run([sys.executable, f"{Q}/cf_real_eval.py", "--weights", w, "--tag", tag],
                       capture_output=True, text=True)
    return json.load(open(o)) if os.path.exists(o) else None


# ---- U5 smoke --------------------------------------------------------------
log("U5 smoke 1ep (V2 COMPLETE 12.5K, standard only)")
sd = train(SMOKE, V2E, 1, 42)
rc = open(f"{sd}/results.csv").read().strip().split("\n")
vals = {h.strip(): v for h, v in zip(rc[0].split(","), rc[1].split(","))}
lo = [float(v) for k, v in vals.items() if "loss" in k and v not in ("", "nan")]
if not lo or not all(np.isfinite(lo)):
    die(f"smoke loss 비정상 {lo}")
ar = open(f"{sd}/args.yaml").read()
for need in ("batch: 32", "fliplr: 0.0", "seed: 42"):
    if need not in ar:
        die(f"smoke args 위반: {need}")
log(f"U5 PASS  loss finite  batch32/fliplr0/seed42")

# ---- args parity C vs E ------------------------------------------------------
import yaml
ca = yaml.safe_load(open(f"{CFR}/{C_ARM}/args.yaml"))
ALLOW = {"data", "name", "save_dir", "project"}
# ---- U6 E 60ep ----------------------------------------------------------------
log("U6 E 12.5K 60ep seed42")
ed = train(E_ARM, V2E, 60, 42)
ea = yaml.safe_load(open(f"{ed}/args.yaml"))
diff = {k: [ca.get(k), ea.get(k)] for k in set(ca) | set(ea)
        if ca.get(k) != ea.get(k)}
bad = {k: v for k, v in diff.items() if k not in ALLOW}
json.dump({"diff": diff, "allowed": sorted(ALLOW), "violations": bad},
          open(f"{Q}/C_vs_E_ARGS_DIFF.json", "w"), indent=2, ensure_ascii=False)
if bad:
    die(f"C/E args 허용 외 차이: {list(bad)}")
log(f"U6 완주  args diff = {sorted(diff)} (전부 허용)")

# ---- U7 공통 evaluator ---------------------------------------------------------
log("U7 same-evaluator (C/E: V1val·V2val·real140)")
S = {"C_v1": synth("C_V1VAL", f"{CFR}/{C_ARM}/weights/last.pt", V1),
     "E_v1": synth("E_V1VAL", f"{CFR}/{E_ARM}/weights/last.pt", V1),
     "C_v2": synth("C_V2VAL", f"{CFR}/{C_ARM}/weights/last.pt", V2C),
     "E_v2": synth("E_V2VAL", f"{CFR}/{E_ARM}/weights/last.pt", V2C)}
RC = real("DATA_C", f"{CFR}/{C_ARM}/weights/last.pt")
RE = real("DATA_E", f"{CFR}/{E_ARM}/weights/last.pt")
RA = real("A0", f"{CFR}/{A_ARM}/weights/last.pt")
FT = (json.load(open(f"{Q}/REAL_FT_REFERENCE.json"))
      if os.path.exists(f"{Q}/REAL_FT_REFERENCE.json") else None)
if not (RC and RE):
    die("real 평가 산출물 없음")
c, e = RC["cbox_paired"], RE["cbox_paired"]


def rel(b, x):
    return (b - x) / max(abs(b), 1e-12)


hits = {"cbox_+3pp": (RE["correct_box_recall"] - RC["correct_box_recall"]) >= 0.03,
        "median_10pct": rel(c["corner_median"], e["corner_median"]) >= 0.10,
        "p90_10pct": rel(c["corner_p90"], e["corner_p90"]) >= 0.10,
        "gross20_10pct": rel(c["gross20"], e["gross20"]) >= 0.10}
harm = {"median": rel(c["corner_median"], e["corner_median"]) <= -0.10,
        "p90": rel(c["corner_p90"], e["corner_p90"]) <= -0.10,
        "cbox": (RE["correct_box_recall"] - RC["correct_box_recall"]) <= -0.03,
        "det": (RE["detection_recall"] - RC["detection_recall"]) <= -0.02}
SIG = "POSITIVE" if (any(hits.values()) and not any(harm.values())) else "NULL_OR_WORSE"
gap = {}
if FT:
    f = FT["cbox_paired"]
    for k in ("corner_median", "corner_p90"):
        cg, eg = c[k] - f[k], e[k] - f[k]
        gap[k] = {"FT": f[k], "C": c[k], "E": e[k], "C_gap": cg, "E_gap": eg,
                  "additional_closure_10K_to_12K5": (cg - eg) / cg if abs(cg) > 1e-9 else None}
json.dump(gap, open(f"{Q}/V2_10K_TO_12K5_GAP_CLOSURE.json", "w"), indent=2, ensure_ascii=False)
WIN = "E_V2_COMPLETE_12K5" if SIG == "POSITIVE" else "C_V2_EARLY10K"
json.dump({"EXTRA_V2_SIGNAL": SIG, "hits": hits, "harm": harm,
           "INTERIM_BEST_DATASET": WIN,
           "★interpretation": ("C→E 는 pure N causal effect 가 아니다 — unique sample 증가와 "
                               "60ep 의 추가 optimization update 가 함께 늘었다. "
                               "'additional V2 training exposure under the same full training recipe'."),
           "scanner": json.load(open(f"{Q}/SCANNER_C_VS_E.json")),
           "real": {"C": c, "E": e, "A": (RA["cbox_paired"] if RA else None),
                    "FT": (FT["cbox_paired"] if FT else None),
                    "C_det": RC["detection_recall"], "E_det": RE["detection_recall"],
                    "C_cbox": RC["correct_box_recall"], "E_cbox": RE["correct_box_recall"]},
           "bootstrap_E_vs_C": RE.get("bootstrap"),
           "synthetic": {k: ({"mAP": v["pose_map50_95"], "median": v["corner_median"],
                              "p90": v["corner_p90"]} if v else None) for k, v in S.items()},
           "gap": gap}, open(f"{Q}/EXTRA_V2_SIGNAL.json", "w"), indent=2, ensure_ascii=False)
log(f"U8 EXTRA_V2_SIGNAL = {SIG}   INTERIM_BEST = {WIN}")

L = [f"**EXTRA_V2_SIGNAL = {SIG}**", "```",
     f"{'':16} {'C V2-10K':>10} {'E V2-12.5K':>11}", "-" * 40]
for lab, kc, ke in (("V1val mAP", "C_v1", "E_v1"), ("V2val mAP", "C_v2", "E_v2")):
    if S[kc] and S[ke]:
        L.append(f"{lab:16} {S[kc]['pose_map50_95']:10.4f} {S[ke]['pose_map50_95']:11.4f}")
        L.append(f"{lab.split()[0]+' median':16} {S[kc]['corner_median']:10.2f} {S[ke]['corner_median']:11.2f}")
        L.append(f"{lab.split()[0]+' p90':16} {S[kc]['corner_p90']:10.2f} {S[ke]['corner_p90']:11.2f}")
L += [f"{'real det':16} {RC['detection_recall']:10.3f} {RE['detection_recall']:11.3f}",
      f"{'real cbox':16} {RC['correct_box_recall']:10.3f} {RE['correct_box_recall']:11.3f}",
      f"{'real median':16} {c['corner_median']:10.2f} {e['corner_median']:11.2f}",
      f"{'real p90':16} {c['corner_p90']:10.2f} {e['corner_p90']:11.2f}",
      f"{'real gross20':16} {c['gross20']:10.4f} {e['gross20']:11.4f}",
      f"{'real bottom':16} {c['bottom_p90']:10.2f} {e['bottom_p90']:11.2f}",
      f"{'real DAY p90':16} {c.get('day_p90',float('nan')):10.2f} {e.get('day_p90',float('nan')):11.2f}",
      f"{'real NIGHT p90':16} {c.get('night_p90',float('nan')):10.2f} {e.get('night_p90',float('nan')):11.2f}",
      "```", ""]
if gap:
    L.append("GAP TO yolo26n-ft")
    L.append("```")
    for k, v in gap.items():
        ac = v["additional_closure_10K_to_12K5"]
        L.append(f"{k:14} FT {v['FT']:7.2f}  C_gap {v['C_gap']:+8.2f}  E_gap {v['E_gap']:+8.2f}  "
                 f"추가closure {'n/a' if ac is None else f'{100*ac:+.1f}%'}")
    L.append("```")
L += ["", f"hits {hits}", f"harm {harm}", f"INTERIM_BEST_DATASET = {WIN}",
      "★ C→E 는 순수 N 효과가 아니다 (unique sample + optimization update 동시 증가).",
      "★ scanner: C train eff 9704 / E train eff 12173, val 은 둘 다 123 (동일).",
      "engineering screen. real 은 EXPLORATORY membership. FINAL dataset 아님(interim)."]
txt = "\n".join(L)
open(f"{Q}/EXTRA_V2_SIGNAL.md", "w").write(txt)
print(txt)
notify(txt[:1800])

# ---- U9 winner seed43 -----------------------------------------------------------
w_arm = E_ARM.replace("SEED42", "SEED43") if WIN.startswith("E") else C_ARM.replace("SEED42", "SEED43")
w_data = V2E if WIN.startswith("E") else V2C
w_tag = "E43" if WIN.startswith("E") else "C43"
log(f"U9 winner seed43: {w_arm}")
wd = train(w_arm, w_data, 60, 43)
R43 = real(f"DATA_{w_tag}", f"{wd}/weights/last.pt")
S43 = synth(f"{w_tag}_V2VAL", f"{wd}/weights/last.pt", V2C)

# ---- U10 replication -------------------------------------------------------------
base = RE if WIN.startswith("E") else RC
b, n = base["cbox_paired"], (R43["cbox_paired"] if R43 else {})
A = RA["cbox_paired"] if RA else {}
repl = bool(R43 and A
            and R43["correct_box_recall"] > RA["correct_box_recall"]
            and n.get("corner_median", 1e9) < A.get("corner_median", 0)
            and n.get("gross20", 1e9) < A.get("gross20", 0))
json.dump({"winner": WIN, "seed42": {"cbox": base["correct_box_recall"],
                                     "median": b["corner_median"], "p90": b["corner_p90"],
                                     "gross20": b["gross20"], "bottom": b["bottom_p90"]},
           "seed43": ({"cbox": R43["correct_box_recall"], "median": n.get("corner_median"),
                       "p90": n.get("corner_p90"), "gross20": n.get("gross20"),
                       "bottom": n.get("bottom_p90")} if R43 else None),
           "V1_A_seed42": ({"cbox": RA["correct_box_recall"], "median": A.get("corner_median"),
                            "p90": A.get("corner_p90"), "gross20": A.get("gross20")} if RA else None),
           "V2_DATA_EFFECT_REPLICATED": repl,
           "criterion": "seed43 에서도 V1-A 대비 cbox↑ median↓ gross↓ 방향 유지"},
          open(f"{Q}/V2_SEED_REPLICATION.json", "w"), indent=2, ensure_ascii=False)
log(f"U10 V2_DATA_EFFECT_REPLICATED = {repl}")
notify(f"✅ **EXTRA_V2 COMPLETE**\nEXTRA_V2_SIGNAL = {SIG}\nINTERIM_BEST = {WIN}\n"
       f"V2_DATA_EFFECT_REPLICATED = {repl}\n{Q}/EXTRA_V2_SIGNAL.md")
if os.path.exists(LOCK):
    os.remove(LOCK)
