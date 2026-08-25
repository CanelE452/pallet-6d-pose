"""COVERAGE_EFFECT_10K — V1-10K vs V2-10K, loss/설정 전부 동일하고 data 만 다름.

smoke(1ep) -> 60ep -> 동일 evaluator 재평가 -> gap closure -> notify. 한 파일에서.
완료 판정은 results.csv 60행 + last.pt 로만 한다.
"""
import json, os, subprocess, sys, time
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
CFR = f"{Y}/runs_camera_facing_loss"
Q = f"{CFR}/ubuntu_cf_loss_queue_20260823T0930"
V1 = f"{Y}/datasets/v1_cf_matched10k/data.yaml"
V2 = f"{Y}/datasets/v2_cf_early10k/data.yaml"
INIT = f"{ROOT}/challenge/weights/pretrained_yolo/yolo26n-pose.pt"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
A_ARM = "CF_A0_V1MATCHED10K_60EP_SEED42_UBUNTU"                  # 기존 A
C_ARM = "CF_DATA_C_V2_EARLY10K_STD_60EP_SEED42_UBUNTU"           # 신규 C
SMOKE = "CF_V2_EARLY10K_STD_SMOKE_SEED42"
LOG = f"{Q}/COVERAGE_LOG.txt"
LOCK = f"{Q}/COVERAGE.lock"


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
    notify(f"❌ COVERAGE_EFFECT_10K: {m}")
    if os.path.exists(LOCK):
        os.remove(LOCK)
    sys.exit(1)


if os.path.exists(LOCK):
    L = json.load(open(LOCK))
    if os.path.exists(f"/proc/{L['pid']}"):
        log("이미 실행 중")
        sys.exit(0)
json.dump({"pid": os.getpid(), "start": time.strftime("%F %T")}, open(LOCK, "w"))


def train(name, data, epochs, seed=42):
    d = f"{CFR}/{name}"
    if os.path.exists(f"{d}/weights/last.pt"):
        log(f"{name} 이미 있음 — 건너뜀")
        return d
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/PURPOSE.md"):
        open(f"{d}/PURPOSE.md", "w").write(
            "[소비처] 논문 — COVERAGE_EFFECT_10K (V1-10K vs V2-10K controlled)\n"
            "[문장]  동일 조건에서 데이터만 V2 로 바꾸면 real keypoint 성능이 개선된다.\n")
    code = f'''
import os, sys, json
sys.path.insert(0, "{ROOT}")
os.environ["A1_CONFIG"] = ""          # ★ 모든 커스텀 loss OFF — STANDARD only
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
    # ★ 전체 로그를 파일로 남긴다. capture_output 의 꼬리만 남기면 CuDNN 경고가
    #   실제 예외를 밀어내 진단이 불가능하다 (2026-08-23 두 번 겪음).
    lf = f"{Q}/_train_{name}.log"
    with open(lf, "w") as fh:
        r = subprocess.run([sys.executable, "-u", sc], cwd=d, stdout=fh,
                           stderr=subprocess.STDOUT, text=True)
    if not os.path.exists(f"{d}/weights/last.pt"):
        tail = ""
        try:
            tail = "".join(open(lf).read().replace("\r", "\n").split("\n")[-25:])
        except Exception:
            pass
        die(f"{name} 산출물 없음 (rc={r.returncode}, 전체로그 {lf}):\n{tail[-1200:]}")
    a = json.load(open(f"{d}/RUNTIME_AUDIT.json"))  # 여기 오면 학습은 성공
    if a["PC_CALL_COUNT"] or a["ROLE_CALLS"]:
        die(f"{name} 커스텀 loss 오염 PC={a['PC_CALL_COUNT']} role={a['ROLE_CALLS']}")
    if "PoseLoss" not in a["criterion"]:
        die(f"{name} criterion 이 standard 가 아님: {a['criterion']}")
    nr = len(open(f"{d}/results.csv").read().strip().split("\n")) - 1
    if nr != epochs:
        die(f"{name} results.csv {nr}행 != {epochs}")
    return d


# ---- STEP 1  smoke ---------------------------------------------------------
log("STEP 1 smoke 1ep (V2 EARLY10K, standard only)")
sd = train(SMOKE, V2, 1)
rc = open(f"{sd}/results.csv").read().strip().split("\n")
hdr, row = rc[0].split(","), rc[1].split(",")
vals = {h.strip(): v for h, v in zip(hdr, row)}
lo = [float(v) for k, v in vals.items() if "loss" in k and v not in ("", "nan")]
if not lo or not all(np.isfinite(lo)) or any(np.isnan(lo)):
    die(f"smoke loss 비정상 {lo}")
ar = open(f"{sd}/args.yaml").read()
for need in ("batch: 32", "fliplr: 0.0", "seed: 42"):
    if need not in ar:
        die(f"smoke args 계약 위반: {need} 없음")
log(f"STEP 1 PASS  loss finite {['%.3f' % x for x in lo]}  batch32/fliplr0/seed42 확인")

# ---- STEP 2  60ep ----------------------------------------------------------
log("STEP 2 CF_DATA_C 60ep seed42 (clean init, resume=False)")
cd_ = train(C_ARM, V2, 60)
log("STEP 2 완주")

# ---- STEP 3  동일 evaluator 재평가 ------------------------------------------
log("STEP 3 same-evaluator 재평가 (A / C, V1val·V2val·real140)")


def synth(tag, w, data):
    o = f"{Q}/SYNTH_{tag}.json"
    if not os.path.exists(o):
        subprocess.run([sys.executable, f"{Q}/cf_synth_eval.py", "--weights", w,
                        "--tag", tag, "--data", data], capture_output=True, text=True)
    return json.load(open(o)) if os.path.exists(o) else None


def real(tag, w):
    o = f"{Q}/REAL_{tag}.json"
    if not os.path.exists(o):
        subprocess.run([sys.executable, f"{Q}/cf_real_eval.py", "--weights", w, "--tag", tag],
                       capture_output=True, text=True)
    return json.load(open(o)) if os.path.exists(o) else None


AW = f"{CFR}/{A_ARM}/weights/last.pt"
CW = f"{CFR}/{C_ARM}/weights/last.pt"
res = {"A_v1val": synth("A_V1VAL", AW, V1), "C_v1val": synth("C_V1VAL", CW, V1),
       "A_v2val": synth("A_V2VAL", AW, V2), "C_v2val": synth("C_V2VAL", CW, V2),
       "A_real": real("A0", AW), "C_real": real("DATA_C", CW),
       "FT_real": real("FT_REFERENCE", "")}
if res["FT_real"] is None:
    res["FT_real"] = (json.load(open(f"{Q}/REAL_FT_REFERENCE.json"))
                      if os.path.exists(f"{Q}/REAL_FT_REFERENCE.json") else None)

# ---- STEP 4  gate + gap closure --------------------------------------------
A, C, FT = res["A_real"], res["C_real"], res["FT_real"]
if not (A and C):
    die("real 평가 산출물 없음")
a, c = A["cbox_paired"], C["cbox_paired"]


def rel(b, x):
    return (b - x) / max(abs(b), 1e-12)


hits = {"cbox_recall_+5pp": (C["correct_box_recall"] - A["correct_box_recall"]) >= 0.05,
        "median_10pct": rel(a["corner_median"], c["corner_median"]) >= 0.10,
        "p90_10pct": rel(a["corner_p90"], c["corner_p90"]) >= 0.10,
        "gross20_10pct": rel(a["gross20"], c["gross20"]) >= 0.10}
harm = {"median": rel(a["corner_median"], c["corner_median"]) <= -0.10,
        "p90": rel(a["corner_p90"], c["corner_p90"]) <= -0.10,
        "cbox": (C["correct_box_recall"] - A["correct_box_recall"]) <= -0.05,
        "det": (C["detection_recall"] - A["detection_recall"]) <= -0.02}
verdict = "POSITIVE" if (any(hits.values()) and not any(harm.values())) else "NEGATIVE"
gap = {}
if FT:
    f = FT["cbox_paired"]
    for k in ("corner_median", "corner_p90"):
        ag, cg = a[k] - f[k], c[k] - f[k]
        gap[k] = {"FT": f[k], "A": a[k], "C": c[k], "A_gap": ag, "C_gap": cg,
                  "closure": (ag - cg) / ag if abs(ag) > 1e-9 else None}
out = {"experiment": "COVERAGE_EFFECT_10K", "verdict": verdict, "hits": hits, "harm": harm,
       "A_arm": A_ARM, "C_arm": C_ARM,
       "gate": "real cbox+5pp | median 10% | p90 10% | gross20 10% 중 하나, 큰 악화 없음",
       "gate_frozen_before_experiment": True,
       "effective_n": {"V1": 9867, "V2": 9704,
                       "note": "declared 9867 vs 9875 이나 corrupt 로 effective 가 163 다르다"},
       "real": {"A": a, "C": c, "FT": (FT["cbox_paired"] if FT else None),
                "A_det": A["detection_recall"], "C_det": C["detection_recall"],
                "A_cbox": A["correct_box_recall"], "C_cbox": C["correct_box_recall"]},
       "bootstrap_C_vs_A": C.get("bootstrap"),
       "synthetic": {k: ({"mAP": v["pose_map50_95"], "median": v["corner_median"],
                          "p90": v["corner_p90"], "gross20": v["gross20"],
                          "bottom_p90": v["bottom_p90"]} if v else None)
                     for k, v in res.items() if k.endswith("val")}}
json.dump(out, open(f"{Q}/COVERAGE_EFFECT_10K.json", "w"), indent=2, ensure_ascii=False)
json.dump(gap, open(f"{Q}/REAL_GAP_CLOSURE.json", "w"), indent=2, ensure_ascii=False)

L = [f"**COVERAGE_EFFECT_10K = {verdict}**", "```",
     f"{'':14} {'A V1-10K':>10} {'C V2-10K':>10}", "-" * 38]
for k, lab in (("A_v1val", "V1val"), ("A_v2val", "V2val")):
    kk = k.replace("A_", "")
    av, cv = res[f"A_{kk}"], res[f"C_{kk}"]
    if av and cv:
        L += [f"{lab+' mAP':14} {av['pose_map50_95']:10.4f} {cv['pose_map50_95']:10.4f}",
              f"{lab+' median':14} {av['corner_median']:10.2f} {cv['corner_median']:10.2f}",
              f"{lab+' p90':14} {av['corner_p90']:10.2f} {cv['corner_p90']:10.2f}"]
L += [f"{'real det':14} {A['detection_recall']:10.3f} {C['detection_recall']:10.3f}",
      f"{'real cbox':14} {A['correct_box_recall']:10.3f} {C['correct_box_recall']:10.3f}",
      f"{'real median':14} {a['corner_median']:10.2f} {c['corner_median']:10.2f}",
      f"{'real p90':14} {a['corner_p90']:10.2f} {c['corner_p90']:10.2f}",
      f"{'real gross20':14} {a['gross20']:10.4f} {c['gross20']:10.4f}",
      f"{'real bottom':14} {a['bottom_p90']:10.2f} {c['bottom_p90']:10.2f}",
      f"{'real day p90':14} {a.get('day_p90',float('nan')):10.2f} {c.get('day_p90',float('nan')):10.2f}",
      f"{'real night p90':14} {a.get('night_p90',float('nan')):10.2f} {c.get('night_p90',float('nan')):10.2f}",
      "```", ""]
if gap:
    L.append("GAP TO yolo26n-ft (closure = (A_gap − C_gap)/A_gap)")
    L.append("```")
    for k, v in gap.items():
        cl = "n/a" if v["closure"] is None else f"{100*v['closure']:+.1f}%"
        L.append(f"{k:14} FT {v['FT']:7.2f}  A_gap {v['A_gap']:+8.2f}  "
                 f"C_gap {v['C_gap']:+8.2f}  closure {cl}")
    L.append("```")
L += ["", f"hits {hits}", f"harm {harm}",
      "★ effective N: V1 9867 vs V2 9704 (corrupt 171/2) — 완전한 matched-N 은 아니다.",
      "engineering screen. real 은 EXPLORATORY membership."]
txt = "\n".join(L)
open(f"{Q}/COVERAGE_EFFECT_10K.md", "w").write(txt)
print(txt)
notify(txt[:1800])
log(f"=== COVERAGE_EFFECT_10K = {verdict} ===")
if os.path.exists(LOCK):
    os.remove(LOCK)
