"""PSPC_ASC_USD_ONLY — train -> 3자 동일 측정 -> gate -> notify. 한 파일에서.

RUN CLASS: EXPLORATORY_METHOD_SCREEN.  paper 채택은 real 평가 후 결정.
완료 판정은 산출물로만 한다.
"""
import csv, json, os, subprocess, sys, time
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
R = f"{ROOT}/challenge/yolo_pose_one_model/runs_pallet_loss"
NAME = "PSPC_ASC_USD_ONLY_V1MATCHED10K_60EP_SEED42"
RUN = f"{R}/{NAME}"
A0W = f"{ROOT}/challenge/yolo_pose_one_model/runs_fixed/V1_FIXED_MATCHED10K_60EP_SEED42_UBUNTU/weights"
A1W = f"{R}/PSPC_A1_USD_ONLY_V1MATCHED10K_60EP_SEED42/weights"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
EP = ["epoch0", "epoch10", "epoch20", "epoch30", "epoch40", "epoch50", "last"]
ARMS = [("A0", A0W), ("A1", A1W), ("ASC", f"{RUN}/weights")]


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def notify(m):
    try:
        subprocess.run([NOTIFY, m], timeout=60)
    except Exception:
        pass


def die(m):
    say("FAIL " + m)
    notify(f"❌ {NAME}: {m}")
    sys.exit(1)


# ---- STEP 1  train --------------------------------------------------------
LAST = f"{RUN}/weights/last.pt"
if os.path.exists(LAST):
    say("이미 학습됨 — 건너뜀")
else:
    os.environ["A1_CONFIG"] = f"{R}/ASC_LOSS_CONFIG.json"
    os.environ["PSPC_CONFIG"] = ""
    from pallet_yolo_loss.loss import PSPCPoseLoss26
    import pallet_yolo_loss.symmetry as SY
    PC = {"n": 0}
    _pc = PSPCPoseLoss26.projective_loss

    def _spy(self, *a, **k):
        PC["n"] += 1
        return _pc(self, *a, **k)
    PSPCPoseLoss26.projective_loss = _spy
    SY.ROLE_CALLS["n"] = 0

    from pallet_yolo_loss.trainer import ASCTrainer
    say("STEP 1 train 60ep seed42  (ASC beta 1.0[0-19] -> ramp[20-29] -> 0.0[30-59])")
    tr = ASCTrainer(overrides=dict(
        task="pose", mode="train",
        model=f"{ROOT}/challenge/weights/pretrained_yolo/yolo26n-pose.pt",
        data=f"{ROOT}/challenge/yolo_pose_one_model/datasets/v1_fixed_matched10k/data.yaml",
        epochs=60, batch=32, imgsz=640, optimizer="SGD", lr0=0.01, lrf=0.01,
        cos_lr=True, close_mosaic=10, pose=12.0, kobj=1.0, warmup_epochs=3.0,
        patience=0, single_cls=True, mosaic=0.3, scale=0.25, hsv_s=0.5, hsv_v=0.35,
        fliplr=0.0, flipud=0.0, erasing=0.4, seed=42, deterministic=True,
        save_period=10, device=0, workers=8, project=R, name=NAME,
        exist_ok=True, resume=False, val=True, plots=False))
    tr.train()
    crit = getattr(tr.model, "criterion", None)
    inner = getattr(crit, "one2many", crit)
    a1c = getattr(inner, "a1", None)
    audit = {"PC_CALL_COUNT": PC["n"], "ROLE_MARGIN_CALL_COUNT": SY.ROLE_CALLS["n"],
             "asc_enabled": bool(getattr(a1c, "asc_enabled", False)),
             "asc_full_end": getattr(a1c, "asc_full_end", None),
             "asc_ramp_end": getattr(a1c, "asc_ramp_end", None),
             "sym_assets": list(getattr(a1c, "sym_assets", ())),
             "lambda_role": getattr(a1c, "lambda_role", None),
             "last_batch_stats": getattr(inner, "a1_stats", None)}
    json.dump(audit, open(f"{RUN}/RUNTIME_AUDIT.json", "w"), indent=2)
    say(f"runtime {audit}")
    if not os.path.exists(LAST):
        die("last.pt 없음")
    if PC["n"] != 0:
        die(f"A2 projective term {PC['n']}회 호출 — 오염")
    if SY.ROLE_CALLS["n"] != 0:
        die(f"role margin {SY.ROLE_CALLS['n']}회 호출 — 계약 위반")
    if not audit["asc_enabled"]:
        die("ASC 가 꺼진 채 학습됐다")

# ---- STEP 2  세 모델을 같은 invocation 으로 --------------------------------
say("STEP 2 measure — A0 / A1 / ASC, 동일 evaluator, v1 val 133")
for arm, wd in ARMS:
    for e in EP:
        tag = f"{arm}_{e}"
        o = f"{R}/A1_MEASURE_{tag}.json"
        w = f"{wd}/{e}.pt"
        if os.path.exists(o) or not os.path.exists(w):
            continue
        r = subprocess.run([sys.executable, f"{R}/a1_measure.py", "--weights", w, "--tag", tag])
        if r.returncode or not os.path.exists(o):
            die(f"{tag} 측정 실패")


def L(tag):
    p = f"{R}/A1_MEASURE_{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None


# ---- STEP 3  비교표 --------------------------------------------------------
rows = []
for e in EP:
    r = {"epoch": e}
    for arm, _ in ARMS:
        d = L(f"{arm}_{e}")
        if not d:
            continue
        r[f"{arm}_map50"] = d.get("pose_map50")
        r[f"{arm}_map"] = d.get("pose_map50_95")
        r[f"{arm}_d_id"] = d["identity_d_id_median"]
        r[f"{arm}_esym"] = d["yaw180best_e_sym_median"]
        r[f"{arm}_flipn"] = int(round(d["flip_rate"] * d["n_frames"]))
        r[f"{arm}_flip"] = d["flip_rate"]
        r[f"{arm}_p50"] = d["corner_px_identity_median"]
        r[f"{arm}_p90"] = d["corner_px_identity_p90"]
    rows.append(r)
cols = ["epoch"] + [f"{a}_{k}" for k in
                    ("map50", "map", "d_id", "esym", "flipn", "flip", "p50", "p90")
                    for a, _ in ARMS]
with open(f"{RUN}/A0_A1_ASC_COMPARISON.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

# channel collapse: 예측 keypoint 가 한 점으로 붕괴했는가
collapse = {}
for arm, _ in ARMS:
    d = L(f"{arm}_last")
    if d:
        sp = [r["px_id"] for r in d["per_frame"]]
        collapse[arm] = int(sum(1 for r in d["per_frame"] if r["px_id"] > 300))
        collapse[f"{arm}_note"] = f"px_id>300 프레임 수 / {len(sp)}"

# ---- STEP 4  gate (하드코딩) ----------------------------------------------
def g(e, arm, k):
    for r in rows:
        if r["epoch"] == e:
            return r.get(f"{arm}_{k}")
    return None


c1a = (g("epoch20", "ASC", "map") is not None and g("epoch20", "A0", "map") is not None
       and g("epoch20", "ASC", "map") >= g("epoch20", "A0", "map") + 0.10)
c1b = (g("epoch20", "ASC", "flip") is not None
       and g("epoch20", "ASC", "flip") < g("epoch20", "A0", "flip"))
cond1 = bool(c1a and c1b)

fa, fs = g("last", "A0", "map"), g("last", "ASC", "map")
pa, ps = g("last", "A0", "p90"), g("last", "ASC", "p90")
ka, ks = g("last", "A0", "flip"), g("last", "ASC", "flip")
c2A = fs is not None and fs >= fa + 0.005
c2B = ps is not None and (pa - ps) / max(pa, 1e-9) >= 0.03
c2C = ks is not None and ks < ka and fs >= fa - 0.005
cond2 = bool(c2A or c2B or c2C)

d_a1, d_asc = g("last", "A1", "d_id"), g("last", "ASC", "d_id")
better_than_a1 = d_asc is not None and d_a1 is not None and d_asc < d_a1
no_collapse = collapse.get("ASC", 0) == 0

verdict = "PASS" if (cond1 and cond2 and better_than_a1 and no_collapse) else "FAIL"
res = {"run_class": "EXPLORATORY_METHOD_SCREEN", "ASC_SIGNAL": verdict,
       "cond1_early_acceleration": {"pass": cond1,
                                    "map20_ge_A0_plus_0.10": c1a, "flip20_lt_A0": c1b},
       "cond2_final_recovery": {"pass": cond2, "A_map_ge_A0+0.005": c2A,
                                "B_p90_3pct_better": c2B, "C_flip_lt_A0_and_map_ge_-0.5pp": c2C},
       "d_id_better_than_A1": better_than_a1, "channel_collapse": collapse,
       "REAL_EVAL_REQUIRED": verdict == "PASS",
       "note": "engineering screen. paper 채택은 real 평가 후. FAIL 이면 schedule sweep·"
               "transition 변경·seed43 금지, symmetry-loss track 종료 후보.",
       "rows": rows}
json.dump(res, open(f"{RUN}/ASC_SCREEN_VERDICT.json", "w"), indent=2)

md = ["# A0 / A1_USD_ONLY / ASC_USD_ONLY", "", "```",
      f"{'epoch':>7} {'mAP50-95 A0/A1/ASC':>28} {'flip n A0/A1/ASC':>20} "
      f"{'d_id A0/A1/ASC':>26} {'p90 A0/A1/ASC':>24}", "-" * 108]
for r in rows:
    def t(k, f):
        return "/".join((f % r[f"{a}_{k}"]) if r.get(f"{a}_{k}") is not None else "-"
                        for a, _ in ARMS)
    md.append(f"{r['epoch']:>7} {t('map','%.4f'):>28} {t('flipn','%d'):>20} "
              f"{t('d_id','%.5f'):>26} {t('p90','%.1f'):>24}")
md += ["```", "", f"**ASC_SIGNAL = {verdict}**", "",
       f"- 조건1 early acceleration: {cond1}  (mAP20 ≥ A0+0.10 = {c1a}, flip20 < A0 = {c1b})",
       f"- 조건2 final recovery: {cond2}  (A={c2A} B={c2B} C={c2C})",
       f"- d_id 가 A1 보다 개선: {better_than_a1}",
       f"- channel collapse: {collapse}", "",
       "engineering screen. real 평가 전 METHOD_SUPPORTED 선언 금지."]
open(f"{RUN}/A0_A1_ASC_COMPARISON.md", "w").write("\n".join(md))

txt = "\n".join(md[:1] + md[2:])
print(txt)
notify(f"**{NAME}  ASC_SIGNAL = {verdict}**\n" + "\n".join(md[2:]))
say("done")
