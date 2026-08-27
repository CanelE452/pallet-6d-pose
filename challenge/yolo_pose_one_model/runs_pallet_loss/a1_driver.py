"""PSPC_A1_USD_ONLY — train -> measure(ep별) -> screen verdict -> notify. 한 파일에서.

RUN CLASS: EXPLORATORY_METHOD_SCREEN.  FINAL A1 이 아니다.
완료 판정은 산출물로만 한다 (exit code·프로세스 존재는 이 프로젝트에서 둘 다 거짓말한 이력).
"""
import json, os, subprocess, sys, time
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
R = f"{ROOT}/challenge/yolo_pose_one_model/runs_pallet_loss"
NAME = "PSPC_A1_USD_ONLY_V1MATCHED10K_60EP_SEED42"
RUN = f"{R}/{NAME}"
A0DIR = f"{ROOT}/challenge/yolo_pose_one_model/runs_fixed/V1_FIXED_MATCHED10K_60EP_SEED42_UBUNTU/weights"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
EPOCHS = ["epoch0", "epoch10", "epoch20", "epoch30", "epoch40", "epoch50", "last"]


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
    os.environ["A1_CONFIG"] = f"{R}/A1_USD_ONLY_CONFIG.json"
    os.environ["PSPC_CONFIG"] = ""
    # A2 projective term 호출 횟수를 실제로 센다 (0 이어야 한다)
    from pallet_yolo_loss.loss import PSPCPoseLoss26
    PC_CALLS = {"n": 0}
    _pc = PSPCPoseLoss26.projective_loss

    def _spy(self, *a, **k):
        PC_CALLS["n"] += 1
        return _pc(self, *a, **k)
    PSPCPoseLoss26.projective_loss = _spy

    from pallet_yolo_loss.trainer import A1SymmetryTrainer
    say("STEP 1 train 60ep seed42  (save_period=10, A0 와 동일)")
    tr = A1SymmetryTrainer(overrides=dict(
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
    a1cfg = getattr(inner, "a1", None)
    audit = {"pc_projective_loss_calls": PC_CALLS["n"],
             "pspc_enabled": bool(getattr(getattr(inner, "pspc", None), "enabled", None)),
             "pspc_lambda_pc": getattr(getattr(inner, "pspc", None), "lambda_pc", None),
             "a1_enabled": bool(getattr(a1cfg, "enabled", False)),
             "a1_sym_assets": list(getattr(a1cfg, "sym_assets", ())),
             "a1_lambda_role": getattr(a1cfg, "lambda_role", None),
             "a1_stats_last_batch": getattr(inner, "a1_stats", None)}
    json.dump(audit, open(f"{R}/A1_USD_ONLY_RUNTIME_AUDIT.json", "w"), indent=2)
    say(f"runtime audit {audit}")
    if not os.path.exists(LAST):
        die("last.pt 없음 — 학습이 산출물 없이 끝났다")
    if PC_CALLS["n"] != 0:
        die(f"A2 projective term 이 {PC_CALLS['n']}회 호출됐다 — A1 오염")
    if not audit["a1_enabled"]:
        die("A1 loss 가 꺼진 채로 학습됐다 — A0 재현일 뿐")

# ---- STEP 2  measure — A0/A1 를 ep 별로 같은 식으로 --------------------
say("STEP 2 measure (v1 val 133, 학습과 같은 이미지 도메인)")
jobs = []
for e in EPOCHS:
    jobs.append((f"A0_{e}", f"{A0DIR}/{e}.pt"))
    jobs.append((f"A1_{e}", f"{RUN}/weights/{e}.pt"))
for tag, w in jobs:
    o = f"{R}/A1_MEASURE_{tag}.json"
    if os.path.exists(o) or not os.path.exists(w):
        continue
    r = subprocess.run([sys.executable, f"{R}/a1_measure.py", "--weights", w, "--tag", tag])
    if r.returncode or not os.path.exists(o):
        die(f"{tag} 측정 실패")


def load(tag):
    p = f"{R}/A1_MEASURE_{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None


# ---- STEP 3  screen verdict (사용자 사전등록 기준, 하드코딩) --------------
say("STEP 3 verdict")
curve = []
for e in EPOCHS:
    a, b = load(f"A0_{e}"), load(f"A1_{e}")
    if not (a and b):
        continue
    curve.append({"epoch": e,
                  "A0_ident": a["identity_d_id_median"], "A1_ident": b["identity_d_id_median"],
                  "A0_yaw180": a["flip_rate"], "A1_yaw180": b["flip_rate"],
                  "A0_best": a["yaw180best_e_sym_median"], "A1_best": b["yaw180best_e_sym_median"],
                  "A0_px50": a["corner_px_identity_median"], "A1_px50": b["corner_px_identity_median"],
                  "A0_px90": a["corner_px_identity_p90"], "A1_px90": b["corner_px_identity_p90"],
                  "A0_map": a.get("pose_map50_95"), "A1_map": b.get("pose_map50_95")})
if not curve:
    die("측정 결과가 하나도 없다")
fin = curve[-1]

# paired bootstrap — 최종 시점, SYM asset 프레임만 (min() 이 적용된 모집단)
SYM = {"scene.usd", "scene_1.usd"}
a, b = load("A0_last"), load("A1_last")
ci = None
if a and b:
    m0 = {r["stem"]: r for r in a["per_frame"]}
    m1 = {r["stem"]: r for r in b["per_frame"]}
    ss = [s for s in sorted(set(m0) & set(m1)) if m0[s]["asset"] in SYM]
    if len(ss) >= 30:
        d = np.array([m0[s]["e_sym"] - m1[s]["e_sym"] for s in ss])
        rng = np.random.default_rng(0)
        bsd = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
        ci = [float(np.percentile(bsd, 2.5)), float(np.percentile(bsd, 97.5)),
              float(d.mean()), len(ss)]

# 판정: yaw180 confusion 감소 또는 더 빠른 수렴  AND  mAP regression <= 2pp
dmap = (fin["A1_map"] - fin["A0_map"]) if (fin["A1_map"] is not None and fin["A0_map"] is not None) else None
map_ok = dmap is not None and dmap >= -0.02
yaw_better = fin["A1_yaw180"] < fin["A0_yaw180"]
faster = any(c["A1_yaw180"] < c["A0_yaw180"] for c in curve[:-1])
sem = yaw_better or faster or (ci is not None and ci[0] > 0)
verdict = "POSITIVE_SCREEN" if (sem and map_ok) else (
    "NEGATIVE_SCREEN_MAP_REGRESSION" if not map_ok else "NEGATIVE_SCREEN")

res = {"run_class": "EXPLORATORY_METHOD_SCREEN", "verdict": verdict,
       "note": "engineering screen. paper final gate 아님. real evaluation 전 METHOD_SUPPORTED 선언 금지.",
       "final": fin, "curve": curve, "paired_bootstrap_sym_last": ci,
       "delta_map50_95": dmap,
       "gate": "yaw180 감소 또는 더 빠른 수렴  AND  pose mAP50-95 regression <= 2pp"}
json.dump(res, open(f"{R}/A1_USD_ONLY_SCREEN.json", "w"), indent=2)

lines = [f"**{NAME}  =  {verdict}**  (EXPLORATORY_METHOD_SCREEN)", "```",
         f"{'epoch':8} {'ident A0->A1':>22} {'yaw180% A0->A1':>20} {'mAP A0->A1':>20}",
         "─" * 74]
for c in curve:
    mp = "  -" if c["A0_map"] is None else f"{c['A0_map']:.4f}->{c['A1_map']:.4f}"
    lines.append(f"{c['epoch']:8} {c['A0_ident']:9.5f}->{c['A1_ident']:9.5f} "
                 f"{100*c['A0_yaw180']:8.2f}->{100*c['A1_yaw180']:7.2f} {mp:>20}")
lines.append("```")
if ci:
    lines.append(f"SYM {ci[3]}프레임 paired bootstrap  ΔE_sym {ci[2]:+.5f}  95%CI [{ci[0]:+.5f}, {ci[1]:+.5f}]")
lines.append(f"Δ mAP50-95 = {dmap:+.4f}" if dmap is not None else "mAP 측정 실패")
lines.append("seed 1개 screen. real evaluation 전 METHOD_SUPPORTED 선언 금지.")
txt = "\n".join(lines)
print(txt)
notify(txt)
say("done")
