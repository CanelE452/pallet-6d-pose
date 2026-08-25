"""ROOTCAUSE FAST SCREEN queue — B10 대기 → M10 → T10 → 평가 → RENDER 결정.

사전등록: ROOTCAUSE_FAST_SCREEN_PREREG.json.  결과 보고 threshold 변경 금지.
"""
import collections, json, os, subprocess, sys, time
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
CFR = f"{Y}/runs_camera_facing_loss"
Q = f"{CFR}/ubuntu_cf_loss_queue_20260823T0930"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
LOG = f"{Q}/ROOTFAST_LOG.txt"
LOCK = f"{Q}/ROOTFAST.lock"
LEAK = set(json.load(open(f"{Q}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
ARMS = [("B10", f"{Y}/datasets/v2_cf_early10k/data.yaml"),
        ("M10", f"{Y}/datasets/fast_m10_n9704/data.yaml"),
        ("T10", f"{Y}/datasets/fast_t10_n9704/data.yaml")]
T0 = time.time()


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def notify(m):
    try:
        subprocess.run([NOTIFY, m], timeout=60)
    except Exception as e:
        log(f"discord 실패(무시): {e}")


if os.path.exists(LOCK):
    L = json.load(open(LOCK))
    if os.path.exists(f"/proc/{L['pid']}"):
        log("이미 실행 중")
        sys.exit(0)
json.dump({"pid": os.getpid(), "start": time.strftime("%F %T")}, open(LOCK, "w"))

TIMES = {}
for arm, data in ARMS:
    d = f"{CFR}/ROOTFAST_{arm}_30EP_SEED42"
    t = time.time()
    def done(dd, ep=30):
        # ★ last.pt 는 매 epoch 갱신되므로 완료 근거가 아니다 (2026-08-24 오판 실제 발생).
        rp = f"{dd}/results.csv"
        return (os.path.exists(f"{dd}/weights/last.pt") and os.path.exists(rp)
                and len(open(rp).read().strip().split("\n")) - 1 >= ep)
    if done(d):
        log(f"{arm} 이미 완료")
    else:
        # B10 은 이미 별도 프로세스로 돌고 있을 수 있다 — 끝날 때까지 대기
        while any(f"--arm {arm}" in (open(f"/proc/{p}/cmdline", "rb").read().decode("utf8", "ignore")
                                     .replace("\x00", " ") if os.path.exists(f"/proc/{p}/cmdline") else "")
                  for p in os.listdir("/proc") if p.isdigit()):
            log(f"{arm} 외부 프로세스 진행 중 — 대기")
            time.sleep(120)
        if not done(d):
            log(f"{arm} 30ep 시작")
            r = subprocess.run([sys.executable, f"{Q}/fast_train_one.py", "--arm", arm,
                                "--data", data], capture_output=True, text=True)
            log(f"{arm} {(r.stdout or r.stderr or '')[-200:].strip()}")
    if not done(d):
        log(f"{arm} 30ep 미완료 — 이 arm 은 결과에서 제외")
        continue
    TIMES[arm] = round((time.time() - t) / 60, 1)
    a = json.load(open(f"{d}/RUNTIME_AUDIT.json"))
    nep = len(open(f"{d}/results.csv").read().strip().split("\n")) - 1
    log(f"{arm} 완료 {nep}ep batches/ep {a['n_train_batches']} PC {a['PC_CALL_COUNT']} "
        f"role {a['ROLE_CALLS']} {a['criterion']}")


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


log("평가 (SAME REAL n=128, last.pt)")
R, NC = {}, {}
for arm, _ in ARMS:
    w = f"{CFR}/ROOTFAST_{arm}_30EP_SEED42/weights/last.pt"
    R[arm] = real(f"RF_{arm}", w)
    NC[arm] = nightc(f"RF_{arm}", w)


def agg(rs):
    cb = [r for r in rs if r.get("correct_box")]
    e = np.concatenate([r["err"] for r in cb]) if cb else np.array([])
    return {"n": len(rs), "cbox": len(cb) / max(len(rs), 1),
            "med": float(np.median(e)) if e.size else None,
            "p90": float(np.percentile(e, 90)) if e.size else None,
            "gross": float((e > 20).mean()) if e.size else None}


TAB = {}
for arm, d in R.items():
    if not d:
        continue
    pf = [r for r in d["per_frame"] if r["frame"] not in LEAK]
    TAB[arm] = {"ALL": agg(pf), "DAY": agg([r for r in pf if r["domain"] == "DAY"]),
                "NIGHT": agg([r for r in pf if r["domain"] == "NIGHT"])}

MQ = json.load(open(f"{Q}/FAST_M10_MATCH_QUALITY.json"))
GOOD = MQ["MATCH_GOOD_ENOUGH"]
b, m, t = NC.get("B10"), NC.get("M10"), NC.get("T10")


def d_(x, y, k):
    return None if not (x and y) else (y[k] - x[k])


# ---- 판정 (사전등록) --------------------------------------------------------
support = None
if b and m:
    support = (d_(b, m, "top1_cbox") or 0) >= 0.10 or (d_(b, m, "margin_median") or 0) >= 0.10 \
        or ((b["wrong_present_frac"] - m["wrong_present_frac"]) >= 0.10)
target_res = None
if m and t:
    target_res = (d_(m, t, "top1_cbox") or 0) >= 0.10
if support and target_res:
    case = "BOTH"
elif support:
    case = "RENDER_SUPPORT"
elif target_res:
    case = "TARGET_RESIDUAL"
else:
    case = "NO_TARGET_EFFECT"

if case in ("RENDER_SUPPORT", "BOTH"):
    RA = "TARGETED_GENERIC_SUPPORT_ADDON" if not GOOD else "KEEP_STOPPED"
elif case == "TARGET_RESIDUAL":
    RA = "NO_RENDER_MOVE_TO_ADAPTATION"
else:
    RA = "KEEP_STOPPED"

out = {"prereg": "ROOTCAUSE_FAST_SCREEN_PREREG.json", "times_min": TIMES,
       "common_N": 9704, "match_quality": MQ, "table": TAB, "night_candidate": NC,
       "CASE": case, "GENERIC_SUPPORT_EFFECT": support, "TARGET_SPECIFIC_RESIDUAL": target_res,
       "RENDER_ACTION": RA,
       "★caveat": ("MATCH_GOOD_ENOUGH=False 이고 M10 은 B10 과 7,990/10,000 겹친다 "
                   "(V2 풀에 OT support 영역이 없어 선택 자유도 2,500 으로도 분포가 안 움직임). "
                   "따라서 M10 은 B10 과 거의 같은 arm 이며, T10−M10 차이를 target identity "
                   "효과로 단정할 수 없다. NIGHT n=28, seed 1개, 30ep screen."),
       "total_min": round((time.time() - T0) / 60, 1)}
json.dump(out, open(f"{Q}/ROOTFAST_SCREEN_RESULT.json", "w"), indent=2, ensure_ascii=False)

L = ["# ROOTCAUSE FAST SCREEN (30ep, common N=9,704)", "", "```",
     f"{'':16} {'B10':>10} {'M10':>10} {'T10':>10}", "-" * 50]
for sc in ("ALL", "DAY", "NIGHT"):
    for k, lab in (("cbox", "cbox"), ("med", "median"), ("p90", "p90"), ("gross", "gross20")):
        row = []
        for a in ("B10", "M10", "T10"):
            v = TAB.get(a, {}).get(sc, {}).get(k)
            row.append("       n/a" if v is None else f"{v:10.3f}" if k in ("cbox", "gross") else f"{v:10.2f}")
        L.append(f"{sc+' '+lab:16} " + " ".join(row))
    L.append("")
L += ["```", "", "## NIGHT candidate", "```",
      f"{'':16} {'B10':>10} {'M10':>10} {'T10':>10}", "-" * 50]
for k, lab in (("any_cbox", "any-cbox"), ("top1_cbox", "top1-cbox"),
               ("cand_per_frame", "cand/frame"), ("wrong_present_frac", "wrong%"),
               ("margin_median", "margin")):
    row = []
    for a in ("B10", "M10", "T10"):
        v = (NC.get(a) or {}).get(k)
        row.append("       n/a" if v is None else f"{v:10.3f}")
    L.append(f"{lab:16} " + " ".join(row))
L += ["```", "", f"MATCH_GOOD_ENOUGH = {GOOD}   (SMD r3d "
      f"{MQ['SMD_T10_vs_M10']['r3d']:+.2f}, elev {MQ['SMD_T10_vs_M10']['elevation']:+.2f})", "",
      f"**CASE = {case}**", f"GENERIC_SUPPORT_EFFECT = {support}",
      f"TARGET_SPECIFIC_RESIDUAL = {target_res}", "",
      f"**RENDER_ACTION = {RA}**", "",
      f"times(min) {TIMES}  total {out['total_min']}", "",
      "★ M10 은 B10 과 7,990/10,000 겹친다 — V2 풀에 OT support 가 없어 실질 대조가 약하다.",
      "  T10−M10 을 target identity 효과로 단정하지 않는다. NIGHT n=28, seed 1, 30ep."]
txt = "\n".join(L)
open(f"{Q}/ROOTFAST_SCREEN_RESULT.md", "w").write(txt)
print(txt)
notify(txt[:1800])
log(f"=== FAST SCREEN 완료  CASE={case}  RENDER={RA} ===")
if os.path.exists(LOCK):
    os.remove(LOCK)
