"""PHASE U6~U9 — N0/N1 FAST ADAPT SCREEN. train -> eval -> verdict -> notify 한 파일."""
import json, os, subprocess, sys, time
import numpy as np

R = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, R)
Y = f"{R}/challenge/yolo_pose_one_model"
CFR = f"{Y}/runs_camera_facing_loss"
Q = f"{CFR}/ubuntu_cf_loss_queue_20260823T0930"
DS = f"{Y}/datasets"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
LOG, LOCK = f"{Q}/NEGSCREEN_LOG.txt", f"{Q}/NEGSCREEN.lock"
INIT = json.load(open(f"{Q}/G38_ADAPT_INIT_LOCK.json"))["G38_INIT_PATH"]
GATE = json.load(open(f"{Q}/NEG_SCREEN_GATE_PREREG.json"))
LEAK = set(json.load(open(f"{Q}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
ARMS = [("N0", f"{DS}/adapt_n0_control/data.yaml"), ("N1", f"{DS}/adapt_n1_negative/data.yaml")]
EPOCHS = 15


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); open(LOG, "a").write(line + "\n")


def notify(m):
    try: subprocess.run([NOTIFY, m], timeout=60)
    except Exception as e: log(f"discord 실패(무시): {e}")


def die(m):
    log("FAIL " + m); notify(f"❌ NEG SCREEN: {m}")
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


def train(name, data, epochs):
    d = f"{CFR}/{name}"
    if done(d, epochs):
        log(f"{name} 이미 완료"); return d
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/PURPOSE.md"):
        open(f"{d}/PURPOSE.md", "w").write(
            "[소비처] 논문 — real adaptation 2x2 screen 의 NEGATIVE 축 (host-internal DELTA)\n"
            "[문장]  real negative supervision 만으로 G38 의 false-positive/랭킹이 개선된다.\n")
    code = f'''
import os, sys, json
sys.path.insert(0, "{R}")
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
    task="pose", mode="train", model="{INIT}", data="{data}",
    epochs={epochs}, batch=32, imgsz=640, optimizer="SGD", lr0=0.002, lrf=0.01,
    cos_lr=True, close_mosaic=10, warmup_epochs=1.0, patience=0,
    single_cls=True, mosaic=0.15, scale=0.25, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
    fliplr=0.0, flipud=0.0, erasing=0.4, seed=42, deterministic=True,
    save_period=5, device=0, workers=8, project="{CFR}", name="{name}",
    exist_ok=True, resume=False, val=True, plots=False))
tr.train()
c = getattr(tr.model, "criterion", None)
json.dump({{"PC_CALL_COUNT": PC["n"], "ROLE_CALLS": SY.ROLE_CALLS["n"],
           "criterion": type(getattr(c, "one2many", c)).__name__,
           "n_train_batches": len(tr.train_loader), "epochs": {epochs},
           "data": "{data}", "init": "{INIT}"}},
          open("{CFR}/{name}/RUNTIME_AUDIT.json", "w"), indent=2)
'''
    sc = f"{Q}/_tr_{name}.py"; open(sc, "w").write(code)
    lf = f"{Q}/_train_{name}.log"
    with open(lf, "w") as fh:
        r = subprocess.run([sys.executable, "-u", sc], cwd=d, stdout=fh,
                           stderr=subprocess.STDOUT, text=True)
    if not os.path.exists(f"{d}/weights/last.pt"):
        tail = "\n".join(open(lf).read().replace("\r", "\n").split("\n")[-25:])
        die(f"{name} 산출물 없음 (rc={r.returncode}):\n{tail[-900:]}")
    a = json.load(open(f"{d}/RUNTIME_AUDIT.json"))
    if a["PC_CALL_COUNT"] or a["ROLE_CALLS"]: die(f"{name} 커스텀 loss 오염")
    if "PoseLoss" not in a["criterion"]: die(f"{name} criterion {a['criterion']}")
    return d


AUD = {}
for arm, data in ARMS:
    log(f"PHASE U7 {arm} smoke 1ep")
    sd = train(f"ADAPT_{arm}_SMOKE_SEED42", data, 1)
    rc = open(f"{sd}/results.csv").read().strip().split("\n")
    vals = {h.strip(): v for h, v in zip(rc[0].split(","), rc[1].split(","))}
    lo = [float(v) for k, v in vals.items() if "loss" in k and v not in ("", "nan")]
    if not lo or not all(np.isfinite(lo)): die(f"{arm} smoke loss {lo}")
    log(f"PHASE U7 {arm} smoke PASS  loss finite {len(lo)}개")
    log(f"PHASE U7 {arm} 15ep")
    d = train(f"ADAPT_{arm}_15EP_SEED42", data, EPOCHS)
    AUD[arm] = json.load(open(f"{d}/RUNTIME_AUDIT.json"))
    AUD[arm]["dir"] = d
    AUD[arm]["actual_epochs"] = len(open(f"{d}/results.csv").read().strip().split("\n")) - 1
    log(f"PHASE U7 {arm} 완료 {AUD[arm]['actual_epochs']}ep batches/ep {AUD[arm]['n_train_batches']}")

if AUD["N0"]["n_train_batches"] != AUD["N1"]["n_train_batches"]:
    die(f"batches/ep 불일치 {AUD['N0']['n_train_batches']} vs {AUD['N1']['n_train_batches']}")

# ---------------- U8 EVAL ----------------------------------------------------
log("PHASE U8 SAME REAL n=128 + NEG heldout 2,689")


def run(script, tag, w):
    o = f"{Q}/{script[0]}_{tag}.json"
    if not os.path.exists(o):
        r = subprocess.run([sys.executable, f"{Q}/{script[1]}", "--weights", w, "--tag", tag],
                           capture_output=True, text=True)
        if not os.path.exists(o): die(f"{script[1]} {tag} 실패: {(r.stderr or r.stdout)[-500:]}")
    return json.load(open(o))


REAL, NC, NS_ = {}, {}, {}
for arm in ("N0", "N1"):
    w = f"{AUD[arm]['dir']}/weights/last.pt"
    REAL[arm] = run(("REAL", "cf_real_eval.py"), f"ADAPT_{arm}", w)
    NC[arm] = run(("NIGHT_CAND", "night_cand_one.py"), f"ADAPT_{arm}", w)
    NS_[arm] = run(("NEGSCORE", "neg_eval_one.py"), f"ADAPT_{arm}", w)
    log(f"  {arm} eval 완료")


def agg(rs):
    cb = [r for r in rs if r.get("correct_box")]
    e = np.concatenate([r["err"] for r in cb]) if cb else np.array([])
    return {"n": len(rs), "cbox": len(cb) / max(len(rs), 1),
            "med": float(np.median(e)) if e.size else None,
            "p90": float(np.percentile(e, 90)) if e.size else None,
            "gross": float((e > 20).mean()) if e.size else None}


TAB = {}
for arm, d in REAL.items():
    pf = [r for r in d["per_frame"] if r["frame"] not in LEAK]
    TAB[arm] = {"ALL": agg(pf), "DAY": agg([r for r in pf if r["domain"] == "DAY"]),
                "NIGHT": agg([r for r in pf if r["domain"] == "NIGHT"])}


def negmetrics(rows):
    y = np.array([r["label"] for r in rows]); s = np.array([r["max_conf"] for r in rows])
    o = np.argsort(-s); ys = y[o]
    P, N = ys.sum(), (1 - ys).sum()
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    rec, fpr = tp / max(P, 1), fp / max(N, 1)
    prec = tp / np.maximum(tp + fp, 1)
    AP = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    AUROC = float(np.trapz(rec, fpr))
    i95 = int(np.searchsorted(rec, 0.95))
    FPR95 = float(fpr[min(i95, len(fpr) - 1)]) if len(fpr) else None
    pos = np.array([r["max_conf"] for r in rows if r["label"] == 1])
    neg = np.array([r["max_conf"] for r in rows if r["label"] == 0])
    negrows = [r for r in rows if r["label"] == 0]
    ncand = np.array([r["n_candidates"] for r in negrows])
    d = {"n_pos": int(pos.size), "n_neg": int(neg.size), "AP": AP, "AUROC": AUROC,
         "FPR@TPR95": FPR95, "pos_p10": float(np.percentile(pos, 10)),
         "pos_median": float(np.median(pos)), "neg_median": float(np.median(neg)),
         "neg_p90": float(np.percentile(neg, 90)), "neg_p95": float(np.percentile(neg, 95)),
         "neg_p99": float(np.percentile(neg, 99)),
         "separation_margin": float(np.percentile(pos, 10) - np.percentile(neg, 90)),
         "neg_cand_mean": float(ncand.mean()), "det": {}}
    for t in (0.05, 0.25, 0.40):
        d["det"][str(t)] = {
            "neg_detect_rate": float((neg >= t).mean()),
            "FP_per_image": float(np.mean([sum(1 for c in r["confs"] if c >= t) for r in negrows])),
            "recall_TPR": float((pos >= t).mean())}
    return d


NM = {a: negmetrics(NS_[a]["rows"]) for a in ("N0", "N1")}
n0, n1 = NM["N0"], NM["N1"]
t0, t1 = TAB["N0"], TAB["N1"]
c0, c1 = NC["N0"], NC["N1"]
NN = c0["n"]

D = {"d_AP": n1["AP"] - n0["AP"],
     "d_FPR_at_TPR95": n1["FPR@TPR95"] - n0["FPR@TPR95"],
     "d_neg_detect_rate_040": n1["det"]["0.4"]["neg_detect_rate"] - n0["det"]["0.4"]["neg_detect_rate"],
     "d_FP_per_image_040": n1["det"]["0.4"]["FP_per_image"] - n0["det"]["0.4"]["FP_per_image"],
     "d_neg_cand_mean_rel": (n1["neg_cand_mean"] - n0["neg_cand_mean"]) / max(n0["neg_cand_mean"], 1e-9),
     "d_neg_p90": n1["neg_p90"] - n0["neg_p90"]}
G = {"d_all_cbox_pp": t1["ALL"]["cbox"] - t0["ALL"]["cbox"],
     "d_night_any_cbox_frames": (c1["any_cbox"] - c0["any_cbox"]) * NN,
     "d_all_median_rel_worse": (t1["ALL"]["med"] - t0["ALL"]["med"]) / max(t0["ALL"]["med"], 1e-9)}
hits = {k: (D[k] >= v["value"] if v["op"] == ">=" else D[k] <= v["value"])
        for k, v in GATE["negative_hits"].items()}
guards = {k: (G[k] >= v["value"] if v["op"] == ">=" else G[k] <= v["value"])
          for k, v in GATE["positive_guards_all_must_hold"].items()}
nhit = sum(hits.values())
SIG = ("HARM" if not all(guards.values())
       else ("POSITIVE" if nhit >= GATE["negative_hits_need"] else "NULL"))

out = {"gate": GATE, "DELTA_NEG": D, "positive_guard_values": G, "hits": hits,
       "n_hits": nhit, "guards": guards, "NEG_SIGNAL": SIG,
       "neg_metrics": NM, "real_table": TAB, "night_candidate": NC,
       "runtime_audit": {a: {k: AUD[a][k] for k in
                             ("n_train_batches", "actual_epochs", "PC_CALL_COUNT",
                              "ROLE_CALLS", "criterion", "data", "init")} for a in AUD},
       "init_lock": json.load(open(f"{Q}/G38_ADAPT_INIT_LOCK.json")),
       "contract": json.load(open(f"{Q}/G38_ADAPT_2X2_CONTRACT.json")),
       "split_lock": json.load(open(f"{Q}/G38_ADAPT_DATA_SPLIT_LOCK.json")),
       "★scope": ("host-internal DELTA 만 유효. Windows P0/P1 과 absolute 비교 금지. "
                  "NEG heldout = 2,689 전체 (FT negative 259 와 내용 교집합 0 실측). "
                  "NIGHT n=28, seed 1, 15ep.")}
json.dump(out, open(f"{Q}/UBUNTU_NEG_SCREEN_RESULT.json", "w"), indent=2, ensure_ascii=False)

f2 = lambda z: "     n/a" if z is None else f"{z:8.3f}"
L = ["# UBUNTU NEGATIVE SCREEN — N0 vs N1 (15ep, exposure 13,554)", "", "## SAME REAL n=128", "```",
     f"{'':16} {'N0':>10} {'N1':>10} {'Δ':>10}", "-" * 50]
for sc in ("ALL", "DAY", "NIGHT"):
    for k in ("cbox", "med", "p90", "gross"):
        a, b = TAB["N0"][sc][k], TAB["N1"][sc][k]
        L.append(f"{sc+' '+k:16} {f2(a)[-10:]:>10} {f2(b)[-10:]:>10} "
                 f"{('       n/a' if None in (a,b) else f'{b-a:+10.3f}')}")
    L.append("")
L += ["```", "## NIGHT candidate", "```", f"{'':16} {'N0':>10} {'N1':>10}", "-" * 40]
for k in ("any_cbox", "top1_cbox", "cand_per_frame", "wrong_present_frac", "margin_median"):
    L.append(f"{k:16} {c0.get(k,float('nan')):10.3f} {c1.get(k,float('nan')):10.3f}")
L += ["```", "## NEG HELD-OUT n=2,689", "```", f"{'':22} {'N0':>10} {'N1':>10} {'Δ':>10}", "-" * 56]
for k in ("AP", "AUROC", "FPR@TPR95", "neg_median", "neg_p90", "neg_p95", "neg_p99",
          "neg_cand_mean", "separation_margin"):
    L.append(f"{k:22} {n0[k]:10.4f} {n1[k]:10.4f} {n1[k]-n0[k]:+10.4f}")
for t in ("0.05", "0.25", "0.4"):
    for k in ("neg_detect_rate", "FP_per_image", "recall_TPR"):
        L.append(f"{k+'@'+t:22} {n0['det'][t][k]:10.4f} {n1['det'][t][k]:10.4f} "
                 f"{n1['det'][t][k]-n0['det'][t][k]:+10.4f}")
L += ["```", "", f"hits {nhit}/6 → {hits}", f"guards {guards}", "",
      f"**NEG_SIGNAL = {SIG}**", "",
      f"batches/ep N0 {AUD['N0']['n_train_batches']} = N1 {AUD['N1']['n_train_batches']} · "
      f"{AUD['N0']['actual_epochs']}ep · init sha {out['init_lock']['G38_INIT_SHA256'][:16]}", "",
      "★ host-internal DELTA 만 유효 — Windows P0/P1 과 absolute 비교 금지.",
      "★ NEG heldout = 2,689 전체. 'FT 학습분 259 제외 2,430' 은 실측상 틀렸다(내용 교집합 0).",
      "★ NIGHT n=28, seed 1, 15ep FAST SCREEN."]
txt = "\n".join(L)
open(f"{Q}/UBUNTU_NEG_SCREEN_RESULT.md", "w").write(txt)
print(txt); notify(txt[:1800])
log(f"=== NEG SCREEN 완료 {SIG} (hits {nhit}/6) ===")
if os.path.exists(LOCK): os.remove(LOCK)
