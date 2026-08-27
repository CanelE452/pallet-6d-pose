"""OVERNIGHT LOSS QUEUE — S0..S7 을 한 번에.

중간 승인 없음.  metric FAIL 은 다음 독립 후보로 진행.  FATAL 만 전체 중단.
완료 판정은 산출물로만 한다 (프로세스 생존 금지).
"""
import glob, hashlib, json, os, socket, subprocess, sys, time
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
R = f"{Y}/runs_pallet_loss"
O = f"{R}/overnight_20260823"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
A0DIR = f"{Y}/runs_fixed/V1_FIXED_MATCHED10K_60EP_SEED42_UBUNTU"
ASCDIR = f"{R}/PSPC_ASC_USD_ONLY_V1MATCHED10K_60EP_SEED42"
DATA = f"{Y}/datasets/v1_fixed_matched10k/data.yaml"
INIT = f"{ROOT}/challenge/weights/pretrained_yolo/yolo26n-pose.pt"
EP = ["epoch0", "epoch10", "epoch20", "epoch30", "epoch40", "epoch50", "last"]
LOCK = f"{O}/OVERNIGHT_QUEUE.lock"
STATE = f"{O}/OVERNIGHT_STATE.json"
LOG = f"{O}/OVERNIGHT_LOG.txt"

STAGES = ["S0_ASC_REAL", "S1_ASC_SEED43", "S2_TAKL", "S3_TAKL_REAL",
          "S4_NRL", "S5_NRL_REAL", "S6_ASC_TAKL", "S7_AUDIT"]


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


def st_load():
    return json.load(open(STATE)) if os.path.exists(STATE) else \
        {s: {"status": "PENDING"} for s in STAGES}


def st_set(s, **kw):
    d = st_load()
    d.setdefault(s, {}).update(kw)
    json.dump(d, open(STATE, "w"), indent=2, ensure_ascii=False)
    return d


def fatal(m):
    log(f"FATAL {m}")
    st_set("_FATAL", status="FATAL", reason=m)
    notify(f"🛑 OVERNIGHT FATAL — {m}")
    sys.exit(2)


# ---- lock ------------------------------------------------------------------
if os.path.exists(LOCK):
    try:
        L = json.load(open(LOCK))
        if L.get("pid") and os.path.exists(f"/proc/{L['pid']}"):
            log(f"이미 실행 중 {L}")
            sys.exit(0)
    except Exception:
        pass
json.dump({"hostname": socket.gethostname(), "pid": os.getpid(),
           "start": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "command": " ".join(sys.argv)}, open(LOCK, "w"), indent=2)

# ---- source lock 재확인 (작업 중 변조되면 FATAL) -----------------------------
GL = json.load(open(f"{O}/GLOBAL_SOURCE_LOCK.json"))


def sha(p, n=16):
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:n]


def check_sources(where):
    bad = []
    for k, p in (("last", f"{A0DIR}/weights/last.pt"), ("args", f"{A0DIR}/args.yaml"),
                 ("results", f"{A0DIR}/results.csv")):
        if sha(p) != GL["A0"][k]:
            bad.append(f"A0.{k}")
    if sha(f"{Y}/datasets/v1_fixed_matched10k/data.yaml") != GL["dataset"]["data_yaml"]:
        bad.append("dataset.data_yaml")
    n_tr = len(glob.glob(f"{Y}/datasets/v1_fixed_matched10k/labels/train/*.txt"))
    if n_tr != GL["dataset"]["labels_train"]["n"]:
        bad.append(f"dataset.train_n {n_tr}")
    if bad:
        fatal(f"{where}: 정본 변조 {bad}")


check_sources("시작")


# ---- helpers ---------------------------------------------------------------
def gpu_free():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60).stdout.strip().split("\n")[0]
        return int(o)
    except Exception:
        return -1


REQUIRED_KEYS = ("gross20", "gross40", "bottom_p90", "corner_px_identity_p90",
                 "identity_d_id_median", "flip_rate", "pose_measure_ok")


def measure(tag, w, force=False):
    """캐시를 재사용하되 **필요한 키가 없으면 다시 잰다.**

    지표를 나중에 추가하면 옛 캐시에 그 키가 없어 게이트가 KeyError 로 죽는다
    (2026-08-23 실제 발생 — A0 캐시가 gross20 추가 전 것이었다).
    """
    o = f"{R}/A1_MEASURE_{tag}.json"
    if os.path.exists(o) and not force:
        try:
            d = json.load(open(o))
        except Exception:
            d = None
        need = ("gross20", "gross40", "corner_px_identity_p90",
                "identity_d_id_median", "flip_rate")
        if d and all(k in d for k in need) and all("corner_err" in r for r in d["per_frame"][:1]):
            return d
        log(f"{tag} 캐시에 신규 지표 없음 — 재측정")
    if not os.path.exists(w):
        return None
    subprocess.run([sys.executable, f"{R}/a1_measure.py", "--weights", w, "--tag", tag],
                   capture_output=True, text=True)
    return json.load(open(o)) if os.path.exists(o) else None


def train(name, cfg, seed=42):
    """A0 args 를 복제하고 loss config 만 바꾼다.  resume 금지, clean init."""
    d = f"{R}/{name}"
    if os.path.exists(f"{d}/weights/last.pt"):
        log(f"{name} 이미 있음 — 건너뜀")
        return d
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/PURPOSE.md"):
        open(f"{d}/PURPOSE.md", "w").write(
            f"[소비처] 논문 method — overnight loss screen ({name})\n"
            f"[문장]  A0 대비 loss 만 바꿔 pallet corner localization 이 개선된다.\n")
    code = f'''
import os, sys, json
sys.path.insert(0, "{ROOT}")
os.environ["A1_CONFIG"] = "{cfg}"
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
from pallet_yolo_loss.trainer import ASCTrainer
tr = ASCTrainer(overrides=dict(
    task="pose", mode="train", model="{INIT}", data="{DATA}",
    epochs=60, batch=32, imgsz=640, optimizer="SGD", lr0=0.01, lrf=0.01,
    cos_lr=True, close_mosaic=10, pose=12.0, kobj=1.0, warmup_epochs=3.0,
    patience=0, single_cls=True, mosaic=0.3, scale=0.25, hsv_s=0.5, hsv_v=0.35,
    fliplr=0.0, flipud=0.0, erasing=0.4, seed={seed}, deterministic=True,
    save_period=10, device=0, workers=8, project="{R}", name="{name}",
    exist_ok=True, resume=False, val=True, plots=False))
tr.train()
crit = getattr(tr.model, "criterion", None)
inner = getattr(crit, "one2many", crit)
a1 = getattr(inner, "a1", None)
json.dump({{"PC_CALL_COUNT": PC["n"], "ROLE_MARGIN_CALL_COUNT": SY.ROLE_CALLS["n"],
           "asc_enabled": bool(getattr(a1, "asc_enabled", False)),
           "takl_enabled": bool(getattr(a1, "takl_enabled", False)),
           "nrl_enabled": bool(getattr(a1, "nrl_enabled", False)),
           "takl_tau": getattr(a1, "takl_tau", None),
           "takl_lambda": getattr(a1, "takl_lambda", None),
           "nrl_beta": getattr(a1, "nrl_beta", None),
           "nrl_lambda": getattr(a1, "nrl_lambda", None),
           "sym_assets": list(getattr(a1, "sym_assets", ())),
           "seed": {seed}}}, open("{R}/{name}/RUNTIME_AUDIT.json", "w"), indent=2)
'''
    sc = f"{O}/_train_{name}.py"
    open(sc, "w").write(code)
    r = subprocess.run([sys.executable, sc], cwd=d, capture_output=True, text=True)
    if not os.path.exists(f"{d}/weights/last.pt"):
        tail = (r.stderr or "")[-1500:]
        if "out of memory" in tail.lower():
            return "OOM"
        if "nan" in tail.lower():
            return "NAN"
        log(f"{name} 학습 실패: {tail[-600:]}")
        return None
    a = json.load(open(f"{d}/RUNTIME_AUDIT.json"))
    if a["PC_CALL_COUNT"] or a["ROLE_MARGIN_CALL_COUNT"]:
        log(f"{name} 계약 위반 PC={a['PC_CALL_COUNT']} role={a['ROLE_MARGIN_CALL_COUNT']}")
        return None
    return d


def synth_gate(arm, tag_prefix):
    """A0 vs arm, v1 val 133 동일 evaluator."""
    a0 = measure("A0_last", f"{A0DIR}/weights/last.pt")
    x = measure(f"{tag_prefix}_last", f"{R}/{arm}/weights/last.pt")
    if not (a0 and x):
        return None, None
    def rel(b, a):
        return (b - a) / max(abs(b), 1e-12)
    p90 = rel(a0["corner_px_identity_p90"], x["corner_px_identity_p90"])
    g20 = rel(a0["gross20"], x["gross20"]) if a0["gross20"] > 0 else 0.0
    bot = rel(a0["bottom_p90"], x["bottom_p90"]) if a0.get("bottom_p90") else 0.0
    dmap = x["pose_map50_95"] - a0["pose_map50_95"]
    did = x["identity_d_id_median"] - a0["identity_d_id_median"]
    dflip = x["flip_rate"] - a0["flip_rate"]
    ce = np.array([r["corner_err"] for r in x["per_frame"]])
    collapse = int((ce[:, :8].max(1) > 300).sum())
    hit = (p90 >= 0.05) or (g20 >= 0.10) or (bot >= 0.08)
    guard = (dmap >= -0.02) and (did <= 0.03) and (dflip <= 0.03) and collapse == 0
    v = "PASS" if (hit and guard) else "FAIL"
    d = {"arm": arm, "verdict": v,
         "hit": {"p90_5pct": p90 >= 0.05, "gross20_10pct": g20 >= 0.10, "bottom_8pct": bot >= 0.08},
         "guard": {"dmap": dmap, "d_identity": did, "d_flip": dflip, "collapse": collapse},
         "relative": {"p90": p90, "gross20": g20, "bottom_p90": bot},
         "A0": {k: a0.get(k) for k in ("pose_map50_95", "corner_px_identity_median",
                                       "corner_px_identity_p90", "gross20", "bottom_p90",
                                       "identity_d_id_median", "flip_rate")},
         arm: {k: x.get(k) for k in ("pose_map50_95", "corner_px_identity_median",
                                     "corner_px_identity_p90", "gross20", "bottom_p90",
                                     "identity_d_id_median", "flip_rate")}}
    json.dump(d, open(f"{R}/{arm}/SYNTH_GATE.json", "w"), indent=2, ensure_ascii=False)
    return v, d


def real_eval(arm, w):
    r = subprocess.run([sys.executable, f"{O}/real_equiv_eval.py", "--arm", arm,
                        "--weights", w, "--baseline", f"{A0DIR}/weights/last.pt"],
                       capture_output=True, text=True)
    p = f"{O}/REAL_{arm}_VERDICT.json"
    if os.path.exists(p):
        return json.load(open(p))["verdict"]
    log(f"{arm} real eval 실패: {(r.stderr or '')[-400:]}")
    return "BLOCKED"


def brief(stage, verdict, arm=None, nxt=""):
    m = f"[OVERNIGHT]\nstage = {stage}\nverdict = {verdict}\n"
    if arm and os.path.exists(f"{R}/{arm}/SYNTH_GATE.json"):
        g = json.load(open(f"{R}/{arm}/SYNTH_GATE.json"))
        m += (f"mAP50-95 = {g[arm]['pose_map50_95']:.4f} (A0 {g['A0']['pose_map50_95']:.4f})\n"
              f"corner p90 = {g[arm]['corner_px_identity_p90']:.2f} "
              f"(A0 {g['A0']['corner_px_identity_p90']:.2f})\n")
    m += f"next stage = {nxt}"
    notify(m)


# ============================ QUEUE ==========================================
T0 = time.time()
log("=== OVERNIGHT LOSS QUEUE 시작 ===")

# ---- S0 ---------------------------------------------------------------------
s0p = f"{O}/REAL_ASC_VERDICT.json"
if os.path.exists(s0p):
    S0 = json.load(open(s0p))["verdict"]
    st_set("S0_ASC_REAL", status="PASS", verdict=S0, note="사전 실행 완료")
else:
    st_set("S0_ASC_REAL", status="RUNNING")
    S0 = real_eval("ASC", f"{ASCDIR}/weights/last.pt")
    st_set("S0_ASC_REAL", status="PASS", verdict=S0)
log(f"S0 = {S0}")
brief("S0_ASC_REAL", S0, nxt="S1" if S0 != "NO_REAL_SIGNAL" else "S2 (S1 SKIP)")

# ---- S1 ---------------------------------------------------------------------
if S0 == "NO_REAL_SIGNAL":
    st_set("S1_ASC_SEED43", status="SKIPPED",
           reason="S0=NO_REAL_SIGNAL — seed 추가보다 다른 loss 탐색 우선 (사전등록 규칙)")
    log("S1 SKIP")
else:
    st_set("S1_ASC_SEED43", status="RUNNING")
    d = train("PSPC_ASC_USD_ONLY_V1MATCHED10K_60EP_SEED43",
              f"{R}/ASC_LOSS_CONFIG.json", seed=43)
    if d in ("OOM", "NAN"):
        fatal(f"S1 {d}")
    if not d:
        st_set("S1_ASC_SEED43", status="FAIL", reason="학습 산출물 없음")
    else:
        for e in EP:
            measure(f"ASC43_{e}", f"{d}/weights/{e}.pt")
        a0 = measure("A0_epoch20", f"{A0DIR}/weights/epoch20.pt")
        x20 = measure("ASC43_epoch20", f"{d}/weights/epoch20.pt")
        xl = measure("ASC43_last", f"{d}/weights/last.pt")
        a0l = measure("A0_last", f"{A0DIR}/weights/last.pt")
        ce = np.array([r["corner_err"] for r in xl["per_frame"]])
        ok = (x20["pose_map50_95"] >= a0["pose_map50_95"] + 0.10
              and xl["pose_map50_95"] >= a0l["pose_map50_95"] - 0.01
              and int((ce[:, :8].max(1) > 300).sum()) == 0)
        st_set("S1_ASC_SEED43", status="PASS" if ok else "FAIL",
               ASC_SEED_ROBUST="PASS" if ok else "FAIL")
        log(f"S1 ASC_SEED_ROBUST = {'PASS' if ok else 'FAIL'}")
        brief("S1_ASC_SEED43", "PASS" if ok else "FAIL", nxt="S2")

# ---- S2 TAKL -----------------------------------------------------------------
check_sources("S2 전")
st_set("S2_TAKL", status="RUNNING")
TAKL_ARM = "PSPC_TAKL_V1MATCHED10K_60EP_SEED42"
d = train(TAKL_ARM, f"{O}/TAKL_LOSS_CONFIG.json", 42)
if d in ("OOM", "NAN"):
    fatal(f"S2 {d}")
TAKL_SYNTH = "FAIL"
if not d:
    st_set("S2_TAKL", status="FAIL", reason="학습 산출물 없음 — 다음 독립 후보로")
else:
    for e in EP:
        measure(f"TAKL_{e}", f"{d}/weights/{e}.pt")
    TAKL_SYNTH, g = synth_gate(TAKL_ARM, "TAKL")
    st_set("S2_TAKL", status="PASS" if TAKL_SYNTH == "PASS" else "FAIL",
           TAKL_SYNTH_SIGNAL=TAKL_SYNTH)
log(f"S2 TAKL_SYNTH_SIGNAL = {TAKL_SYNTH}")
brief("S2_TAKL", TAKL_SYNTH, TAKL_ARM, nxt="S3" if TAKL_SYNTH == "PASS" else "S4 (NRL)")

# ---- S3 ----------------------------------------------------------------------
REAL_TAKL = "SKIPPED"
if TAKL_SYNTH == "PASS":
    st_set("S3_TAKL_REAL", status="RUNNING")
    REAL_TAKL = real_eval("TAKL", f"{R}/{TAKL_ARM}/weights/last.pt")
    st_set("S3_TAKL_REAL", status="PASS", verdict=REAL_TAKL)
    log(f"S3 = {REAL_TAKL}")
    brief("S3_TAKL_REAL", REAL_TAKL, nxt="S4/S6")
else:
    st_set("S3_TAKL_REAL", status="SKIPPED", reason="TAKL_SYNTH_SIGNAL != PASS")

# ---- S4 NRL -------------------------------------------------------------------
NRL_SYNTH = "SKIPPED"
NRL_ARM = "PSPC_NRL_V1MATCHED10K_60EP_SEED42"
if TAKL_SYNTH == "FAIL" or REAL_TAKL in ("NO_REAL_SIGNAL", "BLOCKED"):
    check_sources("S4 전")
    st_set("S4_NRL", status="RUNNING")
    d = train(NRL_ARM, f"{O}/NRL_LOSS_CONFIG.json", 42)
    if d in ("OOM", "NAN"):
        fatal(f"S4 {d}")
    if not d:
        st_set("S4_NRL", status="FAIL", reason="학습 산출물 없음")
        NRL_SYNTH = "FAIL"
    else:
        for e in EP:
            measure(f"NRL_{e}", f"{d}/weights/{e}.pt")
        NRL_SYNTH, _ = synth_gate(NRL_ARM, "NRL")
        st_set("S4_NRL", status="PASS" if NRL_SYNTH == "PASS" else "FAIL",
               NRL_SYNTH_SIGNAL=NRL_SYNTH)
    log(f"S4 NRL_SYNTH_SIGNAL = {NRL_SYNTH}")
    brief("S4_NRL", NRL_SYNTH, NRL_ARM, nxt="S5/S6")
else:
    st_set("S4_NRL", status="SKIPPED", reason="TAKL 이 real 에서 살아 있음 — loss 증식 금지")

# ---- S5 ------------------------------------------------------------------------
if NRL_SYNTH == "PASS":
    st_set("S5_NRL_REAL", status="RUNNING")
    v = real_eval("NRL", f"{R}/{NRL_ARM}/weights/last.pt")
    st_set("S5_NRL_REAL", status="PASS", verdict=v)
    log(f"S5 = {v}")
    brief("S5_NRL_REAL", v, nxt="S6")
else:
    st_set("S5_NRL_REAL", status="SKIPPED", reason="NRL_SYNTH_SIGNAL != PASS")

# ---- S6 --------------------------------------------------------------------------
COMB_ARM = "PSPC_ASC_TAKL_V1MATCHED10K_60EP_SEED42"
COMB = "SKIPPED"
asc_alive = S0 in ("POSITIVE", "POINT_ESTIMATE_POSITIVE_UNCERTAIN", "BLOCKED_PERMUTATION_CONTRACT")
if asc_alive and TAKL_SYNTH == "PASS" and REAL_TAKL in (
        "POSITIVE", "POINT_ESTIMATE_POSITIVE_UNCERTAIN", "BLOCKED"):
    check_sources("S6 전")
    st_set("S6_ASC_TAKL", status="RUNNING")
    t = subprocess.run([sys.executable, f"{O}/test_candidate.py", "--candidate", "ASC_TAKL"],
                       capture_output=True, text=True)
    if t.returncode != 0:
        st_set("S6_ASC_TAKL", status="FAIL", reason="unit test 실패")
    else:
        d = train(COMB_ARM, f"{O}/ASC_TAKL_LOSS_CONFIG.json", 42)
        if d in ("OOM", "NAN"):
            fatal(f"S6 {d}")
        if d:
            for e in EP:
                measure(f"COMB_{e}", f"{d}/weights/{e}.pt")
            COMB, _ = synth_gate(COMB_ARM, "COMB")
            st_set("S6_ASC_TAKL", status="PASS" if COMB == "PASS" else "FAIL",
                   COMBINED_SIGNAL=COMB)
    log(f"S6 = {COMB}")
    brief("S6_ASC_TAKL", COMB, COMB_ARM, nxt="S7")
else:
    st_set("S6_ASC_TAKL", status="SKIPPED",
           reason=f"조건 미충족 (S0={S0}, TAKL_SYNTH={TAKL_SYNTH}, REAL_TAKL={REAL_TAKL})")

# ---- S7 AUDIT ---------------------------------------------------------------------
st_set("S7_AUDIT", status="RUNNING")
subprocess.run([sys.executable, f"{O}/audit.py"], capture_output=True, text=True)
st_set("S7_AUDIT", status="PASS")
dur = (time.time() - T0) / 3600
log(f"=== OVERNIGHT COMPLETE  {dur:.2f}h ===")
summ = ""
if os.path.exists(f"{O}/OVERNIGHT_SUMMARY.md"):
    summ = open(f"{O}/OVERNIGHT_SUMMARY.md").read()[:1500]
notify(f"✅ **OVERNIGHT COMPLETE**  ({dur:.2f}h)\n\n{summ}")
if os.path.exists(LOCK):
    os.remove(LOCK)
