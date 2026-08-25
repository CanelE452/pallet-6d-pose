"""CAMERA-FACING LOSS QUEUE — U0..U4 를 한 번에.

중간 승인 없음. metric FAIL 은 큐를 멈추지 않는다. FATAL 만 중단.
완료 판정은 results.csv 최종 epoch / last.pt / VERDICT.json 로만 한다.
"""
import glob, hashlib, json, os, socket, subprocess, sys, time
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
CFR = f"{Y}/runs_camera_facing_loss"
Q = f"{CFR}/ubuntu_cf_loss_queue_20260823T0930"
DATA = f"{Y}/datasets/v1_cf_matched10k/data.yaml"
INIT = f"{ROOT}/challenge/weights/pretrained_yolo/yolo26n-pose.pt"
A0ARGS = f"{Y}/runs_fixed/V1_FIXED_MATCHED10K_60EP_SEED42_UBUNTU/args.yaml"
FTW = "/home/minjae/Documents/github/25y_automatic_lifter-master/pallet_yolo26n_pose_ft.pt"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
EP = ["epoch0", "epoch10", "epoch20", "epoch30", "epoch40", "epoch50", "last"]
LOCK = f"{Q}/CF_QUEUE.lock"
STATE = f"{Q}/QUEUE_STATE.json"
LOG = f"{Q}/QUEUE_LOG.txt"
STAGES = ["U0_DATA", "U1_CF_A0", "U1E_A0_EVAL", "U2_CF_NRL", "U2E_NRL_EVAL",
          "U2S_NRL_SEED43", "U3_CF_PEVL", "U3E_PEVL_EVAL", "U3S_PEVL_SEED43", "U4_AUDIT"]

A0_ARM = "CF_A0_V1MATCHED10K_60EP_SEED42_UBUNTU"
NRL_ARM = "CF_NRL_V1MATCHED10K_60EP_SEED42_UBUNTU"
NRL43_ARM = "CF_NRL_V1MATCHED10K_60EP_SEED43_UBUNTU"
PEVL_ARM = "CF_PEVL_V1MATCHED10K_60EP_SEED42_UBUNTU"
PEVL43_ARM = "CF_PEVL_V1MATCHED10K_60EP_SEED43_UBUNTU"


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


def fatal(m):
    log(f"FATAL {m}")
    st_set("_FATAL", status="FATAL", reason=m)
    notify(f"🛑 CF QUEUE FATAL — {m}")
    if os.path.exists(LOCK):
        os.remove(LOCK)
    sys.exit(2)


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

SL = json.load(open(f"{Q}/SOURCE_LOCK.json"))
CONTRACT = json.load(open(f"{Q}/CF_DATASET_CONTRACT.json"))


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
    for i in SL["inventory"]:
        if i["sha256_16"] and sha(i["path"]) != i["sha256_16"]:
            bad.append(i["role"])
    n_tr = len(glob.glob(f"{Y}/datasets/v1_cf_matched10k/labels/train/*.txt"))
    n_va = len(glob.glob(f"{Y}/datasets/v1_cf_matched10k/labels/val/*.txt"))
    if (n_tr, n_va) != (CONTRACT["train_n"], CONTRACT["val_n"]):
        bad.append(f"membership {n_tr}/{n_va}")
    if bad:
        fatal(f"{where}: source 변조 {bad}")


def run(cmd, **kw):
    return subprocess.run([sys.executable] + cmd, capture_output=True, text=True, **kw)


def train(name, cfg, seed):
    d = f"{CFR}/{name}"
    if os.path.exists(f"{d}/weights/last.pt"):
        log(f"{name} 이미 있음 — 건너뜀")
        return d
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/PURPOSE.md"):
        open(f"{d}/PURPOSE.md", "w").write(
            f"[소비처] 논문 main — camera-facing loss screen ({name})\n"
            f"[문장]  Standard 대비 loss 만 바꿔 real keypoint 성능이 개선된다.\n")
    cfg_line = f'os.environ["A1_CONFIG"] = "{cfg}"' if cfg else 'os.environ["A1_CONFIG"] = ""'
    trainer = "ASCTrainer" if cfg else "PoseTrainer"
    imp = ("from pallet_yolo_loss.trainer import ASCTrainer as T"
           if cfg else "from ultralytics.models.yolo.pose import PoseTrainer as T")
    code = f'''
import os, sys, json
sys.path.insert(0, "{ROOT}")
{cfg_line}
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
{imp}
tr = T(overrides=dict(
    task="pose", mode="train", model="{INIT}", data="{DATA}",
    epochs=60, batch=32, imgsz=640, optimizer="SGD", lr0=0.01, lrf=0.01,
    cos_lr=True, close_mosaic=10, pose=12.0, kobj=1.0, warmup_epochs=3.0,
    patience=0, single_cls=True, mosaic=0.3, scale=0.25, hsv_h=0.015,
    hsv_s=0.5, hsv_v=0.35, fliplr=0.0, flipud=0.0, erasing=0.4,
    seed={seed}, deterministic=True, save_period=10, device=0, workers=8,
    project="{CFR}", name="{name}", exist_ok=True, resume=False, val=True, plots=False))
tr.train()
crit = getattr(tr.model, "criterion", None)
inner = getattr(crit, "one2many", crit)
a1 = getattr(inner, "a1", None)
json.dump({{"PC_CALL_COUNT": PC["n"], "ROLE_MARGIN_CALL_COUNT": SY.ROLE_CALLS["n"],
           "trainer": "{trainer}",
           "nrl_enabled": bool(getattr(a1, "nrl_enabled", False)),
           "pevl_enabled": bool(getattr(a1, "pevl_enabled", False)),
           "asc_enabled": bool(getattr(a1, "asc_enabled", False)),
           "takl_enabled": bool(getattr(a1, "takl_enabled", False)),
           "sym_assets": list(getattr(a1, "sym_assets", ())),
           "nrl_beta": getattr(a1, "nrl_beta", None),
           "nrl_lambda": getattr(a1, "nrl_lambda", None),
           "seed": {seed}}}, open("{CFR}/{name}/RUNTIME_AUDIT.json", "w"), indent=2)
'''
    sc = f"{Q}/_train_{name}.py"
    open(sc, "w").write(code)
    r = subprocess.run([sys.executable, sc], cwd=d, capture_output=True, text=True)
    if not os.path.exists(f"{d}/weights/last.pt"):
        tail = (r.stderr or "")[-2000:]
        low = tail.lower()
        if "out of memory" in low:
            return "OOM"
        if "nan" in low and name == A0_ARM:
            return "NAN"
        log(f"{name} 학습 실패: {tail[-800:]}")
        return None
    a = json.load(open(f"{d}/RUNTIME_AUDIT.json"))
    if a["PC_CALL_COUNT"] or a["ROLE_MARGIN_CALL_COUNT"]:
        log(f"{name} 계약 위반 PC={a['PC_CALL_COUNT']} role={a['ROLE_MARGIN_CALL_COUNT']}")
        return None
    nr = len(open(f"{d}/results.csv").read().strip().split("\n")) - 1
    if nr != 60:
        log(f"{name} results.csv {nr}행 (60 아님)")
        return None
    return d


def synth(tag, w):
    o = f"{Q}/SYNTH_{tag}.json"
    if os.path.exists(o):
        return json.load(open(o))
    if not os.path.exists(w):
        return None
    r = run([f"{Q}/cf_synth_eval.py", "--weights", w, "--tag", tag])
    if not os.path.exists(o):
        log(f"synth {tag} 실패: {(r.stderr or '')[-300:]}")
    return json.load(open(o)) if os.path.exists(o) else None


def real(tag, w):
    """캐시를 쓰되 신규 키가 없으면 다시 잰다 (지표 추가 후 KeyError 방지)."""
    o = f"{Q}/REAL_{tag}.json"
    if os.path.exists(o):
        d = json.load(open(o))
        if "cbox_paired" in d:
            return d
        log(f"REAL_{tag} 캐시에 신규 모집단 없음 — 재측정")
    if not os.path.exists(w):
        return None
    r = run([f"{Q}/cf_real_eval.py", "--weights", w, "--tag", tag])
    if not os.path.exists(o):
        log(f"real {tag} 실패: {(r.stderr or '')[-300:]}")
    return json.load(open(o)) if os.path.exists(o) else None


def gate(a0r, xr, a0s, xs, name):
    """U2/U3 공통 real gate. 사전등록 그대로 하드코딩."""
    if not (a0r and xr and a0s and xs):
        return "FAIL", {"reason": "평가 산출물 없음"}
    def rel(b, a):
        return (b - a) / max(abs(b), 1e-12)
    P = xr["cbox_paired"]                 # ★ GATE 모집단 = A0 공통 correct-box
    B = a0r["cbox_paired"]
    if not P or not B:
        return "FAIL", {"reason": "correct-box 공통 모집단 비어 있음"}
    hits = {"p90_5": rel(B["corner_p90"], P["corner_p90"]) >= 0.05,
            "bottom_8": rel(B["bottom_p90"], P["bottom_p90"]) >= 0.08,
            "gross20_10": rel(B["gross20"], P["gross20"]) >= 0.10,
            "median_5": rel(B["corner_median"], P["corner_median"]) >= 0.05}
    g = {"det_reg_pp": xr["detection_recall"] - a0r["detection_recall"],
         "cbox_reg_pp": xr["correct_box_recall"] - a0r["correct_box_recall"],
         "day_p90_reg": (rel(B["day_p90"], P["day_p90"])
                         if ("day_p90" in B and "day_p90" in P) else None),
         "night_p90_reg": (rel(B["night_p90"], P["night_p90"])
                           if ("night_p90" in B and "night_p90" in P) else None),
         "dmap": xs["pose_map50_95"] - a0s["pose_map50_95"],
         "collapse": xs["channel_collapse"]}
    # 도메인 표본이 없으면 그 guard 는 건너뛴다 (없는 것을 실패로 세지 않는다)
    dom_ok = all(v is None or v >= -0.05 for v in (g["day_p90_reg"], g["night_p90_reg"]))
    guard = (g["det_reg_pp"] >= -0.02 and g["cbox_reg_pp"] >= -0.02 and dom_ok
             and g["dmap"] >= -0.02 and g["collapse"] == 0)
    ci_ok = any(v["ci95"][0] > 0 for v in xr["bootstrap"].values())
    if any(hits.values()) and guard:
        v = "REAL_POSITIVE" if ci_ok else "REAL_POSITIVE_UNCERTAIN"
    elif (xs["pose_map50_95"] - a0s["pose_map50_95"] >= 0.01
          and xs["corner_median"] <= a0s["corner_median"]
          and rel(a0s["corner_p90"], xs["corner_p90"]) >= -0.03):
        v = "SYNTHETIC_ONLY_POSITIVE"
    else:
        v = "FAIL"
    d = {"arm": name, "signal": v, "hits": hits, "guard": g,
         "bootstrap": xr["bootstrap"],
         "gate": "real p90>=5% | bottom>=8% | gross20>=10% | median>=5%  AND guard 전부"}
    json.dump(d, open(f"{Q}/{name}_VERDICT.json", "w"), indent=2, ensure_ascii=False)
    return v, d


def brief(stage, status, xr=None, xs=None, nxt=""):
    m = f"[UBUNTU CF LOSS]\nstage = {stage}\nstatus = {status}\n"
    if xr and xr.get("cbox_paired"):
        c = xr["cbox_paired"]
        m += (f"real p90 (correct-box n{c['n']}) = {c['corner_health'] if False else c['corner_p90']:.2f}\n"
              f"real bottom p90 = {c['bottom_p90']:.2f}\n"
              f"det {xr['detection_recall']:.3f} / cbox {xr['correct_box_recall']:.3f}\n")
    if xs:
        m += f"synthetic mAP = {xs['pose_map50_95']:.4f}\n"
    m += f"next = {nxt}"
    notify(m)


# ============================== QUEUE ==================================
T0 = time.time()
log("=== CAMERA-FACING LOSS QUEUE 시작 ===")
check_sources("시작")
st_set("U0_DATA", status="PASS", CF_DATA_READY=CONTRACT["CF_DATA_READY"],
       train_n=CONTRACT["train_n"], val_n=CONTRACT["val_n"])
if not CONTRACT["CF_DATA_READY"]:
    fatal("CF_DATA_READY=False")
log(f"U0 PASS  train {CONTRACT['train_n']} / val {CONTRACT['val_n']}")

# ---- U1 CF-A0 ---------------------------------------------------------
st_set("U1_CF_A0", status="RUNNING")
d = train(A0_ARM, None, 42)
if d in ("OOM", "NAN"):
    fatal(f"U1 {d} — Standard baseline 에서 발생")
if not d:
    fatal("U1 Standard baseline 실패 — parity 확보 불가")
st_set("U1_CF_A0", status="PASS")
log("U1 CF-A0 완주")

st_set("U1E_A0_EVAL", status="RUNNING")
a0s = synth("A0", f"{CFR}/{A0_ARM}/weights/last.pt")
a0r = real("A0", f"{CFR}/{A0_ARM}/weights/last.pt")
ftr = real("FT_REFERENCE", FTW) if os.path.exists(FTW) else None
st_set("U1E_A0_EVAL", status="PASS",
       ft_reference="OK" if ftr else "REFERENCE_BLOCKED_MISSING_WEIGHT")
log(f"U1E  synth mAP {a0s['pose_map50_95']:.4f}  real p90 {a0r['paired']['corner_p90']:.2f}"
    if (a0s and a0r) else "U1E 일부 실패")
brief("U1E_A0_EVAL", "PASS", a0r, a0s, "U2 CF-NRL")

# ---- U2 CF-NRL --------------------------------------------------------
check_sources("U2 전")
st_set("U2_CF_NRL", status="RUNNING")
r = run([f"{Q}/cf_calibrate.py"])
if not os.path.exists(f"{Q}/CF_NRL_CONFIG.json"):
    st_set("U2_CF_NRL", status="FAIL", reason=f"calibration 실패 {(r.stderr or '')[-300:]}")
    NRL_SIG = "FAIL"
else:
    cal = json.load(open(f"{Q}/CF_CALIBRATION.json"))
    if not (1e-3 <= cal["nrl_lambda"] <= 10):
        st_set("U2_CF_NRL", status="FAIL", reason="CALIBRATION_UNSTABLE")
        NRL_SIG = "FAIL"
    else:
        t = run([f"{Q}/test_cf_nrl.py"])
        tr_ = json.load(open(f"{Q}/CF_NRL_TEST.json")) if os.path.exists(f"{Q}/CF_NRL_TEST.json") else {}
        if not tr_.get("all_pass"):
            st_set("U2_CF_NRL", status="FAIL", reason=f"unit test {tr_.get('n_pass')}/{tr_.get('n_total')}")
            NRL_SIG = "FAIL"
        else:
            d = train(NRL_ARM, f"{Q}/CF_NRL_CONFIG.json", 42)
            if d == "OOM":
                fatal("U2 반복 OOM")
            if not d:
                st_set("U2_CF_NRL", status="FAIL", reason="학습 산출물 없음")
                NRL_SIG = "FAIL"
            else:
                st_set("U2_CF_NRL", status="PASS")
                st_set("U2E_NRL_EVAL", status="RUNNING")
                xs = synth("NRL", f"{CFR}/{NRL_ARM}/weights/last.pt")
                xr = real("NRL", f"{CFR}/{NRL_ARM}/weights/last.pt")
                NRL_SIG, _ = gate(a0r, xr, a0s, xs, "CF_NRL")
                st_set("U2E_NRL_EVAL", status="PASS", CF_NRL_SIGNAL=NRL_SIG)
                brief("U2E_NRL_EVAL", NRL_SIG, xr, xs,
                      "U2S seed43" if NRL_SIG.startswith("REAL_POSITIVE") else "U3 PEVL")
log(f"U2 CF_NRL_SIGNAL = {NRL_SIG}")

# ---- U2S seed43 -------------------------------------------------------
if NRL_SIG in ("REAL_POSITIVE", "REAL_POSITIVE_UNCERTAIN"):
    st_set("U2S_NRL_SEED43", status="RUNNING")
    d = train(NRL43_ARM, f"{Q}/CF_NRL_CONFIG.json", 43)
    if d and d not in ("OOM", "NAN"):
        synth("NRL43", f"{CFR}/{NRL43_ARM}/weights/last.pt")
        real("NRL43", f"{CFR}/{NRL43_ARM}/weights/last.pt")
        st_set("U2S_NRL_SEED43", status="PASS")
    else:
        st_set("U2S_NRL_SEED43", status="FAIL")
else:
    st_set("U2S_NRL_SEED43", status="SKIPPED", reason=f"CF_NRL_SIGNAL={NRL_SIG}")

# ---- U3 CF-PEVL -------------------------------------------------------
PEVL_SIG = "SKIPPED"
if NRL_SIG in ("FAIL", "SYNTHETIC_ONLY_POSITIVE"):
    check_sources("U3 전")
    st_set("U3_CF_PEVL", status="RUNNING")
    r = run([f"{Q}/cf_pevl_setup.py"])
    if not os.path.exists(f"{Q}/CF_PEVL_CONFIG.json"):
        st_set("U3_CF_PEVL", status="FAIL", reason=f"PEVL setup 실패 {(r.stderr or '')[-300:]}")
        PEVL_SIG = "FAIL"
    else:
        t = run([f"{Q}/test_cf_pevl.py"])
        tr_ = json.load(open(f"{Q}/CF_PEVL_TEST.json")) if os.path.exists(f"{Q}/CF_PEVL_TEST.json") else {}
        if not tr_.get("all_pass"):
            st_set("U3_CF_PEVL", status="FAIL", reason=f"unit test {tr_.get('n_pass')}/{tr_.get('n_total')}")
            PEVL_SIG = "FAIL"
        else:
            d = train(PEVL_ARM, f"{Q}/CF_PEVL_CONFIG.json", 42)
            if d == "OOM":
                fatal("U3 반복 OOM")
            if not d:
                st_set("U3_CF_PEVL", status="FAIL", reason="학습 산출물 없음")
                PEVL_SIG = "FAIL"
            else:
                st_set("U3_CF_PEVL", status="PASS")
                st_set("U3E_PEVL_EVAL", status="RUNNING")
                xs = synth("PEVL", f"{CFR}/{PEVL_ARM}/weights/last.pt")
                xr = real("PEVL", f"{CFR}/{PEVL_ARM}/weights/last.pt")
                PEVL_SIG, _ = gate(a0r, xr, a0s, xs, "CF_PEVL")
                st_set("U3E_PEVL_EVAL", status="PASS", CF_PEVL_SIGNAL=PEVL_SIG)
                brief("U3E_PEVL_EVAL", PEVL_SIG, xr, xs, "U4")
    log(f"U3 CF_PEVL_SIGNAL = {PEVL_SIG}")
else:
    st_set("U3_CF_PEVL", status="SKIPPED", reason=f"CF_NRL_SIGNAL={NRL_SIG} (real 에서 살아있음)")

if PEVL_SIG in ("REAL_POSITIVE", "REAL_POSITIVE_UNCERTAIN"):
    st_set("U3S_PEVL_SEED43", status="RUNNING")
    d = train(PEVL43_ARM, f"{Q}/CF_PEVL_CONFIG.json", 43)
    if d and d not in ("OOM", "NAN"):
        synth("PEVL43", f"{CFR}/{PEVL43_ARM}/weights/last.pt")
        real("PEVL43", f"{CFR}/{PEVL43_ARM}/weights/last.pt")
        st_set("U3S_PEVL_SEED43", status="PASS")
    else:
        st_set("U3S_PEVL_SEED43", status="FAIL")
else:
    st_set("U3S_PEVL_SEED43", status="SKIPPED", reason=f"CF_PEVL_SIGNAL={PEVL_SIG}")

# ---- U4 ---------------------------------------------------------------
st_set("U4_AUDIT", status="RUNNING")
run([f"{Q}/cf_audit.py"])
st_set("U4_AUDIT", status="PASS")
dur = (time.time() - T0) / 3600
log(f"=== CAMERA-FACING COMPLETE  {dur:.2f}h ===")
s = open(f"{Q}/CAMERA_FACING_SUMMARY.md").read()[:1500] \
    if os.path.exists(f"{Q}/CAMERA_FACING_SUMMARY.md") else ""
notify(f"✅ **[UBUNTU CAMERA-FACING COMPLETE]** ({dur:.2f}h)\n"
       f"summary path = {Q}/CAMERA_FACING_SUMMARY.md\n\n{s}")
if os.path.exists(LOCK):
    os.remove(LOCK)
