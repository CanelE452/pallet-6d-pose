"""ADAPTIVE_60EP_CONFIRM 야간 드라이버.

train -> 완주검증 -> checkpoint 수렴분석 -> semantic 진단 -> plots -> 판정 -> 알림.
한 파일에서 끝낸다. 사람 개입 0.

완료 판정은 exit code 가 아니라 results.csv 의 최종 epoch 마크로만 한다.
중간 성능이 나빠도 60ep 를 중단하지 않는다 — 장기 수렴 확인이 이 run 의 목적이다.
"""
from __future__ import annotations
import csv, glob, json, math, os, subprocess, sys, time

ROOT = "/home/minjae/Documents/github/pallet-pose"
D = os.path.join(ROOT, "challenge/yolo_pose_one_model")
RF = os.path.join(D, "runs_fixed")
RUN = "FIXED_OBJECT_BROAD40K_60EP_SEED42_ADAPTIVE_CONFIRM"
OUT = os.path.join(RF, RUN)
LOCK = json.load(open(os.path.join(RF, "FIXED_60EP_CONFIG_LOCK.json")))["locked"]
YOLO = "/home/minjae/anaconda3/envs/pallet-yolo26/bin/yolo"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
FINAL = 60
# 사전 고정 permutation (generator 에서 확인된 proper yaw)
PERMS = {"identity": (0, 1, 2, 3, 4, 5, 6, 7),
         "yaw180": (5, 4, 7, 6, 1, 0, 3, 2),
         "yaw90": (4, 0, 3, 7, 5, 1, 2, 6),
         "yaw270": (1, 5, 6, 2, 0, 4, 7, 3)}
FIVE_EP = {"pose_mAP50": 0.5936, "pose_mAP50_95": 0.08752, "box_mAP50": 0.98339,
           "identity_best": 0.564, "yaw180_best": 0.4202}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def notify(t):
    try:
        subprocess.run([NOTIFY, t], timeout=90)
    except Exception as e:
        log(f"notify 실패 {e}")


def rows_of(p):
    return list(csv.DictReader(open(p)))


def final_mark(run_dir, need=FINAL):
    p = os.path.join(run_dir, "results.csv")
    if not os.path.exists(p):
        return False, None
    r = rows_of(p)
    return (bool(r) and int(float(r[-1]["epoch"])) >= need), r


def train(resume=False):
    cmd = [YOLO, "pose", "train"]
    if resume:
        cmd += [f"model={os.path.join(OUT,'weights/last.pt')}", "resume=True"]
    else:
        for k, v in LOCK.items():
            cmd.append(f"{k}={v}")
    log(" ".join(cmd))
    with open(os.path.join(RF, "train60ep.log"), "a") as fh:
        fh.write(f"\n===== {time.strftime('%F %T')} resume={resume} =====\n")
        fh.flush()
        # 파이프 없음 — 버퍼링으로 진행이 안 보이는 사고를 막는다
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode


def is_oom():
    p = os.path.join(RF, "train60ep.log")
    if not os.path.exists(p):
        return False
    tail = open(p, errors="ignore").read()[-40000:]
    return ("CUDA out of memory" in tail) or ("torch.cuda.OutOfMemoryError" in tail)


def evaluate():
    """checkpoint 별 mAP + semantic assignment 진단."""
    import numpy as np, cv2
    from ultralytics import YOLO as Y
    cv2.setNumThreads(2)
    W = os.path.join(OUT, "weights")
    cks = []
    for e in (10, 20, 30, 40, 50):
        p = os.path.join(W, f"epoch{e}.pt")
        if os.path.exists(p):
            cks.append((e, p))
    if os.path.exists(os.path.join(W, "last.pt")):
        cks.append((60, os.path.join(W, "last.pt")))
    imgs = sorted(glob.glob(os.path.join(D, "datasets/broad40k_fixed/images/val/*.png")))
    rowsout = []
    for ep, ck in cks:
        m = Y(ck, task="pose")
        mt = m.val(data=LOCK["data"], split="val", imgsz=LOCK["imgsz"],
                   batch=16, verbose=False, plots=False)
        best = {k: 0 for k in PERMS}
        nat, orc, n = [], [], 0
        chan = np.zeros(9)
        for p in imgs:
            lp = p.replace("/images/", "/labels/").replace(".png", ".txt")
            if not os.path.exists(lp):
                continue
            f = open(lp).read().split()
            kp = np.array([[float(f[5+3*i]), float(f[6+3*i]), float(f[7+3*i])]
                           for i in range(9)])
            r = m.predict(p, imgsz=LOCK["imgsz"], conf=0.25, verbose=False)[0]
            if r.keypoints is None or not len(r.boxes):
                continue
            H, Wd = cv2.imread(p).shape[:2]
            pr = r.keypoints.xy.cpu().numpy()[0]
            gt = np.stack([kp[:, 0]*Wd, kp[:, 1]*H], 1)
            vis = kp[:, 2] > 0
            if vis[:8].sum() < 4:
                continue
            n += 1
            kc = r.keypoints.conf
            chan += 1 if kc is None else (kc.cpu().numpy()[0] > 0.5).astype(float)
            diag = np.hypot(gt[vis][:, 0].ptp(), gt[vis][:, 1].ptp()) or 1.0
            sc = {}
            for name, pm in PERMS.items():
                idx = list(pm) + [8]
                sc[name] = float(np.median(
                    np.linalg.norm(pr[idx][vis] - gt[vis], axis=1) / diag))
            b = min(sc, key=sc.get)
            best[b] += 1
            nat.append(sc["identity"]); orc.append(sc[b])
        row = {"epoch": ep, "checkpoint": os.path.basename(ck), "n_val": n,
               "box_mAP50": float(mt.box.map50), "box_mAP50_95": float(mt.box.map),
               "pose_mAP50": float(mt.pose.map50), "pose_mAP50_95": float(mt.pose.map),
               "native_fixed_corner_error": float(np.median(nat)) if nat else None,
               "oracle_perm_corner_error": float(np.median(orc)) if orc else None,
               "collapsed_channels": int(sum(1 for i in range(8)
                                             if chan[i]/max(n, 1) < 0.05)),
               "channel_detect_rate": {str(i): round(float(chan[i]/max(n, 1)), 4)
                                       for i in range(9)}}
        for k in PERMS:
            row[f"{k}_best_fraction"] = round(best[k]/max(n, 1), 4)
        rowsout.append(row)
        log(f"  ep{ep:>3} poseMAP50 {row['pose_mAP50']:.4f} "
            f"50-95 {row['pose_mAP50_95']:.4f} "
            f"identity {row['identity_best_fraction']:.3f} "
            f"yaw180 {row['yaw180_best_fraction']:.3f}")
    return rowsout


def plots(rowsout, tr):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm, glob as g
    h = g.glob("/usr/share/fonts/**/NanumGothic.ttf", recursive=True)
    if h:
        fm.fontManager.addfont(h[0])
        plt.rcParams["font.family"] = fm.FontProperties(fname=h[0]).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    e = [r["epoch"] for r in rowsout]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(e, [r["pose_mAP50"] for r in rowsout], "o-", label="pose mAP50")
    ax[0].plot(e, [r["pose_mAP50_95"] for r in rowsout], "s-", label="pose mAP50-95")
    ax[0].plot(e, [r["box_mAP50"] for r in rowsout], "^--", c="#888", label="box mAP50")
    ax[0].scatter([5], [FIVE_EP["pose_mAP50"]], c="r", zorder=5, label="5ep 기준")
    ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title("수렴 곡선")
    ax[1].plot(e, [r["identity_best_fraction"] for r in rowsout], "o-",
               c="#27ae60", label="identity best")
    ax[1].plot(e, [r["yaw180_best_fraction"] for r in rowsout], "s-",
               c="#c0392b", label="yaw180 best")
    ax[1].scatter([5, 5], [FIVE_EP["identity_best"], FIVE_EP["yaw180_best"]],
                  c=["#27ae60", "#c0392b"], marker="x", s=70, label="5ep")
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3); ax[1].set_title("semantic assignment (진단 전용)")
    ax[2].plot([int(float(r["epoch"])) for r in tr],
               [float(r["train/pose_loss"]) for r in tr], label="train pose")
    ax[2].plot([int(float(r["epoch"])) for r in tr],
               [float(r["val/pose_loss"]) for r in tr], label="val pose")
    ax[2].set_xlabel("epoch"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    ax[2].set_title("pose loss")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "CONVERGENCE.png"), dpi=130)


def main():
    ok, _ = final_mark(OUT)
    restarts = 0
    if not ok:
        notify(f"**FIXED 60ep ADAPTIVE_CONFIRM 시작**\n\n"
               f"recipe = paper 60ep artifact (cos_lr True / mosaic 0.3 / "
               f"scale 0.25 / single_cls True / save_period 10)\n"
               f"5ep checkpoint resume 안 함 — pretrained 에서 처음부터.\n"
               f"ORIGINAL_5EP_GATE = FAIL 그대로 보존.\n예상 5~6시간.")
        while True:
            train(resume=(restarts > 0))
            ok, _ = final_mark(OUT)
            if ok:
                break
            if is_oom():
                notify("❌ **TRAIN_BLOCKED_OOM** — batch/imgsz 축소 금지 규칙에 따라 "
                       "중단한다. recipe 변경 없음.")
                log("TRAIN_BLOCKED_OOM"); return
            if restarts >= 1:
                notify("❌ **TRAIN_FAILED** — 2회째 실패. 자동 반복하지 않는다. "
                       "runs_fixed/train60ep.log 확인 필요.")
                log("TRAIN_FAILED"); return
            restarts += 1
            log(f"비-OOM 실패 — 동일 recipe 로 1회 resume 재시도 ({restarts})")
    ok, tr = final_mark(OUT)
    nan = sum(1 for r in tr for v in r.values()
              if _isbad(v))
    log(f"60ep 완주. NaN/Inf {nan}. checkpoint 평가 시작")
    notify(f"**FIXED 60ep 학습 COMPLETE** (재시도 {restarts}회, NaN {nan})\n"
           f"checkpoint 수렴 분석 시작.")
    rowsout = evaluate()
    with open(os.path.join(OUT, "CONVERGENCE.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rowsout[0])); w.writeheader()
        for r in rowsout:
            w.writerow(r)
    try:
        plots(rowsout, tr)
    except Exception as e:
        log(f"plot 실패 {e}")
    first, last = rowsout[0], rowsout[-1]
    obs = {"pose_mAP50_5ep": FIVE_EP["pose_mAP50"],
           "pose_mAP50_60ep": last["pose_mAP50"],
           "pose_mAP50_95_5ep": FIVE_EP["pose_mAP50_95"],
           "pose_mAP50_95_60ep": last["pose_mAP50_95"],
           "identity_best_5ep": FIVE_EP["identity_best"],
           "identity_best_60ep": last["identity_best_fraction"],
           "yaw180_best_5ep": FIVE_EP["yaw180_best"],
           "yaw180_best_60ep": last["yaw180_best_fraction"],
           "train_pose_loss_first_last": [float(tr[0]["train/pose_loss"]),
                                          float(tr[-1]["train/pose_loss"])],
           "collapsed_channels_60ep": last["collapsed_channels"]}
    d_id = obs["identity_best_60ep"] - obs["identity_best_5ep"]
    d_yaw = obs["yaw180_best_60ep"] - obs["yaw180_best_5ep"]
    d_map = obs["pose_mAP50_95_60ep"] - obs["pose_mAP50_95_5ep"]
    if d_id > 0 and d_yaw < 0 and d_map > 0 and obs["yaw180_best_60ep"] < 0.15:
        case, verdict = "H1", "5EP_WAS_INSUFFICIENT_BUDGET"
    elif d_map <= 0 or (abs(d_yaw) < 0.05 and d_id < 0.05):
        case, verdict = "H2", "FIXED_OBJECT_ASSIGNMENT_NOT_LEARNED"
    else:
        case, verdict = "H3", "PARTIAL_FIXED_SEMANTIC_LEARNING"
    res = {"ORIGINAL_5EP_GATE": "FAIL (보존, 변경하지 않음)",
           "ADAPTIVE_60EP_CONFIRM": "USER_AUTHORIZED",
           "observations": obs, "case": case, "verdict": verdict,
           "FIXED_SEMANTIC_LEARNING_CONFIRMED": case == "H1",
           "FIXED_SEMANTIC_AMBIGUITY_PERSISTS": case == "H2",
           "PARTIAL_FIXED_SEMANTIC_LEARNING": case == "H3",
           "restarts": restarts, "nan_inf": nan,
           "paper_main_checkpoint": "last.pt (사전 규칙. 결과 보고 바꾸지 않음)",
           "★permutation_oracle": "진단 전용. main metric 아님.",
           "curve": rowsout}
    json.dump(res, open(os.path.join(OUT, "ADAPTIVE_60EP_VERDICT.json"), "w"),
              indent=1, ensure_ascii=False)
    tbl = "\n".join(
        f"{r['epoch']:>3} | {r['pose_mAP50']:.4f} | {r['pose_mAP50_95']:.4f} | "
        f"{r['identity_best_fraction']:.3f} | {r['yaw180_best_fraction']:.3f}"
        for r in rowsout)
    notify(f"**FIXED 60ep 분석 완료 — {case} {verdict}**\n\n"
           f"```\nep  | poseMAP50 | 50-95  | identity | yaw180\n"
           f"  5 | {FIVE_EP['pose_mAP50']:.4f} | {FIVE_EP['pose_mAP50_95']:.4f} | "
           f"{FIVE_EP['identity_best']:.3f} | {FIVE_EP['yaw180_best']:.3f}\n{tbl}\n```\n"
           f"identity {obs['identity_best_5ep']:.3f}->{obs['identity_best_60ep']:.3f} · "
           f"yaw180 {obs['yaw180_best_5ep']:.3f}->{obs['yaw180_best_60ep']:.3f}\n"
           f"★ ORIGINAL_5EP_GATE = FAIL 그대로. 이건 별도 adaptive 결과다.\n"
           f"★ real 평가·paper 판정은 reviewed fixed real GT 도착 후.")
    log(f"{case} {verdict}")


def _isbad(v):
    try:
        f = float(v)
        return math.isnan(f) or math.isinf(f)
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
