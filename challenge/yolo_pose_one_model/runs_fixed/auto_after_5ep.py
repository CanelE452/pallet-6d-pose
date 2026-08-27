"""5ep 완료 -> 평가 -> 사전등록 gate -> PASS 면 60ep 착수 -> 알림.  한 파일에서 끝낸다.

완료 판정은 exit code 나 프로세스 존재가 아니라 **results.csv 의 최종 epoch 마크**로만 한다.
게이트는 결과를 보기 전에 여기 하드코딩돼 있다 — 실행 후 수정하지 않는다.
"""
from __future__ import annotations
import csv, json, math, os, subprocess, sys, time

ROOT = "/home/minjae/Documents/github/pallet-pose"
D = os.path.join(ROOT, "challenge/yolo_pose_one_model")
RUN5 = os.path.join(D, "runs_fixed/FIXED_OBJECT_BROAD40K_5EP_SEED42")
CF5 = os.path.join(D, "runs_broad40k/b_yolo26n_broad40k_5ep")
FIXED_YAML = os.path.join(D, "datasets/broad40k_fixed/data.yaml")
INIT = os.path.join(ROOT, "challenge/weights/pretrained_yolo/yolo26n-pose.pt")
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
YOLO = "/home/minjae/anaconda3/envs/pallet-yolo26/bin/yolo"
FINAL_EPOCH = 5

# ---- 사전등록 gate (결과 보기 전 고정) --------------------------------------
GATE = {"max_relative_pose_map_collapse": 0.30,   # CF5ep 대비 30% 이상 붕괴 금지
        "require_no_nan": True,
        "require_loss_decreasing": True,
        "require_no_channel_collapse": True}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def notify(text):
    try:
        subprocess.run([NOTIFY, text], timeout=60)
    except Exception as e:
        log(f"notify 실패: {e}")


def rows_of(path):
    return list(csv.DictReader(open(path)))


def wait_final():
    """완료 = results.csv 에 최종 epoch 행이 존재. 프로세스 존재로 판정하지 않는다."""
    while True:
        p = os.path.join(RUN5, "results.csv")
        if os.path.exists(p):
            r = rows_of(p)
            if r and int(float(r[-1]["epoch"])) >= FINAL_EPOCH:
                return r
        time.sleep(20)


def main():
    log("5ep 최종 마크 대기")
    rows = wait_final()
    log(f"5ep 완료 — {len(rows)} epoch")

    # ---- NaN / 손실 추이 ----
    nan = 0
    for r in rows:
        for k, v in r.items():
            try:
                if math.isnan(float(v)) or math.isinf(float(v)):
                    nan += 1
            except (TypeError, ValueError):
                pass
    tr = [float(r["train/pose_loss"]) for r in rows]
    decreasing = tr[-1] < tr[0]

    fx = float(rows[-1]["metrics/mAP50-95(P)"])
    fx50 = float(rows[-1]["metrics/mAP50(P)"])
    fxb = float(rows[-1]["metrics/mAP50(B)"])
    cf = rows_of(os.path.join(CF5, "results.csv"))[-1]
    cf5095 = float(cf["metrics/mAP50-95(P)"])
    cf50 = float(cf["metrics/mAP50(P)"])
    cfb = float(cf["metrics/mAP50(B)"])
    rel = (fx - cf5095) / cf5095 if cf5095 else float("nan")
    rel50 = (fx50 - cf50) / cf50 if cf50 else float("nan")

    # ---- keypoint channel collapse: val 500 에서 채널별 예측 존재 확인 ----
    collapse, per_ch = None, {}
    try:
        sys.path.insert(0, os.path.join(ROOT, "scripts/stage0/model_compare"))
        import numpy as np
        from ultralytics import YOLO as Y
        import glob
        m = Y(os.path.join(RUN5, "weights/last.pt"), task="pose")
        imgs = sorted(glob.glob(os.path.join(
            D, "datasets/broad40k_fixed/images/val/*.png")))[:120]
        cnt = np.zeros(9)
        n = 0
        for im in imgs:
            r = m.predict(im, imgsz=640, conf=0.25, verbose=False)[0]
            if r.keypoints is None or not len(r.boxes):
                continue
            n += 1
            kc = r.keypoints.conf
            if kc is None:
                cnt += 1
                continue
            cnt += (kc.cpu().numpy()[0] > 0.5).astype(float)
        per_ch = {str(i): round(float(cnt[i] / max(n, 1)), 4) for i in range(9)}
        collapse = sum(1 for i in range(8) if cnt[i] / max(n, 1) < 0.05)
        log(f"채널별 검출률 {per_ch}  collapse={collapse}  (n={n})")
    except Exception as e:
        log(f"채널 점검 실패: {e}")

    ok_nan = nan == 0
    ok_dec = decreasing
    ok_ch = (collapse == 0) if collapse is not None else False
    ok_rel = rel >= -GATE["max_relative_pose_map_collapse"]
    go = ok_nan and ok_dec and ok_ch and ok_rel

    verdict = {
        "run": os.path.basename(RUN5), "epochs": len(rows),
        "gate": GATE,
        "fixed": {"pose_mAP50": fx50, "pose_mAP50_95": fx,
                  "box_mAP50": fxb,
                  "train_pose_loss_first_last": [tr[0], tr[-1]]},
        "cf5ep_reference": {"pose_mAP50": cf50, "pose_mAP50_95": cf5095,
                            "box_mAP50": cfb},
        "relative_change_pose_mAP50_95": round(rel, 4),
        "relative_change_pose_mAP50": round(rel50, 4),
        "nan_inf_count": nan, "loss_decreasing": decreasing,
        "channel_collapse": collapse, "per_channel_detect_rate": per_ch,
        "checks": {"no_nan": ok_nan, "loss_decreasing": ok_dec,
                   "no_channel_collapse": ok_ch, "pose_map_not_collapsed": ok_rel},
        "SPECULATIVE_GO_60EP": bool(go),
        "★note": "이건 GPU 를 계속 쓸지 정하는 compute gate 다. "
                 "PAPER_FIXED_OBJECT_VIABLE 판정이 아니다 — 그건 reviewed real GT "
                 "도착 후에만 한다.",
    }
    json.dump(verdict, open(os.path.join(RUN5, "FIXED_5EP_VERDICT.json"), "w"),
              indent=1, ensure_ascii=False)
    log(json.dumps(verdict["checks"], ensure_ascii=False))

    head = (f"**FIXED 5ep 완료 — SPECULATIVE_GO_60EP = {go}**\n\n"
            f"pose mAP50    CF {cf50:.4f} -> FIXED {fx50:.4f}\n"
            f"pose mAP50-95 CF {cf5095:.4f} -> FIXED {fx:.4f}  (상대 {rel:+.1%})\n"
            f"box  mAP50    CF {cfb:.4f} -> FIXED {fxb:.4f}\n"
            f"NaN {nan} · loss {tr[0]:.3f}->{tr[-1]:.3f} · channel collapse {collapse}\n"
            f"게이트: {verdict['checks']}\n")
    if not go:
        notify(head + "\n60ep 시작하지 않음. 원인 진단 필요.")
        log("GATE FAIL — 60ep 시작하지 않는다")
        return
    notify(head + "\n-> 60ep seed42 즉시 착수 (약 4.8시간). "
                  "이건 compute gate 이지 paper viability 판정이 아니다.")

    log("GATE PASS — 60ep 착수")
    cmd = [YOLO, "pose", "train", f"model={INIT}", f"data={FIXED_YAML}",
           "epochs=60", "batch=32", "imgsz=640", "optimizer=SGD", "lr0=0.01",
           "seed=42", "patience=0", "workers=4", "pretrained=True",
           "fliplr=0.0", "device=0", "cache=False", "save_period=10",
           f"project={os.path.join(D, 'runs_fixed')}",
           "name=FIXED_OBJECT_BROAD40K_60EP_SEED42"]
    log(" ".join(cmd))
    t0 = time.time()
    with open(os.path.join(D, "runs_fixed", "train60ep.log"), "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    run60 = os.path.join(D, "runs_fixed/FIXED_OBJECT_BROAD40K_60EP_SEED42")
    p = os.path.join(run60, "results.csv")
    done = os.path.exists(p) and int(float(rows_of(p)[-1]["epoch"])) >= 60
    if done:
        last = rows_of(p)[-1]
        notify(f"**FIXED 60ep seed42 완료** ({(time.time()-t0)/3600:.1f}h)\n\n"
               f"pose mAP50 {float(last['metrics/mAP50(P)']):.4f} · "
               f"mAP50-95 {float(last['metrics/mAP50-95(P)']):.4f}\n"
               f"box mAP50 {float(last['metrics/mAP50(B)']):.4f}\n"
               f"checkpoint = last.pt (사전 규칙)\n"
               f"★ real 평가와 paper 판정은 reviewed fixed real GT 도착 후.")
    else:
        notify("❌ FIXED 60ep 이 60 epoch 마크 없이 종료됐다. 로그 확인 필요: "
               "runs_fixed/train60ep.log")
    log("done")


if __name__ == "__main__":
    main()
