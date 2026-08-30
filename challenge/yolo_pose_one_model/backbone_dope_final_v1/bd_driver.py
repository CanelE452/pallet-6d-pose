"""DOPE on the frozen final dataset — smoke -> train -> verify -> notify.

학습에서 끝내지 않는다.  완료 판정은 exit code 나 프로세스 존재가 아니라
**산출물(체크포인트 파일)** 로만 한다.  중간 실패는 그 자리에서 알리고 종료한다.

평가셋이 아직 완성되지 않았으므로 pose 지표는 내지 않는다 — 학습과 계약 검증까지다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = "/home/minjae/Documents/github/pallet-pose"
NS = f"{ROOT}/challenge/yolo_pose_one_model/backbone_dope_final_v1"
DATA = f"{ROOT}/data/pallet/training_data/dope_final_g38_p0_tex"
TRAIN_DIR = f"{DATA}/train"
OUT = f"{ROOT}/weights/backbone_dope_final_v1"
TRAIN_PY = f"{ROOT}/Deep_Object_Pose/train"
PY = "/home/minjae/anaconda3/envs/pallet-pose/bin/python"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
LOG, LOCK = f"{NS}/DRIVER_LOG.txt", f"{NS}/DRIVER.lock"

# 기존 DOPE 레시피 고정 (weights/dope/dope_cropaug_pretrain/header.txt 와 동일 계약).
# sigma 4.0 은 건드리지 않는다 — sigma<1 은 gradient vanishing.
# batch 16 은 10GB 중 9.6GB 를 써서(94%) 장시간 run 에 여유가 없다 — 12 로 낮춘다.
EPOCHS, BATCH, IMGSZ, LR, SIGMA, WORKERS = 60, 12, 448, 0.0001, 4.0, 8
SAVE_EVERY = 5
SEED = 42


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def notify(m):
    try:
        subprocess.run([NOTIFY, m], timeout=60)
    except Exception as e:
        log(f"discord 실패(무시): {e}")


def die(m):
    log("FAIL " + m)
    notify(f"❌ DOPE final: {m}")
    if os.path.exists(LOCK):
        os.remove(LOCK)
    sys.exit(1)


def ckpts(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".pth"))


def train(outdir, data_dir, epochs, tag):
    """완료 판정 = final 체크포인트 존재.  이미 있으면 건너뛴다."""
    final = f"{outdir}/final_net_epoch_{epochs:04d}.pth"
    if os.path.exists(final):
        log(f"{tag} 이미 완료 -> {os.path.basename(final)}")
        return final
    os.makedirs(outdir, exist_ok=True)
    cmd = [PY, "-u", "train.py",
           "--data", data_dir,
           "--object", "pallet",
           "--epochs", str(epochs),
           "--batchsize", str(BATCH),
           "--imagesize", str(IMGSZ),
           "--lr", str(LR),
           "--sigma", str(SIGMA),
           "--workers", str(WORKERS),
           "--save_every", str(SAVE_EVERY),
           "--manualseed", str(SEED),
           "--loginterval", "200",
           "--outf", outdir]
    lf = f"{NS}/_train_{tag}.log"
    log(f"{tag} 시작  ep={epochs} batch={BATCH} imgsz={IMGSZ} lr={LR} sigma={SIGMA}")
    log("  " + " ".join(cmd))
    t0 = time.time()
    with open(lf, "w") as fh:
        r = subprocess.run(cmd, cwd=TRAIN_PY, stdout=fh, stderr=subprocess.STDOUT,
                           text=True)
    made = ckpts(outdir)
    if not made:
        tail = "\n".join(open(lf).read().replace("\r", "\n").split("\n")[-30:])
        die(f"{tag} 체크포인트 0개 (rc={r.returncode}):\n{tail[-1200:]}")
    log(f"{tag} 종료 {(time.time()-t0)/60:.0f}분  체크포인트 {len(made)}개  "
        f"최신 {made[-1]}")
    return final if os.path.exists(final) else f"{outdir}/{made[-1]}"


def main():
    os.makedirs(NS, exist_ok=True)
    if os.path.exists(LOCK):
        L = json.load(open(LOCK))
        if os.path.exists(f"/proc/{L['pid']}"):
            log("이미 실행 중 — 종료")
            return 0
    json.dump({"pid": os.getpid(), "start": time.strftime("%F %T")},
              open(LOCK, "w"))

    # ---------------------------------------------------------- 데이터 계약
    n_tr = len([f for f in os.listdir(TRAIN_DIR) if f.endswith(".json")])
    n_va = len([f for f in os.listdir(f"{DATA}/val") if f.endswith(".json")])
    if n_tr != 55980 or n_va != 4020:
        die(f"데이터 수 불일치 train {n_tr} val {n_va}")
    log(f"데이터 계약 OK  train {n_tr}  val {n_va}")

    # ---------------------------------------------------------- SMOKE 1ep
    # 512장만 뽑은 소형 셋으로 1ep — 배선이 살아 있는지 먼저 확인한다.
    smoke_dir = f"{DATA}/_smoke512"
    if not os.path.isdir(smoke_dir):
        os.makedirs(smoke_dir, exist_ok=True)
        names = sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith(".json"))
        step = max(1, len(names) // 512)
        for k, n in enumerate(names[::step][:512]):
            stem = n[:-5]
            for ext in (".json", ".png"):
                src, dst = f"{TRAIN_DIR}/{stem}{ext}", f"{smoke_dir}/{stem}{ext}"
                if not os.path.lexists(dst):
                    os.symlink(os.path.realpath(src), dst)
        log(f"smoke 셋 생성 {len(os.listdir(smoke_dir))//2}장")

    sm_out = f"{OUT}/_smoke"
    train(sm_out, smoke_dir, 1, "SMOKE")
    if not ckpts(sm_out):
        die("SMOKE 체크포인트 없음")
    log("SMOKE PASS")
    notify("✅ DOPE final SMOKE 통과 — 본학습 시작 (60ep, 55,980장)")

    # ---------------------------------------------------------- 본학습
    final = train(f"{OUT}/run", TRAIN_DIR, EPOCHS, "FULL")

    # ---------------------------------------------------------- 검증
    made = ckpts(f"{OUT}/run")
    ok = os.path.exists(f"{OUT}/run/final_net_epoch_{EPOCHS:04d}.pth")
    size_mb = os.path.getsize(final) / 1e6
    result = {
        "run": os.path.relpath(f"{OUT}/run", ROOT),
        "final_checkpoint": os.path.relpath(final, ROOT),
        "reached_final_epoch": ok,
        "n_checkpoints": len(made),
        "checkpoint_mb": round(size_mb, 1),
        "dataset": os.path.relpath(DATA, ROOT),
        "n_train": n_tr, "n_val": n_va,
        "recipe": {"epochs": EPOCHS, "batch": BATCH, "imgsz": IMGSZ, "lr": LR,
                   "sigma": SIGMA, "seed": SEED},
        "compare_against": ("challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
                            "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42"),
        "pose_metrics": "NOT_MEASURED — 평가셋 미완성",
    }
    json.dump(result, open(f"{NS}/RESULT.json", "w"), indent=1, ensure_ascii=False)

    verdict = "완주" if ok else f"중단 (최신 {made[-1] if made else '없음'})"
    log(f"VERDICT {verdict}  체크포인트 {len(made)}개  {size_mb:.0f}MB")
    notify(f"{'✅' if ok else '⚠️'} DOPE final {verdict} — "
           f"ckpt {len(made)}개 / {size_mb:.0f}MB / {os.path.basename(final)}. "
           f"평가는 평가셋 완성 후.")
    if os.path.exists(LOCK):
        os.remove(LOCK)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
