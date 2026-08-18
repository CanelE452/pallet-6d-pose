#!/usr/bin/env bash
# Wait for the nano run to finish, probe whether yolo26m-pose fits at batch 32, then
# train it on the same Stage A data.
#
# Batch policy (user decision 2026-08-14): if 32 does not fit, fall back to 16 rather
# than stopping. The prompt's default was "never lower the batch automatically"; this
# overrides it explicitly. lr is scaled with the batch (lr0 = 0.01 * batch / 16) so the
# two options remain comparable in optimiser terms — batch 32 -> lr 0.01, 16 -> 0.005.
#
# Everything else matches the nano run exactly: same data, same seed, same schedule.
set -uo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pallet-yolo26
cd "$(git rev-parse --show-toplevel)"

ROOT=challenge/yolo_pose_one_model
RUNS="$(pwd)/$ROOT/runs"
NOTIFY="$HOME/.claude/hooks/discord-notify.sh"
MODEL=challenge/weights/pretrained_yolo/yolo26m-pose.pt
LOG=$ROOT/runs/stage_a_m_chain.log

say() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------- 1. wait for nano
# Match "bin/yolo pose train", not "yolo pose train": any watcher shell whose command
# line merely CONTAINS the phrase (e.g. a monitor running `pgrep -f "yolo pose train"`)
# would otherwise be mistaken for the trainer and this loop would never exit.
# Only the real process has the interpreter path in its argv.
trainer_running() {
  ps -eo comm,args --no-headers 2>/dev/null |
    awk '$1 ~ /^python/ && /bin\/yolo pose train/ {f=1} END {exit !f}'
}

say "waiting for the nano run to finish"
while trainer_running; do sleep 60; done
NANO_CSV=$ROOT/runs/stage_a_synth_640_b32_seed42/results.csv
NANO_EP=$(( $(wc -l < "$NANO_CSV" 2>/dev/null || echo 1) - 1 ))
say "nano finished at ${NANO_EP} epochs"

# let the GPU settle before measuring
sleep 30
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
say "GPU free: ${FREE} MiB"

# ---------------------------------------------------------------- 2. probe batch 32
say "probing yolo26m-pose at batch 32"
python - "$MODEL" > "$ROOT/runs/_m_probe.txt" 2>&1 <<'PY'
import sys, torch, json
from ultralytics import YOLO
w = sys.argv[1]
ok, peak, err = False, 0.0, ""
try:
    torch.cuda.reset_peak_memory_stats()
    YOLO(w).train(
        data="challenge/yolo_pose_one_model/configs/stage_a.yaml",
        epochs=1, imgsz=640, batch=32, nbs=64, workers=4, optimizer="SGD",
        amp=True, deterministic=True, seed=42, mosaic=0.30, fliplr=0.0,
        fraction=0.004,            # ~300 frames: enough to allocate peak activations
        project="/tmp", name="_m_probe_b32", exist_ok=True, plots=False,
        val=False, verbose=False, save=False,
    )
    ok = True
except torch.cuda.OutOfMemoryError:
    err = "OutOfMemoryError"
except RuntimeError as e:
    err = "RuntimeError:" + str(e)[:120]
except Exception as e:
    err = type(e).__name__ + ":" + str(e)[:120]
peak = torch.cuda.max_memory_reserved() / 1024 ** 3
print(json.dumps({"ok": ok, "peak_gib": round(peak, 2), "err": err}))
PY

VERDICT=$(grep -o '{"ok".*}' "$ROOT/runs/_m_probe.txt" | tail -1)
say "probe result: ${VERDICT:-<no json — see runs/_m_probe.txt>}"

if echo "$VERDICT" | grep -q '"ok": true'; then
  BATCH=32; LR=0.01
else
  BATCH=16; LR=0.005
fi
PEAK=$(echo "$VERDICT" | grep -o '"peak_gib": [0-9.]*' | cut -d' ' -f2)
say "selected batch=${BATCH} lr0=${LR} (probe peak ${PEAK:-?} GiB)"

"$NOTIFY" "[stage_a_m] nano 종료(${NANO_EP} ep) → medium 학습 시작
batch=${BATCH}  lr0=${LR}  (batch32 probe: ${VERDICT:-실패})
같은 Stage A 데이터·seed 42·60 epoch" || true

# ---------------------------------------------------------------- 3. train medium
say "starting medium training"
yolo pose train \
  model=$MODEL \
  data=$ROOT/configs/stage_a.yaml \
  imgsz=640 \
  batch=$BATCH \
  nbs=64 \
  epochs=60 \
  patience=15 \
  optimizer=SGD \
  lr0=$LR \
  lrf=0.01 \
  momentum=0.937 \
  weight_decay=0.0005 \
  warmup_epochs=3.0 \
  warmup_momentum=0.8 \
  warmup_bias_lr=0.1 \
  cos_lr=True \
  amp=True \
  device=0 \
  workers=4 \
  cache=False \
  seed=42 \
  deterministic=True \
  single_cls=True \
  pretrained=True \
  resume=False \
  multi_scale=False \
  fliplr=0.0 \
  flipud=0.0 \
  degrees=0.0 \
  shear=0.0 \
  perspective=0.0 \
  translate=0.10 \
  scale=0.25 \
  mosaic=0.30 \
  close_mosaic=10 \
  mixup=0.0 \
  copy_paste=0.0 \
  hsv_h=0.015 \
  hsv_s=0.50 \
  hsv_v=0.35 \
  save=True \
  save_period=5 \
  plots=True \
  verbose=True \
  project="$RUNS" \
  name=stage_a_m_640_b${BATCH}_seed42 \
  exist_ok=False
RC=$?

CSV=$ROOT/runs/stage_a_m_640_b${BATCH}_seed42/results.csv
EP=$(( $(wc -l < "$CSV" 2>/dev/null || echo 1) - 1 ))
SUM=$(tail -1 "$CSV" 2>/dev/null | awk -F, '{printf "epoch %s | Box mAP50 %.4f | Pose mAP50 %.4f mAP50-95 %.4f | %.1f h", $1, $11, $15, $16, $2/3600}')
if [ "$RC" = "0" ]; then
  "$NOTIFY" "[stage_a_m] medium 학습 완료 (batch=${BATCH})
${SUM}" || true
else
  "$NOTIFY" "[stage_a_m] medium 학습이 실패로 끝났습니다 (exit ${RC}, ${EP} epoch)
로그: ${LOG}" || true
fi
say "medium finished rc=$RC at ${EP} epochs"
