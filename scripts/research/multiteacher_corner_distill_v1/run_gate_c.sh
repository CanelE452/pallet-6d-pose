#!/usr/bin/env bash
# GATE C 드라이버 — 학습에서 끝내지 않는다.
#   STEP 1  C0 학습        STEP 2  C1 학습
#   STEP 3  2D 평가        STEP 4  6D 평가
#   STEP 5  판정(임계 하드코딩)
# 완료 판정은 산출물 존재로만 한다. exit code 나 프로세스 존재로 하지 않는다.
set -u
cd /home/minjae/Documents/github/pallet-pose
R=data/pallet/results/multiteacher_corner_distill_v1
G=$R/gate_c_local_specialist
L=$R/logs
mkdir -p "$L"
PY="conda run -n pallet-yolo26 python"
S=scripts/research/multiteacher_corner_distill_v1

fail () { echo "[GATE_C][FAIL] $1" | tee -a "$L/gate_c_driver.log"; exit 1; }
say  () { echo "[GATE_C] $(date +%H:%M:%S) $1" | tee -a "$L/gate_c_driver.log"; }

say "nvidia-smi before"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tee -a "$L/gate_c_driver.log"

for ARM in C0 C1; do
  say "STEP train $ARM"
  $PY $S/gate_c_train_specialist.py --arm "$ARM" --updates 5000 --cap-minutes 60 \
      > "$L/train_$ARM.log" 2>&1
  [ -f "$G/TRAIN_$ARM.json" ] || fail "TRAIN_$ARM.json 없음 — 학습이 산출물을 남기지 못했다"
  say "  $ARM done: $(python3 -c "import json;d=json.load(open('$G/TRAIN_$ARM.json'));print('updates',d['updates_done'],'real_kp',d['n_real_usable_keypoints'])")"
done

say "STEP 2D evaluation"
$PY $S/gate_c_evaluate.py --arms C0 C1 > "$L/gate_c_eval.log" 2>&1
[ -f "$G/GATE_C_2D.json" ] || fail "GATE_C_2D.json 없음"
tail -8 "$L/gate_c_eval.log" | tee -a "$L/gate_c_driver.log"

say "STEP 6D evaluation"
for ARM in C0 C1; do
  $PY $S/eval_pose_arm.py --arm "$ARM" \
      --out-dir "$(pwd)/$G" >> "$L/gate_c_eval.log" 2>&1
  [ -f "$G/POSE_EVALUATION_$ARM.json" ] || fail "POSE_EVALUATION_$ARM.json 없음"
done

say "STEP verdict"
$PY $S/gate_c_verdict.py 2>&1 | tee -a "$L/gate_c_driver.log"
[ -f "$G/GATE_C_VERDICT.json" ] || fail "GATE_C_VERDICT.json 없음"

say "nvidia-smi after"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tee -a "$L/gate_c_driver.log"
say "COMPLETE"
