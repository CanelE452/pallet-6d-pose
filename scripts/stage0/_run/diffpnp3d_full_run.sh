#!/usr/bin/env bash
# diffpnp3d_full_run.sh — PAPER_S2 Phase 4 full pipeline (detached, unattended).
#
#   Stage A (scratch 60ep, Arm A, DiffPnP3D lambda0.005)
#     -> synthetic-val COMPOSITE select (A best)
#   Stage B (mask FT 15ep, Arm A+B 60:40, +mask_aux, DiffPnP3D lambda0.005, init=A best)
#     -> synthetic-val COMPOSITE select (B best)
#
# All conditions locked to PLAN.md / Q1-GO(lambda=0.005): scratch, batch12, sigma2,
# imagesize400 (aspect_resize squash), DiffPnP gate = pnp_valid_3d & V8 & belief-interior.
# Launch detached:  nohup bash scripts/stage0/_run/diffpnp3d_full_run.sh > <log> 2>&1 &
set -e
ROOT=/home/minjae/Documents/github/pallet-pose
DATA=$ROOT/data/pallet/training_data
IDXDIR=$ROOT/data/pallet/results/paper_s2_scratch_diffpnp/pnp_valid_3d_index
LOGDIR=$ROOT/data/pallet/results/paper_s2_scratch_diffpnp/full_run_logs
STAGEA=$ROOT/weights/paper_s2_stageA
STAGEB=$ROOT/weights/paper_s2_stageB
mkdir -p "$LOGDIR" "$STAGEA" "$STAGEB"

export PYTHONUNBUFFERED=1
export CUDA_MODULE_LOADING=LAZY
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pallet-pose

LAMBDA=0.005
ARMA="$DATA/mixed_v8_train $DATA/v4_split_base $DATA/aug_squash_v2 $DATA/aug_trunc_v2 $DATA/aug_scale_v2"
ARMB="$DATA/paper_4pallet_mask_v1"

echo "=== [$(date '+%F %T')] FULL RUN START (lambda=$LAMBDA) ==="

# ---------------- Stage A: scratch 60ep, Arm A only ----------------
cd "$ROOT/Deep_Object_Pose/train"
if [ -f "$STAGEA/final_net_epoch_0060.pth" ]; then
  echo "[skip] Stage A already complete (final_net_epoch_0060.pth)"
else
  echo "=== [$(date '+%F %T')] Stage A train (scratch 60ep) ==="
  python train.py \
    --data $ARMA \
    --object pallet \
    --batchsize 12 --sigma 2 --imagesize 400 \
    --lr 0.0001 --epochs 60 --workers 6 \
    --manualseed 42 --save_every 3 \
    --outf "$STAGEA" \
    --diffpnp --diffpnp_lambda $LAMBDA --diffpnp_temp 0.1 \
    --diffpnp_warmup 0 --diffpnp_ramp 3000 \
    --diffpnp_index_dir "$IDXDIR" --diffpnp_root "$DATA" \
    > "$LOGDIR/stageA_train.log" 2>&1
  echo "=== [$(date '+%F %T')] Stage A train done ==="
fi

# ---------------- Stage A composite val-select ----------------
cd "$ROOT"
echo "=== [$(date '+%F %T')] Stage A val-select ==="
python scripts/stage0/diffpnp3d/diffpnp3d_val_select.py \
  --weights_dir "$STAGEA" --tag stageA --val full \
  > "$LOGDIR/stageA_valselect.log" 2>&1
A_BEST=$(grep '^BEST_CKPT=' "$LOGDIR/stageA_valselect.log" | tail -1 | cut -d= -f2)
if [ -z "$A_BEST" ] || [ ! -f "$A_BEST" ]; then
  echo "[FATAL] Stage A best checkpoint not found (see stageA_valselect.log)"; exit 4
fi
echo "Stage A best = $A_BEST"

# train.py finetune uses CUMULATIVE epochs (memory: dope-finetune-cumulative-epoch):
# it resumes at (A_best_epoch + 1) and treats --epochs as the ABSOLUTE target.
# So Stage B --epochs must be A_best_epoch + delta, else range(start, epochs+1) is
# empty -> UnboundLocalError at final save. delta = 15 mask-FT epochs.
A_EP=$(basename "$A_BEST" | grep -oE '[0-9]{4}' | head -1)   # e.g. 0042
EPB=$((10#$A_EP + 15))                                       # cumulative target
FINALB=$(printf 'final_net_epoch_%04d.pth' "$EPB")

# ---------------- Stage B: mask FT 15ep, Arm A+B 60:40, init=A best ----------------
cd "$ROOT/Deep_Object_Pose/train"
if [ -f "$STAGEB/$FINALB" ]; then
  echo "[skip] Stage B already complete ($FINALB)"
else
  echo "=== [$(date '+%F %T')] Stage B train (mask FT 15ep -> cumulative ep$EPB, init=A best ep$A_EP) ==="
  python train.py \
    --data $ARMA $ARMB \
    --object pallet \
    --batchsize 12 --sigma 2 --imagesize 400 \
    --lr 0.00005 --epochs $EPB --workers 6 \
    --manualseed 42 --save_every 3 \
    --net_path "$A_BEST" \
    --outf "$STAGEB" \
    --balance_groups "mixed_v8_train|v4_split_base|aug_squash_v2|aug_trunc_v2|aug_scale_v2:60,paper_4pallet_mask_v1:40" \
    --mask_aux --mask_weight 0.01 --mask_warmup 0 \
    --diffpnp --diffpnp_lambda $LAMBDA --diffpnp_temp 0.1 \
    --diffpnp_warmup 0 --diffpnp_ramp 1000 \
    --diffpnp_index_dir "$IDXDIR" --diffpnp_root "$DATA" \
    > "$LOGDIR/stageB_train.log" 2>&1
  echo "=== [$(date '+%F %T')] Stage B train done ==="
fi

# NOTE: Stage B --net_path is the Stage-A heatmap ckpt (numSeg=0). Stage B model has
# numSeg=1 (mask_aux) -> train.py loads strict=False, asserting the ONLY missing keys
# are the fresh m_seg* head. No unexpected keys. (verified: train.py L582-597)

# ---------------- Stage B composite val-select ----------------
cd "$ROOT"
echo "=== [$(date '+%F %T')] Stage B val-select ==="
python scripts/stage0/diffpnp3d/diffpnp3d_val_select.py \
  --weights_dir "$STAGEB" --tag stageB --val full \
  > "$LOGDIR/stageB_valselect.log" 2>&1
B_BEST=$(grep '^BEST_CKPT=' "$LOGDIR/stageB_valselect.log" | tail -1 | cut -d= -f2)
echo "Stage B best = $B_BEST"

echo "=== [$(date '+%F %T')] FULL RUN DONE ==="
echo "Stage A best : $A_BEST"
echo "Stage B best : $B_BEST"
