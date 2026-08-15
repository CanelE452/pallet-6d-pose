#!/usr/bin/env bash
# Waits for STAGE24 seen-eval, then runs held-out PAIRED evals:
#  - voting weights: heatmap-arm vs voting-seg arm (+ control heatmap) on
#    manifest(17-frame testset), filter-val, manual. by_visibility, order-free.
#  - sparse: eval_edgevec_refine control vs sparse-refine on filter-val, manual.
# Even if gate fails, these held-out numbers make the verdict honest/strong.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_MODULE_LOADING=LAZY PYTHONUNBUFFERED=1
L=data/pallet/eval_results/stage24_vec_newdata/logs
MAN=data/pallet/eval_results/stage22_myannot_eval/testset_full8_manifest.txt
run () { conda run -n pallet-pose python "$@"; }

for i in $(seq 1 90); do
  [ "$(cat "$L/STATUS2" 2>/dev/null)" = "SEEN_EVAL_DONE" ] && break
  sleep 30
done
[ "$(cat "$L/STATUS2" 2>/dev/null)" != "SEEN_EVAL_DONE" ] && { echo "seen-eval not done"; exit 1; }
echo "===== held-out paired evals $(date +%H:%M:%S) ====="

VW=weights/stage_screens/stage24_vec_newdata/voting/final_net_voting_unit.pth
CW=weights/stage_screens/stage24_vec_newdata/control/final_net_control_off.pth
SW=weights/stage_screens/stage24_vec_newdata/sparse/final_net_sparse_sparse.pth

# --- voting vs control: manifest(17), filter-val(outside), manual ---
run scripts/stage0/eval_harness/eval_pvnet_heads.py --weights "$VW" "$CW" \
  --vec_mode unit off --arm_tags VOTE CTRL --use_seg_mask --by_visibility \
  --evalset manifest --manifest "$MAN" --tag s24_manifest > "$L/eval_manifest.log" 2>&1
echo "----- voting manifest rc=$? -----"
run scripts/stage0/eval_harness/eval_pvnet_heads.py --weights "$VW" "$CW" \
  --vec_mode unit off --arm_tags VOTE CTRL --use_seg_mask --by_visibility \
  --evalset filter-val --domains outside night --tag s24_filterval > "$L/eval_filterval.log" 2>&1
echo "----- voting filter-val rc=$? -----"
run scripts/stage0/eval_harness/eval_pvnet_heads.py --weights "$VW" "$CW" \
  --vec_mode unit off --arm_tags VOTE CTRL --use_seg_mask --by_visibility \
  --evalset manual --tag s24_manual > "$L/eval_manual.log" 2>&1
echo "----- voting manual rc=$? -----"

# --- sparse: control vs sparse-refine (filter-val, manual) ---
for ES in filter-val manual; do
  run scripts/stage0/eval_harness/eval_edgevec_refine.py --control_weights "$CW" \
    --sparse_weights "$SW" --evalset "$ES" --tag s24 > "$L/eval_sparse_$ES.log" 2>&1
  echo "----- sparse $ES rc=$? -----"
done

echo "EVALALL_DONE" > "$L/STATUS3"
echo "===== EVALALL DONE $(date +%H:%M:%S) ====="
