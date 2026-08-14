#!/usr/bin/env bash
# Reproducible PAPER_S2 single-RGB improvement screen/full fine-tune driver.
#
# Safety defaults:
#   * dry-run unless --execute is supplied;
#   * fixed input=400 and belief grid=50;
#   * existing Stage-B data only, initialized from ep57;
#   * synthetic val + filterval development selection only (sealed final-test is
#     never enumerated); handannot17 is reported but excluded from selection.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEFAULT_CONFIG="$ROOT/config/paper_s2_rgb1_improve_v1.json"
EVALUATOR="$ROOT/scripts/stage0/paper_s2/paper_s2_rgb1_eval.py"
TRAIN="$ROOT/Deep_Object_Pose/train/train.py"

usage() {
  cat <<'EOF'
Usage:
  scripts/stage0/_run/run_paper_s2_rgb1_improve.sh plan [options]
  scripts/stage0/_run/run_paper_s2_rgb1_improve.sh quick [options]
  scripts/stage0/_run/run_paper_s2_rgb1_improve.sh full --arm ARM [options]
  scripts/stage0/_run/run_paper_s2_rgb1_improve.sh eval --phase quick|full [options]

Commands:
  plan    Validate locks and print quick-screen and full-run commands. Never runs.
  quick   Train selected quick arms and evaluate/select them. Dry-run by default.
  full    Fine-tune one selected arm for the full budget and evaluate it. Dry-run.
  eval    Evaluate existing checkpoints only. Dry-run by default.

Options:
  --config PATH       Experiment JSON (default: config/paper_s2_rgb1_improve_v1.json)
  --arms CSV          Quick/eval arms (default: every arm in the config)
  --arm ARM           Full-run arm selected after the quick screen
  --phase PHASE       eval phase: quick or full
  --execute           Actually launch commands (otherwise commands are printed)
  --skip-eval         Train only; do not invoke the evaluator
  --force-existing    Allow writing into a non-empty incomplete arm directory
  -h, --help          Show this help

Environment overrides:
  PAPER_S2_PYTHON     Python executable after environment activation (default: python)

Examples:
  # Inspect exact commands and paths (safe; no writes/training):
  scripts/stage0/_run/run_paper_s2_rgb1_improve.sh plan

  # Run the complete quick screen after smoke tests pass:
  scripts/stage0/_run/run_paper_s2_rgb1_improve.sh quick --execute

  # Run the selected full arm (example only; use quick report's BEST_ARM):
  scripts/stage0/_run/run_paper_s2_rgb1_improve.sh full --arm full --execute
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 2
}

print_cmd() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

json_scalar() {
  local expr=$1
  "$PYTHON_BIN" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {"__builtins__": {}}, {"d": d}))' \
    "$CONFIG" "$expr"
}

json_list() {
  local expr=$1
  "$PYTHON_BIN" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); v=eval(sys.argv[2], {"__builtins__": {}}, {"d": d}); sys.stdout.write("\n".join(map(str,v)))' \
    "$CONFIG" "$expr"
}

ACTION=${1:-help}
if [[ $# -gt 0 ]]; then
  shift
fi

CONFIG="$DEFAULT_CONFIG"
ARMS_CSV=""
FULL_ARM=""
PHASE=""
EXECUTE=0
SKIP_EVAL=0
FORCE_EXISTING=0
PYTHON_BIN=${PAPER_S2_PYTHON:-python}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) [[ $# -ge 2 ]] || die "--config needs a path"; CONFIG=$2; shift 2 ;;
    --arms) [[ $# -ge 2 ]] || die "--arms needs CSV"; ARMS_CSV=$2; shift 2 ;;
    --arm) [[ $# -ge 2 ]] || die "--arm needs a name"; FULL_ARM=$2; shift 2 ;;
    --phase) [[ $# -ge 2 ]] || die "--phase needs quick or full"; PHASE=$2; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --force-existing) FORCE_EXISTING=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

case "$ACTION" in
  help|-h|--help) usage; exit 0 ;;
  plan|quick|full|eval) ;;
  *) die "unknown command: $ACTION (try --help)" ;;
esac

[[ -f "$CONFIG" ]] || die "config not found: $CONFIG"
[[ -f "$TRAIN" ]] || die "trainer not found: $TRAIN"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python not found: $PYTHON_BIN"

BASE_REL=$(json_scalar 'd["locked"]["base_checkpoint"]')
EXPERIMENT=$(json_scalar 'd["experiment"]')
BASE="$ROOT/$BASE_REL"
TEACHER_REL=$(json_scalar 'd["locked"].get("teacher_checkpoint", "")')
TEACHER="${TEACHER_REL:+$ROOT/$TEACHER_REL}"
BASE_EP=$(json_scalar 'd["locked"]["base_epoch"]')
INPUT_SIZE=$(json_scalar 'd["locked"]["input_size"]')
BELIEF_SIZE=$(json_scalar 'd["locked"]["belief_size"]')
SIGMA=$(json_scalar 'd["locked"]["sigma"]')
SEED=$(json_scalar 'd["locked"]["seed"]')
BS=$(json_scalar 'd["locked"]["batch_size"]')
WORKERS=$(json_scalar 'd["locked"]["workers"]')
LR=$(json_scalar 'd["locked"]["learning_rate"]')
QUICK_N=$(json_scalar 'd["locked"]["quick_epoch_size"]')
QUICK_D=$(json_scalar 'd["locked"]["quick_delta_epochs"]')
FULL_N=$(json_scalar 'd["locked"]["full_epoch_size"]')
FULL_D=$(json_scalar 'd["locked"]["full_delta_epochs"]')
BALANCE=$(json_scalar 'd["locked"]["balance_groups"]')
IDX_REL=$(json_scalar 'd["locked"].get("diffpnp_index_dir", "")')
DATA_ROOT_REL=$(json_scalar 'd["locked"].get("diffpnp_root", "")')
IDX="${IDX_REL:+$ROOT/$IDX_REL}"
DATA_ROOT="${DATA_ROOT_REL:+$ROOT/$DATA_ROOT_REL}"
WEIGHTS_ROOT="$ROOT/$(json_scalar 'd["locked"]["weights_namespace"]')"
RESULTS_ROOT="$ROOT/$(json_scalar 'd["locked"]["results_namespace"]')"
EVAL_ROOT="$ROOT/$(json_scalar 'd["locked"]["eval_namespace"]')"
DEV_REAL=$(json_scalar 'd["evaluation"]["development_real"]')
SECONDARY_REAL=$(json_scalar 'd["evaluation"]["secondary_real"]')

mapfile -t DATA_RELS < <(json_list 'd["locked"]["data"]')
mapfile -t COMMON_FLAGS < <(json_list 'd["common_train_flags"]')
mapfile -t REFINE_SCHEDULE_FLAGS < <(json_list 'd["refinement_train_flags"]["schedule"]')
mapfile -t REFINE_QUALITY_FLAGS < <(json_list 'd["refinement_train_flags"]["quality"]')
mapfile -t REFINE_EXTENT_FLAGS < <(json_list 'd["refinement_train_flags"]["extent"]')
mapfile -t REFINE_SPAN_FLAGS < <(json_list 'd["refinement_train_flags"].get("span", [])')
mapfile -t REFINE_SIGNED_FLAGS < <(json_list 'd["refinement_train_flags"].get("signed", [])')
mapfile -t ALL_ARMS < <(json_list 'd["arms"].keys()')

ENABLE_DIFFPNP=0
for token in "${COMMON_FLAGS[@]}"; do
  if [[ "$token" == "--diffpnp" ]]; then
    ENABLE_DIFFPNP=1
    break
  fi
done

if [[ -z "$ARMS_CSV" ]]; then
  ARMS=("${ALL_ARMS[@]}")
else
  IFS=',' read -r -a ARMS <<< "$ARMS_CSV"
fi

arm_exists() {
  local wanted=$1 arm
  for arm in "${ALL_ARMS[@]}"; do
    [[ "$arm" == "$wanted" ]] && return 0
  done
  return 1
}

for arm in "${ARMS[@]}"; do
  arm_exists "$arm" || die "arm '$arm' is not in $CONFIG"
done
if [[ -n "$FULL_ARM" ]]; then
  arm_exists "$FULL_ARM" || die "arm '$FULL_ARM' is not in $CONFIG"
fi

arm_flags() {
  local arm=$1
  json_list 'd["arms"]['"\"$arm\""']["train_flags"]'
}

# -------- invariant and leakage preflight --------
[[ -f "$BASE" ]] || die "base checkpoint missing: $BASE"
if [[ -n "$TEACHER" ]]; then
  [[ -f "$TEACHER" ]] || die "teacher checkpoint missing: $TEACHER"
  [[ "$(realpath "$TEACHER")" == "$(realpath "$BASE")" ]] || \
    die "teacher must equal the locked base checkpoint"
fi
[[ "$BASE_EP" == 57 ]] || die "base epoch drift: expected 57, got $BASE_EP"
[[ "$BASE" =~ 0057\.pth$ ]] || die "base filename is not ep57: $BASE"
[[ "$INPUT_SIZE" == 400 ]] || die "input lock drift: expected 400, got $INPUT_SIZE"
[[ "$BELIEF_SIZE" == 50 ]] || die "belief lock drift: expected 50, got $BELIEF_SIZE"
[[ "$QUICK_N" == 6000 ]] || die "quick epoch_size lock drift: expected 6000, got $QUICK_N"
EXPECTED_QUICK_D=6
if [[ "$EXPERIMENT" == "paper_s2_rgb1_frozen_fusion_v1" ]]; then
  EXPECTED_QUICK_D=3
elif [[ "$EXPERIMENT" == "paper_s2_rgb1_projected_span_v1" ]]; then
  EXPECTED_QUICK_D=3
elif [[ "$EXPERIMENT" == "paper_s2_rgb1_hard_edge_v2" ]]; then
  EXPECTED_QUICK_D=3
elif [[ "$EXPERIMENT" == "paper_s2_selftrain_full7_pl19_v1" ]]; then
  EXPECTED_QUICK_D=3
elif [[ "$EXPERIMENT" == "paper_s2_teacher_hardtail_v5" ]]; then
  EXPECTED_QUICK_D=3
fi
[[ "$QUICK_D" == "$EXPECTED_QUICK_D" ]] || \
  die "quick delta lock drift for $EXPERIMENT: expected $EXPECTED_QUICK_D, got $QUICK_D"
if [[ "$EXPERIMENT" == "paper_s2_selftrain_full7_pl19_v1" \
      && "$ENABLE_DIFFPNP" -ne 0 ]]; then
  die "FULL7 PL19 self-training must not enable DiffPnP"
fi
[[ "$FULL_D" == 15 ]] || die "full delta lock drift: expected 15, got $FULL_D"
grep -Eq 'output_size[[:space:]]*=[[:space:]]*50' "$TRAIN" || \
  die "trainer no longer visibly locks output_size=50"

# Every configured switch must exist in the current trainer source. This catches
# stale experiment JSON before GPU work begins without importing CUDA for a plan.
CONFIGURED_FLAGS=(
  "${COMMON_FLAGS[@]}"
  "${REFINE_SCHEDULE_FLAGS[@]}"
  "${REFINE_QUALITY_FLAGS[@]}"
  "${REFINE_EXTENT_FLAGS[@]}"
  "${REFINE_SPAN_FLAGS[@]}"
  "${REFINE_SIGNED_FLAGS[@]}"
)
for arm in "${ALL_ARMS[@]}"; do
  mapfile -t CHECK_FLAGS < <(arm_flags "$arm")
  CONFIGURED_FLAGS+=("${CHECK_FLAGS[@]}")
done
for token in "${CONFIGURED_FLAGS[@]}"; do
  case "$token" in
    --wd_rank_head|--wd_axis_head|--heatmap_pnp_enhance)
      die "W/D training/prior is forbidden in this experiment: $token"
      ;;
  esac
  if [[ "$token" == --* ]]; then
    grep -Fq -- "\"$token\"" "$TRAIN" || \
      die "configured flag is not declared by train.py: $token"
  fi
done

DATA=()
for rel in "${DATA_RELS[@]}"; do
  case "$rel" in
    data/pallet/training_data/*) ;;
    *) die "non-approved training path in config: $rel" ;;
  esac
  case "$rel" in
    *raw_data*|*/scan|*/scan/*) die "raw_data/scan is forbidden: $rel" ;;
  esac
  abs="$ROOT/$rel"
  [[ -d "$abs" ]] || die "training directory missing: $abs"
  DATA+=("$abs")
done
if [[ "$ENABLE_DIFFPNP" -eq 1 ]]; then
  [[ -n "$IDX" && -d "$IDX" ]] || die "DiffPnP index missing: $IDX"
  [[ -n "$DATA_ROOT" && -d "$DATA_ROOT" ]] || \
    die "DiffPnP data root missing: $DATA_ROOT"
fi

EXCLUDES=$(json_list 'd["evaluation"]["selection_excludes"]')
grep -qx 'final_test' <<< "$EXCLUDES" || die "config must explicitly exclude final_test"
grep -qx 'handannot17' <<< "$EXCLUDES" || \
  die "handannot17 must remain report-only, not a selection set"

QUICK_TARGET=$((BASE_EP + QUICK_D))
FULL_TARGET=$((BASE_EP + FULL_D))

activate_env() {
  if [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-}" == "pallet-pose" ]]; then
    return
  fi
  if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate pallet-pose
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate pallet-pose
  else
    die "pallet-pose conda environment could not be activated"
  fi
}

build_train_cmd() {
  local phase=$1 arm=$2 target=$3 epoch_size=$4 out=$5
  local -n _dest=$6
  local -a EXTRA=()
  local -a REFINE_FOR_ARM=()
  local use_quality=0 use_extent=0 use_span=0 use_signed=0
  mapfile -t EXTRA < <(arm_flags "$arm")
  for token in "${EXTRA[@]}"; do
    case "$token" in
      --corner_quality) use_quality=1 ;;
      --extent_loss) use_extent=1 ;;
      --projected_span_loss) use_span=1 ;;
      --signed_footprint_loss) use_signed=1 ;;
    esac
  done
  if [[ "$use_quality" -eq 1 || "$use_extent" -eq 1 || "$use_span" -eq 1 \
        || "$use_signed" -eq 1 ]]; then
    REFINE_FOR_ARM+=("${REFINE_SCHEDULE_FLAGS[@]}")
  fi
  if [[ "$use_quality" -eq 1 ]]; then
    REFINE_FOR_ARM+=("${REFINE_QUALITY_FLAGS[@]}")
  fi
  if [[ "$use_extent" -eq 1 ]]; then
    REFINE_FOR_ARM+=("${REFINE_EXTENT_FLAGS[@]}")
  fi
  if [[ "$use_span" -eq 1 ]]; then
    REFINE_FOR_ARM+=("${REFINE_SPAN_FLAGS[@]}")
  fi
  if [[ "$use_signed" -eq 1 ]]; then
    REFINE_FOR_ARM+=("${REFINE_SIGNED_FLAGS[@]}")
  fi
  _dest=(
    "$PYTHON_BIN" -u "$TRAIN"
    --data "${DATA[@]}"
    --object pallet
    --batchsize "$BS" --sigma "$SIGMA" --imagesize "$INPUT_SIZE"
    --lr "$LR" --epochs "$target" --epoch_size "$epoch_size"
    --workers "$WORKERS" --manualseed "$SEED"
    --save_every 1 --nb_checkpoints 0 --namefile epoch
    --net_path "$BASE" --outf "$out"
    --balance_groups "$BALANCE"
  )
  if [[ "$ENABLE_DIFFPNP" -eq 1 ]]; then
    _dest+=(--diffpnp_index_dir "$IDX" --diffpnp_root "$DATA_ROOT")
  fi
  if [[ -n "$TEACHER" ]]; then
    _dest+=(--teacher_checkpoint "$TEACHER")
  fi
  _dest+=("${COMMON_FLAGS[@]}" "${REFINE_FOR_ARM[@]}" "${EXTRA[@]}")
}

run_training_arm() {
  local phase=$1 arm=$2 target=$3 epoch_size=$4
  local out="$WEIGHTS_ROOT/$phase/$arm"
  local log="$RESULTS_ROOT/logs/${phase}_${arm}_train.log"
  local final
  final=$(printf '%s/final_net_epoch_%04d.pth' "$out" "$target")
  local -a cmd
  build_train_cmd "$phase" "$arm" "$target" "$epoch_size" "$out" cmd

  echo "[$phase/$arm] target=ep$target epoch_size=$epoch_size"
  print_cmd "${cmd[@]}"
  echo "  log=$log"
  if [[ "$EXECUTE" -eq 0 || "$ACTION" == plan ]]; then
    return
  fi
  if [[ -f "$final" ]]; then
    echo "[skip] completed checkpoint exists: $final"
    return
  fi
  if [[ -d "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit)" \
        && "$FORCE_EXISTING" -eq 0 ]]; then
    die "incomplete non-empty directory: $out (inspect it or use --force-existing)"
  fi
  mkdir -p "$out" "$(dirname "$log")" "$RESULTS_ROOT/manifests"
  cp "$CONFIG" "$RESULTS_ROOT/manifests/${phase}_config_snapshot.json"
  activate_env
  (
    cd "$ROOT/Deep_Object_Pose/train"
    "${cmd[@]}"
  ) >"$log" 2>&1
  [[ -f "$final" ]] || die "training returned without expected checkpoint: $final"
  echo "[done] $phase/$arm -> $final"
}

run_eval() {
  local phase=$1
  shift
  local selected=("$@")
  local arms_arg
  arms_arg=$(IFS=,; echo "${selected[*]}")
  local synth=q1_500
  [[ "$phase" == full ]] && synth=val_1500
  local -a cmd=(
    "$PYTHON_BIN" -u "$EVALUATOR"
    --config "$CONFIG" --phase "$phase" --synthetic "$synth"
    --weights-root "$WEIGHTS_ROOT/$phase"
    --out-dir "$EVAL_ROOT/$phase"
    --arms "$arms_arg"
  )
  echo "[eval/$phase] synthetic=$synth real=$DEV_REAL+$SECONDARY_REAL(report-only)"
  print_cmd "${cmd[@]}"
  if [[ "$EXECUTE" -eq 0 || "$ACTION" == plan ]]; then
    return
  fi
  [[ -f "$EVALUATOR" ]] || die "evaluator not found: $EVALUATOR"
  mkdir -p "$EVAL_ROOT/$phase" "$RESULTS_ROOT/logs"
  activate_env
  "${cmd[@]}" >"$RESULTS_ROOT/logs/${phase}_eval.log" 2>&1
  echo "[done] eval/$phase -> $EVAL_ROOT/$phase"
}

echo "[preflight] PASS config=$CONFIG"
echo "[locks] base=ep$BASE_EP input=${INPUT_SIZE} belief=${BELIEF_SIZE} sigma=$SIGMA"
echo "[paths] weights=$WEIGHTS_ROOT results=$RESULTS_ROOT eval=$EVAL_ROOT"
echo "[leakage] selection=synthetic+$DEV_REAL; $SECONDARY_REAL=report-only; final-test=sealed"

case "$ACTION" in
  plan)
    echo "=== QUICK SCREEN PLAN ==="
    for arm in "${ARMS[@]}"; do
      run_training_arm quick "$arm" "$QUICK_TARGET" "$QUICK_N"
    done
    if [[ "$SKIP_EVAL" -eq 0 ]]; then
      run_eval quick "${ARMS[@]}"
    fi
    echo "=== FULL TEMPLATE (choose --arm after quick report) ==="
    if [[ -n "$FULL_ARM" ]]; then
      template_arm=$FULL_ARM
    elif arm_exists full; then
      template_arm=full
    else
      template_arm=${ARMS[0]}
    fi
    run_training_arm full "$template_arm" "$FULL_TARGET" "$FULL_N"
    if [[ "$SKIP_EVAL" -eq 0 ]]; then
      run_eval full "$template_arm"
    fi
    ;;
  quick)
    for arm in "${ARMS[@]}"; do
      run_training_arm quick "$arm" "$QUICK_TARGET" "$QUICK_N"
    done
    if [[ "$SKIP_EVAL" -eq 0 ]]; then
      run_eval quick "${ARMS[@]}"
    fi
    ;;
  full)
    [[ -n "$FULL_ARM" ]] || die "full requires --arm ARM from the quick report"
    run_training_arm full "$FULL_ARM" "$FULL_TARGET" "$FULL_N"
    if [[ "$SKIP_EVAL" -eq 0 ]]; then
      run_eval full "$FULL_ARM"
    fi
    ;;
  eval)
    [[ "$PHASE" == quick || "$PHASE" == full ]] || \
      die "eval requires --phase quick or --phase full"
    run_eval "$PHASE" "${ARMS[@]}"
    ;;
esac
