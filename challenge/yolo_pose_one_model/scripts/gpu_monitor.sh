#!/usr/bin/env bash
# Sample GPU utilisation / memory / temperature / power while a training run is going.
# Writes CSV next to the run so the training report can quote real numbers.
#
# Usage: bash gpu_monitor.sh <out_csv> [interval_sec]
set -euo pipefail
OUT="${1:?usage: gpu_monitor.sh <out_csv> [interval_sec]}"
INT="${2:-30}"
mkdir -p "$(dirname "$OUT")"
echo "timestamp,util_pct,mem_used_mib,mem_total_mib,temp_c,power_w" > "$OUT"
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits -l "$INT" >> "$OUT"
