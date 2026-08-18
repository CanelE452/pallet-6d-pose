#!/usr/bin/env bash
# Watch a training run and send a Discord message when it reaches a target epoch,
# or when the process dies before getting there (early stop / crash).
#
# Runs detached from the session, so it keeps working after the terminal closes.
#
# Usage: notify_at_epoch.sh <results.csv> <target_epoch> [label]
set -uo pipefail

CSV="${1:?usage: notify_at_epoch.sh <results.csv> <target_epoch> [label]}"
TARGET="${2:?target epoch required}"
LABEL="${3:-stage_a}"
NOTIFY="$HOME/.claude/hooks/discord-notify.sh"

epochs() { echo $(( $(wc -l < "$CSV" 2>/dev/null || echo 1) - 1 )); }

metrics() {
  tail -1 "$CSV" 2>/dev/null | awk -F, '{printf "epoch %s | Box mAP50 %.4f mAP50-95 %.4f | Pose mAP50 %.4f mAP50-95 %.4f | %.1f h", $1, $11, $12, $15, $16, $2/3600}'
}

best_epoch() {
  # highest Pose mAP50-95 so far, and how many epochs ago it was
  awk -F, 'NR>1 && $16!="" {if ($16+0 > b) {b=$16+0; e=$1}} END {print e" "b}' "$CSV" 2>/dev/null
}

while true; do
  N=$(epochs)
  if [ "$N" -ge "$TARGET" ]; then
    read -r BE BV <<< "$(best_epoch)"
    "$NOTIFY" "[$LABEL] ${TARGET} epoch 도달
$(metrics)
best Pose mAP50-95 = ${BV} (epoch ${BE}) / patience 15
곡선 판단이 필요하면 Claude 에게 물어보세요."
    exit 0
  fi
  # Do NOT use `pgrep -f "yolo pose train"`: it matches any shell whose command line
  # contains that phrase, including watcher scripts like this one, so the trainer would
  # look alive forever. Match on the process's own comm/args instead. Note comm is
  # truncated to 15 chars and shows up as "yolo" here, not "python".
  if ! ps -eo comm,args --no-headers 2>/dev/null |
       awk '$1 ~ /^python|^yolo/ && /bin\/yolo pose train/ {f=1} END {exit !f}'; then
    read -r BE BV <<< "$(best_epoch)"
    "$NOTIFY" "[$LABEL] ${TARGET} epoch 도달 전에 학습이 끝났습니다 (조기종료 또는 크래시)
완료 ${N} epoch
$(metrics)
best Pose mAP50-95 = ${BV} (epoch ${BE})"
    exit 0
  fi
  sleep 120
done
