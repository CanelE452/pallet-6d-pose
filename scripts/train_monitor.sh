#!/bin/bash
# 학습 모니터 — GPU + 현재 epoch 를 깜빡임 없이 표시
# 사용법: bash scripts/train_monitor.sh [로그파일] [총epoch]
# 종료: Ctrl+C

LOG="${1:-weights/paper_base_v2_train.log}"
TOTAL_EPOCH="${2:-60}"
INTERVAL=2

# 프로젝트 루트 기준으로 로그 경로 보정
cd "$(dirname "$0")/.." || exit 1

tput civis              # 커서 숨김
clear
trap 'tput cnorm; echo; exit 0' INT TERM   # 종료 시 커서 복원

draw() {
  tput cup 0 0          # 화면 지우지 않고 커서만 홈으로 → 깜빡임 없음

  # --- GPU ---
  read -r GUTIL GMEM_U GMEM_T GTEMP GPOW < <(
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')

  # --- 학습 진행 (로그 마지막 Train Epoch 라인 파싱) ---
  local last ep cur tot pct loss
  last=$(grep "Train Epoch:" "$LOG" 2>/dev/null | tail -1)
  ep=$(echo "$last"   | grep -oE "Train Epoch: [0-9]+" | grep -oE "[0-9]+")
  cur=$(echo "$last"  | grep -oE "\[[0-9]+/" | grep -oE "[0-9]+")
  tot=$(echo "$last"  | grep -oE "/[0-9]+ \(" | grep -oE "[0-9]+")
  pct=$(echo "$last"  | grep -oE "\([0-9]+%\)")
  loss=$(echo "$last" | grep -oE "Loss: [0-9.]+" | grep -oE "[0-9.]+" | cut -c1-7)

  # 학습 프로세스 살아있는지
  local alive
  if pgrep -f "train.py.*paper_base_v2\|train_dope" >/dev/null 2>&1 || pgrep -f "train.py" >/dev/null 2>&1; then
    alive="\033[32m● RUNNING\033[0m"
  else
    alive="\033[31m● STOPPED\033[0m"
  fi

  printf "\033[1m  학습 모니터  (Ctrl+C 종료)\033[0m                 \n"
  printf "  ─────────────────────────────────────────────\n"
  printf "  상태   : %b      \n" "$alive"
  printf "  Epoch  : \033[1;36m%s / %s\033[0m   (배치 %s/%s %s)        \n" \
         "${ep:-?}" "$TOTAL_EPOCH" "${cur:-?}" "${tot:-?}" "${pct:-}"
  printf "  Loss   : %s        \n" "${loss:-?}"
  printf "  ─────────────────────────────────────────────\n"
  printf "  GPU    : \033[1;33m%s%%\033[0m 사용   %s°C   %sW        \n" \
         "${GUTIL:-?}" "${GTEMP:-?}" "${GPOW:-?}"
  printf "  VRAM   : %s / %s MiB        \n" "${GMEM_U:-?}" "${GMEM_T:-?}"
  printf "  ─────────────────────────────────────────────\n"
  printf "  갱신 %ss 주기                          \n" "$INTERVAL"
}

while true; do
  draw
  sleep "$INTERVAL"
done
