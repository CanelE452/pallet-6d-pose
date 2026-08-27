#!/usr/bin/env bash
# 이미 도는 run_paper_generic.sh 의 뒤를 잇는다.  그 파일을 편집하지 않는다
# (실행 중인 bash 는 스크립트를 증분으로 읽어서, 고치면 그 자리에서 깨진다).
set -uo pipefail
ROOT=/home/minjae/Documents/github/pallet-pose
YR=$ROOT/challenge/yolo_pose_one_model
PIPE=$YR/paper_generic_pipeline
NOTIFY=$HOME/.claude/hooks/discord-notify.sh
export PYTHONUNBUFFERED=1
source "$(conda info --base)/etc/profile.d/conda.sh"
say(){ echo "[$(date +%H:%M:%S)] $*"; }

# 앞 드라이버가 판정 JSON 을 낼 때까지 기다린다 (최대 8h)
for i in $(seq 1 480); do
  [ -f "$YR/evaluation/PAPER_YOLO_VERDICT.json" ] && break
  sleep 60
done
if [ ! -f "$YR/evaluation/PAPER_YOLO_VERDICT.json" ]; then
  "$NOTIFY" "⚠️ PAPER_GENERIC 후속: 8h 안에 판정 JSON 이 안 나옴" || true; exit 1
fi
sleep 20
conda activate pallet-pose
say "기전 분석"
python "$PIPE/mechanism_analysis.py" || say "기전 분석 실패 (계속)"
say "판정 + 다음 행동 알림"
python "$PIPE/verdict_driver.py"
say done
