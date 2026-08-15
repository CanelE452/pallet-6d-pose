#!/usr/bin/env bash
# 듀얼부트 우분투에서 실행. 윈도우 파티션을 마운트한 뒤 KEEP 목록만 복사한다.
# 사용법:
#   1) git clone https://github.com/CanelE452/pallet-6d-pose.git ~/FoundationPose
#   2) 윈도우 파티션 마운트 경로 확인 (lsblk -f) 후 아래 SRC 수정
#   3) bash migrate_keep.sh
set -e

# ===== 환경에 맞게 수정 =====
SRC="/mnt/win/Users/minjae/Documents/github/FoundationPose"   # 윈도우 파티션 안 repo 경로
DST="$HOME/FoundationPose"                                      # 우분투에 clone 한 repo 경로
# ===========================

if [ ! -d "$SRC" ]; then echo "ERROR: SRC 경로 없음 → $SRC (마운트/경로 확인)"; exit 1; fi
if [ ! -d "$DST" ]; then echo "ERROR: DST 경로 없음 → $DST (먼저 git clone)"; exit 1; fi

mkdir -p "$DST/weights" "$DST/challenge/weights" \
         "$DST/data/pallet/raw_data" "$DST/data/pallet/training_data"

echo "===== [1/4] weights (challenge 라인) ====="
rsync -avP \
  "$SRC/weights/challenge_track/challengenight" \
  "$SRC/weights/challenge_track/challenge0123" \
  "$SRC/weights/challenge_track/challenge0123_ft_manual" \
  "$SRC/weights/selftrain/r1_outside_loo" \
  "$SRC/weights/misc/f5_noapril_ransac_loo_realonly" \
  "$DST/weights/"
rsync -avP "$SRC/challenge/weights/baseline_v8_A.pth" "$DST/challenge/weights/"

echo "===== [2/4] data — raw 원본 + 평가셋 ====="
rsync -avP "$SRC/data/night" "$SRC/data/outside" "$DST/data/"
rsync -avP \
  "$SRC/data/pallet/raw_data/capture0403middle" \
  "$SRC/data/pallet/raw_data/capture0403noapril" \
  "$SRC/data/pallet/raw_data/real_data" \
  "$SRC/data/pallet/raw_data/models_usd" \
  "$DST/data/pallet/raw_data/"
rsync -avP "$SRC/data/pallet/scan_cleanup" "$DST/data/pallet/"
rsync -avP "$SRC/data/_eval_sets" "$DST/data/"

echo "===== [3/4] data — 학습 base (mixed_v8_train) ====="
rsync -avP "$SRC/data/pallet/training_data/mixed_v8_train" "$DST/data/pallet/training_data/"

echo "===== [4/4] git 미커밋(untracked) 코드 ====="
# scripts 폴더는 통째 동기화 (tracked 는 내용 동일, untracked 만 추가됨)
rsync -avP "$SRC/challenge/yolo_pose" "$DST/challenge/"
rsync -avP "$SRC/challenge/_docs"     "$DST/challenge/"
rsync -avP "$SRC/scripts/data_prep/eval/"      "$DST/scripts/data_prep/eval/"
rsync -avP "$SRC/scripts/data_prep/inference/" "$DST/scripts/data_prep/inference/"

echo ""
echo "===== 복사 완료 ====="
du -sh "$DST/weights" "$DST/data" 2>/dev/null
