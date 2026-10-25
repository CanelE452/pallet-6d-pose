# V261 저장 head CPU replay

```bash
python3 -B /ABS/replay_saved_heads.py \
  --package /ABS/RESTORED_PACKAGE \
  --entry /ABS/TRUE_SAVED_REPLAY_ENTRY.json \
  --output-dir /ABS/FRESH_REPLAY_OUTPUT
```

설치된 NumPy와 PyTorch가 필요합니다. Root는 최종 결과가 저장된 뒤 false entry의 다섯 `relative_path/sha256/size_bytes`와 CODE SHA를 채우고 승인 flag를 true로 둡니다. package 밖의 새 output만 사용하며 원래 절대경로 fallback은 없습니다. 입력은 1784×1186 feature, input-only metadata NPZ, final head 두 개, 저장된 comparison ledger입니다. 별도 prediction 원장은 필요하지 않습니다. ledger의 양쪽 MAIN0 logit과 prediction을 대조합니다.

Torch는 `map_location=cpu, weights_only=True` checkpoint load에만 쓰고 네 tensor를 NumPy FP32로 계산합니다. MAIN0≥0은 E이며, unsupported 입력은 head bias로 분류하지 않고 U를 유지합니다. 고정 primary1237과 material별 D/E 2×3, precision/recall/U 및 기존 좌표오차≤3px 구간을 독립 집계합니다. SELF는 별도 응답 분포이며 SELF 정확도가 아닙니다. v61/legacy는 저장된 응답만 사용합니다.

원 RGB, encoder, 학습, optimizer, 새로운 matching/라벨/좌표/문턱 선택은 없습니다. 이미 반복 노출된 현재 평가의 저장 결과 재현이며 fresh blind 성능 증거가 아닙니다. 가짜 tensor와 metadata 검사만 준비 중 실행했고 실제 다섯 입력은 읽지 않았습니다.
