저장 SYN 출력으로 원본 픽셀 지표 재현
====================================

V269 폴더 구조를 유지하고 CPU에서 실행합니다. NumPy, SciPy, PyTorch, Pillow가 필요합니다.

```bash
python -B /다운로드한/V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE/analyze_syn_pixels_v269.py
```

이미 `syn_mask_diagnostic_v1` 결과가 있으면 235장 전체를 다시 계산하고 기존 수치와 정확히 일치하는지 검증합니다. 이 재검증에는 원본 RGB/M4나 모델이 필요하지 않습니다. 기존 결과가 없는 첫 실행에서는 이 폴더의 `raw6`에 포함된 원본 RGB/M4 여섯 쌍으로 고정 시각화를 함께 생성합니다.

함께 유지할 파일은 이 폴더의 `REPLAY_INPUTS.json`, `SYN_DEV_M4_PACKBITS.npz`, `OPENAI_FULL_CLIP_DEV_LOGITS.npz`, `V267_CS_DEV_LOGITS.npz`와 V269의 `syn_training_v1/COMPLETE.json`, `ALL_CHECKPOINTS_FROZEN_BEFORE_SYN_DEV.json`, 세 `{EVA_STANDARD,EVA_CSA,EVA_DECLIP}_SYN_DEV_MASKS.npz`입니다. 복사된 원본 데이터의 절대 경로는 출처 기록이며 재현할 때 접근하지 않습니다. 입력 파일 SHA를 검사합니다.

M4 packbits는 원본 마스크 `uint8>127`을 원래 H/W 그대로 손실 없이 저장한 정답입니다. 각 모델의 raw32 logit을 원본 H/W로 bilinear 복원(`align_corners=False`)하고 `logit>=0`으로 이진화합니다. 경계는 이진 마스크에서 3×3 침식을 뺀 결과이며 영상 밖은0입니다. 원본 거리1px·3px에서 모두 precision/recall/F1을 계산합니다. 학습용32 soft GT를 확대해서 정답으로 사용하지 않습니다.

결과 키는 `OPENAI_FULL_CLIP`, `V267_CS`, `EVA_STANDARD`, `EVA_CSA`, `EVA_DECLIP`입니다. `micro_IoU`, `micro_Dice`, `boundary1px`, `boundary3px`를 비교할 수 있습니다. 공개 DeCLIP 백본의 COCO 무라벨 증류 이력과 새 mask head의 합성 전용 학습은 구분해야 합니다. 이 결과는 합성 픽셀 분할 성능이며 실사 키포인트 보임/외부 가림 성능을 뜻하지 않습니다.
