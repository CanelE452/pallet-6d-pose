저장 출력으로 합성 DEV 235장의 원본 픽셀 지표 재현
=================================================

V268 폴더 어디에 있든 아래 명령으로 실행할 수 있습니다. Python에 NumPy, SciPy, PyTorch가 필요하며 CPU만 사용합니다.

```bash
python -B /다운로드한/V268_FULL_IMAGE_BOUNDARY_REFINEMENT/replay_syn_pixels_v268.py
```

함께 유지해야 하는 파일은 다음과 같습니다.

- 이 폴더의 `REPLAY_INPUTS.json`, `SYN_DEV_M4_PACKBITS.npz`, `OLD_CS32_DEV_LOGITS.npz`
- V268의 `syn_training_v1/HIGHRES_REFINE_SYN_DEV_MASKS.npz`
- V268의 `syn_training_v1/BOUNDARY_REFINE_SYN_DEV_MASKS.npz`

M4 fixture는 기존 SYN DEV 원본 마스크의 `uint8 > 127` 이진 정답을 가변 크기 그대로 packbits로 손실 없이 저장합니다. 각 원본 크기·프레임 번호·마스크 원본 SHA도 포함합니다. 원본 데이터의 절대 경로는 `REPLAY_INPUTS.json`의 출처 기록에만 남으며 재현 코드가 접근하지 않습니다. 새 두 마스크 파일은 각각 약53MB입니다.

재현 코드는 입력 SHA를 검증하고, 저장된 32/256 격자 raw logit을 각 원본 H/W로 bilinear 복원한 뒤 `logit >= 0`으로 판정합니다. 경계는 마스크에서 3×3 침식 결과를 뺀 내부 경계이며, 영상 밖은0입니다. 원본 픽셀 거리1px와3px 기준을 모두 계산합니다. 출력 `REPLAY_RESULT.json`은 235장 전체의 혼동행렬, IoU, Dice, 경계 precision/recall/F1이 기존 봉인 보고서와 수치상 일치하는지 검증합니다. 다시 실행할 때 기존 결과가 같으면 유지합니다.

학습·모델 추론·실사 데이터·원본 RGB·원본 M4 파일이 필요하지 않습니다. 이 지표는 합성 분할 평가이며 실사 키포인트 보임/외부 가림 평가를 대신하지 않습니다.
