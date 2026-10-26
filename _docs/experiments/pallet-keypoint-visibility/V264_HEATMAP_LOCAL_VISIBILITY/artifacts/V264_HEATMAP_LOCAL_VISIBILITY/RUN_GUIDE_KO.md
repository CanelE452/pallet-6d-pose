# V264 재현 안내

Heatmap 예측 위치 주변의 DINO 공간 특징으로 보임/외부 가림을 판단하는 분류기를 합성으로 학습했습니다. 같은 초기값·배치·모델 크기·업데이트 수를 쓰는 전체 평균 대조군과 비교했습니다. 위치 모델·crop·키포인트 대응·좌표는 고정했고 PnP는 실행하지 않았습니다.

## 결과

합성 DEV에서 외부 가림 정답은 50→54/56점, 보임의 가림 오판은 66→9점으로 개선됐습니다. 실사 PRIMARY D1,147/E90점에서는 보임 정답645→1,028, 외부 가림 정답71→23점입니다. 보임 오판을 줄이는 장점은 있지만 외부 가림 누락이 늘어 **새 모델을 채택하지 않았습니다.**

실사223장은 이번 최적화·normalizer·checkpoint·threshold·설정 선택에 쓰지 않았습니다. 과거 반복 확인한 개발 평가 자료이며 독립 검증은 아닙니다. 자기 가림은 D/E 정답에서 제외하고 별도 응답 표로 집계했습니다.

## 패키지 안에서 가능한 CPU 재현

`artifacts/V264_HEATMAP_LOCAL_VISIBILITY` 폴더에서 실행합니다. 분류기 재현은 Python3.10/PyTorch2.1.1/NumPy1.26.4에서 확인했습니다. 첫 명령은 원래 RGB·DINO 모델·GB 토큰·GPU 없이 제공한 pooled 특징과 최종 분류기로 합성/실사 판정을 다시 계산합니다. 둘째는 표준 라이브러리만으로 제공한 행별 표에서 96개 평가 집계를 다시 계산합니다.

```bash
python -B replay_compact_v264.py --output VERIFY_COMPACT_REPLAY.json
python -B replay_metrics_v264.py --output VERIFY_METRICS_REPLAY.json
```

출력 이름은 새 파일이어야 합니다. 이미 공개한 검증 결과를 덮어쓰지 않습니다. `VERIFY_COMPACT.json`은 최초 검증 때의 코드 바인딩을 보존하며, 마지막 버전의 재현 스크립트는 새 출력 파일 이름을 지정할 수 있습니다. 두 명령은 학습하지 않습니다.

검증 항목은 다음과 같습니다.

- 모든 압축 특징·checkpoint·예측·평가 입력의 SHA256/크기.
- 두 분류기의 동일 seed17 초기값, 동일 FIT 배치1,000×32, 매 배치 D16/E16, 동일25,025 parameter와 최종1,000업데이트 로그.
- 합성 DEV·실사1,784점 판정 완전 일치, CPU/GPU logits 차이≤1e-5.
- 96개 혼동행렬·분모·precision/recall/F1·보류율 등 집계 일치.

## 실제 학습과 특징 추출 경로

`PROTOCOL_BEFORE_FIT.json`에 실제 코드와 원본 입력을 바인딩했습니다. `train_syn_v264.py`가 수행한 학습은 두 번뿐이며 각 최종1,000업데이트를 사용했습니다. `syn_training_v1/BOTH_CHECKPOINTS_FROZEN_BEFORE_SYN_DEV.json` 이후 합성 DEV 분류기 추론을 했고, `NEW_CONFIG_FROZEN_BEFORE_CURRENT.json` 이후에만 실사 토큰을 읽었습니다. 모든 실사 예측을 `CURRENT_PREDICTIONS_FROZEN.json`으로 봉인한 뒤 평가 주석을 결합했습니다.

합성 원자료는 기존 V263과 같은 FIT86 base/DEV24 base입니다. FIT의 D4,626/E190 라벨 중 입력 불가54D/2E를 제외해 D4,572/E188을 표집했습니다. E190은 source family 중복을 포함하고 고유(base,native,variant)95개·encoded RGB93개입니다. DEV에는 native6/7의 외부 가림 사례가 없습니다.

`SYN_POOLED_FEATURES.npz`와 실사 pooled 특징을 제공하므로 새 분류기 forward는 패키지만으로 재현할 수 있습니다. 원본 DINO 토큰 전체와 실사 crop 전체는 새 패키지에 중복 복제하지 않았습니다. 토큰/RGB부터의 재생성에는 `SOURCE_DEPENDENCIES.json`의 V263 자료·추출 코드가 필요합니다. 원래 실행 스크립트의 절대 경로는 출처 기록이며, 휴대용 재현 스크립트는 명시된 패키지 상대 경로로 읽습니다.

원래 GPU 학습 환경은 `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 Python 시작 전에 설정했습니다. 준비 중 CUDA 재현성 환경 설정과 유효 입력 수 assertion에서 멈춘 두 기록은 `PRE_FIT_*`에 보존했습니다. 두 중단 모두 분류기 생성/최적화 이전이며 업데이트0입니다. 학습 성능에 따른 재시도나 실사 결과 이후 추가 학습은 없었습니다.

## 시각화 해석

`ROOT_READOUT_KO.html`에는 실제 crop, 고정 위치 heatmap, 새 분류기가 읽는 Gaussian 국소 token 가중 영역을 나란히 표시했습니다. 공간 특징은 문맥을 포함하므로 원이 엄밀한 RGB 수용 영역을 뜻하지는 않습니다. 색은 상대 위치/가중치 표시이며 보임 확률이 아닙니다.

`evaluation_v1/INDEX_KO.html`의 사례는 유형×정답D/E×두 모델 정오16셀에서 최소 원본 순번으로 선택한 후 RGB를 읽었습니다. 빈 셀도 유지했습니다. G996의 주석/시각적 해석 불일치와 G981/G200의 작은3px 경계 초과는 별도 주의 문구로 남기고 원 주석을 바꾸지 않았습니다.
