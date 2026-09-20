# Large-error refiner v1 — 실행 기록

완료: **NO_PRE_REGISTERED_PASS**. [완료 해석](INTERPRETATION_KO.md), [전체 결과표](RESULTS_KO.md), [검증 결과](AUDIT.json). C/D/E 모두 마지막2,000 step으로 평가했고 원본 모델은 유지했다.

목적: 기존 8px 제한을 해제하는 것과, 큰 입력 오류를 학습하는 공동 보정기가 실사에서 다른 효과를 내는지 한 번의 bounded screen으로 비교.

현재 최종 R0/N2 모델, 논문 표, 라벨, 옛 실험 결과는 수정하지 않는다. 새 모델 자동 승격/commit/push는 하지 않는다.

- A: 기존 N2 seed1. B: 동일 N2, 최대 이동 상한만 이미지 대각선 1%→4%.
- C: 팔레트 전체 ROI와 입력 코너/치수로 8점 잔차를 함께 예측, 일반 합성 학습.
- D: C와 같은 구조/초기값에 합성 입력 큰 오류를 추가.
- E: D의 batch 절반을 수동 클릭 출처가 확인된 실사 정답으로 교체.
- C/D/E 각 seed1, 2,000 step, batch16. 원본 R0 backbone은 고정. 평가를 본 추가 학습이나 threshold 조정 없음.
- 실사 후보 24장 중 9장/38코너만 직접 수동 클릭 확인. 나머지는 제외. 후보 전체 촬영 그룹을 보수적으로 제외한 DEV72와 GREEN150에서 평가.
- 사용자 실패 예제가 속한 eval_pallet07은 학습에서 제외. GREEN150도 새 실험 학습 제외지만 과거 개발에 노출되어 독립 최종 검증이라고 주장하지 않음.
- GT-free 기존 confidence/geometry/flip gate의 통과/실패를 모두 기록하고 실패 시 A로 복귀. ungated/gated 동일 전체 평가분모.
- 복구율(고정 R0 >20px → <=10px), 훼손율(R0 <5px → >10px), PCK10/P90/E_sym/E_fixed. 임의 코너별 GT 재대응 금지.

GPU: CUDA RTX3080, 원격 데스크톱 보존, 다른 compute/80°C 이상이면 중지. 드라이버/전력/OS 설정 변경 및 재부팅 없음.

## 재현

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_large_error_refiner_v1/test_model.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_large_error_refiner_v1/test_contract.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_large_error_refiner_v1/run.py all
```

PROTOCOL.json과 SPLIT.json이 실제 실행 계약. FIT_*.json/체크포인트는 해당 protocol SHA를 기록한다. 250 step마다 복구용 checkpoint, 2,000 step 마지막 결과만 평가.

## 학습 전 preflight 정정

GPU 소량 forward/backward 검사에서 실제 source 2,048행은 모두 train임을 확인했다. 합성 스트레스 256행은 기존 non-train의 heldout135/selection58/calibration63이며, 처음 문안의 “calibration256”은 부정확해 실제 구성으로 정정했다. 학습 전에 feature padding 영역의 projection bias도 마스킹하고 회귀검사를 추가했다. 기존 preflight 문서들은 `*_PREFLIGHT_NOT_TRAINED.json`으로 남겼으며 어떤 최종 모델도 이 문서로 학습하지 않았다. 소량 gradient 검사에는 optimizer step이 없었다.
