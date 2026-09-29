# 공개 큰 오류 사례의 실패 유형: box 불일치와 잘못된 기하 입력

대상: `eval_night08:1779449483432542720`, SEVERE_OCCLUSION, REC_027.

[R0+old GEO와 OLD_REF217+old GEO 그림](cases/FINAL/00_ae7e94e5ad28.png)은 작은 영역의 native2D 점과 그 점에 잘 맞는 pose 재투영을 보인다. 이 모습만으로 다른 실물 instance를 검출했다고 단정하지 않고, 저장된 검출 box·기존 참조 box·native 점·두 최종 pose 후보를 사후 비교했다. 새 추론·학습·후보 재선택은 없다.

## 확인된 사실

1. **기존 평가의 detector–reference box match가 실패한 사례다.** 선택된 box와 참조 box의 IoU는0.026789로 기존0.5 기준보다 낮다. 검출 후보3개의 IoU는0.026789 / 0.019723 / 0.000782다. 저장된 나머지 detection으로 바꾸어도0.5 match를 회복할 후보가 없다. 이 전수 확인은 진단용일 뿐 GT로 실제 선택을 바꾸지 않았다.
2. 선택된 detection의 confidence는0.117387이고 box 크기는137.19×23.14px, 참조 box 크기는419.39×57.17px다. R0·OLD_RAW·OLD_REF·REF_CLEAR43에서 후보 box/score/order는 동일하다. pose-only 학습이 detector를 수정한 결과로 해석할 수 없다.
3. **모든 점이 같은 한 점으로 수치적으로 붕괴한 것은 아니다.** R0의8개 native 코너는 모두 서로 다르고, 최소 pair 거리9.50px, 최대129.70px, 중심화한2D 좌표 rank2다. 다만 점의 footprint는 약129.30×20.47px로 작고 참조 팔레트 위치와 불일치한다. 참조에서 지원되는 동일6코너만 비교해도 R0 footprint129.30×19.35px 대 참조280.39×48.81px다. “작고 어긋난 검출 영역에 모인 좌표”라는 설명이 단순한 point-collapse보다 정확하다.
4. 현재 선택의 재투영 잔차는 작다. R0 약1.213px, OLD_REF 약1.097px인데, 기존 pose 참조 대비 T는 각각487.703cm/482.510cm, R는71.780°/72.429°다. **예측한 점에 cuboid가 잘 맞는 것과 올바른 팔레트 pose를 찾는 것은 다르다.** 작은 apparent footprint와 잘못된 대응/위치가 큰 깊이·위치 오차로 이어졌을 가능성과 일치하지만, 이 한 사례로 물리적 원인을 확정하지 않는다.

위 footprint/거리/IoU는 원시 좌표를 공개하지 않는 기술적 요약이다. box mismatch에 대한 공식2D 점수의 기존 penalty 처리는 바꾸지 않았다. 그림의 초록 점은 legacy2D 참조이며 독립 실측 물리 GT가 아니다.

## W/D 후보를 바꾸는 것만으로 복구되는가?

기존에 저장된 동일 detection의 최종 whole-pose 후보별 참조 오차다. 두 후보의 T/R 최솟값을 하나의 가상 pose로 합치지 않는다.

| 모델/최종 후보 | T cm | R° | 선택 상태 |
|---|---:|---:|---|
| R0 / long-face-front | 559.575 | 25.979 | 미선택 |
| R0 / short-face-front | 487.703 | 71.780 | 선택 |
| OLD_REF / long-face-front | 556.427 | 25.122 | 미선택 |
| OLD_REF / short-face-front | 482.510 | 72.429 | 선택 |

다른 W/D 후보는 R를 낮추지만 T를 더 악화시키고 두 후보 모두 큰 위치 오차가 남는다. 따라서 이 frame은 단순히 “좋은 전체 pose를 GEO가 놓친 것”만으로 설명되지 않는다. 후보 선택의 W/D 모호성과 그 이전의 detector/localization·대응점 입력 실패를 분리해야 한다.

## 보고서에 쓸 수 있는 표현과 피할 표현

- 가능: “참조 box와 모든 저장 검출 box가 불일치했고, 작은 영역에 형성된 native 점을 PnP가 내부적으로 잘 맞추지만 큰 pose 오차가 남는 사례다. W/D 교체만으로 T/R를 함께 회복하지 못한다.”
- 피함: “다른 팔레트를 검출한 것이 확정됐다”, “8점이 한 점으로 collapse했다”, “teacher가 물리 정답을 망가뜨렸다”, “selector만 바꾸면 해결된다.”
- 다른 실물 instance인지, 같은 팔레트의 일부/주변 구조에 반응한 것인지, 참조의 물리적 correspondence가 어디까지 정확한지는 현재 자료로 확정하지 않았다. wrong-instance와 detector/reference mismatch를 같은 뜻으로 사용하지 않는다.

## 재사용 근거

- 공개 그림·선정 이력: [CASES_FINAL.json](CASES_FINAL.json), [CASES_FINAL.md](CASES_FINAL.md).
- 최종 평가와 source bindings: [FINAL_RESULTS.json](FINAL_RESULTS.json).
- 실제 조회: `FINAL_CASE_BINDINGS_PRIVATE.json`이 연결한 각 모델 `PREDICTIONS.json` 및 `CANDIDATES.json`; `FINAL_FRAME_METRICS_PRIVATE.json`; legacy `TRUTH_FOR_DISPLAY_ONLY.json`의 box/valid; 기존 S42 `ORACLE_METRICS_SELECTIONS_PRIVATE.json`의 R0/OLD_REF 후보 metrics.
- 기존 box-match 계약: `scripts/research/pallet_selftraining_paper_closure_v1/evaluate.py`의 IoU≥0.5. 전체 pose 및 평가 denominator는 변경하지 않았다.

이 사례는 이미 공개 승인된 RGB ID의 큰 오류 예시이며,99장 전체의 실패 원인 빈도나 모든 큰 T tail의 원인을 대표한다고 주장하지 않는다.
