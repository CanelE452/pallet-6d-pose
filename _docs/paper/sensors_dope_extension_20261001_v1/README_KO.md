# IEEE Sensors 세 추정기 원고 상태

이 폴더는 YOLO, DOPE, ResNet18에 동결 특징 기반 local correction을 적용한 **미제출 원고 초안**이다. DOPE 결과와 ResNet18 직접 추정기·보정기·통합 런타임 결과까지 삽입됐다. 조판 완료는 독립 TEST, 물리 계측 6D GT, 저자 승인 또는 제출을 뜻하지 않는다.

- [본문 PDF](manuscript.pdf) / [본문 TeX](manuscript.tex)
- [보충자료 PDF](supplementary.pdf) / [보충자료 TeX](supplementary.tex)
- [주장 범위와 한계](CLAIMS_AND_PENDING_KO.md)
- [DOPE 결과 결속 영수증](DOPE_RESULTS_BINDINGS.json)
- [ResNet18 결과 결속 영수증](RESNET_RESULTS_BINDINGS.json)
- [빌드 영수증](DRAFT_BUILD.json) / [PDF 시각 검수 영수증](PDF_VISUAL_REVIEW.json)

## 삽입된 핵심 근거

| 추정기 | 보고하는 역할 | 재사용 DEV 핵심 결과 |
|---|---|---|
| YOLO | 기존 주 실험과 P/D/prior 비교 | 기존 결과와 수치를 그대로 보존 |
| DOPE | 두 번째 동결 특징 인터페이스 | baseline→P 평균: 조건부 2D 중앙값 12.585→7.461 px, full-GT PCK10 19.872→36.775% |
| ResNet18 FULL+P0 | 세 번째 백본의 주 transfer arm | 2D 중앙값 8.001→7.143 px, PCK10 54.68→58.28%, T 11.007→10.001 cm, camera-facing-frame C2 R 3.252→3.018 deg |

ResNet18 FULL은 384×512 RGB와 등록 치수에서 만든 다섯 값 `log W`, `log D`, `log H`, `log(W/D)`, `log(H/sqrt(WD))`를 학습 때부터 함께 받는다. CONSTANT, SHAPE, FULL은 동일 seed·source 순서·augmentation으로 각각 10 epoch, 34,990 updates, 559,800 source exposures를 학습했다. 예전 60-epoch masked-MSE 계획은 고정 usability gate를 통과하지 못한 pilot이며 보고 성능에 쓰지 않는다.

FULL을 동결한 뒤 D0/P0/P5/P5_CONSTANT 각 3 seed를 6,000 updates, batch 16으로 학습했다. P0가 세 번째 백본 transfer의 주 실험이고 D0는 direct-residual control이다. P5와 P5_CONSTANT는 correction head에 치수를 추가한 별도 확장이다. P5−P5_CONSTANT의 2D, T, R 구간에는 모두 0이 포함되므로 치수의 안정적인 추가 이득 또는 T/R 동시 개선으로 주장하지 않는다.

직접 추정기의 FULL−CONSTANT 비교도 혼합 결과다. DEV 중앙값은 8.222→8.012 px이나 plastic은 7.587→7.732 px로 악화한다. Wood의 9.430→8.234 px 개선은 depth 0.59 m가 synthetic TRAIN 최소 0.818 m보다 작은 외삽 조건이다. GREEN150과 0918은 각 데이터셋 안에서 치수가 변하지 않으며, 0918은 59 frame 개선/60 frame 악화와 중앙 변화 +0.031 px라서 치수 인과성을 분리하지 못한다.

## 통합 런타임

RTX 3080에서 동일한 26개 decoded BGR 입력, batch 1, warmup 20, 5 repeats로 1,170개 측정 row를 모두 유지했다. full-path 중앙값은 YOLO 11.092 ms, YOLO+P1 14.779 ms, DOPE 64.585 ms, DOPE+P1 67.411 ms, ResNet18 FULL 9.453 ms, FULL+P0 12.193 ms다. 이 값은 desktop 측정이며 Jetson latency, 동시 throughput 또는 energy 결과가 아니다.

## 해석 제한

- 모든 real-image 정확도는 319장·13세션의 재사용 DEV 근거다. 독립 TEST가 없다.
- T/R reference는 수동 2D 점, 카메라 내부 파라미터와 등록 치수로 재구성했다. 독립 물리 6D 계측값이 아니다.
- Wood는 치수 지원 범위 밖이고 square 세트에서는 치수가 변하지 않는다.
- 세 구현은 각 backbone의 서로 다른 tap, stride, channel 및 missingness 계약을 사용한다. 세 사례는 범용 model-agnostic 정리를 증명하지 않는다.
- 본문 ResNet 보정 전후 pose 표는 canonical camera-facing-frame C2 evaluator만 사용한다. 직접 치수 ablation의 옛 scorer는 `R_physical = R_cf Q`를 먼저 적용하므로 FULL의 R/AUC 수치를 본문 표에 섞지 않는다. 두 경로는 axis-correct 234/319에는 동의하지만 85개 axis error를 다른 회전 convention으로 처리한다.

## 빌드

저장소 루트에서 다음 명령은 결속된 source/table/figure SHA를 확인한 뒤 로컬 cached TeX 자원만 사용해 PDF와 페이지 렌더를 다시 만든다.

```bash
/home/minjae/anaconda3/envs/pallet-pose/bin/python _docs/paper/sensors_dope_extension_20261001_v1/build.py --render
```

`update_resnet_results.py`는 실행 전 60-epoch ResNet 계획용 legacy 삽입기다. 현재 10-epoch 직접 추정기와 P0/P5 실험 표는 세 실험 보고서에서 검산하여 `RESNET_RESULTS_BINDINGS.json`에 직접 결속했으므로 이 legacy 삽입기로 덮어쓰지 않는다. 근거 보고서는 다음과 같다.

- `_docs/experiments/pallet_resnet18_dimension_20261002_v1/REPORT_KO_V2.md`
- `_docs/experiments/pallet_resnet18_dim_refiner_report_20261002_v3/REPORT_KO.md`
- `_docs/experiments/pallet_three_backbone_runtime_20261002_v1/REPORT_KO.md`

대형 checkpoint, 원본 영상·이미지와 feature cache는 공개 원고 자산이 아니다. 저자명·기관·funding·제출 동의와 이미지 공개 권리는 별도 저자 확인이 필요하다.
