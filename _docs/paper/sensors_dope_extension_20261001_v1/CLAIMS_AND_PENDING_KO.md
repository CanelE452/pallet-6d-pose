# 주장 범위와 미완료 검증

| 본문 주장 | 현재 근거 | 허용되는 해석과 제한 |
|---|---|---|
| Local correction을 세 동결 추정기 인터페이스에 실행했다 | 기존 YOLO, 완료된 DOPE P/D, 완료된 ResNet18 P0/D0 | 세 구체 구현의 application evidence다. 임의 backbone에 그대로 적용되는 범용 정리는 아니다. |
| DOPE에서 중앙 2D 오차와 PCK10을 개선했다 | 재사용 DEV에서 baseline→P 평균 12.585→7.461 px, 19.872→36.775%; 3 seeds | 재사용 319장·13세션이며 독립 TEST가 아니다. missing/실패 분모를 유지하고 tail·pose도 함께 보고한다. |
| ResNet18에서 P0가 주 backbone-transfer 결과를 개선했다 | FULL→P0: 8.001→7.143 px, PCK10 54.68→58.28%; paired 2D 변화 −0.859 px [−1.318,−0.621] | P90은 52.054→52.086 px라 tail 우월을 주장하지 않는다. 실제 이미지 선택에 쓰인 DEV가 아니다. |
| ResNet18 P0의 reconstructed-reference R이 개선됐다 | canonical camera-facing-frame C2 R 3.252→3.018 deg; 변화 −0.234 [−0.502,−0.022] | T 변화 −1.007 cm [−1.668,0.187]에는 0이 포함된다. 물리 계측 6D GT가 아니므로 안정적 physical T/R 동시 개선을 주장하지 않는다. |
| 이미지와 치수를 함께 학습하는 직접 ResNet18을 완료했다 | CONSTANT/SHAPE/FULL, 각 10 epochs·34,990 updates·559,800 exposures; TRAIN-only normalization | FULL−CONSTANT는 데이터셋별 혼합 결과다. Wood는 dimension-support 외삽이고 square 세트는 set 안에서 치수가 고정돼 치수 인과성을 분리하지 못한다. |
| 보정 head 자체의 치수 입력을 검사했다 | P5와 동일 구조의 P5_CONSTANT, 각 3 seeds | P5−P5_CONSTANT 2D/T/R interval이 모두 0을 포함한다. favorable PCK10 하나로 안정적 차원 이득을 주장하지 않는다. |
| 동일 환경에서 세 backbone의 비용을 비교했다 | 26 decoded BGR images × 5 repeats, warmup 20, batch 1, RTX 3080, 1,170 rows 모두 유지 | Desktop 단일 GPU inference다. Jetson, file decode, concurrent throughput, energy 또는 독립 latency replication이 아니다. |
| Source-only correction-head fitting을 수행했다 | YOLO/DOPE/ResNet 보정 head는 real pallet training example 0; 고정 source selection/heldout 기록 | Baseline pretraining 이력까지 real-label-free라는 뜻은 아니다. ImageNet/COCO 계보와 real DEV 평가는 구분한다. |

## Pose 수치의 평가 프레임

본문 ResNet FULL↔P0 보정 전후 비교는 한 evaluator 안에서만 계산한다. 이 canonical 경로는 `R_cf`를 `R_gt_representative`와 직접 비교하고 width/depth parity를 axis accuracy로 별도 보고한다. 따라서 표의 R은 **camera-facing-frame, proper-group C2-aware rotation**이고 AUC는 **proper-group C2-aware corresponding-point ADD AUC**다. unrestricted nearest-neighbor ADD-S가 아니다.

직접 치수 ablation의 옛 경로는 `R_physical = R_cf Q`를 먼저 적용한 뒤 proper C2 group으로 점수화한다. 동일 FULL checkpoint와 수치상 일치한 출력(max point difference 0.001163 px)에서 T는 같지만, direct 경로는 R 3.581 deg/AUC 0.31628이고 canonical 경로는 R 3.252 deg/AUC 0.31296이다. 두 경로는 axis-correct 234/319 frame에는 동의하며, 남은 85개 axis error와 회전 convention 때문에 차이가 난다. 직접 실험 수치를 canonical 보정 전후 표로 복사하지 않는다.

## 제출 전에 남은 외부 근거

- 독립 TEST 또는 새 capture session의 prediction-blinded annotation
- 독립 물리 6D pose reference가 필요한 경우의 별도 계측 실험
- 저자명·기관·funding·acknowledgement·제출 동의 확인
- 원본 이미지 또는 qualitative panel을 공개할 권리 확인
- 최종 IEEE Sensors 형식, 페이지 예산, 참고문헌 및 영어 문장 저자 검토

이 항목들은 현재 완료된 학습·평가·런타임 결과를 무효화하지 않지만, 독립 일반화와 물리 T/R 정확도 주장을 제한한다.
