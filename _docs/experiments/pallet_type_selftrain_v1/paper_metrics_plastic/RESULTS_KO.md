# 일반 플라스틱 self-training — 논문 표 기준 재평가

평가일: 2026-09-20 KST. 일반 플라스틱 194장만 사용. 검출·순위 평가는 기존 배경 2,689장을 동일하게 추가했다. 목재·초록색은 제외했다.

사용자가 제시한 AP/6D 스크린샷의 R0는 일반 플라스틱+목재 319장 기준이다. 아래 R0는 일반 플라스틱 194장으로 다시 계산했으므로 그 스크린샷과 수치가 다르다. 기존 일반 플라스틱 R0 원본 결과와 AP, keypoint median/P90, F0 및 MAIN pose 수치가 재현되었다.

학생은 보정된 pseudo-label로 이미 학습한 `runs/PLASTIC/weights/last.pt`이다. R0 → raw flip·LOO 필터 → frozen Replay 보정 → corrected all-8 LOO 필터를 통과한 일반 플라스틱 pseudo-label과 합성 replay로 학습했다. 추가 안정성·면 모양·X자 필터는 사용하지 않았다. 이번 평가는 학생 단독 추론이며 추론 후 Replay 보정이나 평가 이미지 필터를 적용하지 않는다. 새 학습, threshold tuning, checkpoint 선택은 없다.

## 1. F0 / No filter

| 모델 | Kept | Retain | Pass med px ↓ | Reject med px ↑ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|---:|---:|
| R0 합성만 학습 | 194 | 1.000 | 7.689 | — | 0.526 | 1.000 |
| 보정 pseudo-label self-training | 194 | 1.000 | 7.573 | — | 0.505 | 1.000 |

여기서 precision은 검출 AP precision이 아니라 **감독 keypoint가 모두 20px 이내인 이미지의 비율**이다. 해당 이미지는 102 → 98장이다. No filter이므로 recall=1은 자동적으로 유지되며 개선의 증거가 아니다. 기존 M4 정의대로 검출 및 confidence ≥0.5인 corner ≥6의 기본 적격성은 유지했고, 두 모델 모두 194장이 적격이었다.

참고로 이 모집단의 pooled P90은 55.648 → 42.205px, >40px keypoint 비율은 12.804% → 10.788%로 감소했다. 일부 큰 오류는 줄었지만, 모든 점이 동시에 정확한 이미지 수가 늘어난 것은 아니다.

## 2. 검출·오검출 구분·2D keypoint

| 모델 | AP50–95 ↑ | AP50 ↑ | AUROC ↑ | FPR95 ↓ | kp med px ↓ | kp P90 px ↓ |
|---|---:|---:|---:|---:|---:|---:|
| R0 합성만 학습 | 0.7194 | 0.8949 | 0.9890 | 0.0439 | 7.275 | 39.36 |
| 보정 pseudo-label self-training | 0.7000 | 0.8795 | 0.9870 | 0.0658 | 7.546 | 40.09 |

Box IoU≥0.5 매칭 이미지는 186/194 → 190/194이다. AP는 모든 검출 후보와 배경 이미지를 사용하므로 매칭 수가 늘어도 하락할 수 있다. FPR95는 각 모델의 positive TPR 95% 지점에서 측정한 배경 오검출률이며, 고정 confidence threshold에서의 오검출률과는 다르다.

이 표의 keypoint 오차는 IoU≥0.5 매칭 이미지의 감독 keypoint를 pooling한 기존 논문 지표이다. F0 표는 매칭 조건 없이 기본 적격 이미지에서 in-frame 감독점을 pooling하므로 표 1과 오차값·P90의 방향이 다를 수 있다. 두 표 모두 기존 정의를 그대로 유지했으며 유리한 집합으로 변경하지 않았다.

## 3. 6D / frozen MAIN selector

| 모델 | PoseCov ↑ | AxisAcc ↑ | R med ° ↓ | Yaw med ° ↓ | t med cm ↓ | IoU3D ↑ | ADDsym AUC ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 합성만 학습 | 1.000 | 0.7062 | 2.133 | 1.194 | 10.472 | 0.58572 | 0.34476 |
| 보정 pseudo-label self-training | 1.000 | 0.7165 | 2.287 | 1.274 | 10.169 | 0.57654 | 0.33829 |

기존 prediction-only axis selector + SQPnP/RefineLM MAIN 경로만 집계했다. GT axis를 넣는 oracle 결과가 아니다. 6D reference는 수동 2D annotation·카메라 K·치수에서 기하적으로 복원한 기준이며, 외부 장비로 독립 측정한 6D GT가 아니다. ADDsym은 기존 {I, Ry180} 대칭 및 0.1×diameter 범위 AUC이다.

## 판정

일부 큰 2D 오류, 검출 매칭 수, 축 분류, 중앙 translation 오차는 개선됐지만 AP·AUROC·FPR95·matched keypoint 오차·회전·IoU3D·ADDsym은 악화됐다. **이번 일반 플라스틱 self-training이 전체 성능을 개선했다고 볼 수 없으며 R0 교체 근거가 없다.** 단일 seed의 짧은 실험이고 반복 사용한 DEV 평가이므로 독립 최종 검증이나 통계적 유의성 주장은 하지 않는다. 합성 추가학습 control은 앞선 실험에 있지만 이번 요청의 세 표는 R0 vs 학생만 비교한다. Replay 보정만의 인과적 효과를 분리한 표가 아니다.

## 재현 및 보존

- 원시 결과: `data/pallet/results/pallet_type_selftrain_v1/paper_metrics_plastic/SUMMARY.json`
- 원시 후보: `NEG_R0.json`, `NEG_PLASTIC.json`, `PAPER_CACHE_R0.json`, `PAPER_CACHE_PLASTIC.json`
- 표 1 per-frame: `F0_R0.json`, `F0_PLASTIC.json`
- 표 2: `PAPER_2D_R0.json/.csv`, `PAPER_2D_PLASTIC.json/.csv`
- 표 3 per-frame: `POSE_R0.json`, `POSE_PLASTIC.json`
- 프로토콜: 같은 문서 폴더의 `PROTOCOL.json`, `POSE_ID_BRIDGE.json`
- 기존 pose archive는 `session__frame`, 현재 manifest는 `session:frame`을 사용한다. `paper_metrics_plastic_finish.py`가 image/annotation 경로와 object type의 정확한 일치를 검증하여 194개 ID를 일대일 대응시킨다. 좌표·정답·평가 수식은 바꾸지 않는다.
- 최초 `paper_metrics_plastic score` 직접 호출은 위 ID 차이 때문에 실행되지 않는다. 최종 채점 진입점은 아래 finish 명령이다. 최초 lock과 실행 소스는 그대로 보존했다.

```bash
MPLCONFIGDIR=/tmp/pallet-selector-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_type_selftrain_v1.paper_metrics_plastic prepare
MPLCONFIGDIR=/tmp/pallet-selector-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_type_selftrain_v1.paper_metrics_plastic negative --arm R0
MPLCONFIGDIR=/tmp/pallet-selector-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_type_selftrain_v1.paper_metrics_plastic negative --arm PLASTIC
MPLCONFIGDIR=/tmp/pallet-selector-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_type_selftrain_v1.paper_metrics_plastic_finish
```

GPU는 RTX 3080을 사용했고 최대 확인 온도는 60°C였다. 컴퓨터 재부팅·드라이버 변경·기존 결과 덮어쓰기·git push는 하지 않았다.
