# 과거 로컬 상태 정정의 공개 보충 기록

2026-10-16에 확인한 2026-10-01~02 실험 기록입니다. 기존 main 문서와 원본 로컬 수정은 덮어쓰지 않았으며, 당시 로컬 문서3개와 검산에 사용한 작은 근거를 원본 바이트 그대로 보존했습니다. 새 학습·추론·평가를 실행한 결과가 아닙니다.

## 당시 문서와 확인된 사실

| 문서 스냅샷 | 원본 수정 시각(KST) | 확인 결과 |
|---|---|---|
| [ResNet-18 README](dated_snapshot/pallet_resnet18_refiner_20261001_v1/README_KO.md) | 10/02 08:31 | 60epoch 학습 실행은 종료됐으나 기능 검사 실패. 후속 P/D 본학습·실사 성능·속도 측정은 실행되지 않음 |
| [세 기반 README](dated_snapshot/pallet_three_estimator_20261001_v1/README_KO.md) | 10/02 08:31 | DOPE 보고서·속도 측정 완료와 ResNet-18 실패를 구분한 정정 |
| [세 기반 상태](dated_snapshot/pallet_three_estimator_20261001_v1/STATUS_KO.md) | 10/02 01:08 | 실제 queue의 `STOPPED_ON_ERROR / resnet18_head_smoke` 및 완료10단계와 일치 |

[학습 종료 영수증](evidence/pallet_resnet18_refiner_20261001_v1/BASELINE_TRAINING_COMPLETE.json)은60epoch·209,940updates·3,358,800exposures와 실사 학습0을 기록합니다. [고정 TRAIN16 실패 진단](evidence/pallet_resnet18_refiner_20261001_v1/BASELINE_FAILURE_DIAGNOSTIC_KO.md)은 기존 threshold0.1에서 유효점0/144·매칭0/16, 최대 confidence0.005223158을 기록합니다. 학습 실행 종료가 모델 기능 성공을 뜻하지 않습니다. [원래 queue 상태](evidence/pallet_three_estimator_20261001_v1/QUEUE_STATUS.json)도 함께 보존했습니다.

**두 README의 “복구 진단 진행 중”은 최신 상태가 아닙니다.** 복구 진단은10/02 08:27 결과 기록까지 이미 완료됐습니다. [고정 프로토콜](evidence/pallet_resnet18_recovery_20261002_v1/DIAGNOSTIC_PROTOCOL.json)의 동일 TRAIN256·초기값·배치 조건에서 Adam weight decay만 바꾼 폐기용2fit(각500updates)이며, [완료 결과](evidence/pallet_resnet18_recovery_20261002_v1/DIAGNOSTIC_RESULTS_KO.md)는 두 경우 모두 유효점0/144·매칭0/16입니다. 전체60epoch 재학습·refiner 학습·일반화 입증·방법 성공은 없습니다. [초기 수치 비교 중단과 이어 실행](evidence/pallet_resnet18_recovery_20261002_v1/NUMERICAL_CONTINUATION_ADDENDUM_KO.md)도 보존했습니다. 원문은 당시 상태와 오류까지 남기는 스냅샷이므로 수정하지 않았습니다.

DOPE 수치 정정은 이미 공개된 [DOPE 결과](../../pallet_dope_refiner_20261001_v1/REPORT_KO.md)와 일치합니다. P의3seed 통계 평균으로 2D 중앙값12.585→7.461px, PCK10 19.872→36.775%, T/R 중앙값10.046cm/3.441°→8.529cm/2.884°입니다. P90은50.609→50.713px로 악화했고 pose coverage는210/319로 동일합니다. 이 수치는 위 실패한 ResNet-18의 성능이 아닙니다.

## 기존 파일을 덮어쓰지 않은 이유

[기존 ResNet-18 문서](../../pallet_resnet18_refiner_20261001_v1/README_KO.md)와 [세 기반 문서](../../pallet_three_estimator_20261001_v1/README_KO.md)는 게시 당시의 역사 기록으로 보존합니다. 별도 스냅샷에 옛 로컬 정정을 공개하고 이 보충 설명에서 완료·실패·미실행을 구분했습니다. 스냅샷 안의 링크는 원래 경로를 유지하며, 이 묶음의 검산 근거는 위 링크와 [MANIFEST.json](MANIFEST.json)의 대응을 이용합니다. 가중치·대형 원시 데이터는 포함하지 않아 독립 재실행 패키지는 아닙니다.

원본 `.gitignore`와 루트 `readme.md`의 로컬 변경은 기존 공개 그림 예외와10/03·06 보고서 링크를 삭제하므로 반영하지 않았습니다. `ANNOTATION_PROGRESS.md`의 로컬 EVAL541/TRAIN321도 반영하지 않았습니다. 실제 검토 폴더를 쓰기 없이 다시 집계하면 유효862·EVAL862/TRAIN0·오류0이며 기존 main 문서와 일치합니다. 세 변경의 원본 SHA256과 미반영 이유는 manifest에만 기록했습니다. 라벨과 split은 변경하지 않았습니다.
