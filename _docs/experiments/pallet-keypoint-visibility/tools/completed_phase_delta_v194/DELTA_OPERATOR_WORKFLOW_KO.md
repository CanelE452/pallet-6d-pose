# V194 누적 immutable delta 게시 절차

이 도구는 공개 source 그룹 분할을 고정한 두 head의 고정 8000회 학습과 source TRAIN/HOLDOUT 저장 기록을 게시합니다. 실사 개선·새 blind 검증·일반화·모델 채택·전체 목표 해결을 뜻하지 않습니다. 기존 게시 원문과 이전 manifest는 변경하지 않습니다.

1. Root가 V194 보고서·독립 QA·최종 동결을 완료한 뒤 inventory false-entry를 승인합니다. phase_result, independent_report_QA, final_phase_freeze의 실제 metadata와 expected_fields를 바인딩합니다. 그 전에는 phase 열거·해시·복사를 하지 않습니다.
2. source_fit_complete와 source_prediction_seal은 같은 최종 trainer COMPLETE binding입니다. all2_final_states_freeze는 그 COMPLETE의 별도 barrier binding이며, 두 최종 상태가 source 출력 전에 동결됐음을 확인합니다. 양쪽 final checkpoint 2개와 4개 domain×FULL/TRAIN/HOLDOUT×2arm의 저장 출력 24개를 모두 exact path/SHA로 포함합니다. 자체 head는16MiB 미만, 예측 NPZ는32MiB 미만입니다. optional direct target은 기본 제외하며 root가 명시한 경우 fit.training_caches.direct COMPLETE.targets와 정확히 일치하는1개만 허용합니다.
3. 정확 V194 한 폴더만 inventory합니다. V112 이름은 stat 전에 거절하고 symlink·pycache·환경은 따라가지 않습니다. 코드·문서·표·검토 그림은 포함합니다. raw RGB/SAM/feature/cache 배열과 외부 가중치는 제외 사유와 원래 경로를 남깁니다. 새 외부 검토 Markdown도 일반 phase 문서로 포함됩니다.
4. Root가 실제 main을 확인하여 expected_remote_main을 넣고 BASE와 V187~V193의7개 delta/chain/index, BASE와 V187~V193의8개 closing plan을 바인딩합니다. 모든 이전 logical/raw/gzip/descriptor/closing 경로를 보호합니다. BASE 역사 README1개 예외는 유지하며 현재 README의 원문 바이트는 새 suffix와 별도 backup으로 보존합니다.
5. Root build-entry가 기존 copier로 새 archive와 lossless gzip descriptor를 만듭니다. THROUGH_V194 인덱스·AFTER_V194 chain·새 tools만 추가합니다. 독립 byte/gzip/credential/상대링크/parent 검산 후 delta-only≤480MiB push plan을 봉인합니다. Git/main push와 remote/closing은 root의 별도 단계입니다.

검산은 게시 원문과 SHA·경로의 일관성을 확인하며 과학적 성능 검증을 대신하지 않습니다. 가상 roundtrip은 임시 가상 source/repo와 임의의 작은 head/output 바이트만 사용했습니다.
