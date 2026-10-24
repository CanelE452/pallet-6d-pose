# V208 · 저장된 합성 source의 point0와 region 판정

원 BEFORE 부모 41개·native 328개에서 point0 combined는 D 287 / S 41, region은 D 164 / U 123 / S 41이다. point D이면서 region U인 123개는 전체 328개의 **37.50%**, point D 287개의 **42.86%**다. 서로 다른 판정 단위의 저장값을 확인한 결과이며 annotation 오류 수나 모델 원인 증명은 아니다.

## 범위와 분모

| 단위 | 개수 | 해석 |
| --- | --- | --- |
| 현재 TRAIN constructed physical rows | 1968 | 82 attempts × 3 variants × 원 native8 |
| constructed TRAIN attempts | 82 | admitted/기존 weak 상태로 추가 선별하지 않음 |
| 중복 제거 original BEFORE 부모 | 41 | 현재 constructed TRAIN 행이 가리키는 고유 부모 |
| original BEFORE native 조회 | 328 | 41 parents × native0..7 |
| 원 input U | 0 | finite 여부로 보충하지 않고 원 availability 유지 |

## point0와 region 상태

| 저장 판정 | D | E | S | U | UNKNOWN | 전체 |
| --- | --- | --- | --- | --- | --- | --- |
| point0 target-alone | 287 | 0 | 41 | 0 | 0 | 328 |
| point0 combined | 287 | 0 | 41 | 0 | 0 | 328 |
| region | 164 | 0 | 41 | 123 | 0 | 328 |
| target-alone region | 164 | 0 | 41 | 123 | 0 | 328 |

## 교차표

| point0 combined → region | D | E | S | U | UNKNOWN | 전체 |
| --- | --- | --- | --- | --- | --- | --- |
| D | 164 | 0 | 0 | 123 | 0 | 287 |
| E | 0 | 0 | 0 | 0 | 0 | 0 |
| S | 0 | 0 | 41 | 0 | 0 | 41 |
| U | 0 | 0 | 0 | 0 | 0 | 0 |
| UNKNOWN | 0 | 0 | 0 | 0 | 0 | 0 |

## 원 native별 전체 보존

| native | N | point D | point S | region D | region S | region U | point D·region U | input U | sample extent | canvas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 41 | 41 | 0 | 41 | 0 | 0 | 0 | 0 | 41 | 41 |
| 1 | 41 | 41 | 0 | 41 | 0 | 0 | 0 | 0 | 41 | 41 |
| 2 | 41 | 41 | 0 | 17 | 0 | 24 | 24 | 0 | 41 | 41 |
| 3 | 41 | 41 | 0 | 24 | 0 | 17 | 17 | 0 | 41 | 41 |
| 4 | 41 | 41 | 0 | 24 | 0 | 17 | 17 | 0 | 41 | 41 |
| 5 | 41 | 41 | 0 | 17 | 0 | 24 | 24 | 0 | 41 | 41 |
| 6 | 41 | 17 | 24 | 0 | 24 | 17 | 17 | 0 | 41 | 41 |
| 7 | 41 | 24 | 17 | 0 | 17 | 24 | 24 | 0 | 41 | 41 |

## 저장 reference의 원 crop 좌표 범위

| 진단 | 해당 | 분모 |
| --- | --- | --- |
| 저장 reference 존재 | 328 | 328 |
| sample-center extent [0,95] | 328 | 328 |
| canvas extent [-0.5,95.5) | 328 | 328 |

D는 저장된 point의 직접 가시, S는 저장된 자기 가림, E는 외부 가림, U는 저장된 region 판정의 미결정이다. UNKNOWN은 필드/schema를 읽지 못하는 상태와 구분해 표시했다. target-alone와 combined point 집계가 같고 region과 target-alone region 집계도 같다. 원 BEFORE만 조회했으므로 E 0을 외부 가림 학습 범위의 증거로 쓰지 않는다.

## 보존된 reader 이력

v1 실행 자체는 completed/PASS로 종료했지만, reader가 저장된 records_CF 형태를 찾지 못해 point·region 328개가 모두 UNKNOWN이었다. 41개 파일의 사유는 UNKNOWN_OR_AMBIGUOUS_SAVED_CF_RECORD_LIST였다. 원 v1 결과를 보존하고, v2에서 명시적인 records_CF schema adapter로 같은 선택·원 q/L/native/U를 다시 조회했다. 저장값을 고치거나 새 visibility를 계산하지 않았다.

## 해석의 경계

- point0는 선언된 합성 source 내부 ray/기하 판정이며 실제 공개 영상의 물리 native gold가 아니다. region은 별도로 저장된 판정 단위다.
- point D·region U 123개는 서로 다른 판정의 차이다. 실제 annotation 오류 123개, 모델 실패 원인 또는 새 정답으로 해석하지 않는다.
- 원 BEFORE에는 E가 없다. 새 V189 COVER/SHAM의 native point 상태는 UNKNOWN이며 BEFORE 상태·finite-end D/E/D·center owner를 복사하지 않는다.
- raw96 reference index는 47.5 + 96 × (saved reference − original q) / L의 좌표 대수다. q의 GT 오차나 실제 pixel 가시성 판정이 아니다.
- 모든 reference가 canvas 및 sample-center 범위 안이라는 결과는 원영상·crop에서 부위가 보인다는 증거가 아니다. 실제 RGB/valid·source frame observability를 이번 보고서에서 재확인하지 않았다.
- native0..7을 모두 보존했다. native6/7의 이번 point 상태와 과거 finite-end direct target의 UNKNOWN은 서로 다른 필드이며 자동 학습 권한으로 연결하지 않는다.
- TRAIN source만 다뤘다. 보호 평가 점수·영상·모델·NPZ를 새로 읽지 않았고 projection/ray/학습/forward를 실행하지 않았다.
- 이 단계는 저장된 source 정의를 확인한 진단이다. 실사 전이·모델 개선·채택을 입증하지 않는다. 최신 실제 실사 비교 V203은 그대로이며 전체 목표는 미해결이다.

## 원문과 재현

- [수치 JSON](ROOT_RESULTS_KO.json)
- [원328행 CSV](NATIVE328_POINT_REGION_LOCALITY.csv)
- [v2 실제 완료](source_train_native_point_grounding_v2/COMPLETE.json)
- [원328행 저장 JSONL](source_train_native_point_grounding_v2/ALL_ORIGINAL_BEFORE_NATIVE_POINT_STATE_AND_LOCALITY.jsonl)
- [현재 TRAIN 1968행 부모 연결](source_train_native_point_grounding_v2/ALL_CONSTRUCTED_TRAIN_CONTROL_ROWS_TO_BEFORE_PARENT.jsonl)
- [v2 저장 요약](source_train_native_point_grounding_v2/SUMMARY.json)
- [보존 v1 UNKNOWN 요약](source_train_native_point_grounding_v1/SUMMARY.json)
- [v2 schema protocol](saved_schema_reader_v2_c/PROTOCOL_BEFORE_SAME_TRAIN_GT_RECORDS_CF_V2.json)
- [보고서 생성 코드](source_report_preparation_a/build_saved_point_region_report.py)
- [보고서 읽기 범위](source_report_preparation_a/PROTOCOL_BEFORE_SAVED_V2_REPORT.json)

보고서는 저장된 v2 출력만 재집계했다. 독립 saved-GT QA는 별도 B 검토이며 이 생성기가 완료를 선행 주장하지 않는다. HTML은 오프라인 파일·링크 검사 대상이며 브라우저 렌더 검증을 주장하지 않는다.
