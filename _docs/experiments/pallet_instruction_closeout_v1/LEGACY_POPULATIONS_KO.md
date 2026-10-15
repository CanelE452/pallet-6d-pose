# 과거 245·110 출처와 자원 계측 종결

원 지시문의 과거 표와 현재 입력 간 차집합 요구를, 실제로 존재하는 예측 기록과 원자료 목록에 대해 검증했다. `245`라는 수가 서로 다른 두 기록에서 쓰였고, `110`은 평가 장수가 아니라 성공 수였다. 새 평가 집합을 만들거나 기존 192장 분모를 바꾸지 않았다.

원문: `CLI_MINIMAL_BI_ABLATION_20261016.md`, 497줄, SHA256 `182856627cdd61ac58a31d40b7185de6ad86f3adad15eaa00fb612d8add2bca9`. 입력 차집합 요구는 179줄이다. 이전 감사와 봉인된 보고서는 당시 기록으로 보존하고, 이 문서로 후속 확인을 추가한다.

| 실제 기록 | 저장된 수의 의미 | 현재 입력과 비교한 결과 | 종결 상태 |
|---|---|---|---|
| `test_reused_v1` | 49장 × 5방법 = 예측 245행 | 현 TEST49와 ID 집합 동일. PRIMARY192에서 빠진 ID는 REMAINDER143 | 실제 ID 차집합 확인 완료 |
| `region_completion_v1/TEST_N3` | 49장 × 5방법 = 예측 245행 | 현 TEST49와 ID 집합 동일. PRIMARY와 차집합 143 | 실제 ID 차집합 확인 완료 |
| `region_completion_v1/TEST_PF` | 49장 × 5방법 = 예측 245행 | 현 TEST49와 ID 집합 동일. PRIMARY와 차집합 143 | 실제 ID 차집합 확인 완료 |
| `clean_fresh_BI_pose_eval_v4_32_r2` | 동일 PRIMARY192에서 성공 110장 | 192개 ID와 순서까지 동일. 저장된 오차의 `T≤5cm AND R≤5°` 성공 분자가 110 | 별도 110장 평가 집합이 아님 |
| 수기 출처 연결 목록의 archive245 | 312개 수기 출처 중 ZIP 원자료 245개 | 기존 Git 출처 67개와 별개. 실제 ZIP 2개에서 245개 고유 이미지 바이트 SHA 검증 | 실재 출처 목록과 해시 차집합 확인 완료 |

세 예측 파일의 전체 ID·방법 행은 [PREDICTION245_ROWS.csv](evidence/PREDICTION245_ROWS.csv), 110개 성공 ID 및 PRIMARY와의 정확한 차집합은 [LEGACY_POPULATIONS_CLOSEOUT.json](evidence/LEGACY_POPULATIONS_CLOSEOUT.json)에 있다. 이 자료들을 독립적인 245장 또는 110장 평가 결과로 취급하지 않는다.

## 수기 원자료 245개의 실제 출처

기존 [출처 연결 보고서](../pallet_remaining_evidence_connection_20261006_v1/README_KO.md)와 [원본 312행 연결 목록](../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/annotations/HANDOFF_SOURCE_CONNECTION.csv)을 직접 확인했다. 312행은 기존 Git 원자료 67개와 ZIP 원자료 245개로 나뉜다. 후자의 `source_id`는 `annotation_NNNNN` 형식이므로 현재의 `session:frame` ID와 같은 이름 공간이 아니다.

| 공개 별칭 | 크기(byte) | 원자료 JSON 바이트 검증 | 이미지 멤버 바이트 해시 | 경로에 기록된 세션 토큰 |
|---|---:|---:|---:|---|
| SOURCE_ARCHIVE_A | 26,698,730 | 51 | 51 | 20260903의 4개 시간 토큰 |
| SOURCE_ARCHIVE_B | 82,701,495 | 194 | 194 | 20260904의 9개 시간 토큰 |
| 합계 | 109,400,225 | 245 | 245 | 13개 메타데이터 토큰 |

두 컨테이너 전체 SHA256은 각각 `c011c8e9748e897d348520a9b26bc861f06defd908dbc1559266dc0dca701ff7`, `4d975387dbf89ed990b687b7a432af6f900d8b6dae4780c527ff1575c52c0e76`이다. 컨테이너 크기·SHA, 각 JSON 멤버 크기·SHA, 이미지 멤버의 인코딩된 바이트 SHA가 기존 연결 목록과 모두 일치했다. 수기 좌표 JSON을 해석하거나 이미지를 디코딩하지 않았다.

245개의 고유 이미지 SHA와 현재 238개의 고유 이미지 SHA 사이 교집합은 0개, 양쪽 차집합은 각각 245개와 238개다. 현재 PRIMARY192와의 인코딩 바이트 SHA 교집합도 0개다. **이는 동일하게 인코딩된 이미지 바이트가 없다는 결론이다. 재인코딩·크롭된 동일 프레임의 의미적 중복 부재까지 SHA만으로 증명하지 않는다.** 이번 감사는 새 픽셀 대응 검사를 수행하지 않았다. 세션 토큰도 경로에 저장된 메타데이터이며 독립적인 영상 동일성 판정이 아니다.

공개 [DISTINCT245_SOURCES.csv](evidence/DISTINCT245_SOURCES.csv)는 안정된 source ID, 컨테이너 별칭, 각 멤버 경로의 SHA, 원자료·이미지 SHA, 세션 토큰을 보존한다. [CURRENT238_IMAGES.csv](evidence/CURRENT238_IMAGES.csv)는 현재 ID와 이미지 SHA를 제공한다. 개인 원자료 경로·ZIP 원본·이미지·수기 좌표는 이번 패키지에 추가하지 않았다.

실제 기록의 출처·수의 의미·ID 또는 바이트 차집합을 확정했으므로, 원문 179줄의 출처 차이 확인은 이 실제 자료 목록을 대상으로 종결한다. 별도로 존재한다는 근거가 없는 다른 245장/110장 평가 집합을 가정하지 않는다.

## 초기 계측의 복원 불가 부분은 예외로 기록

[원 실행 영수증](../pallet_minimal_bi_ablation_v1/EXECUTION_RECEIPT.json)과 [30행 자원 원장](../pallet_minimal_bi_ablation_v1/RESOURCE_LEDGER.jsonl)을 다시 합산했다. 과거 계측을 새 실험으로 대체하거나 측정되지 않은 초를 만들어 넣지 않았다.

| 저장된 계정 | 재검산한 초 | 해석 |
|---|---:|---|
| GPU 관련 실제 단계 원장 합 | 426.890217 | GPU 단계로 기록된 경과시간 합 |
| CPU solver/analysis 단계 원장 합 | 776.154403 | CPU 단계로 기록된 경과시간 합 |
| GPU 작업을 포함한 보수적 CPU 단계 계정 | 1,021.713615 | 원 영수증의 보수적 단계 합; CPU cap 7,200초 이내 |
| 병렬 worker CPU 합 | 2,152.681296 | worker CPU 시간 합이며 wall time과 다름 |
| GPU 작업을 포함하는 구간 상한 | 1,734.409572 | 원 영수증의 포함 구간; GPU cap 3,600초 이내 |
| 기록 시작부터 공개 전 검증까지의 포착 구간 | 3,333.400213 | 약 55.56분. 모든 초기 작업·최종 push를 포함한 정확한 총 wall time은 아님 |
| 중복 영수증 메타데이터 | 5.252159 | 이미 기록된 단계의 메타데이터. 실행 시간에 재가산하지 않음 |

초기 소스 읽기·짧은 probe·인터프리터 시작과 종료 등의 개별 시간은 완전히 남아 있지 않아 정확히 복원할 수 없다. **이 항목은 완전한 정밀 계측 완료가 아니라 `CLOSED_WITH_DOCUMENTED_MEASUREMENT_EXCEPTION`으로 종결한다.** 원장·영수증의 단계 합과 보수적 구간은 검산했지만, 누락된 시간의 정확한 값이나 정확한 전 과정 총시간을 주장하지 않는다. 자세한 원문과 해시는 [RESOURCE_MEASUREMENT_EXCEPTION.json](evidence/RESOURCE_MEASUREMENT_EXCEPTION.json)에 있다.

전체 RGB→최종 pose 호출을 실행하지 않아 전체 latency/FPS가 NA인 것은 원문 403줄이 허용한다. 이를 근거로 전체 추론이 빨라졌다는 수치를 제시하지 않는다. 기존 검정 구현 확인이 불가능한 경우 p/Holm 값을 NA로 두는 것도 원문 348줄이 허용한다. 측정 공백과 허용된 NA를 구분한다.

## 추가 단독 비교와 공개 상태

원문 151–166줄의 필수 네 경로는 RAW와 N3+SubPix 묶음에 대한 F/B/I 비교였고, 166줄에 개별 모듈 기여를 분리하지 않는다고 명시돼 있다. 이후 사용자가 요청한 N3 단독·SubPix 단독 비교도 같은 238개 입력으로 완료했다. 수치·짝지은 회복/손상·신뢰구간·채택 판단은 [단독 비교 결과](../pallet_single_component_bi_v1/RESULTS_KO.md)와 [구성요소 유지·제거 판단](../pallet_single_component_bi_v1/COMPONENT_DECISIONS.md)에 있다. 기존 네 경로는 새 집계에서도 바이트 및 지표 일치를 검산했다.

이전 최소 실험·후속 guarded 실험·단독 구성요소 실험은 `a118730e8d97cfff15f79726403ddcbefd654ceb`까지 원격 main 게시를 읽기 전용으로 확인했다. 이 SHA는 이번 종결 파일의 미래 게시를 뜻하지 않는다. 현재 사용자의 main 게시 지시가 최초 문서의 연구 브랜치 지시를 대체하며, 이번 종결의 실제 게시 SHA는 최종 게시 영수증에서 별도로 확정한다.

## 재현

[감사 CLI](../../../scripts/research/pallet_instruction_closeout_v1/legacy_populations.py)는 표준 라이브러리만 사용한다. 기존 소스·확정 입력·원 ZIP 폴더·최소 실험 문서 폴더를 읽고 새 출력 폴더를 만든다. 이전 출력이 존재하면 덮어쓰지 않는다.

```bash
python -m scripts.research.pallet_instruction_closeout_v1.legacy_populations \
  --source-root "$SOURCE_ROOT" \
  --current-inputs "$CURRENT_INPUTS_JSON" \
  --archive-dir "$PRIVATE_ARCHIVE_DIR" \
  --minimal-docs "$MINIMAL_DOCS" \
  --output "$NEW_AUDIT_OUTPUT"
python -m unittest scripts.research.pallet_instruction_closeout_v1.test_legacy_populations -v
```

공개 [SEAL.json](evidence/SEAL.json)은 코드 SHA와 출력 5개 파일의 SHA를 제공한다. 자원 중복 계정 분리, cap 초과 시 중단, 원장·영수증 불일치 시 중단의 세 검증을 통과했다. 이번 종결 작업의 새 학습·seed·모델 추론·마스크·PnP·GT 수정·수기 좌표 해석·이미지 디코딩은 모두 0회다. 봉인된 기존 결과·실패 기록·사용자 수정·논문 파일은 수정하지 않았다.
