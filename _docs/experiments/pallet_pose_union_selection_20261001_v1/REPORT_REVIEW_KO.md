# 결과 보고서 독립 검토

이 문서는 `make_report.py`의 사전 코드 검토와, 이후 실제 생성 파일에 대한 검증을 구분한다. 검토자는 새 이미지 추론·학습·GT를 이용한 포즈 채점을 실행하지 않았다. 최초에는 코드가 만들어야 할 수량을 확인했고, **완료된 최종 보고서·공개 CSV·저장된 결과 배열의 일치까지 확인했다. 최종 검증은 PASS**이며 세부 결과는 마지막 절에 있다. 이는 보고서의 정확성에 대한 PASS로, 방법의 성능 기준 통과를 뜻하지 않는다.

사전 검토 대상 `make_report.py` SHA256: `e3f2ab80933965382aa306b39465b68ceacb76cf4de7c0d23055f5cbfe2e2928`. 이후 수정 여부와 실제 출력 검사는 이 문서의 후속 기록으로 확인한다.

## 숫자·단계 구분 검토

보고서는 TRAIN의 정답 기반 whole-pose cost oracle을 실제 배포 가능한 성능으로 표시하지 않는다. TRAIN oracle 통과 후에만 학습한 6개 최종 선택기를 VAL에서 검증하며, 세 UNION seed가 각 비교군의 모든 조건을 통과하지 못하면 실사로 진행하지 않는 사전 중단 규칙을 설명한다. 합성 VAL 실패를 실사 성능 하락을 직접 측정한 결과로 바꾸어 쓰지 않는다.

기존 실사 표는 이번 새 선택기의 결과가 아니라 [앞선 실사 평가](../pallet_pose_stable_improvement_20261001_v1/REPORT_KO.md)의 결과임을 명시한다. 해당 단계의 `RESULTS.json`에서 `NATURAL99/full_population`을 읽어 확인한 값은 다음과 같다.

| 방법 | T 중앙값 cm | R 중앙값 ° | T P90 cm |
|---|---:|---:|---:|
| R0 | 12.4032510683 | 5.2175832874 | 120.4713701029 |
| FULL125 | 11.9864643570 | 4.4106922799 | 120.8240802781 |
| DIVERSE251 seed별 통계 평균 | 12.0037217007 | 4.5672163093 | 129.5027320229 |

보고서의 소수점 여섯 자리 수치 9개는 모두 일치한다. 세 번째 행은 각 seed의 중앙값/P90을 먼저 구한 후 산술 평균한 값이며, seed 전체 예측을 합친 중앙값이 아니다. 원래 보고서와 새 설명 모두 이 구분을 유지한다.

수동 교사의 계보도 기존 보고서와 일치한다. 새 실사 정답을 추가하지 않았으나 기존 Replay 교사가 실사 9장·수동 코너 38개를 썼다는 점을 남기므로, 전체 방법에 실사 GT 감독이 전혀 없었다고 주장하지 않는다.

## CSV 행 수와 변수 재사용

`records`는 TRAIN 후보 목록, `selected`는 TRAIN cost 선택 목록이다. `for model, rows in choices.items()`가 일시적으로 `rows`를 사용하지만 VAL 분기에서는 `rows=[]`를 새로 만든다. 이후 CSV 수량 기록 전에 이를 다시 다른 목록으로 바꾸지 않는다. 따라서 이 사전 검토 버전에는 지적된 `selected/rows` 재사용으로 인한 행 수 오류가 없다.

| 출력 | 기대 행 수 | 유일키 |
|---|---:|---|
| `SOURCE_TRAIN_CANDIDATES.csv` | 2,598×4×2 = 20,784 | `(id, model, hypothesis)` |
| `SOURCE_TRAIN_COST_CHOICES.csv` | 2,598×3 = 7,794 | `(id, model)` |
| `SOURCE_VAL_FRAME_RESULTS.csv` | 1,024×10 = 10,240 | `(id, model)` |
| `SOURCE_VAL_CHECKS.csv` | 3×3×5 = 45 | `(model, baseline, criterion)` |
| `TRAINING_LOG.csv` | 6×330 = 1,980 | `(arm, seed, step)` |

VAL 관련 두 파일은 실제 학습·VAL 단계를 실행한 경우에만 기대한다. 전체 프레임 CSV에서 실패를 삭제하지 않고 T/R `inf`와 `available=False`로 유지한다. 공개 CSV와 원 NPZ의 값 및 JSON 요약은 파일 생성 후 마지막 절의 검사로 확인했다.

## 실제 추론 횟수와 출력 행 수

네 completion receipt가 모두 생성된 뒤 읽은 실제 값이다. 각 refiner에는 5,120개 출력행이 있지만 neural forward는 5,119회다. 따라서 단순히 `5,120×3`을 실제 계산 횟수로 쓰면 3회를 과대 계상한다.

| 모델 | 출력행 | 실제 새 image forward | 기록된 loop 초 |
|---|---:|---:|---:|
| R0 | 5,120 | 4,951 = 신규 4,942 + smoke 9 | 109.050340 |
| DIVERSE251_s1 | 5,120 | 5,119 | 151.437200 |
| DIVERSE251_s2 | 5,120 | 5,119 | 152.964213 |
| DIVERSE251_s3 | 5,120 | 5,119 | 150.907459 |
| 합계 | 20,480 | **20,308** | **564.359213** |

R0의 인증 캐시 178개는 새 forward로 세지 않는다. smoke 9회는 보존된 첫 실패 1회와 고정된 재검사 8회다. 네 모델 모두 `model_state_before == model_state_after`였다. `make_report.py`는 모델별 receipt의 실제 attempt 수를 합산하므로 현재 구조는 20,308회를 보고한다. 최대 예산 20,311회와 구별된다.

loop 초는 이미지 로딩·캐시 검증·출력 저장 등이 포함된 실행 구간의 wall time이다. GPU kernel 시간이나 전체 작업 소요 시간으로 해석하면 안 된다. 보고서는 import·준비·초기 smoke·보고서 작성까지 포함한 전체 시간과 구분한다.

## 링크와 이미지 해석

사전 검토의 Markdown 링크 대상 27개를 확인했다. 기존 실사 보고서·전체 99장 gallery·association 감사·이전 한계 분석은 publication checkout에도 존재한다. 이번 namespace의 코드·감사 문서·smoke 이미지도 실제로 있다. 아직 없는 대상은 현재 단계에서 생성할 TRAIN/VAL CSV·JSON·plot이며, 결과 생성 뒤에 다시 확인해야 한다.

smoke 그림은 결과를 보기 전에 RGB SHA로 정한 8장이다. 실제 합성 RGB와 R0 예측을 보여주며 GT overlay·성능 우수 사례 선택을 하지 않는다. TRAIN oracle plot과 학습된 VAL plot도 서로 다른 그림·제목으로 구분된다. 이 이미지를 실제 자연 가림의 개선 사례라고 표시하지 않는다.

## 발견한 실행 범위 제한과 전달 사항

초기 코드에는 `SOURCE_TRAIN_GATE`가 `STOP_NONFINITE_OR_NONPOSITIVE_TRAIN_SCALE`로 끝나는 경우의 report 분기가 없다. 이 경우 `R0_ONLY_cost_oracle`·`cost_choices` 키가 생성되지 않아 보고서 코드가 중단된다. 루트 담당자에게 전달했으며, 실제로 이 상태가 발생하면 scale-invalid 원인만 보고하고 존재하지 않는 oracle/학습 결과를 만들지 않아야 한다. 정상 scale에서의 TRAIN gate FAIL 또는 이후 VAL FAIL 보고 경로와는 별개다.

이번 실제 실행에서는 sT=2.4636887551191258 cm, sR=1.113474019956766°가 모두 finite·양수이고 TRAIN gate가 통과했다. 따라서 위 분기는 이번 보고서에서 사용되지 않았고 실제 생성 결과의 차단 사유가 아니다.

추가 검토 중 source VAL 참조 무결성 보강을 발견해, 학습 protocol 봉인 전에 루트 승인으로 적용했다. `source_val.py`는 동결 `SOURCE_CONTRACT`가 바인딩한 기존 split lock·geometry·records의 정확한 SHA를 먼저 확인하며, 현재 파일들이 함께 바뀌어도 이를 새 기준으로 신뢰하지 않는다. VAL gate/metric 파일은 routing lock 검증 전에 읽지 못한다. TRAIN label 컨테이너의 SHA 확인과 VAL 정답 값 미접근을 별도 disclosure로 구분했다. 최종 코드 SHA는 `867ac51c6e43808cc5af4661a885b5d5860612a9d7b5be2a1486210555bf0c3b`이며 AST와 가상 입력 self-check가 통과했다. 실제 VAL 채점 또는 학습 실행은 하지 않았다.

## 생성 후 최종 검증: PASS

루트 담당자가 TRAIN·6fit·VAL·보고서 생성을 모두 완료한 뒤 검증했다. `REPORT_DATA.json.complete=true`를 확인한 후에만 저장된 TRAIN/VAL T/R 배열을 읽었다. 이 단계에서는 renderer GT 파일, 원래 평가 GT, PnP 함수 또는 이미지 모델을 사용하지 않았다.

| 확인 항목 | 실제 결과 |
|---|---|
| 공개 CSV 5개 총행 | **40,843행**; 위 표의 다섯 수량과 모두 일치 |
| 행 유일키 | 모든 CSV에서 중복·누락 없음 |
| TRAIN 후보 수치 | 20,784행 T/R/cost·available이 이미 생성된 label NPZ와 일치 |
| TRAIN oracle 선택 | 7,794행 선택·동률 정보가 원 선택 JSON과 일치 |
| TRAIN 실패 보존 | 무효 후보 8행 유지; 4개 oracle 요약 모두 2,598행·실패 1개 유지 |
| VAL 전체 수치 | 10개 방법×1,024행 T/R·available·선택·fallback이 저장된 metric NPZ와 routing JSON에 일치 |
| VAL 기준 CSV | 45개 판정 전부 gate JSON과 일치; 실패 3개 |
| 학습 로그 | 6개 fit×330 update=1,980행, 모든 필드가 원 JSONL trace와 일치 |
| 요약 통계 | TRAIN oracle 4개+VAL 10개, full-population/conditional 중앙값·P90·실패 수 모두 일치 |
| Markdown 수치 표 | 위 14개 source 요약의 표시 정밀도와 실패 수 모두 일치 |
| 최종 Markdown 링크 | 28개 모두 기존 publication base 또는 이번 새 namespace에 존재 |
| 그림 | PNG 3개 정상 decode; smoke 1,680×2,040, 두 plot 각각 2,100×812 |
| 실제 실행 비용 | image forward 20,308회, GPU 실행 loop 합 564.359213초, CPU fit loop 합 2.434941초 |

VAL의 실제 실패 3개는 모두 seed3이며 다음과 같다.

| 비교 | 미통과한 조건 | 보고서의 설명 |
|---|---|---|
| UNION_s3 vs R0_ONLY_s3 | T 중앙값 엄격 개선 | 동일한 T 중앙값이므로 엄격 개선 아님 |
| UNION_s3 vs R0_ONLY_s3 | R 중앙값 엄격 개선 | 0.608126750→0.611122982° |
| UNION_s3 vs R0_GEO | T 중앙값 엄격 개선 | 1.674594149→1.679493903 cm |

seed1·2는 각 세 비교군의 모든 조건을 통과했다. 보고서는 이 부분적 개선과 seed3의 작은 차이를 공개한다. 작은 실패를 모든 방법의 큰 악화로 확대하지 않았으며, 통과한 seed만 골라 실제 평가로 진행하지 않았다. 모든 VAL 방법의 실패 pose 수는 0이다. 최종 상태 `SOURCE_RANKING_NO_JOINT_SIGNAL`, 새 실사 routing/evaluation=false, 안정적 실사 동시 개선 미달성을 명확히 남겼다.

공개한 [모델 파라미터 JSON 6개](model_parameters/)와 [export 목록](MODEL_EXPORTS.json)도 원래 `final.pt`와 독립적으로 대조했다. weight/bias/mean/std 총 **1,698개 float**를 float32로 복원하면 모두 원본과 정확히 일치한다. export 총 49,798 bytes이며, 각 파일은 학습 protocol·원 checkpoint SHA·채택하지 않았다는 상태를 보존한다. 이 검사는 모델을 다시 학습하거나 실제 데이터에 추론한 결과가 아니다.

최종 검사 대상 SHA256은 다음과 같다.

| 대상 | SHA256 |
|---|---|
| `make_report.py` | `bf5abdf7202307e039abef51f4807f6933fe526db26aec7c2d65016c52a4fe81` |
| [최종 결과 보고서](REPORT_KO.md) | `452cf8c1c5ed04463eab9b5adf34a23b7a692f70f0bd427b20f1179ee86d9bba` |

`REPORT_DATA.json`의 코드 바인딩도 위 최종 코드와 일치한다. 이번 검토 범위에서 공개 수치·링크·학습 로그·파라미터 export의 남은 불일치는 발견하지 않았다.
