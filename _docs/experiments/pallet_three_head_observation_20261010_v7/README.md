# 수정된 세 head × 두 실제 관측 공급

이 디렉터리는 수정 감독으로 이미 학습한 GEOMETRY_ONLY·IMAGE_NO_ROLE·IMAGE_ROLE의 마지막3000update checkpoint를 같은 source-only calibration·경계 관측 decoder·강건 point PnP로 비교한다. 쉬움153·중간92, 총245장을 유지한다. severe74는 이번 평가에서 제외하고 이전319장 증거는 보존한다. 이번 새 학습·RGB·seed는0이다.

**실제245장 평가에서 고정 N3→cornerSubPix를 이기지 못했다.** primary IMAGE_ROLE_BOUNDARY_ONLY는 평균 위치10.46185cm·회전11.61153°로 N3의9.75455cm·10.91484°보다 나빴고, 40새 자세·205기본 반환·완전 실패0이었다. H/LOO 계약은 코드와 원행 검산을 통과했으나 선택한 선·교점의 상대 정확도와 충분한 대응 공급이 확보되지 않았다. 감독 오류 수리, corrected9000update 학습 완료, 이번 실제 방법의 실패를 구분한다. 최상위 개선 목표는 미달성이다.

source CAL16forward/256exposure(ROLE 새0), 계약9검사·CPU13검사·source 독립 산술198848개·stream4그룹, GT 없는 fresh245장 inference, scoring2450행, 통계와432CI 검산,1500회 전체 경로 시간,8개 그림 생성·시각 검토가 실제 완료됐다. private weights/GT 없는 공개 CI checker도432슬롯 PASS했고, original gzip6개가 없는 fresh dependency bundle에서 byte 복원·public moment 재검토까지 완료했다. 사후 runtime scalar는 A 오류를 보존하고 B에서148824검사 PASS했다. 이 실험과 저장자료 감사는 완료됐고 개선 목표는 미달성이다. 게시 여부는 별도 publication 영수증으로 확인한다.

| 먼저 볼 자료 | 확인하는 내용 |
|---|---|
| [RESULT_KO.md](RESULT_KO.md) | 사용자 질문의 실제 답, 수치·관측 품질·가림/H·한계·사례 이미지 |
| [REPRODUCE.md](REPRODUCE.md) | 저장 원행 재검산과 private 입력이 필요한 실제 실행의 차이 |
| [BUILD_LEDGER.json](BUILD_LEDGER.json) | 실제 실행·거절·실패 시도·호출량·입력/원행 SHA |
| [PROTOCOL.json](PROTOCOL.json) / [EVALUATION_CONTRACT_KO.md](EVALUATION_CONTRACT_KO.md) | GT 전에 고정한245장·8방법·primary·16대조·432CI와 판정 정의 |
| [INFERENCE_RECEIPT.json](INFERENCE_RECEIPT.json) / [GEOMETRY_SEAL.json](GEOMETRY_SEAL.json) | geometry1960·fixed490·관측735행의 완성/cleanup·GT 미접근·byte 봉인 |
| [source calibration](source_calibration/COMPLETION.json) | 두 새 head16forward, ROLE9파일 exact-byte 재사용/current0 |
| [METRICS.json](METRICS.json) / [METRICS.csv](METRICS.csv) | 전체 운용·NEW·기본 반환, 쉬움/중간별 원행 통계와16개 대조 |
| [VERIFICATION.json](VERIFICATION.json) / [REPROJECTION_CHECKS.json](REPROJECTION_CHECKS.json) | 432CI 슬롯 독립 산술과 H 교체·관측 좌표 보존 |
| [RUNTIME.json](RUNTIME.json) / [FIGURE_BINDINGS.json](FIGURE_BINDINGS.json) | 실제1500경로 시간, 원행과 연결한8PNG |
| [METHOD_CAUSAL_AUDIT_KO.md](METHOD_CAUSAL_AUDIT_KO.md) | 코너 특성·H/LOO·관측 소유권 문제와 아직 없는 같은 sparse 관측 대조 |
| [PUBLIC_CI_CHECKS.json](PUBLIC_CI_CHECKS.json) / [PUBLIC_FRESH_BUNDLE_CHECKS.json](PUBLIC_FRESH_BUNDLE_CHECKS.json) | stdlib 공개 CI1회, 실제 absent→restore와 fresh moment 검토 |
| [성공 runtime scalar B](RUNTIME_VALIDATION_ATTEMPT_B/RUNTIME_VALIDATION_CHECKS.json) / [첫 A 이유](RUNTIME_VALIDATION_ATTEMPT_A/REASON.json) | 공식 runtime1회와 사후 checker2시도를 구분 |
| [ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json) / [PROTECTION_PRE_PUBLICATION.json](PROTECTION_PRE_PUBLICATION.json) | full raw byte parts와 기존 자료·사용자 변경 보존 |

`BOUNDARY_ONLY`는 실제 관측 교점만 sparse fit pool에 넣고, `CORNERWISE_HYBRID`는 기존 native N3와 코너 LOO로 선택한다. 적용 H는 두 경로 모두 final fit에서 제외하며 새 R,t가 나오면 해당 초기 좌표를 재투영으로 대체하고 다시 fit하지 않는다. no-mask native 강건 대조는 별도로 둔다. proposal ID·실제 fit/inlier·출력 좌표는 구분한다. source confidence나 saved-row 검산 PASS는 실사 물리 경계의 소유권 또는 자세 개선의 증명이 아니다.

core/EVAL은 정확도 점수 접근 전에 frozen이며 성능을 보고 설정을 바꾼 재평가는0회다. 공개 저장자료 검토는 private weights/GT 없이 가능한 전용 검사와 실제 전체 경로 재실행을 구분한다. 전용 research branch 게시와 remote SHA 확인은 실제 publication 단계의 영수증을 따른다. main에는 변경하거나 push하지 않는다.
