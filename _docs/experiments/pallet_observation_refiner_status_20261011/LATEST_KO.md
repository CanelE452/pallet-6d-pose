# 최신 확인: 관측 선택·LOO·강건 PnP와 재투영

“가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?”

**현재 알려진 DEV proxy의 쉬움153·중간92, 전체245장에서는 최종 point-consensus 복원의 위치·회전 평균이 모두 개선됐다고 확인하지 못했다. 정확도 목표는 미해결이다.** Severe74는 이번 실행에서 제외했다. 과거 branch/main의 319장 평가 기록과 이번245장 결과를 혼용하지 않는다.

| 전체245 운용 방법 | 새 자세 / fallback | 위치 평균 cm | 회전 평균 deg | ADDsym 평균 cm |
|---|---|---|---|---|
| 고정 N3→cornerSubPix | 고정 대조 | 9.754548028679153 | 10.914841863166323 | 17.028646667097725 |
| N3 관측 ordinary PnP | 244 / 1 | 8.951574173959987 | 10.61763914609911 | 16.21154786029048 |
| 같은 hybrid q의 ordinary PnP | 244 / 1 | 9.207153885736464 | 10.613463948751368 | 16.43968649819084 |
| 같은 hybrid q의 원 point consensus | 79 / 166 | 10.05503053164391 | 11.715323669869344 | 17.868503089664262 |
| 같은 hybrid q의 결정 복원 | 240 / 5 | 10.19473418899381 | 13.639332540977922 | 19.646391027436795 |

수치 산출 NEW는 정확한 물리 자세의 인증이 아니다. 전체 운용245장을 유지하고 fallback의 큰 오차를 삭제하지 않았다. 모든 분산·표준편차·중앙값·P90, 공통 산출/NEW의 paired 비교와 고정 session bootstrap CI는 [V18 결과](../pallet_consensus_decision_repair_20261011_v18/RESULT_KO.md)와 원행·검산 파일에 연결한다. 해당 보고서는 실제 공개 복사본 검산4개가 끝난 뒤 생성됐으며 이미지8개와 실행 증거를 포함한다.

## 실제 고친 것과 남은 문제

- 별도 branch `training.targets()`에서 수치 ray miss를 NONE으로 감독한 2,164 query는 실제 mesh/wire/depth의 POSITIVE/NONE/IGNORE로 수정했다. main N3 문제가 아니다. 최초 읽기 전용 인수인계 당시 재학습은 없었고, 이후 승인된 수정 감독3조건×3,000 update는 별도의 완료된 역사 실험이다. 이번 새 학습·RGB·seed는 0이다. 수정 전 IMAGE_ROLE의 0/319를 수정 후 결과로 주장하지 않는다.
- 같은 support의 다른 비용 해도 거부하던 추가 ambiguity veto161 frame을 원 inlier 수→같은 전체 관측집합의 truncated SSE 결정으로 복원했다. 관측·후보·refit·수치 threshold는 그대로이며 실제 tie1/관측·기하 부족4는 fallback으로 남는다. 이는 산출률을 복원했지만 정확도 개선은 아니다.
- H는 고정 초기 N3의 self-occ 집합이다. LOO는 H∪{시험 corner}를 제외한 다른 관측만 사용한다. 최종 NEW fit에서 H 초기 좌표를 빼고 새 R,t로 한 번 재투영해 교체한다. 그 재투영을 독립 관측처럼 다시 PnP에 넣지 않는다. 마스크 불일치/변경만으로 frame을 거부하지 않는다.
- [방법 진단](../pallet_consensus_decision_repair_20261011_v18/METHOD_DIAGNOSIS_KO.md)과 [대응 원행 검산](../pallet_consensus_decision_repair_20261011_v18/CORRESPONDENCE_AUDIT_CHECKS.json), [저장 readout](../pallet_consensus_decision_repair_20261011_v18/CORRESPONDENCE_AUDIT_READOUT.json)을 구분해서 본다. adopted q133개 중 proxy error 개선80/악화53, >8px인13개는 두 backend의 최종 inlier에 남았다. NEW240 frame 중 known proxy correct FIT<4인75는 독립적인 실제 물리 대응 부족의 인증이 아니다. 실제 edge/corner ownership을 기존 자료만으로 인증하지 못했다.

## 검산·비용·공개 상태

V18 point decision/output 검산626,482 PASS, 정확도1470 saved score/972 moment 수치/45 paired 집합/135 CI 독립 PASS, 실제 fresh 전체 경로150회(20 warmup+130 measured)의 평균798.319045ms와 저장 시간677,122 predicates 독립 PASS를 완료했다. 계산된 mean·variance·sd·median·P90을 원행에서 검산했다. 캐시 replay를 지연으로 쓰지 않았다. 원 V16 full scalar198,134,007 중564 FAIL은 모두 line 경로이며 그대로 보존하고, 이 좁은 point 검사를 전체 V16 PASS로 확대하지 않는다.

[공개 복원 안내](../pallet_consensus_decision_repair_20261011_v18/PUBLIC_REVIEW.md)는 버전별 ordered parts 복원과 네 저장 reader 명령을 제공한다. 공개 byte bundle529개/5,073,312,474B(518 직접 복사+11 ordered parts 복원)와 실제 복사본 reader4 검산이 모두 PASS다. 원 통계 CLI의 상대 경로 오류2개도 계산 전 실패로 보존하고 절대 경로 안내로 바로잡았다. V16–18 정상 commit/push 확인은 진행 중이다. 앞선 V10–15 게시 SHA691a4ece4e719ddfda2aa69a6afb6578f8923511은 [당시 영수증](PUSH_V10_V15_RECEIPT.json)에 있다. 이후 현재 게시 영수증이 생기기 전에는 게시 완료로 주장하지 않는다. main 수정·push·자동 병합·force push는 하지 않는다.

현재 GT는 기존 주석8점+K+등록 cuboid로 만든 GEOMETRIC_PROXY이고 독립 물리 실측/전점 독립 클릭을 보증하지 않는다. 최소 남은 증거는 문제가 된13개 q의 실제 corner와 incident physical edge ownership을 독립적으로 인증하는 것이다. 기존 source-test와 실사 관측 trace는 그 인증을 대신하지 않는다. 새 수동 주석·촬영·RGB·추가 학습·성능에 맞춘 설정 반복을 하지 않고도 확인 가능한 코드·원행·실패·검산은 모두 보존한다.
