# 실행군의 정보·후단 차이

| 경로 | 초기점/이번 추가 학습 | 영상 관측 | 자세 후단 |
|---|---|---|---|
| N0_HISTORY | N3+SubPix 저장 출력 / 0 | 원 추론/기존 F 입력 | 기존 F의 실제 저장 자세 |
| RAW_OBSERVATION_MATCHED | 같은 N0 / 0 | 자동 V의 원q0, fullpoint만 | 공통 fresh점/부분normal adapter |
| LEGACY_RULE_MATCHED | 같은 N0 / 0 | 기존D1 규칙·recover stages·CAL/fixed FACE | 같은 공통 adapter |
| OBS_COMMON (3seed) | 같은 N0 / 합성768/235 | 두cue 합집합+MLP c=(.5,.5), fixed partial | 같은 공통 adapter |
| OBS_BI (3seed) | 같은 N0 / 같은 합성768/235 | 같은 후보/x/모델/cov; c만자동B/I | 같은 공통 adapter |
| E0_HISTORY | 과거 전체 시스템 / 이번0 | 상태별 보정·과거 전체 선택 경로 | 원E0 저장 자세 |
| H1_REP_HISTORY | 원H1 대표seed20261018 / 이번0 | 과거 최종 후보의 학습 선택 | 원H1 저장 자세 |
| PoseFix PF/YOLO anchor | 고정된 서로 다른 두 포트 경로 / 이번0 | 각고정PoseFix+SubPix 출력 | 각원후단의 저장 자세 |

주 대비는 같은 seed의 COMMON–BI뿐이다. 합성 자산 분할/C1·C2, CAL 실사 감독 이력, 양cue, 공통B/I noise처리를 공유했다. 다른 전체 시스템 간 차이를 c의 효과로 귀속하지 않는다. 모든 새 matched 경로가 N0로 시작하므로 이번 결과로 N3 단독/SubPix 단독 제거 효과를 새로 주장하지 않는다. 기존 비교군과 실패 기록을 모두 유지했다.
