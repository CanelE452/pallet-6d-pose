# 가시성·정사각형 CLI 마감

기존 사람 입력3030점과 보호된71점을 합쳐3101개 참조 코너의 가시성 분류를 검산했다. 누락0점이다. 좌표·원본 주석·고정 예측은 변경하지 않았다.

| 분류 | 참조 코너 |
|---|---:|
| DIRECT_VISIBLE | 2313 |
| EXTERNAL_OCCLUDED | 281 |
| SELF_OCCLUDED | 462 |
| OUT_OF_FRAME | 45 |
| UNKNOWN | 0 |

가시성별 표는 프레임마다 전체 물체 대칭을 먼저 선택한 뒤 원래 GT 코너ID로 나눴다. 각 가시성 집단에서 대칭을 다시 선택하지 않았다. DOPE의 매칭 성공 프레임 내부 결측 코너도 PCK 분모와 벌점에 유지하고 조건부 중앙값/P90에서 제외했다.

[가시성 결과](VISIBILITY_RESULTS.md) · [seed별 결과](VISIBILITY_PER_SEED.md) · [3101점 출처](STATIC_VISIBILITY_MERGE_AUDIT.json) · [모든 코너 원시 지표](PER_POINT_SCORES.csv)

정사각형119장은 manual_declared602점과 manual_in_frame600점을 모든 방법에 각각 동일하게 적용했다. 두 모드를 섞지 않았으며 기존 고정 결과에 대한 전수 회귀검산을 통과했다. N3 대Base와 N3 대N2 비교는 별도 열로 보존했다.

[119장 두 모드 결과](SQUARE119_RESULTS.md) · [Base/N2 대비 N3 변화](SQUARE119_COMPARISONS.md) · [119장 지표·분모·제한](SQUARE119_RESULTS.json)

119장은 단일 촬영 세션이다. 독립6D 참조가 없으므로 이동·회전 정확도는x이며 물리적6D 정답을 재구성하지 않았다. 모든 영상의 치수가 같아 치수 입력의 인과 효과는 식별할 수 없다. 세션 일반화 신뢰구간은NA로 둔다.

기존 정사각형150장은7개 세션의 가림 없음103/중간44/어려움3으로 별도 재집계했다. 과거 고정R0/N0/N2 결과만 재사용했다. 직접 클릭681점과 PnP 포함1200점 proxy를 별도 모드로 유지했으며119장과 합산하지 않았다. 이150장에 대한 동일 계약N3/DOPE/ResNet 결과는x다.

[기존150장 별도 결과](HISTORICAL_SQUARE150.md) · [출처와 제한](HISTORICAL_SQUARE150.json)

실행 검산72개PASS, 실제 경과4.679초. 새 학습·optimizer update·추론·PnP·PDF·업로드·push는0회다. 새 표의 원고 반영 여부는 별도 원고 작업의cell map에서 확인해야 한다.

[실행 명령·입력 SHA-256·회귀검산](EXECUTION_VALIDATION.json)

[전체 대칭과 조건부 관측 분모 의미 검증 4개](SEMANTIC_TEST_VALIDATION.json)도 통과했다. 기계 실행으로 새 사람 판정이나 클릭 기록을 생성하지 않았다.

[최초 실행과 최종 실행을 모두 포함한 실제 CPU 비용](ACTUAL_CPU_COST_LEDGER_KO.md)을 별도로 보존했다.
