# CLI 실행 지시문 — 상태별 자세 후보의 선택 실패 32장과 후보 부재 46장 진단

작업: 합의한 다음 분석을 구현·실행하고, 실제 원인 근거와 수정 우선순위까지 보고한다.
경로: 0 → 2 → 3. 계획 문서만 작성하고 멈추지 않는다.
저장소: `CanelE452/pallet-6d-pose`
결과 기준: `research/statewise-correspondence-diag-20261016-v1`, `407de865445e4987614f50f893d2ccea78d507b6`.
작성 시 확인한 통합 main: `2fe1969b7a999d9a8d460703be4bfd11317e7f07`.

**이번 목표는 기존 e보다 좋은 방법을 이미 만들었다고 주장하는 것이 아니다. 성공 후보가 있는데도 선택하지 못한 32장에서 어떤 점수·선택 단계가 문제인지 확인하고, 후보 자체가 없는 46장과 분리해 다음 수정 한 가지를 결정하는 것이다.**

기존 N3·SubPix·e와 S/B/I/O/T/U 상태별 처리라는 연구 방향은 유지한다. 이전 952개 자세 재계산, 관측 전달 복원, A/B guard, 성분 제거 실험을 다시 수행하지 않는다. 이번에는 **고정 후보에 대한 점수·잔차 계산과 저장 결과 분석**을 한다.

## 0. 목적·실행 권한·중단 경계

### 0.1 이 분석으로 바꿀 판단

| 질문 | 왜 필요한가 | 확인 후 달라지는 판단 |
|---|---|---|
| 다른 W/D 가설의 좋은 후보를 놓친 10장은 왜 생겼는가? | 가설 선택 오류와 최종 단계에서만 생긴 좋은 후보를 구분해야 함 | 초기 가설 고정 정책, 단계 간 평가 일관성 중 무엇을 고칠지 결정 |
| 같은 가설 안의 좋은 후보를 놓친 22장은 왜 생겼는가? | W/D를 맞춰도 남는 선택 오류가 있음 | 후보 비용 항·관측 출처·tie-break·점수 구현 중 수정 대상을 결정 |
| 성공하는 114장에서도 같은 현상이 있는가? | 실패32장만 보고 만든 점수는 기존 성공을 망칠 수 있음 | 실패에만 맞춘 사후 설명인지, 전체에 공통인 결함인지 구분 |
| 후보가 없는 46장에서는 어느 단계가 제한되는가? | 선택기만 바꿔서는 해결할 수 없는 집단을 구분해야 함 | 관측·대응·진입 조건·시작 후보 중 다음에 확인할 위치 결정 |

[진단 설계·미검증] 아래 분석이 원인을 얼마나 분리할지는 실제 저장 근거에 달려 있다. 큰 비용, 작은 공분산, 큰 회전 오차만 보고 특정 항이나 참조가 틀렸다고 단정하지 않는다.

### 0.2 이번에 바로 실행할 것

- 기존 소스·보고서·원시 후보·관측 packet·seal 읽기와 필요한 분석 코드 작성.
- 고정 R/t에서 원래 목적함수·잔차·제약을 평가하는 계산. 좌표·후보·관측·가중치를 바꾸지 않는다.
- 원래 선택 순서와 최종 출력의 재집계, stage별 후보 순위·점수 차이·관측별 기여 분석.
- 참조를 사용하는 사후 진단: 성공 후보의 존재, 실제 선택과의 비교, 기존 방법별 손익 비교.
- 고정된 원 후보 은행에서만 수행하는 제한된 점수 항 영향 진단(8절). 새로운 자세를 만들지 않는다.
- 원영상과 **이미 저장된** mask·선·후보의 시각화. 새로운 검출·선 탐색·분할은 하지 않는다.
- 공개 가능한 새 분석 코드·작은 결과·재현 안내의 별도 브랜치 commit·push.

### 0.3 이번에 하지 않는 것

- 새 학습·CAL 재fit·증류·새 오류합성·새 seed·가중치 변경: **0회**.
- 새 YOLO/N3/PoseFix/상태 분류/분할기 forward: **0회**.
- 새 PnP·SLSQP·least-squares·pose refinement·후보 생성: **0회**.
- 이전 네 경로×238장의 replay를 다시 실행해 비용 분해를 얻는 행위: 금지.
- PRIMARY 오차를 보고 loss/weight/threshold를 탐색하거나 좋은 후보 레이블로 선택기를 학습하는 행위: 금지.
- 32장이나 특정 ID에만 새 경로 적용, 좋은 121장만 최종 평가, 참조·W/D·C2 규약 재정의: 금지.
- N3/SubPix 제거, O/T/U 삭제, 모델 투영을 관측점으로 재주입, 논문·운영 기본값 교체: 금지.
- Jetson/FLOPs/TensorRT, 새 논문 PDF 제작: 해당 없음. 현재 선택 실패 원인 판단에 필요하지 않다.

분석 코드의 구현·키 매핑·배포 오류는 스스로 수정하고 이전 시도를 보존한다. **실제 방법의 계산 오류를 발견하면 최소 반례와 수정안까지 작성하되, 기존 추정기를 고쳐 전체 성능을 새로 산출하는 작업은 별도 승인 대상으로 남긴다.**

## 1. 결과 기준과 작업공간을 먼저 고정

[확인·원보고서] PRIMARY192=TEST49+REMAINDER143, CAL46 별도이며 전체 장부는238장이다. e 최종 성공114, 실패78. 원 e consumer 범위에서 성공 후보가 있는 영상146, 선택 family 안에서는136, 최종 선택은114다. 최종 실패는 선택손실32(가설10+같은 가설22)와 관측/제안 제한46으로 나뉜다. 46 중4장은 진입 조건 때문에 후보 생성이 없었다. 좋은 후보가 모두 사라진 pruning loss는0이다.

위 값은 결과를 맞춰야 할 목표가 아니라 **같은 원자료를 읽었는지 검산할 기준**이다. 다르면 원인을 확인하고, 임의로 ID·후보·정답을 변경해 숫자를 맞추지 않는다.

### 1.1 읽을 기존 문서와 산출물

다음 접두부를 이 문서 안에서만 `DIAG_ROOT`라고 부른다. 기존 코드 변수라고 가정하지 않는다.

```text
_docs/experiments/pallet_statewise_correspondence_diagnosis_v1/
```

필수로 읽을 기존 파일:

- `DIAG_ROOT/REPORT_KO.md`
- `DIAG_ROOT/ORACLE_LIMITS.md`
- `DIAG_ROOT/OBSERVATION_SCHEMA_MAPPING.md`
- `DIAG_ROOT/PLAN_AND_CONTRACT.md`
- `DIAG_ROOT/REPRODUCE.md`
- `DIAG_ROOT/POSEFIX_TRANSFER_AUDIT.md`
- `DIAG_ROOT/oracle/ORACLE_SUMMARY.json`
- `DIAG_ROOT/oracle/ORACLE_FRAME_TABLE.csv`
- `DIAG_ROOT/oracle/ORACLE_ROWS.jsonl.gz`
- `DIAG_ROOT/oracle/ORACLE_SEAL.json`
- `DIAG_ROOT/oracle/DIAGNOSTIC_CASE_STAGES.json`
- `DIAG_ROOT/metrics/ROW_METRICS.csv`
- `DIAG_ROOT/metrics/PAIR_TRANSITIONS.csv`
- `DIAG_ROOT/metrics/SCORE_SEAL.json`
- `DIAG_ROOT/evidence/EXTRACTION_SEAL.json`
- `DIAG_ROOT/evidence/REPLAY_PARITY.json`
- `DIAG_ROOT/evidence/SOURCE_AND_INPUT_MANIFEST.json`

대형 파일의 실제 위치는 이전 실행 폴더·원 manifest에서 확인한다. 공개 디렉터리에 파일이 없다고 기록이 없다고 결론 내리거나, 이름만 같은 다른 실행을 연결하지 않는다. 필요한 파일만 원 로컬 자료/검증된 Release에서 복원하고 거대한 archive 전체 재다운로드를 기본 동작으로 삼지 않는다.

### 1.2 재사용할 실제 코드

```text
scripts/research/pallet_statewise_correspondence_diagnosis_v1/
    candidates.py
    oracle.py
    lineage.py
    typed_replay.py
    parity.py

scripts/research/pallet_visibility_curriculum_20261013_v1/
    select_frozen_wd_partial_refinement_v4_13e.py
    statewise_two_wd_family_v4_9.py
    evaluate_global_outline_pose_v4_10.py
    statewise_global_outline_solver_v4_10.py
    statewise_global_outline_solver_v4_11.py
    statewise_partial_BI_outline_solver_v4_13.py
    statewise_partial_BI_u_solver_v4_13.py
```

실제 활성 backend와 의존 모듈은 원 packet의 source hash와 위 호출부에서 끝까지 확인한다. 문서의 클래스명·헤더 설명만으로 현재 실행 기능을 단정하지 않는다.

이전 `e_trace.py`만 다시 읽는 것은 부족하다. 새 lineage는 기존 추적의 **legacy partial72와 실제 V47 correction 출처**를 수정했다. 그 수정 결과를 재사용하고 과거 추적의 잘못된 full/partial 분류로 되돌아가지 않는다.

### 1.3 원격 버전과 로컬 작업 보존

작성 시 main은 통합 커밋 `2fe1969b...`로 갱신되었고, 분석 원결과 브랜치는 `407de865...`다. 실제 실행 시 HEAD/원격/dirty 작업을 확인하고 관련 파일의 차이를 기록한다. 통합 커밋이라는 이유만으로 수학 소스·원자료가 동일하다고 가정하지 않는다.

새 코드·결과 경로와 브랜치 **제안 이름**:

```text
scripts/research/pallet_selection_score_diagnosis_v1/
_docs/experiments/pallet_selection_score_diagnosis_v1/
research/selection-score-diag-20261016-v1
```

존재하면 새 버전을 사용한다. 원본 소스, 예측, label, 사용자 수정, 운영 기본값을 덮어쓰지 않는다. `reset --hard`, `clean -fd`, force-push, 다른 작업 종료는 하지 않는다. 현재 main에서 새 브랜치를 만들 수 있지만 **수치 근거는 원407de865 결과와 실제 실행 source SHA로 바인딩**한다.

## 2. 데이터·성공·후보 범위 계약

### 2.1 자료 역할과 그룹

- 전체238장의 같은 ID·RGB SHA·K·등록 치수·물리 원점·native/physical Q·참조를 유지한다.
- PRIMARY192 안에서 기존 오라클 행을 읽어 `FAMILY_SELECTION_LOSS`10, `WITHIN_FAMILY_SELECTION_LOSS`22, `NO_SUCCESS_CANDIDATE`46, `FINAL_SUCCESS_CONTROL`114를 만든다. 이 이름들은 **이번 신규 분석용 별칭**이다.
- 그룹 이름은 기존 `earliest_observed_failure` 등 실제 필드와 매핑한 뒤 사용한다. 예상 필드가 없으면 새로 지어내지 않는다.
- 네 그룹이 서로 겹치지 않고192장을 정확히 구성하는지 ID로 검산한다.
- CAL46은 별도로 같은 분석을 한다. 원기록은 CAL 선택실패3장이 모두 family 단계이고 같은-family 선택실패는0이었다. 이 부족한 지원 범위를 숨기지 않는다.
- TEST49·추가143은 이미 반복 개발에 노출되었다. 분석 전에 규칙을 저장하더라도 새로운 독립 test가 되지 않는다.

### 2.2 평가 의미

성공은 **동일한 자세 후보 하나**의 중심 위치 오차≤5cm AND physical/C2 회전 오차≤5°다. 위치 최소 후보와 회전 최소 후보를 합친 가상 성공은 만들지 않는다. 직사각형의90° W/D 혼동을 C2 대칭으로 지우지 않는다.

여기서 참조는 기존 기하 재구성값이다. 이 참조에서 성공하는 후보를 `reference-successful candidate`라고 부르고, 독립 물리 참자세나 semantic owner 정답이라고 부르지 않는다. 모델 등록의 일치는 영상점의 물리적 의미가 맞다는 보증이 아니다.

### 2.3 후보 은행 범위

분석 대상으로 삼을 것은 원 e consumer가 실제 읽은 accepted winner, feasible samples, explicit starts와 원 기록/검증된 replay의 끝점이다. 원래 G10/G13은 구분한다.

- 과거 WD/P1 내부에서 생성했다가 버린 후보, V15/V17에서 뒤늦게 추가한 다른 후보, V33 또는 새 SINGLE 후보를 원래 은행에 합치지 않는다.
- PoseFix와 N0의 최종 자세도 비교군일 뿐 원래 e의 후보가 아니다.
- consumer-generated coverage complete와 전체 역사적 generated bank complete를 구분한다.
- 4개 admission 실패의 fallback은 출력으로는 존재하지만 가짜 generated 후보로 추가하지 않는다.
- 최초 입력·끝점·feasible·family winner·최종 출력의 역할과 동일성 연결을 보존한다. R/t가 같거나 근처라는 이유로 단계 기록을 합치지 않는다.

## 3. 먼저 원래 ‘선택’을 재현 — 자세를 다시 풀지 말 것

기존 selector를 **이미 계산된 후보 레코드**에 적용해 어떤 후보가 골라지는지 복원한다. optimizer나 initializer를 호출하지 않는다.

### 3.1 원래 선택의 실제 구조

[확인·소스] `select_frozen_wd_partial_refinement_v4_13e.select_same_family`는 G10이 선택한 W/D family를 고정한 뒤 **G13의 그 family 출력**을 읽는다. G10과 G13의 관측·목적함수는 다르다.

[확인·소스] `statewise_two_wd_family_v4_9.solve_two_wd_families`는 수락된 family의 objective와 index로 family를 고르고 정확한 동률은 ORIGINAL을 우선한다. G10 outline 내부는 유효·중복 제거 후보를 `(objective, initialization name)`으로 정렬한다. G13 및 위임 backend의 정확한 규칙도 직접 확인한다.

분해해야 할 순서는 다음과 같다.

1. 각 단계·family에서 후보가 어떤 입력·관측을 받고 생성되었는가?
2. 수락/유효성/중복 제거/branch 범위 중 실제 비교 대상에 들어가는 조건은 무엇인가?
3. family 내부 winner가 어떤 점수와 tie-break로 선택되는가?
4. G10 family selector가 무엇을 읽고 어느 family를 선택하는가?
5. e가 G13 같은-family의 어느 결과 또는 fallback을 내보내는가?

### 3.2 PASS 기준

같은 저장 후보·순서·source를 사용해 원 G10/G13 각 family winner, G10 family, e 최종 선택을 재현한다. native/physical 변환·출처·후보 index·accepted/fallback을 함께 비교한다. 부동소수 비교 허용치는 기존 원검산의 `atol=1e-9, rtol=0` 등 실제 고정 규칙을 인용하고 완화하지 않는다. 순서·ID·불리언은 정확히 비교한다.

성공 후보의 점수가 더 낮아 보이는데 원 프로그램은 다른 후보를 골랐다면, 바로 ‘선택기 버그’라 하지 않는다. **stage/family 불일치, 점수 단위, 수락 여부, fallback, 중복 제거, 원래 조회하지 않은 후보, tie-break, 잘못된 cost 연결**부터 확인한다. 같은 실제 eligible pool과 같은 comparator라는 조건이 확인된 뒤에만 구현 불일치로 판정한다.

이 단계가 실패한 영상은 선택 비용 귀속에서 `SELECTION_REPLAY_UNRESOLVED`로 남긴다. 나머지 독립적으로 성립한 분석은 계속하되 해당 영상을 분모에서 지우지 않는다.

## 4. 비용 항·제약·관측별 점수 장부

### 4.1 고정 자세 평가만 허용

저장된 objective_components를 우선 쓴다. 부족하면 **같은 R/t, 같은 family의 XYZ/Q, 같은 관측 packet, 같은 cfg**에서 원 cost/residual/constraint 함수를 평가한다. 이는 새로운 pose fit이 아니다.

공개 코드가 현재 main과 원 실행에서 달라졌으면 원 source SHA에 해당하는 버전을 사용한다. 분석용 추출 함수는 계산을 복제하기보다 원 평가 함수를 호출하거나, 원함수와 parity를 보장한 읽기 전용 계측을 사용한다. 전역 monkeypatch로 다른 실행을 오염시키지 않는다.

코스트 함수가 gradient까지 계산하면 비용을 기록한다. gradient 생략용 최적화는 scalar 값의 동등성을 먼저 입증한 경우에만 사용하고, 성능 좋게 보이도록 정규화·샘플 수를 바꾸지 않는다.

### 4.2 원 objective 복원 표

새 `OBJECTIVE_MAP.md`에 다음을 실제 코드와 연결한다.

| 구분 | 확인할 내용 |
|---|---|
| Full B/I 점 비용 | 활성 ID, 직접 교점/복귀점 출처, 원 px 잔차, whitening, 공분산, reliability, 강건화, 합/평균 규칙 |
| Partial B/I 선 비용 | 원 normal·offset·분산·reliability, 법선 방향 잔차, 완전한 점 손실 철회 여부 |
| 전역 outline 비용 | 양방향 거리, 모델/관측 표본 수, arc 정규화, 강건화, 실제 factor weight |
| 초기 자세 prior | cfg의 요청값이 아니라 실제 활성 weight와 실제 scalar. outline 경로는0인 코드가 있으므로 stage/위임 경로별로 확인 |
| O/T/U·유한 tangent 등 | soft cost인지 hard constraint인지, 실제 slack과 통과 여부. hard constraint를 합산 비용처럼 취급하지 않음 |
| 기타 | 실제 cost에 있는 추가 항만 기록. 모호성·rank·출처·개수 같은 진단 필드를 비용으로 더하지 않음 |

[확인·소스] G10 outline은 partial/BI objective에 `outline_factor_weight × global_outline_robust_mean_loss`를 더한다. `global_outline_factor_weight`와 loss를 이중 가중하지 않는다. G13 partial은 `removed_complete_BI_2D_likelihood_ids`에 포함한 점을 full2D loss에서 제외하고 normal 비용을 더한다. 탄젠트 구간은 range 제약이다.

실제 모든 backend가 이 패턴이라고 가정하지 않는다. 위임 경로의 prior·normalizer는 원 호출을 따라 확인한다. 비활성 항은 `inactive, reason`으로, 있어야 하는데 복원 못한 항은 `unavailable, reason`으로 구분한다.

### 4.3 후보별·관측별 기록

아래는 **이번 신규 출력의 논리 필드**다. 원파일에 같은 키가 있다고 가정하지 말고 mapping을 작성한다.

- 영상/녹화/분할, stage/family/branch, candidate ID와 원 시작점 이름·원 순서.
- 원 R/t/Q와 후보 역할, retained/accepted/selected/fallback/dedup 상태.
- 비용 총합, 실제 가중 후 구성항, raw·whitened·robust 잔차, 실제 count·sum·normalizer.
- B/I state와 관측 종류·출처·물리 owner 가설·활성/철회, 공유 source group.
- outline 양방향 세부값, 제약별 slack/validity, 비용·선택에서의 실제 역할.
- 참조 전용 별도 표의 T/R/동일 후보 success와 posterior 진단 그룹.

가중 후 구성항 합이 저장 총비용과 맞는지 전량 검산한다. 맞지 않으면 남은 항·상수·비선형 집계를 찾는다. ‘몇 %는 설명되지 않지만 대충 맞음’으로 통과시키지 않는다.

원 px·cm·정규화 cost를 같은 척도처럼 비교하지 않는다. 큰 절대항이 선택을 뒤집었다는 뜻은 아니므로 **후보 간 비용 차이**를 중심으로 분석한다.

## 5. 같은 family 선택 실패22장 — 어떤 관측이 잘못된 후보를 선호했나?

각 영상에서 실제 선택된 G13 후보를 s라 한다. 같은 G13·선택 family의 원 retained-feasible 후보 중 **동일 후보 성공 기준을 만족하는 전체 집합**을 보존한다.

대표 성공 후보 g는 이 성공 집합에서 **원래 comparator상 가장 앞서는 후보**로 선택한다. 이는 참조를 쓰는 사후 진단이다. 가장 작은 GT 오차의 한 후보만 뽑아 과장하지 말고, 성공 후보 전체의 비용 순위와 범위도 함께 기록한다.

### 5.1 원가중 비용의 쌍 차이

같은 목적함수에서 실제 가중 후 항을 J_k라 두고 다음을 계산한다.

```text
Delta_k = J_k(g) - J_k(s)
Delta_total = J(g) - J(s)
```

- `Delta_k > 0`: 그 항은 실제 선택 s를 선호한다.
- `Delta_k < 0`: 그 항은 참조상 성공 후보 g를 선호한다.
- 가산 구조가 확인된 경우 `sum(Delta_k) == Delta_total`인지 검산한다.
- 거의 동률이면 원 double 값과 tie-break를 그대로 보여준다. 작은 Delta_total을 분모로 비율을 폭발시키지 않는다.

이름은 **‘후보 선택 비용 기여’**다. 이를 모든 프레임에서의 실제 오류 원인이나 새로운 방법의 인과효과로 확대하지 않는다.

### 5.2 결정적 항에서 관측까지 내려가기

s를 선호하는 비용을 실제 점/선/outline 방향으로 분해한다. 대표 수치만이 아니라 전체 관측별 기여를 보존하고 큰 양·음 기여를 함께 보여준다.

확인할 경쟁 설명:

- fallback 점이 두 후보를 구분하는 근거를 과도하게 제공하는가?
- 실제 교점·선이 가정된 physical owner와 충돌하는가? 독립 owner 정답이 없으면 미검증으로 남긴다.
- full2D 철회가 실제 objective에 반영됐는가? 이미 수정한 추적 오류를 다시 모델 오류로 세지 않는다.
- 같은 근거가 점·선·outline으로 반복되었는가? source hash가 다르다고 통계적으로 독립인 것은 아니다.
- 공분산·정규화·관측 개수 때문에 비용 크기가 달라지는가? 합산 자체는 정상적인 likelihood일 수 있으므로 무조건 평균으로 바꾸지 않는다.
- silhouette가 비슷해 두 후보의 깊이/물리 축을 구분하지 못하는가? 비슷하다는 결론은 실제 동일 목적함수 값·투영으로 보여준다.
- 소수의 큰 회전 오류인지, 위치/회전이5cm/5° 경계를 조금 넘는 오류인지? 전체 선택손실을 W/D swap으로 부르지 않는다.

**최종 문장은 ‘어느 함수의 어느 cost가, 어떤 원관측을 통해, 어느 후보를 얼마만큼 선호했는가’까지 내려가야 한다.** 원관측 품질을 확인하지 못하면 ‘weight가 틀렸다’가 아니라 ‘이 항이 선택을 결정했으며 관측/불확실성 가정은 검증 필요’라고 쓴다.

## 6. family 선택 실패10장 — G10 결정과 G13 후보를 섞지 말 것

이10장은 최종 G13 retained-feasible 전체에는 성공 후보가 있지만, G10이 고정한 family 안에는 그 성공 후보가 없는 집단이다. **G10 당시에도 좋은 후보가 있었다는 뜻은 아니다.**

각 영상에 다음 두 표를 따로 만든다.

1. **G10 실제 의사결정 표:** 두 family의 당시 winner R/t·accepted·비용·구성항·선택 index·tie-break. 이 표의 두 후보만으로 G10이 그 결정을 내린 이유를 설명한다.
2. **G13 최종 도달 가능성 표:** 두 family의 실제 winner와 모든 retained-feasible 성공 후보·순위. G10이 선택한 family와 G13 자체 선택, 최종 e 출력을 구분한다.

그 뒤 다음을 분류한다.

- G10에서도 성공 후보가 있었지만 자기 점수/가설 선택에서 놓침.
- G10 은행에는 없고 G13 관측 패키지에서 성공 후보가 생김.
- G10이 선택한 family 안의 d 결과는 실패하고 반대 family 내부에는 성공 후보가 있음. 반대 family winner도 성공하는지는 별도 확인.
- G13 자체 family 재선택 역시 성공하지 못함. 원32장·10장 전부를 가설 스위치 하나로 해결할 수 있다고 가정하지 않음.
- 원기록·동등성 확인 부족으로 세부 판정 불가.

[확인·기존 결과] G13 자체 선택은 전체112/192, 기존 e는114/192였다. 따라서 ‘G13 점수로 그냥 통일’은 이미 자동 해결책이 아님을 유지한다.

필요한 **고정 후보의 동일 패키지 재평가**는 허용한다. 예: 같은 family의 선택 후보 s와 성공 후보 g를 각각 G10 패키지로 평가하고, 따로 G13 패키지로 평가한다. 양 family 간에는 해당 family의 원 XYZ/Q·관측 역할을 정확히 사용한다.

단, `G10(s)`와 `G13(g)`를 빼거나, G13의 성공 후보를 G10에 원래 있었던 후보로 꾸미지 않는다. 이러한 재평가는 `counterfactual_fixed_pose_evaluation`이며 새 optimizer·실행된 원후보·배포 성능이 아니다. 적법한 원패키지로 평가할 수 없으면 계산하지 말고 이유를 기록한다.

## 7. 성공114장과 CAL46을 대조해 사후 정당화를 방지

32장만 분석해서 ‘실패를 선호한 비용 항’을 찾는 것은 불충분하다. 같은 측정과 분석 규칙을 성공114장에도 적용한다.

- 성공 영상의 실제 선택 s와, 같은 단계/가설의 comparator상 가장 앞서는 참조 실패 후보 b를 비교한다.
- 해당 실패 후보가 없으면 없음으로 보고한다. 후보를 새로 만들지 않는다.
- 실패22장에서 문제처럼 보이는 항이 성공114장에서는 잘못된 후보를 배제하는 근거인지 확인한다.
- family 대조도 원 accepted family winner 기준으로 수행한다. 다른 family라고 항상 실패인 것은 아니다.
- 후보 수·영상 수·녹화 수를 구분한다. 후보 수천 개를 독립 영상 수천 장처럼 집계하지 않는다.
- 전체192/기존49/추가143/CAL46별로 수·비용차 분포·대응 관계를 보존한다. 평균±표준편차와 원행을 제공한다.

CAL46에서는 이미 학습/교정에 사용된 관측·상태 모델이라는 제한을 유지한다. CAL의 동일-family 선택실패0이라면 PRIMARY22장을 레이블로 새 선택기를 학습해 놓고 CAL 검증이라고 부르지 않는다.

실패32장이 모두 추가143에 속한다는 기존 분포도 ID로 확인한다. 분할 이름을 보고 다른 점수식을 적용하지 않는다.

## 8. 선택적 부가 진단 — 고정 후보에서만 점수 항의 순위 영향 확인

이 절은 새로운 refiner/selector를 개발하는 단계가 아니다. **원 cost의 가산 복원이 완료되고 원 선택 재현을 통과했을 때만** 수행한다. 복원이 안 되면 해당 없음과 이유를 쓰고7절·9절을 계속한다.

### 8.1 범위

[진단 설계·미검증] 실제 활성 additive loss family마다 한 번씩, **이미 계산한 비용표에서 그 family의 기여만 빼었을 때 후보 순위가 어떻게 달라지는지** 계산한다. 예: full B/I 전체, partial-normal 전체, outline 전체. 실제 활성 prior가 없는 패키지에는 prior 실험을 만들지 않는다.

- 원래 eligible/retained-feasible 후보, 대응, 분모, hard guards, tie-break는 고정.
- 관측 재생성·pose 재최적화·가중치 grid search·두 항 이상 조합·점별 유리한 삭제: 금지.
- full B/I를 다시 B/I/fallback/교점별 여러 신규 selector로 쪼개 탐색하지 않는다. 세부 출처는 비용 분석에만 사용한다.
- 기본 선택과 **실제 활성 additive loss family별 단일 제외**만 허용하며 목록은 참조별 성과표를 만들기 전에 기록한다. 적응적으로 후보 방법 수를 늘리지 않는다.
- 하드 영역/부분 tangent 제약을 제거하는 실험은 하지 않는다. 이미 feasible 후보에 점수만 다르게 붙이는 진단이다.
- family 단계(G10)와 같은-family 내부 단계(G13)를 한 번에 변경하지 않는다. G10만 변경해 기존 same-family d 출력을 읽는 진단과, 원 G10 family를 유지하고 G13 내부 순위만 바꾸는 진단을 분리한다.

### 8.2 보고

동일192장 전체에서 회복·손상·양쪽 실패·평균 오차 변화를 모두 보고한다. 32장만의 회복률로 결론 내리지 않는다. 전체 결과와114장 성공 보존, 원49/추가143을 같이 보여준다.

반드시 다음 제한을 붙인다.

> 원 목적함수로 생성한 고정 후보 은행 위의 사후 순위 영향 진단이다. 항을 제거한 목적함수로 후보를 새로 생성·최적화한 방법의 성능이 아니며, 새 선택기로 채택하지 않았다.

참조를 본 뒤 이 진단 중 가장 좋은 결과를 최종 방법으로 고르지 않는다. 결과가 좋으면 다음 승인용 변경안의 근거로만 사용한다. 좋지 않은 진단도 삭제하지 않는다.

## 9. 후보 부재46장 — 선택기 문제로 억지로 설명하지 않기

현재 consumer 은행에서 좋은 후보가 없는46장의 장부를 별도로 만든다. 4개 admission/no-generation과 나머지 후보 존재·공동성공 부재를 구분하고 수를 검산한다.

이미 있는 원 시작점→끝점→유효성→선택 로그로만 다음을 확인한다.

- solver 진입을 막은 실제 코드 조건/상태/관측 수는 무엇인가?
- 초기 N0/PoseFix/e 자세와 각 원 family의 좋은/나쁜 방향은 무엇인가?
- 기록된 어떤 후보도 위치·회전을 동시에 만족하지 못하는가? 위치만/회전만 좋은 후보가 다른 경우를 따로 보여준다.
- 시작점 자체가 모두 부정확한가, 후보 생성 후 계속 같은 잘못된 대응을 따르는가? 이는 로그에 보이는 범위로 한정한다.
- 실제 점·선 근거 부족, full/partial 정보, 과신한 fallback, 초기 조건부 S 판정, 원 mask/owner 불일치의 관찰이 있는가?
- 원 지지/제약이 서로 충돌하는가? 기록된 finite slack과 활성 항을 근거로 쓴다.
- 참조/물리 축/Q/K가 미확정이라는 기존 별도 기록이 있는가? 큰 오차 자체를 참조 오류의 증거로 바꾸지 않는다.

`wood_day_01:005249`, `wood_night_01:029428`, `wood_night_01:031344`, `wood_night_01:031426`, `plastic_day_01:011640`은 기존의 무성공 후보 사례다. 이번 선택32장의 대표 사례로 다시 사용하지 않는다. `plastic_day_01:010995`는 원래 e 성공이 후단에서 손상된 사례로만 다룬다.

각 행에는 ‘확인된 사실 / 경쟁 설명 / 아직 부족한 증거 / 다음에 필요한 최소 확인’을 쓴다. 로그만으로 관측과 후보 생성의 인과 기여를 분리하지 못하면 `OBSERVATION_VS_PROPOSAL_NOT_ISOLATED` 같은 **새 분석 상태**로 남긴다. ‘더 넓게 탐색하면 된다’, ‘모델을 학습하면 된다’로 자동 결론 내리지 않는다.

32와46의 고정 분류는 이번 기록 범위에 대한 것이다. 46개가 본질적으로 해결 불가능하다는 뜻은 아니고, 32개가 새 점수로 모두 회복된다는 뜻도 아니다.

## 10. 다른 잘되는 방법과 비교 — 기존 계약 내에서만

기존 N0(N3+SubPix+F), POSEFIX_RAW, E0의 동일192장 저장 결과를 재사용한다. 원래19경로 전체를 다시 추론하거나 통계를 전부 새로 만들 필요는 없다.

선택32·후보부재46·성공114별로 N0/PoseFix가 성공하는지 교차표를 만든다. 이를 통해 **e 후보에는 없지만 다른 모델의 출력은 성공인 사례**와 모든 기존 방법이 실패한 사례를 구분한다.

- 다른 방법의 성공 pose를 e 후보에 합쳐 오라클 상한을 높이지 않는다.
- 필요하면 다른 방법의 실제 고정 pose를 같은 e cost에 읽기 전용으로 대입할 수 있다. 반드시 `EXTERNAL_POSE_NOT_IN_E_BANK`로 구분하며 e의 원래 선택 후보로 취급하지 않는다.
- 외부 pose의 native/physical Q 변환을 정확히 확정할 수 없으면 대입하지 않는다.
- PoseFix RAW와 PoseFix+SubPix를 구분한다. 전체192용 PF+SubPix 단일 규칙 미확립 상태는 이번에 해결하려고 범위를 넓히지 않는다. 기존 공통95장 표는 참고로 연결하되 CAL과 TEST를 합쳐 독립 성능으로 쓰지 않는다.
- PoseFix의 Gaussian·좌표 손실·오류합성·과거 replay 분석은 이미 있다. 또 같은 문헌·실험을 재조사해서 작업량을 채우지 않는다.

## 11. 이미지 검토와 보고 형태

정량 장부는32/46/114를 전량 제공한다. 갤러리는 읽을 수 있게 제한하되 실제 후보·관측이 연결되어야 한다.

- 선택32장 각각에 실제 선택 후보와 원순위상 최선의 참조성공 후보를 같은 RGB·같은 crop·같은 축척으로 나란히 표시한다.
- 비용표는 같은 stage의 raw/weighted 항과 Delta를 보여준다. 가설10장은 G10 결정과 G13 결과를 두 줄로 구분한다.
- complete 점, partial 선/유한 지지, O/T/U 영역, model projection을 다른 표시로 구분한다. S의 투영은 관측점이 아니다.
- 46장에는 장별 정량 링크와 기존 이미지 링크를 연결한다. 초기/최종/후보 R/t가 없는 단계를 새로 만들어 그리지 않는다.
- 본문 대표는 고정 기준으로6장 이하: family10에서 기존 손상량 순 상위2, 내부22에서 상위2, 성공114에서 같은 항이 유효한 반례2. 같은 녹화가 반복되면 다음 녹화의 다음 사례를 택하고 이 규칙을 먼저 저장한다. 참조상 T/R의 의미가 다르므로 순서는 회전/위치를 섞은 임의 합이 아니라 기존에 정의된 최종 위치 오차 순으로 고정한다.
- 갤러리에 기존 사진을 쓰는 허가 범위를 지킨다. 새로운 RGB 공개는 승인 없이 하지 않는다.
- 숫자는 실제 저장된 후보·관측과 연결하고 이미지/배치용 좌표가 metric을 바꾸지 않게 한다.
- 새 서술은 내부 파일 ID 나열보다 ‘어떤 관측이 어떤 후보를 왜 고르게 했는가’를 먼저 설명한다.

PNG/PDF 형식 변환 자체를 별도 연구 실험으로 세지 않는다. 기존 실제 이미지 출력 코드가 있으면 재사용한다.

## 12. 산출물·검산·예산

### 12.1 필요한 산출물만 만들기

아래 이름은 **이번 신규 제안**이다. 원 field를 이런 이름이라고 가정하지 않는다.

```text
PLAN_AND_CONTRACT.md
SOURCE_BINDINGS.json
INPUT_AND_SELECTION_CHECK.json
OBJECTIVE_MAP.md
SELECTION_COMPONENT_LEDGER.csv.gz
SELECTION_PAIR_TABLE.csv
FAMILY10_CASES.csv
WITHIN22_CASES.csv
NO_CANDIDATE46_CASES.csv
SUCCESS114_CONTROLS.csv
SCORE_INFLUENCE_DIAGNOSTIC.csv      # 8절이 성립할 때만
BASELINE_CROSS_TABLE.csv
RESULTS_KO.md
NEXT_CHANGE_PROPOSAL_KO.md
VALIDATION.json
COST.json
REPRODUCE.md
gallery/INDEX_KO.md
```

대형 원저널을 새 폴더에 중복 복제하지 않는다. 입력·소스 해시와 필수 경로를 참조한다. 보고·갤러리에 보이는 숫자는 동일 장부에서 생성한다. 독립 검산은 핵심 수치·선택 재현·합계 등 의사결정에 필요한 범위에서 수행하고 검사 개수를 목표로 만들지 않는다.

### 12.2 필수 검산

1. 동일238 ID/분할·192 분모·114+10+22+46과 원참조 및 오라클 coverage.
2. 원 selector 재현과 step/family/candidate 역할·tie-break 일치.
3. 비용 총합과 실제 가중 구성항, 후보쌍 Delta 합 일치.
4. 철회된 full2D, 원예측 fallback, hard range·S 항의 역할 보존.
5. reference-success 정의가 동일 후보의 T/R 동시 충족인지.
6. 8절을 수행하면 whole192의 회복−손상과 성공수 차이, 모든 전이군의 연속오차 합 일치.
7. 기존 예측 R/t·가중치·정답·운영 파일·원봉인 해시 불변.
8. 실제 학습 update/NN forward/PnP/수치 optimizer/새 후보 생성 호출이0인지 실행 경로로 확인.

결측은NA와 이유를 남긴다. 판정 불가를 실패0이나 효과0으로 대체하지 않는다. 관측의 의미적 owner 정답까지 검증한 것처럼 검산 범위를 확대하지 않는다.

### 12.3 실행 예산

[추정·운영 상한] 이번 기본 상한은 **CPU 분석 단계 elapsed 7,200초, GPU 사용0, CPU worker 최대3개**다. 이전 승인 진단과 같은 CPU 상한을 쓰되 이번에는 신규 pose solve가0이므로 반복 대규모 최적화에 이 예산을 소모하지 않는다. 예측 완료시간이나 방법 채택 기준으로 이 숫자를 쓰지 않는다.

이전 후보 오라클 후처리는 약110초였지만 새 항별 비용 계산은 더 비쌀 수 있다. 전체 후보 수·관측량을 읽고 원 CAL 목록 처음6장의 참조 독립 파일럿으로 처리량·메모리를 확인한다. 전체 계산이 상한을 넘을 전망이면 동일 평가를 유지한 스트리밍·정확 캐시를 우선 적용하고, 그래도 초과하면 필요한 남은 예산을 보고한다. 영상·후보를 유리하게 줄이지 않는다.

입력 복원 wall time, cost 계산 elapsed, worker CPU 합, 갤러리, 검산 시간을 구분해 COST에 기록한다. full-RGB 지연시간이나 FPS로 환산하지 않는다. 과거 시간 기록을 새 실행 시간처럼 복사하지 않는다.

### 12.4 중단과 계속 기준

관련 분석만 중단할 조건:

- 원 source·관측·후보·참조 해시 불일치.
- 동일 candidate/family를 확정할 수 없음 또는 비용/원 선택 parity 실패.
- 본질적으로 새 최적화·학습·관측 생성 없이는 다음 계산이 안 됨.
- 원자료 권한·게재 허가·자원 상한 문제.

막힌 항목과 독립적인 나머지 분석은 계속한다. 데이터 접근 가능 여부를 반복 질문하기 전에 원 manifest·로컬 실행 폴더·Release를 확인한다. 다만 다른 자료로 임의 대체하지 않는다.

## 13. 최종 판단 — 같은 가설을 반복하지 않고 다음 수정 한 가지를 제안

`RESULTS_KO.md`는 다음 순서로 쓴다.

1. 이번 새 계산 범위와 원 선택 재현 상태.
2. family10 / 내부22에서 어떤 실제 결정이 실패했는지.
3. 비용 항→원관측까지 확인한 근거와 반례.
4. 성공114장에서는 같은 항이 무엇을 보호하는지.
5. 후보부재46장의 확인된 제약과 미해결 공백.
6. N0/PoseFix와의 동일 집합 비교가 알려주는 점.
7. 구현 오류 / 관측·가중치 모델 한계 / 이산 선택정책 / 정보 부족을 구분한 판정.
8. 다음 변경안 하나와, 다른 대안을 지금 하지 않는 근거.

후보 판정 예시(결과를 미리 채우지 말 것):

| 발견 | 다음에 합리적인 변경 | 이 분석만으로 하면 안 되는 주장 |
|---|---|---|
| 같은 eligible pool에서 더 낮은 원cost 후보를 실제 선택하지 않음 | 실제 comparator/기록 연결 구현 수정 검토 | stage를 섞은 숫자로 버그 확정 |
| 같은 family에서 잘못된 관측의 한 항이 반복적으로 좋은 후보를 누름 | 해당 관측 출처·불확실성/모델의 최소 수정안 | 높은 비용항이면 무조건 삭제 |
| family10에서 G13에만 좋은 후보가 생기는데 G10 family를 고정 | 단계 간 가설 선택 일관성의 최소 변경안 | G13 자체 선택이 이미112라는 반례를 무시 |
| 좋은·나쁜 후보의 기존 영상 근거가 구별되지 않음 | 추가 정보/관측 지원 또는 모호성 보존의 필요조건 | 새 ranking 학습으로 무조건 분리 가능 |
| 후보부재46에 성공 pose가 없음 | 관측·owner·초기 후보의 다음 최소 확인 | 선택기만으로46장 회복 |

새 알고리즘·학습을 제안할 때만 필요한 선행을 직접 확인한다. 기존 코드/문헌 감사에서 확인한 표준 방법과 실제 문제·가정·정보·감독의 차이를 적고, 미확인 효과는 `[추정·미검증]`으로 남긴다. 널리 알려진 오류처리를 새 발명처럼 이름 붙이지 않는다.

새 변경안은 최다1개를 우선 추천하되, 확인 근거가 부족하면 변경을 억지로 확정하지 않고 **판단을 바꾸는 최소 확인 한 가지**로 끝내도 된다. ‘32장 모두 고칠 수 있음’이나 ‘146/192 달성’을 약속하지 않는다.

새 선택기 학습이 필요하다는 판단이면 사용 감독·CAL 지원 범위·기존 시도와 차이·독립 확인 공백을 밝히고 승인 요청안만 쓴다. PRIMARY32를 학습시키거나 weights를 바꾸는 실행은 하지 않는다.

## 14. 게시와 사용자에게 줄 마지막 답

새 분석의 코드·집계·공개 가능한 갤러리·실제 재현 명령을 새 연구 브랜치에 commit·push한다. 기존 main/연구 브랜치를 강제 변경하지 않는다. 대형 후보·원관측·원RGB를 Git에 무차별로 넣지 말고 검증된 manifest와 허용된 작은 증거를 게시한다. 실제 원격 SHA를 다시 읽고 게시 상태를 확인한다.

마지막 답에는 아래를 포함한다.

- 실제 브랜치·commit SHA와 원결과407de865 연결.
- 32장 중 family10/내부22에서 확인한 원인과 미확정 수.
- 원관측의 어떤 항이 어떤 후보 선택을 뒤집는지 대표 실제 수치·그림.
- 그 항을 단순히 제거했을 때 잃는 기존 성공도 함께 보고(8절 수행 시).
- 후보부재46의 다음 확인 대상과 남은 source/semantic/참조 공백.
- 새 자세 추론·학습을 실행하지 않았다는 범위, 운영 기본값 미변경.
- 가장 우선하는 수정 한 가지, 그것이 새 학습을 요구하는지, 아직 효과 미검증임을 명시.

**이번 작업은 상태별 방법의 기여를 버리는 것이 아니라, 그 방법이 이미 만들어 둔 후보와 관측을 왜 잘못 평가하는지 찾는 작업이다. 기존에 성공한 결과를 잃지 않으면서 실제 다음 수정으로 이어지는 근거를 남겨라.**

---

## 근거 인덱스 — 이 지시문 작성 시 직접 확인한 범위

새 설계의 성능 근거가 아니라 기존 구현·결과 연결용이다.

- `407de865.../DIAG_ROOT/ORACLE_LIMITS.md` §7: 146→136→114, 실패32/46, CAL·TEST·추가143 분리, 후보 consumer 범위.
- `407de865.../DIAG_ROOT/OBSERVATION_SCHEMA_MAPPING.md`: 실제 full/typed/legacy 관측 활성, V47 correction, 후보/관측 출처와 source group.
- `407de865.../select_frozen_wd_partial_refinement_v4_13e.py`: G10 family 고정 후 G13 같은-family 숫자 출력.
- `407de865.../statewise_two_wd_family_v4_9.py`: accepted family comparator와 ORIGINAL exact tie, physical Q/XYZ.
- `407de865.../statewise_global_outline_solver_v4_10.py`: actual cost 구성, feasible/dedup, `(objective, name)` 선택, 전역 outline 가중·정규화와 prior0.
- `407de865.../statewise_partial_BI_u_solver_v4_13.py`: full2D 철회·partial normal soft cost·tangent range hard constraint.
- `407de865.../statewise_partial_BI_outline_solver_v4_13.py`: typed partial과 outline 결합 및 위임 backend.
- 통합 main `2fe1969...`에서 `ORACLE_LIMITS.md`는 같은 blob `a2c546ece6678a31004cb4b809a1ecfcee7be2b8`로 확인했다. 다른 실행 의존 파일의 동일성은 CLI가 source manifest 기준으로 확인해야 한다.
