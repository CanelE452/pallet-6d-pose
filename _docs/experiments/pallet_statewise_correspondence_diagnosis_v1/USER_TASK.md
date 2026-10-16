# CLI 실행 지시문 — 상태별 관측 기여 유지, 대응 전달·해 선택·PoseFix 차이 진단

작업: 합의한 네 우선순위의 감사·제한된 재계산·전체 검산·결과 보고를 실행한다.
경로: 0 → 2 → 3. 계획만 다시 작성하고 끝내지 않는다.
저장소: `CanelE452/pallet-6d-pose`
확인 기준 커밋: `96360c61418c435602875f5f34ca8ea4c694fbdc`
이 문서는 앞서 수행한 최소 B/I·성분 제거·guard A/B 실험을 다시 시키는 문서가 아니다.

## 0. 최상위 목적과 이번에 지켜야 할 기여

**목적은 사용자가 제안한 ‘가시 상태에 따라 영상 관측과 기하 제약을 다르게 처리하여 하나의 강체 자세를 추정한다’는 방법을 유지하면서, 현재 손상을 만드는 관측 전달·물리 대응·자세 해 선택의 문제를 분리하고 수정 근거를 확보하는 것이다.**

결과를 본 사용자가 판단할 것은 다음 네 가지다.

1. e가 실제 사용한 관측의 종류와 물리 대응을 다음 단계까지 전달하면 무엇이 달라지는가?
2. 같은 관측을 써도 자세 후보를 다시 선택하는 과정 때문에 손상이 발생하는가?
3. 큰 실패에 좋은 후보가 없었는가, 중간 검사에서 버렸는가, 최종 선택만 잘못했는가?
4. 수정이 기존 성공과 이미 실패한 큰 오차를 함께 개선하는가? 개선하지 못했다면 정확히 어느 원인이 남는가?

**기여를 비용 비교만으로 낮추거나, 상태별 방법 전체를 폐기하지 않는다.** 다만 새롭다는 주장과 실제 효과를 입증했다는 주장은 구분한다. ‘검색에서 못 찾음’을 ‘세계 최초’로 바꾸지 않는다.

### 0.1 상태별 처리의 불변 원칙

- B: 보이는 외곽 코너. 물체–배경 경계의 영상 근거를 사용한다.
- I: 보이는 내부 코너. 물체 내부 접합·선의 영상 근거를 사용한다.
- S: 초기 기하에 조건화된 자체 가림. 해당 코너를 가시 영상점처럼 국소 질의하거나 정확한 2D 대응점으로 넣지 않는다. 최종 자세로 얻은 모델 투영은 관측점과 분리한다.
- O: 외부 가림으로 판정된 경우. 근거와 사용 범위가 확인된 기존 영역 제약만 보존하며 영역 중심을 정확한 관측점으로 바꾸지 않는다.
- T: 화면 밖/잘림. 기존의 독립 근거·조건부 판정·보류를 구분한다. 초기 투영만으로 화면 밖임을 확정한 사실로 쓰지 않는다.
- U: 불확실하여 판정을 보류한 상태. 정밀 점을 새로 만들어 넣지 않는다.
- B/I는 기하학적 역할이며, 실제 보임/가림 판정과 같은 변수가 아니다.
- 한 방향의 선만 확인된 partial 관측은 full 2D point로 승격하지 않는다.
- ‘S에 영상 질의가 없다’는 뜻은 S 위치를 지정한 국소 질의·프롬프트·정확한 점 인자를 제외한다는 것이다. 전체 RGB를 읽는 CNN/분할기의 수용영역까지 S와 무관하다고 확대하지 않는다.

실제 데이터에서 비활성인 O/T 항은 ‘해당 없음: 유효 제약 또는 추가 효과 미확인’으로 보고한다. 상태 개념을 유지한다는 이유로 효과 없는 항을 강제로 추가하지도, B/I-only 절제가 실패했다는 이유로 전체 S/B/I/O/T/U 방법이 실패했다고 결론 내리지도 않는다.

### 0.2 승인된 실행과 제외 범위

**승인된 것:** 기존 소스/가중치/입력/원시 저널 읽기, 대응·관측 출처 감사, 동등한 추출기 구현, 아래 네 개의 고정 진단 경로 이내의 자세 재계산, 기존 후보 전량 복원/계측, 전체192장 재집계, PoseFix 원문과 저장된 comparator의 차이 분석, 공개 가능한 작은 코드·결과 게시.

**이번에는 하지 않는 것:** 새 모델 학습·미세조정·증류·CAL 재fit·신규 오류합성 학습, 새 seed, 새 촬영/Blender 렌더, 새 정답/축 승인/레이블 수정, 새로운 손실·guard·임계값 탐색, Jetson/FLOPs/TensorRT 작업, 논문/제목/표의 자동 변경, 운영 기본값 교체.

N3와 SubPix는 현재 유지한다. 이미 끝난 ‘둘을 없애도 되는가’ 실험을 처음부터 반복하지 않는다. 학습 optimizer update는 0회다. 자세를 구하는 수치 최적화는 아래 승인 범위에서 허용하며 학습 업데이트와 별도 집계한다.

이전 `CLI_MINIMAL_BI_ABLATION_20261016.md`의 O/T 추가 금지는 **그 B/I-only 절제 경로**에 대한 것이었다. 이번에는 기존 전체 e가 실제 사용한 O/T/U·부분 선 등을 복원하는 것이 허용된다. 새 영역·새 계수·새 관측을 발명하는 것은 허용하지 않는다.

## 1. 최신 근거부터 읽고 실제 호출 관계를 연결

다음은 위 커밋에서 확인한 기존 경로다. 최신 HEAD가 다르면 관련 diff를 먼저 읽고 영향과 소스 해시를 기록한다. 강제로 과거 HEAD로 reset하지 않는다.

### 1.1 결과·진단 문서

- `_docs/experiments/pallet_guarded_bi_final_v1/FINAL_RECOMMENDATION_KO.md`
- `_docs/experiments/pallet_guarded_bi_final_v1/METHOD_KO.md`
- `_docs/experiments/pallet_guarded_bi_final_v1/RESULTS_KO.md`
- `_docs/experiments/pallet_instruction_closeout_v1/RESULTS_KO.md`
- `_docs/experiments/pallet_posefix_gap_audit_v1/REPORT_KO.md`
- `_docs/experiments/pallet_posefix_gap_audit_v1/summary/ALL_ROWS.csv`
- `_docs/experiments/pallet_posefix_gap_audit_v1/axis_011640/CASE_REPORT_KO.md`
- `_docs/experiments/pallet_posefix_gap_audit_v1/pointed_cases/POINTED_CASES.json`
- `_docs/experiments/pallet_posefix_gap_audit_v1/pointed_cases/process/DATA.json`

### 1.2 관측 전달·원래 e 복원 코드

- `scripts/research/pallet_guarded_bi_final_v1/e_inputs.py`
- `scripts/research/pallet_guarded_bi_final_v1/guard.py`
- `scripts/research/pallet_posefix_gap_audit_v1/e_trace.py`

`e_inputs.py`는 e의 자세와 **N3/SubPix의 옛 이미지 좌표**를 물리 모델 꼭짓점 대응으로 재번호하여 조합한다. 이때 `physical_vertex_bijection`은 모델 등록을 검산하지만 영상에서의 의미 대응이 옳다는 것을 검증하지 않는다.

`e_trace.py`의 `load_e_trace`는 원래 e 이전 관측을 원저널에서 복원하고, 원래 e의 최종 자세와 현재 E0의 숫자를 연결할 수 있다. 새 E 후단의 관측을 원래 e의 중간 결과로 대신하지 않는다.

특히 확인할 기존 필드는 `e_likelihood_mode`, `explicit_complete_2D_likelihood_revoked`, `final_complete_measurements`, `archived_visible_observations`, `rays`, `regions`, `selected_candidate_audit`다. **필드가 존재한다는 것만으로 그 값이 실제 optimizer에서 사용됐다고 가정하지 않는다.** loss/constraint를 구성하는 원래 호출부까지 따라가고 실제 활성 항을 확인한다.

`e_trace.py`의 CLI가 외부 JSON에서 특정 outlier 부분집합을 읽을 수 있으므로, 이번에는 **명시적 전체238 ID 목록**을 넘긴다. 이전 71장 또는 121장 분석 파일을 전체 집합으로 대신하지 않는다.

### 1.3 원래 e의 저장 경로

아래 긴 경로를 이 문서 안에서만 `BOUNDARY_ROOT`라고 부른다. 기존 코드 변수명으로 가정하지 않는다.

```text
_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/
```

확인된 복원 출처:

- `BOUNDARY_ROOT/fixed_wd_archived_tangent_eval_v4_15/`의 봉인된 입력과 protocol
- `BOUNDARY_ROOT/cached_native_edge_candidates_v4_17/`의 원래 `SAME_FROZEN_E` 예측과 봉인
- `BOUNDARY_ROOT/evaluation_v3/`의 CAL/TEST recovery 저널
- `_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/remainder_after_test49_v1/fresh_observations_v4_14/`

파일명·입력 키·실제 solver 모듈은 위 복원 코드와 manifests에서 확인하여 연결한다. V33의 다른 후보를 원래 e의 후보로 합치지 않는다. final-only compact archive에는 원래 시작점·전체 후보가 없을 수 있다.

### 1.4 PoseFix 비교 확인

- `scripts/research/pallet_sensors_submission_v1/prior_model.py`
- `scripts/research/pallet_sensors_submission_v1/prior_data.py`
- `scripts/research/pallet_sensors_submission_v1/prior_inference.py`
- 실제 학습 호출·checkpoint·후처리는 위 파일과 결과 manifest에서 따라가서 확인한다.
- `_docs/experiments/pallet_oracle_mechanism_followup_v1/CANDIDATE_CARDS.md`
- `_docs/experiments/pallet_oracle_mechanism_followup_v1/RELATED_WORK_AND_TRANSFER.md`

현재 port는 RGB crop과 KP Gaussian map을 쓰는 `PoseFixPallet9`이며 heatmap/coordinate loss를 이미 포함한다. 따라서 ‘Gaussian을 넣자’, ‘좌표 loss를 더하자’를 확인 없이 새로운 아이디어로 다시 제안하지 않는다.

## 2. Phase 0 — 목적 연결과 경쟁 설명

실행 시작 문서는 아래 질문에 짧게 답하고 진행한다.

| 우선순위 | 왜 필요한가 | 결과가 바꾸는 다음 판단 |
|---|---|---|
| 1 관측 전달 감사 | 모델 등록과 영상 의미 대응, full point와 partial line을 구분 | 관측 재주입/형식 변환 수정이 우선인가 |
| 2 같은 관측의 국소 보정 대 재선택 | 관측 오류와 다른 자세 해 선택을 분리 | 국소 보정·대응 보존이 손상을 막는가 |
| 3 e 후보 상한·선택 진단 | 좋은 후보 부재와 pruning/선택 실패를 분리 | 관측 생성과 점수 중 어느 쪽을 수정할 것인가 |
| 4 전체 손상·회복과 PoseFix 비교 | 소수 성공·손상 사례를 전체로 확대하지 않기 | 원래 기여에서 어떤 효과가 지지되고 무엇이 미검증인가 |

경쟁 설명은 유지한다.

- 거의 같은 점에서도 다른 PnP 해가 선택되어 손상될 수 있다.
- 올바른 모델 vertex 재번호가 잘못된 영상 corner identity를 고치지는 않는다.
- 상태/마스크가 틀렸거나 fallback 점이 부정확해서 좋은 후보가 생성되지 않을 수 있다.
- 실제 관측선과 해당 물리 변의 대응이 틀렸을 수 있다.
- 같은 이미지 근거를 교점과 선으로 중복 사용하여 과신할 수 있다.
- physical 등록·참조축·카메라 정보에 불확실성이 남을 수 있다.
- 작은 일부 사례만 보면 좋지만 전체 큰 오차가 결과를 뒤집을 수 있다.

**코드에서 가능한 경로를 발견한 것, 사례에서 연관을 관찰한 것, 동일 조건 개입으로 원인을 분리한 것을 각각 구분한다.** 이전 보고서의 원인 해석도 이번 실험이 확인해야 할 가설이며 모든 실패의 확정 원인으로 복사하지 않는다.

## 3. 작업공간·자료·평가 계약 고정

### 3.1 원본 보존

현재 HEAD/원격/dirty 파일/다른 작업/실행 환경을 확인한다. 별도 연구 브랜치 또는 worktree를 사용하고, 기존 실험과 사용자의 미커밋 수정을 보존한다. `reset --hard`, `clean -fd`, 강제 push, 다른 프로세스 임의 종료, 과거 기록 덮어쓰기를 금지한다.

새 경로 제안은 아래와 같다. **이번에 만드는 이름**이며 존재하면 새 버전을 사용한다.

```text
scripts/research/pallet_statewise_correspondence_diagnosis_v1/
_docs/experiments/pallet_statewise_correspondence_diagnosis_v1/
```

원 RGB·대형 예측 저널은 로컬의 읽기 전용 원본을 우선 재사용한다. 없으면 공개 manifest에서 필요한 바이트를 확인하고 선택적으로 복원한다. 공개 archive 전체를 무조건 다시 받거나 소스·모델을 전체 복제하지 않는다.

### 3.2 분모·초기화

- PRIMARY: TEST49 + 추가143 = **192장** 고정.
- CAL: **46장** 별도. 따라서 기본 감사 장부는 **238장**이다.
- 기존 seed/고정 실행 하나를 유지한다. 새 seed와 유리한 실행 선택은 0회.
- severe와 새 정사각형 집합은 추가하지 않는다. 기존 영상 전체 등급을 새로 정의하거나 재분류하지 않는다.
- 영상 전체 등급에는 사용자가 육안으로 식별하기 어려운 그림자도 포함했고 거리는 주요 기준이 아니었다. 이 등급을 외부 가림의 면적 정답이나 코너별 S/O 상태 정답으로 사용하지 않는다.
- TEST49/추가143도 반복 개발 자료다. 새로운 독립 test라고 부르지 않는다.
- ID·RGB SHA·K·왜곡 처리·치수·모델 원점·코너 순서·CF와 physical 등록 Q·회전 대칭·평가 참조 해시를 연결한다.
- 성공은 현재 확정 계약의 **중심 위치 오차 ≤5cm AND physical/C2 회전 오차 ≤5°**다. 110장/245장 예전 표의 집계·부등호·seed 수를 섞지 않는다.
- C2 동등성과 잘못된 W↔D 대응을 구분한다. 직사각형 오류를 줄이기 위해 임의의 90° 동등성을 추가하지 않는다.

### 3.3 추론과 참조의 분리

추론·관측·후보 생성·후보 정렬에는 GT pose, 사람 가시성, 성공/실패, 오차, 점수 기반 사례 ID 분기를 넣지 않는다. e도 GT가 아니라 기존 모델의 예측이다.

oracle과 실패 분류는 모든 고정 후보/예측이 봉인된 뒤 별도 분석 프로세스에서만 수행한다. 이미 사례와 개발 결과를 본 상태이므로 봉인이 새로운 독립성을 만들지는 않는다.

원참조·Q·치수를 수정해야만 결과가 나오는 문제를 발견하면 영향을 받는 비교를 중단하고 별도 계약 오류로 보고한다. 답에 맞도록 registration이나 annotation을 다시 fit하지 않는다. 영향 없는 나머지 감사·재집계는 계속한다.

## 4. 우선순위 1 — 관측 전달과 물리 대응 감사

### 4.1 전체238장의 관측 계보 복원

같은 ID에 대해 아래를 따로 복원한다.

A. N3/SubPix의 원예측과 공통 F 초기 자세.
B. **원래 e 이전** 상태·마스크·RGB 관측·선·부분 제약·불확실성.
C. 원래 e의 후보/물리 대응 선택, 활성 점/선/영역 항, 최종 pose.
D. **e 이후** `E_FROZEN` 입력을 만드는 재번호, 새 상태·마스크·B/I 관측.
E. 현재 E1/EA/EB 재계산에서 실제 활성화된 관측과 선택 pose.

A/B/C/D/E를 섞지 않는다. 원래 e에 없었던 후단 mask/선으로 원래 e 과정을 설명하지 않는다.

### 4.2 신규 표준 관측 묶음

아래는 **새 감사 출력용 논리 필드**이며 기존 JSON에 있다고 가정하지 않는다. 실제 필드와의 mapping 표를 먼저 만든다.

- 원관측 식별자, 원파일·행·원 RGB 해시.
- 생성 시점(원 N3 / 원 e 이전 / e의 선택 / 새 e 후단).
- native corner/edge ID와 physical owner 후보.
- 확정 대응인지 복수 대응 후보인지, 근거가 모델 등록인지 영상 관측인지.
- B/I/S/O/T/U와 그 판정의 출처, 조건화한 초기 pose 가설.
- 관측 종류: complete image point / partial line / bounded region / inherited predictor point / model projection / unknown.
- 실제 좌표 또는 선/영역의 원 파라미터.
- 원 불확실성·가중치·robust loss·활성/철회 여부.
- 동일 영상선·교점에서 파생되어 정보를 공유하는 source group.
- 사용된 가설 family, native/physical 등록, 원점·단위.

**정밀도, 물리 대응, 관측 종류는 서로 다른 속성이다.** covariance가 작아도 대응이 틀릴 수 있고, 2D 좌표가 저장돼 있어도 실제 likelihood는 partial일 수 있다.

### 4.3 필수 검사

1. 원래 e에서 `explicit_complete_2D_likelihood_revoked`였던 항이 후단에서 full point로 다시 활성화됐는지 전량 확인.
2. `visible_observations` 목록에 남아 있다는 사실과 full 2D loss의 활성 여부를 구분.
3. `is_actual_image_observation` 같은 기록 flag만 믿지 않고 raw predictor fallback과 실제 line intersection을 생성 경로로 구분.
4. `physical_vertex_bijection`의 유일 모델 등록과 영상 의미 대응 검증을 분리. 전자가 PASS여도 후자는 미검증일 수 있음.
5. e가 바꾼 W/D/순열/물리 등록과 이전 관측의 owner가 일관되게 운반되는지 확인.
6. S와 model projection이 점·선 측정으로 재진입하지 않는지 확인. O/T/U 영역 중심도 정확한 점으로 승격하지 않음.
7. 교점과 그 교점을 만든 선을 동시에 사용하는 경우 근거 중복 여부 확인. 이 단계에서 새 covariance/loss를 만들지는 않음.
8. 잘못된 단위/전치/재번호 구현을 찾으면 입력·출력 동등성 오류와 모델의 의미 대응 오류를 구분.

산출물: 전체 observation lineage, full→partial→full 전이 수, 원예측 재주입 수, owner 변경 수, state 변경 수, 관측 중복 의존 관계, 최초로 의미가 바뀌는 함수·행.

특정 사례에서 발견한 우려를 전체 원인으로 확정하지 않는다. 의미 대응 정답이 없는 장면은 `SEMANTIC_OWNER_UNVERIFIED`라는 **새 보고 상태**로 남기며 임의 배정하지 않는다.

## 5. 우선순위 2 — 같은 관측에서 국소 자세 보정과 해 재선택 분리

### 5.1 먼저 원래 결과를 재현

변경하지 않은 상태로 가능한 모든 기존 baseline 숫자를 재집계한다. 필요한 고정 replay에서 pose/Q/가설/관측/수락/fallback이 기존 기록과 맞는지 확인한다. 기존 코드의 수치 허용치를 찾아 적용하고, 결과를 맞추기 위해 늘리지 않는다.

`e_trace` 추출의 E0 numeric bridge와 solver 재현은 다른 검사다. 같은 숫자를 복사해온 것이 원 solver를 재현한 것은 아니다.

### 5.2 네 경로만 사전 고정

다음은 **이번 진단용 새 별칭**이다. 기존 코드의 R0/N0 같은 이름을 재사용하지 않는다.

| 진단 | 관측 묶음 | 연속 최적화와 이산 선택 |
|---|---|---|
| REOBS_MULTI | 현재 E 후단이 재관측한 B/I와 기존 후단의 원 가중치·규칙 | 현재 후단의 고정 후보 생성·선택 재현 |
| REOBS_SINGLE | REOBS_MULTI와 완전히 같은 관측·목적함수·제약 | E0에서 한 번 시작, 현재 discrete 대응/W-D assignment 고정 |
| TYPED_MULTI | 원래 e가 사용한 typed 관측·원 owner 가설·활성/철회·원 가중치·기존 영역/부분 선 | 원래 e 계열의 고정 후보 생성·선택을 해당 관측 계약으로 재현 |
| TYPED_SINGLE | TYPED_MULTI와 완전히 같은 관측·목적함수·제약 | E0에서 한 번 시작, E0가 선택한 discrete assignment 고정 |

**SINGLE은 단일 시작점·고정 이산 대응을 뜻하며, 전역해 유일성이나 수치해가 반드시 가까운 basin에 머문다는 보장이 아니다.** 원래 존재하지 않는 임의 회전/이동 상한을 새로 넣지 않는다. 실제로 멀리 이동하거나 ambiguity가 남으면 그대로 기록한다.

### 5.3 비교 해석의 제한

TYPED_MULTI가 원래 e의 동일 관측·후보·선택을 정확히 재현하면 E0와 같은 결과가 나오는 것이 정상이다. 이 경우 **동등한 복원 대조군**으로 표기하며 별도의 정확도 개선 방법으로 세지 않는다. 잘못된 후단을 제거하여 E0로 복귀한 것을 e 자체보다 향상된 결과라고 쓰지 않는다.

- 같은 행 묶음의 SINGLE↔MULTI에서 관측 좌표·형태·가중치·활성 항·제약·허용 해를 바꾸지 않는다. 바뀌는 것은 시작점/후보 탐색·선택 정책이다. 탐색량 차이도 보고한다.
- REOBS↔TYPED는 상태·점 집합·부분 선·영역까지 다른 **관측 패키지 전체의 비교**다. 차이를 ‘재번호 하나의 순수 인과효과’로 부르지 않는다.
- 두 관측 묶음 사이의 raw 목적함수 값은 차원·가중치·측정 수가 달라 직접 크기를 비교하지 않는다.
- 실제 기존 함수로 동일 목적함수를 유지할 수 없으면 `NOT_ISOLATED`로 표시하고 차이 목록을 제시한다. full point만 받는 최소 솔버에 partial 좌표를 가짜 점으로 넣지 않는다.
- 기존 O/T/U·부분 선은 원래 e에서 실제 활성화됐던 범위만 TYPED에 보존한다. 없는 영역을 새로 만들지 않는다.
- 여러 가설 사이에 관측 owner를 옮길 때는 해당 가설에 대한 기존 명시적 변환/대응을 사용한다. 정답 기반 가까운 코너로 재할당하지 않는다.
- TYPED replay로 원래 e의 후보/점수를 재현하지 못한다면, 먼저 실패 원인을 해결한다. 그래도 못 하면 해당 경로를 미완료로 남기고 1·3·4의 가능한 분석은 완료한다.

### 5.4 원래 기여를 유지하는 최소 구현 수정

이 단계의 구현 변경은 observation packet 운반, 활성 full/partial 구분, 원 owner·Q 보존, 동일 관측의 한 시작점/후보 재선택 wrapper로 제한한다. 새 신경망·관측선 탐색·재학습·임계값 sweep을 추가하지 않는다.

모든238장에 같은 규칙을 적용한다. 알려진 손상 사례의 ID를 보고 다른 경로를 선택하지 않는다. E0를 초기값·fallback으로 사용하는 모든 경로는 `uses_e_initialization=True`에 해당하는 새 설명을 붙이고 e 없는 독립 방법이라고 부르지 않는다.

수락 실패 시 기존의 **자기 초기값 복귀 규칙**을 사용한다. 기존 코드에 충분한 규칙이 없으면 조용히 새 정책을 발명하지 말고 경로를 막고 보고한다. 수락 여부와 정확도 성공 여부는 따로 저장한다.

### 5.5 보존·연속성 검사

실제 장면에 새 GT를 만들지 않는 순수 코드 검사를 포함한다.

- 같은 관측을 같은 표현으로 직렬화·복원하면 각 활성 residual/constraint와 pose가 보존되는가?
- 모델 좌표와 영상 대응을 함께 유효한 순열로 바꾼 경우 물리 pose/재투영이 보존되는가?
- 의미 없는 채널 재번호와 physical correspondence 변경을 구분하는가?
- 한 무한 직선의 endpoint 순서만 반전했을 때 기존 수직거리 비용이 같다는 점을 확인하되, 이 검사가 올바른 끝점 대응을 판별한다고 주장하지 않음.
- 사용하지 않는 display 필드·내부 ID를 바꿔도 예측이 바뀌지 않는가?
- S projected readout 및 철회된 complete point의 표시 좌표를 바꿔도, 해당 값이 비활성 측정인 한 solver 측정 residual이 바뀌지 않는가? 초기 S를 결정한 coarse pose까지 바꾼 다른 검사는 여기 섞지 않음.

원래 e나 최소 B/I가 이 속성을 이미 위반한다고 가정하지 말고 검사 결과로 보고한다.

## 6. 우선순위 3 — 원래 e 후보의 상한과 선택 단계 진단

### 6.1 먼저 후보 기록이 충분한지 확인

원 저널에서 아래 단계를 구분하여 수집한다.

1. 초기 가설/시작점.
2. 실제 생성된 자세 후보와 각각의 관측/물리 대응 패키지.
3. 관측 지지·기하·부등식 검사에서 거부된 후보.
4. 수락 또는 유효 후보.
5. family 선택과 family 내부 refinement.
6. 최종 선택과 fallback.

`solution.candidates`, `feasible_pose_samples` 같은 이름을 발견해도 **전체 생성 후보가 들어 있다고 가정하지 않는다.** 실제 append/filter/sort/truncate 호출부를 확인한다. 최종 pose만 있는 compact 자료로 후보 상한을 계산하지 않는다.

필요한 원자료가 있으면 스트리밍 추출한다. 누락이면 **동일 원코드·같은 입력·같은 seed·같은 후보예산**으로 후보 로깅만 추가한 재실행을 한 번 허용한다. 소스 계측이 RNG 순서·후보 순서·tie-break·부동소수점 연산을 바꾸지 않았는지 확인한다. 변경했다면 원실행 재현이 아니라 별도 계측 실행으로 표기한다.

GT를 보고 새 W/D, 후보, 시작값을 추가하지 않는다. 재실행 비용이 예산을 넘으면 없는 bank를 꾸미지 않고 `CANDIDATE_COVERAGE_UNKNOWN`으로 남긴다.

### 6.2 분석용 oracle

모든 후보·가설·선택을 봉인한 다음 분석 프로세스에서 기존 참조와 비교한다.

- joint-success oracle: **같은 후보 하나**가 위치≤5cm AND 회전≤5°를 만족하는지 확인.
- translation-only minimum과 rotation-only minimum은 각기 별도 기술 통계다. 다른 후보의 좋은 위치/회전을 합쳐 가상 성공 pose를 만들지 않는다.
- `generated`, `after_support`, `feasible`, `selected` 단계마다 같은 분모의 도달 가능 성공 수를 계산.
- 순수 기존 bank와 SINGLE 등 새 진단 bank를 섞지 않는다.
- PoseFix/N3의 좋은 pose를 원래 e bank에 없었는데 e의 후보였던 것처럼 추가하지 않는다. 교차 모델 합집합 oracle이 필요하면 별도의 privileged 진단으로만 표시하고 이번 주상한과 분리한다.
- reference가 기하 재구성값이므로 oracle도 **기존 참조 기준의 후보 상한**이다. 물리적 참자세에 대한 절대 상한이 아니다.

### 6.3 전체 실패 원인표

다음 보고 분류는 **새 분석용 정의**다. 코드의 기존 상태명이라고 주장하지 않는다.

- `OBSERVATION_OR_PROPOSAL_LIMIT`: 확인된 전체 생성 bank에 공동 성공 후보 없음. 더 나은 관측/모델/탐색이면 절대 불가능하다는 뜻은 아님.
- `PRUNING_LOSS`: 생성 단계에는 공동 성공 후보가 있으나 지정된 검사 후 사라짐.
- `SELECTION_LOSS`: 선택 대상에 성공 후보가 남아 있었지만 기존 image-only 점수가 다른 후보를 선택.
- `REFINEMENT_LOSS`: 선택 중간 후보는 성공이나 같은 branch 내부 후속 최적화가 실패로 이동.
- `CANDIDATE_RECORD_INCOMPLETE`: bank 누락/저장 일부/재현 실패로 위 판정 불가.
- `REFERENCE_OR_REGISTRATION_UNRESOLVED`: physical axis·K·참조의 해석 공백이 판정에 영향을 줄 가능성. 다른 기계적 단계 판정과 별도 flag로 남김.

서로 중복될 수 있는 원인은 주원인 하나로 억지 할당하지 않는다. earliest observed failure stage와 추가 flags를 함께 제공한다.

특히 위치가 크게 틀리고 회전이 약90°인 경우, 화면 CF 상자 유사성과 physical W/D 오류를 분리한다. symmetry 최소화가 어떤 회전을 무시하는지 원 scorer 식으로 검산한다.

## 7. 우선순위 4 — 192장 전체 손상·회복·큰 실패와 PoseFix 비교

### 7.1 비교 경로

기존 RAW+F, N3-only+F, SubPix-only+F, N3+SubPix+F(N0), 최소 B/I(N1), e(E0), guard A/B 결과를 **수치 재집계**로 보존한다. 모두 다시 추론할 필요는 없다. 주요 원인은 N0/N1/E0와 위 네 새 진단 경로를 중심으로 설명한다.

PoseFix RAW는 같은192장에 이미 있는 고정 결과로 포함한다. PoseFix+SubPix는 기존 cap/모델선택 계약과 결과가 있는 CAL/TEST 구간에서만 먼저 연결한다. 전체192에 사용할 **단일 기존 규칙이 이미 확정**되어 있고 필요한 원예측이 완전하면, 그 규칙을 전체 누락 구간에 한 번 적용해 비교 범위를 완성하는 것은 허용한다. 단순 SubPix/F 재계산을 먼저 사용하며 새 학습은 하지 않는다.

두 cap 중 PRIMARY 결과가 좋은 것을 골라 쓰지 않는다. 정확한 규칙/입력 계약을 확인할 수 없으면 RAW192와 +SubPix 공통부분집합을 분리하고 `POSEFIX_SUBPIX_FULL_COHORT_NOT_ESTABLISHED`로 보고한다. 이 경우 ‘PoseFix를 이겼다’ 또는 예전 Table II의 최상 경쟁군보다 우수하다고 일반화하지 않는다.

### 7.2 결과표와 전이표

각 경로에 대해 PRIMARY192, TEST49, 추가143, CAL46을 별도로 보고한다.

- 공동 성공 수/N 및 비율.
- 위치·회전 평균±표본 SD와 최대값. 변동 단위 cm/°와 분산 cm²/°² 구분.
- 원래 기록에 있는 큰 오차 기준(예: 위치>50cm, >100cm)을 그대로 재사용하고 기준을 명시.
- pose unavailable, solver rejected, fallback, ambiguous-but-accepted를 각각 집계.
- 기존→변경의 양쪽 성공 / 회복 / 손상 / 양쪽 실패 4분할.
- 위치/회전의 개선 총합·악화 총합과 평균 차이에 대한 4분할별 기여.
- 성공으로 넘어간 영상뿐 아니라 양쪽 실패 집단에서 오차가 얼마나 커졌는지 별도 강조.
- 원49 보존 여부와 손상 ID를 기록. 원49가 반복 개발 집합이라는 제한도 유지.

새 경로에 pose unavailable이 생기면 이진 성공은 실패로 분모에 남긴다. 연속 오차는 관측 가능 N과 전체 N을 같이 쓰고 available-only 평균과 공통 available paired 평균을 분리한다. 실패값을0으로 넣거나 실패를 없애 평균을 낮추지 않는다. 임의의 유한 penalty를 새로 정해서 전체 평균으로 포장하지 않는다.

### 7.3 부트스트랩과 주장 범위

현재192의 실제 세션/원녹화 관계로 짝지은 군집 부트스트랩을 한다. 기존10,000회와 RNG seed20260917을 재사용하되, 과거 다른 ID 집합의 재표집 배열을 재사용하지 않는다. 반복 개발 자료의 기술적 불확실성 분석이며 독립 검증이 아니다.

우선 방향을 확인할 대비는 `TYPED_SINGLE−E0`, `TYPED_SINGLE−TYPED_MULTI`, `TYPED_SINGLE−REOBS_SINGLE`다. 모두 새 진단이며, 마지막은 패키지 전체의 대비다. 실행 불가/목적함수 불일치가 있으면 그렇게 표기하고 유리한 다른 대비를 주대비로 교체하지 않는다.

성공률·평균 위치·평균 회전의 짝 차이 CI를 보고한다. 이를 여러 비교에 대한 확증적 유의성으로 말하지 않는다. p값을 추가하려면 검정 가족과 양측 방향을 새 결과 전 고정하고, 기존 연구의 Holm 가족을 임의로 축소하지 않는다. 이번 기본 산출물에는 CI/전이 수만으로도 충분하다.

‘유의하지 않음’을 동등성으로, 한두 건 순증가를 안정적 우월성으로, mean만 개선을 모든 지표 개선으로 바꾸지 않는다.

### 7.4 사례 표시

아래는 **이미 보고된 진단 사례**이며 독립 평가 표본이 아니다.

- `wood_night_01:029428`: 작은 B 이동, 적은 대응점, 다른 pose branch.
- `plastic_day_01:010995`: 정확한 E0 뒤 재관측 후 큰 회전 손상.
- `plastic_day_01:011640`: CF 기하 유사성과 physical 축 오류의 구분.
- `wood_day_01:005249`: PoseFix가 성공하고 e가 크게 실패한 경우.
- `wood_night_01:031426`, `wood_night_01:031344`: 여러 초기 모델이 함께 크게 실패한 경우.

실제 입력 manifest에서 ID를 확인하며 없으면 이름을 비슷하게 만들어 대체하지 않는다. 실제 원 RGB와 저장 좌표만 표시한다. 이미지·가림·선 정답을 생성형으로 수정하지 않는다.

원래 e 이전 관측 → active full/partial/region 및 owner → 후보들 → 선택 → 후단 재관측 → 새 결과를 같은 장면에서 표시한다. 표에는 point의 픽셀 오차와 전체 pose T/R을 분리한다. 개선·악화·무변화·모든 방법 실패를 함께 포함하고 각 사례선택 이유를 표시한다.

## 8. PoseFix에서 가져올 원리 — 이번에는 진단과 다음 학습 필요성 판단에 사용

### 8.1 원문이 말하는 것과 현재 구현을 구분

`PoseFix: Model-Agnostic General Human Pose Refinement Network (2019/CVPR)`의 핵심 원리는 다음과 같다.

- 영상과 초기 pose를 함께 사용하여 초기 pose를 수정한다.
- 초기 입력을 Gaussian heatmap으로 표현하여 정확한 한 점뿐 아니라 주변 정보를 사용한다.
- 원문 오류합성에는 작은 오차(jitter), 같은 사람의 좌우 혼동(inversion), 다른 사람 부위와 혼동(swap), 큰 오류(miss), 이미 좋은 입력(good)이 포함된다.

이 원문의 사람 좌우 inversion과 팔레트의 W/D·native/physical 대응 오류가 같은 현상이라고 단정하지 않는다. swap은 다른 사람 instance의 관절 혼동이므로 배경 선을 잡은 것과 자동으로 동일시하지 않는다.

원문은 good 입력을 포함한다고 해서 ‘정상 입력 절대 보존’이나 ‘가려진 점의 보장된 복원’을 증명하지 않는다. Gaussian blob 또한 통계적으로 교정된 신뢰구간이 아니다.

현재 port의 `SourceRGB.item`은 확인 범위에서 frozen R0 코너와 박스를 입력으로 사용한다. 이 사실과 학습 전체에 별도 오류합성이 있는지는 다른 문제다. 실제 trainer·augmentation·checkpoint 계보를 확인하고, ‘원문 오류합성 그대로 재현’이라고 가정하지 않는다.

### 8.2 이번에 할 PoseFix 분석

1. 원문, 저장소의 현재 RAW comparator, 과거 replay/large-error 시도의 입력·오류합성·손실·가중치·crop·후처리 차이표를 만든다. 이미 있던 변화와 아직 시험하지 않은 변화를 구분한다.
2. CAL46의 **기존 참조**로 원 N3 및 e/후단의 잔차·상태·물리 대응/가설 변화를 요약한다. 증거가 없는 오류 종류는 ‘미확인’으로 둔다. 인간 오류합성 비율을 팔레트에 그대로 복사하지 않는다.
3. 전체 PRIMARY에서는 예측 봉인 뒤, PoseFix 성공/e 실패와 그 반대, 양쪽 성공/실패를 모두 분석한다. 사례 하나에서 PoseFix가 잘됐다고 원인을 학습 방식 하나로 확정하지 않는다.
4. 양쪽 좋은 입력에서 추가 보정이 얼마나 움직이고 망치는지, 보이는 같은 물리 코너의 작은 잔차와 다른 물리 대응의 오류를 분리한다. 상태별 오차 수와 분모를 함께 보고한다. 상태(S/B/I/O/T/U)는 관측 가능성의 분류이고, jitter/inversion/swap/miss는 예측 오류의 분류다. 둘을 일대일로 치환하지 않는다. 허용된 C2 대칭을 오류나 교정 성공으로 잘못 세지 않는다.
5. 기존 학습용 오류 범위가 실제 CAL 대응 오류를 포함하는지 소스·기존 통계로 확인한다. 없다면 `COVERAGE_NOT_VERIFIED`이지 무조건 새 학습을 시작할 이유는 아니다.

**초기 좌표/heatmap은 탐색 안내이며 정밀 관측으로 확정하지 않는다는 원리**를 우선순위1의 source/type 감사에 연결한다. PoseFix로 교정한 S/O/T 점을 그대로 원래 상태별 방법의 실제 B/I 영상 관측으로 승격하지 않는다.

### 8.3 허용하지 않는 확대

이번 실행에서는 새 PoseFix 재학습, RGB branch 추가 학습, 모델 증류, 새 sigma sweep, 새 오류합성 학습 데이터를 만들지 않는다. 과거 실패한 방법을 이름만 바꿔 반복하지 않는다.

학습이 필요하다고 판단되면 결과 끝에 **하나의 변경안**만 제시한다. 그 변경안에는 (a) 기존 오류지원 공백, (b) 기존 실패 시도와 바뀌는 조건, (c) CAL/학습·선택·평가 분리, (d) 보이는 B/I에만 학습 관측을 쓰고 S/O/T를 기존 정책으로 유지하는 계약, (e) 같은 예산의 표준 대안과 good 입력 손상 평가를 포함한다. 실제 fit은 별도 승인을 기다린다.

## 9. 자원·자동 실행·중단 조건

이미 승인했던 범위를 과도하게 키우지 않는다.

- 새 fit/학습 update/새 데이터/새 seed: **0**.
- 핵심 새 진단 경로: **4개 이내**, 각각 최대238장. 최대952개 최상위 새 pose-run 슬롯이다. 동일 결과 재사용은 따로 집계한다.
- 원래 e의 후보 로깅 replay가 필요하면 **고정 프로토콜 1회×238장 이내**에서 수행하고, TYPED_MULTI와 동일 실행이면 중복 계산하지 않는다. 내부 후보 solve 수는 별도 집계한다.
- 원문/소스 읽기, 기존 후보 채점, bootstrap은 위 pose-run 수와 분리하되 실제 비용을 모두 기록한다.
- CAL에서 참조/성공을 보지 않고 세션 균형·ID hash로 최대6장을 골라 동등성/시간 파일럿을 한다. 뒤에 같은 행을 다시 계산할 때는 동일 해시의 결과를 재사용할 수 있다.
- **[추정·운영 상한] GPU 관련 프로세스 누적 경과시간60분, CPU 솔버·분석 단계 누적 경과시간120분.** 이전 지시문의 상한을 유지한다. 기존 E 후단 약2.25초/이미지 측정은 작은 CAL 패널의 참고값일 뿐 full e replay 시간 보장이 아니다. 파일럿으로 전량 비용을 다시 예측한다.
- GPU 실행은 동시에1개, 무거운 CPU worker는 최대3개. 다른 사용자의 작업을 중단하지 않는다. disk·RAM·VRAM을 먼저 확인한다.
- 데이터 복원·코드 작성·검산을 포함한 총 wall time 및 worker core-time을 별도로 기록한다.

예산 초과가 예상되면 불리한 영상을 빼거나 후보 budget을 줄이지 말고 필요한 시간·남은 경로·추가 예산을 보고한다. 예산 내에서 가능한 읽기 전용1·3·4는 계속한다. timeout을 numerical failure와 구분하고 전체 coverage에 남긴다.

중단해야 하는 것은 GT 유입, 봉인/좌표/참조 계약 불일치, 무허가 원자료 공개, 원본 훼손 위험, 자원 초과다. 패키징·경로·입출력·타입 등 승인 범위의 구현 오류는 스스로 고치고 검산한다. 오류 수정 전 시도도 남긴다. 수치가 나쁘다는 것은 중단 사유가 아니라 보고할 결과다.

## 10. 작업 완료·채택 판단을 분리

### 작업 완료

네 우선순위의 결과·공백·재현 증거가 갖춰졌는지로 판단한다. 새 방법이 이겨야만 실행이 완료되는 것은 아니다. 실행 불가인 경로가 있으면 왜인지·어디까지 완료했는지 정확히 쓴다.

### 알고리즘 판단

- 대응 의미와 full/partial 관측 보존이 실제 손상을 줄이면: **상태별 방법의 구현 개선 근거**로 보고한다. 보존 규칙 자체가 최초라는 주장은 선행과 별도로 확인한다.
- 같은 관측의 SINGLE이 손상을 줄이지만 큰 실패를 못 고치면: 국소 정밀화와 가설 전환을 분리할 근거다. global hypotheses를 모두 폐기하지 않는다.
- 좋은 후보가 이미 있는데 선택만 실패하면: 다음 과제는 그 가설을 실제 영상 근거로 구별하는 scoring이다. GT oracle을 운영 selector로 넣지 않는다.
- 좋은 후보가 없으면: 관측/대응 지원 또는 후보 생성의 문제로 남긴다. scoring 튜닝만 반복하지 않는다.
- 좋아지는 지표와 악화하는 지표가 섞이면: 조건부 유지/보류. 단일 최고 성공 수만 고르지 않는다.
- 어떤 진단도 원인을 좁히지 못하면: 남는 원인·부족한 증거를 보고한다. ‘기여를 포기한다’나 ‘논문 불가’로 결론을 대체하지 않는다.

원49 유지, 전체 평균·큰 실패, 전체192 성공률을 함께 표시한다. 허용 손실폭/비열등성 margin을 새로 만들어 동등성을 선언하지 않는다. 결과를 보고 방법별로 좋은 ID를 골라 routing하지 않는다. 운영 기본값과 논문 수정은 별도다.

## 11. 산출물과 검산

다음은 **이번에 생성할 새 산출물 이름** 제안이다. 기존 파일과 혼동하지 않는다.

1. `PLAN_AND_CONTRACT.md`: 기여·범위·네 진단·모집단·지표·자원 고정.
2. `SOURCE_AND_INPUT_MANIFEST.json`: 실제 HEAD/dirty dependency/환경/원저널·모델 바인딩.
3. `OBSERVATION_LINEAGE.jsonl.gz`, `OBSERVATION_TYPE_AND_OWNER_AUDIT.csv`: 전체238 관측 계보와 최초 변화 위치.
4. `OBSERVATION_SCHEMA_MAPPING.md`: 새 감사 필드 ↔ 실제 원코드/필드 대응.
5. `REPLAY_PROTOCOL.json`, `REPLAY_PARITY.json`, `PREDICTIONS.jsonl.gz`: 네 진단의 실제 활성 항·시작점·후보·선택·fallback.
6. `CANDIDATE_MANIFEST.json`, `CANDIDATE_STAGE_DIAGNOSTICS.csv`, `ORACLE_LIMITS.md`: bank 완전성·생성/검사/선택 상한과 공백.
7. `ROW_METRICS.csv`, `PAIR_TRANSITIONS.csv`, `COHORT_SUMMARY.csv`: 전체 분모·원49·추가143·CAL.
8. `POSEFIX_TRANSFER_AUDIT.md`: 원문/현재 port/과거 실패·비교모델 차이, 기존 좋은 입력 보존과 오류지원 분석, 새 학습 필요 여부.
9. `CONTRIBUTION_EVIDENCE.md`: 핵심 아이디어, 선행에서 이미 알려진 부분, 이번 검증이 지지/부분지지/미지지한 것, 아직 확인할 신규성 범위.
10. `REPORT_KO.md`, `REPRODUCE.md`, 작은 실제 gallery, 테스트/실행량/한계 기록.

독립 검산은 같은 score helper를 그대로 다시 호출하는 것만으로 끝내지 않는다. 저장 R,t·물리 등록·고정 참조에서 중심 거리와 회전식을 별도 짧은 구현으로 계산하고 다음을 검사한다.

- 전체 ID와 각 분모, 참조 해시, 네 진단의 입력 차이 범위.
- 공동 성공이 단일 R,t 쌍에서 계산됐는지.
- 회복−손상 = 성공 수 차이, 개선+악화 합/N = 평균 차이.
- no-op/기존 재현에서 states/owner/fallback/branch가 보존되는지.
- oracle 결과가 운영 예측/후보 순서/선택 파일에 역류하지 않았는지.
- 표·갤러리·보고서·CLI 단일 이미지 결과가 같은 저장 좌표에서 나왔는지.
- `accepted`, ambiguity, 물리 정확도 성공이 서로 구분되는지.
- 성능에 영향 없는 숫자 차이와 다른 discrete choice를 구분하는지.

새 API는 실제로 존재하는 입력/출력 계약과 테스트를 제공한다. 파일·프로그램 이름만 만든 빈 scaffold나 요약 Markdown만으로 실행 완료를 선언하지 않는다.

## 12. 게시와 최종 응답

기존 원고·운영 코드·가중치·실험을 덮어쓰지 않는다. 새 코드/보고서/공개 가능한 작은 수치/동일 공개 RGB의 허용 범위 그림은 별도 연구 브랜치에 commit·push한다. 개인 경로·비공개 이미지·새 raw archive 공개는 기존 허용 manifest를 먼저 확인한다. 실패한 push를 완료라고 말하지 않는다. 대형 raw는 로컬 증거 바인딩을 우선 보존하고 별도 승인 없는 대형 공개를 하지 않는다.

최종 보고 첫 부분은 다음 순서로 쓴다.

1. **기여의 어느 부분을 유지했고, 실제 구현에서 어느 연결이 잘못됐거나 미검증이었나?**
2. 네 우선순위 각각의 완료 범위, 핵심 관측과 반사실 결과.
3. 전체192장·원49·추가143·CAL46의 숫자와 손상/회복/큰 오차.
4. PoseFix에서 가져올 원리와 현재 port가 이미 구현한 것, 새 학습이 정말 필요한지.
5. 현재 채택 가능한 수정/보류할 수정/다음 한 의사결정.
6. 소스 SHA, 보고서·재현·갤러리 경로, 브랜치/push 상태, 실제 시간/검산/남은 공백.

‘감사 PASS’를 ‘정확도 향상 PASS’로, ‘관측 계보 완전’을 ‘물리 GT 정확’으로, ‘다른 논문에서 같은 약자를 못 찾음’을 ‘최초성 증명’으로 바꾸지 않는다.

## 13. 선행 원리와 근거 — 실행 전 읽기

### PoseFix: Model-Agnostic General Human Pose Refinement Network (2019/CVPR)

원문의 §4 오류합성과 §5.1–5.2 영상+pose/완만한 입력 표현을 확인한다. 본 지시문은 원문을 그대로 학습하지 않는다. 오류 종류별 진단, good 입력 손상 분석, 초기 예측과 영상 관측의 구분을 가져온다. 아래 arXiv는 **CVPR 출판 논문의 저자 공개본**이지 arXiv-only 연구라는 뜻이 아니다.

```text
https://openaccess.thecvf.com/content_CVPR_2019/html/Moon_PoseFix_Model-Agnostic_General_Human_Pose_Refinement_Network_CVPR_2019_paper.html
https://arxiv.org/pdf/1812.03595
https://github.com/mks0601/PoseFix_RELEASE
```

### EPOS: Estimating 6D Pose of Objects With Symmetries (2020/CVPR)

복수의 2D–3D 대응을 유지하고 강건하게 pose를 구하는 선행이다. 현재 방법과 같다는 주장이 아니라, ‘대응 모호성을 유지’ 자체를 새 원리로 주장할 수 없다는 인접 근거다. EPOS 전체를 새로 학습하지 않는다.

```text
https://cmp.felk.cvut.cz/epos/
https://ieeexplore.ieee.org/document/9157391/
```

### Uncertainty-Aware Camera Pose Estimation From Points and Lines (2021/CVPR)

점과 선의 불확실성을 함께 처리한 선행이다. 작은 covariance와 올바른 의미 대응은 다른 문제다. 현재 complete point/partial line을 일관되게 전달하는 검사의 근거로 사용하며 새 solver 전체를 들여오지 않는다.

```text
https://alexandervakhitov.github.io/uncertain-pnp/
https://www.iri.upc.edu/publications/show/2495
```

### Object Pose Estimation With Statistical Guarantees: Conformal Keypoint Detection and Geometric Uncertainty Propagation (2023/CVPR)

키포인트의 영역 제약을 pose 집합으로 전파하는 선행이다. O/T에 점 대신 영역을 쓰는 아이디어만으로 최초성을 단정하지 않도록 확인한다. 기존 CAL 최대오차 타원이나 Gaussian heatmap을 이 논문의 통계 보장과 동일시하지 않는다. 이번에는 conformal fit이나 보장 주장을 추가하지 않는다.

```text
https://research.nvidia.com/labs/aspire/publication/yang.pavone.cvpr2023/
```

**남길 기여 서술의 방향:** 가시 상태에 따라 영상에서 측정한 점·부분 선과 비관측 영역·모델 투영의 역할을 구분하고, 그 물리 대응을 하나의 강체 자세 계산까지 유지하는 팔레트 보정 방법을 제시하며 실제 추가 효과와 실패 경계를 검증한다. 세부 구현의 신규성은 직접적인 선행 대조로 확인하고, 효과는 위 네 진단에서 확인한 범위만 주장한다.
