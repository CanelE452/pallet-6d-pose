# CLI 실행 지시문 — 경쟁 보정기 전체 비교 확정 + 상태별 e의 신뢰도 결합 오류 교정

**작업:** 다른 보정 방법보다 최종 팔레트 6D pose 성공률이 좋은지 같은 자료·같은 지표로 먼저 확인하고, 차이를 만드는 구체적 오류가 확인될 때 **수정 한 가지**를 구현·검산·평가한다. **경로: 0 → 2 → 3.** 분석 계획만 작성하고 멈추지 말고 실행 가능한 범위에서 재현 → 비교 → 진단 → 한 가지 수정 → 검산 → 결과 게시까지 수행하라.

- 저장소: `CanelE452/pallet-6d-pose`.
- 작성 시 확인한 `origin/main`: `b843726ee2a78d7d29b79dafecdfc9d982b266fd`. 시작 전 `fetch` 후 실제 SHA와 새로운 관련 diff를 보고하고, 저장소가 갱신되어 있으면 최신의 정식 결과를 우선 확인하라.
- **최상위 성능 목표:** 기존 전체 상태별 경로 `e`를 중심으로, 동일 PRIMARY192에서 **N3+SubPix와 강한 PoseFix+SubPix를 최종 pose 성공률로 비교**한다. 다른 보정기보다 더 높은 정확도가 가능한지를 판단한다. 경량화/Jetson 측정/새 논문 집필은 이번 목적이 아니다.
- 논문에 바로 사용 가능한 **공정한 평가**와 기존 자료에서 수행하는 **탐색적 방법 개발**을 반드시 구별하라. 모든 실사는 이미 개발에 노출되었으며 독립 시험이 아니다.
- 새 결과를 좋게 만들 목적으로 영상/seed/평가 지표/치수/축/참조/결측 판정/비교군을 선택하거나 바꾸지 마라. 실패 결과도 게시한다.

## A. 현재 확정 상태를 재현하는 것이 첫 번째 합격 조건

용어: `N0` = `YOLO→N3→SubPix→공통 F`, `E0` 또는 `e` = 기존 전체 **S/B/I/O/T/U 상태별 관측 + 부분 선/영역/외곽 + 계층 자세 선택** 경로, `POSEFIX_RAW` = 저장된 synthetic-only PRIOR1 seed1 last6000의 첫 uncapped 출력과 기존 F. 이 셋은 서로 다른 **전체 시스템 경로**이므로 완전히 동일한 upstream-only 보정기 비교처럼 서술하지 마라.

기존 PRIMARY192(`TEST49 + REMAINDER143`, CAL46 별도)는 동일 physical/C2 참조, 동일 영상·동일 좌표/치수/K/Q/대칭 규약에서 이진 성공 `(translation <= 5 cm) AND (rotation <= 5 deg)`를 사용한다. **직사각형의 90° W/D 뒤집힘은 동등 회전으로 처리하지 않는다.** 자세 미산출은 분모에 실패로 남긴다.

실제 원장부를 검산해 확인할 기존 PRIMARY192 값:

| 경로 | 공동 성공 | 위치 평균(cm) | 회전 평균(°) |
|---|---:|---:|---:|
| R0, RAW YOLO+F | 66/192 | 14.536 | 14.282 |
| N0, N3+SubPix+F | 108/192 | 11.306 | 10.631 |
| E0, 기존 전체 e | 114/192 | 14.112 | 13.001 |
| POSEFIX_RAW(+F, SubPix 없음) | 80/192 | 13.836 | 11.640 |

검산 근거: `_docs/experiments/pallet_statewise_correspondence_diagnosis_v1/REPORT_KO.md`, `metrics/RESULTS.json`, `metrics/ROW_METRICS.csv`, `POSEFIX_TRANSFER_AUDIT.md`. README 등의 이름만 보고 별칭을 짐작하지 말고 실제 cache와 함수 경로를 추적하라. CAL46은 교정에 사용된 집단이므로 PRIMARY192와 합치지 않는다. 과거 `TEST49`라는 이름도 독립 test를 의미하지 않는다.

**반드시 저장할 교차 집계:** E0와 N0는 BOTH_SUCCESS=89, E_ONLY=25, N0_ONLY=19, BOTH_FAILURE=59로 알려져 있다. 총 성공 차이 +6장은 회복25−손상19다. 이 재현이 어긋나면 기존 결과와 새로운 결과의 계약을 복원할 때까지 뒤 단계로 진행하지 마라.

## B. 최우선: PoseFix+SubPix의 PRIMARY192 비교 공백을 채운다

**현재 공백:** `POSEFIX_SUBPIX_FULL_COHORT_NOT_ESTABLISHED`. PRIMARY192에서 비교한 80/192는 `POSEFIX_RAW`이지, 기존 논문 Table II의 강한 `PoseFix+SubPix`가 아니다. 따라서 PF_RAW와의 큰 차이로 "최강 경쟁군을 이겼다"고 쓰면 안 된다.

다음 실제 기존 자료를 먼저 읽어라.

```
_docs/experiments/pallet_statewise_correspondence_diagnosis_v1/POSEFIX_TRANSFER_AUDIT.md
_docs/experiments/pallet_statewise_correspondence_diagnosis_v1/metrics/POSEFIX_SUBPIX_COMMON95.json
_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/posefix_sub_comparison_v1/PROTOCOL.json
_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/posefix_sub_comparison_v1/compare.py
scripts/research/pallet_sensors_submission_v1/prior_model.py
scripts/research/pallet_sensors_submission_v1/posefix_contract_math.py
scripts/research/pallet_training_free_compare_20261007_v1/methods.py
scripts/research/pallet_dim_conditioned_p_v1/pose.py
challenge/evaluation_v2/pnp_selector.py
```

공통95장에 **두 고정 경로**가 이미 존재한다. ① `POSEFIX_SUBPIX_PF_ANCHOR`: raw PoseFix 보정량은 uncapped, SubPix의 추가 이동만 **PoseFix 출력점 기준** 원영상 대각선 1%로 제한. ② `POSEFIX_SUBPIX_YOLO_ANCHOR`: PoseFix+SubPix 총 이동을 **원 YOLO 점 기준** 원영상 대각선 1%로 제한. 둘 다 기존 창 5×5, `zeroZone=(-1,-1)`, `maxCount=40`, `epsilon=.001`, 최소 영상 크기15 조건을 기록했다. **한 설정을 택해두고 나머지를 숨기지 말고 두 경로를 각각 독립적인 사전 정의 민감도 비교로 유지하라.** 원래 코드와 문서가 우선이며, 위 요약과 실제 구현이 충돌하면 구현·원프로토콜을 보고하고 정합성을 복구하라.

진행 순서:

1. 원95장(=CAL46+TEST49)의 입력 ID, 원 PF `seed1_REAL_DEV.npz` tensor `points[selected_rows,0,1]`, checkpoint SHA, RGB·카메라·치수·Q, 후처리 선형 순서, 원래 `F`의 W/D 선택·SQPnP·RefineLM 호출을 확인한다. PF_RAW/두 SubPix 경로를 절대 혼용하지 말라.
2. **참조 자세를 읽기 전에** 현재 구현으로 기존 원95장의 출력·점·최종 pose를 복원해 성공 수 PF_RAW=66/95, PF-anchor=71/95, YOLO-anchor=70/95, N0=74/95, E0=86/95를 재현한다. 기존 봉인 pose와 허용차/동일성 검사를 기록하라. 별도 기술 차이는 모두 보고하라. 재현 실패 시 수치를 임의로 보정하지 말고 중단 원인 확인.
3. 허용된 REMAINDER143의 입력·저장 PF 출력이 존재하는지 확인하고, **두 원95장 정의를 그대로** 143장에 각각 적용해 TOTAL192 두 경로를 채운다. 기존 모델 forward/학습 없이 캐시를 우선 사용한다. 캐시가 없고 기존 고정 checkpoint의 필수 forward가 필요한 경우에는 실행 계약·seed·전처리·해시·비용을 먼저 명시하고 같은 정책으로만 실행한다. 미확정 자료 접근 정책이나 보호 세션은 우회하지 않는다.
4. 두 설정 중 **PRIMARY192 또는 COMMON95 결과를 보고 성능 좋은 것을 대표로 선택하지 마라.** 두 설정을 둘 다 기록하고, 비교 대상을 하나로 확정해야 한다면 이미 기록된 사전 연구 이유가 있어야 하며 없으면 `TWO_FIXED_COMPARATORS_NO_GLOBAL_WINNER`라고 기록한다.
5. 동일한192장 E0/N0/PF_RAW/PF-anchor/PF-YOLO-anchor의 `image_id`-별 예측, 5cm5° 성공, 위치·회전 평균±표본SD, 중앙값/P90/P95/max, >50/>100cm, 회전>45°, 불가용/오류/fallback, 세션·원49/추가143 구분을 표로 만든다. **E0 대비 PF 각각의 성공 건수 차이, paired 회복·손상과 차이를** 구한다. 가능한 한 세션 단위 군집 bootstrap 95% CI도 계산하고 세션 수와 사용된 검정 범위를 명시한다. 유의하지 않은 차이를 동등성으로 쓰지 않는다.
6. 정확도 계산기는 selector 프로세스와 분리한다. 참조·성공 후보 라벨·오라클·과거 평가 수치가 후처리 선택 또는 데이터 변형에 유입되지 않도록 I/O 경계 및 독립 재집계로 확인한다.
7. 245장 원고 Table II/III, 110장, 기존 95장, 새192장 **서로 다른 모집단의 수치를 하나의 성능표에 섞지 않는다.** 245장이나 다른 seed 결과를 192장의 누락값으로 채우지 않는다.

이 단계 결과물: `POSEFIX_192_PROTOCOL_LOCK.json`, `SOURCE_AND_INPUT_BINDINGS.json`, `POSEFIX_PARITY95.json`, `POSEFIX_192_PREDICTIONS.jsonl.gz`, `FAIR_192_COMPARISON.csv`, `PAIRED_TRANSITIONS.csv`, `FAIR_COMPARISON_REPORT_KO.md`, `VALIDATION.json`. 경로·형식은 신규 namespace 안에서 일관되게 확정하라. 실제 규칙을 복원할 수 없으면 **공백 사유와 필요한 증거를 명시하고 강한 비교를 했다는 주장 없이** 다음 가능한 감사만 진행하라.

## C. 우리 방법 E0에서 실제로 보정할 수 있는 문제를 구분한다

직전 실패한 접근을 자동으로 되풀이하지 않는다. 다음은 이미 수행된 결과다.

- RAW+최소B/I 82/192, N3+SubPix+최소B/I 108/192, 기존 전체 E0 114/192. N3/SubPix를 현재 삭제할 근거 없음.
- E0에서 소비한 후보 중 공동 성공 후보를 가진 영상 146/192, 실제 선택 성공114/192. 차이32장은 **정답을 아는 후보 오라클**이지 실현 가능한 추가 성능이 아니다. 최종 실패78장 중 가설 선택 문제10, 같은 가설 내부 선택22, 소비한 후보 집합에 성공 후보가 없는46이다.
- 동일 family 내부 선택 실패22/22에서 옛 whole-outline 항이 실패 후보를 선호했지만, 외곽 전체 제외는 9회복/11손상이었다.
- 새 콘 지원 whole-parent 결측 처리: 선택 6장 모두 2/6, 회복0. 새 RGB 가시 경계 기능: 기존2/6 대비 회복1·손상1=2/6. 후자는 **모델 보조 수동 입력**이고 자동 시스템이 아님.
- VISIBLE_SUPPORT의 `wood_night_01:030898`에서 성공 후보0 대 선택된 실패 후보2: 외곽선 비용 0.391463 vs 0.723937(성공 선호), **full_BI 비용 12.409336 vs 11.609887(실패 선호)**, 합산 12.800799 vs 12.333823(실패 선택). 그러나 이미 CAL total-error covariance/reliability override가 들어 있어 단순한 '불확실성 모듈 없음' 문제가 아님.
- `wood_day_01:005286`의 가시 경계 거리 자체가 실패를 선호하고, `plastic_day_01:020955`는 G10 family가 다른 가설을 고르며, 참조의 geometry-rank ambiguity도 남아 있음. **full_BI 하나의 수정이 모든 실패를 해결한다고 가정하지 마라.**

**Phase C1: 수정 전에 정보 계약을 감사한다.**

`E0`의 원래 full_BI/partial/S/O/T/U 출처와 실제 점별 likelihood를 사용하여, 후보의 native 물리 corner owner, 물리 Q/phase, 2D 관측 좌표/상태, measured vs fallback vs geometry-projected, 각 공분산 원본·변환·override 경로·좌표계 단위, reliability, robust 손실, point/line/outline에 같은 RGB 근거가 재사용됐는지 확인하라. 특히 CAL46에 대해 출처별 표본 수와 분포를 보고하고, 참조가 기하 재구성이라는 제한을 명시한다.

코드와 데이터의 후보 우선 감사 범위:

```
_docs/experiments/pallet_visible_boundary_pilot_v1/REPORT_KO.md
_docs/experiments/pallet_visible_boundary_pilot_v1/NEXT_DECISION_KO.md
_docs/experiments/pallet_visible_boundary_pilot_v1/SELECTED_COMPONENTS.csv
_docs/experiments/pallet_statewise_correspondence_diagnosis_v1/REPORT_KO.md
_docs/experiments/pallet_selection_score_diagnosis_v1/OBJECTIVE_MAP.md
_docs/experiments/pallet_selection_score_diagnosis_v1/SELECTION_RULE_MAP.md
scripts/research/pallet_visibility_curriculum_20261013_v1/joint_ot_pnp.py
scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_partial_BI_outline_solver_v4_13.py
scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_global_outline_solver_v4_10.py
```

**가장 먼저 고칠 대상이 단순 구현·전달 오류인지 판정한다.** 예: 실제 소스에서 잘못된 Q 적용·좌표계의 px/정규화 혼동·공분산 제곱/역행렬의 부호·full과 partial의 중복 계수·앞단의 철회된 부분 선을 정밀 2D point로 다시 승격, 모델 꼭짓점 재번호와 영상 semantic owner의 불일치 등. `참조상 오차가 크다`만으로 오류라고 확정하지 말고 원 정의·물리 대응·동일 입력에서 검산해야 한다. 부정확한 2D 참조의 투영값을 독립 물리 GT로 취급하지 마라. CAL의 4세션·불균형 상태 레이블로 복잡한 신규 모델을 fit하지 마라.

**Phase C2: 확인된 원인 하나만 수정·재평가한다.**

1. 먼저 `MECHANISM_AND_EDIT_PROPOSAL.md`에 **원코드 위치 → 실제 관측 불일치 → 왜 발생하는지 → 영향을 받을 장면 유형 → 하나의 변경 → 반례·부작용 → 성공/실패 판정 기준**을 명시한다. 저렴한 소스·고정후보 검증으로 가설이 틀리면 코드 변경 없이 `NO_SUPPORTED_CHANGE`로 종료한다.
2. 명확한 실제 결함이 확인되면 **새 연구 namespace의 opt-in 경로 한 가지**로만 수정한다. 기존 E0, N0, PF 경로를 덮어쓰지 않는다. 새 global weight sweep, 여섯 사례 전용 조건문, frame ID/재료/시간별 하드코딩, 정답 참조를 통한 gating, 취약 6장 경계 재작성은 금지한다.
3. 통계적 오차 교정이 필요하다면 CAL46의 실제 세션 단위 분리를 명확히 하고, 이미 사용된 CAL임을 표시하라. 단일 간단한 교정/진단이 정당화되지 않으면 임의 숫자를 정해 보정하지 마라. CAL에서 맞는다는 이유로 PRIMARY192 참조를 본 뒤 재튜닝하지 말라.
4. 기존 입력/출력·원 후보를 바꾸지 않는 **고정후보 선택 진단**인지, 새 관측/새 최적화를 포함한 **새 전체 방법**인지 구분하고, 후자라면 후보 생성/실행 시간을 비롯한 평가 범위가 달라진 사실을 명시한다. 둘을 섞어 우월성을 주장하지 않는다.
5. 후보선택 변경이 정당화되면 **프로토콜을 봉인한 뒤 PRIMARY192에 정확히 한 번 적용**, E0/N0/두 PF+SubPix와 paired 비교한다. 전체192와 원49/추가143, CAL46을 나눠서 **회복·손상·공동실패 tail**, position/rotation mean±SD/P90/P95/max, >50/100cm 및 >45° 회전, 가설 전환·실패·fallback을 전량 보고하라. 좋은 6장만 따로 골라 합격시키지 마라.
6. 수정이 성능을 낮추거나 tail을 더 악화시키면 그대로 보고하고 운영 채택을 보류하라. **목표는 성공률 최고지만, 평균/꼬리 오차 악화는 반드시 함께 공개한다.** 성공률의 1건 차이, 통계적 우월성, 실사용 정확도/독립 일반화는 서로 다른 주장이다.
7. 메커니즘이 전혀 지지되지 않거나 대상 입력이 자동으로 생성되지 않으면 억지로 성능 향상 방법을 만들지 마라. 가장 작은 추가 데이터/독립 신호가 무엇인지 근거와 함께 한 가지만 제안한다.

이번에는 기존 6장을 사용해 여러 임계값·가중치 조합을 고르거나, 전체192 수치가 높을 때까지 반복 조정하지 않는다. 192는 이미 반복 개발에 사용한 **탐색적 진단 집합**이며 새로운 독립 test가 아니다. 중요한 최종 주장에는 추후 실제로 격리된 독립 세션 평가가 필요하다는 한계를 보고서에 유지한다.

## D. 보고 형식과 완료 조건

`_docs/experiments/pallet_accuracy_closeout_20261017_v1/`와 `scripts/research/pallet_accuracy_closeout_20261017_v1/` 등 **신규 namespace**에 결과를 만든다. 과거 저장 경로·임시 파일·사용자 수정·모델 가중치·논문 PDF/LaTeX·운영 기본값을 변경하지 않는다. 코드 리뷰와 tests가 필요한 범위에서 가장 짧은 재사용 경로를 선택하라. 필요 없는 전체 재학습, 전체 238장 모델 추론, 새로운 마스크 생성은 실행하지 않는다.

최종 `REPORT_KO.md`는 이 순서를 따른다:

1. 사용한 SHA, 입력 계약, 실행된/미실행 단계 및 막힌 이유.
2. **원래 성능표와 동일192 PoseFix+SubPix 두 설정의 결과를 먼저** 보고. `E0−최강 경쟁군` 격차를 정수 건수 및 %p로 계산하되 둘 중 더 나은 설정을 대표로 사후 선발하지 않았음을 명시.
3. `E0(114)−N0(108)`의 paired 25회복·19손상과, 두 PF 경로 대비 각각의 회복·손상.
4. CAL point/partial/outline 공분산·owner·출처의 실제 확인 결과. `CONFIRMED_BUG / OBSERVATION_MODEL_MISMATCH / INSUFFICIENT_EVIDENCE` 중 어느 상태인지 구분.
5. 한 가지 수정이 있었다면 후보·최종 pose의 동일192 paired 전체 비교, 실제 이득과 손해, 새로운 tail 실패 및 성능 차이 CI. 변경이 없으면 `NO_SUPPORTED_CHANGE`와 증거.
6. 비교의 비대칭성: E0는 실사 CAL 사용, PF는 synthetic-only 사전학습, 서로 다른 전체 경로와 실사 개발 반복 사용. 결과가 좋더라도 독립 일반화 주장 금지.
7. **마지막 한 줄 의사결정**: `ADOPT_FOR_FURTHER_VALIDATION`, `REJECT`, `BLOCKED` 중 하나와 구체적인 근거. 결과를 좋게 보이기 위해 보고 집합 변경 금지.

최소 검사: 95장 parity, 192장 ID·분모 exact, 수식/점/좌표 단위 및 기존 속성 불변, no-PF-anchor mixing, no-G10/G13 objective cross-stage comparison, no GT/ref reads within selector, 동일 pose 하나에서 계산한 5cm5°, 수치 독립 재집계, 단계별 소요시간·환경 기록, 실패·누락·출처 SHA 전량. 오류가 나면 조용한 fallback 대신 실패를 그대로 보고한다.

**Git:** 사용자 작업 폴더를 stash/reset/clean/force-push 하지 마라. 별도 연구 브랜치에 변경을 commit·push하고, 원격 SHA를 다시 읽어 검증하라. 기본 브랜치 병합이 허용되고 충돌·보호 문제가 없으며 **실험 결과 보존만 수행**하는 범위라면 `main`에 보고서/연구 코드만 안전하게 반영하라. 운영 기본값·논문·원 예측·모델은 자동 교체하지 마라. 필요한 접근 권한이나 실질적 범위 확대가 발생하면 억지로 우회하지 말고 필요한 승인을 보고한다.

**마지막 사용자 보고:** 실제 원격 커밋 SHA, 실행 명령/소요시간, 두 PoseFix+SubPix의 PRIMARY192 실제 성공수, E0와의 격차, 수정한 단일 원인/코드 또는 수정 없음, N0/E0/PF와의 전체표, 바로 다음 필요한 것 한 가지. 새 방법이 다른 모든 보정법보다 낫다고 입증되지 않았다면 그렇게 말하지 않는다.
