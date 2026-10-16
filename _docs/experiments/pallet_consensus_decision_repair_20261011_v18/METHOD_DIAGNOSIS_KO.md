# 보정기와 자세 선택의 현재 진단

사용자가 확인한 LOO의 self-occlusion 제외는 구현되어 있다. 그러나 현재 방법은 정확도 목표를 해결하지 못했다. 아래 수치는 ROOT가 완료한 고정 245프레임 평가·저장 진단에서 전사했으며, 이 문서를 작성하면서 원행 해독·새 계산·모델·학습·PnP·최적화를 수행하지 않았다.

## 해결된 코드 문제와 남은 정확도 문제

기존 2,164개 감독 자료의 수치적 target 문제는 수정된 READY 자료·SHA 결속으로 처리했다. 현재 고정 학습 경로는 [retrain.py:100–129](../../../scripts/research/pallet_kp_supervision_repair_20261010_v1/retrain.py#L100)의 수정 자료를 사용한다. 이를 현재 실사 대응점이나 자세가 정확하다는 증명으로 해석하지 않는다. 이번 진단에서 해당 감독 자료를 다시 수정하거나 재학습하지 않았다.

V18은 별개의 결정 규칙 문제를 고쳤다. S15가 같은 inlier ID를 가진 비동등 후보를 비용 차이와 무관하게 거절하던 추가 규칙을 제거하고, 같은 전체 관측에 대한 count → truncated SSE 순서를 복원했다. 원 best와 모든 후보·기하 검사는 그대로 보존했다. 161행이 복원됐고, 실제 비동등 numerical tie 1행과 부족 관측·기하 4행은 fallback으로 남았다. [S15 solver.py:309–325](../../../scripts/research/pallet_factor_subset_consensus_20261011_v15/solver.py#L309), [V18 solver.py:94–126](../../../scripts/research/pallet_consensus_decision_repair_20261011_v18/solver.py#L94).

이 수정은 기능적으로 확인됐지만 정확도를 해결하지 않았다. NEW는 수치적 자세 산출 상태이며 물리적 정확도·ownership·전역 유일성 인증이 아니다.

## LOO와 각 관측의 의미

- LOO는 같은 native N3 bank에서 `H ∪ {k}`를 제외해 자세를 구한 뒤, 그 자세에 대한 native 점과 실제 decoded q의 잔차를 비교한다. q의 잔차가 고정 8px 안이고 native보다 개선될 때 실제 q를 채택한다. [selection.py:132–177](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/selection.py#L132). 제외된 H와 center는 보존된다. 이 소스 감사에서 H/k fit 누출이나 누락된 native-vs-q 비교 오류는 발견하지 못했다.
- heldout 자세는 남은 잡음 있는 native 관측에 의존한다. 초기 N3의 H와 Base feature/role도 조건부 입력이다. 따라서 LOO 잔차 개선은 실제 물리 corner 대응이나 독립 자세 정확도 개선을 보장하지 않는다.
- 감독은 지정된 physical source wire의 edge/axis·소유·깊이·검색 범위를 검사한다. decoder의 q는 두 incident edge의 TLS 교점이다. 임의 ray 교점을 곧바로 3D corner로 쓰는 단순 경로는 아니다. [actual_wire_proposal.py:98–147](../../../scripts/research/pallet_kp_supervision_gate_20261010_v1/actual_wire_proposal.py#L98), [observations.py:148–173](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py#L148). 그러나 실사에서 예측 edge ownership이 맞다는 독립 인증은 없다.
- finite support, 실제 fit rank/weak 검사와 normal-incidence 검사는 기하적 승인 조건이다. 물리 대응점의 진실이나 보정 정확도의 인증으로 바뀌지 않는다. [V14 pipeline.py:41–109](../../../scripts/research/pallet_stable_incidence_guard_20261011_v14/pipeline.py#L41).
- 승인 NEW에서는 H가 fit·score에 들어가지 않고 저장된 최종 projection으로 한 번 표시된다. 재투영 좌표를 다시 fit하지 않는다. fallback은 원 all8 N3 출력을 보존한다. 잘못된 H만으로 전체 프레임 실패를 선언하거나, 맞는 H만으로 성공을 선언하지 않는다.

## 동일 q의 일반 PnP와 복원 경로

ROOT가 완료한 전체 245행 저장 평가의 종합 평균이다. 기준은 기존 GEOMETRIC_PROXY이며 독립적인 물리 GT가 아니다.

| 방법 | T (cm) | R (degree) | ADD (cm) |
|---|---:|---:|---:|
| N3_SUBPIX | 9.75455 | 10.91484 | 17.02865 |
| 같은 hybrid q의 일반 PnP | 9.20715 | 10.61346 | 16.43969 |
| V18 복원 point consensus | 10.1947 | 13.6393 | 19.6464 |

같은 q·H·K·치수·head/CAL·native 원본을 결속했음에도 복원 consensus의 종합 T/R은 일반 PnP와 N3보다 나쁘다. 따라서 추가 veto 제거의 성공과 자세 정확도 목표의 성공을 구분해야 한다. [METRICS.json](METRICS.json), [VERIFICATION.json](VERIFICATION.json).

## 최소 저장 비교는 이미 실행했다

[correspondence_audit.py](../../../scripts/research/pallet_consensus_decision_repair_20261011_v18/correspondence_audit.py)는 기존 POSTHOC 490행과 V17 hybrid의 scored/posthoc를 조인했다. 모든 245프레임, 두 방법 490행을 보존하고 같은 q/H/reference, 실제 fit/inlier ID, 저장 기하와 corner ID의 알려진 cuboid face 포함 관계를 검사했다. 실제 freeze/run은 ROOT가 수행했고 complete/PASS, 예외 없음, 생산 호출 0을 기록했다. 새 실험 제안이나 새 fit의 결과가 아니다. [protocol](CORRESPONDENCE_AUDIT_PROTOCOL.json), [checks](CORRESPONDENCE_AUDIT_CHECKS.json), [summary](CORRESPONDENCE_AUDIT_SUMMARY.json), [행별 근거](CORRESPONDENCE_AUDIT_ROWS.jsonl.gz).

두 방법 모두 비가림 adopted q는 133개다. 기준 좌표 오차가 native보다 개선된 것은 80개, 악화된 것은 53개다. 이는 선택용 LOO 잔차와 후단 기준 좌표 오차가 다른 양임을 보여준다. 해당 q 중 기준 좌표 오차 >8px인 13개는 두 방법 모두 최종 pose-fit inlier로 남았다.

| 방법 | NEW / fallback | N3보다 T/R 모두 개선 | T/R 모두 악화 | 혼합·동일 |
|---|---|---:|---:|---:|
| 같은 q 일반 PnP | 244 / 1 | 63 | 69 | 113 |
| V18 복원 | 240 / 5 | 55 | 74 | 116 |

다음 분포는 승인 NEW 범위(일반 PnP 244행, 복원 240행) 기준이다. `bad`는 기존 기준 좌표 오차 >8px인 최종 inlier, `correct`는 기준 좌표 오차 ≤8px인 실제 fit 점이다. 물리적 진실의 분류가 아니다. fallback은 이 분포에 포함되지 않으며 별도 1행 / 5행으로 기록했다. UNKNOWN을 correct로 합치거나 누락 개수를 추정하지 않았다.

| 방법·분포 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 일반 PnP: bad final | 95 | 59 | 44 | 12 | 18 | 5 | 3 | 8 |
| 복원: bad final | 93 | 45 | 44 | 21 | 20 | 6 | 3 | 8 |
| 일반 PnP: correct fit | 20 | 7 | 25 | 16 | 37 | 31 | 50 | 58 |
| 복원: correct fit | 20 | 11 | 22 | 22 | 30 | 31 | 47 | 57 |

ROOT의 추가 닫힌 저장 readout은 승인 NEW 범위를 따로 기록했다. known proxy correct 점 4개 미만을 실제 물리 대응점 부족으로 단정하지 않는다.

| 승인 NEW 범위 진단 | 같은 q 일반 PnP (244행) | 복원 (240행) |
|---|---:|---:|
| known proxy correct fit <4 행 | 68 | 75 |
| known proxy correct final <4 행 | 76 | 74 |
| UNKNOWN fit 점 / final 점 | 5 / 2 | 3 / 3 |
| bad proxy final ≥1 행 | 149 | 147 |
| correct final ID 집합의 알려진 cuboid face 포함 행 | 48 | 52 |

수치적 rank6 또는 inlier 수가 기준 좌표에 가까운 대응점 4개를 뜻하지 않는다. 반대로 correct fit 점이 많다는 사실만으로 좋은 자세를 인증하지도 않는다. face-ID 기록은 고정 cuboid의 조합 정보이며 일반 coplanarity·절대 conditioning·물리 ownership 판정이 아니다. [CORRESPONDENCE_AUDIT_READOUT.json](CORRESPONDENCE_AUDIT_READOUT.json)은 입력 SHA에 묶인 ROOT의 저장 label/ID 집계이며 새 GT·오차 norm·moment·CI 계산이 아니다. schema와 limits는 [audit 소스](../../../scripts/research/pallet_consensus_decision_repair_20261011_v18/correspondence_audit.py)에 고정돼 있다.

양쪽 NEW인 240행 중 dimension 선택이 다른 행은 9개, 실제 fit ID가 다른 행은 73개다. 전체 245행에서 복원을 같은 q 일반 PnP와 직접 비교하면 양축 개선 77 / 악화 64 / 혼합·동일 104다. 선택 차이를 기록한 사실만으로 특정 branch나 fit ID를 악화의 원인으로 확정하지 않는다.

V18의 의미상 known H 비교에서도 wrong 61행은 양축 개선 13 / 악화 22 / 혼합 26이고, matching 184행은 개선 42 / 악화 52 / 혼합 90이다. 이 조건별 저장 결과는 마스크 오차를 자세 실패와 등치할 수 없음을 보여준다. V17 legacy known 정의와 V18의 UNKNOWN·UNANNOTATED 제외 정의를 같은 정의라고 강제하지 않았다.

## 현재 결론과 한계

추가적인 wrong ID·누락 POSITIVE·H fit 누출이라는 확정 코드 오류는 이번 소스 감사에서 찾지 못했다. 기존 감독 수정, LOO의 조건부 검증, 수치적 합의 승인, 실사 ownership·정확도는 서로 다른 문제다. 같은 q의 닫힌 저장 비교는 backend의 fit/branch 선택 차이를 보여주지만, 음의 성능의 단독 원인을 증명하지 않는다.

V16 전체 scalar audit의 564 실패는 계속 FAIL로 보존한다. V18은 완료된 전체 trace에서 POINT_ONLY 실패가 0임을 결속한 좁은 범위만 사용한다. 전체 line audit가 PASS라고 주장하지 않는다. [VALIDATION_CHECKS.json](VALIDATION_CHECKS.json).

기존 자료만으로 실사 예측 edge의 physical ownership과 독립적인 실제 pose ground truth를 검증 완료할 수는 없다. 남은 한계를 코드 문제로 확정하거나 새 threshold·설정·seed·학습으로 덮지 않았다. 현재 정확도 목표는 미해결이다.
