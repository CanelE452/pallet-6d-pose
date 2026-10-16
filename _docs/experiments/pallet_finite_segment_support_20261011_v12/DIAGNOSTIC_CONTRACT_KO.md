# 유한 물리 edge의 관측 지원과 별도 물리 branch 진단

이 문서는 V11 출력이 좋지 않다는 이유로 설정을 조정하는 계획이 아니다. 기존 C2의 실제 관측 의미에서 확인하지 않은 두 필요조건을 저장된 GT-free 기하만으로 진단한다. 기존 모델·decoder·좌표·H·치수·초기화·loss·optimizer·선택·NEW/fallback·원행은 바꾸지 않는다. 새 fit·PnP·SVD·모델·학습·RGB·GT 채점·bootstrap은 0이다. 공개 결과의 전체 245장(Clean153, Moderate92)을 유지하며 분류 실패나 불확실성이 있는 프레임도 원행에서 지우지 않는다.

## 확인할 두 가지

[V5 residual_jacobian](../../../scripts/research/pallet_partial_line_independent_20261010_v5/solver.py)은 등록 3D edge의 두 투영 끝점에서 관측 무한직선까지의 법선 잔차를 사용한다. 실제 `support_points`와 `query_radii_px`는 소유 edge/CAL/선 합의를 검증할 때 보존되지만, 최종 유한 투영 segment가 실제 관측 구간을 설명하는지 확인하지 않는다. 멀리 있는 작은 segment도 같은 무한선에 놓일 수 있다. 이는 실제 영상의 edge 소유권이 잘못됐다는 증명이 아니라, 부분적인 **유한 물리 변 관측**을 해석할 때 남은 검사다.

각 보존 candidate의 `projected[a], projected[b]`로 닫힌 유한 segment를 만들고 원래 CAL query의 실제 선택 위치에서 그 segment까지의 최소 Euclidean 거리, segment parameter, 첫 끝점 앞/둘째 끝점 뒤의 접선 방향 초과 길이를 기록한다. 해당 query의 원래 radius와 거리 차이, 투영 무한직선까지의 법선 거리, 원 source line의 자체 잔차도 별도 기록하여 법선 불일치와 유한 구간 초과를 구별한다. radius는 decoder의 조건부 bin 분산과 합성 CAL에서 만든 관측 불확실성 surrogate다. 물리 진실을 보장하는 오차 상한이나 완전한 2D covariance가 아니다. Euclidean radius는 실제 검색선 방향보다 넓은 보수적인 진단 해석이다. 진단에서는 새 slack/threshold를 추가하거나 radius를 수정하지 않는다. 포함 여부가 틀렸다고 프레임을 버리거나 원래 pose를 바꾸지 않는다. 0길이 segment는 점까지의 거리로 기록하며 별도 degenerate 표식을 남긴다.

둘째로 서로 다른 물리 branch가 같은 전체 실제 factor를 기존 8px inlier로 설명하는지 확인한다. 저장된 candidate의 `accepted`, 모든 양의 camera depth, 관측/모델 법선 Jacobian rank6, `final_inliers == fit_input_ids`, `final_line_inliers == fit_line_edges`를 읽는다. 저장된 `numerically_equivalent_to_selected=False` 대안이 선택 후보와 모두 이 조건을 만족하면 별도 `observation_consensus_ambiguity_with_selected`를 기록한다. 기존 `numerical_objective_tie`, cost와 차이, R,t, condition과 전체 candidate를 그대로 보존한다. 새 수치 동점 threshold나 통계적 likelihood를 만들지 않는다. 두 후보의 8px 합의가 같다는 것은 어느 후보가 GT와 가까운지, 실제 같은 물체 대칭인지, 올바른 소유 edge인지 증명하지 않는다. rank는 지역 정보이며 이 검사는 전역 유일해를 증명하지 않는다.

## 입력 권위와 실행

[audit.py](../../../scripts/research/pallet_finite_segment_support_20261011_v12/audit.py)는 V11의 490 기하 행·245 ledger·seal·완료 receipt·source population proof·원 검산 실패·별도 전체 review PASS/protocol/STARTED와 그 모든 바이트 입력을 고정한다. 원 `KeyError('best_candidate')` 95검사 실패를 성공으로 바꾸지 않는다. V7 원 기하 1,960행과 관측 735행의 전체 순서를 같이 읽고, 각 원행의 `IMAGE_ROLE_BOUNDARY_ONLY`와 `IMAGE_ROLE` 관측을 semantic SHA로 연결한다. 원 선택 query의 위치와 line 지원점이 정확히 같고, semantic edge/등록 끝점/원 line record가 동일해야 한다. 채택 코너에서 이미 쓴 선과 제외한 선은 중복 사용하지 않는다. H 초기 2D 좌표는 실제 fit ID에 없어야 한다. 두 운용 view는 동일한 한 번의 filtered LOCAL 제안/candidate를 공유해야 한다.

검사 대상은 모든 보존 candidate다. weak gate로 기본 출력을 반환한 경우, insufficient/수치 실패/다중해로 새 pose가 없는 경우도 프레임 상태와 빈 후보 목록을 유지한다. 진단에서 97개 rank-view NEW와 94개 safe-view NEW는 저장된 출력 상태를 각각 세며, 이를 새 실행으로 세지 않는다. 원 solver candidate가 없는 프레임을 지원 검사가 성공한 프레임으로 대신 세지 않는다.

새 `DIAGNOSTIC_PROTOCOL.json`을 arithmetic 전에 한 번 고정하고 `DIAGNOSTIC_STARTED.json` 뒤에만 저장된 segment 거리와 집계를 계산한다. 출력은 `SEGMENT_ROWS.jsonl.gz` 245행과 `DIAGNOSTIC_RECEIPT.json`이다. 실패 prefix는 `INTERRUPTED_SEGMENT_ROWS.jsonl.gz`에 보존하고 자동 재시도/덮어쓰기하지 않는다. gzip close, raw flush/fsync/close 오류는 첫 오류를 덮지 않고 별도 기록한다. 모든 입력과 protocol SHA를 끝에 다시 확인한다. 이 코드는 읽기 전용 진단이며 운용 필터나 새 성능 평가 구현이 아니다. 루트가 게시 상태와 보호 자료를 확인하고 읽은 후 독립적으로 동결·실행한다.

```bash
python -m scripts.research.pallet_finite_segment_support_20261011_v12.audit freeze
python -m scripts.research.pallet_finite_segment_support_20261011_v12.audit run
```

## 실행 전 저장 증거 세 사례

이 사례는 GT나 성능을 조건으로 뽑지 않았다. V11 GT-free 기하의 원 순서에서 safe NEW이고 선택/비동일 대안이 모두 동일한 전체 factor의 8px inlier를 가진 첫 세 사례다. 출처는 V11 `GEOMETRY_SEALED.jsonl.gz`, SHA `440577156615adc0e0e9a0e7b0fb389ebf66318cdb7753b280e0631a1a304809`, 18,224,586bytes다. 아래 값은 저장된 원 수치를 전사한 것으로 새 fit/GT/수치 재계산이 아니다. 전체 발생률은 아직 이 문서의 작성 단계에서 계산하지 않았다.

| ID | U / line / H | 선택·대안 dim | 선택·대안 soft cost | 저장된 대안 cost 차이 | 선택·대안 observed condition |
| --- | --- | --- | --- | --- | --- |
| eval_noapril:1775201421043401984 | [0,4,5] / [2] / [6] | 1 / 0 | 1.6590069853486256 / 6.4191418751540965 | 4.760134889805471 | 5.007615771999426 / 4.997068134506671 |
| eval_noapril:1775201428535961600 | [0,1,4,5] / [2] / [6] | 1 / 0 | 8.359655333022062 / 32.59622741485401 | 24.23657208183195 | 1.7467523277724581 / 1.900635318925052 |
| eval_noapril:1775201436800711168 | [0,3,4,5] / [] / [6] | 1 / 0 | 12.093262337284699 / 52.8116112066337 | 40.718348869349 | 7.44934870481828 / 8.47434841321627 |

첫 사례의 저장된 선택 z는 3.979279473585493m, 대안 z는 3.191943817467752m다. 선택 점 잔차는 [0.760884885871842, 0.9213126545210328, 0.9278536027146487]px, 선 RMS는 1.0199871509997709px다. 대안 점 잔차는 [1.9588037923609412, 0.5456062758860614, 0.4070029557403375]px, 선 RMS는 2.9803285201209495px다. 셋 모두 저장된 `multiple_solutions=True`, `unresolved_ambiguity=False`, 대안 `numerically_equivalent_to_selected=False`, `numerical_objective_tie=False`, weak gate 미적용이다. 서로 다른 물리 branch가 수치 동점은 아니어도 기존 관측 합의를 동시에 만족할 수 있음을 보인다. 이 세 사례만으로 대안이 더 정확하다고 주장하지 않는다.
