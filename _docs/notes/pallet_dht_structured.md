# 점·Deep Hough 결합 — structured v2 완료 결과

2026-09-09 현재 개발 목표는 외부에서 일시정지됐다. 아래 실험과 감사는 완료됐지만, **점과 DHT를 결합해 실사 성능을 일관되게 높인다는 전체 목표는 미달성**이다. 불가능함을 증명한 결과도 아니다. 추가 학습·실사 추론·GT 수정은 하지 않는다. 후속 합성 전용 v3/v3b는 [local refinement 기록](pallet_dht_local.md)에 분리했다.

## 구현과 검증 범위

v2는 고정된 YOLO/DHT P4 특징에서 8코너와 12선분의 영상 근거를 읽어 **전체 점 번호배치 후보 하나**를 선택하도록 학습한 검증기다. C4 번호배치, DHT 교점 이동, 작은 이동·크기 변경 후보를64슬롯에 담고 원래 배치를 보존했다. 후보는 예측만으로 생성하며, 합성 GT는 후보 오차 감독과 calibration에 사용했다. 공식 평가에는 GT 기반 임의 번호 변경을 적용하지 않았다. 초기 문서의 구상과 달리 실행된 모델 입력은 GT-free 후보 bank다.

`point_only`, `point_segment`, `point_segment_hough` 세 arm을 seed1, 각2,000 optimizer step/batch16, 131,458파라미터로 실제 학습했다. 초기 가중치·이미지 순서·교란은 같았다. 새 검증기 학습1,792장/calibration256장/합성 검증512장을 분리했다. 전체 backbone을 다시 학습한 실험은 아니다. 세 arm은 같은 DHT 교점 후보를 사용하므로 point_only도 모든 DHT 정보를 제거한 대조군은 아니다.

세 arm 모두 calibration margin0.5를 선택했고, 합성 검증의 두 C4 교란에서 각각 약96–97% 평균오차 감소와 사전 합성 진입 조건을 통과했다. 인위적 교란 복구와 실사 일반화는 구별한다. [프로토콜](../../data/pallet/results/pallet_dht_structured_v2/PROTOCOL.json), [훈련 감사](../../data/pallet/results/pallet_dht_structured_v2/provenance/independent_audit/TRAINING_AUDIT.json), [합성 평가 감사](../../data/pallet/results/pallet_dht_structured_v2/provenance/independent_audit/SYNTHETIC_AUDIT.json).

## 실제 실사 결과

현재 baseline은 논문 R0가 아니라 **기존 pallet_dht_joint_v1/hough_joint_seed1 최종 모델의 저장 예측**이다. 출처는 [decoder 원본 프로토콜](../../data/pallet/results/pallet_dht_decoder_probe_v1/TRAIN_PROTOCOL.json)의 backbone에 고정돼 있다.

실사 DEV319장/13세션, centroid 포함 감독2,818점 중 원래 매칭·예측 유효성을 충족한 **309장/2,738점**이 아래 오차의 분모다. 누락·미매칭80점과 coverage97.1611%를 별도로 유지했다. centroid 제외 코너 전용 분모2,429점과 아래9점 지표를 섞지 않는다.

| arm | mean px | median px | P90 px | 변경 프레임 |
|---|---:|---:|---:|---|
| baseline | 22.454100 | 6.896978 | 41.487329 | 0 |
| point_only | 22.973853 | 6.919365 | 44.153423 | eval_pallet09:1778653664407620608 한 장, 악화 |
| point_segment | 22.081301 | 6.882053 | 39.813888 | plastic_day_01:005838 한 장, 개선 |
| point_segment_hough | 22.454100 | 6.896978 | 41.487329 | 0, 모든 원래 점 유지 |

full DHT-score arm의 실사 개선은 확인되지 않았다. point_segment의 이득도 한 장에 집중됐고, 이미 여러 번 검토한 DEV와 학습 seed1개 결과다. 독립 FINAL, 반복 seed의 안정된 개선, 새6D pose 개선으로 확대 해석하지 않는다. [공식 실사 결과](../../data/pallet/results/pallet_dht_structured_v2/REAL_RESULTS_seed1.json), [좌표·선택](../../data/pallet/results/pallet_dht_structured_v2/REAL_PREDICTIONS_seed1.json).

## 개선한 한 장에서 교점 후보가 필요했는가

완료 후 별도 사전 기록을 만들고 저장된 point_segment 점수·가중치·margin0.5를 유지한 채 교점 이동 후보만 제거했다. cal256/val512 각각4상태와 real319 원래 상태, 총3,391항목에서 남은10후보의 실제 FP32 좌표가 원본의 정확한 부분집합임을 검증했다. 원래 순서도 유지했다. 새 forward·재학습·GT 기반 선택·보정값 변경은 없었다.

005838은 후보45(C4 quarter3+코너5 half-snap) 대신 후보3(C4-only)을 선택했다. 후보3의 저장 점수 gap0.798528이 margin0.5를 넘어 GT 없이 선택됐다. 실사 median/P90 개선은 그대로이고 평균은22.081301→22.080791px로 약간 더 낮았다.

| 사례005838의 공식 감독9점 | mean px | median px |
|---|---:|---:|
| baseline | 118.580959 | 125.665459 |
| C4-only 후보3 | 5.011946 | 4.084113 |
| 기존 선택 후보45, 선 snap 포함 | 5.167237 | 4.084113 |

사후 GT 기준으로 번호배치 변경이 평균113.569013px를 줄였고, 추가 선 snap은 평균0.155291px를 늘렸다. 코너5도4.988218→6.385833px로 악화했다. **이번 실사 이득을 DHT 교점 보정의 기여라고 주장할 근거는 없다.** 합성 국소변형에서는 교점 후보가 평균 약0.0396px를 더 줄이는 대신 양호점1개를10px 밖으로 이동시켰다. 모든 상태 결과를 보존했다.

이는 같은 학습된 scorer에서 후보의 필요성을 본 진단이다. 후보 수가 함께 줄고 기반 DHT 학습 특징은 남으므로, 모든 DHT 정보 제거 또는 동일 조건 재학습 대조군이 아니다. [결과와 한계](../../data/pallet/results/pallet_dht_structured_v2/provenance/proposal_removal_diagnostic/RESULTS.md), [실제 선택](../../data/pallet/results/pallet_dht_structured_v2/provenance/proposal_removal_diagnostic/SELECTIONS.json), [검증](../../data/pallet/results/pallet_dht_structured_v2/provenance/proposal_removal_diagnostic/RECEIPT_AUDIT.json).

## 의자가 겹친 원래 사례

`eval_pallet07:1778652166837872128`에서 v2 세 arm은 모두 후보0을 선택했다. 현재 [좌표·오차 JSON](../../data/pallet/results/pallet_dht_structured_v2/REAL_FRAME_METRICS_seed1.json)의 mask는 `[T,T,T,F,T,T,T,T,T]`다. 7코너+centroid, 총8감독점의 same-ID 오차는 baseline과 세 arm 모두 **mean237.222411 / median270.064622 / P90294.888926px**다. 이 사례는 해결되지 않았다.

과거 첨부 그림의 median267.929370px는 **이전 R0 teacher**의 다른 예측이다. GT와 mask는 같지만 checkpoint와 좌표가 현재 v2와 다르다. 당시 정확한 값267.92936996828024와 R0 checkpoint970a0913…4f7은 [이전 provenance](../../data/pallet/results/hough_line_visual/case_recheck_20260908/provenance/provenance.json)에 있다. 현재 baseline checkpoint는0960fb32…4d37이다. 서로 다른 baseline 수치를 같은 결과로 섞지 않는다.

v3/v3b는 예측 선 주변 ±16 input pixels에서 찾는 국소 보정이고 최종 변위 상한은4 input pixels다. 이640×480 사례의 gain16/21에서는 최대5.25 raw pixels이며 실제 gate/scale로 더 줄어든다. 고정 ID와 이 이동 상한으로 수백 픽셀의 번호배치 오류를 복구할 수는 없다. 이는 해당 국소 구조의 제한이며 모든 전역 점·선 구조가 불가능하다는 결론은 아니다. v3/v3b는 이 실사 프레임에 새로 적용하지 않았다.

## GT 정확성의 기존 증거와 한계

P9041.487329px는 저장 좌표와 공식 same-ID mask로 재현됐지만 GT 전체의 픽셀 정확성 인증은 아니다. GT에는 직접 클릭·PnP 투영·자동 중심·외삽·출처 미상이 섞여 있고, 감독에는 가려진 구조상 코너가 포함된다. v>0 또는 낮은 reprojection만으로 물리적 가시 경계의 정확성을 보장하지 않는다.

기존 GT-only 검토에서 원사례0/1/2/4/7은 보이는 외곽 부근,5/6은 의자에 가려 확인 불가,3은 화면 밖 v0였다. 좌우·상하 필요조건도 정면 면 선택과 전체 ID의 정확성을 보장하지 않는다. plastic_night_01:037376의 높이 끝점 등 재검토 후보는 있지만 대체 좌표·주석 오차 분포·정량 오차 상한을 새로 측정하지 않았다. 모델 오차만 보고 GT를 고치거나 작은 GT 오차가 전체 큰 오차를 설명한다고 결론내리지 않는다. [GT 감사 결론](../../data/pallet/results/pallet_dht_gt_audit_v1/AUDIT_CONCLUSION.json), [GT-only 기하 검토](../../data/pallet/results/pallet_dht_gt_audit_v1/specialist_gt_review.md), [재검토 목록](../../data/pallet/results/pallet_dht_gt_audit_v1/GT_REVIEW_QUEUE.json).

초기 가설은 무한선 점수와 soft DLT만으로 C4 의미 배치를 구별하지 못한 이전 결과에서 출발했다. v2는 전체 선분과 끝점 영상 근거를 추가해 이를 실제 학습으로 검증했다. 이번 한계를 근거로 성공 기준을 바꾸거나 전체 목표를 완료 처리하지 않는다.
