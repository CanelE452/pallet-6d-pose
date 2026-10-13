# 현재 v3 요구사항별 독립 완료 감사

선언된 조건부 처리 방법과 현재 출력 계약은 확인되었다. 소스·출력의 미해결 material 불일치는 발견하지 않았다. 물리 상태의 완전한 판정, 모든 실제 T의 복원, 전역 유일성, 기존 PnP 대비 우수성은 입증되지 않았다.

코어 수정 없이 현재 source19개와 evaluation/smoke snapshot·seal, 모델·파라미터·체크포인트 binding을 확인했다. 아래 감사는 current v3에 적용하며 이전 v2 리뷰의 seed 투영 고정·물리 guard 해석을 대체한다. 물리적으로 불일치했던 과거 oracle CAL mechanism은 성공 근거로 사용하지 않았다.

| 요구사항 | 확인 결과 |
|---|---|
| 초기 S 영상 배제 | main98 + fresh 원/교란6 runtime에서 S feature/mask prompt/보정/영역/PnP term 0. S raw는 영상 단계 전에 제거된다. |
| BI 보정·고정 | v2→v3 98건의 q/cov 및 상태·마스크·영역 등 882개 upstream 필드가 정확히 같다. readout은 frozen BI q를 그대로 보존한다. |
| 관측 오차·물리 모델 지원 | seed projection 고정 off. 각 BI whitening norm≤3, model nonself 2° 및 closed canvas 제약을 독립 수치로 다시 확인했다. 이는 가정한 noise budget이며 보정된 확률이 아니다. |
| O 영역 | 실제 외부 가림 CAL eligible7점/2sessions의 raw-centered empirical max 모델. S/TEST로 fit하지 않았다. finite-confidence 보장은 없다. |
| T 영역 | finite T 좌표를 만들지 않는다. side가 독립적으로 확인되지 않으면 화면 밖 4-side union이며 cap256 초과는 unresolved다. |
| 모든 O/T 공동 처리 | 6DOF SLSQP 연속 최적화의 각 callback에 모든 individual 영역을 넣고 최종 scalar 검산한다. old finite poses만 scoring하는 방법이 아니다. |
| BI2+OT와 모호성 | accepted 가족3건은 항상 ambiguous, OT_solved=False, uniqueFalse. 표본 투영은 전체 연속 가능집합이나 정밀 해가 아니다. |
| 실패·fallback | 공동 실패49건은 O/T None/UNRESOLVED. valid all-guard nonfamily BI seed만 fallback 허용; 현재 실패 출력은 original coarse의 S-only fallback이다. |
| S readout | accepted 또는 명시 fallback의 원 native 모델 투영으로만 채운다. BI 관측을 투영으로 덮어쓰지 않으며 U는 미해결이다. |
| native/Q/9번째 중심 | 원 순서·치수 유지. Q=R_cf_originalᵀR_physical_original, R_physical_final=R_cf_final Q. 9번째 center는 original metadata 그대로다. |
| GT 분리·전체 population | ordered CAL46/TEST49 + 별도 real crop3의 총98건 예측을 봉인한 후 score했다. prediction/candidate generation에 좌표 GT를 사용하지 않는다. |
| fresh 전체 API | CAL 실제 RGB3사례의 Tiny/SAM/current API. 상태·BI·domain이 cached stage와 같고 S 좌표 교란에도 모두 정확 불변. 98건 전체 새 RGB 실행으로 표현하지 않는다. |
| 테스트 | 현재 core/test/log SHA가 95 distinct pass receipt와 일치한다(기본90 + physical5). 이 독립 감사에서 pytest를 다시 실행하지는 않았다. |

자동 검산은 33 묶음, 4,137개 확인에서 실패0이다. 자세와 투영으로 BI283개 및 O/T60개 제약을 독립 계산했다. accepted main49건은 primary CAL18/46, TEST29/49, crop2/3으로 구성되며 main의 개별 O/T48개 제약이 모두 pass한다. 숫자 근거는 [INDEPENDENT_COMPLETION_EVIDENCE_V3.json](INDEPENDENT_COMPLETION_EVIDENCE_V3.json), 요구사항·해시는 [INDEPENDENT_COMPLETION_AUDIT_V3.json](INDEPENDENT_COMPLETION_AUDIT_V3.json)에 있다.

‘S RGB 0’은 초기 기하가 S로 고정한 ID의 보장이다. 실제 HUMAN SELF 중 CAL13/TEST13점은 B/I로 분류되어 영상 근거를 사용했다. 실제 물리 S가 모두 정확히 배제되었다고 확대 해석할 수 없다. 최종 모델 가시성이 달라도 초기 S는 재라벨하거나 RGB를 조회하지 않는다. 현재 RGB 상태 판정은 TEST actual O12 중 O11/B1이고, actual DIRECT286 중 O11 오분류가 남는다.

자연 T 라벨은0이다. real crop의 upstream KP/K는 변환해 재사용했고 cropped backbone을 새로 추론하지 않았다. 첫 crop은 실제 T4/T5 중 T4를 O4로 놓쳤다. 둘째 crop은 실제 T5를 U5로 놓쳐 공동 해가 없다. 셋째 crop은 T0/T3을 잘못 조건부 판정하고 실제 T5를 U5로 놓쳤다. 셋째 가족 표본의 일부 투영은 약1.111×10⁸px까지 넓다. conditional feasibility를 물리 T 복원 성공이나 신뢰구간으로 표현하지 않는다.

같은 BI-only 대비 accepted 자세 차이는 score 기준으로 CAL2/TEST7건이다. 기준은 R matrix norm>10⁻⁶ 또는 translation>10⁻⁷m이며 TEST7 중 한 건은 Δt≈3.74×10⁻⁷m의 수치적 차이다. 모두 실질적 개선 사례의 수를 뜻하지 않는다. summary의 TEST8은 실패 fallback 차이1건도 포함하므로 accepted7과 구분한다.

TEST49 전체에 fallback을 포함하면 original PnP는 3.363cm/6.527°/40성공, current v3는 3.514cm/6.568°/39성공이다. 우수성은 입증되지 않았다. TEST는 이미 재사용된 탐색 평가이며 derived pose reference를 쓴다. 작은 CAL O 표본, 가정 covariance, 조건부 초기 상태 및 유한 다중 시작점 탐색의 한계도 남는다.

현재 [README_KO.md](README_KO.md), [METHOD_DETAILS_KO.md](METHOD_DETAILS_KO.md), [RESULTS_V3.json](RESULTS_V3.json)는 이 범위와 지표를 구분한다. [current_figures_v3](current_figures_v3/index.html)의 첫 두 패널은 S 좌표를 제외하고, final family는 finite samples와 clipped projection 범위를 명시한다. crop의 실제 T 누락·false T를 숨기지 않는다.
