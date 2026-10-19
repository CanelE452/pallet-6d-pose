# 팔레트 B/I 전체 실행 결과

**전체 성능:** 최종 MATCHED_BI의 대표 seed20261018 성공은 109/192로 원 H1의 120/192 대비 -11건이다. T/R 평균은 11.733cm/9.605°이며, 원 H1은 12.073cm/11.549°다. 성공·평균·tail을 함께 비교하며 한 지표의 개선을 전체 지배로 해석하지 않는다.
MATCHED_COMMON은 110/192로 원 H1 대비 -10건이며, T/R 평균은 12.538cm/9.870°다.
대표 seed의 PF 대비 절대오차 차이: F_COMMON−POSEFIX_SUBPIX_PF_ANCHOR: Δ성공-2, ΔT평균+2.689cm/ΔR평균-0.669°; F_COMMON−POSEFIX_SUBPIX_YOLO_ANCHOR: Δ성공-1, ΔT평균+3.160cm/ΔR평균-0.724°; F_BI−POSEFIX_SUBPIX_PF_ANCHOR: Δ성공-3, ΔT평균+1.884cm/ΔR평균-0.934°; F_BI−POSEFIX_SUBPIX_YOLO_ANCHOR: Δ성공-2, ΔT평균+2.355cm/ΔR평균-0.989°. 두 군 모두192 available이면 저장 평균차는 같은192 ID의 paired mean delta와 대수적으로 같다. 별도 PF confidence interval은 새로 계산하지 않았다.
**B/I 추가 효과:** MATCHED_BI−MATCHED_COMMON의 PRIMARY 성공 차이는 세 seed 순서로 **-1/+3/-10건**이다. 일관된 B/I 추가 성공 이득은 관찰되지 않았다. 이는 고정 정책 묶음의 관측 결과이며, 원 H1 대비 변화와 구별한다. 아래 촬영군·paired interval을 함께 읽어야 하며 독립 신규 실사 확증은 **0건**이다.
현재 권고는 N3/SubPix를 유지하는 것이다. 제거군 근거가 없으므로 이 실행으로 제거를 권하지 않는다. 등록된 최종 구성을 저장 지표대로 평가하고 새 실사에서 독립 확증한다.

과학·파생 산출물과 이 보고서의 준비 완료를 나타낸다. 공개 패키지·Git commit/push·원격 SHA 확인은 별도 ROOT 단계이며 `whole_campaign_complete=false`다. 대표 seed는 사전 지정20261018이고 좋은 seed로 바꾸지 않았다.

## 1. 실제 완료와 재사용

| 단계 | 상태 | 실제 범위 |
|---|---|---|
| P0 | COMPLETE_EXACT_REUSE | 원 입력·기준·고정 ID·PF tensor 정확 재사용 |
| S1 | COMPLETE_NEW | D1+PF2 union과 own H1 |
| D_QCOV | DIAGNOSTIC_ONLY | 고정 공통 graph q/Σ 진단; 전체 방법과 분리 |
| S2 | COMPLETE_NEW | 전체 normal C0/CP/CB와 own H1 |
| S3 | COMPLETE_NEW | COMMON/BI generic support 제안+union |
| S4 | COMPLETE_NEW | PL/CONTOUR/IMAGE 실제 새 pose+union |
| F | COMPLETE_NEW | 고정 VAL 선택 auxiliary를 양쪽에 동일 적용한 matched 정책 |
| R | COMPLETE_NEW | 독립 재집계·전체 gallery·scatter·비용·재현/보고서; 공개는 대기 |

실사238=PRIMARY192(TEST49+REMAINDER143)+CAL46을 모두 유지했다. 실패/no-pose/fallback은 이진 분모에서 빠지지 않는다. 연속오차는 available-only, sample variance/SD는 ddof1이며 trimming은 없다. CAL 및 반복 사용 PRIMARY는 개발 자료다. 독립 checker 일치는 같은 자료의 계산 검증이며 새 독립 test를 만들지 않는다.

## 2. 생략·고정한 것과 비교 범위

원 spine의 N3/SubPix를 고정한 채 추가 후보·관측 정책을 비교했다. 제거군을 실행하지 않았으므로 이 결과로 N3/SubPix가 필요 없거나 이득의 원인이라고 말할 수 없다. 큰 관측 모델의 재학습, 새 시퀀스 수집, human material-owner annotation은 하지 않았다. 현재 라이브러리·등록 자산과 비학습 grayscale DIS를 사용했으며 DeepIM/SRT3D 논문 전체 재현을 주장하지 않는다.
새 MATCHED_COMMON/BI는 동일한 가용 cue·mask·geometry 및 후단 역할 특징을 가진 강한 대조군이다. 원120 H1과 full-normal 후보/특징·일부 입력 채널 범위가 달라 원 방법의 직접 parity나 모든 B/I 정보 제거군으로 부르지 않는다. native S는 모델 투영만 유지하며 새 이미지 측정·regression·seed 관측으로 추론하지 않았다. [방법 상세](METHODOLOGY_KO.md)와 [구조 검토](STRUCTURE_REVIEW_NOTES_KO.md)를 함께 본다.

## 3. 전체 방법의 성능과 tradeoff

성공은 T≤5cm이고 C2 회전오차≤5°다. C2는 Y축180°만 허용하며90° 혼동은 오류다. 아래 성공 열은20261018/19/20이며 정적 N0/E0/PF는 한 번의 동결 결과다. T/R 평균과 tail은 대표 seed의 저장값이다.

| 군 | PRIMARY 성공 | T 평균 cm | R 평균 ° | T>50 / >100 / R>45 | no-pose |
|---|---:|---:|---:|---:|---:|
| N0 | 108 | 11.306 | 10.631 | 8/5/20 | 0 |
| E0 | 114 | 14.112 | 13.001 | 15/3/26 | 0 |
| POSEFIX_SUBPIX_PF_ANCHOR | 112 | 9.848 | 10.539 | 10/3/20 | 0 |
| POSEFIX_SUBPIX_YOLO_ANCHOR | 111 | 9.377 | 10.594 | 9/2/20 | 0 |
| D1_H1_MLP | 120/117/121 | 12.073 | 11.549 | 9/5/21 | 0 |
| S1_UNION | 119/118/120 | 11.614 | 10.284 | 7/5/19 | 0 |
| S2_C0 | 117/119/118 | 11.318 | 10.655 | 8/5/20 | 0 |
| S2_CP | 115/112/112 | 11.143 | 10.275 | 7/5/19 | 0 |
| S2_CB | 111/114/113 | 12.752 | 11.229 | 10/6/20 | 0 |
| S3_COMMON | 120/124/122 | 11.624 | 10.656 | 8/5/20 | 0 |
| S3_BI | 119/118/120 | 11.492 | 10.322 | 7/5/19 | 0 |
| S4_PL | 123/123/122 | 11.488 | 10.659 | 9/5/20 | 0 |
| S4_CONTOUR | 117/118/124 | 11.499 | 10.642 | 8/5/20 | 0 |
| S4_IMAGE | 117/118/118 | 12.573 | 12.884 | 10/5/24 | 0 |
| F_COMMON | 110/114/117 | 12.538 | 9.870 | 9/5/17 | 0 |
| F_BI | 109/117/107 | 11.733 | 9.605 | 8/4/17 | 0 |

[전체40 군 PRIMARY: 평균·sample variance·SD·중앙값·P90/P95·최댓값·tail](tables/final_report/FULL40_PRIMARY_METRICS.csv) · [네 population 전체160행 원정밀 값](tables/PUBLIC_METRICS40/ALL_POPULATION_METRICS.csv) · [오차 예산40](tables/ERROR_BUDGET40_V1/ERROR_BANDS.csv). PF 두 anchor의 한 결과를 세 독립 PF 실험으로 늘리지 않았다. 오차 예산은 저장 scalar의 기술적 분해이며 새 사전등록 검정이나 selector 이득 추정이 아니다.

## 4. B/I−COMMON과 촬영군 차이

| 비교 | seed | recovery / damage | Δ 성공 | Δ percentage point | 6촬영군95% CI, pp |
|---|---:|---:|---:|---:|---:|
| S3_role_policy | 20261018 | 0/1 | -1 | -0.521 | [-1.117, 0.000] |
| S3_role_policy | 20261019 | 0/6 | -6 | -3.125 | [-4.902, -1.136] |
| S3_role_policy | 20261020 | 0/2 | -2 | -1.042 | [-3.279, 0.000] |
| final_matched_BI_effect | 20261018 | 5/6 | -1 | -0.521 | [-5.106, 3.352] |
| final_matched_BI_effect | 20261019 | 5/2 | +3 | 1.562 | [-3.093, 4.255] |
| final_matched_BI_effect | 20261020 | 3/13 | -10 | -5.208 | [-7.843, 2.778] |

MATCHED BI−COMMON의 촬영군별 Δ성공(세 seed 순): REC_001:-4, REC_002:+3, REC_007:+0, REC_025:+0, REC_039:+0, REC_044:+0 / REC_001:+0, REC_002:+4, REC_007:+0, REC_025:+0, REC_039:+0, REC_044:-1 / REC_001:-4, REC_002:-7, REC_007:+0, REC_025:+0, REC_039:+0, REC_044:+1.
[모든30 paired 비교·4 population·각 recording/named-session interval·leave-one-cluster 결과](tables/PUBLIC_CONTRASTS30/ALL_POPULATION_PAIRED_CONTRASTS.json)에 전체 회복/손상 ID를 유지한다. 6 recording cluster,10000 draws의 저장 CI를 복사했으며 세 seed가 새 독립 이미지 수를 늘리지는 않는다.
최종 auxiliary 구성은 **S2_CP**다. actual own-VAL235 세 seed의 median success→median loss→실측 production-wall 순서로 사전 고정한 결과를 사용했다. PRIMARY로 후보를 재순위화하지 않았다. 등록된 all-negative 개발 fallback bit는 평가됐고 실제 선택의 `all_negative_fallback=false`다. 원120 fallback을 새 이름으로 실행한 것으로 쓰지 않는다.

## 5. 확인된 연결: 후보 생성→준입→선택

| 군 | 원31 선택18/19/20 | 원41 generated/eligible; 선택 | PF후38 generated/eligible; 선택 |
|---|---:|---:|---:|
| S1_UNION | 0/0/1 | 3/3; 0/0/0 | 0/0; 0/0/0 |
| S2_C0 | 10/10/9 | 11/11; 0/0/1 | 10/10; 0/0/1 |
| S2_CP | 8/5/5 | 6/6; 0/0/0 | 5/5; 0/0/0 |
| S2_CB | 4/6/5 | 6/6; 0/0/0 | 5/5; 0/0/0 |
| S3_COMMON | 1/6/3 | 7/7; 1/1/1 | 4/4; 1/1/1 |
| S3_BI | 0/4/2 | 7/7; 1/1/1 | 4/4; 1/1/1 |
| S4_PL | 1/3/1 | 10/10; 3/3/3 | 7/7; 3/3/3 |
| S4_CONTOUR | 0/0/6 | 12/12; 3/3/3 | 9/9; 2/2/2 |
| S4_IMAGE | 1/1/1 | 11/11; 0/0/0 | 8/8; 0/0/0 |
| F_COMMON | 5/7/9 | 13/13; 1/0/0 | 12/12; 1/0/0 |
| F_BI | 4/8/3 | 12/12; 0/1/1 | 11/11; 0/1/1 |

집합31/41/38와 PF-new-good3/old12는 원래 ID로 고정했다. [다섯 cohort 모든 군·seed 수치와 transition ID](tables/final_report/FIXED_COHORTS.json)를 남기며 새 군의 자기 실패 평균끼리 빼지 않는다. good 후보 coverage가 늘어도 실제 H1 선택이 좋아진다는 보장은 없다.
PL은 기존 native point/typed-normal과 pose에 따라 갱신한 native-outline evidence를 쓰며 이 branch는 CAD/proxy renderer를 쓰지 않는다. 따라서 PL 개선을 인증 material CAD·전체 RGB·물리 owner 정합의 성공으로 설명할 수 없다. IMAGE의 good endpoint 존재 후 낮은 H1 순위는 저장된 feature/logit 연관으로 설명 가능한 손실 위치이며 원인 개입 실험은 아니다.
[S1/D 짧은 근거](S1_D_EVIDENCE.md) · [S3/S4 짧은 근거](S3_S4_EVIDENCE.md). q/Σ/가족 전환 D는 제한 공통 graph 진단이며 전체 H1 B/I 효과와 합치지 않는다. 원 D는Σ+Γ donor이고 corrected D는COMMON reliability Γ를 고정한다.

## 6. 미식별 원인과 최소 추가 증거

남은 오차를 feature 하나, proxy, 관측 cue 또는 학습 분포의 단일 원인으로 식별하지 못했다. 필요한 최소 증거는 독립 physical correspondence/owner 확인, signed axes·instance geometry의 독립 등록 검사와 새 촬영 실사 검증이다. 특정 loss feature의 원인 효과에는 다른 요소를 고정한 추가 개입이 필요하다. 현재 결과를 본 뒤 threshold·seed·pose·F 구성을 다시 고르지 않았다.
S3 kept-original의 정확 factor ID와 PL 매 update outline 원좌표가 모두 저장된 것은 아니므로 완전한 좌표 ancestry를 주장하지 않는다. raw support가 없는 old factor는 UNKNOWN_CORRELATION으로 유지될 수 있으며 source-ID/pixel8px dedup은 인증된 물리 독립성이 아니다. 고정 row/endpoint/support/cost binding의 기록 범위를 방법 문서에 명시했다.

## 7. 독립 확증·자산·공개 한계

human owner 인증은0이며 exact real CAD identity와 signed axes도 미인증이다. 합성 VAL235의 exact named GLB 부재/registered proxy 사용은 확인된 연결·분포 한계이지 이 결과만으로 형상이 실제 틀렸다는 실증이 아니다. 자동 structural proxy를 bare cuboid material CAD나 독립 physical truth로 승격하지 않는다. 물리 owner/자산 인증은 `NOT_APPLICABLE_WITH_REASON`의 별도 한계이고 실제 자동 S3/S4 전체를 BLOCKED로 바꾸지 않는다.
[전체 transition gallery](galleries/F/index.html)와 [full-axis scatter](figures/FULL_T_horizontal_R_vertical/index.html)는 파생 시각자료다. 제한된 pixel/AI 검토는 human 실사 정답이나 전량 pixel 검토가 아니다. 원24/private RGB·GT/Rt·weights·banks·등록 자산은 공개 패키지에서 제외하고 기존 승인 PRIMARY derived 범위만 공개한다.

## 8. 실제 실행 비용·재사용·완료 범위

새 H1 fit은 실제 **33회/등록33회**이며 tiny real H1 prediction forwards는 **7854**, 새 backbone prediction forward는 **0**다. full-normal minimize 알려진 lower bound는 **1262703**이고 D는 별개 786+64회다. S3 objective/Jacobian·DIS/PnP·render는 다른 단위로 분리하여 하나의 총 호출 수로 더하지 않는다.
원 D4 prefix 112387회는 복구된 base 누적값에 이미 포함되어 다시 더하지 않는다. exact reuse는 복사된 FRAME의 intrinsic 비용을 새 optimizer 비용으로 청구하지 않았다. 원 중단8 arm의 미기록 작업, 구형 import/finalization 등은 성공적 복구 뒤에도 **UNKNOWN_NOT_ZERO**다.
문서화된 START→비용 측정 cutoff elapsed는 **23615.45459151268초**이며 범위는 `scientific+derivedcomplete;cost/report/package/Gitpending`다. 공개/Git 후 최종 workflow elapsed가 아니다. child wall 합·서로 겹친 service elapsed·내부 fit timer를 campaign elapsed로 쓰지 않고, inclusive service CPU와 그 자식/내부 타이머도 중복 합산하지 않는다.
추가 파생 작업 서비스의 inclusive CPU/elapsed는 229.706619초/9318.209903초다. master 감사 child의 외부 CPU/wall은 18.010686초/19.709193260001484초로 내부 timer 밖 출력 write/import/teardown을 포함한다. 이 값들도 자식·내부 범위와 중복 합산하지 않는다.
[Typed 비용 원장](tables/costs/PUBLIC_COST_SUMMARY.json) · [파생 서비스 비용](tables/costs/DERIVED_SERVICE_PUBLIC.json) · [cache/환경/모델/교정기/실제argv/source 인덱스](PUBLIC_PROVENANCE_INDEX.json) · [재현 절차](REPRODUCE.md). 이 보고서는 저장 scalar와 봉인 메타데이터만 연결했다. 새 pose/GT 오차계산·fit·forward·render·bootstrap·선택은0회다.
