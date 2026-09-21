# Occlusion robust refiner — 실행 결과

[확인] **E1 무학습 진단 수행. E2는 clean pool 무결성 gate에서 STOP, 학습 0 update.** E2 성능 FAIL이 아니라 입력 조건 미충족이다. E3~E6·commit·push는 실행하지 않았다.

## 문제와 이번 차이

[확인] 기존 clean8 학생 fine-tuning은 occlusion 평가에서 일관된 개선이 없었다. 이번 계획은 frozen R0를 유지하고 보정기만 학습하며, 가린 RGB에서 R0를 실제 재추론한 입력과 clean pseudo target을 분리하는 것이다. 기존 pose-only 실험을 이름만 바꿔 재실행하지 않았다.

[추정] 이 대조가 실행되면 artificial occlusion 입력과 source replay의 복원/보존 효과를 구분할 수 있다. 이번에는 gate에서 멈췄으므로 그 효과를 결론낼 수 없다.

## E1-A: 현재 후보로 얼마나 복구할 여지가 있는가

| [확인] 집합 | R0 PCK10 | N2 | N3 | 합성 PoseFix | Replay | 진단 oracle | 최고 실제 대비 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRIMARY_OCC96 | 41.79% | 47.49% | 46.95% | 46.13% | 47.49% | 51.29% | +3.80pp |
| CLEAN_NONCAD69 | 84.42% | 87.86% | 87.50% | 86.96% | 88.04% | 89.31% | +1.27pp |
| GREEN150_MANUAL | 82.23% | 81.35% | 미제공 | 80.18% | 77.39% | 83.70% | +1.47pp |
| DEV72_REFERENCE_UNKNOWN | 53.51% | 59.46% | 58.56% | 57.84% | 60.18% | 62.88% | +2.70pp |
| CAD18 | 41.18% | 66.91% | 67.65% | 54.41% | 63.97% | 78.68% | +5.15pp |
| SELECTED_CLEAN8 | 37.50% | 64.06% | 65.62% | 43.75% | 73.44% | 76.56% | +0.00pp |

[확인] CAD18/clean8 oracle에는 R0/N3/Replay의 기존 자기 가림 PnP 후보도 포함한다. 다른 집합에서는 이 후보가 없고 GREEN N3 저장본도 없어 포함하지 않았다. 모든 집합은 별도 분모이며 합산하지 않았다. 전체 지표와 후보 가용성은 E1_CANDIDATE_HEADROOM에 있다.

[확인] occlusion96은 737코너, R0/N2/Replay 모두 동일 검출에서 매칭88/96이다. N2와 Replay PCK10은 같은47.49%지만 Replay는 R0 >20→≤10 복구4개 / R0 <5→>10 손상2개(좋은116개 중1.72%), N2는 복구0/손상0이다. P90은 R0 64.14px, N2 63.95px, Replay65.64px다.

[확인] GREEN 수동681코너에서는 Replay가 PCK10 77.39%로 R0 82.23%보다 낮다. 좋은344코너 중14개(4.07%)를 손상시켰다. 진단 oracle도 최고 실제 대비+1.47pp이며 누락된 N3 후보에 대해서는 판단하지 않는다.

[추정] 현재 확인한 후보 집합의 occlusion 선택 여지는 중간(+3.80pp)이며, 선택기만으로 큰 오류 전부를 해결할 수 있다는 근거는 없다. oracle도 >20px 꼬리가 남는다. 후보 생성/표현 개선 가능성과 target-domain 적응 가능성은 별도 질문이다.

## E1-B: geometry

[확인] camera/pallet movement parity가 검증된 physical clean/occlusion pair는 감사한 자료에서 발견하지 못했다. CAD18×3후보의 자기 가림 출력54개를 재현했다. 40개 hidden-only 적용,14개 reliable visible point<6으로 원본 유지. 사용점 분포·재투영 residual·hidden set·각 치수축±5% 민감도를 기록했다.

[확인] GT-visible external occlusion oracle은 미실행이다. 숨은 코너 GT 일부는 PnP 유래일 수 있다. 기하 sanity를 독립 물리 복구 검증으로 해석하지 않는다. GREEN의 과거 카메라 메타데이터 불일치를 이번 작업에서 해결했다고 주장하지 않는다.

## E1-C 및 E2 gate

[확인] 기존 pool/scout/expanded 후보 2276행, 이미지 SHA 중복 제거 2240장. 기존 메타데이터와 세션 분리까지 확인한 eligible clean 후보 0장. RGB_NOT_YET_REVIEWED나 condition 미기록을 clean으로 간주하지 않았다. 수도레이블 생성·E2 학습은 미실행이다.

[확인] 기존 선택CAD8은 평가 세션 제외 규칙에 따라 이번 학습으로 자동 전환하지 않았다. 이전 실험의 역할 변경은 그 실험에 한정된 것이며 이번 지시를 덮어쓰지 않는다.

## 최종 질문에 대한 답

1. [확인] clean pseudo-label로 보정기가 적응했는가? **미검증** — E2 미실행.
2. [확인] occluded-R0 input이 CLEAN보다 좋은가? **미검증** — 기존 학생 결과로 대체 답변하지 않는다.
3. [확인] source replay가 필요한가? **이번 대조에서는 미검증**. 이전 Replay 결과는 과거 별도 설정의 증거다.
4. [확인] 새 보정기가 N2/N3/Replay보다 나은가? **새 모델 없음**. frozen 후보의 차이만 측정했다.
5. [확인] 큰 오류 복구와 손상은? occlusion96 Replay 복구4/손상2, N2 복구0/손상0; GREEN Replay 복구1/손상14. 세부 분모와 모든 후보는 E1 표 참조.
6. [확인] 실행을 막은 병목은 **조건 확인된 disjoint clean pool 부재**다. [추정] 기존 후보들의 제한된 oracle 여지는 후보 생성/표현 병목도 시사하지만 원인 확정은 아니다.
7. [확인] 지금 E3/E5/representation 신규 학습으로 넘어가지 않는다. 먼저 평가와 분리된 clean 후보≥8장의 조건을 확인하거나 데이터 역할 변경에 대한 별도 지시가 필요하다. 새 코너 좌표 annotation이 필수라는 뜻은 아니다.

## 산출물·한계

- [확인] [후보별 전체 수치](E1_CANDIDATE_HEADROOM.md), [기하 검사](E1_GEOMETRY_RECOVERABILITY.md), [pool 감사](E1_PSEUDOLABEL_POOL.md), [갤러리](../../../outputs/pallet_occlusion_refiner_transfer_v1/index.html).
- [확인] 갤러리는 모집단별 개선10/악화 또는 최소개선10/seed1 랜덤10이다. clean8은 각8장만 제공한다. GT는 그림과 사후 점수에만 쓰며 좋은 사례만 보고하지 않는다.
- [확인] CLEAN_NONCAD69는 역사적 이름이며 external occlusion 없음 집합이다. truncation까지 없다는 뜻은 아니며 E1_CONDITION_SUBGROUPS에 분리 집계했다.
- [확인] 재사용 DEV·점별 unknown/PnP GT·후보 누락·독립 paired physical occlusion GT 부재가 한계다. 통계적 유의성 주장은 하지 않는다.
- [확인] GPU 추론은 RTX3080에서 수행, 관측 최대58°C. 기존 모델·annotation·split·논문표 변경 없음. 기존 CuDNN workaround 경고는 있었으나 실행 완료했으며 패키지/시스템 변경 없음.
