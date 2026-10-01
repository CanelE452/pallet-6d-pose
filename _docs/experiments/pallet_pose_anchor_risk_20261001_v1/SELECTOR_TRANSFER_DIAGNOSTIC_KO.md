# 고정 selector의 source→실사 전환 진단

source45/45 통과 뒤 실사 원래 조건과 개입 대조군을 합친5개 항목 중2개만 통과했다. 새 성능이나 선택 규칙을 시험하지 않고 실제 고정692개 선택과 기존519개 anchored oracle 행을 대조했다. raw GT 재열람·metric 재계산·이미지 forward·PnP·fit은 모두0회다. 이미 저장된 GT 유래 오차를 쓰는 사후 진단이며 배포 정책이 아니다.

| 모집단 | 모델 | anchor | safe 개선 | 동오차 non-anchor | T만 악화 | R만 악화 | 양축 악화 | oracle 개선 기회 | 놓친 기회 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NATURAL99 | R0_ONLY | 84 | 5 | 0 | 4 | 1 | 5 | NA | NA |
| NATURAL99 | UNION_s1 | 93 | 3 | 0 | 1 | 1 | 1 | 51 | 48 |
| NATURAL99 | UNION_s2 | 90 | 4 | 0 | 1 | 4 | 0 | 51 | 47 |
| NATURAL99 | UNION_s3 | 89 | 6 | 0 | 1 | 2 | 1 | 53 | 47 |
| CLEAN29 | R0_ONLY | 29 | 0 | 0 | 0 | 0 | 0 | NA | NA |
| CLEAN29 | UNION_s1 | 26 | 0 | 0 | 2 | 0 | 1 | 8 | 8 |
| CLEAN29 | UNION_s2 | 26 | 1 | 0 | 1 | 0 | 1 | 9 | 8 |
| CLEAN29 | UNION_s3 | 24 | 2 | 0 | 1 | 2 | 0 | 12 | 10 |
| WOOD45 | R0_ONLY | 45 | 0 | 0 | 0 | 0 | 0 | NA | NA |
| WOOD45 | UNION_s1 | 45 | 0 | 0 | 0 | 0 | 0 | 14 | 14 |
| WOOD45 | UNION_s2 | 45 | 0 | 0 | 0 | 0 | 0 | 13 | 13 |
| WOOD45 | UNION_s3 | 43 | 1 | 0 | 0 | 0 | 1 | 17 | 16 |

Anchor는 기존 R0 operational GEO의 같은 후보 identity다. Safe 개선은 두 오차 모두 anchor 이하이면서 한 축 이상 엄격히 작을 때다. R0_ONLY는2후보이므로4후보 UNION oracle 기회 회수율을 적용하지 않는다. JSON은 모든 모집단·recording·seed별 집계와692행을 제공한다.

## 후보·expert·W/D 전환

| 모집단 | UNION seed | 실제 refiner 선택 | 실제 W/D 전환 | oracle refiner 선택 | oracle W/D 전환 | 정확 oracle identity 일치 |
|---|---:|---:|---:|---:|---:|---:|
| NATURAL99 | 1 | 6 | 0 | 43 | 25 | 47 |
| NATURAL99 | 2 | 9 | 0 | 42 | 25 | 46 |
| NATURAL99 | 3 | 10 | 0 | 45 | 26 | 48 |
| CLEAN29 | 1 | 3 | 0 | 8 | 1 | 18 |
| CLEAN29 | 2 | 3 | 0 | 9 | 1 | 19 |
| CLEAN29 | 3 | 5 | 0 | 12 | 1 | 16 |
| WOOD45 | 1 | 0 | 0 | 12 | 2 | 31 |
| WOOD45 | 2 | 0 | 0 | 11 | 2 | 32 |
| WOOD45 | 3 | 2 | 0 | 15 | 2 | 28 |

## NATURAL99 recording별 놓친 기회

| recording | seed | frame 수 | anchor | safe 개선 | unsafe | oracle 기회 | 놓친 기회 |
|---|---:|---:|---:|---:|---:|---:|---:|
| REC_007 | 1 | 33 | 31 | 2 | 0 | 21 | 19 |
| REC_007 | 2 | 33 | 28 | 2 | 3 | 20 | 18 |
| REC_007 | 3 | 33 | 30 | 2 | 1 | 22 | 20 |
| REC_021 | 1 | 2 | 2 | 0 | 0 | 0 | 0 |
| REC_021 | 2 | 2 | 2 | 0 | 0 | 0 | 0 |
| REC_021 | 3 | 2 | 1 | 1 | 0 | 2 | 1 |
| REC_022 | 1 | 16 | 14 | 0 | 2 | 6 | 6 |
| REC_022 | 2 | 16 | 14 | 1 | 1 | 5 | 4 |
| REC_022 | 3 | 16 | 12 | 2 | 2 | 5 | 3 |
| REC_025 | 1 | 27 | 25 | 1 | 1 | 13 | 12 |
| REC_025 | 2 | 27 | 26 | 0 | 1 | 15 | 15 |
| REC_025 | 3 | 27 | 25 | 1 | 1 | 15 | 14 |
| REC_027 | 1 | 12 | 12 | 0 | 0 | 5 | 5 |
| REC_027 | 2 | 12 | 11 | 1 | 0 | 6 | 5 |
| REC_027 | 3 | 12 | 12 | 0 | 0 | 4 | 4 |
| REC_041 | 1 | 9 | 9 | 0 | 0 | 6 | 6 |
| REC_041 | 2 | 9 | 9 | 0 | 0 | 5 | 5 |
| REC_041 | 3 | 9 | 9 | 0 | 0 | 5 | 5 |

## TRAIN과 실사에서 관측한 차이

| 모델 | TRAIN target anchor/유효행 | TRAIN 선택 anchor/유효행 | NATURAL99 oracle anchor/frame | NATURAL99 선택 anchor/frame |
|---|---:|---:|---:|---:|
| R0_ONLY | 2454/2597 | 2575/2597 | NA/99 | 84/99 |
| UNION_s1 | 1812/2597 | 2452/2597 | 48/99 | 93/99 |
| UNION_s2 | 1817/2597 | 2457/2597 | 48/99 | 90/99 |
| UNION_s3 | 1835/2597 | 2498/2597 | 46/99 | 89/99 |

전체 TRAIN 분모는2,598행이고 실패1행을 유지했다. 위 anchor 비율만 정의 가능한2,597행을 표시한다. 실사173행은 모두 available이었다. 실제 선택/오차의 exact tie는 JSON에 별도 집계하며 임계값은 사용하지 않았다.

**관측:** natural에서 UNION은 대부분 anchor에 머물며 oracle의 pointwise-safe 개선 기회를 대부분 놓친다. 이번 실사 실패 항목은 공동 중앙값·불확실성·recording 민감도이고 tail/clean 보호 항목은 통과했다. 따라서 이번 결과를 많은 교정으로 tail이 무너진 실패라고 요약하면 틀리다. R0_ONLY의 W/D 교환에는 실제 악화도 있어 단순히 가설 전환을 늘리는 정책 역시 근거가 없다.

**추정과 한계:** TRAIN에서도 target보다 anchor를 더 자주 선택하며, 실사 oracle의 개선 기회 구성도 TRAIN과 다르다. 이는 보수적 supervision/의사결정과 domain shift를 검토할 근거다. 그러나 이 집계만으로 class 빈도, risk margin, 기하 특징, RGB 보정 품질 중 무엇이 원인인지 분리할 수 없다. 189특징의 표현력 부족이나 합성학습의 불가능성을 확정하지 않는다. source45는 해당 고정 합성 VAL의 비교 조건이며 실사 recording에 대한 보증이 아니었다.

## 다음 구조 변경 하나의 설계 제안 — 고정 RBF context64 추가

현재189특징은 후보 기하94·anchor 절댓값 차이94·identity1이며 직접 RGB appearance를 담지 않는다. 그 위의 공유 선형 결합만으로 서로 다른 기하·confidence·anchor 관계를 충분히 구분하는지와, 합성→실사 전이 또는 이미지 정보 부족인지는 현재 결과로 분리되지 않는다. 다음 한 가지 구조 실험은 기존189에 TRAIN에서 고정한 RBF64를 이어 붙인253차원 scorer다. 현재 target·risk margin·후보·원래 정규화·loss 계수·추론 argmin은 유지하고 고정 특징 map만 바꾼다.

센터 후보는 eligible TRAIN2,598행의 R0 및 세 frozen refiner의 유효 후보에서 얻은 기존189 context다. 반복된 R0 항목은(frame ID, expert, hypothesis) identity로 한 번만 세고, 이 identity의 SHA256 순으로 서로 다른 벡터64개를 고른다. label·오차·VAL·실사는 센터 선정에 쓰지 않는다. 양의 센터 쌍별 제곱거리의 median을 bandwidth 제곱값으로 한 번 고정하고 `exp(-||x-center||²/(2*bandwidth²))`64개를 추가한다. 센터64개나 양의 유한 bandwidth를 얻지 못하면 중단하며 다른개수·scale을 시험하지 않는다. 원래189 이후 추가 정규화는 없고 invalid 후보는253개 모두0으로 둔다.

이는 새 RGB 정보를 만드는 방법이 아니라 기존 특징들의 비선형 상호작용을 추가하는 최소 검증이다. 센터·bandwidth·특징을 target 품질 평가 전에 동결한다. 기존189weight 뒤에0을64개 붙인 score/objective 동등성을 확인하고, 동일한 margin CE+L2로 네 모델 각1회·zero init·같은solver 예산과 강볼록 인증을 적용한다. source45와 복구된 실사 원래/개입5개 AND 조건, 모든 seed·실패행·대조군은 유지한다. 센터 선정과 loss는 TRAIN만 사용하고, 런타임에는 기존 RGB에서 나온 후보기하·치수·K와 고정센터만 필요하며 GT나risk margin을 전달하지 않는다.

원래 gate 하나라도 실패하면 중단하며 threshold·margin배율·센터수·bandwidth sweep이나 좋은 seed 선택으로 구제하지 않는다. 이 구조가 TRAIN 적합을 개선해도 실사에서 RBF가 source support 밖으로 벗어나거나 직접RGB 단서가 부족할 수 있다. 따라서 실패는 모든 비선형 방법의 불가능성 증명이 아니고, 성공도 현재 표현력 부족만이 원인이었다는 인과 증명이 아니다.

[기존 방법 감사](../pallet_pose_union_selection_20261001_v1/PRIOR_AND_METHOD_AUDIT_KO.md)의 Stage4 관계기하+RGB context MLP 음성 결과, [objective 감사](../pallet_pose_selector_objective_audit_20261001_v1/PRIOR_OBJECTIVE_AUDIT_KO.md)의 utility·보존·soft-cost 실패, [pairwise 실패](../pallet_pose_selector_pairwise_20261001_v1/REPORT_KO.md)를 그대로 유지한다. 일반 비선형 선택기는 이미 시험됐으므로 발명이나 해결책으로 주장하지 않는다. 제한된 관련 코드 RBF/Nyström 검색에서는 현재 고정 pool·target·risk loss와 같은 이 구조의 실행을 찾지 못했다. 새 fit은 아직 실행하지 않았으며 별도 사전 protocol이 필요하다.

[전체 집계·692행·해시](SELECTOR_TRANSFER_DIAGNOSTIC.json) · [실사 고정 결과](REAL_RESULTS.json) · [TRAIN 검산](TRAIN_CONVERGENCE_KO.md) · [기존 oracle 진단](../pallet_pose_pareto_anchor_20261001_v1/REAL_FEASIBILITY.json)
