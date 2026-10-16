# 후속 방법 정리용 실패 원인 조사

기존 봉인 결과를 읽은 개발용 조사다. 새 모델·마스크·이미지 관측·PnP solve·학습·정답 변경은 수행하지 않았다. 모든 판정 특징은 예측만으로 먼저 계산해 FEATURE_SEAL.json에 봉인한 뒤 기존 점수와 별도 결합했다. PRIMARY가 이미 개발에 노출되었으므로 이후 결과를 독립 검증으로 부르지 않는다.

## 확인한 원인

- PRIMARY R1의 5손상, N1의 14손상은 모두 기존 solver accepted=True이다. accepted는 관측/기하 제약 만족이며 정확도 보증이 아니다.
- 모든 손상은 양의 최소 깊이를 가지며, visible Jacobian rank=6이다. 이 두 조건만으로는 손상을 구분하지 못한다.
- R1 손상 3/5, N1 손상 3/14는 B/I 3점이다. 3점의 6개 영상좌표를 정확히 맞춘 목적함수 0의 P3P 해도 서로 다른 자세가 될 수 있다. R1 2/3, N1 3/3은 기존 ambiguity flag가 이를 검출했다. rank=6은 국소 민감도일 뿐 전역 유일성 증거가 아니다.
- R1 회복 21건, N1 회복 14건은 모두 B/I 4점 이상이며 ambiguity 검출은 0건이다. 이 사실은 원인 조사 결과이며 사전 독립 증거가 아니다.
- R1 손상 3/5와 N1 손상 5/14는 초기 F가 같은 보정 관측의 기존 Mahalanobis≤3, 기하 2°, canvas, depth guard를 이미 통과했다. 그러나 회복 R1 11/21, N1 9/14도 그러므로 초기 admissibility만으로 갱신을 모두 막으면 이득도 버린다.
- B/I 4~6점 손상도 존재한다. 작은 좌표 보정, U 점 제외, B=2px/I=3px fallback과 선 교점 약1.5px 불확실도의 재가중으로 초기 F와 다른 절충점을 만든다. 큰 선 오차만의 문제라고 결론내릴 수 없다.

## 가장 분명한 실제 사례

N1 wood_night_01:029428: 3개의 B 관측 이동은 0.06/0.17/0.18px뿐이다. 초기 F는 같은 관측 guard를 전부 통과하며 최대 Mahalanobis=0.599다. 그래도 3점 재PnP가 목적함수 0인 다른 해를 선택해 초기 대비 이동30.9cm, 회전45.65°가 된다. 정답 오차는 초기4.00cm/3.75° → 보정26.87cm/42.61°다. 3점 재풀이 과정의 해 선택이 직접적인 문제다.

## 고정 후보 최대 2개

A. AMBIGUITY_VETO: 기존 solution.ambiguity.detected 또는 local_underdetermined 또는 pose_family_from_two_BI이면 자기 초기 F를 유지한다. 임계값 추가 없이 기존 ambiguity 설정(목적함수 gap0.05, 회전0.25°, 이동/스케일0.001)을 그대로 사용한다. B/I 3점인데 ambiguity=False인 R1 wood_183705:000706은 이 규칙만으로 보호되지 않는다. 수학적 중복 관측을 요구해 B/I≥4를 추가할지는 CAL 평가 전 방법 정의로 고정해야 한다.

B. ADMISSIBILITY_FIRST: A에 더해 자기 초기 F가 기존 Mahalanobis3·기하2°·canvas·positive_depth guard를 모두 만족하면 갱신하지 않는다. 새로운 수치 임계값은 없다. 이득을 버릴 위험도 커 CAL46 선택 후 동결된 PRIMARY192 개발 진단으로 비교해야 한다.

GT 점수로 후보를 고르거나 사례별 경로를 선택해서는 안 된다. 추가 threshold 탐색/세 번째 후보는 제안하지 않는다. 기본 방법을 바꾸지 않는다.

## 전체 손상 목록

| 경로 | ID | B/I 수 | 이동관측 수 | 초기F admissible | ambiguity | 초기→최종 T cm | 초기→최종 R ° |
|---|---|---:|---:|---|---|---|---|
| R1 | eval_pallet07:1778652170735118080 | 3 | 3 | False | True | 2.42 → 44.89 | 1.38 → 4.57 |
| N1 | eval_pallet07:1778652170735118080 | 3 | 2 | False | True | 2.46 → 16.22 | 1.07 → 4.96 |
| N1 | wood_183705:000586 | 4 | 3 | False | False | 0.74 → 5.79 | 1.00 → 8.30 |
| R1 | wood_183705:000706 | 3 | 1 | False | False | 3.14 → 8.76 | 4.66 → 13.13 |
| N1 | wood_183705:000706 | 3 | 1 | False | True | 2.77 → 9.42 | 4.02 → 102.05 |
| R1 | plastic_day_01:010207 | 5 | 2 | True | False | 1.99 → 10.64 | 0.35 → 1.60 |
| R1 | plastic_day_01:011067 | 5 | 2 | True | False | 3.22 → 12.09 | 0.92 → 1.30 |
| N1 | plastic_day_01:012545 | 6 | 2 | False | False | 3.55 → 11.62 | 0.70 → 3.08 |
| N1 | plastic_night_01:038630 | 4 | 3 | True | False | 3.85 → 12.81 | 2.75 → 0.92 |
| N1 | plastic_night_01:039268 | 5 | 3 | False | False | 2.87 → 7.15 | 2.04 → 2.35 |
| N1 | plastic_night_01:040448 | 5 | 4 | False | False | 4.30 → 6.85 | 0.69 → 1.90 |
| N1 | wood_day_01:004481 | 4 | 3 | False | False | 1.51 → 5.21 | 0.45 → 2.40 |
| N1 | wood_day_01:004799 | 4 | 1 | True | False | 3.68 → 5.66 | 3.61 → 1.85 |
| N1 | wood_day_01:026673 | 5 | 3 | False | False | 1.37 → 6.91 | 4.20 → 4.31 |
| R1 | wood_night_01:029428 | 3 | 3 | True | True | 1.86 → 2.67 | 2.25 → 41.83 |
| N1 | wood_night_01:029428 | 3 | 3 | True | True | 4.00 → 26.87 | 3.75 → 42.61 |
| N1 | wood_night_01:030607 | 5 | 2 | True | False | 4.29 → 7.31 | 0.57 → 2.38 |
| N1 | wood_night_01:030825 | 5 | 3 | False | False | 0.19 → 7.75 | 2.49 → 2.86 |
| N1 | wood_night_01:031853 | 4 | 4 | True | False | 4.84 → 7.06 | 3.36 → 1.92 |

## 파일

- PREDICTION_ONLY_FEATURES.json: 476행, 상태·관측 이동·공분산·초기/최종 재투영 잔차·guard·라인 support·시작/ambiguity. GT 미포함.
- FEATURE_SEAL.json: 기존 예측·관측 입력 SHA와 특징파일 SHA.
- SCORE_JOINED_FORENSICS.json: 기존 점수와 별도 결합한 개발 조사 파일.
- audit.py: 특징 계산 및 별도 점수 결합 재현.
