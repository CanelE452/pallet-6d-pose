# 상태별 B/I 검증 과정 갤러리

실제 저장된 초기 입력·자기 상태/영상 측정·B/I PnP 제안·A/B 최종 결과를 같은 전체 원영상 위에 표시했다. 정답 선은 그리지 않았다. 화면 밖 모델은 원영상 경계에서 잘라 표시한다.

CAL 선택 경로: **EB**. 사례 선택은 사후 진단이며 추론과 경로 선택에 영향이 없다.

존재하지 않는 고정 범주: 없음

## wood_night_01:029428

명시된 P3P 실패 사례

[개별 HTML](case00/index.html) · [실제 과정/좌표](case00/PROCESS.json)

| 경로 | 위치·회전 오차 | 교체·유지 사유 |
|---|---|---|
| E0 | T 8.997cm / R 11.044° · 실패 | Exact independently registered frozen E pose |
| E1 | T 1.460cm / R 21.205° · 실패 | None |
| EA | T 8.997cm / R 11.044° · 실패 | rule_A_requirements_failed |
| EB | T 8.997cm / R 11.044° · 실패 | rule_A_requirements_failed |

![01_own_initial](case00/01_own_initial.png)
![02_states_BI_rays](case00/02_states_BI_rays.png)
![03_unguarded_BI_proposal](case00/03_unguarded_BI_proposal.png)
![04_guarded_final](case00/04_guarded_final.png)

## wood_night_01:031344

CAL 선택 경로의 PRIMARY 최대 위치오차

[개별 HTML](case01/index.html) · [실제 과정/좌표](case01/PROCESS.json)

| 경로 | 위치·회전 오차 | 교체·유지 사유 |
|---|---|---|
| E0 | T 260.412cm / R 65.519° · 실패 | Exact independently registered frozen E pose |
| E1 | T 168.368cm / R 86.550° · 실패 | None |
| EA | T 260.412cm / R 65.519° · 실패 | rule_A_requirements_failed |
| EB | T 260.412cm / R 65.519° · 실패 | rule_A_requirements_failed |

![01_own_initial](case01/01_own_initial.png)
![02_states_BI_rays](case01/02_states_BI_rays.png)
![03_unguarded_BI_proposal](case01/03_unguarded_BI_proposal.png)
![04_guarded_final](case01/04_guarded_final.png)

## plastic_day_01:007881

EA vs E0 회복 · ID순 첫 사례

[개별 HTML](case02/index.html) · [실제 과정/좌표](case02/PROCESS.json)

| 경로 | 위치·회전 오차 | 교체·유지 사유 |
|---|---|---|
| E0 | T 5.343cm / R 0.959° · 실패 | Exact independently registered frozen E pose |
| E1 | T 4.874cm / R 1.038° · 성공 | None |
| EA | T 4.874cm / R 1.038° · 성공 | rule_A_requirements_passed |
| EB | T 5.343cm / R 0.959° · 실패 | rule_B_observed_ray_worsened |

![01_own_initial](case02/01_own_initial.png)
![02_states_BI_rays](case02/02_states_BI_rays.png)
![03_unguarded_BI_proposal](case02/03_unguarded_BI_proposal.png)
![04_guarded_final](case02/04_guarded_final.png)

## eval_noapril:1775201430551447040

EA vs E0 손상 · ID순 첫 사례

[개별 HTML](case03/index.html) · [실제 과정/좌표](case03/PROCESS.json)

| 경로 | 위치·회전 오차 | 교체·유지 사유 |
|---|---|---|
| E0 | T 4.069cm / R 0.354° · 성공 | Exact independently registered frozen E pose |
| E1 | T 7.626cm / R 1.091° · 실패 | None |
| EA | T 7.626cm / R 1.091° · 실패 | rule_A_requirements_passed |
| EB | T 4.069cm / R 0.354° · 성공 | rule_B_observed_ray_worsened |

![01_own_initial](case03/01_own_initial.png)
![02_states_BI_rays](case03/02_states_BI_rays.png)
![03_unguarded_BI_proposal](case03/03_unguarded_BI_proposal.png)
![04_guarded_final](case03/04_guarded_final.png)

## plastic_night_01:039093

EB vs E0 회복 · ID순 첫 사례

[개별 HTML](case04/index.html) · [실제 과정/좌표](case04/PROCESS.json)

| 경로 | 위치·회전 오차 | 교체·유지 사유 |
|---|---|---|
| E0 | T 16.861cm / R 84.893° · 실패 | Exact independently registered frozen E pose |
| E1 | T 2.170cm / R 1.040° · 성공 | None |
| EA | T 2.170cm / R 1.040° · 성공 | rule_A_requirements_passed |
| EB | T 2.170cm / R 1.040° · 성공 | rule_B_requirements_passed |

![01_own_initial](case04/01_own_initial.png)
![02_states_BI_rays](case04/02_states_BI_rays.png)
![03_unguarded_BI_proposal](case04/03_unguarded_BI_proposal.png)
![04_guarded_final](case04/04_guarded_final.png)

## plastic_day_01:010995

EB vs E0 손상 · ID순 첫 사례

[개별 HTML](case05/index.html) · [실제 과정/좌표](case05/PROCESS.json)

| 경로 | 위치·회전 오차 | 교체·유지 사유 |
|---|---|---|
| E0 | T 0.761cm / R 0.250° · 성공 | Exact independently registered frozen E pose |
| E1 | T 6.663cm / R 84.136° · 실패 | None |
| EA | T 6.663cm / R 84.136° · 실패 | rule_A_requirements_passed |
| EB | T 6.663cm / R 84.136° · 실패 | rule_B_requirements_passed |

![01_own_initial](case05/01_own_initial.png)
![02_states_BI_rays](case05/02_states_BI_rays.png)
![03_unguarded_BI_proposal](case05/03_unguarded_BI_proposal.png)
![04_guarded_final](case05/04_guarded_final.png)

## wood_night_01:030652

NB vs N0 회복 · ID순 첫 사례

[개별 HTML](case06/index.html) · [실제 과정/좌표](case06/PROCESS.json)

| 경로 | 위치·회전 오차 | 교체·유지 사유 |
|---|---|---|
| N0 | T 1.500cm / R 5.548° · 실패 | Unchanged own common F; no original numerical-acceptance flag available |
| N1 | T 1.867cm / R 2.131° · 성공 | None |
| NA | T 1.867cm / R 2.131° · 성공 | rule_A_requirements_passed |
| NB | T 1.867cm / R 2.131° · 성공 | rule_B_requirements_passed |

![01_own_initial](case06/01_own_initial.png)
![02_states_BI_rays](case06/02_states_BI_rays.png)
![03_unguarded_BI_proposal](case06/03_unguarded_BI_proposal.png)
![04_guarded_final](case06/04_guarded_final.png)

## plastic_night_01:039268

NB vs N0 손상 · ID순 첫 사례

[개별 HTML](case07/index.html) · [실제 과정/좌표](case07/PROCESS.json)

| 경로 | 위치·회전 오차 | 교체·유지 사유 |
|---|---|---|
| N0 | T 2.867cm / R 2.044° · 성공 | Unchanged own common F; no original numerical-acceptance flag available |
| N1 | T 7.149cm / R 2.349° · 실패 | None |
| NA | T 7.149cm / R 2.349° · 실패 | rule_A_requirements_passed |
| NB | T 7.149cm / R 2.349° · 실패 | rule_B_requirements_passed |

![01_own_initial](case07/01_own_initial.png)
![02_states_BI_rays](case07/02_states_BI_rays.png)
![03_unguarded_BI_proposal](case07/03_unguarded_BI_proposal.png)
![04_guarded_final](case07/04_guarded_final.png)
