# N3만·SubPix만의 실제 B/I 과정

PRIMARY에서 N3_BI/SubPix_BI와 결합 N1의 성공 여부가 다른 네 고정 범주의 ID순 첫 사례를 사용했다. 중복을 합쳤고 없는 범주는 대체하지 않았다. 각 초기화는 자기 상태·마스크·B/I와 PnP를 사용한다.

모든 그림은 같은 전체 원영상 범위이며 S/O/T/U 실제 측정 마커와 정답 선은 없다. 화면 밖 모델은 원영상 경계에서 잘라 표시한다. JPG는 quality90 표시본이고 원본RGB는 바이트 그대로 복사했다.

없는 범주: 없음

## plastic_day_01:010995

N3_BI만 성공 / N1 실패

[개별 HTML](case00/index.html) · [실제 과정/좌표](case00/PROCESS.json)

| 경로 | 위치·회전 오차 |
|---|---|
| N3_F | T 5.648cm / R 1.923° · 실패 |
| N3_BI | T 2.337cm / R 0.368° · 성공 |
| SUBPIX_F | T 6.321cm / R 83.235° · 실패 |
| SUBPIX_BI | T 6.814cm / R 83.989° · 실패 |
| N1 | T 6.821cm / R 84.001° · 실패 |

![N3_ONLY_01_initial](case00/N3_ONLY_01_initial.jpg)
![N3_ONLY_02_states](case00/N3_ONLY_02_states.jpg)
![N3_ONLY_03_BI_observations](case00/N3_ONLY_03_BI_observations.jpg)
![N3_ONLY_04_final](case00/N3_ONLY_04_final.jpg)
![SUBPIX_ONLY_01_initial](case00/SUBPIX_ONLY_01_initial.jpg)
![SUBPIX_ONLY_02_states](case00/SUBPIX_ONLY_02_states.jpg)
![SUBPIX_ONLY_03_BI_observations](case00/SUBPIX_ONLY_03_BI_observations.jpg)
![SUBPIX_ONLY_04_final](case00/SUBPIX_ONLY_04_final.jpg)

## eval_noapril:1775201414728017664

N3_BI실패 / N1만 성공

[개별 HTML](case01/index.html) · [실제 과정/좌표](case01/PROCESS.json)

| 경로 | 위치·회전 오차 |
|---|---|
| N3_F | T 6.577cm / R 1.319° · 실패 |
| N3_BI | T 5.215cm / R 0.776° · 실패 |
| SUBPIX_F | T 3.876cm / R 1.914° · 성공 |
| SUBPIX_BI | T 3.645cm / R 0.734° · 성공 |
| N1 | T 3.209cm / R 0.699° · 성공 |

![N3_ONLY_01_initial](case01/N3_ONLY_01_initial.jpg)
![N3_ONLY_02_states](case01/N3_ONLY_02_states.jpg)
![N3_ONLY_03_BI_observations](case01/N3_ONLY_03_BI_observations.jpg)
![N3_ONLY_04_final](case01/N3_ONLY_04_final.jpg)
![SUBPIX_ONLY_01_initial](case01/SUBPIX_ONLY_01_initial.jpg)
![SUBPIX_ONLY_02_states](case01/SUBPIX_ONLY_02_states.jpg)
![SUBPIX_ONLY_03_BI_observations](case01/SUBPIX_ONLY_03_BI_observations.jpg)
![SUBPIX_ONLY_04_final](case01/SUBPIX_ONLY_04_final.jpg)

## plastic_night_01:036949

SUBPIX_BI만 성공 / N1 실패

[개별 HTML](case02/index.html) · [실제 과정/좌표](case02/PROCESS.json)

| 경로 | 위치·회전 오차 |
|---|---|
| N3_F | T 7.453cm / R 1.807° · 실패 |
| N3_BI | T 7.202cm / R 2.220° · 실패 |
| SUBPIX_F | T 4.297cm / R 1.921° · 성공 |
| SUBPIX_BI | T 4.178cm / R 2.156° · 성공 |
| N1 | T 5.429cm / R 1.741° · 실패 |

![N3_ONLY_01_initial](case02/N3_ONLY_01_initial.jpg)
![N3_ONLY_02_states](case02/N3_ONLY_02_states.jpg)
![N3_ONLY_03_BI_observations](case02/N3_ONLY_03_BI_observations.jpg)
![N3_ONLY_04_final](case02/N3_ONLY_04_final.jpg)
![SUBPIX_ONLY_01_initial](case02/SUBPIX_ONLY_01_initial.jpg)
![SUBPIX_ONLY_02_states](case02/SUBPIX_ONLY_02_states.jpg)
![SUBPIX_ONLY_03_BI_observations](case02/SUBPIX_ONLY_03_BI_observations.jpg)
![SUBPIX_ONLY_04_final](case02/SUBPIX_ONLY_04_final.jpg)

## eval_noapril:1775201415399297536

SUBPIX_BI실패 / N1만 성공

[개별 HTML](case03/index.html) · [실제 과정/좌표](case03/PROCESS.json)

| 경로 | 위치·회전 오차 |
|---|---|
| N3_F | T 7.009cm / R 1.276° · 실패 |
| N3_BI | T 4.450cm / R 0.814° · 성공 |
| SUBPIX_F | T 10.335cm / R 1.780° · 실패 |
| SUBPIX_BI | T 5.994cm / R 1.026° · 실패 |
| N1 | T 2.672cm / R 0.740° · 성공 |

![N3_ONLY_01_initial](case03/N3_ONLY_01_initial.jpg)
![N3_ONLY_02_states](case03/N3_ONLY_02_states.jpg)
![N3_ONLY_03_BI_observations](case03/N3_ONLY_03_BI_observations.jpg)
![N3_ONLY_04_final](case03/N3_ONLY_04_final.jpg)
![SUBPIX_ONLY_01_initial](case03/SUBPIX_ONLY_01_initial.jpg)
![SUBPIX_ONLY_02_states](case03/SUBPIX_ONLY_02_states.jpg)
![SUBPIX_ONLY_03_BI_observations](case03/SUBPIX_ONLY_03_BI_observations.jpg)
![SUBPIX_ONLY_04_final](case03/SUBPIX_ONLY_04_final.jpg)
