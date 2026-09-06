# SOLVER SWAP v1 — SQPnP+RefineLM vs 미분가능 Gauss-Newton PnP

생성 2026-09-05T16:00:16.215476+00:00 · 학습 0 step · 새 추론 0 회 · 76.1s

GATE 0 (S0 이 정본 POSE_EVALUATION_{ARM}.json 재현) = **PASS** — 따라서 아래 D arm 수치는 solver 만의 차이다.

축 가설 선택기는 모든 arm 에서 SQPnP 기반으로 고정했다. 바뀐 것은 pose read-out 뿐이다.

## R0  (짝지은 프레임 n=319)

| solver | rot med° | yaw med° | trans med cm | IoU3D | ADD AUC |
|---|---:|---:|---:|---:|---:|
| S0_SQPNP_LM | 2.262 | 1.231 | 7.897 | 0.6032 | 0.4285 |
| D1_GN_LS | 2.296 | 1.241 | 7.897 | 0.6000 | 0.4272 |
| D2_GN_HUBER | 2.296 | 1.302 | 7.897 | 0.6000 | 0.4281 |
| D3_SQPNP_GN | 2.262 | 1.231 | 7.897 | 0.6032 | 0.4285 |
| D4_GN_HUBER_CONF | 2.296 | 1.285 | 7.897 | 0.6000 | 0.4279 |

## R0_CONT  (짝지은 프레임 n=318)

| solver | rot med° | yaw med° | trans med cm | IoU3D | ADD AUC |
|---|---:|---:|---:|---:|---:|
| S0_SQPNP_LM | 2.499 | 1.261 | 8.544 | 0.5943 | 0.4091 |
| D1_GN_LS | 2.507 | 1.295 | 8.544 | 0.5933 | 0.4079 |
| D2_GN_HUBER | 2.538 | 1.316 | 8.544 | 0.5928 | 0.4085 |
| D3_SQPNP_GN | 2.499 | 1.261 | 8.544 | 0.5943 | 0.4091 |
| D4_GN_HUBER_CONF | 2.539 | 1.316 | 8.444 | 0.5928 | 0.4083 |

## R1_NAIVE  (짝지은 프레임 n=319)

| solver | rot med° | yaw med° | trans med cm | IoU3D | ADD AUC |
|---|---:|---:|---:|---:|---:|
| S0_SQPNP_LM | 2.372 | 1.245 | 7.796 | 0.5897 | 0.4204 |
| D1_GN_LS | 2.376 | 1.268 | 7.796 | 0.5889 | 0.4192 |
| D2_GN_HUBER | 2.401 | 1.236 | 7.796 | 0.5870 | 0.4207 |
| D3_SQPNP_GN | 2.372 | 1.245 | 7.796 | 0.5897 | 0.4204 |
| D4_GN_HUBER_CONF | 2.398 | 1.236 | 7.848 | 0.5869 | 0.4191 |

## R2_CONF  (짝지은 프레임 n=319)

| solver | rot med° | yaw med° | trans med cm | IoU3D | ADD AUC |
|---|---:|---:|---:|---:|---:|
| S0_SQPNP_LM | 2.477 | 1.326 | 7.785 | 0.5995 | 0.4158 |
| D1_GN_LS | 2.477 | 1.326 | 7.785 | 0.5979 | 0.4144 |
| D2_GN_HUBER | 2.579 | 1.341 | 7.532 | 0.5979 | 0.4161 |
| D3_SQPNP_GN | 2.477 | 1.326 | 7.785 | 0.5995 | 0.4158 |
| D4_GN_HUBER_CONF | 2.534 | 1.341 | 7.664 | 0.5969 | 0.4136 |

## R3_CONF_REPROJ  (짝지은 프레임 n=319)

| solver | rot med° | yaw med° | trans med cm | IoU3D | ADD AUC |
|---|---:|---:|---:|---:|---:|
| S0_SQPNP_LM | 2.349 | 1.207 | 7.737 | 0.5998 | 0.4149 |
| D1_GN_LS | 2.349 | 1.207 | 7.737 | 0.5998 | 0.4141 |
| D2_GN_HUBER | 2.414 | 1.295 | 7.652 | 0.5998 | 0.4156 |
| D3_SQPNP_GN | 2.349 | 1.207 | 7.737 | 0.5998 | 0.4149 |
| D4_GN_HUBER_CONF | 2.419 | 1.302 | 7.595 | 0.5997 | 0.4143 |

## R4_CONF_REMOVE  (짝지은 프레임 n=319)

| solver | rot med° | yaw med° | trans med cm | IoU3D | ADD AUC |
|---|---:|---:|---:|---:|---:|
| S0_SQPNP_LM | 2.524 | 1.276 | 8.055 | 0.5997 | 0.4120 |
| D1_GN_LS | 2.524 | 1.276 | 8.055 | 0.5997 | 0.4120 |
| D2_GN_HUBER | 2.629 | 1.282 | 8.011 | 0.5997 | 0.4135 |
| D3_SQPNP_GN | 2.524 | 1.276 | 8.055 | 0.5997 | 0.4120 |
| D4_GN_HUBER_CONF | 2.551 | 1.276 | 7.930 | 0.5995 | 0.4126 |

## R5_PROPOSED  (짝지은 프레임 n=319)

| solver | rot med° | yaw med° | trans med cm | IoU3D | ADD AUC |
|---|---:|---:|---:|---:|---:|
| S0_SQPNP_LM | 2.535 | 1.294 | 8.827 | 0.5868 | 0.4001 |
| D1_GN_LS | 2.535 | 1.294 | 8.827 | 0.5846 | 0.3983 |
| D2_GN_HUBER | 2.535 | 1.302 | 8.827 | 0.5846 | 0.4008 |
| D3_SQPNP_GN | 2.535 | 1.294 | 8.827 | 0.5868 | 0.4001 |
| D4_GN_HUBER_CONF | 2.534 | 1.309 | 8.690 | 0.5816 | 0.4006 |

## 사전등록 판정

사전등록 규칙은 개선을 부등호로만 정의한다. 그래서 baseline 과 사실상 같은 solver 도 ACCEPT 가 될 수 있다. 마지막 두 칸이 그 구분이다.

| solver | 판정 | 허용범위 밖 악화 없음 | 개선 지표 수 (R0+R5, 8칸) | 최대 상대변화 | 실질 |
|---|---|---|---:|---:|---|
| D1_GN_LS | **REJECT** | False | 3 | 1.48e-02 | REAL_DIFFERENCE |
| D2_GN_HUBER | **REJECT** | False | 4 | 1.48e-02 | REAL_DIFFERENCE |
| D3_SQPNP_GN | **ACCEPT_SOLVER_SWAP** | True | 5 | 3.67e-08 | NO_CHANGE |
| D4_GN_HUBER_CONF | **REJECT** | False | 4 | 1.54e-02 | REAL_DIFFERENCE |

## R0 짝별 개선/악화

| solver | rotation 개선/악화 | translation 개선/악화 |
|---|---|---|
| D1_GN_LS | 166/153 | 140/179 |
| D2_GN_HUBER | 171/148 | 142/177 |
| D3_SQPNP_GN | 167/152 | 140/179 |
| D4_GN_HUBER_CONF | 151/168 | 165/154 |

## solver health

| solver | 풀린 프레임 | init/guard fallback |
|---|---:|---:|
| D1_GN_LS | 2232 | 0 |
| D2_GN_HUBER | 2232 | 0 |
| D3_SQPNP_GN | 2232 | 0 |
| D4_GN_HUBER_CONF | 2232 | 0 |

> PAPER_EVAL 319장은 반복 사용된 development set. 어떤 결과도 held-out/final/SOTA 로 부르지 않는다.
