# 고정 세 사례의 저장 후보·비용 진단 (V32 R2)

SOURCE-C는 PRIMARY 192장에서 T≤5cm 120장·T≤5cm/R≤5° 117장으로 e의 115/114보다 늘었지만, TEST는 43/42이고 최대 T 오차는 191.313cm다. 기본 방법 e와 현재 관측·모델·선정 규칙은 변경하지 않았다.

이 진단은 전체 평가 봉인 후 ROOT가 지정한 세 실패 사례를 설명하기 위한 사후 분석이다. 사례 선정이나 오차는 방법 실행에 공급하지 않았다. e는 비교 예측이며 정답이 아니다. 새 참조·GT·마스크·모델·NN·TLS·PnP·최적화·후보 생성을 수행하지 않았다. 원 RGB 세 장은 바이트 그대로 복사해 직접 확인했다.

원 수치는 [EVIDENCE.json](EVIDENCE.json), 초기 후보와 native endpoint 보존은 [COARSE_TRACE_V1.json](COARSE_TRACE_V1.json)에 있다. 초기 coarse의 T/R은 ROOT가 이미 저장된 PROCESS_DATA만으로 계산한 [별도 사후 결과](../clean_fresh_BI_fixed_initial_pose_diagnosis_v4_32_v1/RESULT.json)를 인용했다. 본인이 평가 참조를 다시 열거나 T/R을 새로 계산한 값이 아니다.

표의 N은 native 영상 비용, O/S는 기존 coarse regularizer, C는 가중된 **전경 포함 비용**이다. C는 hard 영역 제약이 아니다. J=N+O+S+C이며 단위는 목적함수 값이다. ORIGINAL/SWAPPED는 서로 다른 W/D 대응 가설이다. `min joint`는 혼합 제약의 저장 최솟값으로, 전체를 픽셀 단위로 읽으면 안 된다.

## wood_night_01:031344

[원 RGB](wood_night_01__031344.png) · 상태 0→7: `I B B B B B S U` · old partial IDs `[0, 1, 3, 4, 5]` · S eligible `[6]` · O regions 0

ROOT의 별도 사후 초기 coarse 오차: **168.7714cm / 86.7259°**. 아래 모든 신규 arm은 accepted=True이며 finite fallback이 아닌 선택 결과다.

| 방법 | family | T cm | R ° | 새 선 | N | O | S | C | J | feasible 후보 | 후보 J 범위 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e | 부모 비교 | 260.4117 | 65.5187 | — | — | — | — | — | — | — | — |
| ALL-N | ORIGINAL | 193.5675 | 88.5720 | 1 | 6.021572 | 0.000000 | 1.847168 | 0.000000 | 7.868740 | 79 | [7.868740, 136.772780] |
| ALL-C | ORIGINAL | 176.2967 | 89.9003 | 1 | 6.269412 | 0.000000 | 1.830817 | 3.683908 | 11.784137 | 94 | [11.784137, 138.475558] |
| SOURCE-N | ORIGINAL | 199.9483 | 89.0042 | 1 | 5.769580 | 0.000000 | 1.849091 | 0.000000 | 7.618670 | 79 | [7.618670, 130.702628] |
| SOURCE-C | ORIGINAL | 179.9380 | 90.2655 | 1 | 6.028954 | 0.000000 | 1.832298 | 3.738689 | 11.599940 | 95 | [11.599940, 132.405406] |

**원 ORIGINAL coarse는 네 arm 모두 feasible 후보로 남았다.** 실제 selector 별칭은 `ORIGINAL_WD/OWN_INDEPENDENT_INITIAL_COARSE`다. 요청 당시의 `ORIGINAL_INITIAL` 토큰은 없다. R/t가 같은 저장 UNREFINED 레코드의 accepted=True와 아래 min joint만 확인했으며, 보관되지 않은 coarse 개별 guard 배열은 재구성하지 않았다.

| 방법 | 원 coarse N | 원 coarse O+S | 원 coarse C | 원 coarse J | coarse min joint | 선택 pose와 일치한 저장 refinement 이름 |
|---|---|---|---|---|---|---|
| ALL-N | 27.550409 | 0.105692 | 0.000000 | 27.656101 | 0.163977 | FRESH_SQPNP_12 |
| ALL-C | 27.550409 | 0.105692 | 1.443031 | 29.099132 | 0.163977 | FRESH_SQPNP_12 |
| SOURCE-N | 27.458631 | 0.105692 | 0.000000 | 27.564323 | 0.163977 | FRESH_SQPNP_12 |
| SOURCE-C | 27.458631 | 0.105692 | 1.443031 | 29.007354 | 0.163977 | FRESH_SQPNP_12 |

위 선택 pose들은 fresh SQPNP 시작의 연속 refinement 결과다. 각 선택 family의 시작 목록에는 별도의 `OWN_INITIAL_COARSE` 또는 `OWN_INDEPENDENT_INITIAL_COARSE`도 있다. 저장된 전체 seed/start R/t·모든 trial 경로를 다시 생성하거나 검증한 것은 아니다.

| 방법 | 활성 edge | 3D축 | owners | 실제 inlier | 실제 span px | 모델 길이 px | 선 각도차 ° | 실제 support min px |
|---|---|---|---|---|---|---|---|---|
| ALL-N | [0, 1] | X | 0,1 | 59 | 58.027 | 64.274 | 1.988 | 12.711 |
| ALL-C | [0, 1] | X | 0,1 | 59 | 58.027 | 65.539 | 1.877 | 13.729 |
| SOURCE-N | [0, 1] | X | 0,1 | 59 | 58.027 | 63.712 | 2.313 | 12.176 |
| SOURCE-C | [0, 1] | X | 0,1 | 59 | 58.027 | 65.182 | 2.179 | 13.388 |

확인된 구조: 새 선은 0–1 한 개이고 owner0은 SELF 실제 support, owner1은 같은 named edge의 MODEL_CONSTRAINT_ONLY다. 59개 실제 inlier의 span은 58.027px이다. 나머지 normal/tangent 및 complete q 항도 남아 있다. 원 q0과 q4는 fallback이며 약 0.00064px로 겹친다. 이는 약한 영상 배치의 단서일 뿐 물리 코너의 잘못된 대응이 확정됐다는 뜻은 아니다.

coarse에서 선택 pose로 N은 약 27.5→5.8~6.3으로 감소하지만 S는 0.106→약 1.83~1.85로 증가한다. 저장 J가 이 교환을 선호했다는 것은 확인된다. 선택 support slack은 모두 12px 이상이고 최소 joint는 거의 0이다. 최소 joint 전체를 새 선 support 제약이 타이트하다는 증거로 해석하지 않는다. 낮은 비용의 큰 회전 오차가 어떤 q/old partial/약한 기하의 조합에서 생겼는지는 이 정지 후보 비교만으로 분해 확정할 수 없다.

## wood_night_01:031426

[원 RGB](wood_night_01__031426.png) · 상태 0→7: `B U B U B O S S` · old partial IDs `[0, 4]` · S eligible `[6, 7]` · O regions 1

ROOT의 별도 사후 초기 coarse 오차: **174.7580cm / 81.6873°**. 아래 모든 신규 arm은 accepted=True이며 finite fallback이 아닌 선택 결과다.

| 방법 | family | T cm | R ° | 새 선 | N | O | S | C | J | feasible 후보 | 후보 J 범위 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e | 부모 비교 | 208.2242 | 83.7512 | — | — | — | — | — | — | — | — |
| ALL-N | SWAPPED | 13.0345 | 8.8504 | 0 | 0.095279 | 0.000000 | 0.837634 | 0.000000 | 0.932913 | 8 | [0.932913, 2.849650] |
| ALL-C | ORIGINAL | 191.3134 | 82.3232 | 0 | 0.664359 | 0.000000 | 0.933739 | 2.500146 | 4.098244 | 13 | [4.098244, 7.647074] |
| SOURCE-N | SWAPPED | 13.0345 | 8.8504 | 0 | 0.095279 | 0.000000 | 0.837634 | 0.000000 | 0.932913 | 8 | [0.932913, 2.849650] |
| SOURCE-C | ORIGINAL | 191.3134 | 82.3232 | 0 | 0.664359 | 0.000000 | 0.933739 | 2.500146 | 4.098244 | 13 | [4.098244, 7.647074] |

**원 ORIGINAL coarse는 네 arm 모두 feasible 후보로 남았다.** 실제 selector 별칭은 `ORIGINAL_WD/OWN_INDEPENDENT_INITIAL_COARSE`다. 요청 당시의 `ORIGINAL_INITIAL` 토큰은 없다. R/t가 같은 저장 UNREFINED 레코드의 accepted=True와 아래 min joint만 확인했으며, 보관되지 않은 coarse 개별 guard 배열은 재구성하지 않았다.

| 방법 | 원 coarse N | 원 coarse O+S | 원 coarse C | 원 coarse J | coarse min joint | 선택 pose와 일치한 저장 refinement 이름 |
|---|---|---|---|---|---|---|
| ALL-N | 0.228768 | 0.996881 | 0.000000 | 1.225649 | 0.124426 | FRESH_SQPNP_1 |
| ALL-C | 0.228768 | 0.996881 | 3.692022 | 4.917670 | 0.124426 | FRESH_SQPNP_0 |
| SOURCE-N | 0.228768 | 0.996881 | 0.000000 | 1.225649 | 0.124426 | FRESH_SQPNP_1 |
| SOURCE-C | 0.228768 | 0.996881 | 3.692022 | 4.917670 | 0.124426 | FRESH_SQPNP_0 |

위 선택 pose들은 fresh SQPNP 시작의 연속 refinement 결과다. 각 선택 family의 시작 목록에는 별도의 `OWN_INITIAL_COARSE` 또는 `OWN_INDEPENDENT_INITIAL_COARSE`도 있다. 저장된 전체 seed/start R/t·모든 trial 경로를 다시 생성하거나 검증한 것은 아니다.

새 활성 edge/owner가 모두 0이다. 관측 q와 남은 old partial 항, O/S 및 C만으로 최종 비용이 구성되므로 이 사례를 새 선 보정 효과로 설명할 수 없다.

**C는 native endpoint를 버린 것이 아니라, 보존한 상태에서 다른 family를 더 낮은 J로 선택했다.** ALL과 SOURCE는 이 사례에서 같은 결과다.

| C 후보 | family | N | O | S | C | J |
|---|---|---|---|---|---|---|
| 보존한 native winner (`SAME_POLICY_NATIVE_0`) | SWAPPED | 0.095279 | 0 | 0.837634 | 4.857064 | 5.789976 |
| 최종 C winner | ORIGINAL | 0.664359 | 0 | 0.933739 | 2.500146 | 4.098244 |

C 감소 2.356917이 N+S 증가 0.665185보다 커서 J가 1.691732 낮아졌다. 이에 따라 native 13.0345cm/8.8504°에서 C 191.3134cm/82.3232°로 악화됐다. 이는 **저장 후보와 고정 목적함수의 선호**를 보이는 반례다. 전체 최적해·마스크가 틀렸다는 인과·고유 depth/scale 또는 마스크 소유권을 증명하지 않는다. C는 관측 전경 포함 비용이며 가림을 의미하는 hard 제약이 아니다.

## eval_noapril:1775201428535961600

[원 RGB](eval_noapril__1775201428535961600.png) · 상태 0→7: `B B B B B U S U` · old partial IDs `[0, 3, 4]` · S eligible `[6]` · O regions 0

ROOT의 별도 사후 초기 coarse 오차: **3.4494cm / 0.6746°**. 아래 모든 신규 arm은 accepted=True이며 finite fallback이 아닌 선택 결과다.

| 방법 | family | T cm | R ° | 새 선 | N | O | S | C | J | feasible 후보 | 후보 J 범위 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e | 부모 비교 | 2.8801 | 0.6132 | — | — | — | — | — | — | — | — |
| ALL-N | ORIGINAL | 3.3902 | 0.4272 | 4 | 0.196019 | 0.000000 | 0.003011 | 0.000000 | 0.199031 | 41 | [0.199031, 109.887743] |
| ALL-C | ORIGINAL | 0.8124 | 0.6595 | 4 | 0.305576 | 0.000000 | 0.002701 | 0.127758 | 0.436035 | 55 | [0.436035, 117.447413] |
| SOURCE-N | SWAPPED | 40.7352 | 89.1843 | 0 | 0.124116 | 0.000000 | 0.012895 | 0.000000 | 0.137011 | 46 | [0.137011, 112.083714] |
| SOURCE-C | SWAPPED | 40.9213 | 89.2478 | 0 | 0.144694 | 0.000000 | 0.012494 | 0.298083 | 0.455271 | 61 | [0.455271, 119.507310] |

**원 ORIGINAL coarse는 네 arm 모두 feasible 후보로 남았다.** 실제 selector 별칭은 `ORIGINAL_WD/OWN_INDEPENDENT_INITIAL_COARSE`다. 요청 당시의 `ORIGINAL_INITIAL` 토큰은 없다. R/t가 같은 저장 UNREFINED 레코드의 accepted=True와 아래 min joint만 확인했으며, 보관되지 않은 coarse 개별 guard 배열은 재구성하지 않았다.

| 방법 | 원 coarse N | 원 coarse O+S | 원 coarse C | 원 coarse J | coarse min joint | 선택 pose와 일치한 저장 refinement 이름 |
|---|---|---|---|---|---|---|
| ALL-N | 0.456187 | 0.002565 | 0.000000 | 0.458752 | 0.273643 | FRESH_SQPNP_20 |
| ALL-C | 0.456187 | 0.002565 | 0.093733 | 0.552484 | 0.273643 | FRESH_SQPNP_20 |
| SOURCE-N | 0.804033 | 0.002565 | 0.000000 | 0.806598 | 0.273643 | FRESH_SQPNP_21 |
| SOURCE-C | 0.804033 | 0.002565 | 0.093733 | 0.900330 | 0.273643 | FRESH_SQPNP_21 |

위 선택 pose들은 fresh SQPNP 시작의 연속 refinement 결과다. 각 선택 family의 시작 목록에는 별도의 `OWN_INITIAL_COARSE` 또는 `OWN_INDEPENDENT_INITIAL_COARSE`도 있다. 저장된 전체 seed/start R/t·모든 trial 경로를 다시 생성하거나 검증한 것은 아니다.

| 방법 | 활성 edge | 3D축 | owners | 실제 inlier | 실제 span px | 모델 길이 px | 선 각도차 ° | 실제 support min px |
|---|---|---|---|---|---|---|---|---|
| ALL-N | [1, 2] | Y | 1,2 | 12 | 11.109 | 18.023 | 1.686 | 7.237 |
| ALL-N | [1, 5] | Z | 1 | 29 | 31.118 | 64.334 | 0.315 | 14.663 |
| ALL-N | [2, 3] | X | 2,3 | 219 | 216.004 | 226.083 | 0.237 | 15.210 |
| ALL-N | [4, 5] | X | 4 | 33 | 31.153 | 175.873 | 0.383 | 14.818 |
| ALL-C | [1, 2] | Y | 1,2 | 12 | 11.109 | 18.120 | 2.067 | 7.992 |
| ALL-C | [1, 5] | Z | 1 | 29 | 31.118 | 65.820 | 0.018 | 17.208 |
| ALL-C | [2, 3] | X | 2,3 | 219 | 216.004 | 227.661 | 0.416 | 15.230 |
| ALL-C | [4, 5] | X | 4 | 33 | 31.153 | 176.914 | 0.599 | 14.994 |

ALL은 실제 선 네 개와 owner1/2/3/4를 활성화했다. SOURCE는 RGB nonfallback q 보존 정책으로 그 새 owner/factor가 모두 빠져 새 활성선이 0이다. SOURCE의 큰 오차를 새 선 영향으로 설명할 수 없다. ALL은 q-normal 대체/잔존 tangent와 해당 CAL population까지 함께 달라지므로 네 선만의 독립적 인과 효과로 결론내리지 않는다.

SOURCE-N의 feasible ORIGINAL 최저 J는 **0.684655**(N0.681942+S0.002713), 선택 SWAPPED는 **0.137011**(N0.124116+S0.012895)이다. 좋은 own ORIGINAL coarse도 J0.806598로 후보에 남았지만 비용이 더 높았다. SOURCE-C에서도 ORIGINAL 최저 J0.767219보다 선택 SWAPPED J0.455271이 낮다. 즉 이 실패는 원 coarse가 누락되거나 guard로 배제돼 발생한 것으로 볼 수 없다.

보관된 SWAPPED 시작 감사의 `OWN_INDEPENDENT_INITIAL_COARSE`는 SOURCE에서 min joint −0.825470으로 거절돼 있다(ALL은 −165.248193). 이는 ORIGINAL coarse 후보의 탈락이 아니며, 해당 별칭의 family/pose 의미와 개별 guard 원인은 별도 STATE 정지 probe 범위다. 이 보고서는 보관된 최솟값 이상의 이유를 추정하지 않는다.

## 확인 범위와 한계

원 ORIGINAL coarse는 12/12 신규 arm에서 feasible selector에 남았고, C의 대응 native winner도 6/6 보존됐다. 따라서 이 세 사례에서 그 endpoint 누락 자체는 원인이 아니다. 후보군 안에서 J 최소를 고른 산술은 확인했으며, global optimum이나 보관하지 않은 trial/seed 전체의 올바름은 주장하지 않는다.

원 q·상태·공분산·실제선 provenance·support/방향·선택 비용·후보 범위는 증거 JSON에 보존했다. q/old partial의 실제 물리 코너 정체, 독립 가림 마스크, 초기 coarse 개별 guard, 최적화 중간 궤적은 이 자료만으로 확정하지 않았다. S의 영상 점/2D 항이나 U 정밀 점을 만들지 않았다. S 모델 prior 및 기존 U 영역은 계속 활성이다.

이 진단은 방법 수정·튜닝·재학습·GT 기반 query/후보/family 선택이나 기본 채택을 수행하지 않는다. 초기 coarse의 개별 guard 정지 재검산은 STATE의 별도 append-only probe이고, 평가 참조 wire 그림은 ACTUAL의 별도 권한 범위다.

## 봉인된 근거

- [원 계획](PLAN.json), [초기 source seal](SOURCE_SEAL.json), [저장 관측·선택 증거](EVIDENCE.json)
- [추가 coarse trace 계획](COARSE_TRACE_PLAN_V1.json), [source](add_coarse_trace.py), [source seal](COARSE_TRACE_SOURCE_SEAL_V1.json), [결과](COARSE_TRACE_V1.json)
- [기존 실제 전체 감사](../clean_fresh_BI_pose_eval_v4_32_r2/independent_root_audit_v1/AUDIT.json), [고정 SCORE_SEAL](../clean_fresh_BI_pose_eval_v4_32_r2/SCORE_SEAL.json)
- [이 보고서의 저장값 QA 및 범위](CLOSURE_RECEIPT_V1.json)
