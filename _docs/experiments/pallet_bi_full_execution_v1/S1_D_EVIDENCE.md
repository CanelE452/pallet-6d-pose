# S1/D 저장 근거

S1 PRIMARY 성공(세 seed): 119/118/120.
원 D1+두 PF anchor union과 own-H1 결과다. 신규PF cost1348과 실제 old-cost fallback4를 포함한1352 평가를 constructed work로 빼지 않는다. 별도 postseal diagnostic130는 다른 범위다.
D는 oldΣ+Γ와 COMMONΓ고정 진단을 구별한다. 제한 WD 공통 graph에서 source-compatible103/source-changing135를 분리하며 전체 H1 정책의 인과 분해로 바꾸지 않는다.

| D variant | PRIMARY cell | 성공/192 |
|---|---|---:|
| D_SIGMA_GAMMA_CONFOUNDED | Q0S0 | 107/192 |
| D_SIGMA_GAMMA_CONFOUNDED | Q1S0 | 107/192 |
| D_SIGMA_GAMMA_CONFOUNDED | Q0S1 | 106/192 |
| D_SIGMA_GAMMA_CONFOUNDED | Q1S1 | 107/192 |
| D_SIGMA_GAMMA_CONFOUNDED | ACTUAL_COMMON | 107/192 |
| D_SIGMA_GAMMA_CONFOUNDED | ACTUAL_BI | 107/192 |
| D_FIXED_COMMON_GAMMA | Q0S0 | 107/192 |
| D_FIXED_COMMON_GAMMA | Q1S0 | 107/192 |
| D_FIXED_COMMON_GAMMA | Q0S1 | 106/192 |
| D_FIXED_COMMON_GAMMA | Q1S1 | 107/192 |
| D_FIXED_COMMON_GAMMA | ACTUAL_COMMON | 107/192 |
| D_FIXED_COMMON_GAMMA | ACTUAL_BI | 107/192 |

| D variant | contrast | family switches | within-before-family Δsuccess | family-readout Δsuccess | total Δsuccess |
|---|---|---:|---:|---:|---:|
| D_SIGMA_GAMMA_CONFOUNDED | Q1S0_minus_Q0S0 | 7 | 2 | -2 | 0 |
| D_SIGMA_GAMMA_CONFOUNDED | Q1S1_minus_Q0S1 | 5 | 3 | -2 | 1 |
| D_SIGMA_GAMMA_CONFOUNDED | Q0S1_minus_Q0S0 | 5 | -1 | 0 | -1 |
| D_SIGMA_GAMMA_CONFOUNDED | Q1S1_minus_Q1S0 | 3 | 0 | 0 | 0 |
| D_FIXED_COMMON_GAMMA | Q1S0_minus_Q0S0 | 7 | 2 | -2 | 0 |
| D_FIXED_COMMON_GAMMA | Q1S1_minus_Q0S1 | 5 | 3 | -2 | 1 |
| D_FIXED_COMMON_GAMMA | Q0S1_minus_Q0S0 | 5 | -1 | 0 | -1 |
| D_FIXED_COMMON_GAMMA | Q1S1_minus_Q1S0 | 3 | 0 | 0 | 0 |

[전체 D scalar](tables/D_PUBLIC_SCALARS_V3/POPULATION_METRICS.csv) · [고정 cohort](tables/final_report/FIXED_COHORTS.json) · [전체 보고서](FINAL_REPORT_KO.md).
