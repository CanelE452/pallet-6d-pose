# PGBC_OVERFIT_GATE — NOT RUN

16-frame correction overfit 는 실행하지 않았다.

사유: feasibility gate G0/G1/G2 가 전부 FAIL 이고, 특히 **G1 FEATURE_OBSERVABILITY 는
사전 지정된 종료 규칙**("FAIL 이면 frozen-feature PGBC 전체 중단")에 해당한다.
"통과한 component 만 구현한다"는 규칙에 따라 구현·학습 단계로 진입하지 않았다.

판정 근거와 수치: `PGBC_COMPONENT_DECISION.md`
게이트 원자료: `pgbc_gate.json`, `pgbc_g0_residual_capacity.csv`,
`pgbc_g1_probe_samples.csv`, `pgbc_g1_probe_diagnostics.json`, `pgbc_g2_leave_one_out.csv`
