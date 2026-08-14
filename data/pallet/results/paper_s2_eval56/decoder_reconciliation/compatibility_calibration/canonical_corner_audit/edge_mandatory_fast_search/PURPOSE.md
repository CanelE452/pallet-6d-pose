# PURPOSE — partial-edge supervision ablation, B0 fresh train (2026-08-12)

[소비처] 논문 line-stage 학습 recipe 결정 — 화면에 일부만 남은 structural edge 를
         supervision 에 포함하는 것이 그 population 예측에 실제로 도움이 되는지.
         결과가 truncation robustness 에 대한 recipe 판단 근거가 된다.

[문장] "endpoint 가 프레임 밖으로 잘린 edge(T1_PARTIAL)를 supervised loss 에 포함하면,
        그 population 의 line 예측이 포함하지 않을 때보다 개선된다" 는 주장을,
        B0(T1 제외) 대비 B1(=historical P0, T1 포함)로 LINE_DEV T1 @25,545 에서 검증한다.

사전등록: 82ec010 (stratification lock) + 이번 commit (ablation lock)
러너:     scripts/stage0/partial_edge_supervision_ablation.py
factor:   T1_PARTIAL 의 supervised-loss 포함 여부 하나
preflight: mask isolation ISOLATED / P0 reuse QUALIFIED — 전부 통과
결과:     partial_supervision_result.json  (위 결과 root)
