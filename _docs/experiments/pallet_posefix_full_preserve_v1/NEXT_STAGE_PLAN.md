# 다음 단계 — 하나만

**REPRESENTATION_GEOMETRY_DIAGNOSIS**

보존만으로 해결되지 않은 큰 오류의 candidate/representation 제약을 문헌·구조 비교부터 진단한다. 이번 작업에서 새 geometry나 학습은 하지 않는다.

## 지시문의 10개 질문에 대한 답

1. FULL B3 6개 중 동일 코너 유지 5개. FP 총복구 5개.
2. BASE 정답 손실 17→16 (+1개 감소).
3. PCK10 50.35→50.35% (+0.00pp). 전체 gain/loss 전이로 검산했다.
4. N2 정답 손실 31→31.
5. source clean FULL 91.35→FP 92.24%; BASE 94.02%.
6. source stress FULL 67.71→FP 67.41%.
7. GREEN FULL 79.15→FP 79.44%.
8. CLEAN17 PCK10 90.44→91.91%; CAD18 62.50→66.91%.
9. 복구 유지 gate=True. source stress와 실사 B3 변화로 적응 약화 여부를 해석하되 λ 한 점만으로 강도/목표 충돌을 인과적으로 분리했다고 주장하지 않는다.
10. 현재 다음 병목 후보: REPRESENTATION_GEOMETRY_DIAGNOSIS. >40px 복구는 FULL 0, FP 0. 이는 representation ceiling 가설과 양립하지만 그 자체로 원인을 증명하지 않는다.

Blind review: REVIEW_PENDING. Data: CANDIDATE_AVAILABLE. 신규 추가학습 없음. single-seed 반복 DEV 결과이며 독립 generalization/외부가림 복원 주장 없음. 최종 모델/논문 표 변경 없음.
