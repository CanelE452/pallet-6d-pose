# Corrected-pseudo self-training paper closure

[결과·과정·표·실제 이미지 보고서](REPORT_KO.md) · [영문 원고 PDF](../../paper/selftraining_submission_v1/manuscript.pdf) · [LaTeX](../../paper/selftraining_submission_v1/manuscript.tex) · [최종 감사](PAPER_AUDIT_KO.md)

새 학습 없이 기존 공정 비교를 재사용했다. 일반 플라스틱 반복 DEV128장에서 raw→corrected 학생 PCK10은 47.51→51.47%, 같은 D9 최종 ADDsym AUC는 0.33472→0.35902. R0는49.14%/0.33796.

수동 가시점66개에서는 teacher가44→50/66으로 개선되지만, 학생 PCK10은43→43/66 동률이다. 기존128장 reference의 pooled 개선을 새 독립 검증 또는 모든 점의 개선으로 확대하지 않는다. P90 악화도 유지한다.

Teacher 실사 감독9장38점; 학생217 unique real RGB; real/synthetic각2560노출;320 update. 기존 learning-rate2개와 order반복2개의 모든 matched control을 보고했다. 독립 확인 데이터는 인증된 것이 없어 DEV-only 원고로 마감한다. 방법 개발 STOP; 저자 검토와 제출 준비는 별도다.

## Audit / reproduction

- [입력·학습 공정성](CORE_COMPARABILITY_AUDIT.json)
- [전체 2D·6D·난도·recording 결과](CORE_RESULTS.json)
- [66점 품질과 학생 민감도](PSEUDO_LABEL_QUALITY.json)
- [실사 감독 예산](MANUAL_SUPERVISION_BUDGET.json)
- [짝지은 변화](PAIRED_ANALYSIS.json)
- [공개 오류 측정 행](MEASUREMENT_ROWS.json)
- [재현 문서](../../paper/selftraining_submission_v1/REPRODUCIBILITY.md)
- [제출 전 저자 확인 사항](../../paper/selftraining_submission_v1/SUBMISSION_READINESS.md)

기존 local-refiner 원고와 과거 실험은 보존했다. 추가 학습·촬영·어노테이션을 이 작업에서 요청하지 않는다.
