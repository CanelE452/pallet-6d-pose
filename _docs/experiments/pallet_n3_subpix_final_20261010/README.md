# N3→cornerSubPix 3-seed 최종 실사 검증 (2026-10-10)

상태: **COMPLETE**. 동일 REAL_DEV 319장·13세션, N3 seed 1/2/3, 새 학습 0. BASE/N3/SUBPIX/N3→SUBPIX를 같은 최종 F로 비교했다.

[한국어 결과 보고서](REPORT_KO.md) · [프레임 추적 안내](REVIEW_GUIDE_KO.md) · [재현 명령](REPORT_KO.md#8-재현)

[전체 수치 CSV](RESULTS.csv) · [통계 JSON](METRICS.json) · [짝비교/95% CI](PAIRED_COMPARISONS.json) · [원행 gzip JSONL](PREDICTIONS.jsonl.gz)

[입력 해시 감사](INPUT_AUDIT.json) · [감사 부가본](INPUT_AUDIT_SUPPLEMENT.json) · [입력/방법 lock](INPUT_AND_METHOD_LOCK.json) · [seed1 전319 parity](SEED1_PARITY.json) · [실행/검증](EXECUTION_AND_VERIFICATION.json) · [독립 검산](INDEPENDENT_VERIFICATION.json) · [R/t 독립 재계산](EVALUATOR_CONTRACT_REVIEW.json) · [과거 증거 검산](HISTORICAL_EVIDENCE_VERIFICATION.json) · [그림/권리 manifest](FIGURE_INDEX.json)

원본 RGB의 공개 배포 권한이 확인되지 않아 좌표만으로 사례를 그렸다. 반복 개발자료와 기하 재구성 참조의 한계, PCK/큰오류/손상 등 부정적 결과를 보고서에 함께 보존했다.

| PNG | 내용 |
| --- | --- |
| [figures/01_method_overview.png](figures/01_method_overview.png) | Fixed integrated refinement pipeline |
| [figures/02_pose_accuracy_by_seed.png](figures/02_pose_accuracy_by_seed.png) | Pose mean by seed and mean across seeds |
| [figures/03_paired_improvement_ci.png](figures/03_paired_improvement_ci.png) | Paired session cluster confidence intervals |
| [figures/04_grade_breakdown.png](figures/04_grade_breakdown.png) | All annotation difficulty grades retained |
| [figures/05_error_distribution.png](figures/05_error_distribution.png) | Untrimmed error distributions |
| [figures/06_corner_damage_and_hypothesis.png](figures/06_corner_damage_and_hypothesis.png) | Corner quality, movement cap and hypothesis diagnostics |
| [figures/07_qualitative_examples.png](figures/07_qualitative_examples.png) | Post-hoc coordinate-only improvement and failure examples |

[재현 코드](../../../scripts/research/pallet_n3_subpix_final_20261010/) · [집계](../../../scripts/research/pallet_n3_subpix_final_20261010/summarize.py) · [독립 검산](../../../scripts/research/pallet_n3_subpix_final_20261010/verify.py) · [그림 생성](../../../scripts/research/pallet_n3_subpix_final_20261010/figures.py) · [보고서 생성](../../../scripts/research/pallet_n3_subpix_final_20261010/report.py)
