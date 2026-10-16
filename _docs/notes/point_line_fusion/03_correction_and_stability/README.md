# 보정량·후보 선택·안정성

V13~V19. 선 근거·RGB·가시 실루엣, 직접 보정 벡터, 보정 적용률. 큰 복원 이득과 추가 선의 효과를 분리한다.

[전체 노트 목차](../../README.md) · [현재 점·선 연구 종합](../STATUS_AND_NEXT_STEPS.md)

> 각 원문은 실험 당시의 기록이다. 당시의 중단/일시정지/다음 제안은 최신 상태를 뜻하지 않는다. 현재 결론은 종합 문서를 우선한다.

| 문서 | 내용 |
|---|---|
| [pallet_candidate_gain_v13.md](pallet_candidate_gain_v13.md) | V13: frozen candidate gain evaluation |
| [pallet_final_fusion_v14_architecture.md](pallet_final_fusion_v14_architecture.md) | 최종 후보: 점 유지·복원·선 보정의 학습한 선택 |
| [pallet_final_fusion_v14_results.md](pallet_final_fusion_v14_results.md) | V14 완료: 최종 점·선 구조와 기존 방법 비교 |
| [pallet_stability_v15_design.md](pallet_stability_v15_design.md) | V15: 악화 억제와 어려운 조건 개선을 함께 유지할 수 있는가 |
| [pallet_stability_v15_results.md](pallet_stability_v15_results.md) | V15 결과: 악화 억제·소폭 개선 확인, 큰 복원 이득 유지 실패 |
| [pallet_line_evidence_v16_design.md](pallet_line_evidence_v16_design.md) | V16: 후보별 선 구간의 영상 근거와 경계 역할 |
| [pallet_line_evidence_v16_results.md](pallet_line_evidence_v16_results.md) | 후보별 선 특징: 심한 잘림은 개선, 전체 안정성은 미달 |
| [pallet_detail_v17_design.md](pallet_detail_v17_design.md) | V17: RGB 세부 정보와 가시 실루엣 보조 정답을 분리한 실험 |
| [pallet_detail_v17_results.md](pallet_detail_v17_results.md) | 고해상도 RGB와 가시 마스크 검증: 전체 안정성 기준 미달 |
| [pallet_vector_v18_design.md](pallet_vector_v18_design.md) | V18 보정 벡터 실험 설계 |
| [pallet_vector_v18_results.md](pallet_vector_v18_results.md) | 선 특징으로 보정 방향·크기 직접 학습: V18 |
| [pallet_vector_v18_scale_feasibility.md](pallet_vector_v18_scale_feasibility.md) | V18 보정 유지율의 가능성 검산 |
| [pallet_retention_v19_design.md](pallet_retention_v19_design.md) | V19 적용률 아키텍처 설계 |
| [pallet_retention_v19_results.md](pallet_retention_v19_results.md) | V19: 좋은 보정은 유지하고 잘못된 보정은 줄일 수 있는가 |
