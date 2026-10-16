# 자세 기하·대칭·PnP

C4 대칭과 점ID, 학습측 DiffPnP 및 평가 solver 비교. 점·선 융합의 최근2D 지표와 별도 트랙이다.

[전체 노트 목차](../README.md) · [현재 점·선 연구 종합](../point_line_fusion/STATUS_AND_NEXT_STEPS.md)

> 각 원문은 실험 당시의 기록이다. 당시의 중단/일시정지/다음 제안은 최신 상태를 뜻하지 않는다. 현재 결론은 종합 문서를 우선한다.

| 문서 | 내용 |
|---|---|
| [c4-rotation-symmetry.md](c4-rotation-symmetry.md) | 정사각 팔레트 회전 대칭 (C4) 계열 |
| [diffpnp-yolo.md](diffpnp-yolo.md) | 학습측 DiffPnP — 미분가능 PnP 를 YOLO pose loss 에 넣는다 (diffpnp_yolo_v1) |
| [pnp-solver-swap.md](pnp-solver-swap.md) | 평가 solver 교체 — SQPnP+RefineLM → 미분가능 Gauss-Newton PnP (solver_swap_v1) |

## 치수 입력 아키텍처

- [C1 기존 loss 유지 · C4 전체 loss 선택 실제 검증](c4-joint-loss-check.md) — 3군×3시드. 점 선택과 전체 선택은 가중치·예측까지 동일. 학습 전 대비 C4 개선, C1 악화; 선택 기준과 추가 학습 효과를 구분.
- [점 기반 논문 초안과 시각화](point_only_paper/README.md) — 선 제외, 실제 보정 구조·loss·비교표·기여 범위. 대칭 통합 설계와 완료 실험을 구분.
- [이전 대칭 loss의 실제 결과와 설명 정정](why-previous-symmetry-loss-did-not-win.md) — 실물 C4는 이미 학습했다. 번호만 바뀐 실패, 추가 위치 개선 미확인, 공동 head 위치 실패를 구분.
- [코너 번호와 실제 위치 오류 재분석](corner-id-versus-spatial-error.md) — 새 종류에서 번호 영향을 제외하면 8.29→7.07px, 세 시드 개선. 실물 정사각 대칭과 합성 메시의 차이, 기존 실패·개선 사례 해석 수정.
- [코너 대응 선택과 작은 보정 분리](role-selection-and-residual.md) — 실제 구조와 손실·GT 대상 연결 대조. 본 종류 새 장면의 조건부 개선, 새 종류 안정성은 미해결.
- [새 팔레트 치수·기하 구조 검증](unseen-dimensions-screen.md) — 학습에 없는 두 메시, 4군×3시드 실제 비교. 기하 관계 추가 이득 미확인; 큰 코너 ID 오류·보정 범위·GT 역할 경계 원인 진단과 이미지.
- [기존 대비 아키텍처 개선량 비교표](architecture-comparison-overview.md) — 최신 치수·대칭, 실사 조건별 대표 구조, V44 새 장면을 평가셋별로 구분.
- [치수 × 대칭 네 군 실제 결과](dimension-symmetry-2x2-results.md) — 재부팅 없이 GPU 복구, 기존 출력층12모델 학습 완료. 작은 치수 효과, 안정적 기존 성능 개선은 미확인.
- [공유 대화 원문과 실제 실행 대조](shared-dimension-symmetry-review.md) — 원래 요청한 공동 대칭 로스, 완료 범위와 남은 네 군 비교.
- [기존 구현·학습 진행 상태](dimension-input-status.md)
- [팔레트 구조를 활용하는 논문 기여 후보 설계](pallet-dimension-architecture-proposal.md) — 최초 잠정안. 이후 일부 보정 구조의 새 자산 검증은 위 문서에서 확인.
