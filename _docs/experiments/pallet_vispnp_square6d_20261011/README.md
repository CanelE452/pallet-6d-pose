# VIS PnP / square 6D 감사 (namespace 20261011)

[확인] A **SUPPORTED** (A2 실사), B **BLOCKED_REFERENCE_PROCEDURE_CONFLICT**. 실제 실행일은 2026-10-10이다. 사전등록 gate와 동일 참조 절차의 불일치를 기록하며 새 학습·새 주석 없이 종료했다.

SUPPORTED는 실사 주경로의 5 cm·5° 성공률 증가에 한정한다. 회전 평균·위치 중앙값/P90은 악화했고 합성 성공률은 세 seed 모두 감소했다. 전체 6D 지표 개선으로 확대하지 않는다.

[한국어 보고서](REPORT_KO.md) · [방법](METHOD_KO.md) · [재현](REPRODUCE.md) · [사전등록 lock](METHOD_LOCK.json) · [A0 parity](A0.json)

[정확도](METRICS.json) · [짝 비교](PAIRED.json) · [판정](VERDICT.json) · [실패·손상·전환 ID](FAILURES.json) · [square 입력 감사](SQUARE_INPUT_AUDIT.json) · [square 차단 기록](SQUARE_REFERENCE_POSES.json) · [그림 manifest](FIGURE_INDEX.json)

원본 RGB는 공개하지 않는다. 공개 사례는 실제 수치 코너·마스크이며 비공개 실제 RGB sheet와 구분한다. 좋은 seed나 영상으로 다시 선택하지 않고 나쁜 결과도 유지했다.

| PNG | 내용 |
| --- | --- |
| [figures/01_method_flow.png](figures/01_method_flow.png) | VIS 마스크와 기존 F의 적용 범위 |
| [figures/02_all_vis_accuracy.png](figures/02_all_vis_accuracy.png) | 모든 경로·seed ALL/VIS 정확도 |
| [figures/03_primary_paired_ci.png](figures/03_primary_paired_ci.png) | 사전등록 주비교의 짝 95% CI |
| [figures/04_subgroups.png](figures/04_subgroups.png) | 모든 고정 하위집단 |
| [figures/05_hidden_damage_switch.png](figures/05_hidden_damage_switch.png) | 숨김·손상·전환·fallback·미산출 |
| [figures/06_numeric_cases.png](figures/06_numeric_cases.png) | 실제 코너 좌표와 fit 마스크 사례 |
| [figures/07_square_manual_counts.png](figures/07_square_manual_counts.png) | 정사각형 수동 코너 분포와 절차 충돌 |

[새 코드](../../../scripts/research/pallet_vispnp_square6d_20261011/) · [기하 규칙](../../../scripts/research/pallet_vispnp_square6d_20261011/visibility.py) · [대응점 adapter](../../../scripts/research/pallet_vispnp_square6d_20261011/adapter.py) · [단위검사](../../../scripts/research/pallet_vispnp_square6d_20261011/test_adapter.py) · [B 감사](../../../scripts/research/pallet_vispnp_square6d_20261011/square_audit.py)
