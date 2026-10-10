저장된 v2 자세 원행의 세션 bootstrap 신뢰구간을 별도 구현으로 수치 검산했습니다. 새로운 자세 평가나 성능 실험이 아닙니다. [결과](RESULT_KO.md), [사전 잠금](PROTOCOL.json), [원행](ROWS.jsonl.gz), [검산](CHECKS.json)을 함께 확인합니다.

기존 v2 [REVIEW_CHECKS](../pallet_boundary_corner_refiner_20261010_v2/REVIEW_CHECKS.json)는 `CI95_numeric_replay=false`였고 그대로 보존했습니다. 이번 별도 검사는 공개된 1,715개 scored 원행과 기존 10,000×13 세션 multiplicity를 읽어 144개 CI와 combined 별칭 48개를 확인했습니다. 원래 통계 함수를 import하지 않았습니다.

재현은 새 연구 checkout의 저장소 루트에서, 저장소 안의 새 출력 폴더로 실행합니다. helper가 output도 repository-relative binding으로 기록하므로 아래처럼 절대경로를 지정합니다. 실제 환경은 Python 3.10.20, NumPy 1.26.4입니다. 기존 결과에 덮어쓰지 않습니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m scripts.research.pallet_boundary_bootstrap_audit_20261010_v1.audit freeze --output "$PWD/_docs/experiments/pallet_boundary_bootstrap_replay_local"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m scripts.research.pallet_boundary_bootstrap_audit_20261010_v1.audit run --output "$PWD/_docs/experiments/pallet_boundary_bootstrap_replay_local"
```

기존 참조 오차가 입력입니다. 물리 pose GT와 scorer를 독립 재실행한 것으로 표현하지 않습니다. model/neck/head/PnP/학습/RGB/새 bootstrap draw는 모두 0입니다. [목표 완료 감사의 시점별 기록](GOAL_COMPLETION_SNAPSHOT.json)은 API 실행·검산과 아직 미달성인 실제 정확도 목표를 구분합니다.

기존 mechanism 재현 명령의 출력 경로는 [REPRODUCIBILITY_ERRATA](REPRODUCIBILITY_ERRATA.md)의 정정 명령을 사용합니다.
