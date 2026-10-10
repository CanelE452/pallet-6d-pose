공개 재현 안내의 출력 경로 오류를 확인했습니다. 기존 mechanism 감사의 `binding(path)`는 `path.relative_to(ROOT)`를 사용하고 bootstrap 감사는 resolve 후 repository 내부 경로를 요구합니다. 따라서 기존 mechanism 안내의 `/tmp/...` 예시는 지원되는 경로가 아닙니다. freeze가 protocol을 쓴 뒤 출력 SHA를 표시할 때 예외가 날 수 있습니다. 이를 성공한 산술 run이나 실제 방법의 오류로 해석하지 않습니다.

기존 코드·protocol·성공/실패 원행·보고서 bytes는 보존합니다. 재현 안내는 아래 명령으로 정정합니다. 새 연구 checkout의 저장소 루트에서 실행하고, 목적지는 존재하지 않는 **저장소 내부의 절대경로**여야 합니다. 사용자 원본 checkout이나 이미 게시된 결과 폴더에 실행하지 않습니다.

```bash
python -B -m scripts.research.pallet_corner_mechanism_audit_20261010_v1.audit freeze --output "$PWD/_docs/experiments/pallet_corner_mechanism_replay_local"
python -B -m scripts.research.pallet_corner_mechanism_audit_20261010_v1.audit run --output "$PWD/_docs/experiments/pallet_corner_mechanism_replay_local"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m scripts.research.pallet_boundary_bootstrap_audit_20261010_v1.audit freeze --output "$PWD/_docs/experiments/pallet_boundary_bootstrap_replay_local"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m scripts.research.pallet_boundary_bootstrap_audit_20261010_v1.audit run --output "$PWD/_docs/experiments/pallet_boundary_bootstrap_replay_local"
```

별도의 `verify_mechanism` 검산기는 공개 receipt를 읽고 `/tmp/...json`에 새 receipt를 쓰는 것을 지원합니다. 이 검산기와 freeze/run 드라이버의 출력 제약을 혼동하지 않습니다. 이번 정정은 code inspection으로 확인한 경로 계약 오류이며, 동일 산술 실험이나 새 자세 평가를 반복 실행하지 않았습니다. 기존 실험 결과의 수치나 모델 결정 규칙은 변경하지 않았습니다.
