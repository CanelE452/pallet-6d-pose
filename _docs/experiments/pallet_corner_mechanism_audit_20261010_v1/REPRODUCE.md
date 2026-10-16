이 감사는 새로운 실사 성능 실험이나 학습을 추가하지 않았습니다. 먼저 [BUILD_LEDGER](BUILD_LEDGER.json)의 완료 횟수와 첫 RGB 검사 실패를 확인합니다. 수치 비교에 사용한 실사 참조는 기존 GEOMETRIC_PROXY DEV이며 독립 실측 GT가 아닙니다.

공개 저장소 루트에서 Python 3.10과 NumPy로 기존 원행의 산술 분해를 새 출력 폴더에 재현할 수 있습니다. 아래 경로는 존재하지 않는 새 폴더여야 합니다.

```bash
python -B -m scripts.research.pallet_corner_mechanism_audit_20261010_v1.audit freeze --output /tmp/pallet-corner-mechanism-replay
python -B -m scripts.research.pallet_corner_mechanism_audit_20261010_v1.audit run --output /tmp/pallet-corner-mechanism-replay
```

저장된 공개 결과의 별도 산술 검산은 NumPy·OpenCV·Torch 없이 Python 표준 라이브러리로 실행합니다. `--output`에 새 파일을 지정합니다. 이 검산은 배포 추론이나 PnP를 호출하지 않습니다.

```bash
python3 -B -m scripts.research.pallet_corner_mechanism_audit_20261010_v1.verify_mechanism --output /tmp/pallet-corner-mechanism-independent-check.json
```

검산기에서 `--root`를 생략하면 코드가 위치한 공개 저장소를 읽습니다. 별도 원행 폴더의 산술 실행과 공개 결과 검산은 각각 다른 입력을 읽으므로 두 명령의 출력 경로를 혼동하지 않습니다. PNG는 저장된 423개 실사 교체 후보와 CAL 348개를 모두 표시했습니다. 그림과 보고서의 최종 SHA는 FIGURE_BINDINGS에 있습니다. report.py는 덮어쓰기 금지이며 기존 결과에 다시 실행하지 않습니다.

다음 검사는 외부 읽기 전용 자료가 필요합니다. 원본 RGB·전체 Base/N3 가중치·source mesh/feature cache는 새로 게시하지 않았습니다. receipt에 상대 경로·크기·SHA가 있으며, 동일 자료를 확보하지 못하면 해당 검사까지 독립 재현했다고 주장할 수 없습니다.

| 완료한 검사 | 진입점과 필요한 외부 자료 |
| --- | --- |
| source 계약 | `source_contract.py --source-root SOURCE_ROOT --baseline-root FROZEN_BASELINE_ROOT --features FEATURES_NPY --cache-manifest CACHE_MANIFEST_JSON --mesh ACTUAL_MESH_NPZ --output NEW_CHECK_JSON` |
| native CAL 코너 계약 | `cal_corner_contract.py freeze/run --root REPLAY_REPO --source-root SOURCE_ROOT`; REPLAY_REPO에는 기존 입력과 코드가 있어야 하고 새 감사 결과 목적지는 비어 있어야 함 |
| RGB 원 실패와 재개 | `rgb_contract.py`와 `rgb_contract_resume.py` 각각 `freeze/check --source-root SOURCE_ROOT --features FEATURES_NPY --cache-manifest CACHE_MANIFEST_JSON --protocol NEW_PROTOCOL_JSON --output NEW_CHECK_JSON` |
| 실제 배포 API 수명 | `deployment_smoke.py freeze/preflight/run --source-root SOURCE_ROOT --baseline-root FROZEN_BASELINE_ROOT --fits CORRECTED_BUNDLE --output NEW_SMOKE_DIR`; 실제 환경·가중치와 공개 보호 snapshot의 일치를 먼저 확인 |

표의 `freeze/run`, `freeze/check`, `freeze/preflight/run`은 각각 순서대로 실행한 별도 stage 이름이며 하나의 CLI 인자가 아닙니다. 새 RGB 감사는 원 실패의 tuple/list 비교 한 줄만 JSON canonical 비교로 바꿨습니다. 원 코드·사전 잠금·실패 receipt는 보존했습니다. 원 실패를 성공 결과로 대체하지 않습니다.

실제 GPU 사용 검사는 사전에 고정된 첫 영상 하나에서 두 인스턴스를 순차로 실행했습니다. 기존 봉인 출력과 좌표·R,t·상태의 일치만 확인했습니다. 새 GT 채점·성능 집계·latency 측정이 아니며, 다른 이미지를 골라 성공 사례를 찾는 반복도 하지 않았습니다. 같은 프로세스에서 여러 활성 Pipeline의 동시 사용은 지원하지 않습니다.

source ID/bin/기하 및 RGB 3채널의 검증을 전체 19채널 일치, 실사 물리 경계 소유권 인증 또는 자세 정확도 개선으로 확대하지 않습니다. 현재 정확도 목표는 미달성입니다.
