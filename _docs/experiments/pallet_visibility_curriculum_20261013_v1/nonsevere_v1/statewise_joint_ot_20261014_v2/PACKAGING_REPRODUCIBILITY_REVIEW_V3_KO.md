# 공개 패키지와 재현성 점검 v3

현재 연구 소스와 예측은 수정하지 않았다. 이전 공개 목록, 새 상태별 연구 폴더, 현재 실행 설정과 의존 경로를 읽기 전용으로 점검했다. 이 보고서는 업로드 전 검토 시점의 목록이며 실제 업로드 확인 영수증은 아니다.

이전 manifest의 1,200개 파일은 모두 존재한다. 새 소스 22개와 새 연구 자료 279개를 추가한 후보 목록은 **1,501개, 252,522,558 bytes**다. 최대 개별 파일은 24,263,575 bytes이며 100MB를 넘는 파일은 없다. 이 보고서 2개는 자기참조 해시를 피하려고 후보 계산에서 제외했다. 최종 업로드 manifest에는 별도로 포함해야 한다.

## 포함과 제외

- 이전 `PUBLICATION_MANIFEST_V3.json`의 경로와 현재 연구 소스, 봉인 예측·프로토콜·실행 스냅샷·검토·문서를 포함하는 후보 목록을 JSON에 기록했다.
- 새 `PARAMETERS.npz` 3개는 작은 fitted 모델 파라미터로 포함한다. 현재 기본 state head는 67,428 bytes다. 기존 Tiny `best_source_CAL.pt` 130,710 bytes도 실행에 필요하며 이전 공개 목록에 있다.
- `NON_S_FEATURES.npz` 2개와 `NON_S_RGB26_FEATURES.npz` 1개는 원 특징 배열이므로 제외한다. 캐시/pyc/vendor/대형 외부 checkpoint도 제외한다.
- 실제 원 RGB와 `ot_domains/real_crops`의 원 crop PNG 3개는 제외한다. 현재 `figures/index.html`의 주석 PNG/SVG 3쌍은 모두 포함한다. 내장 실제 사진이 있는 그림은 공개할 주석 결과물이므로 원 입력 영상의 별도 배포 여부와 구분한다.
- 참조되지 않은 역사적 v1/v2 그림 preview는 최종 curator가 제외할 수 있다. 이 검토의 넓은 후보 목록에는 남겼다.

이전 manifest와 현재 SHA가 달라진 파일은 아래 문서 6개다. 새 설명을 반영한 변경이며, 이 파일들을 이전 hash 그대로 복사하면 최종 manifest가 틀리므로 업로드 직전 재해시해야 한다. 이전 runtime source 경로의 hash 변경은 발견하지 않았다.

- `_docs/experiments/pallet_visibility_curriculum_20261013_v1/PUBLICATION.md`
- `_docs/experiments/pallet_visibility_curriculum_20261013_v1/README_KO.md`
- `_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/README_KO.md`
- `_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/region_completion_v1/PAPER_DRAFT_EN.md`
- `_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/region_completion_v1/README_KO.md`
- `_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/region_completion_v1/STATEWISE_METHOD_AND_RESULTS_EN.md`

## 보호된 사용자 파일

`STATEWISE_PUBLICATION_REMOTE_RECEIPT_V1.json`의 6개 보호 파일 SHA를 현재 파일과 비교했다. **6개 모두 정확히 일치**하며 후보 공개 범위에도 들어 있지 않다. `.gitignore`, `readme.md`, 기존 ResNet18 README, Three-estimator README/STATUS, annotation progress 파일이다. 상세 원 hash와 현재 hash는 JSON에 있다. 파일 내용의 의미나 주석 좌표는 읽지 않았다.

## 현재 한 장 실행

`RUNTIME_CONFIG_V3.json`이 상대경로로 참조하는 state model, O/T model, mask descriptor, boundary mask config는 모두 존재하고 공개 후보 범위에 있으며 바인딩 SHA와 일치한다. current CLI의 local import closure 17개도 모두 포함된다. 다른 연구 폴더의 `foundation.py`와 `uncertainty.py`는 이전 manifest에 포함되어 있다.

| 자산 | 공개 묶음 | 현재 로컬 검증 |
|---|---|---|
| 기본 fitted state `PARAMETERS.npz` | 포함 | SHA 일치 |
| O/T JSON 모델 및 입력 K/3D/초기 pose | 포함 | 설정 SHA 일치 |
| 작은 source-CAL Tiny 가중치 | 포함 | SHA 일치 |
| ImageNet ResNet18, 46,830,571 bytes | 제외, 아래 bootstrap으로 복원 | SHA 일치 |
| MobileSAM vendor와 40,728,226-byte checkpoint | 제외, 아래 bootstrap으로 복원 | 고정 commit의 18개 runtime 자산 SHA 일치 |
| `EXAMPLE_INPUT.json`의 원 RGB | 제외, 사용자가 로컬에서 제공 | 기록 file SHA 일치 |

저장소 루트에서 공개 foundation을 복원한다. bootstrap은 기존 파일의 SHA가 다르면 덮어쓰지 않고 중단한다. Git 및 네트워크가 필요한 준비 단계이며 이번 점검에서는 다운로드하지 않았다.

```bash
python -m scripts.research.pallet_visibility_curriculum_20261013_v1.bootstrap_foundation
python -m scripts.research.pallet_visibility_curriculum_20261013_v1.bootstrap_foundation --check-only
python -m scripts.research.pallet_visibility_curriculum_20261013_v1.bootstrap_joint --descriptor _docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/state_visibility/bounded_ownership_spatial_v3/MODEL.json
```

ResNet18은 공식 PyTorch endpoint의 고정 SHA를 검사한다. MobileSAM은 `f706ad9c4eb7f219c00d9050e46328518ffb65d2` commit과 `MOBILESAM_ASSET_MANIFEST.json`의 파일 SHA를 검사한다.

`EXAMPLE_INPUT.json`에는 원 native KP, K, 등록 3D 모델 및 초기 pose가 있어서 이 CLI는 detector를 다시 실행하지 않는다. 다른 실제 image backbone 결과에도 같은 입력 계약을 제공해야 한다. 원 detector를 새로 실행하려면 해당 detector의 로컬 자산이 별도로 필요하다.

```bash
python -m scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_joint_cli_v2 --image data/evaluation/pallet_eval_v1/dev_existing/sessions/eval_outside/rgb/1778651519235162880.png --input _docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/EXAMPLE_INPUT.json --config _docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/RUNTIME_CONFIG_V3.json --output /tmp/pallet-statewise-current-v3-example.json --device cuda
```

출력은 새 경로를 사용한다. CLI의 `--image`는 독립 인자이고 EXAMPLE의 `image`/`image_sha256` metadata를 자동 검증하지 않는다. 같은 예제를 정확히 재현할 때는 입력 RGB file SHA `ca549694bf816f0400e7b2fa733fe5617abc2ccb4b9374094ac9595015177467`를 먼저 확인해야 한다. 원 RGB가 없으면 이 예제 명령은 실행할 수 없다. CUDA가 없으면 `--device cpu`를 지정할 수 있으나 속도·수치적 동일성은 이번 실행 증거의 보장 범위가 아니다.

## 재현 가능한 범위

공개 예측·프로토콜·실행 source snapshot으로 해시와 readout 계약을 감사할 수 있다. 봉인 V2 cache의 고정 state/BI/domain/K/3D 값만 받아 `solve_joint_ot_pnp` 또는 replay의 `_run`을 직접 호출하는 수학 재실행도 원 영상·대형 모델·원 특징 없이 가능하다.

반면 **기존 `statewise_joint_solver_replay_v3` CLI는 패키지 단독으로 실행되지 않는다.** 솔버 계산에는 영상이 필요 없지만, 그대로 보존한 `crop_bindings()`가 원 crop PNG 3장의 SHA를 다시 계산한다. 정확한 역사적 protocol 재실행에는 그 3개 파일을 로컬에 준비해야 한다. 원 RGB와 저장된 crop transform으로 픽셀을 재구성할 수 있더라도 PNG 재인코딩이 기록된 file SHA를 보존한다고 보장할 수 없다. 이 의존성을 숨기기 위해 봉인된 소스를 수정하지 않았다.

새로운 실제 RGB CLI와 저장 smoke는 원 RGB, Tiny, 복원한 ResNet18/MobileSAM, 호환 Python 환경이 필요하다. 새 학습·95개 전체 RGB 재추론에는 원 데이터·주석·upstream 예측 자료가 별도로 필요하다. 원 특징 NPZ를 제외하므로 재학습 시 특징도 다시 추출해야 한다. 봉인된 결과의 공개는 이 외부 데이터 전체를 배포한다는 뜻이 아니다.

## 확인한 실행 환경

현재 실제 smoke와 numerical tests는 Python 3.10.20, NumPy 1.26.4, SciPy 1.12.0, OpenCV 4.9.0.80, PyTorch 2.1.1+cu118, torchvision 0.16.1+cu118, timm 0.9.16 환경을 사용했다. 저장 smoke/병렬 helper는 threadpoolctl도 사용한다. fitted forward에 scikit-learn은 필요 없고 학습 때만 사용한다. 그림 재생성에는 matplotlib, 테스트에는 pytest가 필요하다. 이는 관찰한 환경 정보이며 모든 버전 조합의 호환성 증명은 아니다.

최종 manifest 생성 시 현재 문서, 새 보고서 2개, 최종 그림 경로를 다시 해시하고 정책상 제외 자산이 섞이지 않았는지 확인해야 한다. 이 보고서는 기존 소스·모델·봉인 예측을 변경하지 않았다.

## 새 clone 경로에서의 경로 재배치

현재 config/model 입력에는 기존 `/home/minjae` 작업공간의 절대 경로가 없다. config 4개 경로는 config 파일의 parent, fitted 배열은 MODEL의 parent, ResNet18/Tiny는 현재 모듈에서 계산한 저장소 ROOT, MobileSAM은 boundary config parent를 기준으로 해석한다. 모델의 protocol/feature-seal SHA는 추론 중 추가 artifact를 여는 경로가 아니다. common에 남은 역사적 `/dev/shm` OLD/SCRATCH 상수도 현재 CLI 경로에서는 사용하지 않는다.

새 임시 저장소 경로에 current CLI의 17개 local source와 bound config/model/로컬 asset을 같은 상대 위치로 복사했다. 그 위치에서 새 Python 프로세스로 import, 기본 fitted 배열 load, 모델/자산 SHA 검증을 실행했고 ROOT/ResNet18/Tiny/MobileSAM 경로가 모두 새 저장소 안으로 재배치됨을 확인했다. 기존 작업공간 경로를 요구하지 않았다. 실제 RGB forward나 네트워크 복원까지 실행한 clean-room 시험은 아니다. 이 점검은 소스를 수정하지 않았다.

깊은 `code_snapshot` 디렉터리는 당시의 소스 bytes를 보존한 증거다. 직접 별도 Python package로 실행하면 `parents[3]` ROOT가 기록된 repo layout과 달라질 수 있으므로, 역사적 소스를 재실행할 때는 기록된 repo-relative 경로로 복원한 checkout을 사용해야 한다.
