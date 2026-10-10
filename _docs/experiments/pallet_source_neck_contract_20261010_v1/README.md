# 고정 CAL 16채널 neck 입력 계약 검사

**입력 계약은 PASS이며, 자세 정확도 문제는 미해결입니다.** 현재 고정 Base extractor가 기존 CAL 128장의 기록된 query 위치에서 만든 16개 neck 채널은 원래 FP16 캐시와 모두 일치했습니다. 이 결과는 경계 모델의 유용성이나 실사 위치·회전 개선을 입증하지 않습니다.

결과와 한계는 [RESULT_KO.md](RESULT_KO.md), 고정 조건은 [PROTOCOL.json](PROTOCOL.json), 실제 결과는 [CHECKS.json](CHECKS.json), 영상별 기록 128행은 [ROWS.jsonl.gz](ROWS.jsonl.gz)에 있습니다. 구현은 [check.py](../../../scripts/research/pallet_source_neck_contract_20261010_v1/check.py)입니다.

## 비교 범위

- 기존 분할의 CAL 인덱스 `768..895` 전부를 사용했습니다. 결과를 보고 영상을 골라내지 않았습니다.
- 캐시 `features.npy`의 `features[768:896,:,3:19,:]`를 비교했습니다. 영상당 84개 query × 16채널 × 65개 bin, 총 **11,182,080개 FP16 값**입니다. 채널 `3..10`은 P3의 8개 평균 그룹, `11..18`은 P4의 8개 평균 그룹입니다.
- 기존 Base 좌표로 query를 만들었습니다. 새 Base의 후보 선택·좌표도 별도로 기록해 detector 변화가 숨겨지지 않게 했습니다.
- 원래 `model.py`의 query 및 neck AST만 컴파일했습니다. letterbox affine, bilinear sampling, zero padding, `align_corners=False`, FP16 반올림을 유지했습니다. RGB stencil·초기 자세·ROLE head를 실행하지 않았습니다.

실제 실행은 한 번, 종료 코드는 0입니다. 원본 BGR decode와 Base `predict`는 각각 128회, detector model forward는 내부 lazy warmup 1회를 포함해 **129회**입니다. N3·ROLE·PnP·GT 채점·ray·학습·새 RGB 생성은 모두 0회입니다. 이 검사는 시간 측정이 아닙니다.

실행 원장은 [BUILD_LEDGER.json](BUILD_LEDGER.json)에 있습니다. AST 기반 scope 검사와 함께 freeze 1회·preflight 1회·실제 GPU 비교 1회를 수행했으며, 원래 query/neck fragment의 compile/exec는 실행 본체에서 1회입니다. AST parse는 여러 단계와 함수 내부에서 호출되므로 전체 호출 횟수를 계측하지 않았고 `NA`로 구분합니다.

## 재현에 필요한 입력

Python ≥3.9와 CUDA 환경이 필요합니다. 실제 실행 버전은 Python 3.10.20, NumPy 1.26.4, OpenCV 4.9.0, Torch 2.1.1+cu118, Ultralytics 8.4.60, CUDA 11.8, cuDNN 8700, RTX 3080입니다. 전체 입력 SHA·bytes는 `PROTOCOL.json.authoritative_inputs`에 있습니다.

공개 저장소 외에 다음 읽기 전용 자료가 필요합니다. 공개 receipt만으로 GPU 비교를 재실행할 수는 없습니다.

| 인자 | 필요한 내용 |
|---|---|
| `--source-root` | 원래 CAL PNG들과 `scripts/research/pallet_line_pose_v1/features.py`가 있는 source checkout |
| `--baseline-root` | 같은 SHA의 `features.py`가 있는 baseline checkout |
| `--features` | 원래 `[1024,84,28,65]` FP16 `features.npy` |
| `--cache-manifest` | 해당 원본 PNG·Base 좌표가 기록된 `CACHE_MANIFEST.json` |
| `--base-weights` | 고정 Base `best.pt`; SHA `970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7` |

저장소의 이전 성공한 RGB 계약 protocol/checks, source 분할·준비 receipt, 원래 model/edge/preparation 코드도 필요합니다. 대상·실사 GT를 새로 읽어 query나 검사 조건을 정하지 않습니다.

## 새 출력 디렉터리에서 재현

아래는 독립 재현을 위한 명령 예입니다. 이 문서를 작성하면서 실행한 명령은 아닙니다. 입력 경로를 설정하고 저장소 루트에서 실행합니다. `NECK_RECHECK_OUTPUT`은 저장소와 source/baseline/cache/weights 경로의 바깥에 있는 **새 검사 전용 디렉터리**로 지정합니다.

```bash
export NECK_PYTHON=/path/to/environment/bin/python
export NECK_SOURCE_ROOT=/path/to/source-checkout
export NECK_BASELINE_ROOT=/path/to/baseline-checkout
export NECK_FEATURES=/path/to/readonly-cache/features.npy
export NECK_CACHE_MANIFEST=/path/to/readonly-cache/CACHE_MANIFEST.json
export NECK_BASE_WEIGHTS=/path/to/readonly-weights/best.pt
export NECK_RECHECK_OUTPUT=/dev/shm/neck-contract-independent-check

neck_args=(
  --source-root "$NECK_SOURCE_ROOT"
  --baseline-root "$NECK_BASELINE_ROOT"
  --features "$NECK_FEATURES"
  --cache-manifest "$NECK_CACHE_MANIFEST"
  --base-weights "$NECK_BASE_WEIGHTS"
  --output "$NECK_RECHECK_OUTPUT"
)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$NECK_PYTHON" -B -m scripts.research.pallet_source_neck_contract_20261010_v1.check freeze "${neck_args[@]}"
"$NECK_PYTHON" -B -m scripts.research.pallet_source_neck_contract_20261010_v1.check preflight "${neck_args[@]}"
"$NECK_PYTHON" -B -m scripts.research.pallet_source_neck_contract_20261010_v1.check run "${neck_args[@]}"
```

`freeze`는 현재 저장소의 tracked/sparse 파일과 source checkout 상태를 새 `PROTECTION_BEFORE.json`에 기록하고, 코드·입력·128개 PNG를 해시로 고정한 새 `PROTOCOL.json`을 씁니다. 모델을 import하거나 GPU를 실행하지 않습니다. `preflight`도 모델 실행 없이 그 고정 조건과 보존 상태를 확인합니다. **새 출력에서 `freeze`를 생략하고 기존 protocol만 복사하는 재현 방법은 지원하지 않습니다.** protection의 위치와 SHA도 입력 binding이기 때문입니다.

`run`은 연구 CPU/GPU 경합과 GPU 온도 조건을 확인한 뒤 exclusive `STARTED.json`을 기록하고 실제 비교를 한 번 수행합니다. 데스크탑 rustdesk는 quiet guard의 명시적 예외입니다. 현재 backend flag는 실행 전에 고정되며 결과를 본 뒤 바꾸지 않습니다.

`--output` 기본값은 이 문서의 디렉터리입니다. 이미 완료된 기본 디렉터리에는 재실행할 수 없습니다. `freeze`, `preflight`, `run` 모두 기존 `STARTED.json`, `CHECKS.json`, `ROWS.jsonl.gz`를 덮어쓰지 않습니다. 따라서 `preflight`는 완료된 결과를 재검산하는 명령이 아니라, 아직 시작하지 않은 새 검사 출력의 준비 검사입니다. 출력 디렉터리 및 대상 파일의 symlink와 입력 경로와의 ancestor/descendant 중첩을 거부합니다. 저장소 안의 다른 디렉터리도 출력으로 허용하지 않습니다.

실제 시작 후 불일치·실패가 생기면 그 출력과 claim을 보존합니다. 자동 재시작은 없습니다. bit 또는 수치 불일치가 하나라도 있으면 PASS 조건을 충족하지 못하며 허용오차를 뒤늦게 정하지 않습니다. 다른 버전·GPU·과거 backend flag의 수치 차이가 있을 수 있으므로 독립 재현의 일치를 미리 보장하지 않습니다.

## 결과를 읽는 방법과 한계

`CHECKS.json`의 `channels`는 채널별 bit/numeric mismatch와 절대 차이 평균·중앙값·P90·P99·최댓값입니다. 이번에는 16채널 각각 698,880개 값의 모든 차이가 0입니다. `ROWS.jsonl.gz`에는 원본 PNG·decoded BGR SHA, query SHA, 새 Base 좌표 차이, neck shape/dtype/device, live/cache FP16 SHA와 채널별 차이가 있습니다. 원래 full neck tensor나 private cache는 공개 원행에 다시 저장하지 않았습니다.

이 검사는 현재 extractor와 저장된 source CAL neck의 수치 계약을 확인했습니다. 모든 실사 입력이나 feature의 물리적 의미·domain transfer·경계 소유권을 증명하지 않습니다. N3가 사이에 들어가는 전체 경로의 수치 재생은 이 검사에서 하지 않았습니다. 과거 준비 당시 TF32·cuDNN·라이브러리 버전은 기록되지 않았고, 이번 일치가 그 과거 환경을 복원했다는 뜻도 아닙니다. 원본 private 캐시의 생성 진위를 공개 SHA만으로 인증하지 않습니다. 실제 자세 개선 여부는 별도 실사 평가의 문제로 남아 있습니다.
