# source CAL 16채널 neck 입력 계약 결과

**입력 계약은 PASS이며, 자세 정확도 문제는 미해결입니다.** 현재 고정 Base extractor를 기존 CAL 128장에 실행해 기록된 query 위치의 16개 neck 채널을 원래 source FP16 캐시와 비교한 결과 모든 값이 일치했습니다. 이 검사로 경계 정제 방법이 위치·회전을 개선했다고 결론낼 수 없습니다.

## 실제 결과와 실행 범위

[CHECKS.json](CHECKS.json)은 `passed=true`, `complete=true`, `status=PASS`입니다. 검사 process는 실제 실행 한 번 후 exit 0으로 종료됐습니다. 영상별 [ROWS.jsonl.gz](ROWS.jsonl.gz)는 128행이며, 사후 성능으로 고른 subset이 아닌 원래 CAL 인덱스 `768..895` 전부입니다.

| 항목 | 실제 결과 |
|---|---:|
| 비교 FP16 값 | 11,182,080 = 128 × 84 × 16 × 65 |
| FP16 bit 불일치 / 수치 불일치 | 0 / 0 |
| 절대 차이 평균 / 중앙값 / P90 / P99 / 최댓값 | 모두 0 |
| 채널별 값 수 | 16채널 각각 698,880 |
| fresh Base 선택 후보 불일치 영상 | 0 |
| fresh Base 좌표 불일치 영상 | 0; 사전 고정 `atol=1e-7 px`, `rtol=0` |
| 원본 BGR decode / detector predict / neck stencil | 128 / 128 / 128 |
| 실제 detector model-root / first-layer forward | 129 / 129 |
| 내부 초기화 | lazy warmup 1; constructor model-root forward 0 |
| N3 / ROLE head / 초기·최종 PnP / GT 채점 / ray / 학습 / 새 RGB | 모두 0 |
| 시간 측정 구간 / 새 자세 정확도 집계 | 0 / 없음 |

forward 계수는 단순히 `predict` 호출 수를 옮겨 쓴 값이 아닙니다. 모델 전체 `PoseModel`의 hook은 constructor 이전에 설치했고, first-layer hook은 constructor 이후 설치했습니다. 실제 root forward 129회 전부가 predict 단계에서 발생했으며 first-layer 계수와 일치합니다. lazy warmup은 첫 predict 내부에서 발생합니다.

## 고정된 계산 계약

비교 대상은 `features[768:896,:,3:19,:]`입니다. P3의 8개 channel-mean 그룹과 P4의 8개 그룹을 합친 16채널을 검사했습니다. query는 새로운 좌표로 치환하지 않고 원래 cache manifest의 Base 9점 좌표로 만들었습니다. 새 Base의 선택 후보 및 좌표는 별도로 비교·기록했습니다.

[check.py](../../../scripts/research/pallet_source_neck_contract_20261010_v1/check.py)는 원래 `model.py`의 `query_geometry` 35–45행과 neck stencil 71–80행을 AST로 추출합니다. 원래 model 모듈을 import하거나 그 초기 자세·ROLE head·RGB stencil을 실행하지 않습니다. 현재 source와 baseline extractor의 파일 SHA가 같은지도 확인했습니다. `canvas_affine`, P3/P4 hook, reflected border 100, letterbox, 640 기준 grid, bilinear sampling, zero padding, `align_corners=False`를 유지합니다. FP32 neck 값이 원래 RGB concatenation에서 FP64로 승격된 뒤 FP16으로 반올림되는 순서도 유지했습니다.

PASS 조건은 **FP16 bit 및 수치 불일치 모두 0**으로 비교 전에 고정했습니다. 좌표 검사만 기존 `1e-7 px` parity 허용오차를 사용합니다. neck 값에 결과를 보고 정한 성공 허용오차를 적용하지 않았습니다. 실패 시 자동 재실행이 없고, 이번에는 실패·설정 변경·GPU 재실행이 없었습니다.

| 실행 전 고정 설정 | 값 |
|---|---|
| Torch / OpenCV CPU threads | 4 / 1 |
| `cuda.matmul.allow_tf32` | false |
| `cudnn.allow_tf32` | true |
| `cudnn.benchmark` / `cudnn.deterministic` | false / false |
| OpenCV optimized | true, 실제 receipt에 기록 |
| 실행 환경 | Python 3.10.20; NumPy 1.26.4; OpenCV 4.9.0; Torch 2.1.1+cu118; Ultralytics 8.4.60 |
| GPU backend | RTX 3080; CUDA 11.8; cuDNN 8700 |

원래 preparation의 Torch/OpenCV thread 설정은 1/1로 코드에 남아 있으나, 그 당시 TF32·cuDNN backend flag와 라이브러리 버전은 receipt에 기록되지 않았습니다. 현재 조건에서 원래 FP16 값과 일치했음을 확인했을 뿐, 과거 환경을 알고 복원했다고 주장하지 않습니다.

## 코드·입력·산출물 binding

전체 binding은 [PROTOCOL.json](PROTOCOL.json)의 `authoritative_inputs`와 [CHECKS.json](CHECKS.json)에 있습니다. 공개 경로는 저장소 상대 경로이고, 외부 dependency는 basename·SHA·bytes로 기록했습니다. 원본 PNG 128장의 경로·byte SHA와 decoded BGR SHA도 각 원행과 protocol에 있습니다.

| 대상 | SHA-256 |
|---|---|
| 이번 `check.py` | `0c9576a1a80854cc07c18db2da371e8f2d1feeb981c354e587d309ef6b2e11b5` |
| 고정 `PROTOCOL.json` | `78869f68e7b011c1a9d049d2a21e75e1d2147f86adae7b447c7cbd0df1d68f7d` |
| exclusive `STARTED.json` | `6cbd5a06ce15c66f051a1b48a029ddecf77c8a5e9fe90ffb6996a49bf85cc638` |
| PASS `CHECKS.json` | `b4fb634a594c6933a3a533e23bcd1a8c47a96a51f94d4979ba672963fb2adacb` |
| 128행 `ROWS.jsonl.gz` | `d27bad405dc7f5f5e4f3a045b1b38a84beb304d49c3f3ca0090eaee2b8bbb81c` |
| `PROTECTION_BEFORE.json` | `ea4309d01742827f7dea8ec549a04ab129633bebd57e4c32246a2a2508caa380` |
| source·baseline `features.py` 각각 | `296d81b2adde5ce74192f0012e554ae546c95f309a82dd808024ee34105721ab` |
| 원래 `model.py` | `4e4856abdc3a1c11a920560f1b110d76787975a15ddd637c93c2ec210dbefa3a` |
| fixed Base `best.pt` | `970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7` |
| 기존 `features.npy` | `9b7de2284cbc10a0ac5283c3076d7ba9c886c2e6c3933a76aff81cd9b45787eb` |
| 기존 `CACHE_MANIFEST.json` | `6bcb29e8184d0ca8240b95cff53a92c56ec987c3cf58396a87c25a0a27756316` |

기존 cache 313,098,368 bytes와 manifest 1,540,952 bytes를 읽기 전용으로 사용했습니다. 검사 전후 전체 입력 binding 및 원본 PNG 128장 binding이 그대로였습니다. 당시 snapshot의 기존 저장소 tracked/sparse **16,584개 파일**과 source checkout의 사용자 변경·main HEAD도 보존 검사 PASS입니다. source HEAD는 `7e92fdcefb0a37bdee0aef95d3e86c572e23b967`로 유지됐으며 기존 사용자 status/diff SHA도 일치했습니다.

## 실행량과 의미 한계

이 작업에서는 AST 기반 scope 검사와 함께 freeze 1회, 모델 실행 없는 metadata preflight 1회, 실제 GPU 비교 1회를 수행했습니다. 원래 query/neck fragment의 compile/exec는 실행 본체에서 1회입니다. AST parse는 freeze·preflight·run 준비 검사·run 본체와 함수 내부에서 여러 번 호출되며, 전체 호출 횟수를 별도로 계측하지 않아 [BUILD_LEDGER.json](BUILD_LEDGER.json)에 `NA`로 표시했습니다. 추가 학습·평가·PnP·ray·시간 측정·RGB 생성은 없었습니다. 실행 전후 resource snapshot에서 다른 연구 CPU/GPU process가 없었고, desktop rustdesk는 명시적 예외였습니다. snapshot이 짧은 경합을 모두 배제한다고 주장하지 않으며 latency 자료로 사용하지 않습니다. 모델 hook과 IO canary는 종료 시 복원됐고 cleanup error는 0입니다.

이 결과는 기존 CAL 128장에 대한 현재 Base extractor·neck stencil·source cache 수치 일치를 보여 줍니다. 실사 경계의 물리적 소유권, 여러 선의 공통 편향, neck feature의 정보량, domain transfer 또는 위치·회전 개선을 증명하지 않습니다. N3가 사이에 들어가는 전체 실사 경로를 수치 재생하지 않았고, public SHA가 private cache의 생성 진위를 인증하지도 않습니다. 원래 정확도 목표가 달성됐다고 보고하지 않습니다.

재현은 [README.md](README.md)의 새 외부 `--output` 절차를 따릅니다. 완료된 기본 디렉터리의 code/protocol/receipts/원행을 덮어쓰는 명령은 지원하지 않습니다. 이 README와 보고서 작성 단계에서는 새 모델 실행이나 숫자 재계산을 하지 않았습니다.
