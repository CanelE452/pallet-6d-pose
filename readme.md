# Pallet 6D Pose Estimation — Geometry-aware Self-Training

팔레트 6D 포즈 추정을 위한 기하학적 제약 기반 준지도 도메인 적응 프레임워크.

## 2026-10-01 큰 pose 오류의 학습 비용을 반영한 선택기 검증

입력은 **RGB 이미지 한 장과 팔레트 치수**, 기존 카메라 보정 K입니다. 학습 시 R0보다 큰 T/R 오류에 비용을 더한 선택기 4개는 합성 검증 **45/45 통과** 후 실사 173장 전체를 평가했습니다. 자연 가림 99장의 세 seed 중앙값 평균은 R0 대비 T **12.403→12.081cm**, R **5.218→5.224°**입니다. 실사 안정성 조건은 **2/5 통과**로, **T·R의 안정적인 동시 개선은 아직 미달성입니다.**

- [상세 결과·실제 예측 이미지·치수·학습 및 T/R 그래프](_docs/experiments/pallet_pose_anchor_risk_20261001_v1/REPORT_KO.md)
- [13개 모델×실사 173장 전체 결과 2,249행](_docs/experiments/pallet_pose_anchor_risk_20261001_v1/REAL_FRAME_RESULTS.csv)
- [실제 네 선택기의 최종 파라미터](_docs/experiments/pallet_pose_anchor_risk_20261001_v1/model_parameters/)

## 2026-10-01 R0와의 관계를 입력한 선택기 검증

기존 특징에 R0 기준 후보와의 차이 및 기준 후보 여부를 추가한 189차원 선택기 4개를 실제 학습했습니다. 합성 VAL의 검증은 38/45에서 **43/45 통과**로 나아졌지만, seed 3의 T P90 조건 두 개가 실패했습니다. **실사 T·R의 안정적인 동시 개선은 아직 미달성입니다.**

- [상세 결과·T/R 비교·학습 그래프·이미지와 치수](_docs/experiments/pallet_pose_anchor_context_20261001_v1/REPORT_KO.md)
- [전체 8192개 source 결과](_docs/experiments/pallet_pose_anchor_context_20261001_v1/SOURCE_VAL_FRAME_RESULTS.csv)
- [실제 네 모델의 최종 파라미터](_docs/experiments/pallet_pose_anchor_context_20261001_v1/model_parameters/)

## 2026-10-01 R0 양축 보존 타깃의 실제 학습

R0보다 T와 R을 모두 악화시키지 않는 후보로 학습 정답을 제한했습니다. 참조를 보는 실사 진단은 원래5개 기준을 모두 통과했지만, 새로 학습한 선택기4개는 합성 VAL에서38/45 통과·7개 실패였습니다. **실제 모델의 안정적인 T·R 동시 개선은 아직 미달성입니다.**

- [전체 결과·이미지와 치수·실제 학습 그래프](_docs/experiments/pallet_pose_pareto_anchor_20261001_v1/REPORT_KO.md)
- [실제 네 선택기의 최종 파라미터](_docs/experiments/pallet_pose_pareto_anchor_20261001_v1/model_parameters/)
- [합성 VAL 전체8192행](_docs/experiments/pallet_pose_pareto_anchor_20261001_v1/SOURCE_VAL_FRAME_RESULTS.csv)

## 2026-10-01 실제 이미지·치수와 후보 개선 가능성

입력은 **RGB 이미지 한 장과 팔레트 치수**이며 기존 카메라 보정 K를 유지합니다. 참조를 이용해 기존 후보를 고르는 진단에서, 현재 학습 비용을 완벽하게 최소화해도 자연 가림 T P90이129.60cm로 원래 허용 한계126.49cm를 넘었습니다. **실제 모델의 안정적인 T·R 동시 개선은 아직 달성하지 못했습니다.**

- [상세 수치·실제 RGB6장·치수·후보 투영 이미지](_docs/experiments/pallet_pose_real_union_feasibility_20261001_v1/REPORT_KO.md)
- [두 진단×3seed×173장 전체 CSV](_docs/experiments/pallet_pose_real_union_feasibility_20261001_v1/FRAME_RESULTS.csv)
- [독립 수치 검증과 판정 한계](_docs/experiments/pallet_pose_real_union_feasibility_20261001_v1/VERIFICATION_KO.md)

## 2026-10-01 후보 쌍별 감독 구조 검증

같은94특징·네 후보에서 W/D 비교와 원본/보정 결과 비교를 함께 학습했습니다. 신규3개 fit은 수렴했지만 합성 검증45개 조건 중16개가 실패했습니다. TRAIN에서도 쌍별 정확도의 작은 개선이 최종 후보 선택으로 이어지지 않았습니다. **실사 T·R의 안정적 동시 개선은 아직 달성하지 못했습니다.**

- [직전 학습 결과·구조도·전체 T/R 그래프](_docs/experiments/pallet_pose_selector_pairwise_20261001_v1/REPORT_KO.md)
- [독립 TRAIN 수렴·최종 선택 검증](_docs/experiments/pallet_pose_selector_pairwise_20261001_v1/TRAIN_CONVERGENCE_KO.md)
- [source VAL1024장×8모델 전체 CSV](_docs/experiments/pallet_pose_selector_pairwise_20261001_v1/SOURCE_VAL_FRAME_RESULTS.csv)

## 2026-10-01 선택기 수렴 통제와 실패 원인 검증

입력은 **이미지 한 장과 팔레트 치수**이며 기존 카메라 보정 정보를 유지합니다. 선택기4개를 실제 학습해 수렴을 확인했지만, 합성 검증45개 조건 중4개가 실패했습니다. **실사 T·R의 안정적 동시 개선은 아직 달성하지 못했습니다.** 아래에 전체 수치·그래프·학습 파라미터와 독립 검증을 공개합니다.

- [직전 수렴 통제 결과·인증·T/R 그래프](_docs/experiments/pallet_pose_selector_convergence_20261001_v1/REPORT_KO.md)
- [후보 선택·학습 목적함수·검출 데이터의 실패 원인](_docs/experiments/pallet_pose_selector_objective_audit_20261001_v1/REPORT_KO.md)
- [source VAL1024장×8모델 전체 결과 CSV](_docs/experiments/pallet_pose_selector_convergence_20261001_v1/SOURCE_VAL_FRAME_RESULTS.csv)
- [실제 최종 선택기4개의 파라미터](_docs/experiments/pallet_pose_selector_convergence_20261001_v1/model_parameters/)

## 2026-10-01 원래 pose와 보정 pose의 공동 선택 검증

**실사 T·R의 안정적 동시 개선은 아직 달성하지 못했습니다.** 원래 R0 후보를 보존하면서 보정 모델의 후보를 함께 고르는 방법을 합성 데이터로 검증합니다. 상세 보고서는 합성 정답으로 고르는 진단적 가능성, 실제 학습한 선택기의 검증 결과, 기존 실사 결과를 구분하고 전체 수치·이미지·실행 기록을 제공합니다.

- [직전 선택기 검증 결과·현재 T/R 상태·이미지](_docs/experiments/pallet_pose_union_selection_20261001_v1/REPORT_KO.md)
- [합성 데이터의 좌표계·대칭성·기하 오류 감사](_docs/experiments/pallet_pose_union_selection_20261001_v1/SOURCE_CONTRACT_KO.md)
- [캐시 재현 검사와 실행 순서 수정 이력](_docs/experiments/pallet_pose_union_selection_20261001_v1/RUNTIME_AND_SOURCE_METHOD_KO.md)
- [새 선택기 6개의 실제 학습 파라미터 JSON](_docs/experiments/pallet_pose_union_selection_20261001_v1/model_parameters/)

## 2026-10-01 T·R 안정적 개선 대조 실험

같은 251장·학습량에서 단일 촬영과 여러 촬영의 데이터 구성을 비교하고, 각 군을 3개 seed로 반복했습니다. **6회 학습은 완료됐지만 T·R의 안정적인 동시 개선에는 실패했습니다.** 아래 보고서에 T·R 중앙값·P90·실패 수, 사전 안정성 기준 판정과 모든 자연 가림 사례를 공개합니다.

- [상세 결과·판정·그래프](_docs/experiments/pallet_pose_stable_improvement_20261001_v1/REPORT_KO.md)
- [자연 가림 99장 전체 비교](_docs/experiments/pallet_pose_stable_improvement_20261001_v1/GALLERY_NATURAL99.md)
- [프레임별 전체 수치 CSV](_docs/experiments/pallet_pose_stable_improvement_20261001_v1/FRAME_RESULTS.csv)
- [실행 계약과 입력 제외 이력](_docs/experiments/pallet_pose_stable_improvement_20261001_v1/PROTOCOL_EFFECTIVE.json)
- [현재 개선 상태·검출 오류 이미지·후보 선택의 한계](_docs/experiments/pallet_pose_joint_recovery_20261001_v1/REPORT_KO.md)

## 2026-09-30 clean→가림 pose 진단

현재 natural99·clean29의 동일 T/R 평가, 실제 입력 이미지와 보정 전후 비교, 지시문 이행 검증을 함께 공개합니다. 새 학습은 0회이며 아래 결과는 반복 사용 DEV의 진단입니다.

- [상세 결과 보고서와 그래프](_docs/experiments/pallet_pose_diagnosis_20260930_v1/REPORT_KO.md)
- [자연 가림 99장 전체 이미지 비교](_docs/experiments/pallet_pose_diagnosis_20260930_v1/GALLERY_NATURAL99.md)
- [입력·모델·평가 계약](_docs/experiments/pallet_pose_diagnosis_20260930_v1/INPUTS_AND_METHOD_KO.md)
- [누락 보완·검증 및 남은 한계](_docs/experiments/pallet_pose_diagnosis_20260930_v1/REVIEW_CORRECTIONS_KO.md)
- [전체 문서·수치·재현 안내](_docs/experiments/pallet_pose_diagnosis_20260930_v1/README.md)

**3단계 파이프라인:**
1. Isaac Sim 합성 데이터 생성 + DOPE 학습
2. Geometric Filter + Pseudo-label 생성
3. Fine-tuning + Self-Training 반복

## Pre-trained Weights

학습된 weight 는 Hugging Face Hub 에 공개되어 있습니다 — `weights/` 폴더는 이 repo 에 포함되지 않으니 아래에서 다운로드하세요.

**Repo:** [`CanelE452/pallet-pose-dope-weights`](https://huggingface.co/CanelE452/pallet-pose-dope-weights)

```
파일                                                      설명                          NN <20px
─────────────────────────────────────────────────────────────────────────────────────────────────
mixed_v8/final_net_epoch_0060.pth                         Synthetic-only baseline       18.9%
v8_ablation_C_coord_edge/final_net_epoch_0065.pth         Loss ablation BEST            38.4%
                                                          (coord + edge log-ratio)
f5_noapril_ransac_loo_realonly/final_net_epoch_0096.pth   Self-training BEST (F5)       60.5%  ★
                                                          (RANSAC + LOO filter)
```

다운로드:
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='CanelE452/pallet-pose-dope-weights',
    local_dir='weights',
)"
```

또는 개별 파일:
```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='CanelE452/pallet-pose-dope-weights',
    filename='f5_noapril_ransac_loo_realonly/final_net_epoch_0096.pth',
    local_dir='weights',
)"
```

## Quick Start

```bash
# 환경 설정
conda create -n pallet-pose python=3.10
conda activate pallet-pose
pip install -r requirements.txt

# Weight 다운로드 (위 섹션 참조)

# Step 1: 합성 데이터 생성 (Isaac Sim 필요)
bash scripts/data_prep/isaac_sim/generate_all.sh

# Step 1: DOPE 학습
bash scripts/train_dope.sh

# Step 1: Fine-tune (기존 weight에서 이어서)
bash scripts/train_dope.sh --finetune

# 평가 (synthetic val)
python scripts/data_prep/eval/evaluate_on_val.py \
    --weights weights/misc/f5_noapril_ransac_loo_realonly/final_net_epoch_0096.pth \
    --val_dir data/pallet/training_data/val
```

## 디렉토리 구조

```
FoundationPose/
├── config/
│   ├── default.yaml                  # 전체 설정 (모델, 학습, 카메라, 팔레트 스펙)
│   └── stage3_selftrain.yaml         # Self-training 하이퍼파라미터
├── scripts/
│   ├── data_prep/
│   │   ├── isaac_sim/                # Isaac Sim 합성 데이터 파이프라인
│   │   │   ├── gen_replicator_data.py    메인 생성 스크립트
│   │   │   ├── generate_all.sh           배치 생성 (64프레임/배치, 자동 재시작)
│   │   │   ├── sdg_config.py             설정 상수
│   │   │   ├── sdg_scene.py              씬 조립 (warehouse, 조명, 배경)
│   │   │   ├── sdg_distractors.py        적재물/디스트랙터 배치
│   │   │   ├── sdg_usd_xform.py          USD 에셋 조작
│   │   │   ├── sdg_math.py               좌표 변환, 카메라 매트릭스
│   │   │   └── sdg_annotation.py         NDDS JSON 생성
│   │   ├── blender/                  # Blender 렌더링 파이프라인 (대안)
│   │   ├── evaluate_on_val.py        # 평가 (PCK@3/5/10px, PnP Reproj)
│   │   ├── visualize_annotations.py  # GT keypoint overlay 시각화
│   │   ├── visualize_inference.py    # 추론 결과 시각화 (belief map + keypoint)
│   │   ├── merge_and_validate.py     # 배치 병합 + JSON 검증
│   │   └── verify_keypoints.py       # 키포인트 기하 검증
│   ├── self_training/                # Self-Training 파이프라인
│   │   ├── self_train.py                 메인 루프
│   │   ├── geometric_filter.py           3단계 기하 필터
│   │   ├── pnp_solver.py                 EPnP + RANSAC
│   │   ├── augmentations.py              Weak/Strong augmentation
│   │   └── metrics.py                    6D 포즈 메트릭
│   ├── dope/
│   │   └── run_dope_live.py          # 실시간 추론 (RealSense D435i, native)
│   ├── train_dope.sh                 # DOPE 학습 스크립트 (config 기반)
│   └── launch_tensorboard.py         # TensorBoard 실행
├── Deep_Object_Pose/                 # DOPE 구현 (VGG-19 backbone)
│   ├── train/train.py                    학습 루프
│   ├── common/models.py                  네트워크 정의
│   └── common/utils.py                   데이터 로더 (CleanVisiiDopeLoader)
├── data/pallet/
│   ├── training_data/
│   │   ├── train/                    # 병합된 학습 데이터 (NDDS 포맷)
│   │   └── val/                      # 검증 데이터
│   ├── real_data/                    # 실제 이미지 (~1924장, RealSense D435i)
│   ├── hdri/                         # HDRI 배경 (5종 산업 환경)
│   └── models_usd/                   # USD 팔레트 모델 (4종)
├── weights/
│   ├── pallet_category/              # Pretrain weight (ep60)
│   ├── pallet_v11/                   # Fine-tune weight (ep91)
│   └── pallet_v11_far/              # 원거리 보강 weight (ep121) ← 최신
└── _docs/                            # 연구 설계 문서
```

## 환경 설정

### 필수 요구사항

- Python 3.10+
- NVIDIA GPU (CUDA 11.8+, 8GB+ VRAM)
- conda

### Python 환경

```bash
conda create -n pallet-pose python=3.10
conda activate pallet-pose
pip install -r requirements.txt
```

주요 패키지: PyTorch 2.1.1+cu118, albumentations, opencv, open3d, trimesh

### Isaac Sim (합성 데이터 생성 시 필요)

- **버전**: Isaac Sim 4.5.0
- Isaac Sim 내장 Python으로 실행 (conda 환경 아님)
- 필수 환경변수:
  ```bash
  export OMNI_KIT_ACCEPT_EULA=YES
  export CUDA_MODULE_LOADING=LAZY
  export PYTHONUNBUFFERED=1
  ```

### 플랫폼별 주의사항

| 항목 | Windows | Ubuntu |
|------|---------|--------|
| `config/default.yaml` → `paths.python_exe` | `C:/Users/.../python.exe` | conda python 경로로 변경 |
| `train.workers` | `0` (필수) | `4` 이상 가능 |
| `generate_all.sh` 프로세스 정리 | `wmic`/`taskkill` | `pgrep`/`kill` |
| Isaac Sim | standalone 설치 | standalone 설치 |

## 설정 파일

### `config/default.yaml` — 전체 설정

모든 학습/평가 파라미터를 관리하는 단일 설정 파일.

```yaml
model:
  input_size: 448               # 네트워크 입력 해상도
  num_keypoints: 9              # 8 cuboid corners + 1 centroid

train:
  pretrain:                     # scratch 학습
    epochs: 60, batch_size: 4, lr: 1e-4
  finetune:                     # fine-tune
    epochs: 91, batch_size: 4, lr: 5e-5
  sigma: 4.0                   # belief map Gaussian std (>=2 유지)
  workers: 0                   # Windows: 0, Ubuntu: 4+

pallet:
  width: 1.1                   # KS T-11 규격 (meters)
  depth: 1.1
  height: 0.15

camera:                         # RealSense D435i 내부 파라미터
  fx: 615, fy: 615, cx: 320, cy: 240
```

### `config/stage3_selftrain.yaml` — Self-Training

```yaml
geometric_filter:
  tau_reproj: 5.0               # Reproj error 임계값 (px)
  tau_ratio_min/max: [0.5, 2.0] # 변 길이 비율
  min_keypoints: 5              # PnP 최소 keypoint 수
```

## 파이프라인 상세

### Step 1: 합성 데이터 생성

Isaac Sim Replicator로 NDDS 포맷 합성 이미지 생성.

```bash
# 단일 실행
python scripts/data_prep/isaac_sim/gen_replicator_data.py \
    --renderer PathTracing \
    --num_frames 100 \
    --output_dir data/pallet/training_data/test \
    --seed 42 \
    --hdri_dir data/pallet/hdri

# 배치 실행 (64프레임/배치, 자동 재시작)
bash scripts/data_prep/isaac_sim/generate_all.sh
```

**현재 데이터 구성:**
- 기존 근거리 2000장 + v11 리프터 시점 2000장 + 원거리 2000장 = **6000장 train**
- val 1500장

**카메라 3모드** (리프터 마운트 60%, 높은 시점 25%, 바닥 레벨 15%):
- Mode A: h=0.2~0.5m, dist=1.5~8.0m (리프터 포크 마스트)
- Mode B: h=0.5~1.2m, dist=1.0~6.0m (운전석/점검)
- Mode C: h=0.05~0.3m, dist=1.0~5.0m (바닥 레벨)

### Step 1: DOPE 학습

```bash
# Scratch 학습 (60 epochs)
bash scripts/train_dope.sh

# Fine-tune (기존 weight에서 이어서)
bash scripts/train_dope.sh --finetune

# 특정 weight에서 fine-tune
bash scripts/train_dope.sh --finetune --net_path weights/custom/net.pth
```

`train_dope.sh`는 `config/default.yaml`에서 모든 설정을 읽음.

### Step 2-3: Self-Training

```bash
python scripts/self_training/self_train.py \
    --config config/stage3_selftrain.yaml
```

3단계 Geometric Filter:
- A: Augmentation Consistency (약한 aug 간 keypoint 일관성)
- B: 변 길이 일관성 (대각 변 비율 0.5~2.0)
- C: 물리적 크기 규격 비율 (팔레트 1.1m 기준)

### 평가

```bash
python scripts/data_prep/evaluate_on_val.py \
    --weights weights/misc/pallet_v11_far/final_net_epoch_0121.pth \
    --val_dir data/pallet/training_data/val \
    --output_dir data/pallet/eval_results/latest
```

**메트릭:**
- PCK@3/5/10px — keypoint 정확도
- PnP 성공률 — 6D 포즈 복원 가능 비율
- Reproj error — PnP 재투영 오차 (px)

### 시각화

```bash
# GT annotation overlay
python scripts/data_prep/visualize_annotations.py \
    --data_dir data/pallet/training_data/val

# 추론 결과 시각화 (belief map + keypoint + cuboid)
python scripts/data_prep/visualize_inference.py \
    --weights weights/misc/pallet_v11_far/final_net_epoch_0121.pth \
    --num_syn 10 --num_real 10
```

## 현재 학습 결과

| 메트릭 | Pretrain (ep60, 2K) | v11 (ep91, 4K) | v11_far (ep121, 6K) |
|--------|---------------------|----------------|---------------------|
| PCK@3px | 48.2% | 51.5% | **56.6%** |
| PCK@5px | 52.9% | 55.7% | **59.7%** |
| PCK@10px | 58.2% | 60.6% | **63.8%** |
| PnP 성공률 | 78% | 84.5% | 82.5% |
| Reproj mean | 182px | 140px | **110px** |
| Real kps 평균 | ~3.5/9 | 4.4/9 | **6.3/9** |

**Best weight**: `weights/misc/pallet_v11_far/final_net_epoch_0121.pth`

## 실시간 추론 (RealSense D435i)

Intel RealSense SDK + pyrealsense2 만 있으면 native 로 실행 가능 (Docker 불필요).

```bash
# 1. RealSense SDK 설치 (한 번만, Windows installer)
#    https://www.intelrealsense.com/sdk-2/

# 2. Python 바인딩
conda activate pallet-pose
pip install pyrealsense2

# 3. 카메라 USB 연결 후 실행
python scripts/dope/run_dope_live.py \
    --realsense \
    --weights weights/misc/f5_noapril_ransac_loo_realonly/final_net_epoch_0096.pth
```

키 조작: `q`=종료, `s`=프레임 저장, `b`=belief map 토글, belief 클릭 → keypoint 자동 threshold 튜닝

## Gotchas

- **sigma=4.0 유지** — sigma<1은 gradient vanishing 발생
- **Isaac Sim ~2분/프레임** — 메모리 누수로 64프레임마다 자동 재시작
- **Windows workers=0** — multiprocessing fork 미지원, config에서 설정
- **밝기 skip**: mean<40 또는 mean>240이면 프레임 폐기
- **조명 범위**: DomeLight 2000-3500, Main light 100K-300K (과다 밝기 방지)
- **ORIENTATION_OVERRIDES 수정 금지** — 팔레트 모델별 보정 값 (검증 완료)

## 연구 문서

현재 논문 실험 표:

- `_docs/paper/final/generated/` — 최종 논문 표 (Table 1~4 + diagnostic)
- `_docs/archive/paper_pre_final_20260903/legacy_paper_outputs/evaluation_tables/` — SUPERSEDED 옛 템플릿
- `_docs/archive/paper_support_20260830/` — 표 밖의 이전 support 문서 보관본

아래 `_docs/method/`·`_docs/preprocessing/` 문서는 과거 세대 참고자료이며 현재
paper contract나 keypoint convention을 정의하지 않는다:

- `method/overview.md` — 전체 파이프라인 설계
- `method/step1_synthetic_data.md` — 합성 데이터 생성 가이드
- `method/step2_geometric_filter.md` — Geometric Filter 설계
- `preprocessing/keypoint_definition.md` — legacy Y=UP 키포인트 기록
- `survey/survey-6d-pose-estimation.md` — 6D Pose 분야 서베이
