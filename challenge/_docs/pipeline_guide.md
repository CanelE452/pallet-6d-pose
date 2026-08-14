# Pseudo-label + Finetune 파이프라인 가이드

`challenge/scripts/` 의 세 도구를 묶어 self-training 한 사이클을 자동 실행한다.

```
manual GT  ─┐
            ├─► STAGE 1: 1차 ft (manual 만)
            │   └─► stage1 weight
            │
            └─► STAGE 2: stage1 모델로 pseudo-label 생성 (다른 시퀀스)
                │
                └─► pseudo GT
                    │
                    ├─► STAGE 3: 2차 ft (manual + pseudo)
                    │   └─► stage2 weight (최종)
                    │
                    └─► challenge/weights/finetuned/ 에 정리
```

## 파일 구성

```
challenge/scripts/
├── make_pseudo_gt.py   시퀀스 → pseudo NDDS GT (sanity gate 통과 frame만)
├── merge_dataset.py    여러 GT 디렉토리 → train/val NDDS 데이터셋
└── run_pipeline.sh     위 3 단계 자동 실행 (사용자 자리 비울 때 1줄 실행)
```

## 가장 흔한 사용 (한 줄)

```bash
bash challenge/scripts/run_pipeline.sh
```

내부 default 설정:
```
MANUAL_GT       = challenge/data/01_real/manual_gt/capturepallet07_manual_gt
PSEUDO_SEQ      = data/outside/capturepallet09         (pseudo-label 만들 시퀀스)
BASELINE        = challenge/weights/baseline_v8_A.pth
STAGE1_EP       = 15  (manual 만 짧게)
STAGE2_EP       = 15
STRIDE_PSEUDO   = 3   (3 frame 마다 1개 추론 → 시간 절감)
MAX_REPROJ_PSEUDO = 8.0
```

다른 시퀀스로 바꿔 실행:
```bash
PSEUDO_SEQ=data/outside/capturepallet11 bash challenge/scripts/run_pipeline.sh
```

여러 manual 디렉토리 사용하려면 스크립트 내 `--inputs` 부분을 직접 늘리거나, 한
디렉토리로 미리 합쳐두는 것이 편하다.

## 단계별 수동 실행 (디버깅용)

### Step 1 — manual GT 만으로 1차 ft

```bash
# 데이터셋 준비
python challenge/scripts/merge_dataset.py \
    --inputs challenge/data/01_real/manual_gt/capturepallet07_manual_gt \
    --out challenge/data/03_derived/_train_stage1 \
    --val_fraction 0.2

# 학습
bash scripts/train_dope.sh --finetune \
    --net_path challenge/weights/baseline_v8_A.pth \
    --train_dir challenge/data/03_derived/_train_stage1/train \
    --val_dir   challenge/data/03_derived/_train_stage1/val \
    --exp_name challenge_ft_stage1 \
    --struct_loss --struct_coord 0.003
```

### Step 2 — 1차 모델로 pseudo-label 생성

```bash
python challenge/scripts/make_pseudo_gt.py \
    --seq data/outside/capturepallet09 \
    --weights weights/challenge_ft_stage1/final_net_epoch_NNNN.pth \
    --out_dir challenge/data/01_real/pseudo_gt/capturepallet09_pseudo_gt \
    --stride 3 --min_kp 6 --max_reproj 8
```

옵션:
- `--stride N` : N frame 마다 추론 (1=모든 frame, 3=1/3)
- `--max N` : 최대 N frame 처리
- `--min_kp / --max_reproj / --thr / --z_max` : gate 임계값 override

저장 경로 + overlay sample (`_overlay/`) 검증.

### Step 3 — manual + pseudo 병합 + 2차 ft

```bash
python challenge/scripts/merge_dataset.py \
    --inputs challenge/data/01_real/manual_gt/capturepallet07_manual_gt \
             challenge/data/01_real/pseudo_gt/capturepallet09_pseudo_gt \
    --out challenge/data/03_derived/_train_stage2 \
    --val_fraction 0.15 \
    --manual_to_val_only          # manual 일부를 held-out val 로

bash scripts/train_dope.sh --finetune \
    --net_path weights/challenge_ft_stage1/final_net_epoch_NNNN.pth \
    --train_dir challenge/data/03_derived/_train_stage2/train \
    --val_dir   challenge/data/03_derived/_train_stage2/val \
    --exp_name challenge_ft_stage2 \
    --struct_loss --struct_coord 0.003
```

## Pseudo-label 품질 관리

`make_pseudo_gt.py` 는 `challenge/config/task.yaml` 의 `inference.gates` 를 그대로
사용 (기본 — `min_kp=7, reproj=8px, z=[0.3,5.0]m`).

baseline 이 outdoor 에서 약하므로 첫 round 의 pseudo-label 은 거의 안 만들어질
수 있다. 그래서 파이프라인은 manual 만으로 한 번 ft 한 뒤 그 모델로 pseudo 를
시도한다. stage1 모델은 outdoor 에 약간 더 강해서 pseudo 가 잘 나옴.

만약 그래도 부족하면 gate 완화:
```bash
python challenge/scripts/make_pseudo_gt.py --seq ... --min_kp 5 --max_reproj 12
```

너무 완화하면 false positive 가 학습에 들어가서 collapse 위험. 권장 minimum:
`min_kp=5, max_reproj=12`.

## 결과 검증

학습 끝나면 `challenge/weights/finetuned/` 에 가중치가 정리됨. live GUI 로 확인:

```bash
python challenge/scripts/run_live.py \
    --seq data/outside/capturepallet11 --seq_fps 5 --seq_loop \
    --weights challenge/weights/finetuned/final_net_epoch_NNNN.pth
```

수치 비교 (baseline vs ft):
```bash
# baseline
python challenge/scripts/seq_stats.py --all --thr 0.10 --min_kp 4 --stride 10 --max 100

# ft
python challenge/scripts/seq_stats.py --all --thr 0.10 --min_kp 4 --stride 10 --max 100 \
    --weights challenge/weights/finetuned/final_net_epoch_NNNN.pth
```

## 출력 / 로그 위치

```
challenge/data/03_derived/_train_stage1/{train,val}/   stage1 학습용 NDDS
challenge/data/03_derived/_train_stage2/{train,val}/   stage2 학습용 NDDS (manual+pseudo)
challenge/data/<seq>_pseudo_gt/             생성된 pseudo GT
challenge/data/<seq>_pseudo_gt/_overlay/    pseudo 검증 overlay 이미지
weights/challenge_ft_stage1/                stage1 학습 weight
weights/challenge_ft_stage2/                stage2 학습 weight
challenge/weights/finetuned/                최종 정리된 weight 사본
challenge/_docs/pipeline_runs/YYYYMMDD_HHMMSS/   각 run 의 로그
```

## 주의사항

- baseline 학습 데이터는 dims = (1.1, 1.1, 0.15) 인데 challenge 는 (1.1, 1.3, 0.11).
  ft 시 keypoint 회귀만 학습되므로 dims 직접 영향은 없지만, **PnP 단계에서는 dims
  가 정확해야** depth/3D pose 가 맞는다. `task.yaml` 이 이미 새 dims 로 갱신됨.
- pseudo-label gate 가 너무 보수적이면 stage2 가 stage1 과 동일해질 수 있음. log
  의 `Pseudo GT: N frames` 확인 후 N < 10 이면 stage2 건너뜀.
- ft 가 끝나면 `challenge/weights/finetuned/` 의 최신 epoch 가중치를 `task.yaml`
  의 `baseline.weights` 로 갱신하면 그 다음부터 run_live 가 ft 모델 사용.
- 학습 중에 자리 비워도 ntfy.sh 알림이 `train_dope.sh` 안에서 전송됨 (`config/default.yaml` 의 `notification.ntfy_topic`).
