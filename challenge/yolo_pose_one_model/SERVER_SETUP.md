# 학교 서버에서 medium 을 batch 32 로 학습하기

3080(10GB)에서는 `yolo26m-pose` 가 batch 8 이 한계라 nano(batch 32)와 조건이 달랐다.
24GB 서버에서 batch 32 로 돌리면 **nano 와 완전히 같은 조건**이 되어 비교가 깨끗해진다.

## 1. 옮길 것 — 3개

```
challenge/yolo_pose_one_model/datasets/stage_a/          51 GB   ★필수
challenge/weights/pretrained_yolo/yolo26m-pose.pt        49 MB
challenge/yolo_pose_one_model/scripts/  + configs/       1 MB 미만
```

**원본 데이터는 옮기지 않아도 된다.**
```
data/pallet/training_data/paper_release/  (G 40k 원본)   21 GB   불필요
challenge/data/02_synthetic/training/v1, v2 (T 원본)     16 GB   불필요
```
`stage_a` 는 padding·라벨 변환이 끝난 완성본이고 내부 symlink 도 같은 폴더를 가리키므로
원본 없이 자립한다. 약 37 GB 를 아낀다.

## 2. 전송

symlink 는 2026-08-15 에 전부 상대경로로 바꿔 두었다(`relink_relative.py`).
그래서 `-a` 로 그대로 옮기면 서버에서도 정상 동작한다.

```bash
rsync -avP \
  challenge/yolo_pose_one_model/datasets/stage_a \
  계정@서버:/서버경로/challenge/yolo_pose_one_model/datasets/

rsync -avP challenge/weights/pretrained_yolo/yolo26m-pose.pt \
  계정@서버:/서버경로/challenge/weights/pretrained_yolo/

rsync -avP challenge/yolo_pose_one_model/scripts challenge/yolo_pose_one_model/configs \
  계정@서버:/서버경로/challenge/yolo_pose_one_model/
```

★ 만약 rsync 가 `-a` 없이 실행돼 symlink 가 실파일로 복사되면 용량이 약 14 GB 늘어난다.
   문제는 없지만 디스크가 빠듯하면 주의.

### 전송 검증 (서버에서)
```bash
cd /서버경로/challenge/yolo_pose_one_model
for d in images/train labels/train images/val labels/val; do
  echo "$d: $(ls datasets/stage_a/$d | wc -l)"
done
# 73916 / 73916 / 4009 / 4009  이어야 한다

# 깨진 symlink 가 없는지
find datasets/stage_a -xtype l | wc -l      # 0 이어야 한다
```

## 3. 서버 환경

```bash
conda create -n pallet-yolo26 python=3.10 -y
conda activate pallet-yolo26
pip install "ultralytics==8.4.60"
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 드라이버에 맞춰
python -c "import torch,ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
```
CUDA toolkit 은 설치할 필요 없다 — torch wheel 이 런타임을 번들한다. **드라이버만** 맞으면 된다
(Ada 계열이면 ≥525).

## 4. data yaml 의 경로 갱신

`configs/stage_a.yaml` 의 `path` 는 이 머신 절대경로가 박혀 있다. 서버에서 재생성한다.
```bash
python scripts/build_stage_dataset.py --stage stage_a --yaml-only
```

## 5. GPU 확인 후 학습

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```
GPU1 만 써야 한다면 `CUDA_VISIBLE_DEVICES=1` 로 **GPU0 을 아예 안 보이게** 막는다.

```bash
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/train_stage_a_m.sh 32 > runs/m_b32.log 2>&1 &
```

### batch 판단 기준 (3080 실측에서 외삽 [추정])
```
GPU1 free      권장 batch
────────────────────────────
  ~9 GB          8
 ~12 GB         16
 ~18 GB         32
 ~24 GB         32  (48은 경계)
```

★ **Ultralytics 는 OOM 이 나면 batch 를 스스로 낮춘다** (32→16→8, 3회까지).
   그러면 run 이름·`args.yaml` 은 32 인데 실제는 8 이 된다. 반드시 확인할 것:
```bash
grep -c "Reducing to batch" runs/m_b32.log     # 0 이어야 정상
# step 수로도 검증: 73916/batch = batch32 면 2310, b16 이면 4620, b8 이면 9240
```

## 6. 완료 후 가져올 것

```
runs/stage_a_m_640_b32_seed42/weights/    best.pt · last.pt · epoch*.pt   (약 1.5 GB)
runs/stage_a_m_640_b32_seed42/results.csv args.yaml
```
가중치만 가져오면 이 머신에서 평가·비교할 수 있다.

## 참고 — 이 머신에서 이미 확인된 것

```
nano   yolo26n-pose  batch 32  8.45 h   Pose mAP50-95 0.9595 (60 epoch 완주)
medium yolo26m-pose  batch 32  OOM (probe 8.34 GiB 로 "통과" 했으나 실학습 실패)
                     batch 16  OOM (깨끗한 GPU + expandable_segments 에서도)
                     batch  8  정상 6.29 GiB · epoch 1,650 s · 60 epoch 약 25 h
```
서버 batch 32 가 성공하면 nano 와 유효 batch(64)·BN 통계까지 동일해진다.
