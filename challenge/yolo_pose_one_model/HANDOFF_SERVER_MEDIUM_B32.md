# 인수인계 — 학교 서버에서 yolo26m-pose 를 batch 32 로 학습

이 문서 하나만 읽고 끝까지 진행할 수 있게 썼다. 작성 2026-08-15, 작성자 = 로컬(RTX 3080) 세션.
받는 쪽 = 학교 서버(24GB GPU) 세션.

---

## 0. 한 줄 요약

**로컬에서 만든 `datasets/stage_a`(51GB)를 서버로 옮기고, `yolo26m-pose` 를 batch 32 로
60 epoch 학습한 뒤, `weights/` 만 로컬로 돌려보내면 된다.**

---

## 1. 배경 — 왜 서버에서 다시 돌리나

과제용 팔레트 6D pose 모델. 리프터가 정지 후 RealSense RGB 한 장에서 팔레트 bbox + 9 keypoint 를
예측하고, 기존 배포 후처리(PnP + depth)가 yaw / lateral offset / forward distance 로 바꾼다.

```
이미 끝난 것 (로컬 RTX 3080 10GB)
  nano   yolo26n-pose  batch 32   8.45 h   60 epoch 완주   Pose mAP50-95 0.9595
못 한 것
  medium yolo26m-pose  batch 32   OOM      10GB 로는 불가
                       batch 16   OOM
                       batch  8   가능하나 nano 와 조건이 달라짐(BN 통계)
```

★ 목적은 **nano 와 완전히 같은 조건(batch 32)** 의 medium base 를 만드는 것이다.
   nano 가 real 에서 정확도가 부족할 경우의 대안으로 쓴다.
   이번 라운드는 **합성 데이터만** 학습한다. real 데이터는 이후 finetune 단계이며 여기선 쓰지 않는다.

---

## 2. 로컬에서 서버로 옮길 것 — 3개뿐

```
① challenge/yolo_pose_one_model/datasets/stage_a/       51 GB   ★필수
② challenge/weights/pretrained_yolo/yolo26m-pose.pt     47 MB
③ challenge/yolo_pose_one_model/scripts/ + configs/     1 MB 미만
```

**원본 합성 데이터는 옮기지 않는다** (약 37GB 절약).
```
data/pallet/training_data/paper_release/       (G 원본 40k)  21 GB   불필요
challenge/data/02_synthetic/training/v1, v2    (T 원본 20k)  16 GB   불필요
```
`stage_a` 는 padding·라벨 변환이 끝난 완성본이고, 내부 alias symlink 도 **같은 폴더 안**을
가리키는 상대경로라 원본 없이 자립한다.

### 전송
```bash
rsync -avP challenge/yolo_pose_one_model/datasets/stage_a \
      계정@서버:/서버경로/challenge/yolo_pose_one_model/datasets/
rsync -avP challenge/weights/pretrained_yolo/yolo26m-pose.pt \
      계정@서버:/서버경로/challenge/weights/pretrained_yolo/
rsync -avP challenge/yolo_pose_one_model/scripts challenge/yolo_pose_one_model/configs \
      계정@서버:/서버경로/challenge/yolo_pose_one_model/
```
`-a` 를 반드시 쓴다(symlink 보존). 빼면 실파일로 복사돼 14GB 늘어난다.

---

## 3. 데이터셋 구조 — 이대로 두면 된다

```
datasets/stage_a/
├── images/
│   ├── train/   73,916   (실파일 55,959 + 상대 symlink 17,957)
│   └── val/      4,009
└── labels/
    ├── train/   73,916   (이미지와 같은 이름, 확장자만 .txt)
    └── val/      4,009
```

```
파일명 규칙
  G__f0000.png              generic synth (여러 팔레트 자산, 40k 중 38,002 train)
  T__v1__part_000__000000.png   target synth (과제 팔레트 palletobj, 17,957 train)
  T__..._rep1.png           위의 2배 alias — symlink. generic:target 을 1:1 로 맞추기 위함
구성 비율
  train  generic 38,002 : target 35,914(원본 17,957 x2)  = 1.06 : 1
  val    generic  1,998 : target  2,011                  (alias 없음, 각 1회)
```

### 이미지 — 해상도가 제각각인 게 정상이다
```
G__f0000.png              920 x 680
T__v1__part_000__000000.png   840 x 680
```
원본 해상도가 G 는 4종(640x480 / 960x540 / 720x480 / 560x560), T 는 640x480 이고
**전부 상하좌우 100px reflect padding** 을 적용했기 때문이다. YOLO 라벨은 이미지별로
정규화되므로 크기가 달라도 학습에 문제없다. **다시 padding 하지 말 것.**

### 라벨 — YOLO pose 32 토큰
```
0 cx cy w h  x0 y0 v0  x1 y1 v1 ... x8 y8 v8
예: 0 0.246763 0.380727 0.340307 0.184770 0.076609 0.390831 2 ...
```
- class 는 항상 0 (pallet)
- keypoint 9개, visibility 는 **2(보임) 또는 0(안 보임)만** 쓴다. 1 은 쓰지 않는다
- v=0 인 점의 좌표는 0,0 으로 기록돼 있다

---

## 4. 키포인트 계약 — 절대 바꾸지 말 것

```
0 near_top_left     1 near_top_right     2 near_bottom_right   3 near_bottom_left
4 far_top_left      5 far_top_right      6 far_bottom_right    7 far_bottom_left
8 centroid

{0,1,4,5} = 윗면,  {2,3,6,7} = 아랫면
0~3 = 카메라에 가까운 면(리프터가 진입하는 fork-pocket 면)
left/right 는 IMAGE 기준이지 물체 고정이 아니다
```
로컬에서 배포 PnP reprojection 으로 실증했다 (as-is 0.00~3.75px vs near/far 뒤집으면 8~31px).

```yaml
# configs/stage_a.yaml 에 이미 들어 있음
kpt_shape: [9, 3]
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]
```
★ `flip_idx` 는 **데이터 계약 기록용**이다. 이번 학습에서 `fliplr=0.0` 이므로 실제로는 안 쓴다.
   좌우 flip 을 켜지 말 것.

---

## 5. 서버 환경

```bash
conda create -n pallet-yolo26 python=3.10 -y
conda activate pallet-yolo26
pip install "ultralytics==8.4.60"
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 드라이버에 맞춰
python -c "import torch,ultralytics;print(torch.__version__,torch.cuda.is_available(),ultralytics.__version__)"
```
- **CUDA toolkit 설치 불필요** — torch wheel 이 런타임을 번들한다. 드라이버만 맞으면 된다(Ada ≥525)
- ultralytics 는 **8.4.60 이상**이어야 YOLO26 을 지원한다. 8.0.x 는 불가

---

## 6. 학습 전 두 가지

### (1) data yaml 의 path 갱신 — 필수
`configs/stage_a.yaml` 의 `path:` 는 로컬 절대경로가 박혀 있다. 서버 경로로 바꿔야 한다.
```bash
cd /서버경로/challenge/yolo_pose_one_model
python scripts/build_stage_dataset.py --stage stage_a --yaml-only
```
★ Ultralytics 는 `path` 가 상대경로면 `SETTINGS['datasets_dir']` 기준으로 풀어서 엉뚱한 곳을
   본다. 반드시 절대경로여야 한다(위 스크립트가 자동 계산한다).

### (2) GPU 확인
```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```
특정 GPU 만 써야 하면 `CUDA_VISIBLE_DEVICES` 로 **다른 카드를 아예 안 보이게** 막는다.
`device=1` 보다 안전하다.
```
free 용량 기준 권장 batch (로컬 3080 실측에서 외삽 [추정])
   ~9 GB → 8      ~12 GB → 16      ~18 GB → 32      ~24 GB → 32 (48은 경계)
```

---

## 7. 학습 실행

```bash
cd /서버경로     # repo 루트
CUDA_VISIBLE_DEVICES=1 nohup bash challenge/yolo_pose_one_model/scripts/train_stage_a_m.sh 32 \
  > challenge/yolo_pose_one_model/runs/m_b32.log 2>&1 &
```

스크립트가 쓰는 설정 (nano 와 동일, 모델과 batch 만 다름):
```
model=yolo26m-pose.pt   imgsz=640   batch=32   nbs=64
epochs=60   patience=15   optimizer=SGD
lr0=0.01  lrf=0.01  momentum=0.937  weight_decay=0.0005
warmup_epochs=3.0  warmup_momentum=0.8  warmup_bias_lr=0.1  cos_lr=True
mosaic=0.30  close_mosaic=10  translate=0.10  scale=0.25
hsv_h=0.015  hsv_s=0.50  hsv_v=0.35
fliplr=0  flipud=0  degrees=0  shear=0  perspective=0  mixup=0  copy_paste=0
amp=True  seed=42  deterministic=True  single_cls=True  multi_scale=False
save_period=5   name=stage_a_m_640_b32_seed42   exist_ok=False
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   (스크립트가 export)
```
**이 설정을 바꾸지 말 것.** nano 와 비교하려면 동일해야 한다.

---

## 8. ★★ 반드시 확인할 함정 3가지

### (1) Ultralytics 가 batch 를 몰래 낮춘다
OOM 이 나면 32 → 16 → 8 로 스스로 줄이고 계속 진행한다. 그러면 run 이름과 `args.yaml` 은
32 인데 **실제는 8** 이 된다. 로컬에서 실제로 당했다.
```bash
grep -c "Reducing to batch" runs/m_b32.log        # 0 이어야 정상
grep "Reducing to batch" runs/m_b32.log           # 나오면 실제 batch 확인
# step 수로도 검증 (73,916 / batch)
#   batch 32 → 2,310   batch 16 → 4,620   batch 8 → 9,240
```
축소가 일어났다면 **중단하고 로컬에 알릴 것.** 그 상태로 끝까지 돌리면 비교가 무의미해진다.

### (2) project 는 절대경로여야 한다
`project=` 가 상대경로면 Ultralytics 가 `SETTINGS['runs_dir']`("runs") 아래로 풀어서
`runs/pose/challenge/...` 같은 엉뚱한 곳에 저장한다. 스크립트는 이미 절대경로를 쓴다.

### (3) 프로세스 감시에 `pgrep -f` 를 쓰지 말 것
`pgrep -f "yolo pose train"` 은 **그 문자열을 포함한 모든 셸**(감시 스크립트 자신 포함)을
잡는다. 로컬에서 이것 때문에 자동화가 2시간 30분 멈췄고, `pkill -f` 로 자기 셸을 두 번 죽였다.
```bash
# 감지
ps -eo comm,args --no-headers | awk '$1 ~ /^python|^yolo/ && /bin\/yolo pose train/ {f=1} END {exit !f}'
# 정리
ps -eo pid,comm --no-headers | awk '$2=="yolo"{print $1}' > /tmp/pids.txt && xargs -r kill < /tmp/pids.txt
```
comm 은 15자로 잘려 이 경우 `yolo` 로 보인다. `^python` 만 보면 놓친다.

---

## 9. 정상 진행 판단 기준

로컬 nano(batch 32) 와 medium(batch 8) 의 실측이다. 서버 medium(batch 32) 은 이 사이 어딘가여야 한다.
```
                epoch 1        epoch 1 Pose mAP50-95     60 epoch 총 시간
nano   b32        580 s              0.5806                8.45 h
medium b8       1,698 s              0.7732               25 h (추정)
medium b32     서버에서 측정          0.75~0.80 예상 [추정]   4090 이면 6~9 h [추정]
```
```bash
tail -3 runs/stage_a_m_640_b32_seed42/results.csv | cut -d, -f1,2,11,16
# epoch, time(초), Box mAP50, Pose mAP50-95
```
- epoch 1 의 Pose mAP50-95 가 0.7 대면 정상
- NaN 이 나오면 즉시 중단하고 알릴 것
- patience 15 로 조기 종료될 수 있다(정상)

---

## 10. 끝나면 로컬로 돌려보낼 것

```
runs/stage_a_m_640_b32_seed42/weights/      best.pt · last.pt · epoch{0,5,...,55}.pt   약 1.5 GB
runs/stage_a_m_640_b32_seed42/results.csv
runs/stage_a_m_640_b32_seed42/args.yaml
runs/m_b32.log                              (batch 축소 여부 확인용)
```
```bash
rsync -avP runs/stage_a_m_640_b32_seed42 로컬계정@로컬:/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/runs/
```
평가·최종 선택은 로컬에서 한다(평가 스크립트와 nano 결과가 거기 있다).

---

## 11. 하지 말아야 할 것

```
- 데이터에 padding 을 다시 적용 (이미 100px 적용됨)
- convert_to_camera_facing_v4.py 를 이 데이터에 실행 (2D 휴리스틱이 저앙각에서 오판, 데이터가 깨진다)
- fliplr / flipud 켜기
- imgsz 를 640 이외로 변경 (배포 계약)
- batch 를 임의로 낮추고 계속 진행 (낮춰야 하면 로컬에 알리고 lr 재조정 협의)
- 기존 로컬 run(stage_a_synth_640_b32_seed42) 을 덮어쓰기
- real 데이터를 이 학습에 섞기 (이번 라운드는 합성 전용)
```

---

## 12. 참고 — 로컬에 있는 문서

```
challenge/yolo_pose_one_model/README.md                  전체 파이프라인
challenge/yolo_pose_one_model/PURPOSE.md                 목적·범위·봉인 규칙
challenge/yolo_pose_one_model/SERVER_SETUP.md            서버 이전 절차(이 문서의 축약판)
challenge/yolo_pose_one_model/contract/pallet_pose_contract.yaml   계약 단일 출처
challenge/yolo_pose_one_model/reports/01~05              데이터 인벤토리·계약충돌·split·감사·smoke
_docs/history/2026-08-15.md                              오늘 작업 전말(함정 포함)
```
