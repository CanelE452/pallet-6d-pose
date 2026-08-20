# YOLO26 pallet-pose — 아키텍처와 학습 명세

이 문서의 모든 수치는 **실행 기록에서 읽은 것**이다. README 요약이 아니라
`runs*/args.yaml`(Ultralytics 가 실제 실행값으로 남긴 파일), `train_*.sh`,
체크포인트 내부 구조가 1차 출처다.

---

## 1. 아키텍처

```
                         yolo26n            yolo26m
parameters               3,043,704          23,543,044
scale                    n                  m
task                     pose               pose
nc (클래스)               1  (pallet)        1
kpt_shape                [9, 3]             [9, 3]      x, y, visibility
head                     Pose26             Pose26
strides                  8 / 16 / 32        8 / 16 / 32
입력                      640x640            640x640
```

단일 스테이지다 — 한 장 들어가서 **bbox 1개 + 키포인트 9개**가 나온다.
pose 는 모델이 내지 않는다. 키포인트를 PnP 에 넣어 6-DoF 를 복원한다.

### 키포인트 규약 — camera-facing (object-fixed 아님)

```
0 near_top_left      1 near_top_right      <- 카메라를 향한 면 (포크 포켓)
2 near_bottom_right  3 near_bottom_left
4 far_top_left       5 far_top_right       <- 반대 면
6 far_bottom_right   7 far_bottom_left
8 centroid

top = {0,1,4,5}   bottom = {2,3,6,7}
```

인덱스 0-3 은 **항상 카메라 쪽 면**이지 물체의 특정 면이 아니다.
→ 좌우 비대칭이므로 **`fliplr` 는 반드시 0.0**. 켜면 순서가 조용히 망가진다.
불가피하게 켜야 하면 `flip_idx: [1,0,3,2,5,4,7,6,8]` (항등 순열은 틀림).

---

## 2. 학습 2단계

```
STAGE A   합성 전용 pretrain   yolo26{n,m}-pose (COCO) -> stage_a
FT        real + negative     stage_a -> ft
```

### STAGE A 데이터 (`datasets/stage_a`)

```
train  73,916   generic 38,002 + target 35,914 (그중 alias 17,957 = x2 반복)
val     4,009   generic  1,998 + target  2,011
```
`generic` = 일반 팔레트 합성, `target` = 사용자 팔레트(palletobj) 합성.
target 을 x2 반복해 대략 1:1 로 맞췄다.

**★ 이 73,916장은 전부 팔레트를 포함한다. negative 가 0장이다.** 이게 FT 를 하게 된
직접적인 이유다 — 모델이 "팔레트 없는 장면" 을 한 번도 못 봤다.

### FT 데이터 (`datasets/ft_a` / 서버의 `ft_m`)

```
real       157장 x20 반복 = 3,140     실촬영 GT
negative   259장 x6  반복 = 1,554     배포영상에서 팔레트 없는 프레임 (빈 라벨)
synthetic         subsample = 12,000  stage_a 에서 심볼릭 링크
                    train_total 16,694    negative 비중 9.3%
val                             1,000
seed 42
```

반복은 **유일 이름의 심볼릭 링크**로 만든다 — 같은 경로를 두 번 적으면 로더가
중복 제거해 버린다.

왜 이 비율인가: real 157장은 합성 73,916 의 0.2% 라 그냥 이어붙이면 매 epoch 거의
안 뽑힌다. 반대로 real 만 쓰면 합성에서 배운 키포인트 구조를 잊는다. negative 9.3%
는 Ultralytics 권장(0~10%) 안이다.

**누수 규칙** — eval 정본 161장은 학습에서 제외. real 157장은
night01~07 · pallet02/03/04/05/08 · forklift_20260528 이고, 평가셋의
night08/09 · pallet07/09 는 들어가지 않았다. 같은 세션의 인접 non-eval 53장도 제외
(프레임 누수 방지). 그래서 real 이 400 → 157 로 줄었다 — **실수가 아니라 의도**다.

---

## 3. 하이퍼파라미터 (args.yaml 실측)

```
                    stage_a_n    stage_a_m         FT (n·m 공통)
model             yolo26n-pose  yolo26m-pose      stage_a best.pt
epochs                      60            60                   40
patience                    15            15                    0   ★
batch                       32             8   ★                32
nbs (nominal)               64            64                   64
imgsz                      640           640                  640
optimizer                  SGD           SGD                  SGD
lr0                       0.01          0.01                0.002   ★
lrf                       0.01          0.01                 0.01
momentum                 0.937         0.937                0.937
weight_decay            0.0005        0.0005               0.0005
warmup_epochs              3.0           3.0                  1.0   ★
warmup_momentum            0.8           0.8                  0.8
cos_lr                    True          True                 True
amp                       True          True                 True
seed                        42            42                   42
deterministic               -             -                 True
single_cls                  -             -                 True
```

### 손실 가중치 (세 run 동일)

```
box 7.5    cls 0.5    dfl 1.5    pose 12.0    kobj 1.0
```

### 증강 (세 run 동일, mosaic 만 다름)

```
hsv_h 0.015   hsv_s 0.50   hsv_v 0.35
translate 0.10   scale 0.25
degrees 0   shear 0   perspective 0   flipud 0   fliplr 0.0  ★
mosaic  0.30 (stage_a)  ->  0.15 (FT)   ★
close_mosaic 10   mixup 0   copy_paste 0   cutmix 0
erasing 0.4   auto_augment randaugment
```

★ 표시한 FT 변경점의 이유:
```
lr0 0.01 -> 0.002     그대로 두면 합성에서 배운 걸 흔든다
epochs 60 -> 40, warmup 3 -> 1
mosaic 0.30 -> 0.15   real 157장이 mosaic 으로 과하게 합성되면 negative 신호가 흐려진다
patience 15 -> 0      아래 §5 참조 — 필수다
나머지는 stage_a 와 동일하게 둔다. 비교 가능해야 하므로.
```

---

## 4. 실제 실행 결과

```
run                     epochs 완주   mAP50-95(B)   mAP50-95(P)   소요
stage_a_n                  60/60          0.9608        0.9595    8.4 h
stage_a_m (로컬)           12/60          0.9249        0.9338    5.5 h   ← 미완주
ft_b (= 배포 n_ft)         40/40          0.9474        0.9571    2.6 h
ft_a (patience 15)         16/40          0.9398        0.9448    0.55 h  ← 조기종료
```

**★ 로컬 `stage_a_m` 은 batch 8 · 12 epoch 에서 끝났다.** batch 32 로 돌리려다
OOM 이 나서(`_aborted_m_b16_oom_20260815`, `_aborted_m_actual_b8_20260815`) batch 를
8 로 낮춘 판이다. 배포된 `yolo26m-ft` 는 이 판이 아니다 — §6.

---

## 5. ★ 밟은 함정

### (1) EarlyStopping 이 개선 구간을 잘라먹는다 → `PATIENCE=0`

`patience=15` 로 돌린 `ft_a` 는 16 epoch 에서 멈췄고 `best.pt` 가 **epoch 1** 이 됐다.
Ultralytics fitness 가 epoch 1 을 최고로 잡아 카운터가 거기서부터 돈 것이다.
실제로는 계속 개선 중이었다:

```
ep16   val 0.3366/0.3532   mAP-P 0.9448     <- 여기서 끊김
ep39   val 0.2957/0.2990   mAP-P 0.9573     <- 40 완주 시
```

그래서 배포본은 `patience=0` 으로 다시 돌린 `ft_b` 다.

### (2) Ultralytics 가 OOM 시 batch 를 몰래 낮춘다

`train_stage_a.sh` 주석: "batch is fixed at 32. If it OOMs, diagnose the GPU;
do NOT lower the batch." 두 스크립트 모두 **free VRAM < 7168 MiB 면 학습을 시작하지
않고 중단**한다 — batch 를 낮춰 조용히 다른 실험이 되는 걸 막기 위해서다.

### (3) 추론 시 100px reflect padding 이 필요하다

학습이 패딩된 프레임으로 이뤄져서, 추론에서 빼먹으면 **검출률이 아니라 키포인트
정확도**가 떨어진다.

```
                detection    kp err median    kp err p90
padded (정답)   157/161          7.38 px        26.75 px
unpadded        157/161          8.49 px        28.42 px
```

배포 계약:
```python
padded = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
r = model.predict(padded, imgsz=640, conf=0.4, verbose=False)[0]
i = int(np.argmax(r.boxes.conf.cpu().numpy()))     # 최고신뢰 instance
kps = r.keypoints.xy.cpu().numpy()[i] - 100        # (9,2) 원본 좌표
```
PnP 는 `SOLVEPNP_SQPNP` + `refineLM`, 키포인트 conf >= 0.5 만 사용.
index 8(centroid)도 실제 대응이라 넣어도 된다.

---

## 6. ★ yolo26m-ft 는 이 머신에서 학습되지 않았다

```
release/pallet-pose-yolo26m-ft/pallet_yolo26m_pose_ft.pt   47.6 MB   08-16 13:10
~/Downloads/best.pt                                        47.6 MB   08-16 13:06
```

4분 차이다. medium 은 **서버에서** 학습해 내려받았다
(`yolo26m_b32_server_code.zip`, `yolo26m_finetune_pkg.zip`). 서버 레시피는
`README_FINETUNE.md` 에 있고 nano 와 동일하며, base 만
`runs/stage_a_m_640_b32_seed42/weights/best.pt` (batch 32 완주판) 로 바꾼 것이다.

**그 b32 stage_a_m 학습 로그는 이 머신에 없다.** 로컬 `stage_a_m_640_b8_seed42`
(12 epoch) 와 혼동하면 안 된다. medium 쪽 pretrain 의 정확한 epoch·시간은
`UNKNOWN` 으로 남긴다 — 추정으로 채우지 않는다.

---

## 7. 해석 주의

FT 의 주효과는 **정확도 향상이 아니라 false positive 제거**다. real 157장은 합성
73,916 의 0.2% 라 도메인 적응은 부수적이다. 합성 지표(mAP)가 올랐다고 성공이라
읽으면 안 된다 — 판정은 배포영상 FP 와 real 161 정본에서 한다.

알려진 한계(release README):
- 한 프레임에 팔레트 1개 가정 (`single_cls=True`)
- real 학습 데이터가 저앙각에 쏠려 있어 높은 시점은 미검증
- 플라스틱 팔레트 위주 — 목재 팔레트는 별개
- 하단 모서리 검출이 pretrain 0.427 → finetune 0.000 으로 떨어졌다.
  더 오래 학습해도 안 고쳐진다. 배경 이미지 구성 문제지 학습 부족이 아니다.
