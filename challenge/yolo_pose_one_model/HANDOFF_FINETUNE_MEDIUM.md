# 인수인계 — yolo26m-pose 를 real + negative 로 finetuning

이 문서만 보고 실행할 수 있게 썼다. nano(yolo26n)로 **먼저 다 해봤고**, 그때 밟은
함정과 판정 결과를 그대로 옮겼다. §5(함정)와 §7(해석)은 반드시 읽어라 — 안 읽으면
"합성 지표가 올랐으니 성공" 이라는 잘못된 결론에 도달한다.

---

## 0. 한 줄 요약

합성만으로 학습한 `stage_a_m`(medium) 위에, **real GT 157장 + 배포환경 negative 259장**
을 얹어 40 epoch finetuning 한다. 목적은 **정확도 향상이 아니라 false positive 제거**다.

---

## 1. 배경 — 왜 하는지

배포 환경(지게차 장착 카메라) 영상에서 **팔레트가 없는데 리프터 자기 포크와 울타리를
팔레트로 검출**했다. 원인은 명확하다:

- stage_a 학습셋 73,916장이 **전부 팔레트를 포함**한다. negative 가 0장이다.
- 즉 모델은 "팔레트가 없는 장면" 을 한 번도 본 적이 없다.
- 추론 코드도 `box conf 최대 instance 선택` 이라 항상 뭔가를 고른다.

그래서 팔레트가 없는 프레임을 **빈 라벨(background)** 로 넣어준다. 이게 이 학습의 전부다.

---

## 2. 옮길 것 — 이 zip 하나 (371MB)

```
ft_pkg/
├─ README_FINETUNE.md          이 문서
├─ ft_real_neg/
│  ├─ images/train/            416장 (real__ 157 + neg__ 259) — 이미 100px 패딩됨
│  ├─ labels/train/            416개 (neg__ 는 0바이트 = background)
│  ├─ _prepare_real_train.json 변환 기록
│  └─ _build_ft.json           조립 기록
├─ scripts/
│  ├─ build_ft_dataset.py      oversample + 합성 subsample 조립
│  ├─ train_ft_a.sh            학습 (BASE/DATA/EPOCHS/PATIENCE/RUN_NAME 환경변수)
│  ├─ prepare_real_ft.py       (참고용 — 재변환할 일 없으면 안 씀)
│  └─ eval_ft_fp.py            판정 (로컬에서 돌리는 게 낫다, §8)
└─ runs_ft/
   ├─ PURPOSE.md               목적·지표 (train_ft_a.sh 가 존재를 검사한다)
   └─ forklift_raw_conf.json   negative 프레임 목록
```

**합성 12,000장은 넣지 않았다.** 서버에 이미 `datasets/stage_a` 가 있고,
`build_ft_dataset.py` 가 거기서 심볼릭 링크로 가져간다. 중복 전송 불필요.

### 서버에 푸는 위치

```bash
cd <repo>/challenge/yolo_pose_one_model
unzip ft_pkg.zip -d /tmp/ft_pkg

mkdir -p datasets/ft_m/images/train datasets/ft_m/labels/train
cp /tmp/ft_pkg/ft_real_neg/images/train/* datasets/ft_m/images/train/
cp /tmp/ft_pkg/ft_real_neg/labels/train/* datasets/ft_m/labels/train/
cp /tmp/ft_pkg/scripts/* scripts/
mkdir -p runs_ft && cp /tmp/ft_pkg/runs_ft/* runs_ft/
chmod +x scripts/train_ft_a.sh
```

---

## 3. 전제 — medium base weight 가 있어야 한다

```bash
ls -la runs/stage_a_m_640_b32_seed42/weights/best.pt
```

**이게 없으면 이 작업을 시작하지 마라.** 합성 pretrain(stage_a_m)이 먼저 끝나야 한다.
finetuning 은 그 위에 얹는 것이다.

---

## 4. 실행

### (1) 데이터셋 조립

```bash
python scripts/build_ft_dataset.py --out datasets/ft_m \
  --real-repeat 20 --neg-repeat 6 --synth 12000 --synth-val 1000
```

반복은 유일한 이름의 심볼릭 링크로 만든다. 같은 경로를 두 번 나열하면 로더가 중복
제거해 버리기 때문이다.

기대 출력:
```
real_total 3140 / neg_total 1554 / synth_train 12000 / train_total 16694
negative 비중 9.3%
```

**왜 이 비율인가**: real 157장은 합성 73,916장의 0.2% 라 그냥 이어붙이면 매 epoch
거의 안 나온다. 반대로 real 만 쓰면 합성에서 배운 keypoint 구조를 잊는다
(catastrophic forgetting). negative 9.3% 는 Ultralytics 권장(0~10%) 범위다.

### (2) 학습

```bash
BASE=challenge/yolo_pose_one_model/runs/stage_a_m_640_b32_seed42/weights/best.pt \
DATA=challenge/yolo_pose_one_model/datasets/ft_m/data.yaml \
EPOCHS=40 PATIENCE=0 RUN_NAME=ft_m_real157_neg259_synth12k \
bash scripts/train_ft_a.sh
```

`PATIENCE=0` 은 **필수다** — 이유는 §5(1).

nano 기준 40 epoch 에 2.6시간이었다. medium 은 파라미터가 훨씬 많으니 더 걸린다
(4090 이면 nano 대비 대략 2~3배로 잡아라 [추정]).

---

## 5. ★★ 함정 — nano 에서 실제로 밟은 것들

### (1) EarlyStopping 이 개선 구간을 잘라먹는다 → `PATIENCE=0`

nano 1차 시도는 `patience=15` 로 돌렸더니 **16 epoch 에서 멈추고 `best.pt` 가 epoch 1**
이 됐다. Ultralytics fitness 가 epoch 1 을 최고로 잡아 카운터가 거기서부터 돈 것이다.
그런데 실제로는 계속 개선 중이었다:

```
ep16  val 0.3366/0.3532  mAP-P 0.9448   <- 여기서 끊겼다
ep39  val 0.2957/0.2990  mAP-P 0.9573   <- 40 완주했을 때
```

### (2) Ultralytics 가 OOM 시 batch 를 몰래 낮춘다

`args.yaml` 에는 32 라고 적혀 있는데 실제로는 16 이나 8 로 돈다. 반드시 확인:

```bash
grep -c "Reducing to batch" <로그>            # 0 이어야 한다
# step 수로도 검증: 16,694 / batch
#   batch 32 -> 522    batch 16 -> 1,044    batch 8 -> 2,088
```

### (3) `project` 는 절대경로여야 한다

상대경로면 Ultralytics 가 `SETTINGS['runs_dir']` 기준으로 풀어서 엉뚱한 곳에 저장한다.
`train_ft_a.sh` 는 이미 절대경로로 만든다.

### (4) 프로세스 감시에 `pgrep -f` / `pkill -f` 를 쓰지 마라

자기 자신(감시 명령)이 매칭돼 오탐하거나, 자기 셸을 죽인다. 실제로 두 번 겪었다.

```bash
# 감지
ps -eo pid,args | awk '/bin\/yolo pose train/ && !/awk/ {print $1}'
# 정리 — PID 를 모아서 kill
ps -eo pid,args | awk '/bin\/yolo pose train/ && !/awk/ {print $1}' | xargs -r kill
```

### (5) GPU 를 남이 쓰고 있는지 먼저 봐라

`train_ft_a.sh` 는 free VRAM 이 부족하면 중단한다. **batch 를 낮추지 말고** 원인을 찾아라.

```bash
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
```

---

## 6. 정상 진행 판단

```bash
# 데이터 스캔 로그 — negative 가 background 로 잡혔는지
#   기대: "16694 images, 1554 backgrounds, 0 corrupt"
grep -aoE "[0-9]+ images, [0-9]+ backgrounds, [0-9]+ corrupt" <로그> | tail -1

# 진행
python -c "
import csv,sys
rows=list(csv.DictReader(open('runs_ft/ft_m_real157_neg259_synth12k/results.csv')))
g=lambda r,k: float(r[[c for c in r if c.strip()==k][0]].strip())
print(len(rows),'epoch, 경과', round(g(rows[-1],'time')/60), '분')
for r in rows[-3:]:
    print(int(g(r,'epoch')), g(r,'val/box_loss'), g(r,'metrics/mAP50-95(P)'))"
```

`backgrounds` 가 1554 가 아니면 negative 라벨이 빈 파일이 아닌 것이다 — 멈추고 확인해라.

---

## 7. ★ 결과 해석 — 여기서 틀리기 쉽다

### 합성 val mAP 로 성공을 판단하지 마라

nano 에서 확인된 것: **합성 val mAP 를 0.9448 → 0.9573 으로 올려도 real 성능은
전혀 변하지 않았다.**

```
                      FP율@0.05  eval검출   kp med    forklift 검출
base (합성만)            50.6%     88.2%    9.30px      558/911
FT 16 epoch              0.0%     96.9%    6.95px      479/911
FT 40 epoch (완주)        0.0%     97.5%    7.38px      479/911   <- real 은 동일
```

40 epoch 완주분과 16 epoch 분을 같은 프레임끼리 짝지어 검정하면 **Wilcoxon p=0.83**
으로 구분되지 않는다. 즉 합성 목적함수를 더 내려도 real 로 전이되지 않는다.

### 진짜 판정 지표 3개

1. **FP율** — 팔레트 없는 259 프레임에서 검출이 나오는 비율. base 는 conf 0.05 에서
   50.6%, finetune 후 0.0%. **단 negative 는 학습에 들어갔으므로 in-sample 이다.**
2. **eval 정본 161장** — 학습에서 제외한 유일한 셋. 여기 개선이 진짜다
   (검출 88.2%→96.9%, kp 9.30→6.95px).
3. **영상 육안** — 사람이 봐야 한다.

### 검출 수가 줄어드는 건 정상이다

forklift 558 → 479 로 줄지만, base 가 잡던 것 중 사라진 81장을 전수로 보면 대부분
울타리·화단·방수포·건물이다. **base conf ≥ 0.9(확실한 팔레트) 440장은 100% 유지된다.**

### 알려진 부작용 — 잘린/원거리 팔레트가 죽는다

```
30s 프레임 (하단 심하게 잘린 팔레트)
  base 0.427  ->  FT16 0.028  ->  FT40 0.000
```

**학습을 더 돌려도 회복되지 않는다.** 학습량 문제가 아니라 negative 구성 문제로 본다.
medium 에서도 같은 증상이 나올 것으로 예상한다 [추정]. 나오면 다음을 시도해라:

1. negative 259장 재검수 — 멀리 작게 찍힌 팔레트가 섞였는지 확인해 제거
2. 잘린/원거리 팔레트를 positive 로 보강 (추가 어노)
3. 임시 운용: conf threshold 를 0.25 로 낮춤 (FP 가 0 이라 여유가 있다)

---

## 8. 평가는 로컬에서

`eval_ft_fp.py` 는 real GT 161장과 forklift_raw 911 프레임이 필요한데, 그건 서버에
없다. **학습된 weight 만 로컬로 가져오면 된다** (medium best.pt 약 40MB).

```bash
# 서버 -> 로컬
runs_ft/ft_m_real157_neg259_synth12k/weights/best.pt
runs_ft/ft_m_real157_neg259_synth12k/weights/last.pt
runs_ft/ft_m_real157_neg259_synth12k/results.csv
```

로컬에서:
```bash
python scripts/eval_ft_fp.py --conf 0.4 --weights <base.pt> <ft_m_best.pt>
```

---

## 9. 키포인트 계약 — 절대 바꾸지 마라

```
kpt_shape: [9, 3]
flip_idx : [0,1,2,3,4,5,6,7,8]      # 항등. fliplr=0.0 이므로 실제로 안 쓰이지만
                                     # 좌우 반전은 절대 켜지 마라 (0123 이 좌우 비대칭)
0~3 = 카메라를 향한 가까운 면, 4~7 = 반대편, 8 = centroid
{0,1,4,5} = 위,  {2,3,6,7} = 아래
padding = 100px BORDER_REFLECT_101, 라벨은 패딩된 크기로 정규화
```

`fliplr=0.0`, `flipud=0.0`, `degrees=0.0` 은 `train_ft_a.sh` 에 이미 박혀 있다.
바꾸면 keypoint 순서가 깨진다.
