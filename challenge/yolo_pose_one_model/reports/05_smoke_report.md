# 05 — Smoke report

```
명령   bash challenge/yolo_pose_one_model/scripts/run_smoke.sh
run    challenge/yolo_pose_one_model/runs/smoke_b32_seed42/
데이터 datasets/smoke   train 512 (G 256 + T 256) / val 128 (G 64 + T 64)
설정   yolo26n-pose.pt · imgsz 640 · batch 32 · nbs 64 · epochs 2 · SGD lr0 0.01 · mosaic 0.2
       fliplr 0 · seed 42 · deterministic · AMP · workers 4
```

## 통과 조건

```
조건                              결과      근거
──────────────────────────────────────────────────────────────────────────────
CUDA OOM 없음                     PASS      완주. peak GPU_mem 5.58 G / 9.62 G
GPU 에 다른 대형 process 없음      PASS      학습 전 free 9,038 MiB (게이트 7,168 MiB)
NaN/Inf loss 없음                 PASS      전 스텝 유한
label warning 없음                PASS      Ultralytics 경고 0
학습 loss 감소                    PASS      아래 표
validation 정상 종료              PASS      2/2 epoch 모두 완료
checkpoint 생성                   PASS      weights/best.pt, last.pt (각 6.5 MB)
예측 overlay keypoint 순서 정상    N/A       2 epoch 로는 판단 불가 — 아래 참조
SQPnP 최소 1 sample 성공          N/A       예측 기반은 위와 같은 이유로 불가 — 아래 참조
```

## loss 추이

```
                epoch1 -> epoch2
box_loss         3.249  -> 2.310
pose_loss       10.42   -> 9.904
kobj_loss        0.702  -> 0.674
cls_loss         6.563  -> 3.420
dfl_loss         0.047  -> 0.030
rle_loss        29.36   -> 18.64
```
전 항목 감소. cls 와 rle 가 크게 떨어지고 **pose_loss 는 거의 움직이지 않았다**
(10.42 → 9.90). 512장 2 epoch(32 step)이라 정상 범위지만, Stage A 에서
pose_loss 가 계속 정체하면 그때는 문제 신호다 — 감시 대상으로 남긴다.

## val 지표 (의미 없음, 기록용)

```
epoch2   Box  P 0.455  R 0.281  mAP50 0.256  mAP50-95 0.113
         Pose P 8.3e-05 R 0.0078 mAP50 6.1e-07
```
512장 2 epoch 산물이다. **이 숫자를 성능으로 인용하지 않는다.**

## keypoint 순서 검증을 무엇으로 대신했나

2 epoch 모델의 예측은 keypoint 가 한 점에 뭉쳐 있어 순서를 눈으로 판정할 수 없다
(`runs/smoke_b32_seed42/val_batch0_pred.jpg`). 대신 두 가지로 대체 검증했다.

1. **Ultralytics 가 읽은 GT** — `val_batch0_labels.jpg` 를 직접 열어 확인.
   bbox 와 9 keypoint 가 팔레트 위에 정상 배치. 로더가 우리 라벨을 제대로 파싱한다.
2. **우리 overlay** — `reports/overlays/smoke_train_{G,T}/` 각 100 장.
   near(초록 0-3) / far(파랑 4-7) / centroid(마젠타 8) 가 계약대로 그려진다.
   T 예시에서 0:nTL 좌상 → 1:nTR 우상 → 2:nBR 우하 → 3:nBL 좌하 순서를 육안 확인.
3. **PnP** — GT 라벨 기준 reprojection 은 이미 계약 감사에서 통과했다
   (G 0.00 px / T 3.49 px / near-far 뒤집으면 8~24 px). reports/02 §1.

## ★ 발견 — G 에는 라벨되지 않은 팔레트가 함께 찍혀 있다

`val_batch0_labels.jpg` 를 눈으로 보다 확인했다. G 프레임에는 라벨 객체가 항상 1개인데
(`objects` 길이 = 1, 표본 2000/2000), 같은 화면에 **다른 팔레트가 함께 보이는 프레임이
많다**. 예: `G__f0231`(좌하단 나무 팔레트 2개 비라벨), `G__f0748`(우측 다수),
`G__f10025`(좌측 녹색 팔레트 상자).

`records.jsonl` 이 이를 뒷받침한다.
```
n_context_visible = 3  인 프레임   21,778 / 40,000  (54%)
n_context_visible = 0              15,738           (39%)
context_screen_area_ratio 예       0.245 (idx 2)
```
컨텍스트 자산의 종류를 적은 필드가 없어 "그 3개가 전부 팔레트인지"는 메타데이터로는
확정할 수 없다 [추정]. 그러나 육안으로 팔레트인 사례가 확인된다 [확인].

**영향**: 라벨 없는 팔레트는 배경으로 학습되어 detection recall 을 떨어뜨릴 수 있다.
과제의 1순위 지표가 recall 이므로 무시할 수 없다.

**이번 라운드 처리**: 그대로 진행한다. G 를 고치려면 재렌더가 필요해 범위 밖이고,
G 는 어차피 geometry 사전학습용이며 과제 팔레트 성능은 T val 로 본다.
real finetune 후 recall 이 낮으면 **가장 먼저 의심할 원인**으로 기록해 둔다.

## 부수 수정

`project=` 를 상대경로로 주면 Ultralytics 가 `SETTINGS['runs_dir']`("runs") 아래로
풀어 `runs/pose/challenge/...` 에 저장된다. 절대경로로 바꿨고
(`RUNS="$(pwd)/$ROOT/runs"`), smoke 산출물은 의도한 위치로 옮겼다.

## 판정

**PASS** — full Stage A 로 진행 가능.
