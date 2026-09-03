# Strong-teacher track — 착수 감사

V1~V5 가 전부 **selection** 을 건드려 실패했으므로, 이번엔 **teacher** 를 바꾼다.
그 전에 §1~§4 감사와, 21 시간을 쓰기 전에 공짜로 얻을 수 있는 증거를 먼저 모았다.

`data/pallet/results/paper_strong_teacher_v1/PRIOR_TRACK_IMMUTABILITY_LOCK.json`
— V1~V5 산출물 **150 항목** 봉인 (pseudo 라벨 집합해시 2 개 포함).

## §2 금지 teacher 확인

real keypoint / real pose supervision 이 들어간 것:

```text
challenge/yolo_pose_one_model/release/pallet-pose-yolo26n-ft/     금지
challenge/yolo_pose_one_model/release/pallet-pose-yolo26m-ft/     금지
challenge/yolo_pose_one_model/release/pallet-pose-yolo26n-livegt/ 금지
challenge/.../runs_ft/       (model=best.pt, data=ft_a)            금지
challenge/.../runs_live_gt/  (model=pallet_yolo26n_pose_ft.pt)     금지
challenge/weights/*_ft_manual/                                     금지
```

`challenge/weights/pretrained_yolo/yolo26m-pose.pt` 는 COCO pretrained 다 — teacher 가
아니라 **init 후보**다.

## §3 T0 확인

```text
challenge/yolo_pose_one_model/spatial_concat_scratch/runs/
  YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt

model  yolo26n-pose.pt (COCO pretrained)
data   datasets/g38_legacy_v1v2_p0_tex20k
epochs 60   batch 32   imgsz 640   seed 42   lr0 0.01   fliplr 0.0   mosaic 0.3
```

**real frame 0 확인**: train 55,980 장이 전부 synthetic 데이터셋으로의 symlink 다
(`stage_a`, `legacy_v1v2_p0_10k`, `legacy_v1v2_p0_tex10k`).  `raw_data` · `01_real` ·
`real_data` 경로가 **0 건**.

## §4 기존 source-only medium 감사 — 사용 불가

repo 전체에서 `yolo26m-pose` 로 학습한 run 은 셋뿐이고, 완료본은 하나다.

```text
run                              model            data                  결과
_aborted_m_b16_oom_20260815      yolo26m-pose.pt  configs/stage_a.yaml  중단(OOM)
_aborted_m_actual_b8_20260815    yolo26m-pose.pt  configs/stage_a.yaml  중단
stage_a_m_640_b8_seed42          yolo26m-pose.pt  configs/stage_a.yaml  12 / 60 epoch
```

**두 가지 이유로 §5 요건을 못 채운다.**

1. **데이터셋이 R0 와 다르다.**  `stage_a`(73,916 장) 대 R0 의
   `g38_legacy_v1v2_p0_tex20k`(55,980 장).  §5 는 T0 와 T1 의 train membership SHA 가
   같을 것을 요구한다.
2. **60 epoch 중 12 에서 멈췄다.**  checkpoint selection contract 를 만족하지 않는다.

따라서 재사용하지 않는다.  T1 을 새로 학습해야 한다.

## T1 학습 비용 — 추정이 아니라 실측

중단된 medium run 의 `results.csv` 에서:

```text
stage_a_m_640_b8_seed42   12 epoch 누적 19,685 s  ->  ~1,640 s/epoch
                          (73,916 장, batch 8, nbs 64)
```

R0 데이터셋은 55,980 장으로 0.757 배다.  같은 batch/nbs 라면

```text
~1,240 s/epoch  ->  60 epoch  ~=  21 시간
```

GPU 는 RTX 3080 10 GB 이고 medium 은 batch 16 에서 OOM 이 난 이력이 있다(위 aborted).
batch 8 · nbs 64 면 R0 의 batch 32 · nbs 64 와 **effective batch 가 같다**(§6).

## 21 시간을 쓰기 전에 — consensus 절반을 공짜로 확인

§13 의 합의 메커니즘은 T1 없이도 방향을 볼 수 있다.  이미 있는 T0 캐시에
original 과 horizontal flip 두 view 가 들어 있다.

**주의: 이건 §13 계약이 아니다.**  §13 은 valid teacher 3 개 이상을 요구하고, 2 view
에서는 median 이 곧 평균이다.  탐색용이며 gate 판정에 쓰지 않는다.

```text
set                                  n_kp   NME med   NME p90   px med   px p90  gross20   cat40
R0 baseline / ALL                    1979    0.0195    0.0857     6.45    29.93    0.152   0.079
consensus (accepted only) / ALL      1284    0.0182    0.0553     5.71    22.25    0.115   0.049
consensus (same keypoints) / ALL     1807    0.0189    0.0993     6.36    38.41    0.160   0.099
```

**세 번째 줄이 정직한 비교다.**  첫 두 줄은 모집단이 다르다 — 합의가 받아들인 것만 세면
어려운 keypoint 1,979 -> 1,284 로 35% 가 빠진다(ambiguity 448, disagreement 84).

같은 keypoint 집합에서 **좌표만** 바꾸면:

```text
NME median   0.0195 -> 0.0189   미미한 개선
NME p90      0.0857 -> 0.0993   악화
gross20      0.152  -> 0.160    악화
cat40        0.079  -> 0.099    악화
```

야간도 같다 — gross20 0.106 -> 0.141, cat40 0.046 -> 0.099.

즉 **자기 자신의 flip 과 평균 내면 중앙값은 조금 좋아지고 꼬리는 나빠진다.**  flip view 는
어려운 곳에서 어긋나고, 두 점의 중점은 좋은 예측을 나쁜 쪽으로 끌어당긴다.

그리고 "accepted only" 의 개선은 **coverage selection** 이다 — V1~V5 가 다섯 번
보여준, 학생으로 전이되지 않는 바로 그 종류다.

## 지금 상태

```text
supporting evidence      2-view consensus 는 같은 keypoint 에서 좌표를 개선하지 못한다
                         (꼬리는 오히려 악화)
still untested           medium capacity 가 nano 보다 localisation 이 나은가 (T1 gate)
                         4-view median 이 2-view 평균과 다르게 작동하는가
cost to test             T1 학습 ~21 시간 (실측 기반)
```

T1 gate(§10)는 consensus 와 **독립된 질문**이다 — medium 이 nano 보다 잘 찍는지는
그 자체로 검증할 가치가 있고, PASS 하면 4-view median 도 2-view 평균과 다르게 나올 수
있다.  다만 21 시간은 사용자 결정이 필요한 규모라 여기서 멈춘다.
