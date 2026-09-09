# NEXT_RECOMMENDATION — MATCHED_CAPACITY (지시문 초안, 학습 미착수)

> `low_angle_diversity_v1` screen 이 `NO_SIGNAL` 로 끝나 §28 대로 생성한다.
> **이 문서는 계획일 뿐이고 이번 실행에서 medium 학습을 시작하지 않았다.**

## 왜 capacity 가 다음인가

지금까지 닫힌 레버 (전부 이 저장소의 실측):

```
loss           LC covariance 계열 5ep matched screen -> t +18% · depth +25% 악화
               (memory: lc-covariance-loss-screen-negative)
data 구성      저앙각 diverse swap 5ep matched screen -> median 미달, plastic 반대
               (memory: low-angle-diverse-swap-screen-negative)
PnP solver     평가측·학습측 양쪽 REJECT
self-training  강한 base + 자기예측 PL = no new signal
```

남아 있고 **matched training 으로 기각된 적이 없는** 것이 model capacity 다.
기존 관측에서 medium 이 corner · IoU3D · rotation · ADD 에서 nano 보다 나았던
사례가 있으나, 데이터·recipe·init 계열을 맞춘 비교가 아니었다
(memory: `arch-baseline-synthetic-cannot-select-architecture` — synthetic 으로는
architecture 를 못 고르고 real 에서 축마다 승자가 다르다).

## 설계 (결과 보기 전 고정할 것)

```
arm N   = yolo26n   (현재 R0 계열)
arm M   = yolo26m
same    data manifest (R0 train 55,980 / val 4,020)
        init family (같은 pretrained 계열, 각 크기의 공식 pose 가중치)
        recipe (R0 args.yaml), epochs, optimizer, LR, scheduler, batch*, imgsz,
        augmentation, val, stopping policy, evaluation
유일 차이 = backbone 크기
```

`batch` 는 medium 이 VRAM 에 안 맞으면 조정이 불가피하다. 그 경우
**optimizer update 수를 맞추고** batch 차이를 confound 로 명시한다
(이 저장소 GPU 는 RTX 3080 10 GB, nano 가 batch 32 에서 6.8 GB 를 쓴다 —
medium 은 batch 축소가 거의 확실하다).

## 평가

PAPER_EVAL 319, 정본 evaluator, MAIN 경로. last.pt (best.pt 금지).
primary = translation median · depth median.
반드시 함께 = p90 · lateral · rotation median/p90 · yaw · corner median/p90 ·
detection coverage, 그리고 ALL / plastic / wood / Low / Mid / High / Far /
Clean / Occlusion.

★**p90 을 primary 와 나란히 본다.** 이번 screen 에서 중앙값은 안 움직였는데
p90 이 -8~61% 움직였다. 중앙값만 보면 신호를 놓친다.

## 사전등록 gate `[추정][미검증]`

```
M vs N:  depth median >= 5% 개선  AND  translation median >= 5% 개선
         AND plastic 둘 다 개선
         AND p90 악화 없음
         AND detection coverage 손상 없음
```

## 하지 않을 것

새 synthetic render · self-training · DiffPnP · 새 loss 항 · C4/square 비율 ·
real GT fine-tuning. capacity 하나만 바꾼다.

## 착수 전 필수

`data/pallet/results/<track>/PURPOSE.md` 에 [소비처]·[문장] 두 줄,
그리고 medium 가중치가 `pallet-yolo26` env 에서 로드되는지 먼저 확인
(memory: `yolo26-weights-need-pallet-yolo26-env`).
