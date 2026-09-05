# GATE C — 국소 semantic corner 전문가

R0 좌표는 **검색 중심만** 제공하고 최종 위치는 국소 RGB 가 정한다.
학습 crop 의 중심을 정확한 GT 에 두지 않고, SOURCE_DEV 에서 실제로 측정한
R0 coarse residual 벡터(3,175개, median 1.88px / p99 33.1px)를 복원추출해 흔들었다.

```
구조        작은 U-Net, 855,815 파라미터 (teacher-time only)
입력        64x64 RGB crop + keypoint semantic ID 0~7 embedding
출력        국소 잔차 heatmap · visibility · uncertainty · 투영 cuboid 변 방향 2개
예산        5,000 optimizer update (양 arm 모두 완주) · 60분 캡 미도달
checkpoint  last only
```

## 판정

```
REAL_LOCAL_SPECIALIST = STOP
```

사전등록 조건 6개 중 4개는 통과했으나 두 개에서 막혔다.

```
조건                                        결과
────────────────────────────────────────────────────
C1 의 p90 >=10% 개선 OR gross20 >=15% 감소   실패 (0.69% / 1.20%)
median 악화 <= 5%                            통과 (오히려 11.06% 개선)
R0-good harm rate <= 10%                     통과 (1.45%)
PoseCov 하락 <= 1%p                          통과 (0)
IoU3D 또는 ADDsym AUC 악화 없음               통과 (ADDsym +0.0066)
C1 이 C0 보다 좋을 것                         실패 (p90 43.59 vs 42.17)
```

## 수치 (visible 코너 1,594)

```
arm                      median     p90   gross20   IoU3D   ADDsym AUC
─────────────────────────────────────────────────────────────────────────
R0                         6.36   43.89     0.157   0.603        0.4285
C0  SYN_LOCAL              5.83   42.17     0.154   0.603        0.4373
C1  SYN_PLUS_REAL_SOFT     5.66   43.59     0.156   0.601        0.4351

R0 대비                  median     p90   gross20
─────────────────────────────────────────────────
C0                       -8.32%  -3.92%    -1.99%
C1                      -11.06%  -0.69%    -1.20%
```

## 결과가 말하는 것 — 국소 증거는 **정밀도**를 주고 **구조**를 못 준다

중앙값은 실제로 좋아진다. C1 이 6.36 → 5.66 px 로 11.1% 개선했고, 이미 맞힌 코너를
망가뜨리는 비율은 1.45% 로 낮다. 학습된 국소 모델은 고전 CV 선택기가 못 한 일을 한다
(Gate B 의 prediction-only 선택기는 median 을 11.2% **악화**시켰다).

그런데 꼬리는 그대로다.

```
R0 가 >20px 틀린 코너에서 전문가가 <=10px 로 구제한 비율   C0 0.0020   C1 0.0020
```

498개 중 한 개다. 사실상 0 이다.
반경 12~32px 안의 외형은 "이미 거의 맞은 점을 몇 px 당기는" 정보는 갖고 있지만
"여기가 아니라 저기다" 를 말해 주지 않는다. Gate B 의 밀도 통제가 예고한 그대로다.

불확실성 gate 는 거의 모든 코너를 통과시켰다 (C0 refined 2,550 / kept 2).
즉 전문가는 자기가 틀린 자리를 스스로 알아채지 못했다.

## real soft supervision 은 아무것도 더하지 못했다

C1 은 C0 에 high-consensus real target 과 photometric consistency 를 더한 arm 이다.
median 은 조금 더 내려갔지만(5.83 → 5.66) 꼬리는 오히려 나빠졌다(42.17 → 43.59).

원인은 표본이다. 미라벨 1,000장에서 합의 임계를 통과한 keypoint 가 **116개**뿐이었고,
그것도 Gate D 에서 보듯 R0 가 이미 맞히는 자리들이다. 116개를 5,000 step 동안
반복 노출한 것이 real 감독의 전부다. 이 arm 은 "real soft supervision 이 무력하다" 가
아니라 **"이 파이프라인이 real supervision 을 충분히 만들어내지 못한다"** 를 보여준다.

## 실행 이력 (숨기지 않는다)

첫 실행은 77분 만에 산출물 0 으로 중단했다. 원인은 두 겹이었다.

```
1  step 마다 840x680 PNG 6장을 콜드로 디코딩 -> 3.1 초/step (5,000 step = 4시간+)
2  torch 를 import 한 부모를 fork 한 프로세스 풀 / OpenMP 런타임에서 100% CPU 스핀
   (GPU 0% · IO 0 · wchan 0 · libgomp 매핑이 유일한 단서)
```

고친 것은 실행 방식뿐이다 — crop 사전추출(79,080개, 18초), 프로세스 풀 spawn 전환,
`OMP_NUM_THREADS=1` · `torch.set_num_threads(1)`, 무버퍼 로그, 하트비트, 주기 체크포인트,
step 내부에서도 발동하는 캡. 결과적으로 3.1 초/step 이 0.02 초/step 이 됐다(50 step/s).
**구조 · 5,000 update · jitter 분포 · arm 구성 · 임계는 한 글자도 바꾸지 않았다.**
