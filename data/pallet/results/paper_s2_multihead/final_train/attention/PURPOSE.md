# ATTENTION / BELIEF 시각화 — PURPOSE

**[소비처]** 방향결정 — 최종 모델(SPLIT_LATE 2-head, FINAL40K seed1/2 step_25000)의
line branch 가 무엇을 근거로 엣지를 찍는지 눈으로 확인해, `rho` 가 병목이라는
기존 판정([theta-only line = 회전 이득 확증, rho 가 범인])의 **기전**을 가른다.
부수적으로 논문 정성 그림(§ 정성 분석) 후보.

**[문장]** "line branch 의 12 role query attention 이 해당 엣지 주변을 보는지(국소
증거) 아니면 전역/한 곳으로 붕괴하는지" 가, theta 는 이미 좋은데 rho 만 나쁜
비대칭을 설명한다 — 한 곳으로 붕괴해 있으면 방향은 알아도 위치는 모르는 것이
구조적으로 당연하고, 국소인데도 rho 가 나쁘면 원인은 attention 이 아니다.

## 학습 아님

이 폴더는 **추론 + 시각화만** 한다. 가중치를 만들지 않는다.
읽는 체크포인트: `weights/paper_s2/paper_s2_multihead/screen_A1_CORNER_LINE_FINAL40K_seed{1,2}/step_25000.pth`

## 판정 지표 (그림을 그리기 전 스모크에서 확인)

```
집중도     entropy < ln(2500)=7.824 nats 보다 확실히 낮을 것
peak       max >> 균일 4.0e-4
role별상이  12 role 평균 상관 < 0.9  (전부 같은 곳 보면 role별 그림 무의미)
```
셋 중 하나라도 실패하면 "어디를 보는가" 그림은 의미가 없으므로 그리지 않고
그 사실 자체를 보고한다.

## 대상

REAL_DEV open set 56장 (`ft_f0f3_eval.OPEN_SETS` = eval_canonical 3폴더).
FINAL_TEST 4세션은 열지 않는다.
