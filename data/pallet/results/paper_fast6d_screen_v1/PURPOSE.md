# PURPOSE — paper_fast6d_screen_v1

[소비처]
"어떤 pose formulation 이 short training 까지 갈 자격이 있는가" 를 정하는 근거.
self-training 계열이 전부 닫힌 뒤, **학습 없이** 기존 신호만으로 후보를 거른다.
결과는 사용자 판단으로 소비되며, GO 가 나와도 자동 학습은 없다.

[문장]
"monocular RGB → full 6D 를 유지한 채, YOLO bbox 나 구조 신호를 pose 계산에
붙이면 synthetic-only R0 보다 실제 6D pose(IoU3D · ADDsym AUC)가 좋아진다"
— 또는 그 반증.

## 이번 단계 범위

새 학습 0 · 새 checkpoint 0 · depth 0 · self-training 0 · parameter sweep 0.
Direct 3DoF 전환 금지.  기존 실험 artifact 는 전부 읽기 전용.

## 판단 지표 (결과 보기 전 고정 — FAST_6D_SCREEN_LOCK.json)

```
primary    IoU3D median ↑ · ADDsym AUC ↑
secondary  PoseCov · rotation · yaw · translation median
DOPE arm   pooled kp median · p90 · gross20
promotion  ΔIoU3D >= +0.020 OR ΔADDsym >= +0.020, 그리고 다른 지표 악화 없음
```

★ **목적함수가 줄었다는 이유만으로 PASS 금지.** 기각된 predicted-seed DiffPnP 가
predicted reprojection 을 11.96→6.35px 로 줄이면서 GT reprojection 과 rotation 은
악화시켰다. 판정은 GT 기준으로만 한다.

## S1 이 존재하는 이유

S2(bbox) 의 이득이 bbox 때문인지, 그냥 pose 를 한 번 더 최적화해서인지 갈라야
한다.  S1 은 bbox 를 쓰지 않고 translation 만 다시 맞추는 control 이다.

## 적용범위 한계 (측정 전 선언)

population 은 PAPER_EVAL positive 319 로, **이미 개발에 쓰인 셋**이다.
결과가 좋아도 held-out · independent · final 이라고 부르지 않는다.
이 screen 은 method selection 전용이다.

## 실패 시

GO arm 이 0 개면 `FAST_6D_SCREEN = NO_PROMOTABLE_SIGNAL` 로 끝내고 추가
architecture 학습을 열지 않는다.  실패한 arm 을 파라미터 탐색으로 구조하지 않는다.

NEXT_ACTION = USER_REVIEW_FAST_6D_SCREEN
