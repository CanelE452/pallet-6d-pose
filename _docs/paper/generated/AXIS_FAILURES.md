# 파국적 corner 오차는 위치 실패가 아니라 축 배정 실패다

진단 전용이다.  threshold·pool·model 을 바꾸지 않는다.

`AXIS_PERMUTED` = identity 의 **최대** 코너 오차가 25 px 를 넘는데, 수직축 회전
또는 거울 재배정 중 하나에서 최대 오차가 25 px 밑이면서 identity 의 절반 밑으로
떨어지는 프레임.  코너가 제자리에 있고 라벨만 돌아갔다는 뜻이다.

판정에 중앙값을 쓰지 않는다.  90 도 순열은 8 코너 중 절반만 맞히므로 오차가
이봉분포가 되고 median 이 작은 쪽 봉우리에 앉는다 — 초판에서 이걸 "좌우 뒤집힘"
으로 잘못 읽었다.

## 프레임 판정

```text
model          domain                      OK      AXIS_PERMUTED  OTHER_PERMUTATION         MISLOCATED
────────────────────────────────────────────────────────────────────────────────────────────────
R0             daytime                     33                  1                  0                 36
R0             nighttime                   30                  1                  0                 19
R0             none                       158                 13                  0                 28
R2_CONF        daytime                     27                  4                  0                 39
R2_CONF        nighttime                   26                  2                  0                 22
R2_CONF        none                       150                 11                  0                 38
R5_PROPOSED    daytime                     24                  4                  0                 42
R5_PROPOSED    nighttime                   24                  2                  0                 24
R5_PROPOSED    none                       151                 10                  0                 38
```

## 전체 (도메인 합)

```text
model           frames                 OK      AXIS_PERMUTED  OTHER_PERMUTATION         MISLOCATED
────────────────────────────────────────────────────────────────────────────────────────────
R0                 319                221                 15                  0                 83
R2_CONF            319                203                 17                  0                 99
R5_PROPOSED        319                199                 16                  0                104
```

## 어떤 순열이었나

```text
model                   yaw90         yaw180         yaw270         mirror             기타
──────────────────────────────────────────────────────────────────────────────────────────
R0                          8              0              7              0              0
R2_CONF                    11              0              6              0              0
R5_PROPOSED                10              0              6              0              0
```

## R0 는 맞았는데 R5 가 어긋난 프레임

```text
R2_CONF        새로 어긋남   3   고쳐짐   2
    eval_night09:1779449631842893312           identity max   153.6 ->         yaw90 max   15.2  centroidΔ   4.6
    eval_pallet07:1778652138515809024          identity max   286.7 ->         yaw90 max   24.1  centroidΔ   2.7
    eval_pallet09:1778653664407620608          identity max   206.5 ->         yaw90 max    7.5  centroidΔ   3.1
R5_PROPOSED    새로 어긋남   3   고쳐짐   2
    eval_night09:1779449631842893312           identity max   156.0 ->         yaw90 max   17.7  centroidΔ   5.5
    eval_pallet07:1778652138515809024          identity max   287.7 ->         yaw90 max   22.0  centroidΔ   2.5
    eval_pallet09:1778653664407620608          identity max   207.8 ->         yaw90 max   13.0  centroidΔ   4.0
```

