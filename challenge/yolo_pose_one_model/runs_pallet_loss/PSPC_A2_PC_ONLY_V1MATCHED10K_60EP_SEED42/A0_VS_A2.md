# A0 vs A2 (PC_ONLY) — 결론: **귀인 불가**, PC_SIGNAL = False

## 사전등록 게이트

```
PC_SIGNAL = corner p90 relative >=10% 개선  OR  projective p90 relative >=20% 개선
            AND  pose mAP50-95 regression <= 2pp
결과       = False
```
게이트는 결과를 보고 바꾸지 않았다.

## 관측

```
              A0 standard   A2 PC_ONLY
pose mAP50      0.8502        0.2869
pose mAP50-95   0.7290        0.0838
box  mAP50      0.9826        0.9823
identity-best   0.917         0.453
yaw180-best     0.062         0.479
corner p90      0.2017        0.7071      +250.6%
projective p90  0.0422        0.2856      +577.1%
```

학습 곡선 (train pose loss):
```
ep      1      2      5     10     20     30     40     50     60
A0   9.712  8.673  7.566  6.851  6.329  3.888  2.707  1.902  1.080
A2   9.726  8.627  7.503  6.813  6.348  5.958  5.607  5.386  5.487
```

## ★ 무엇이 실제로 일어났나

**ep 20 까지 A0 와 A2 는 사실상 같다** (train pose 6.329 vs 6.348). PC 는 설계대로
작은 섭동이다. 갈라지는 지점은 **ep 20 → 30** 이고, 거기서 A0 만 급락한다
(6.329 → 3.888). 그 시점이 앞서 확인한 **yaw180 트랩 탈출** 시점이다
(V1_10K A0 의 yaw180: ep20 0.376 → ep30 0.094).

A2 는 끝까지 탈출하지 못했다 (yaw180 ep30 0.443 → ep60 0.479).

**box mAP50 은 두 arm 이 동일하다** (0.9826 vs 0.9823). 검출은 멀쩡하고
keypoint **배정**만 갇혀 있다. NaN 0, gradient 폭주 없음.

## 왜 "PC loss 가 해롭다" 고 결론내지 않는가

이 트랩의 탈출은 **run 마다 크게 다르다** — 이미 실측된 것:
```
seed42 full40K   ep60 까지 탈출 못 함 (yaw180 0.159)
seed43 full40K   ep10 에 이미 탈출     (yaw180 0.055)
V1_10K seed42    ep20~30 에 탈출       (yaw180 0.376 -> 0.094)
```
탈출은 knife-edge 사건이고, loss 에 어떤 섭동을 줘도 궤적이 다시 굴려진다.
**arm 당 n=1 로는 "PC 가 미탈출을 유발했다" 와 "궤적 재추첨" 을 가를 수 없다.**

트랩을 통제하려고 둘 다 identity-best 인 64 프레임만 비교했지만, A2 는 전역적으로
무너져 있어(pose mAP50-95 0.084) 그 부분집합에서도 통제가 성립하지 않았다:
```
[n=64]  corner p90  A0 0.0891 -> A2 0.3369  (+278%)
        proj   p90  A0 0.0355 -> A2 0.2984  (+742%)
        proj   med  A0 0.0129 -> A2 0.1804  (14배)
```

## 설계상 약점 (다음 개정에 반영할 것)

`dist(GT_centroid, line(pred_i, pred_j))` 는 **직선이 centroid 를 지나기만 하면 0** 이다.
두 endpoint 를 그 직선을 따라 미끄러뜨리거나 늘려도 loss 가 변하지 않는다.
즉 corner 위치를 유일하게 구속하지 않는 **퇴화 방향**이 존재한다.
GT centroid 를 쓴 덕에 centroid shortcut 은 막았지만(T7 실증), 이 퇴화는 남아 있다.

보완 후보 (이번 범위 밖):
```
- 대각선 위에서의 centroid 의 내분비(GT 는 0.5)를 함께 구속
- 네 대각선의 교점이 한 점이 되도록 하는 항
```

## 판정

```
PC_SIGNAL         = False        (사전등록 게이트)
귀인              = 불가          (n=1, knife-edge 트랩)
METHOD_SUPPORTED  = Pending real  (reviewed GT 미도착이므로 real 평가 안 함)
```

**"PC loss 는 해롭다" 는 이 자료로 말할 수 없다.** 말할 수 있는 것은
"이 조건의 단일 run 에서 PC arm 이 트랩을 벗어나지 못했고, 그 이후 모든 지표가
그 사실에 지배됐다" 이다.

귀인하려면 arm 당 seed 를 늘리거나(최소 3), 트랩 자체를 제거해야 한다.
브리프가 "A1/A3 symmetry gate 가 열리기 전 추가 sweep 금지" 라고 했으므로
여기서 멈춘다.

## 참고 — checkpoint 이식성

A2 의 `last.pt` 는 `PSPCPoseModel` 클래스를 pickle 하고 있어, 로드하려면
프로젝트 루트가 import path 에 있어야 한다 (`import pallet_yolo_loss`).
