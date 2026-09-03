# FAST teacher probe — 학습 없이 R0 를 이길 수 있는가

`FAST_NO_TRAIN_TEACHER_WORKS = NO`

teacher 단계에서 세 후보가 전부 실패했으므로 **student 를 시작하지 않았다**.
새 학습 0 회.  YOLO26m 60ep(~21시간)는 보류 상태 그대로다.

## 판정 요약

```text
Teacher            ALL NME   Night NME   NME p90   gross20   cat40   축 순열
R0 (single view)    0.0190      0.0203    0.0795     0.144   0.084    0.048
FAST-A flip 평균     0.0189      0.0224    0.0993     0.160   0.099    0.044
FAST-B 640+960       0.0190      0.0204    0.0996     0.146   0.100    0.049
FAST-C R0+G38        0.0183      0.0208    0.1054     0.159   0.099    0.068
```

R0 열은 candidate 마다 **모집단이 달라 값이 조금씩 다르다** — 각 후보가 값을 낸
keypoint 에서만 R0 를 같이 재기 때문이다(아래 §비교 규칙).

**셋 다 같은 자리에서 실패한다: 중앙값은 미미하게 좋아지고 꼬리는 뚜렷하게 나빠진다.**

## 비교 규칙 — 이게 결론을 가른다

candidate 가 어려운 keypoint 를 버리고 쉬운 것만 남기면 지표가 저절로 좋아진다.
V1~V5 를 다섯 번 속인 것이 그 selection 효과다.

그래서 gate 는 **candidate 가 값을 낸 keypoint 에서 R0 도 같이 재는 paired 비교**만
쓴다.  coverage 는 따로 적는다.

```text
후보      coverage        비교에 쓴 keypoint
FAST-A    230/319 frames  1,801
FAST-B    221/319         1,671
FAST-C    235/319         1,801
```

이 대조를 안 쓰면 세 후보 모두 "R0 보다 낫다" 로 잘못 보고된다.

## FAST-A — flip 두 view 의 단순 평균

새 추론 0 회 (기존 캐시 재사용).

flip cache 계약을 먼저 감사했다 — `|orig - flip|` 중앙값 3.80 px 로 **이미 unflip +
index 복원 상태**다.  한 번 더 mirror 하면 95 px 가 된다.  재mirror 하지 않았다.
(V2 에서 이 실수로 1.9 px 가 127 px 이 된 이력이 있다.)

```text
set              n_kp   NME med   NME p90   px p90   gross20   cat40
ALL / R0         1807    0.0193    0.0784    29.48     0.142   0.079
ALL / FAST-A     1807    0.0189    0.0993    38.41     0.160   0.099
night / R0        192    0.0239    0.0836    21.44     0.109   0.052
night / FAST-A    192    0.0224    0.1097    33.95     0.141   0.099

PASS A1 ALL NME   0.0193 -> 0.0189
PASS A2 Night NME 0.0239 -> 0.0224
FAIL A3 p90       0.0784 -> 0.0993
FAIL A4 gross20   0.142  -> 0.160
```

두 점의 평균은 좋은 예측을 나쁜 쪽으로 끌어당긴다.  두 view 가 어긋나는 곳이 곧
어려운 곳이라, 중점은 어느 쪽도 아니게 된다.

## FAST-B — 640 + 960 네 view 의 좌표별 median

새 추론 638 회 (imgsz 960, original + flip).  OOM 없음.  960 한 값만 봤고 sweep 하지
않았다.

```text
set              n_kp   NME med   NME p90   px p90   gross20   cat40
ALL / R0         1671    0.0191    0.0820    29.31     0.138   0.081
ALL / FAST-B     1671    0.0190    0.0996    38.96     0.146   0.100

PASS B1 ALL NME   0.0191 -> 0.0190
PASS B2 Night NME 0.0213 -> 0.0204
FAIL B3 p90       0.0820 -> 0.0996
FAIL B4 gross20   0.138  -> 0.146
FAIL B5 cat40     0.081  -> 0.100
```

해상도를 올려 view 를 늘려도 같은 패턴이다.  네 view 는 여전히 **같은 모델**이라
어려운 곳에서 함께 틀린다.

## FAST-C — 다른 source-only checkpoint 를 넣은 median

membership 을 **결과 보기 전에 동결**했다
(`FAST_C_MEMBERSHIP_FREEZE.json`).  호환 후보 셋을 감사해 전부 기록하고,
§11 이 예시로 지명한 `G38_ONLY_60EP` 를 택했다 — 결과를 보고 고르지 않기 위해서다.

```text
호환성 확인   kpt_shape [9,3] · flip_idx [1,0,3,2,5,4,7,6,8] · nc 1  전 후보 동일
              real supervision 0 (각 데이터셋 train symlink 에 raw_data/01_real 0 건)
              inference padding PAD 100 BORDER_REFLECT_101 공통
```

```text
set                n_kp   NME med   NME p90    px p90   gross20   cat40
ALL / R0           1801    0.0190    0.0795     30.79     0.144   0.084
ALL / FAST-C       1801    0.0183    0.1054     38.97     0.159   0.099
day / R0            281    0.0225    0.0820     40.07     0.246   0.103
day / FAST-C        281    0.0258    0.4372    178.81     0.356   0.214
night / R0          168    0.0203    0.0640     20.40     0.101   0.060
night / FAST-C      168    0.0208    0.0492     19.26     0.095   0.060

PASS C1 ALL NME   0.0190 -> 0.0183
FAIL C2 Night NME 0.0203 -> 0.0208
FAIL C3 p90       0.0795 -> 0.1054
FAIL C4 gross20   0.144  -> 0.159
FAIL C5 cat40     0.084  -> 0.099
축 순열            0.048  -> 0.068   (q>=.75  0.104 -> 0.159)
```

**가장 나쁘다.**  주간 p90 이 0.0820 -> 0.4372 (40 px -> 179 px)로 무너지고 축 순열이
0.048 -> 0.068 로 는다.  다른 데이터로 학습한 모델을 median 에 넣으면 그 모델이 크게
틀린 프레임에서 합의가 끌려간다.

야간만 유일하게 조금 나아진다(gross 0.101 -> 0.095, p90 0.0640 -> 0.0492) — 다만
C2 가 실패했고 나머지 넷도 실패라 gate 는 FAIL 이다.

## 왜 전부 같은 방향으로 실패했나

세 후보 모두 **중앙값은 개선, 꼬리는 악화**다.  합의는 여러 관측이 **독립적으로**
틀릴 때 이득이 되는데, 여기서는 그렇지 않다.

- FAST-A / FAST-B 는 **같은 가중치**의 다른 view 다.  R0 가 틀리는 곳에서는 flip 도
  960 도 같이 틀린다.  평균/median 은 그 오류를 지우지 못하고 좋은 관측만 오염시킨다.
- FAST-C 는 다른 모델이지만 **더 나쁘다** — 독립성을 얻은 대신 품질이 떨어지는
  관측을 median 에 넣었기 때문이다.

즉 **관측을 더 모으는 것만으로는 R0 의 teacher ceiling 을 넘지 못한다.**

## 결과

```text
[FAST-A]  새 추론 0     GATE FAIL  (A3 p90, A4 gross20)
[FAST-B]  새 추론 638   GATE FAIL  (B3 p90, B4 gross20, B5 cat40)
[FAST-C]  새 추론 638   GATE FAIL  (C2~C5)

BEST FAST TEACHER            없음
teacher better than R0       NO
student training started     NO

FAST_NO_TRAIN_TEACHER_WORKS = NO
```

## 60ep source-teacher 를 돌릴 가치가 있나

이 probe 가 답하지 **못하는** 것을 분명히 해 둔다.

```text
확인됨    같은 R0 의 view 를 늘리는 것으로는 안 된다 (FAST-A, FAST-B)
확인됨    비슷한 급의 다른 source-only nano 를 섞는 것도 안 된다 (FAST-C)
미확인    더 큰 용량(YOLO26m)이 nano 보다 keypoint 를 잘 찍는가
```

FAST-C 가 실패한 이유는 "앙상블이 안 된다" 가 아니라 **넣은 모델이 R0 보다 못했기
때문**일 수 있다.  용량을 올린 teacher 는 그 점에서 다르다 — 이 probe 로는 배제되지
않는다.

다만 세 번 모두 꼬리가 나빠졌다는 사실은, teacher 를 바꿔도 **꼬리(gross·
catastrophic)가 개선되지 않으면 student 는 움직이지 않을 것**임을 시사한다.  V1~V5 가
보여준 것이 정확히 그것이다 — 중앙값 개선은 학생에게 전이되지 않았다.

따라서 21 시간을 쓴다면 **T1 gate 의 판정 기준은 중앙값이 아니라 p90 과 gross20 이어야
한다.**  그 둘이 개선되지 않으면 medium teacher 도 같은 결말이 된다.
