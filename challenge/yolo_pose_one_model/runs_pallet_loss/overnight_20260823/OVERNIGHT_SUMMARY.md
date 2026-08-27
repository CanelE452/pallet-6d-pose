# OVERNIGHT LOSS QUEUE — 2026-08-23

> screen dataset = V1_FIXED_MATCHED10K (A0 대비 변화를 loss 에 귀속시키기 위한 것).
> 모든 판정은 **engineering screen** 이다. real 평가 전 METHOD_SUPPORTED 선언 금지.

## 1. RUN STATUS

```
stage            status     verdict                            reason
------------------------------------------------------------------------------------------------
S0_ASC_REAL      PASS       NO_REAL_SIGNAL                     
S1_ASC_SEED43    SKIPPED    -                                  S0=NO_REAL_SIGNAL — seed 추가보다 다른 loss 탐색
S2_TAKL          PASS       PASS                               
S3_TAKL_REAL     PASS       NO_REAL_SIGNAL                     
S4_NRL           PASS       PASS                               
S5_NRL_REAL      PASS       NO_REAL_SIGNAL                     
S6_ASC_TAKL      SKIPPED    -                                  조건 미충족 (S0=NO_REAL_SIGNAL, TAKL_SYNTH=PA
S7_AUDIT         RUNNING    -                                  
```

## 2. SYNTHETIC (v1 val 133, 동일 evaluator)

```
model       mAP50-95  corner med      p90  gross20    flip  ep20 mAP
----------------------------------------------------------------------
A0            0.7197        6.25    28.03   0.1600   6.77%    0.0429
ASC42         0.7268        6.48    27.57      nan   5.26%    0.4479
TAKL42        0.7399        6.78    34.59   0.1771   8.27%    0.2829
NRL42         0.7560        6.26    31.33   0.1799   6.77%    0.4664
```

## 3. REAL (EXPLORATORY — canonical semantic accuracy 아님)

```
arm         N     det  corner med      p90  gross20  verdict
------------------------------------------------------------------------
ASC        52   0.471       63.49   137.64   0.9135  NO_REAL_SIGNAL
  (A0)     52   0.450       65.55   104.00   0.9327
NRL        44   0.400       69.45   145.03   0.9574  NO_REAL_SIGNAL
  (A0)     44   0.450       67.20   108.11   0.9233
TAKL       46   0.357       63.01   103.10   0.9158  NO_REAL_SIGNAL
  (A0)     46   0.450       66.29   103.15   0.9429
```

★ real 평가기 건전성 진단 (A0):
```
  G(4) 최소   median 60.36
  전체 8! 최소 median 60.36
  Hungarian   median 60.36
  box IoU     median 0.506
```
세 값이 같다 = 순열/규약 문제가 아니라 **모델이 real 에서 실제로 실패**한다.
즉 real 평가는 현재 loss 후보를 구분할 해상도가 없다.

## 4. CONVERGENCE (mAP50-95 가 처음 넘는 epoch)

```
model          0.3     0.4     0.5     0.6     0.7     AULC
--------------------------------------------------------------
A0              30      30      40      40      59    0.3664
ASC42           20      20      30      40      50    0.4690
TAKL42          30      30      30      40      50    0.4338
NRL42           20      20      30      40      40    0.5255
```

## 5. BEST CANDIDATE

```
BEST_OVERNIGHT_CANDIDATE = S2_TAKL
EVIDENCE_LEVEL           = SYNTHETIC_ONLY_PROVISIONAL
WHY                      = synthetic gate 통과, real 미확인
WHAT_FAILED              = S1_ASC_SEED43, S6_ASC_TAKL
```

## 6. 다음날 자동 실행하지 않는 것

추가 hyperparameter sweep / seed 44,45 / dataset 변경 / 새 architecture /
self-training / paper final claim — 전부 사용자 확인 후.

## 근거 태그

- [확인] 위 모든 수치는 disk artifact 에서 읽었다.
- [추정] 메커니즘 해석은 추정이다.
- [미검증] real 전이·seed 일반화는 확립되지 않았다. ASC 는 현재 **convergence acceleration** 만 확립됐고 final accuracy 우위는 주장하지 않는다.