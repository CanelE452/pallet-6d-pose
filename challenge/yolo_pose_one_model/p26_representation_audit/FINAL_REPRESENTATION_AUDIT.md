# P26 REPRESENTATION AUDIT — Y0 frozen, training-0

[CONTRACT]
checkpoint:      runs_posecls_g38/Y26_G38_Y0_VANILLA_30EP_SEED42/weights/last.pt
sha256:          37f904b975db3e95297af5acb51f6e99360f4b59245cef04d0511af3f5a189b1          기존 audit 값과 일치 [확인]
commit:          96ddf1967ecee2759e5d36578a84f2e4eb021efe          기대값과 일치 [확인]
ultralytics:     8.4.60 · torch 2.1.1+cu118 · RTX 3080
dataset counts:  synth val 1,998 · real 128 (DAY 100 / NIGHT 28) · real negative 2,689
training runs:   0
fuse:            호출 안 함 (Pose26.fuse 가 module 을 지우므로)

[M0 PARITY]  PASS
max diffs:   candidate count 0 · conf 0.0 · box 0.0 · keypoints 0.0   (n=64, exact zero)
provenance:  매핑된 source cell 의 sigmoid(logit) vs final conf  max diff 7.04e-08
             64/64 프레임 매핑 성공 → provenance 가 맞다는 독립 증거

[CANDIDATES]
S+           1,997  (missing 1)
R+             118  (correct 없음 10 프레임)
RW              77  (wrong 없음 51 프레임)
RW_RANKFAIL     12
RN           2,258  (NO_CANDIDATE 431 / 2,689)

source-level distribution (★ 그룹마다 크게 다르다 — level-matched 필수)
```
group          P3      P4      P5
S+           1.2%   25.1%   73.7%
R+           0.0%   11.9%   88.1%
RW          28.6%   41.6%   29.9%
RW_RANKFAIL 33.3%   58.3%    8.3%
RN           7.1%   31.6%   61.3%
```

[R+ vs RW — PRIMARY]   5NN AUROC balanced [CI95], logit 은 스칼라 직접 AUROC
```
scope     neck_in            cls1               cls_pen            logit      n(R+/RW)
P3        —                  —                  —                  —          0/22
P4        0.825[0.70,0.94]   0.798[0.68,0.91]   0.694[0.55,0.85]   0.605      14/32
P5        0.876[0.81,0.92]   0.879[0.83,0.93]   0.910[0.85,0.94]   0.966      104/23
ALL       (합칠 수 없음)      0.934[0.91,0.95]   0.932[0.92,0.94]   0.918      118/77
```

**P5 에서 붕괴가 없다.** neck_in 0.876 → cls_pen 0.910 → logit 0.966 으로 오히려 올라간다.
P5 는 R+ 의 88% 가 사는 level 이다. P4 는 내려가지만 R+ n=14 로 표본이 얇다.

[R+ vs RW_RANKFAIL]
```
scope     cls1               cls_pen            logit      n
P4        0.633[0.55,0.72]   0.541[0.24,0.78]   0.520      14/7
ALL       0.875[0.78,0.95]   0.859[0.76,0.95]   0.835      118/12
```
RANKFAIL 은 P3/P4 에 92% 가 몰려 있다(P5 1 개). ALL 수치는 level 교락을 포함한다.

[R+ vs RN]
```
scope     neck_in            cls1               cls_pen            logit      n
P4        0.941[0.84,1.00]   0.967[0.89,1.00]   0.957[0.88,1.00]   0.739      14/713
P5        0.977[0.95,0.99]   0.951[0.93,0.98]   0.960[0.94,0.97]   0.932      104/1385
ALL       —                  0.950[0.93,0.97]   0.962[0.95,0.98]   0.912      118/2258
```
real negative 는 representation 에서 **매우 잘 분리된다**. CASE D 아님.

[S+ vs R+]
```
scope              neck_in   cls1    cls_pen   logit
raw                  —       0.952   0.914     0.790
level-matched P5    0.967    0.949   0.902     0.827
size-matched P5     0.964    0.949   0.902     0.601*
DAY   (P5)          0.964    0.947   0.894     0.584*
NIGHT (P5)          0.967    0.986   0.934     0.733*    n_real=15
```
*표시는 5NN 값(부분집합). level·size 를 맞춰도 gap 이 0.9 대로 유지되고 **NIGHT 이 DAY 보다 크다**.
gap 은 tower 를 지나며 줄어든다(0.967 → 0.902) — classification tower 가 domain 차이를 일부 정규화한다.

[LOGIT RANKING]   같은 프레임 R+ vs RW score margin
```
scope   n     median     p10       p90       frac(R+ > RW)
ALL     67   +0.5294   -0.0804   +0.9280        0.821
DAY     49   +0.5976   -0.0089   +0.9435        0.878
NIGHT   18   +0.3150   -0.1968   +0.8066        0.667
```
**cls_pen 0.910 / logit 0.966 인데도 프레임의 17.9%(NIGHT 33.3%)에서 wrong 이 이긴다.**

cross-scale 기여 (보조):
```
같은 level  n=28  margin median +0.687  RW 승률 0.143
다른 level  n=39  margin median +0.359  RW 승률 0.205
```
다른 level pair 가 더 자주 뒤집히지만 같은 level 에서도 14.3% 뒤집힌다 —
cross-scale calibration 만으로 설명되지 않는다.

linear readout probe (cls_pen 5-fold CV vs 실제 logit, balanced):
```
R+ vs RW  P5   linear 0.951   actual logit 0.972   → readout 손실 없음
R+ vs RN  P5   linear 0.957   actual logit 0.936   → 손실 +0.021 (미미)
```
**readout 이 정보를 버리는 것이 아니다.**

[KEYPOINT SECONDARY]
R+ 의 9kp median 16.52 px · p90 60.04 px · 20px 초과 36.4%
GOOD_KP vs BAD_KP separability (P5, n=67/37):  pose_pen 0.699 · cls_pen 0.659
→ representation 이 keypoint 품질을 **약하게만** encode 한다.
   (primary gate 와 분리해 읽는다)

[CAUSE]
PRIMARY   : OBJECTIVE_BOTTLENECK        (CASE C)
SECONDARY : DOMAIN_REPRESENTATION_GAP   (CASE E)

근거:
- CASE A 기각 — neck_in 에서 이미 0.876(P5) 로 높다.
- CASE B 기각 — P5 에서 cls tower 를 지나며 오히려 올라간다(0.876 → 0.910 → 0.966).
  P4 의 하락(0.825 → 0.694 → 0.605)은 R+ n=14 로 표본이 얇아 SECONDARY 로만 기록.
- CASE C 성립 — cls_penultimate 까지 separability 가 높고(0.910) logit 도 높은데(0.966)
  **프레임 단위 ranking 이 17.9%(NIGHT 33.3%) 실패**한다. linear probe 로 readout 손실이
  없음을 확인했으므로 남는 것은 무엇을 높게 만들라고 가르쳤는가 = objective/assignment 다.
- CASE D 기각 — R+ vs RN 이 0.96 으로 가장 잘 분리된다.
- CASE E 성립(SECONDARY) — level·size matched 후에도 S+ vs R+ 가 0.90~0.97 이고
  NIGHT(0.934) > DAY(0.894). 다만 gap 이 tower 를 지나며 줄어들어 최종 병목은 아니다.

[NEXT]  설계안만. 학습하지 않는다.
1) 선행(training-0): **assignment / target-score audit**.
   G38 학습 배치에서 TAL assigner 가 만든 target_scores 를 R+ 대응 anchor 와 distractor
   위치에서 비교한다. 지금 관측은 "표현은 가르는데 순위가 뒤집힌다" 이므로,
   학습 시 target 자체가 distractor 를 충분히 낮추지 않았을 가능성을 먼저 확인해야 한다.
2) 그 다음 후보(loss): **within-image pairwise ranking term**.
   같은 이미지의 (R+, RW) 쌍에 margin ranking 을 걸어 절대 target 을 낮추지 않고
   **상대 순서만** 강제한다. Y1(=cls target 에 q_pose 를 곱해 절대값을 깎는 방식)은
   이미 반증됐으므로(NIGHT top1 14/28 → 8/28) 절대 target 을 건드리는 설계는 제외한다.
architecture(backbone/neck/head) 변경은 이 audit 이 지지하지 않는다.

[VERIFY]
checkpoint SHA before/after:  37f904b975db3e95297af5acb51f6e99360f4b59245cef04d0511af3f5a189b1 / 37f904b975db3e95297af5acb51f6e99360f4b59245cef04d0511af3f5a189b1   동일
mtime before/after:           1787641648 / 1787641648   동일
training_runs = 0
model parameter diff = 0  (가중치를 로드만 하고 어떤 optimizer step 도 실행하지 않음)

ARCHITECTURE TARGET:
LOSS_OBJECTIVE
