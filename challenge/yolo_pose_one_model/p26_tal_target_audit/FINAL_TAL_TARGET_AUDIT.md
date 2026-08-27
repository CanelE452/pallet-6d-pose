# P26 TAL TARGET AUDIT — Y0 frozen, training-0

[CONTRACT]
checkpoint    runs_posecls_g38/Y26_G38_Y0_VANILLA_30EP_SEED42/weights/last.pt
sha256        37f904b975db3e95297af5acb51f6e99360f4b59245cef04d0511af3f5a189b1   기대값·representation audit 과 동일 [확인]
commit        96ddf1967ecee2759e5d36578a84f2e4eb021efe   기대값과 동일 [확인]
ultralytics   8.4.60 · torch 2.1.1+cu118 · RTX 3080
data          G38 val 1,998 · real DEV128 (DAY 100 / NIGHT 28)   ★real128 = diagnostic/dev set
guards        training runs 0 · optimizer steps 0 · backward 0 · model.train() 0 · fuse 0

[M0 PARITY]  PASS
candidate 수 0 · conf 0.0 · box 0.0 · keypoints 0.0  (n=64, exact zero)
provenance   sigmoid(class_logit) vs final conf  max 7.04e-08

[SOURCE TAL FORMULA]   설치본 실측, 기억·문서 인용 아님 [확인]
one2one     TaskAlignedAssigner(topk=7, topk2=1, alpha=0.5, beta=6.0)  → GT 당 anchor 1 개
one2many    TaskAlignedAssigner(topk=10, topk2=10, alpha=0.5, beta=6.0)
align       align_metric = sigmoid(cls)^0.5 · CIoU^6.0
target      target = one_hot · fg_mask · norm_align_metric
            norm_align_metric = (align_metric · pos_overlaps / (pos_align_metrics+eps)).amax
            non-assigned anchor 의 target = 정확히 0
cls loss    BCE(pred, target).sum() / max(target.sum(),1) · hyp.cls
            → P3/P4/P5 가 하나의 합·하나의 분모를 공유한다 (level 별 항 없음)
kp quality included?   NO — TAL 은 cls score 와 box CIoU 만 본다

[REAL R+/RW]   n=67 paired
```
target ordering   R+ target median 0.7882 · RW target 0.0000 (100%)
                  frac(target R+ > RW) 0.896   나머지 10.4% 는 0 vs 0 동률 (역전 아님)
                  RW 가 fg 인 비율 0.000 — assigner 가 distractor 를 positive 로 준 적 없음
logit ordering    frac(logit R+ > RW)   ALL 0.821 · DAY 0.878 · NIGHT 0.667
                  logit margin median   ALL +5.17 · DAY +5.76 · NIGHT +1.87
NIGHT             18 쌍 중 6 역전 (33.3%)
RANKFAIL          n=12 (17.9%) · 그 중 NIGHT 50% · R+ 가 assigned anchor 인 비율 0.667
                  RANKFAIL 에서도 target margin median +0.586 (target 은 여전히 옳다)
```

[SYNTH S+/SW]   n=210 paired
```
target ordering   frac(target S+ > SW) 0.948 · SW target 0.0000
logit ordering    frac(logit S+ > SW) 0.948
rank failure      11/210 = 5.2%      (real 17.9% · NIGHT 33.3%)
★ distractor 자체가 드물다   SYNTH 210/1,998 = 10.5%   vs   REAL 67/128 = 52.3%
```

[TARGET QUALITY]
```
corr target~IoU     REAL +0.973 · SYNTH +0.987     → target 은 사실상 IoU 그 자체
corr target~kpErr   REAL -0.441 · SYNTH -0.374     → IoU 를 매개로 한 간접 상관
GOOD_KP vs BAD_KP (kp err 20px 기준)
  SYNTH  target 0.9628 vs 0.9382 · logit 3.10 vs 2.61 · IoU 0.9635 vs 0.9383
         kp error 2.23 px vs 78.90 px  → keypoint 가 79px 틀려도 target 이 거의 같다
  REAL   target 0.9032 vs 0.7334 (IoU 0.9033 vs 0.7339 과 동행 — 별도 pose 신호 아님)
```
→ CLASS_TARGET_POSE_MISMATCH 성립

[LEVEL CALIBRATION]
```
same-level    n=28  frac(R+>RW) 0.857  margin median +5.95
cross-level   n=39  frac(R+>RW) 0.795  margin median +4.05
    representation audit 의 실패율 14.3% / 20.5% 와 일치 [확인]
REAL logit median   P3 correct n=0 / wrong -4.61 · P4 -1.25 / -1.95 · P5 +2.14 / -5.74
oracle headroom     12 실패 → 6   (offsets P3 -3.0, P4 -1.2, P5 0)
```
★ 이 oracle 은 신뢰할 수 없다 — 최적 P3 offset 이 탐색 격자 경계(-3.0)이고, DEV 에서
  R+ 가 P3 에 하나도 없어 P3 를 눌러도 비용이 0 이다. DEV 에 과적합된 상한이다.

[CAUSE]
PRIMARY   : T3 WITHIN_IMAGE_RANKING_OBJECTIVE
SECONDARY : T4 REAL_RANKING_GENERALIZATION_GAP   (+ T2 를 메커니즘으로 병기)

근거
- T1 기각 — RW 의 target 이 100% 정확히 0 이고 fg 로 뽑힌 적이 없다. assigner 가 순서를
  뒤집은 사례가 0 이다. RANKFAIL 에서 R+ 가 assigned 가 아닌 33% 는 topk2=1 설계상 GT 당
  anchor 가 하나뿐이라 생기는 것으로 순서 역전이 아니다.
- T5 약함 — cross-level 이 더 자주 실패하지만(20.5% vs 14.3%) oracle 회수분이 DEV
  과적합 아티팩트다. backbone/head 변경 근거로 쓸 수 없다.
- T3 성립 — target 은 "R+ 0.79, RW 0.00" 으로 모호할 수 없이 옳은데 최종 logit 이
  프레임의 17.9%(NIGHT 33.3%)에서 역전한다. synthetic 에도 hard pair 가 11 개 존재한다.
  pointwise BCE 가 within-image 순서를 강제하지 못한다.
- T4 병기 — 같은 objective 인데 synthetic 5.2% vs real 17.9% vs NIGHT 33.3%.
  단순 objective 실패로만 단정하지 않는다.
- T2 병기 — target ≡ IoU 이고 keypoint 품질은 target 에 들어가지 않는다. synthetic 에서
  79px 틀린 검출이 2px 검출과 사실상 같은 target 을 받는다.

[PAIRWISE LOSS JUSTIFIED?]
YES — T3 의 세 조건(① target ordering 명확히 정상 ② 같은 프레임에서 logit 빈번 역전
③ synthetic 에도 hard pair 존재)이 모두 충족된다.

★ 다만 신호가 얇다는 것을 설계 전제로 박아야 한다. synthetic 프레임의 89.5% 는
  distractor 자체가 없어 쌍을 만들 수 없고, 쌍이 있는 210 개 중 hard 는 11 개다.

[ARCHITECTURE]
BACKBONE/NECK CHANGE SUPPORTED?   NO — representation 은 이미 가른다(cls_pen 0.910).
CLS HEAD CHANGE SUPPORTED?        NO — level 증거가 DEV 과적합이라 근거 부족.
LOSS/ASSIGNMENT SUPPORTED?        YES (loss 만). assignment 자체는 정상 → assigner 변경 NO.

[NEXT]  설계안만. 학습하지 않는다.
```
L_total = L_stock_YOLO + lambda_rank * L_rank
          stock cls BCE 삭제 금지 · absolute positive target 수정 금지

pair    같은 이미지 안에서
        s_pos = assigned anchor (fg_mask=1) 의 logit
        s_neg = unassigned anchor 중 logit 최고
                ★ IoU<0.5 로 한정하지 않는다 — synthetic 은 그런 후보가 10.5% 뿐
form    L_rank = softplus(margin - (s_pos - s_neg))  또는 동등한 logistic pairwise
```
lambda·margin 은 이번 audit 에서 정하지 않는다. 구현·학습도 하지 않는다.

착수 전 필요한 선행 집계(training-0): synthetic 각 프레임에서 unassigned anchor 의 logit
분포와 s_pos 대비 hard 비율. 신호가 없으면 이 loss 도 무의미하다.

[VERIFY]
training runs = 0 · optimizer steps = 0 · backward = 0 · model.train() 호출 0 · fuse 0
checkpoint sha before/after   37f904b975db3e95297af5acb51f6e99360f4b59245cef04d0511af3f5a189b1 / 37f904b975db3e95297af5acb51f6e99360f4b59245cef04d0511af3f5a189b1   동일
mtime before/after            2026-08-25 16:07:28.570120660 +0900 / 2026-08-25 16:07:28.570120660 +0900   동일
