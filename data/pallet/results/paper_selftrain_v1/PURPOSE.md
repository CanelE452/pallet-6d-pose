# PURPOSE — YOLO26 geometry-filtered self-training (paper MAIN track)

[소비처]
논문 본문 M2 (Table 2, target-domain adaptation under daytime/nighttime) ·
M3 (Table 3, component ablation) · M4 (filter quality) · M5 (robustness),
그리고 M1 의 Proposed row.  `_docs/paper/EXPERIMENTS.md` 가 최종 소비처다.

[문장]
"라벨 없는 target-domain 이미지에서, YOLO detection confidence 로 1차 선별한 뒤
single-keypoint-removal reprojection consistency 와 horizontal-flip keypoint
consistency 로 pseudo-label 을 걸러 self-training 하면, synthetic-only baseline 과
naive self-training 대비 주간·야간 모두에서 2D keypoint 정확도가 개선된다."

## 판단 지표 (결과 보기 전에 고정)

primary  = PAPER_EVAL 의 supervised keypoint location median px (↓) · detection rate (↑)
secondary= AUROC (↑) · FPR95 (↓) · box AP50-95 (↑)
pose     = POSE_METRICS_STATUS 가 READY 가 되기 전까지 보고하지 않는다.

성공 판정: R5(Proposed) 가 R0(synthetic-only) 와 R1(naive ST) 를 Daytime·Nighttime
평균에서 모두 상회하고, 어느 한 조건에서도 catastrophic degradation 이 없다.
실패해도 결과를 지우거나 threshold 를 다시 고르지 않는다 — FAILURE_ANALYSIS.md 를 쓴다.

## 사전등록 (결과 보고 수정 금지)

```
teacher            R0 YOLO26n synthetic-only, 단 하나 (static teacher, one round)
                   sha 970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7
pool               metric_split_lock.md §1.6 pl_pool, balanced Day/Night
TAU_BOX            unlabeled pool 만 보고 결정 (PAPER_EVAL GT 사용 금지)
tau_reproj         0.05   (canonical_filters 무차원 default)
tau_remove         0.05
tau_flip           0.05
keypoint validity  kp_conf >= 0.5, 최소 valid corner 6/8
학습               모든 arm 동일 init·exposure·epoch·seed·augmentation, fliplr=0.0 유지
checkpoint 선택    고정 최종 epoch (last.pt).  PAPER_EVAL 보고 고르지 않는다.
```
