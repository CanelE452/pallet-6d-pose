# PURPOSE — h9 Naïve ST (filter 없음)

[소비처] 논문 Method 비교표의 **Naïve ST 행** (Synthetic only / Naïve ST / Reproj+flip ST / Ours(loo+flip)
× ADD(m)↓ · Yaw(°)↓ · Detection rate(%)↑). 현재 이 행만 학습된 모델이 없어 표가 미완성.

[문장] "필터 없는 self-training(검출되면 전부 PL)은 PL 오염으로 pose 정확도가 baseline 이하로
악화된다 — 개선의 원인은 self-training 자체가 아니라 기하 self-consistency 필터(loo+flip)다."

## 구성 (h8 loo+flip 과 완전 동일, filter_type 만 다름 = paired 비교)
- base: `weights/paper_s2_stageB/net_epoch_0057_noseg.pth` (synthetic-only s2)
- synthetic anchor: `data/pallet/training_data/aug_squash_v2` (2212)
- real unlabeled: `data/pallet/real_unlabeled_ralph_{outside,night,noapril}` (500/500/170, eval GT 홀드아웃)
- filter_type: **none** (peak≥0.3 검출 ≥5kp 이면 전부 PL — 무필터)
- rounds 2, epochs/round 3, lr/seed 등 config/stage3_selftrain.yaml 동일

## 판단 지표 (self-domain, THRESH 0.3, manual GT)
- ADD(m) median · yaw(°) median · det rate(N/eval_gt_n) 를 R0 / loo+flip 과 같은 하네스로 비교.
- 기대: Naïve 는 PL 수 ↑(무필터) 이지만 ADD/yaw 가 R0 이하로 악화. 만약 Naïve 가 loo+flip 과
  동등하면 "필터가 원인"이라는 논문 주장이 무너지므로 그대로 보고한다.

## caveat
pseudo-GT(2D클릭→PnP) floor nonzero (outside 0.027 / night 0.028 / noapril 0.078 m),
noapril N=15~18 소표본, unpaired N(모델별 검출셋 다름).
