# A1. paper_base 합성학습 성능 (base)  (논문 Table)

> 상태: **미시작** (paper_base 학습은 다른 머신) | 의존: paper_base 학습 완료
> 구분: **다시** (모델 자체가 camera-facing 새 base)

## 목적 (한 줄)
논문용 base 모델 `paper_base`(camera-facing 합성 + squash + truncation padding, v1/v2 제외)의 기준선 성능 확립.

## 판단 지표
합성 val에서 **PCK@3/5/10 · 검출률 · reproj(9kp)**. self-training 전 출발점.

## 설정
- 모델: `paper_base` (학습 데이터 = `mixed_v8_train`(camfacing) + `aug_squash` + `aug_trunc` + `aug_scale`)
- 평가: 합성 val (TBD: held-out 합성셋) + real held-out 일부
- 비교 레퍼런스: `dope_cropaug_pretrain`(squash 없는 전신)

## 방법
1. paper_base 학습 (scratch, train_dope.sh)
2. 합성 val + real 평가
3. dope_cropaug_pretrain 대비 squash 효과 1차 확인

## 결과 (2026-06-06, 합성 val 200 frame, order-free 매칭)
```
model        PCK@3   PCK@5   PCK@10   corner2d_med   PnP%    reproj_med   vol_ratio_med
─────────────────────────────────────────────────────────────────────────────────────
paper_base   0.952   0.988   0.995    11.7px         85.5%   14.1px       1.60
```
- keypoint 위치 매우 정확(PCK@5 0.988). same-index PCK 낮음(@3 0.28)=convention 순서차 → order-free로 봐야 (memory evaluate-on-val-convention-bug).
- volume_ratio median 1.60 = PnP scale 과대(monocular, dims 1.1/1.3/0.11). keypoint는 양호하나 PnP depth 약제약.
- 학습: scratch 60ep, mixed_v8(camfacing 9000)+aug_squash(2819)+aug_trunc(3929)+aug_scale(1592)=17,340. epoch당 ~18.6분(workers=0 병목, 다음 학습은 workers↑).
- 산출: `weights/paper_base/final_net_epoch_0060.pth`, `eval_results/eval_summary.json`.

## 결론
keypoint 품질이 self-training 출발점으로 충분(PCK@5 0.99, corner 11.7px). 단 이는 **합성 val** 기준 — real 일반화는 D1/D2에서. self-training(C1)으로 real 개선 기대.
TODO: dope_cropaug_pretrain(squash 없는 전신) 동일 평가로 squash 효과 1차 비교(A2와 연계).
