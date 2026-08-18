# PURPOSE — PPD long-run (L0/M0/M1, 3k/1k)

[소비처]
PalletGraph-6D 논문의 architecture 결정 — learned 5-class polarity line head를
main method에 넣을지, mask branch(M0/M1)를 유지할지 삭제할지, 그리고 full-pose
CGR을 재개할지. `_docs/audits/PPD_ARCHITECTURE_DECISION.md` 로 귀결된다.

[문장]
"학습된 5-channel polarity-aware semantic line map은, 고정된 oracle unsigned
SAI-U candidate set에서 upright candidate를 group-disjoint held-out synthetic과
real mechanism-val N87에서도 선택할 수 있다" — 이 문장이 참인지 거짓인지를
32-frame 암기가 아닌 일반화 수준에서 판정한다.

## 판단 지표 (사전 고정)

primary (progression gate = H3):
- candidate polarity accuracy >= 0.95
- vertical inversion rate <= 0.05
- fixed indexed reprojection: unsigned baseline 대비 70% 이상 감소
- valid candidate 감소 0, NaN/Inf 0

diagnostic (진행 중단 조건 아님, 기존 FAIL 유지):
- pixel precision / recall / macro F1 (H2: historical FAIL)
- mask IoU / Dice (H1: historical FAIL)

## 실행 범위

- training root 단독: `data/pallet/training_data/paper_4pallet_mask_v1`
- split: locked full split train 3039 / val 1045 / untouched 5916 (group overlap 0)
- 20 epoch 고정, early stop 없음, loss/pos_weight/threshold 변경 없음
- checkpoint selection은 synthetic validation만 사용 (N87·untouched 사용 금지)
- predicted map은 candidate re-ranking만, candidate 생성에 사용하지 않음

## 한계 (미리 명시)

- paper_4pallet_mask_v1은 V=8 clean full-view라 close-range/truncation robustness를
  학습했다고 주장할 수 없다.
- N87은 architecture-selection용 mechanism set이며 final-test가 아니다.
- candidate-pair availability와 conditional polarity accuracy를 합쳐 쓰지 않는다.
