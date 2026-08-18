# PURPOSE — stage-wise bias loss, 5-epoch screen

[소비처]
DOPE refinement 를 loss 수준에서 고칠 수 있는지의 결정.
`_docs/audits/STAGEWISE_GO_STOP_GATE.md`.  조건부 DiffPnP 재개 여부도 여기서 갈린다.

[문장]
"기존 Gaussian MSE 에 full-map GT mass / wrong-peak rank / distance / progress
loss 를 더하면, stage4→6 이 confidence 만 올리는 대신 위치를 교정하게 되어
far/depth signed bias 와 confident-wrong tail 이 real mechanism set 에서 줄어든다."

## 판단 지표 (사전 고정)

primary: F2 far median -15% / signed bias -20% / >50px tail -15% /
sharpen-without-correction -30% / F2 paired improved>worsened /
canonical PnP >=72/87 / fixed indexed reproj -10%
guard: near +5% 이내 / 새 PnP failure 0 / 새 >100px 0 / NaN·negative depth 0 /
stage map collapse 없음

## 실행 범위

- architecture 불변.  신규 prediction branch 없음.  belief stage 4~6 만 학습.
- canonical Stage-B 6 roots + 60:40 sampler, 정확히 5 epoch, checkpoint selection 없음.
- canonical PnP 는 centroid 를 correspondence 로 포함한다(실측 확인된 canonical 경로).
- N87 은 epoch0/epoch5 두 번만.  final-test 미사용.
- proposal / router / PGBC / graph / edge / mask / line 재도입 없음.

## 한계 (미리 명시)

- 5 epoch 는 방향성만 본다.
- N87(87 frame, F2 35)은 architecture go/no-go 전용이며 일반화 수치가 아니다.
- lambda 는 gradient-norm 으로 한 번 고정한다.  결과를 보고 재조정하지 않는다.
