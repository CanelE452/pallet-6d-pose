# PURPOSE — corner proposal replacement, 5-epoch architecture screen

[소비처]
PGBC 계열의 다음 결정 — corner identity 를 frozen feature 뒤 corrector 가 아니라
feature 자체에 형성시키는 방향이 맞는지.  `_docs/audits/CORNER_REPLACEMENT_GATE.md`.

[문장]
"ep57 을 초기값으로 마지막 VGG block·belief stage 4~6·corner proposal branch 를
전체 canonical dataset 으로 5 epoch 공동 학습하면, far/depth corner 의 signed bias 와
>50px confident-wrong tail 이 감소한다."

## 판단 지표 (사전 고정)

primary: F2 far/depth median error -15% / signed bias -20% / >50px tail -20% /
paired improved > worsened / predicted PnP >= 72/87 / fixed indexed reproj -10%
guard: near median +5% 이내 / 새 PnP failure 0 / 새 >50px frame 0 /
gate median 0.02~0.98 / C1-base 파국 회귀 없음

## 실행 범위

- ep57 initialization (scratch 아님), canonical Stage-B 6 roots + 60:40 sampler 그대로
- 정확히 5 epoch, checkpoint selection 없음, epoch5 고정 평가
- N87 은 epoch0 / epoch5 두 번만, mechanism screen (final-test 아님)
- graph / edge / mask / line / DiffPnP / centroid proposal 미사용

## 한계 (미리 명시)

- 5 epoch 는 최종 성능이 아니라 방향성만 본다.
- N87 은 architecture go/no-go 전용이며 일반화 수치가 아니다.
- mixed_v8_train 에는 dimensions_m 이 없어 query 의 dims 는 0 으로 들어간다(조작 대신 flag).
