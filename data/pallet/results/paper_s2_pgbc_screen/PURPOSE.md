# PURPOSE — PGBC feasibility gates (G0/G1/G2, 학습 없음)

[소비처]
Pallet Graph Belief Corrector 를 구현할지, 어떤 component 로 구현할지의 결정.
`_docs/audits/PGBC_COMPONENT_DECISION.md` 로 귀결된다.  구현 착수 전 관문이다.

[문장]
"frozen ep57 belief 위의 bounded residual 은 far/depth corner 를 실제로 옮길 수 있고
(G0), frozen 50x50 feature 는 GT corner 위치를 wrong peak 와 구별할 정보를 갖고 있으며
(G1), 나머지 7 corner 는 여덟 번째 corner 를 base 보다 정확히 예측한다(G2)."

## 판단 지표 (사전 고정, 결과 보고 조정 금지)

- G0: F2 far corner 의 80% 이상에서 error 50% 이상 감소
- G1: 모든 fold 에서 AUC 또는 accuracy >= 0.75, GT>wrong pair 비율 >= 0.70
- G2: F2 far median error 와 signed bias 모두 20% 이상 감소

## 실행 범위

- 학습 0회.  기존 mechanism cache(87 frame, checkpoint SHA c0055fe7...) 재사용.
- G1 만 shared feature 를 위해 frozen forward 87회 추가.
- N87 은 mechanism capability screen 이며 final-test 가 아니다.
- edge/mask/line/vector/offset/voting branch 없음.

## 한계 (미리 명시)

- G0 의 oracle residual 은 GT 를 사용한다.  capacity 상한 측정이지 학습 가능성 증명이 아니다.
- G1 probe 는 GT 위치를 알고 feature 를 뽑는다.  "정보가 있다"는 말이지
  "학습이 그것을 찾아낸다"는 말이 아니다.
- G2 는 predicted 7 corner 만 사용한다(GT 미사용).  단 F2 프레임은 나머지 7개도
  틀렸을 수 있어, 낮은 수치가 graph 무용을 뜻하지는 않는다.
