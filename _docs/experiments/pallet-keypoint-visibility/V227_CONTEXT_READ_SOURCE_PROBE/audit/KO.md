# V227 저장 모델 진단

두 모델의 736개 저장 logit은 CPU에서 차이 0으로 재현했고 부호도 모두 같다. 두 state는 각각 4,561 parameters·6 tensors이며 유한하다. CONTROL optimizer state는 사용 classifier 2개, TREAT는 6개로 각 300 update다. DEV 606행의 네 block dot-product+bias 합 오차는 CONTROL 9.54e-7, TREAT 1.91e-6 이하다.

| DEV303, truth D267/E36 | 전체 logit CM [[D→D,D→E],[E→D,E→E]] | condition32+bias만 남긴 CM |
|---|---|---|
| CONTROL fixedQ | [[258,9],[2,34]] | [[240,27],[9,27]] |
| TREAT attention | [[258,9],[4,32]] | [[242,25],[10,26]] |

condition32+bias 부분만으로도 신호가 남지만 이것은 학습 후 영상 항을 뺀 partial-logit/OOD 진단이다. geometry/native 정보만이 원인이라거나 RGB가 불필요하다는 증명이 아니다. 실제 전체 logit에는 local/read/contextCLS 기여가 함께 있다. 정확한 606행 가산분해와 각 CM/P/R은 [REVIEW](REVIEW.json), [CSV](ALL606_DEV_ADDITIVE_LOGIT_CONTRIBUTIONS.csv)에 보존했다.

점수·checkpoint를 보기 전에 원 DEV 행 순서 첫 D2/E2를 고정했다. 네 query는 고유 frame3개이며 원 RGB와 같은 q8 context224·local96을 표시했다. local96 RGB/valid는 frozen feature 사실표의 decoded SHA와 일치한다. [고정 4패널](FIXED4_GALLERY_KO.html)은 실제 tool-view로 모두 확인했다. 색축은 0..0.02, 균일 attention은 1/256이다. Attention은 모델 읽기 가중치이며 정답·가림 mask·교정 query가 아니다. entropy는 각 패널과 REVIEW에 기록했다.

고정 대조에서 TREAT의 D 정답 수는 같고 E 정답 수는 34→32였다. 사전 SOURCE 분기 기준 실패를 유지하며 epoch/seed/head 변형·재학습·새 current 추론을 추가하지 않는다. 이 DEV는 이미 노출된 SOURCE이고 weak role 정답을 사용하므로 독립 실용 성능이나 물리적 native 좌표 gold를 입증하지 않는다. 새 GT·pose·reference·보호 데이터는 읽지 않았다.
