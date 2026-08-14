# PURPOSE — VCR-DOPE Gate 0 (viewpoint x corner-role bias repeatability)

[소비처]
교수님이 제기한 "heatmap systematic bias" 가설의 판정, 그리고 VCR-DOPE 구현 착수 여부.
`_docs/audits/VCR_LOSO_GATE.md`.  FAIL 이면 architecture 구현 없이 종료.

[문장]
"DOPE far/depth corner 의 signed bias 는 랜덤 오차가 아니라 camera viewpoint 와
corner physical role 로 예측 가능하며, held-out session 에서도 보정이 전이된다."

## 판단 지표 (사전 고정, 결과 보고 조정 금지)

B2/B3 중 하나가 B0 대비: F2 signed bias -25% / F2 far median -15% /
>50px tail -15% / paired improved>worsened / near +5% 이내 /
PnP >=72 또는 (감소없이 reproj -8%) / 새 >100px 0
+ view necessity: B1(role-constant) 대비 추가 이득 4항목 중 1개 이상

## 실행 범위

- 학습 0 step.  기존 mechanism cache 재사용.  ep57 read-only.
- leave-one-session-out 8 fold.  test session 은 fitting 에 절대 미사용.
- feature basis 8개·ridge lambda 1e-3 은 결과 전에 고정.
- centroid 는 predicted 그대로 유지, canonical PnP 는 centroid 포함.
- 기존 실패 경로(residual/graph/proposal/line/DiffPnP) 재도입 없음.

## 한계 (미리 명시)

- N87 valid corner ~519, fold 당 test corner 약 65개.  소표본이다.
- GT pose 로 view 를 계산한다.  Gate 0 는 "bias 가 view 로 설명되는가" 만 답하고,
  inference 에서 view 를 읽을 수 있는지는 Gate 1 의 질문이다.
- N87 은 mechanism development set 이며 논문 일반화 수치가 아니다.
