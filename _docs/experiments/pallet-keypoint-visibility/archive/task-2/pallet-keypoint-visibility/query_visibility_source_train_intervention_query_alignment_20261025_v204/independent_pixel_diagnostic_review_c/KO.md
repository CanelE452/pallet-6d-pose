# V204 A TRAIN 픽셀 진단 독립 검토

PASS. 원 TRAIN 메타데이터와 저장 CSV만 대조했다. whole654행, window3270행, 집계90행, source×native0–7 분모16행의 identity·분수·범위·단조성·거리 경계를 독립 확인했다. 원 NPZ/pixel/endmask/target/모델은 열지 않아 원 차이이미지의 신경망/픽셀 계산을 독립 재실행한 것은 아니다.

TRAIN2784/U112/UNKNOWN2457, known327=109 D-E-D(합성82+공개weak27)를 보존한다. 합성82는41base/41component, 공개weak27은12frame/1component다. 중앙2×2 maxabsRGB≥1 변화는 COVER45/109(SYN30/82, PUBLIC15/27), SHAM6/109(SYN2/82, PUBLIC4/27); h3에서는 COVER65/109, SHAM9/109다. 고정 h=.5/1.5/3/6/12는4/16/36/144/576 integer pixel centers이며 threshold1/8/32를 바꾸지 않았다.

C는 사전 metadata 규칙으로 A의 첫2를 제외한 V189_BASE069_NATIVE3, V189_BASE088_NATIVE0, Q0033_D2 카드3개만 tools.view_image로 봤다. label/q/L/native와 clean96·확대 표시·변화heatmap·고정square가 구분된다. 공개카드에서는 donor가 COVER의 q왼쪽과 SHAM의 q부근에 보인다. 이 관찰로 정확한 native 끝부위/instance/가림 원인을 gold로 인증하지 않았다.

**query중심 불변은 finite-end 약라벨 오류의 증거가 아니며, SHAM의 query중심 변화도 재라벨 근거가 아니다.** 두 감독 질문의 차이를 보여 주는 source 입력진단이다. 실제 모델의 단서사용·실사 실패인과·개선·새학습/모델채택은 주장하지 않는다.

[독립 수치 QA](NUMERICAL_QA.json) · [실제3뷰](ACTUAL_FIXED3_CARD_VISUAL_QA.json) · [라벨/손실 계약](../label_loss_contract_c/KO.md)
