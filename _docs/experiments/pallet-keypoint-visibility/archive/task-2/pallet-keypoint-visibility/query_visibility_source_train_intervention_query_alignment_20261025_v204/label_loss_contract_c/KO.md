# V204 TRAIN label/loss 계약 검토

**직접 E는 선택한 finite-end의 COVER 상태이지 초기 query 픽셀의 foreign owner나 기존 native E가 아니다.** 원 q/L/native를 옮기지 않는다. 합성은 등록끝 patch의 inner0.2 convention, 공개 실사는 기존 semantic-end weak authority다. 다른7점·inputU·미확정 상태는 UNKNOWN이며 인간 D/E/S 평가권위나 물리 native gold를 바꾸지 않는다.

원component **TRAIN 2784행**만 `control_join_key`로 결합했다. U112/UNKNOWN2457을 보존했고 known327은 D218/E109, 곧 **109 D-E-D triplets(합성82+realweak27)**다. 같은 triplet의 q/L/native/availability는 모두 동일하다. 메타데이터의 RGB96+valid SHA 기준327→312고유이며 서로 다른 D/E가 같은 SHA를 가진 경우는0이다. 원 pixels/target NPZ는 열지 않았으므로 full input identity나 모든 라벨의 완전한 타당성 인증은 아니다.

|감독|정의·경로|
|---|---|
|old visible/foreign2|더 넓은 등록 end patch에서 원96 sampler에 양의 visible/foreign mass가 있는지. 서로 배타적이지 않음.|
|old synthetic owner|query central4의 incident+other=T, foreign=F, BG raw mass. class조건 추출이며 row정규화/UNKNOWN보충0.|
|old public/CF owner|기존 중앙owner weak 관찰만. Public/CF support는 전부−1. YCB-V donor CF의 COVER F/SHAM T는 selected anchor owner 보조감독.|
|new direct|선택 finite-end의 BEFORE0/COVER1/SHAM0. DIRECT행에 oldsupport/owner/quality loss를 새로 붙이지 않음.|

특히 **TRAIN 합성82개 COVER는 저장 broad diagnostic이 모두 `[V=1,F=1]`인 동시에 direct E=1**이다. 작은 finite-end는 막고 바깥 등록patch의 visible witness를 남기도록 만든 결과다. broad V를 direct D로 바꾸면 task가 달라진다. 별도 logits에 이 값들이 존재하는 것은 hard-label 모순이 아니다.

손실은 old192의 다섯 평균(BCE2+owner CE3)을 다시 평균한 값에 balanced direct32(16D/16E) BCE를 weight1로 더한다. 각 old loss 계수는0.2, direct aggregate는1이다. Direct class balancing 아래 BEFORE/COVER/SHAM의 기대 비중은 .25/.50/.25다. 모두 같은 encoder/pool/shared64에 backprop하므로 gradient 간섭은 가능하지만 **현재 code/metadata로 실제 상충이나 실사 실패 인과를 입증하지 않았다.** V202의 nominal fit mask는 모든 pool에 같이 적용하며 제외TRAIN jitter를 HOLD로 바꾸지 않는다.

source shortcut 위험도 제한해서 해석해야 한다. 합성 COVER/SHAM은 같은 plane texture·alpha mass를 쓰고 비편집 배경을 복사하므로 색/foreign 존재만으로 E가 결정되지는 않는다(SHAM도 donor가 있다). 그러나 SHAM의 고정 +X 이동과 합성경계·위치는 label과 상관될 수 있다. 모델이 이를 썼는지는 측정하지 않았다. 입력에는 target/variant/authority/pose/GT enclosure를 넣지 않는다. 이 위험은 금지된 evaluation 또는 명시적 GT label 입력누출과 다르다.

다음의 구분 실험은 보호score를 보고 선택하지 않고 SOURCE TRAIN에서 별도로 사전고정할 수 있다.

1. A의 고정 query-window edit와 finite-end label을 나란히 보고 querycenter foreign와 finite-end E를 분리한다(라벨/qL 변경0).
2. 새 source control에서 같은 end·q/L·donor·mass를 유지하고 SHAM 방향/other-part 위치를 균형화한다. 기존14proof/UNKNOWN을 유지해 위치·렌더단서와 end遮挡 의미를 구분한다.
3. 별도허가한 고정 TRAIN batch에서 loss별 shared gradient norm/cosine을 측정해 실제 optimization 간섭을 확인한다. 이번 검토의 forward/gradient/fit은0이다.

[코드 근거](CODE_EVIDENCE.json) · [TRAIN 전행 join](TRAIN_ONLY_CONTROL_LABEL_JOIN2784.jsonl) · [실제메타데이터 census](TRAIN_METADATA_LABEL_LOSS_CENSUS.json) · [scalar checks](INVENTED_SCALAR_LOSS_CHECKS.json)
