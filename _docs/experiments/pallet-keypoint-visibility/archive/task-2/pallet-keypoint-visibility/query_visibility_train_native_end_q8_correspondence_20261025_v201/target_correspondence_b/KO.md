# TRAIN 부위·native·query 대응 확인

D/E/D 감독은 **원 query 중심의 소속**이 아니라, 기존에 지정한 **selected finite end**를 BEFORE에서 보이고 COVER에서 가리고 SHAM에서 남기는 약한 감독이다. 원 q/L/native8은 세 variant에서 그대로다. 다른7 native는 각 대조에서 UNKNOWN이며, 배경/SELF/negative로 바꾸지 않는다.

| 권한 | TRAIN triplet | 원 source 그룹 | q가 후보 end 상자 안 | 상자까지 거리 median/max (crop96 px) |
|---|---:|---:|---:|---:|
| 합성 constructive finite-end |82|41 base|3|5.887 / 33.019|
| 공개 실사 RGB semantic-end weak |27|12 frame|5|4.949 / 23.434|

모든109개·327개의 선언된 약한 D/E/D와 all8 variant 2,616행을 보존한다. 나머지7slot의 2,289행은 UNKNOWN이다. 기존 선택12의288행에서는 known36·inputU24이며 q/L/native/원ID가 세 variant에서 같음을 CSV로 확인했다. 원 target 배열은 다시 열지 않았다. 상자는 보수적인 부위 후보이고, 상자 중심은 원 정확 endpoint가 아니다. 위 거리는 원 물리 좌표 오차·query quality·라벨 오류나 성능 원인이 아니다.

합성은 native slot을 저장된 permutation으로 canonical role에 연결한 뒤, 기존 anchor index0을 공유하는3면 patch를 anchor 기준0.2로 줄인 유한 영역이다. public-real은 RGB 구조상 관찰한 같은 재료 끝의 후보 상자와 주변 witness이다. Real27개의 canonical 매핑은 모두 미제공 UNKNOWN이다. 이 차이는 weak 권한 태그로 보존하며, 사용 가능한 selected-end weak 감독을 무조건 완벽한 mesh/pose 증명이 있어야 한다는 새 gate로 막지 않는다. 기존 human D/E 의미도 부정하지 않는다.

Native 입력 슬롯별 known triplet은 0..7 순서로 **35/26/13/22/6/7/0/0**이다. 그렇지만 합성 canonical6은9개, canonical7은4개가 native2/3를 통해 포함된다. 입력 native6/7 known0만으로 관련 D/E 부위가 빠졌다거나 SELF 필터가 잘못됐다고 결론낼 수 없다. 허용된109 known snapshot에는 native6/7의 원 SELF/D/E/OOF 사유가 없다. 선택12의 native6/7은 각36행 모두 inputavailable이지만 direct target은 by-design UNKNOWN이다.

기존 선택순서의 domain별 첫 사례 Q0041_D3와 V189_BASE020_NATIVE4에 대해 contact2장과 BEFORE context2장을 실제로 보았다. 공개 사례는 q 표시와 오른쪽 상판 끝 후보가 분리돼 있고, 가림 대조는 끝 후보 쪽에서 일어난다. 합성 사례는 작은 끝 cue와 foreign plane 위치 차이를 보인다. 이 네 장은109개 전수 이미지 판독·물리 native 인증·학습 가능성의 증거가 아니다. HTML은 브라우저로 렌더하지 않았다.

다음 한 가지 작업은 A의 **같은 원 initializer group/native 키로 복원된 q8**을 이 고정 TRAIN109 표에 metadata로만 join해 상대 방위·순서의 입력 가용성을 확인하는 것이다. q8은 deployment에서 실제 얻을 수 있는 초기 좌표여야 하고, end 상자/GT/nominal 위치는 감독 진단 전용으로 분리한다. SELF 증강에서는 old active anchor로 jitter를 역산하는 누출을 피해야 한다. 이번 단계는 새 head·fit·데이터/문턱/모델 선택을 구현하지 않는다.

- [전체109 대응 CSV](ALL109_TRAIN_FINITE_END_QUERY_CORRESPONDENCE.csv)
- [요약 JSON](SUMMARY.json) · [source 그룹](SOURCE_GROUP_COUNTS.csv) · [native/canonical 교차](NATIVE_CANONICAL_COUNTS.csv)
- [입력·감독 계약](ACTIONABLE_TARGET_AND_INPUT_CONTRACT.json) · [실제4장 관찰](ACTUAL_FIXED_TRAIN_VISUAL_OBSERVATIONS.json)
- [분석 코드](analyze_train_target_correspondence.py) · [규칙과 한계](ANALYSIS_CONTRACT_AND_LIMITS.json)
