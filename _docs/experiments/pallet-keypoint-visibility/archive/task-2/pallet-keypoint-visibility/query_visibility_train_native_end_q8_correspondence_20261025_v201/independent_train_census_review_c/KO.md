# TRAIN census·q8 교차 검토

독립 검토 **PASS**. B의 원 TRAIN 109개 control 메타데이터와 선택12의 288행 매핑을 별도 스칼라 식으로 재계산했다. 8,188개 검사가 일치했고 최대 부동소수 차이는 5.69×10⁻¹⁴(고정 허용치 10⁻¹⁰)이었다. B/A producer를 다시 실행하지 않았다.

|TRAIN control|triplet|원 source 그룹|query가 weak 후보 영역 안|영역까지 거리 중앙값, crop px|
|---|---:|---:|---:|---:|
|합성 finite-end|82|41|3/82|5.8873|
|공개 실사 semantic weak|27|12|5/27|4.9493|

109 triplet은 327개 선언된 weak D/E/D와 다른7슬롯 UNKNOWN 2,289개를 유지한다. 이 수는 독립109프레임이나 물리 native gold가 아니다. 영역까지 거리는 weak enclosure와 query의 관계이며 GT 좌표오차가 아니다. native6/7의 known triplet은0이지만 canonical6/7은 각각9/4개로 native2/3에서 감독됐다. 원 SELF·OOF 등의 제외원인을 이 표에서 추정하지 않는다.

A의 최종 census 코드·요약·두 CSV 산술과 선택12 q8의 q/L/U/ID/native 288행 교차 join도 일치했다. TRAIN **64,160 occurrence/inputU 1,145**이며 모두 finite q/L라는 사실이 availability나 visibility를 뜻하지 않는다. C는 A의 65k metadata census나 원 initializer prefix4,096행을 재독립 실행하지 않았다. 원 합성 q8이 실제 initializer 출력이고 GT 투영이 아니라는 결론은 A의 코드·최종 receipt 권위로 확인했다. 원 anchor와 active-only crop jitter의 산술복원 가능성은 GT 오차 복원이나 현재 모델의 shortcut 사용 증거가 아니다.

C는 [root 메타데이터 그림](../scientific_readout_v1/TRAIN_QUERY_AND_WEAK_END.png)을 실제 도구로 **1회** 봤다(원2250×780, 도구2048×710). 3개 패널·82/27·41/12·3/82·5/27·거리/한계 문구가 읽힌다. B의 기존 source PNG 4회 뷰는 별도 receipt이며 C가 원 RGB/crop을 봤다는 뜻이 아니다. 전체109이미지·288이미지·브라우저 렌더 검증 주장은 없다.

[독립 계산](COMPUTED_CHECKS.json) · [검토 receipt](REVIEW.json) · [A 최종 계보](../q8_provenance_a/REVIEW.json) · [B 최종 census](../target_correspondence_b/REVIEW.json)

배열·헤더·원 픽셀·가중치·모델·예측점수·평가/held 타깃을 읽지 않았으며 새 라벨·학습·입력/임계값/모델 선택을 하지 않았다. 실사 실패 원인과 모델 개선은 이 검토로 입증되지 않았다.
