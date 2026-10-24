# V218 독립 실제 데이터 검산

weak9 triplet와 별도 GT-query 360행의 저장 픽셀·메타데이터 검산이 모두 PASS입니다. 원본과 생성 PNG의 SHA를 각 lane에서 먼저 닫은 후 decode했고, 마지막에 같은 닫힌 목록을 재검사했습니다. 생산자 함수는 import하지 않았습니다.

|검산|결과|
|---|---|
|weak finite-part triplet|9개 / 27 variant, FIT7·DEV2 case|
|COVER 원 ROI OR alpha·원 VALID96·동일 pattern/mass|9/9 일치|
|COVER target 변화 픽셀|1130, 1169, 1053, 939, 957, 1205, 953, 859, 1418|
|SHAM target 변화|전부 0|
|새 GT-query family|360행 / 45 group / nominal contact15|
|독립 sampler의 RGB96 byte parity|360/360|
|strict·any-support RLE|360/360 일치, mass 평균 차이 0|
|새 family SOURCE_FIT / SOURCE_DEV|240 / 120|
|원 unique query / 원 U|96 / 11 그대로; candidate·state 반복 U provenance51|
|생성 query inframe / nominal target inframe|209 / 210|
|crop96 전체 strict-valid|181행; 나머지도 삭제·음성라벨화하지 않음|

이 family의 **q8 자체가 저장된 nominal SOURCE GT에서 유래**합니다. 추가 GT pose/ref/ROI/mask feature 채널을 배포 입력에 넣지 않는다는 사실과, GT 유래 q8을 새 SOURCE 학습용 입력으로 만들었다는 사실은 구분해야 합니다. 원 initializer96을 교정하거나 deployment 성능을 측정한 결과가 아닙니다. 원 input U와 새 family availability도 별도입니다.

weak COVER/SHAM은 post-RAW96의 동일 checker pattern 개입이고 finite witness OR는 약한 bag/영역 selector입니다. target 밖은 IGNORE이며 native D/E 또는 재질 pixel gold가 아닙니다. 이 검산은 RGB/alpha/validity 관계를 확인했으며 intended-instance/native-part의 의미나 자연 가림을 새로 인증하지 않았습니다. 모든 native 라벨은 여기서 UNKNOWN이고 admission·학습·모델 forward·PnP는 0입니다. SOURCE_DEV는 모든 future fit/normalizer에서 제외합니다.

B의 실제 UI image view는 0회입니다. 15 contact는 nominal state0만 표시하며, root/A/C의 실제 view와 의미 검토는 별도 근거입니다.

[weak 실제 QA](actual_weak_v1/REVIEW.json) · [GT-family 실제 QA](actual_GT_v1/REVIEW.json) · [sampler code 사전검토](GT_FAMILY_CODE_PREFLIGHT_V2.json) · [원 weak gallery](../weak_part_pair_controls_v1/PORTABLE_GALLERY_KO.md) · [새 GT-query gallery](../public_source_gt_query_jitter_v1/PORTABLE_GALLERY_KO.md)
