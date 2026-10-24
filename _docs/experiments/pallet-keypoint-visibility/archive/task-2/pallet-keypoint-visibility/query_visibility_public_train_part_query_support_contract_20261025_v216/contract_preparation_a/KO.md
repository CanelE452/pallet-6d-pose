# SOURCE part·query support 약관찰 계약

기존 V215의 12프레임·원 native96행·q/L·RAW96·원 U11을 그대로 쓴다. 새 프레임이나 점을 고르지 않는다. 이번 준비는 값 없는 template와 SOURCE 내부 역할을 고정한 것이며 이미지 관찰·새 라벨·학습 admission은 아직 없다.

관찰자는 같은 12프레임을 각각 확인하고 아래 여섯 항목만 짧게 기록한다. 원 U11은 모든 의미축 UNKNOWN으로 남긴다. 나머지 85행에 양성 라벨 개수를 채울 의무는 없다.

|항목|기록할 근거|
|---|---|
|선택 instance|full frame과 원crop에서 같은 선택 pallet를 식별할 수 있는가. 다른 팔레트·목재·상자의 존재만으로 YES가 되지 않는다.|
|의도 part|원 native slot이 가리키는 끝·코너·블록 등을 같은 물체의 연결·배치로 설명할 수 있는가. 원 native 순서를 바꾸지 않는다.|
|국소 witness|실제 보이는 feature의 종류·짧은 설명·양성근거와 원96 inclusive 정수 ROI. 0..95 범위이며 최소면적, q포함, 중심일치 조건은 없다.|
|주소화|원 q/native와 같은 frame의 맥락으로 그 feature를 묻는다는 양성 weak 근거가 있는 `QUERY_NATIVE_SUPPORTED`, 추가 ROI selector가 있어야 특정되는 `EXPLICIT_ROI_ONLY`, 또는 `UNKNOWN`.|
|query support|원 q의 ON/NEAR/OFF/UNKNOWN과 별도 근거. 이를 native visibility의 답으로 복사하지 않는다.|
|visibility|주소화된 intended part의 D/E/SELF/UNKNOWN. E에는 외부 전경이 실제 그 부위를 가리는 양성 경계·연속성 근거가 필요하다. absence·배경·다른 물체@q·qOFF는 E가 아니다.|

주소화는 완벽한 mesh 코너, 정확 3D crosswalk나 수학적 유일성 인증을 요구하지 않는다. 관찰 가능한 같은 part의 연결과 원 q8 맥락으로 어떤 feature를 묻는지 설명하는 약근거다. 반대로 임의의 crop feature를 두 사람이 똑같이 골라도 이 설명이 없으면 current unconditioned native head의 D/E로 옮길 수 없다. feature가 원crop 안에서 보이고 주소화 근거가 충분하다면 q가 OFF라는 이유만으로 그 visible-part 관찰을 버리지 않는다.

## 두 권위와 두 관찰자

A와 C는 서로의 새 값을 보기 전에 같은 frame의 전체 맥락과 원crop/contact를 실제 확인하고 각 96행을 봉인한다. 기존 V215의 서로 다른 6+6 appearance 관찰은 이 같은-frame 두-reader 의미합의를 대신하지 않는다.

첫째, instance·part·query-native 주소화와 D/E가 두 관찰자에게 일치하면 SOURCE intended-part visibility의 **weak 후보**를 만들 수 있다. D에는 보이는 witness가, E/SELF에는 양성 가림근거 ROI가 필요하다. SELF와 UNKNOWN은 D/E loss 대상이 아니다. 이는 물리 native point gold나 실사 사람 정답과 같지 않다.

둘째, native 대응이 미확정이어도 명시적 국소 feature와 ROI는 보존할 수 있다. 같은 feature 종류와 겹치는 두 ROI는 spatial-agreement 후보이며 union을 남긴다. 종류·겹침만으로 같은 의미부위나 instance가 증명되지는 않으므로 root의 근거 대조가 필요하다. `EXPLICIT_ROI_ONLY`는 미래 ROI-conditioned auxiliary 또는 공간 loss의 후보이며 native primary target이 아니다. 불일치·UNKNOWN은 그대로 보존한다.

paired control은 원 q/L/native/valid, 동일 donor RGB-alpha pattern·scale·rotation과 alpha mass를 고정하고, 두 witness ROI union 전체 opaque COVER 대 union을 건드리지 않는 NOTCOVER/SHAM을 실제 pixel proof로 확인한 때에만 witness-covered 1 대 0 후보가 된다. 주변 samepart witness와 alpha 밖 RGB를 보존하고 실패·SHAM불가·실제변화불가를 남긴다. q/ROI를 옮기거나 줄여 통과시키지 않는다. 이 post-sampler 개입의 target을 native E나 query support로 복사하지 않는다. COVER가 remote feature만 바꿔 q는 그대로일 수도 있고 SHAM이 feature를 보존하면서 q근방을 바꿀 수도 있다.

모든 후보의 `training_admission`은 false다. 미래 loss·입력·ROI conditioning 계약과 root의 별도 admission 전에는 기존 native head에 넣지 않는다. 이 계약은 기존 V212 point target이나 weak44 admission도 수정하지 않는다.

## SOURCE 내부 역할

[B가 봉인한 component split](../component_split_preparation_b/INTERNAL_SOURCE_FIT_DEV_COMPONENT_SPLIT.json)을 그대로 사용한다. 원 role은 전부 TRAIN으로 유지하며 별도 내부 역할은 FIT 8frame/64행/U9, DEV 4frame/32행/U2다. 같은 component/source-group의 모든 native와 파생 pair는 같은 내부 역할을 상속한다. 알 수 없는 component에 fallback하지 않는다.

미래 fit은 DEV component를 새 loss뿐 아니라 PUBLIC/CF 등 모든 기존 보조 loss·pool·normalization에서도 제외해야 한다. 제외된 원 TRAIN을 complement로 HOLDOUT으로 바꾸지 않는다. 이 내부 DEV는 기존 appearance 노출과 old auxiliary 접근 가능성이 있는 SOURCE 개발 분리이며 새 blind 검증이나 기존 보호 HOLDOUT이 아니다.

## 근거와 실행 경계

[계약 JSON](ANNOTATION_AND_SUPERVISION_CANDIDATE_CONTRACT.json) · [행 schema](ANNOTATION_ROW_SCHEMA.json) · [빈 96행 template](INDEPENDENT_ANNOTATION_TEMPLATE96.json) · [순수 검증·join 함수](part_query_support_contract.py) · [가상검사](INVENTED_CHECKS_FINAL.json) · [실제관찰 전 protocol](PROTOCOL_BEFORE_ACTUAL_SAMEFRAME_TWO_READER_ANNOTATION.json)

V212 SOURCE point0의 actual-image ray state와 crop-canvas 적격성은 query-pixel presence나 region/finite-feature visibility와 다르다. 기존 weak44도 3frame의 local-feature 권위이며 새로운 12frame의 native 라벨을 대신하지 않는다. 역사적 HANDOFF의 native→dynamic 순열 방향을 새 public point mapping으로 재사용하거나 이중 적용하지 않는다.

실제 image view는 이 동결 계약·split·정확한 생성 PNG whitelist를 바인딩한 root entry 뒤에만 진행한다. 원 source array·GT·모델·점수·보호 평가 body는 열지 않으며, 현재 준비로 학습·모델/문턱 선택이나 목표 해결을 주장하지 않는다.
