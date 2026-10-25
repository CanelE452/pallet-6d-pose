# V253 의미 연결 및 SOURCE FIT crop 준비

V252는 공유 Linear41→32/ReLU 뒤 출력2개입니다. YCB output1 BCE는 공유 trunk를 학습하지만 pallet용 output0 readout을 직접 감독하지 않습니다. YCB condition32가0이므로 그 loss의 condition-column 직접 gradient도0입니다. 공유 gradient 경로만으로 actual pallet visibility 의미 전이가 입증되지는 않습니다.

SYN MAIN은 렌더링된 실제 원장면의 point0 target-alone D/combined D·E와 crop locality 감독입니다. literal q-owner·whole96·물리적 native gold·whole-pallet presence는 아닙니다. YCB는 associated-r surface AUX/nativeUNKNOWN입니다. 실제 pallet 배포는 원 initializer의 주소/q8/L/U로 condition32와 같은 bank 정책을 계산하는 adapter가 필요하고, SOURCE row cache를 새 입력의 정답 ID처럼 조회할 수 없습니다. 물리적 instance/expected-part 대응은 UNKNOWN을 허용해야 합니다.

기존 V155 common scalar 외부가림 출력, V160 native-slot hidden residual, V166 finite-part CF→same external scalar CODE는 이미 있습니다. 이번 제한 감사는 실행·성공을 재검증하거나 새 model recipe를 고르지 않았습니다.

Crop 수식상 q의 RAW96 주소는 정확히(47.5,47.5), 중앙47/48 픽셀 사이입니다. clean96과 valid PNG/선택 miniNPZ는 원 cache pixels를 보존하고 cross는 별도 표시 카드에만 얹습니다. CODE는 Root 동결정책의36cell/first2 FIT/빈cell을 유지하고, B 봉인 manifest를 읽은 뒤 선택 FIT 숫자행만 mmap에서 복사합니다. 전체 Source파일 SHA/header가 DEV bytes를 포함한다는 사실을 공개합니다. 전체210MB를 새 package에 복사하지 않습니다.

`render_source_fit_cards.py --entry ROOT_TRUE_ENTRY.json`는 Root 별도 승인 후에만 실행됩니다. 성공 산출물은 clean RGB/valid PNG, 카드, OVERVIEW.png, SELECTED_FIT_RAW96.npz, CASE_DATA.json, FIT_TILE_SELECTION.json, PORTABLE_GALLERY_KO.html, MANIFEST/COMPLETE입니다. intended part/instance 시각해석은 UNKNOWN일 수 있으며 새 정답 라벨이나 인과 일반화를 만들지 않습니다. 현재 준비는 CODE/metadata만이며 pixels/CP/model 실행0입니다.
