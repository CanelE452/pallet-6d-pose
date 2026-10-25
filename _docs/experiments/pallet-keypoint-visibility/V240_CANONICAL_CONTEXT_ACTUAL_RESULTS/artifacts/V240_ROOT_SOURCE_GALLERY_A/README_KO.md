V239 canonical paired SOURCE gallery를 위한 V240 메타데이터 접합입니다. 실제 CSV·캐시·가중치·특징·GT·보호자료는 읽지 않았습니다. 기존 build_gallery.py508cb74c와 false template은 수정하지 않았습니다.

ROOT_GALLERY_SCOPE_PENDING_APPROVAL.json과 ROOT_GALLERY_ENTRY_PENDING_APPROVAL.json은 실제 완료된 fit/report/freeze/all7617 QA 바인딩을 채웠으며 실행승인 값은 false입니다. Root의 별도 승인 후 true entry로 정확 1회 실행합니다.

실행 코드가 고정한 output directory는 ../V239_SOURCE_GALLERY_A/gallery_v1입니다. 기존 준비 파일을 수정하지 않고 해당 fresh child에만 출력합니다. 고정12 정책의 빈 bucket/중복/고유 수를 보존하고, 실제 CSV에서 CASE_IDS를 먼저 봉인한 뒤 선택 group 픽셀만 해석합니다. SOURCE DEV 재사용과 YCB nativeUNKNOWN, 원 q/L 유지 및 원인 확정 불가 한계를 유지합니다.
