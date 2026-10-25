# Pallox 공식 자료의 제공 범위 확인

2026-10-25 공식 공개 metadata를 다시 확인했습니다. DOI `10.48804/DJLDFS`의 최신 공개판은 **1.0**, 공개일은 **2026-05-15**, dataset 라이선스 표기는 **CC-BY-4.0**입니다. 접근 권한은 open이며 catalog의 1,234개 파일은 restricted=false입니다. `fileAccessRequest=true`라는 서비스 설정만으로 승인 필요 자료라고 해석하지 않았습니다. [KU Leuven 공식 metadata](https://rdr.kuleuven.be/api/datasets/:persistentId/?persistentId=doi:10.48804/DJLDFS), [공식 dataset](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/DJLDFS).

| 확인 대상 | 공식 문서가 제공하거나 설명하는 범위 | 이번 QA의 한계 |
|---|---|---|
| RGB·depth | 640×480, L515, depth는 16bit PNG/mm | 실제 이미지·depth를 열지 않음 |
| pose·K·mesh | frame별 SE3, translation metres, K와 textured mesh | 실제 pose·mesh 수치 검증 없음 |
| mask·visibility | binary instance mask 및 projected-mesh 기반 visible/truncated/occluded 통계 | per-query native·owner·SELF 주석으로 간주하지 않음 |
| 가림 전후 | 같은 촬영 위치에서 무가림 및 cardboard box 가림 RGB-D를 촬영했다고 논문에 명시 | pair-ID 표·파일 순서·픽셀 정합 보증은 확인되지 않음 |
| 주석 생성 | 수동 대응점 기반 pose와 prompt-point+SAM mask | 독립 pixel correction·물리 정확도 인증은 확인되지 않음 |

자료 형식은 [공식 README](https://rdr.kuleuven.be/api/access/datafile/356944), 촬영·주석 방법은 [저자 논문 §4.1](https://www.scitepress.org/Papers/2026/144556/144556.pdf)에 근거합니다. 논문은 단일 물리적 pallet **bin**을 기술하며, 모든 pallet 형상으로 일반화한 자료라고 하지 않습니다. Dataset 라이선스와 논문의 별도 출판 라이선스를 혼동하지 않았습니다.

공식 설명의 **284 annotated frames**와 전체 catalog 수를 구분했습니다. Catalog는 RGB322·depth322·mask290·pose284를 열거합니다. 이는 metadata 집계이며 284행의 완전한 대응을 새로 검증한 결과가 아닙니다. [공식 v1.0 catalog](https://rdr.kuleuven.be/api/datasets/:persistentId/versions/1.0?persistentId=doi:10.48804/DJLDFS).

RGB와 depth가 같은 크기라는 설명은 픽셀 등록의 증명이 아닙니다. 확인한 문서에서 명시적인 RGB–depth alignment/calibration 보증을 찾지 못했습니다. README는 pose를 한 곳에서 camera-to-object라 부르지만, 뒤에서는 `ob_in_cam`과 `p_cam=T@p_obj`로 설명합니다. 방향 불일치를 그대로 남기며 여기서 재투영·정합으로 해결하지 않았습니다. README/catalog의 거리1.5–8m와 논문2–9m도 합치지 않았습니다. [README](https://rdr.kuleuven.be/api/access/datafile/356944), [논문 §4.1](https://www.scitepress.org/Papers/2026/144556/144556.pdf).

native8/keypoint별 D/E/SELF/U, 완전 amodal mask, per-part 외부가림 주석은 확인한 문서에 별도 제공 항목으로 명시되지 않았습니다. Mask의0은 선택한 부재의 외부가림·SELF를 독립 확정하지 않습니다. 이는 제공된 저자 pose/mask를 무효라고 판정한 것이 아니라, 확인한 annotation authority와 미확인 해석을 구분한 것입니다.

V66 기존 provenance의 공식 metadata SHA `1e6d8dd4…`와 README SHA `06344d86…`는 이번 조회와 byteexact 일치했습니다. 기존의 로컬 정합·split·내용 중복 검사 결과는 재검산하거나 새 판단에 사용하지 않았습니다. 현재223은 사용자/Root가 직접 촬영한 별개 자료라는 진술로만 기록하며, 현재 영상·GT·점수 비교는 하지 않았습니다.

원본 RGB/depth/mask/pose/mesh 다운로드, 모델·배열·평가셋 접근, 계정·설치·학습·모델/데이터/architecture 선택은0입니다. 직접 인용은0단어입니다. [정확한 URL·SHA와 범위](REVIEW.json).
