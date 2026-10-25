# V257 공개 Pallox native-point teacher 준비 범위

기존 공식 API snapshot과 C의 public-held identity 경로 JSON 두 개만 읽었습니다. 이 문서는 필요 자료를 닫는 준비 기록이며 geometry 검증 완료·학습/평가 승인·모델/자료 선택이 아닙니다. 새 다운로드, RGB/depth/pose/mask/mesh 파일의 stat/hash/header/body 접근은 모두 0입니다.

공식 snapshot은 [DOI10.48804/DJLDFS](https://doi.org/10.48804/DJLDFS)의 RELEASED1.0, 2026-05-15 공개, [CC BY4.0](https://creativecommons.org/licenses/by/4.0/)를 기록합니다. RGB-D, object-in-camera4×4 pose, segmentation 및 visibility statistics 제공은 **publisher 주장**입니다. API 파일목록에서는 cam_K6개, pallox OBJ/MTL/texture와 README를 식별했습니다. 숫자값·파일내용·투영/등록 정확도는 읽지 않았습니다. 정확한 fileID/크기/MD5는 [inert 요구 목록](INERT_PUBLIC_FILE_REQUIREMENTS.json)에 보존했습니다. MD5는 publisher metadata이며 새 asset SHA 검증이 아닙니다.

| 공개 directory | RGB | depth | pose TXT | SAM1 mask | 4종 동명 stem |
| --- | --- | --- | --- | --- | --- |
| 0.55m_hall | 48 | 48 | 28 | 30 | 28 |
| 0.55m_lab | 65 | 65 | 52 | 56 | 52 |
| 1.23m_hall | 48 | 48 | 48 | 48 | 48 |
| 1.23m_lab | 65 | 65 | 60 | 60 | 60 |
| 1.92m_hall | 40 | 40 | 40 | 40 | 40 |
| 1.92m_lab | 56 | 56 | 56 | 56 | 56 |

파일명 수준으로 RGB/depth322쌍, pose284개, mask290개, 4종 stem 공통284개입니다. 이것은 dataset의284 annotated-frame 설명과 연결 가능한 **목록**이며, 모든 RGB가 native teacher를 갖는다는 뜻이 아닙니다. 같은 stem도 registration/instance/pose 일치 증명이 아닙니다.

| 닫아야 할 질문 | 이미 알려진 것 | 아직 필요한 권위/검증 |
| --- | --- | --- |
| 투영 convention | cam_K, mesh, pose와 RGB-D 파일이 공식 목록에 있음 | README·calibration의 수치/영상크기·왜곡·RGB-depth 정렬·depth scale·mesh/pose 단위·축·행렬방향·pixel center, Root가 지정한 정확한 한 frame의4종 파일 join |
| SAM mask와 point visibility | 파일명이 sam1 mask, 공식 설명은 segmentation | mask의 수동 검증/instance 대응, 관측 depth와 target-alone render의 정합, self/external 구분 권위. mask밖을 E/SELF로 바꿀 수 없음 |
| physical native mapping | 대상 OBJ 존재 | mesh에 연결되는 안정된 native/part ID·좌표·symmetry/permutation 정의. bbox/OBJ 극점이나 texture만으로 native gold를 만들 수 없음 |
| public-held capture 중복 | 기존 route는 MR6D val/000001:0..44 및 고정 HF revision | KU/Pallox↔MR6D upstream attribution·capture/file identity 대응. 동일/다른 이름과 CC-BY만으로 중복/비중복 단정 불가 |

`README.md` ID356944, `assets/mesh/pallox.obj` ID357978가 정확한 공개 pointer입니다. 여섯 directory의 `cam_K.txt`는 모두198B와 동일한 publisher MD5를 갖지만 내적 calibration 값/유효성은 UNKNOWN입니다. 각 directory의 `0000` RGB/depth/pose/mask bundle은 파일ID를 찾을 수 있는 metadata 예시로만 열거했으며 학습/평가 frame을 선정하지 않았습니다. occluded-reference pair table 및 native-keypoint map은 이 이름 목록에서 확인되지 않았습니다. 없는 것으로 단정하지 않고 README/공식 권위 추가 확인이 필요한 상태입니다.

기존 publicheld45는 MR6D SOURCE_VALIDATION이라는 **기존 identity receipt의 결론**만 재사용했습니다. 원 held metadata/payload/score를 새로 열지 않았고 Pallox와의 repack/영상 중복 여부는 UNKNOWN을 유지합니다. current223과의 인간 확인 사실을 publicheld45의 비중복 증거로 전용하지 않습니다. 다음 접근은 Root가 exact 자료·용도·held 제외 범위를 별도 봉인해야 하며, 이 영수증은 그 권한을 주지 않습니다.

[검토 영수증](REVIEW.json) · [inert exact 공개 파일 요구 목록](INERT_PUBLIC_FILE_REQUIREMENTS.json)
