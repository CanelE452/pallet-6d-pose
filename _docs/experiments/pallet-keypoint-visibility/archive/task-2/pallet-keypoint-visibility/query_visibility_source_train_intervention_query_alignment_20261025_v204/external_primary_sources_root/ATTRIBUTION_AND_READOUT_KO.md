# 공식 공개 source metadata와 출처

MR6D는 Anas Gouda 외 저자의 데이터이며 publisher가 CC-BY-4.0을 명시합니다. 이번에 저장한 것은 공개 API 파일목록·객체명·model bounds·README이며 영상·GT frame payload는 새로 받지 않았습니다. 원문 파일은 수정하지 않았습니다. 기존 source와 동일 revision이므로 새로운 데이터셋 발견으로 보지 않습니다.

- [MR6D publisher와 license](https://huggingface.co/datasets/anas-gouda/mr6d)
- [MR6D 논문](https://arxiv.org/abs/2508.13775)
- [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
- Revision: 3c4ed8453312655499c3d3a3cf38b4929fae77ee
- [이번 repository metadata](MR6D_REPOSITORY_API_METADATA.json)
- [객체명](MR6D_models_names.json): object1은 wooden Euro pallet입니다.
- [모델 bounds](MR6D_models__models_info.json): object1 크기800×1200×144mm.
- 이번 파일목록에는 scene_camera가 있는 directory가 dynamic40/mr14/odyn19/val16개입니다. 논문의 scene92와 같은 집계라고 가정하거나 빠진3개를 채우지 않았습니다.
- dynamic/mr/odyn과 기존 val/publicheld 원촬영의 대응을 확인하지 못해 새 source admission을 하지 않았습니다.

KU Leuven의 Yanming Wu·Eric Demeester가 공개한 Pallox 자료도 metadata와 publisher README만 받았습니다. Publisher는 CC-BY-4.0을 명시합니다. 문서의 총 fully-annotated284와 API의 RGB322·pose284·mask290을 구분합니다. mask 파일명은 SAM1이며 native keypoint D/E 어노테이션으로 승격하지 않습니다. 원 README와 metadata를 바꾸지 않았습니다.

- [KU Leuven 원 데이터 DOI](https://doi.org/10.48804/DJLDFS)
- [publisher API](https://rdr.kuleuven.be/api/datasets/:persistentId/?persistentId=doi:10.48804/DJLDFS)
- [보존한 metadata](DJLDFS_DATASET_API_METADATA.json)
- [보존한 publisher README](DJLDFS_PUBLISHER_README.md)

YCB-V 공식 목록은21개 household object이며 팔레트는 포함하지 않습니다. 이번 로컬 source에서의59 donor 역할은 별도 TRAIN metadata 감사로 확인했습니다. PalLoc6D는 공식 대학 저장소의50,000 합성 이미지·3D 모델 자료로서 실사 native 라벨로 취급하지 않았고 새 payload를 받지 않았습니다.

- [YCB-V 공식 객체 목록](https://github.com/yuxng/YCB_Video_toolbox/blob/master/classes.txt)
- [YCB-V 공식 annotation format](https://github.com/yuxng/YCB_Video_toolbox)
- [PalLoc6D 대학 저장소](https://tore.tuhh.de/entities/product/4b51fbbb-6b30-4468-80c3-fd6d25419310)

초기 MR6D API 요청의2MiB cap으로 JSON이 잘린 실패는 [기록](MR6D_METADATA_INITIAL_CAP_FAILURE.json)에 보존했습니다. cap을16MiB로 늘려 EOF와 JSON을 확인했고 완성 metadata는4,876,470바이트입니다. 이미지·모델 다운로드나 학습 재실행은 없었습니다.
