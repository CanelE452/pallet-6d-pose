# V210 SOURCE native-point head 학습 준비 확인

같은 작은 C61 모델의 입력·구조를 유지하면서, 기존 finite-end 감독과 실제 SOURCE point0 감독을 분리할 준비를 마쳤다. 이 단계는 코드·metadata 확인이며 새 모델 학습이나 성능 개선 결과가 아니다.

원 component TRAIN 합성56320행 중 기존 C61 fit 대상은7040행(SELF NOMINAL3520+BEFORE_ANCHOR3520)이다. 고유 원본은110 base·440 scene·3520 native query다. 증강 행을 독립 장면으로 세지 않는다.

기존 direct TRAIN은 D218/E109=327행이며 합성 finite-end82 triplet과 공개 semantic-weak27 triplet에서 나온다. 입력 native6/7의 direct E는0이지만, 합성 COVER를 원 declared permutation으로 대응한 canonical0..7은 모두 노출되어 있다. native와 canonical을 혼동하지 않는다.

다음 SOURCE point 조회를 위해 정확110개 GT JSON의 기록 binding만 확보했다. 이 준비 단계의 GT·배열·영상·가중치 접근과 새 fit은0이며, point 상태는 아직 이 단계에서 확인하지 않았다. 후속 V211의 실제 조회는 별도 결과로 기록한다.

원 q/L/native/U, RAW96 sampler, 8채널 입력과 original q8 F64 상대좌표 계약을 보존한다. GT reference·crop target·label role은 입력하지 않으며 GT recenter·좌표 추정·PnP는 없다.

- [전체 검토·모델 계약](head_training_code_readiness_a/KO.md)
- [TRAIN metadata census](head_training_code_readiness_a/SOURCE_TRAIN_COVERAGE_CENSUS.json)
- [입력 native별 기존 direct 노출](head_training_code_readiness_a/DIRECT_TRAIN_NATIVE_EXPOSURE.csv)
- [다음110 GT binding 제안—이 단계에서는 미열람](head_training_code_readiness_a/SOURCE110_TRAIN_GT_BINDING_PROPOSAL_NOT_OPENED.json)
- [코드 의존 binding](head_training_code_readiness_a/CODE_DEPENDENCY_BINDINGS.json)
- [검토 JSON](head_training_code_readiness_a/REVIEW.json)

현재223·publicheld45·SOURCE HOLDOUT·COMMON·oldVAL은 학습·보정·모델 선택에 사용하지 않는다. 공개 실사 SOURCE는 사용자가 허용한 별도 출처이며 합성 전용이라고 부르지 않는다. 학습 적합·실사 전이·모델 채택·전체 목표 해결을 주장하지 않는다.
