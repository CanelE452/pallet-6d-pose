# 기존 YCB 교사의 실제 consumer 연결 확인

이전 c60406fd 보고서의 downstream UNKNOWN은 원본을 보존하고 여기서 좁게 보완했다. V109 CORPUS_SEAL(934a9a3a…)을 V110/C10 design이 정확히 pin한다. 완료 FIT_COMPLETE→MODEL_FREEZE와 RUN_STARTED→root actual entry는 trainer 3baff03b…·design b7862cca…를 같은 SHA로 가리킨다. V111/C11도 완료 metadata→trainer f34ce3b4…→동일 C10 소비 함수로 이어진다.

C10 `join_ycb`는 D0/E1/S·U−1을 검사하며, known D/E·원 TRAIN·available만 sampling pool과 BCE에 들어간다. YCB_D/YCB_E는 단일 external logit의 **직접 감독**이다. 단지 간접 AUX로 저장되었다고 볼 수 없다. S/U를 E로 합치는 경로는 이 코드에 없다. 공통 손실의 0.25 matched hinge는 old/synthetic triples에 적용되며 자연 YCB 자체가 matched triple인 것은 아니다. C11은 label·pool·draw·loss를 그대로 재사용한다.

입력 q는 GT surface reference r에 jitter를 준 source emulation이다. 교사는 r의 ray/nearest-r evidence로 만들었고, 검토한 consumer는 q 위치 owner/depth를 다시 판독하지 않는다. 따라서 associated-r 감독의 실제 사용은 확인되지만 literal-q entity presence 교사와 같다고 바꾸어 해석하지 않는다. logical auxiliary slot의 pallet native 대응은 UNKNOWN이다.

C10은 frozen ResNet18 영상 descriptor1541+T/C 관계, C11은 frozen DINO T/Q/F/C/P+scalar1928과 영상 관계를 받는다. GT reference/teacher/depth/mask/pose/native를 forward에 추가하는 계약이 아니다. 이것은 기존 소비 경로의 사실이며 새 모델 제안이 아니다.

이 확인은 코드·root entry·완료 metadata의 계보 점검이다. 원 GT/target/lineage/feature 배열, 실제 draw log/gradient, 모델 weight, 과거 score/성능은 읽거나 재현하지 않았다. 다른 모든 후속 경로의 부재도 주장하지 않는다. 새 YCB 학습을 첫 시도라고 표현할 근거는 없으며, 이후 설계·학습 선택에는 참여하지 않았다.
