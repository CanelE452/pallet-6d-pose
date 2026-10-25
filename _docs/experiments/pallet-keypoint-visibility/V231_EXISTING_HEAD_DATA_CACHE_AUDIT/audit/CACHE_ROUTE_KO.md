# V231: 기존 공개 YCB SOURCE cache 경로만 확인

C10/C11 실제 run metadata가 가리키는 feature receipt/protocol/code를 확인했다. NPZ·RGB·weight의 stat/hash/header/decode는 하지 않았으며 모델 설계나 학습 선택에는 참여하지 않았다. 모든 asset SHA는 봉인 metadata에 기록된 값이다.

| cache | 봉인 feature shape | 새 YCB 감독과 연결되는 입력 표현 |
|---|---|---|
| V110 natural_surface_rgb_features/SOURCE_RGB_FEATURES.npz (0bad9866…) | [11652,1541] | frozen ResNet18 target512/local96full512/central32_512 + cosine2·validfraction3 |
| V111 spatial_query_features/SOURCE_SPATIAL_QUERY_FEATURES.npz (8c3e4dcf…) | [11652,1928] | frozen DINO T=contextCLS,Q=initial-q patch sample,F=local96CLS,C=central32CLS,P=center2×2 pooled + scalar8 |

둘 다 old3652 units/33984 occurrences와 새 YCB8000 units/occurrences의 **mixed cache**다. 코드에서 새 YCB i,k는 occurrence33984+8i+k, feature unit3652+8i+k로 고정된다. 새 1000 instance는 원 TRAIN이고 700 frame·8000 logical slots이며 native correspondence는 UNKNOWN이다. 독립 component 수/분할은 검토한 입력·feature schema에서 확정하지 않았다. 전체 mixed NPZ를 eager decode하고 나중에 mask하는 것은 현재 허용되지 않는다. 미래 별도 root entry와 TRAIN 선별이 body 해석보다 앞서야 한다.

새YCB raw96/valid와 target-context224 RGB는 두 producer에서 일시 생성하고 SHA facts만 남긴다. 검토한 출력 목록에는 새YCB crop RGB tensor cache가 없다. 원 fullRGB는 INPUTS1000의 encoded image path/SHA로 연결되며 현재 존재·bytes는 재확인하지 않았다. C11의 full256 patch token은 일시 사용하고 Q/P로 pool/sample하므로 전체 token cache도 없다. 다른 미검토 역사 전체의 부재는 주장하지 않는다. old_RGB_refs는 mixed old-family pointer이므로 따라가지 않았다. V113 탐색도 하지 않았다.

입력 q8 자체는 GT reference에 jitter를 준 SOURCE emulation이다. 따라서 GT-origin을 지우고 “모든 GT 영향0”이라고 부르지 않는다. 다만 r/pose/GTmask/depth/teacher/native semantics가 추가 forward 채널로 들어가는 계약은 아니다. GT target은 V109 RAW_SURFACE_TEACHER8000_GT_ONLY.npz라는 별도 sidecar이며 i,k/원 q·L/lineage로 join된다. associated-r D/E 직접 BCE 및 S/U mask는 이미 확인한 C10/C11 소비 계약이다.

경로·SHA·slice·join keys는 SOURCE_CACHE_ROUTE.json에 기록했다. 기존 cache 위치를 안다는 사실은 payload 접근, crop 재생성 또는 새 학습의 실행 승인이 아니다.
