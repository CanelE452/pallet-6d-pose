# 보존 loss 고정 계약

합성 TRAIN normal 입력에서 frozen PRIOR1과 native synthetic GT의 거리 ≤5px인 유효 코너만 보호. 거리 단위는 crop scale을 역변환한 원본 prepared-image px. center8 제외. SourceData/cache_features의 supervised native 채널 계약과 동일한 하나의 전체 identity 순열을 고정하고 점별/학생별 permutation 선택은 하지 않음. eval에서는 기존 whole-object symmetry evaluator를 그대로 유지.

T=1 KL(teacher || student), 공간합 후 batch8의 선택 코너 전체 평균. microbatch별 mask 수로 가중하여 정확한 batch 평균 구현. source stress task와 별도로 동일 source row의 normal 입력 forward; 추가 source identity/exposure는 아니지만 추가 compute이며 compute-matched가 아님. teacher logits은 float32 캐시하여 재사용. real/OCC/stress에는 preserve 0.

TRAIN warmup10, 11번째 batch의 regularization 포함 task gradient와 preserve gradient norm 비율 0.25로 lambda를 1회 정함. clipping/재탐색 없음. warmup state/optimizer 폐기 후 strict PRIOR1 reset.
