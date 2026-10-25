# V248 공개 conditional pixel AUX builder — CODE ONLY

Root가 `build_public_pixel_aux.py --entry ROOT_TRUE_ENTRY.json`의 sample/features 두 단계를 따로 허가한다. sample은 TRAIN22 certificate의 지정8개 Boolean member로 새 AUX query를 만들고, features는 저장 sample COMPLETE와 RGB44+공식 DINO에서 pixel1154를 만든다. 원native352·원SOURCE7617·oldprefix는 입력하지 않는다. V32/E32, L64, seed17,17FIT/5DEV와 shortfall 무보충을 고정했다.

Crop은 Root addendum의 V90 pure crop(F64→half-up uint8 CHW96)이다. V107 F32 context math는 사용하지 않으며 원native crop과 byte 동일을 주장하지 않는다. AUX_SAMPLES.npz는 protocol의11keys다. native_slot/SELF=null, 원sampler training_admission=false를 보존하고 Root conditional AUX_FIT admission은 별도 field로 기록한다.

False template은 실행을 거절한다. Root scope는 {approved:true,entry_contract:...}이며, contract는 완성 entry에서 root_scope와 그 자체의 metadata whitelist binding만 제거한 사전이다. feature entry에 sample COMPLETE와5개 metadata output binding을 metadata_whitelist에, frame44 RGB binding을 RGB_whitelist44/payload_whitelist에 추가한다. sample_complete와 조건부 admission은 Root가 나중에 채운다. 공식cfg는 V239 actual entry의 원사전과 동일해야 하며 cfg code closure는 outer whitelist에 합집합해 두었다.

별도 independent featureQA gate는 추가하지 않는다. trainer 내부 optimizer 전 globalQA에 연결한다. 실제 certificate/RGB/weight/model/feature/fit 실행0이며 준비검사는 false-entry/role metadata와 만든 배열의 crop 경계2예뿐이다.
