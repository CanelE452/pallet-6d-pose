# V263 실행 인터페이스

공식 frozen DINOv2 ViT-S/14 reg4 block8 (0부터), `norm=True`, 16×16×384 patch tokens를 사용한다. CLS pooling을 위치 입력으로 사용하지 않는다. 저장 FP16, head forward FP32. 모든 4개 학습 설정은 원 native0..7 one-hot 8채널을 동일하게 broadcast한다. 입력 ID의 GT 재배열은 없다.

`spatial_head_v2.SpatialPointHead.forward(tokens[N,256,384], support[N,16,16], native_slot[N])`은 별도 semantic VIS E-logit, 위치 heatmap/soft-argmax subpixel 좌표, location MATCH/NO_MATCH/OUT_OF_SEARCH logits를 반환한다. 모델 파라미터 21,989개이며 별도 offset head가 없다. VIS threshold는 logit>0이고 입력 미지원은 U이다. LOCATION 상태를 D/E로 변환하지 않는다. D이며 위치 cue가 약하면 VIS=D이고 location supervision은 unknown이다.

`train_spatial.fixed_targets(...)`의 인수는 `input_available` (원 available과 해당 crop central4 유효성 포함), `direct_cue_known/direct_cue`, `identity_known/identity_correct`, `reference_xy`, 원 `q0`, 입력만으로 만든 `crop_side`다. fixed LOC mask는 available∧known DIRECT two-direction cue∧known ID∧in-search다. RAW96 검색 범위는 normalized [0,1)이며 경계 밖 정답을 border로 clamp하지 않는다. Semantic VIS는 A의 combined D0/E1/SU−1을 별도로 전달한다. GT pose/reference는 loss/진단 sidecar이며 crop/encoder 입력이 아니다.

`fit_tensors(spatial, tokens, support, native_slot, fit, reference_uv, loc_valid, visibility_target, location_target, mode, device)`는 원 FIT 전체 분모에서 PCG17 동일 1000×32 draw를 만들고 seed17, AdamW lr0.001/wd0.0001으로 1000회 갱신한다. masked/U 행은 유지하며 loss만 mask한다. 반환값은 `(model, result, logs1000, draws[1000,32])`다. numeric DEV token 행은 이 함수에서 읽지 않는다. Root가 model/state/log/draw를 저장·freeze한 뒤 `infer_tensors`를 호출한다. 실제 SYN/현재 데이터가 이 모듈에 자동 발견되거나 로드되지 않는다.

Crop simple은 원 predicted input scale L×Root 고정 factor다. Geometry는 고정 원 solver의 INITIAL_T, K, registered dimensions만 project한 span×Root 고정 fraction이다. GT pose는 허용하지 않는다. 초기 solver 실패는 표시하고 simple fallback을 사용하며 행을 삭제하지 않는다. 두 factor와 input packet binding은 실제 feature 생성 전에 Root가 고정한다.

실행 순서는 두 LOC crop fit→두 LOC final freeze→SYN DEV에서 crop 및 세 고정 acceptance rule만 선택→선택 crop의 VIS 및 JOINT fit→모든 final/config freeze→기존223 반복 개발 평가다. LOC/VIS/JOINT loss는 fixed GT mask를 사용하며 predicted VIS로 LOC를 gate하지 않는다. Acceptance는 semantic VIS를 사용하지 않으며 `ACCEPTANCE_RULES_BEFORE_FIT.json`에 원값을 고정했다. Conservative verified precision은 verified-correct accepted / **ALL accepted**이며 unverified accepted도 분모에 남는다. .90 이상 rule 중 coverage 최대, 없으면 all-abstain/q0 fallback이다.

임의 입력 공식 encoder probe는 초당497.72행(batch8), state 전후 bb4c byte 동일을 확인했다. 그 probe는 이전 V1 head 21,799개(별도 offset 포함)의 측정이며 V2 학습 결과가 아니다. V1 code/receipt를 보존했다. V2는 생성 데이터 9개 확인과 생성 optimizer1회만 통과했다. 실제 SYN/current/held 학습 또는 성능 주장은 없다. 16×16 coarse field, soft-argmax의 다봉 평균, 입력 외관/직접 cue의 차이와 confidence 미보정 한계를 유지한다.
