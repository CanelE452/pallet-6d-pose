# V263 실행 기록과 재추출 순서

실제 SYN 7,040행(880 group; FIT5,504/DEV1,536)에서 네 설정을 seed17/각1,000update로 실행했다. 현재223장은 모든 checkpoint와 SYN 선택 설정이 동결된 뒤 읽은 반복 개발 평가다. 신규 blind 평가나 추가 실제 데이터 학습은 아니다. Source cue는 등록 material anchor와 두 geometric visible-edge 방향의 교사 증명이며, photometric direct observability는 미측정이다.

모델은 frozen official DINOv2 ViT-S/14 reg4의 중간 block8(0부터) normalized16×16×384 patch tokens에 원 native0..7 one-hot8을 broadcast한 392채널을 사용한다. 21,989개 파라미터의 작은 공유 stem과 별도 semantic VIS/위치 heatmap·location-state 출력을 학습했다. heatmap soft-argmax가 subpixel 좌표이며 learned offset head는 없다. D+약한 위치 cue는 VIS D/LOC unknown이다. E/S는 location NO_MATCH(해당 native 직접 관측점 없음), amodal r heatmap supervision0이며 S의 semantic VIS target은 unknown이다. 위치 상태를 semantic E로 바꾸지 않는다.

원 실행 순서와 입력:

1. `extract_simple_features.py --entry ACTUAL_SIMPLE_FEATURE_ENTRY.json`: 원 SYN NPZ의 `rgb96` NHWC 및 `valid96`만 읽는다. simple crop은 원 q/L이며 factor1.0이다. compact 재생에서는 이 원 NPZ를 그대로 쓴다.
2. `extract_geometry_features.py --entry ACTUAL_GEOMETRY_FEATURE_ENTRY.json`: B의 ORIGINAL 고정 solver INITIAL_R_cf/centroid와 registered extents/K로 projected bbox maxspan×0.25를 정한다. 중심은 원 q다. GT pose는 사용하지 않는다. 8/880 solver 실패는 simple fallback이며 행을 삭제하지 않는다. fullRGB는 바인딩된 NPZ의 `RGB_uint8`만 읽었다.
3. RGB96를 원 V105 `preprocess_views` 정의대로 mean-fill/224 bilinear antialias/Imagenet normalize하고 full view만 frozen block8에 넣는다. centre view와 pooled CLS는 입력하지 않는다. 원available=False는 token/support0이다. FP16로 저장하고 head forward는 FP32다. 두 encoder 실행의 state 전후 bb4c 및 첫 batch repeat가 일치했다.
4. `run_arm.py --entry ACTUAL_LOC_SIMPLE_ENTRY.json`, 이어 `ACTUAL_LOC_GEOMETRY_ENTRY.json`: LOC 두 설정 각각 1,000회. `LOC_BOTH_FINAL_FROZEN_BEFORE_SYN_DEV.json` 생성 후에만 SYN DEV head readout 및 `select_syn_crop.py`를 실행했다.
5. crop 선택은 세 고정 rule의 conservative verified-correct / **ALL accepted** precision≥0.90 중 전체 query coverage 최대, 동률 raw eligible median error, 재동률 simple이다. unverified accepted도 분모에 남긴다. INITIAL_GEOMETRY+OPEN을 선택했다. 두 crop 모두 보류여도 VIS/JOINT는 실행하도록 사전 고정했다.
6. `run_arm.py --entry ACTUAL_VIS_ENTRY.json`, 이어 `ACTUAL_JOINT_ENTRY.json`: 선택 crop에서 각각 1,000회. AdamW lr0.001/wd0.0001/batch32, 모든 네 설정의 seed17 초기 state와 PCG17 draw ledger SHA가 일치한다. `finalize_source.py`가 네 final/log/draw 동결과 SYN-only 최종 rule 선택 후 `ALL4_FINAL_FREEZE.json`을 저장했다.
7. 그 뒤 `infer_current.py --entry ACTUAL_CURRENT_INFERENCE_ENTRY.json`를 실행했다. 원223 canonical regular RGB만 열었다. physical INITIAL_T와 physical XYZ dimensions를 사용했고, CF native_xyz에 physical T를 곱하지 않았다. 모델/설정 변경0. 미선택 LOCsimple의 current forward0. ORIGINAL+선택LOC+VIS+JOINT 각1,784행을 저장·동결한 뒤 A/B가 별도 late 정답/고정 solver 평가를 수행한다.

원 명령은 지정 MLPY Python의 `-B`로 실행했다. 동일 출력 경로에 재실행하면 보존 guard로 거부한다. 재생은 fresh 출력과 명시적 재배치 바인딩을 사용해야 한다. artifact loader는 pinned ModuleType+compile bytes이며 named path 외 탐색, alias symlink, cache/.pyc, v112를 허용하지 않는다. 공식 encoder runtime·weight·upstream vendor CODE는 dependency manifest로 고정한 외부 의존성이다. 모든 transitive runtime import를 번들한 offline 실행물이라고 주장하지 않는다.

공개용 compact crop 재추출은 SIMPLE 원 NPZ(23MB)와 `GEOMETRY_RGB96_VALID96_REPLAY.npz`(31MB)의 NCHW RGB96/valid96에서 위 3단계를 같은 batch32/order로 수행하면 된다. geometry compact 파일은 동일 bound RGB/q/side/V90 definition을 다시 실행하여 저장했고 전체 저장 support/central4와 byte equality를 확인했다. 원 feature 실행에서 RGB 배열 자체 SHA를 남기지 않아 원 배열과의 retrospective byte 비교라고 부르지 않는다. feature token NPY 두 개(각1.38GB)는 재계산 가능하여 공개 제안에서 제외하고 SHA/shape를 남긴다. 현재 token350MB도 같은 crop cache/정의로 재추출 가능하다.

현재 readout 계약: LOC-only VIS는 미학습 N/A여서 원 solver support를 유지한다. VIS-only는 q0를 유지한다. JOINT는 두 출력을 사용한다. LOC acceptance에 predicted VIS를 쓰지 않는다. solver에서 원 available인 semantic predicted E(logit>0)만 제외하며, legacy U/invalid의 원 solver support를 새로 삭제하지 않는다. current에서 threshold/crop/rule/model 재선택과 추가 학습은 없다. 정답 native/SELF·가림 및 pose 성과는 별도 A/B 보고서의 고정 분모로 읽어야 한다.

V1 offset-head와 random-input probe도 보존했다. throughput497.72행/s는 V1 head를 포함한 임의 입력 시험이며 V2 성능 결과가 아니다. Source 학습은 모두 성공했으며, current metadata key `canonical_mappings223` 추정이 literal `canonical_mappings`와 달랐던 1회 inspection 실패도 별도 기록했다.
