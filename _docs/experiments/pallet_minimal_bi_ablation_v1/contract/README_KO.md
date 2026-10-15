# 고정 238장 입력 계약

CAL 46장, 기존 TEST 49장, 나머지 143장을 원래 순서로 모두 유지했다. PRIMARY는 TEST+REMAINDER 192장이다. 각 이미지의 RAW YOLO와 N3_SUBPIX는 같은 기존 검출 실행·선택 후보·원본 영상·confidence를 공유한다. 새 YOLO/N3 추론, 학습, seed 선택은 없었다.

비공개 대형 산출물의 위치는 실행자가 `PALLET_ARTIFACT_ROOT`로 지정한다. 최종 `inputs_v2/INPUTS.json`과 `inputs_v2/INPUTS.jsonl.gz`는 같은 238개 predictor-only 행이다. [입력 계약](INPUT_CONTRACT_V1.json), [준비 영수증](INPUT_PREPARATION_RECEIPT_V1.json), [촬영·소스 보충 계약](SOURCE_CAPTURE_AND_INPUT_CLOSURE_V1.json), [경로 설정 V2 재현](INPUT_BUILDER_REPLAY_V2.json)에 SHA를 기록했다. 참조 파일은 별도 scorer manifest로만 연결하며 수치 입력에는 포함하지 않는다.

각 행에는 `id/partition/session/cluster_id/image/image_sha256/raw_hw/K/xyz/selected_index/candidate_metadata`와 두 `arms`가 있다. `RAW_YOLO`와 `N3_SUBPIX` 각각의 `native_points/keypoints_conf/prediction_support/coarse_pose`를 사용한다. 두 팔의 자체 state·mask·B/I는 이후 각각 계산한다. 기존 F는 accepted/fallback 판정을 저장하지 않았으므로 해당 메타데이터는 null(NA)이다.

RAW F는 기존 102개를 그대로 사용했고 동일 F 함수로 전부 대조한 최대 차이는 0이었다. 누락 136개만 원래 `pose.infer`의 prediction-only 함수와 동일 SQPnP/LM·W/D selector로 CPU 1개에서 계산했다. N3_SUBPIX F는 238개 모두 저장 값을 유지했다. 두 팔의 coarse pose는 각각 238개 모두 available이다. 과거 gain10 입력과 q·confidence·support·자세·K·치수·영상·선택 후보가 10장 모두 정확히 일치한다. gain10은 사후 선택 사례이며 전체 결과의 근거를 대신하지 않는다.

치수는 물리 XYZ(metre)다. 이전 `dimensions_whd_m`라는 필드의 실제 값도 이 자료에서는 XYZ이므로 이름만 보고 축을 바꾸지 않았다. R143의 과거 `raw_points` 필드는 실제 N3_SUBPIX 좌표여서 RAW q0로 사용하지 않았다. `N0_BASE_REPLAY_seed1`도 RAW YOLO로 사용하지 않았다.

원본 238개 영상과 238개 선택 캐시 SHA가 기존 기록과 일치하고 영상 SHA 중복은 0이다. 원촬영 출처 map으로 CAL 4·TEST 4·REMAINDER 2·PRIMARY 6개 군집을 고정했다. 낮/밤의 plastic/wood 세션은 각각 같은 녹화 군집으로 합쳤다. CAL과 PRIMARY 녹화 군집은 겹치지 않는다. 이 map은 콘텐츠·출처 기반 묶음이며 독립 촬영을 새로 인증한 것은 아니다.

옛 245/110 표의 정확한 ID 원장은 아직 특정하지 못했다. 일부 보고서의 245는 49장×5개 방법의 예측 수이고, 다른 245는 별도 촬영 주석 수다. 이 숫자들을 238장의 모집단으로 간주하거나 차집합 탈락 이유를 추측하지 않는다. 현재 실행은 봉인된 정확한 238개 ID를 사용한다.

최초 입력 준비는 1.667초 wall / 1.456초 CPU, 경로를 외부 설정으로 바꾼 최종 V2는 1.305초 wall / 1.305초 CPU였다. V1·중간 V2도 비공개 원위치에 보존했다. 각 준비는 CPU 프로세스 1개·BLAS/OpenCV 1 thread였으며 238회 F 호출 중 102회는 저장 값 대조이고 136회가 누락 자세 계산이다. 세 차례 준비의 F 호출 합계는 714회이며, 원본 고유 누락 ID 수는 136개다. V2의 전체 238개 행과 INPUTS JSON 바이트는 최초 V1과 정확히 같다. 입력 준비에서 영상 decode·NN·GPU·참조 loader 호출은 모두 0이다. 별도 소스 조사 및 단위검사 시간은 이 구간에 포함하지 않았다.

```bash
env PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m scripts.research.pallet_minimal_bi_ablation_v1.inputs verify \
  --source-root "$PALLET_SOURCE_ROOT" --output "$PALLET_ARTIFACT_ROOT/inputs_v2"
```
