# 다른 연구자가 검토하는 순서와 실험 계약

이 실험은 전체 319장의 위치·회전을 함께 개선했다는 근거를 얻지 못했다. [RESULT_KO.md](RESULT_KO.md)의 전체 운용 결과와 새 자세 산출률을 먼저 읽고, 아래 원행과 코드를 통해 이 결론을 검토할 수 있다. 그림은 저장된 결과를 설명하는 파생 자료다. 그림을 만든 뒤 모델·임계값·가설·자세를 다시 선택하지 않았다.

## 공개 파일만으로 검산하기

저장소 루트에서 아래 명령을 실행한다. Python 3.9 이상 표준 라이브러리만 사용하며 원본 RGB, detector/N3 가중치, private cache, 환경변수 또는 GPU가 필요하지 않다.

```bash
python3 scripts/research/pallet_observation_refiner_20261009_v1/review_verify.py
python3 scripts/research/pallet_observation_refiner_20261009_v1/review_verify.py \
  --frame-id eval_noapril:1775201415399297536
```

`--root`는 저장소 루트 또는 이 결과 디렉터리를 받는다. `--output`으로 검산 JSON 경로를 지정할 수 있으며 기본은 `REVIEW_CHECKS.json`이다. `--require-manifest`는 최종 게시 bundle manifest가 있는 경우 그 해시 검사를 필수로 요구한다. 매 실행마다 파일을 다시 읽고 원행 수치와 투영을 다시 계산한다. 기존 PASS 파일을 재사용하는 명령이 아니다.

공개 검산은 23개 방법의 319개 ID와 세션, 새 자세/fallback/실패 수, 원행 평균·표본분산·SD·중앙값·선형보간 P90·최대값, JSON/CSV 일치, 저장된 R/t/K/치수의 숨은 코너 투영과 fit 제외, 957개 학습 관측의 식별자·float32 해시·argmax·교점 연결, 실제 시간 원행 600개의 통계를 검사한다. 결과의 `schema`는 `public_review_checks_v1`이며 `checks[].name/status/details`, `problems`, `verified_inputs`, `limits`, 선택적인 `frame_detail`을 읽으면 된다.

이 명령은 **저장된 참조 오차의 재집계**를 검산한다. 원본 정답의 물리적 정확성, detector/head를 원영상에서 다시 실행한 결과, private 학습 캐시의 모든 target을 공개 파일만으로 재현하지는 않는다. 기존 원본과 가중치를 사용하는 전체 재실행은 [REPRODUCE.md](REPRODUCE.md)의 환경변수와 의존성이 필요하다. 당시의 더 넓은 private 검산 39개는 [VERIFICATION.json](VERIFICATION.json)에 보존되어 있다.

실제 public checker는23방법7,337행, pose 분포138개, 신규2,457행의 hidden 투영3,353개, 학습 관측957행, runtime600행/20 timing 통계를 다시 계산했다. private GT뿐 아니라 bootstrap CI 재생성, 실제 detector/training/timing 재실행, 서명된 provenance도 이 public 검사 범위 밖이다. loss-only 입력이 있으면 첫 배치16×84 target의 분류 count와 균일 logits의 해석적 CE loss도 검산한다. 실제 PyTorch gradient 재실행은 아래의 별도 명령이다.

## 입력과 좌표 계약

실사 모집단은 13세션·319개 고정 ID다. [INPUTS.json](INPUTS.json)의 `frames[]`는 `id/session/raw_hw/K/xyz/points/selected_index/candidate_metadata/image_sha256`을 담는다. `points`에 저장된 입력은 `BASE`와 `N3_SUBPIX`이며, 정확도 비교에서 기존 좌표를 그대로 재사용했다. detector나 N3를 새로 학습하지 않았다. 별도의 실제 시간 측정에서는 원 RGB에서 detector와 해당 경로를 다시 실행했다.

| 계약 | 구현 및 확인 위치 |
|---|---|
| 이미지 좌표 | 원본 native pixel. `raw_hw=[height,width]`, solver `image_size=(width,height)` |
| 코너/중심 | cuboid 코너 ID 0–7만 fit에 사용. 중심 ID8은 출력에서도 입력 그대로 보존 |
| 카메라 | 원본 K 그대로, distortion=None. padding·letterbox는 feature sampling에만 적용 |
| 치수 | registry의 W/D/H를 solver의 `xyz=(width,height,depth)` meter로 바꾸는 순열 `[0,2,1]` |
| 물체→카메라 | `X_camera = R_cf @ X_object + centroid`; `centroid` 단위 meter |
| 치수 가설 | W/H/D와 D/H/W 두 대응 가설. W=D이면 한 가설로 합침 |
| physical 회전 | 정규 치수는 `R_physical=R_cf`; W/D 교환 가설은 `R_cf @ rotY(pi/2)` |
| 허용 대칭 | 고정 proper C2. W/D 대응 가설 두 개를 허용하는 것과 90° 회전을 평가상 동치로 취급하는 것은 별개 |
| 보존 항목 | 선택 detector candidate, confidence/box 등 metadata, 중심, 원본 RGB·가중치·사용자 변경 |

코너 순서는 `(-w,-h,-d),(+w,-h,-d),(+w,+h,-d),(-w,+h,-d),(-w,-h,+d),(+w,-h,+d),(+w,+h,+d),(-w,+h,+d)`이며 각 축 부호는 실제 half extent에 곱해진다. 이 8개는 **기하 참조 코너**다. 실제 팔레트 메쉬에 이 코너를 연결하는 모든 bounding 경계가 물리적으로 존재한다고 가정하지 않았다.

[CONTRACT_AUDIT.json](CONTRACT_AUDIT.json)의 `coordinates/symmetry/physical_geometry/reference`와 [solver.py](../../../scripts/research/pallet_observation_refiner_20261009_v1/solver.py)의 `cuboid/project`를 함께 보면 축·단위를 확인할 수 있다. 실사 참조는 기존 기하 재구성값이다. 독립적으로 측정한 물리 위치·회전 정답이 아니다.

## 무학습 방법이 달라지는 지점

아래 여섯 조건을 BASE와 N3_SUBPIX 좌표 각각에 적용했다. 필수12×319=3,828행에 shared319행을 더한 [PREDICTIONS.jsonl.gz](PREDICTIONS.jsonl.gz)의 총행 수는4,147이다. 모든 조건의 동일 입력·K·치수 가설은 같은 `HypothesisBank`를 공유한다.

| 접미어 | fit에서 제외하는 점 | 자세 계산 | 진단/배포 구분 |
|---|---|---|---|
| `NO_MASK_STANDARD` | 없음 | 모든 eligible 점의 표준 SQPnP/IPPE + LM | 배포 가능 대조 |
| `NO_MASK_ROBUST` | 없음 | 유한 4점 부분집합 합의 | 배포 가능 대조 |
| `GEOM_NOSELF_STANDARD` | 예측 자기 가림 H | 표준 해법 + LM | 배포 가능 대조 |
| `GEOM_NOSELF_ROBUST` | 예측 자기 가림 H | 같은 유한 합의 | 배포 가능 대조 |
| `ORACLE_NOSELF_ROBUST` | 사람 자기 가림 | 같은 유한 합의 | `ORACLE_MASK_AND_PHASE` |
| `ORACLE_VISIBLE_ROBUST` | 사람 직접 가시 이외 전부 | 같은 유한 합의 | `ORACLE_MASK_AND_PHASE` |

예측 H는 고정 초기 자세와 convex cuboid proxy에서 계산했다. 물체 좌표의 각 코너에서 카메라 중심 `-Rᵀt`로 향하는 ray(`camera_object−corner`)에 대해 각 축 face cosine의 최댓값이 `-sin(2°)`보다 작으면 자기 가림으로 제외한다. 실제 메쉬 visibility, 외부 물체 가림, 잘림, 영상의 경계 유무를 완벽하게 분류하는 장치가 아니다.

사람 상태는 `DIRECT_VISIBLE/SELF_OCCLUDED/EXTERNAL_OCCLUDED/OUT_OF_FRAME/UNANNOTATED`다. oracle은 고정 기준선의 기존 whole-object 평가 순열로 canonical 사람 상태를 native ID에 옮긴다. GT를 사용한 이 phase와 사람 마스크는 배포 입력이 아니다. `UNANNOTATED`를 직접 가시로 바꾸지 않았다.

기존 `BASE/SUBPIX/N3/N3_SUBPIX` 4개 대조는 [FIXED_CONTROLS.jsonl.gz](FIXED_CONTROLS.jsonl.gz)의 과거 원행이다. 이번 표준 해법은 기존 POSE 구현과 다른 표준 SQPnP/IPPE 경로이므로, 기존 BASE를 `NO_MASK_STANDARD`와 같은 구현이라고 부르면 안 된다. [INITIAL_POSE_PARITY.json](INITIAL_POSE_PARITY.json)은 이번 mask/fallback용으로 실제 다시 계산한 BASE/N3_SUBPIX 초기 자세 638개의 기존 metric 일치 증거다.

`SHARED_BOUNDARY_GEOM_ROBUST`는 기존 무학습 경계 실험의 실제 두 선 교점만 사용한 별도 대조다. 그 경계의 real 물리 소유권은 예측 proxy이며 source 메쉬 정답을 사용한 관측과 동일하게 취급하지 않는다.

## 강건 솔버: 가설 생성, 공유, 점수, 정제

1. 코너0–7에서 finite, sentinel `(-1,-1)` 아님, 원영상 내부인 점만 공통 `eligible` pool에 넣는다. confidence나 GT 정확도로 pool을 바꾸지 않는다. 각 arm의 scoring pool은 `U=eligible−excluded`다. hidden ID는 자동으로 excluded에 합친다.
2. 첫 robust solve에서 공통 eligible pool의 모든 4-ID 부분집합을 생성한다. 8점이면 치수 가설당 `C(8,4)=70`, W/D 두 가설이면 최대140개다. 좌표·K·치수·4점 ID가 같으면 mask 간 재사용한다. 마스크가 허용한 U에 generator ID4개가 모두 포함된 가설만 해당 arm에서 경쟁한다.
3. 정확한 평면은 orthonormal local plane으로 옮겨 generic IPPE를 실행하고, 비평면은 SQPnPGeneric을 실행한다. 각 호출의 모든 반환 해를 보존한다. IPPE_SQUARE를 사용하지 않았다. 4/5점은 [SOLVER_CHECKS.json](SOLVER_CHECKS.json)의 수치·다중해 검증을 거쳤으며 기존 6점 조건의 숫자만 낮춘 구현이 아니다.
4. 비유한 R/t, 잘못된 회전, 전체8코너의 비양수 camera depth를 배제한다. 이미지/3D rank가 부족한 부분집합은 생성 전에 제외한다.
5. **같은 U 전체**에서 pixel 잔차를 계산한다. robust 점수는 inlier 수(`≤8 px`) 최대, 다음 `Σmin(r²,64)` 최소 순이다. 마지막 tie-break는 치수 index·subset lexicographic·solution index·generator다. 초기 자세 prior, GT/사람 visibility, head confidence를 점수에 넣지 않는다.
6. 동일 projected start를 합친 상위 최대3개 후보를 각각 그 후보의 inlier(최소4)로 다시 SQPnP/IPPE 생성하고 LM 정제한다. 정제 후보와 원래 부분집합 후보를 함께 같은 점수로 경쟁시킨다. 따라서 LM이 고정 목적함수를 악화시키면 원래 후보가 최종 선택될 수 있다. 최종 `fit_input_ids`가 이 경우 원래4개인 것은 실제 실행 이력이며 전체 U로 정제했다고 표시하지 않았다.
7. robust는 최종4개 이상 inlier가 필요하다. 정규화된 6-column projection Jacobian의 수치 rank가6보다 작으면 실패다. 약한4점 합의, 조건수, 두 대안의 R/t·inlier gap·score gap을 저장한다. `multiple_solutions=true`는 다른 subset/치수의 대안까지 포함한 뜻이다. 동등한 native PnP 해가 여러 개인 경우만 뜻하지 않는다.

표준 arm은 동일 U·치수·generic 해법에서 전체 SSE가 가장 작은 seed를 골라 U 전체로 LM을 한 번 실행한다. robust 합의 gate를 사용하지 않는다. 두 arm 모두 수치 자세 반환을 정확한 물리 자세 성공으로 부르지 않는다. 낮은 잔차의 일관된 오대응이나 나쁜 배치는 큰 실제 오차를 만들 수 있다.

원행의 `solver.eligible/used/excluded/inliers/fit_input_ids/generator_ids/residuals_used_px/geometry/alternatives/operation_counts/hypothesis_bank_counts`를 비교하면 생성·fit·최종 합의를 분리해 볼 수 있다. `used`는 scoring pool이고 최종 inlier 수가 아니다. `fit_input_ids`와 `inliers`도 항상 같은 집합은 아니다. mask 공유 bank의 누적 호출 수는 각 행마다 더하지 않고 [POSE_EXECUTION.json](POSE_EXECUTION.json)의 bank별 최종 `counts`를 합산해야 한다.

## 숨은 코너 교체와 출력 상태

새 R,t가 구해지면 제외한 자기 가림 H의 **초기 좌표를 최종 투영으로 교체**한다. 이 투영은 독립 관측이 아니며 다시 PnP에 넣지 않는다. 원래 H와 최종 자세에서 다시 계산한 H가 달라도 프레임을 취소하지 않는다. `hidden_initial/hidden_after/hidden_set_changed/reprojected_ids`에 기록한다.

| `output_status` | 의미 | 통계 처리 |
|---|---|---|
| `NEW_POSE` | 이번 관측으로 새 자세 계산 | 신규 조건부와 전체 운용 둘 다 포함 |
| `BASELINE_FALLBACK` | 새 solver 실패, 고정 초기 자세 반환 | 전체 운용 포함, 신규 산출에는 포함하지 않음 |
| `POSE_FAILURE` | 새 solver와 초기 자세 모두 없음 | ID 유지, 실패 수·전체 AUC 분모 유지 |
| `HISTORICAL_FIXED_CONTROL` | 과거 단순 대조 원행 재사용 | 새 자세 계산이나 fallback으로 세지 않음 |

새 solver의 실패 이유는 `solver.reason`에서 `insufficient_observations`, `insufficient_consensus`, `degenerate_or_numerical_generation_failure`, `numerical_rank_deficient`를 구별한다. 바깥 `actual_pose`는 fallback이면 **초기 자세**다. 이를 새 solver의 결과로 읽으면 산출률이 잘못 보인다.

## 정확한 source 감독과 세 소형 학습

기존 G38/P0/TEX 60,000개는 asset inventory다. 모든60,000장에 per-frame mesh ray 검사를 했다는 뜻이 아니다. 정밀 source 감사 패널은 P0 128장+G38 32장이며 기존 TEX32장 보조 감사도 별도로 수행했다. [SYNTH_SUPERVISION_AUDIT.json](SYNTH_SUPERVISION_AUDIT.json)에는 full mesh 비교가 가능했던 source subset, 마스크 IoU·누락 asset·렌더 환경·bindings가 있고 [TEX_ARCHIVE_SAMPLE_AUDIT.json](TEX_ARCHIVE_SAMPLE_AUDIT.json)에 TEX 패널을 기록했다. 정확한 감독이 가능한 기존 P0 RGB1,024개를 사용했으며 새 RGB와 새 기본장면은0개다. source raycast 보조 마스크는 실제 scene.usd 삼각형에서 생성했다. box/hull을 실제 팔레트 마스크로 대체하지 않았다.

[SOURCE_FAMILY_SPLIT.json](SOURCE_FAMILY_SPLIT.json)의 `records[]`는 `id/family/partition/source/raw`를 저장한다. split은 train768/calibration128/source_test128이다. 학습 배치는 train index0–767에서만 나온다. source-test는 고정 probe에만 쓰고 model/threshold/checkpoint를 고르지 않았다. calibration 분할을 보유했지만 이번 고정 설정을 그 점수로 보정하지 않았다. source family 선택은 실사 GT나 source score를 사용하지 않았다.

12개 declared edge마다 초기 RGB 키포인트의 선분을 1/8,…,7/8에서 sampling하므로 영상당84 query다. query의 초기 법선 방향으로 −32…+32 native px의65개 후보를 둔다. target은 query 법선과 실제 source projected edge의 교점이다. 3D segment에 대한 perspective-correct 위치, actual mesh 최근접 거리, ray depth, 원래 delivered visible mask를 확인한다. 물리 거리 허용은 diagonal×1e−5, depth 허용은 diagonal×0.001, 외부 mask 근방은 명목1.5px다.

| target | 저장/손실 | 원천 판단 |
|---|---|---|
| positive | `lo/hi` 인접 두 bin에 subpixel `weight`를 배분 | 검색창 안·물리 edge 위·depth와 visible mask가 관측을 지지 |
| no-match | index65의 분류 loss | 검색창/segment 밖 또는 물리 edge가 자기/외부 가림으로 관측 불가 |
| ignore | `valid=false`, loss와 gradient0 | 비물리 bounding edge, 퇴화 query, 원영상 밖 등 감독을 정당화할 수 없음 |

비물리 수직 bounding edge는 no-match를 억지 학습시키지도 않고 ignore 처리했다. source 86,016 query의 총계는 positive17,825 / no-match35,281 / ignore32,910이다. [SUPERVISION_PREPARATION.json](SUPERVISION_PREPARATION.json)의 `target_counts`와 [training.py](../../../scripts/research/pallet_observation_refiner_20261009_v1/training.py)의 `targets`를 함께 확인한다.

모델은 28→32 Conv1d(kernel3)→ReLU→32→32 Conv1d(kernel3)→ReLU, 1×1 candidate score와 pooled hidden→1 none score다. 총5,890 parameter이며 65후보+none의66 logits를 출력한다. input tensor는 image×84×28×65다.

| channel index | 정보 | 정규화/샘플링 |
|---|---|---|
| 0–2 | brightness, normal/tangent Sobel | brightness[-1,1], Sobel/4; 원영상 bilinear, reflected boundary |
| 3–18 | 고정 detector P3/P4 각8 group mean | frozen neck; bilinear, zero padding, `align_corners=false` |
| 19–24 | offset, query center x/y, tangent x/y, 초기 edge length | offset/32, x/width·y/height, length/image diagonal |
| 25–27 | predicted projected-hull boundary/internal/unavailable | RGB 초기 표준 자세에서 얻은 one-hot proxy role |

여기서 hull은 **역할 feature**에만 쓴 proxy다. 감독 마스크를 hull로 만든 것이 아니다. source cache와 live input 모두 전체 feature를 FP16으로 반올림한 뒤 head를 FP32로 계산한다.

| 학습 arm | zero한 channel | 동일하게 유지한 것 |
|---|---|---|
| `GEOMETRY_ONLY` | image0–18 | 기하19–24 및 예측 기하 role25–27, 전체 architecture/parameter 수 |
| `IMAGE_NO_ROLE` | role25–27 | 영상과 기하 channel, 전체 architecture/parameter 수 |
| `IMAGE_ROLE` | 없음 | 전체 architecture/parameter 수 |

`GEOMETRY_ONLY`는 순수 좌표 기하뿐 아니라 예측 기하 role도 보유한다. 따라서 GEOMETRY_ONLY와 IMAGE_ROLE 차이는 image channel 추가이고, IMAGE_NO_ROLE과 IMAGE_ROLE 차이는 role 추가다. 두 비교의 실사 후단 자세를 source 분류 점수와 구별해야 한다.

loss는 logits의 log-softmax를 positive 두 bin의 soft weight 또는 none index에 적용한다. 각 영상의 valid query 평균을 계산한 뒤 valid query가 있는 영상끼리 평균한다. invalid query는0, 전부 invalid면 연결된0 loss다. 6D loss·PnP 역전파·새 N3 학습·대형 분할망·CLIP/DINO를 사용하지 않았다.

각 arm은 seed1, batch16, 3,000 update, AdamW(lr0.001, weight_decay0.0001), 100-update linear warmup 후3,000까지 cosine→0, gradient norm clip10, 마지막 checkpoint만 사용했다. 정확한 같은 초기 텐서 SHA는 `d40067b3…6d231`, 같은 3,000×16 배치 index SHA는 `016ad882…f0b31`이며 전체값은 [LEARNING_PROTOCOL.json](LEARNING_PROTOCOL.json)에 있다. 독립 checker가 실제 npy와 checkpoint metadata를 읽은 증거는 [VERIFICATION.json](VERIFICATION.json)이다.

[TRAIN_LOGS.jsonl](TRAIN_LOGS.jsonl)은 매 update의 기록이 아니라 각 arm step1 및100배수의 `kind=formal`, step1000/2000/3000의 `kind=source_curve`, 별도100-update `throwaway_preflight`를 저장한다. 정식 update9,000 / 정식 RGB노출144,000이며 preflight와 probe 비용까지 head forward9,268 / RGB노출148,288이다. [TRAINING_COMPLETION.json](TRAINING_COMPLETION.json)의 `checkpoints[].updates/exposures/first_step`와 [SOURCE_LEARNING_CHECKS.json](SOURCE_LEARNING_CHECKS.json)의 초기값·순서·budget·positive/no-match gradient·ignore0·ablated channel gradient 검사를 대조한다.

원 학습 `first_step.target_gradient.ignored` 검사는 all-invalid mask(queries0)의 연결된0 loss를 확인한 것이었다. 상세 보고서 추가 시 [REVIEW_LOSS_MASK_CHECK.json](REVIEW_LOSS_MASK_CHECK.json)에 **실제 첫 정식 배치16장 target**(positive249/no-match635/true-ignore460)을 넣고 per-logit gradient를 추가로 검사했다. 실제 ignore460개 모두0, valid query 모두 비영이며 analytical CE derivative와 일치했다. 이는 CPU loss-only 검산이고 detector/head forward·update·seed는 추가하지 않았다. 원9000 업데이트 증거와 구별한다. Python 3.9 이상과 PyTorch가 설치되어 있으면 private RGB/cache/weight 없이 아래 명령으로 다시 계산할 수 있다.

```bash
python3 scripts/research/pallet_observation_refiner_20261009_v1/review_loss_mask.py \
  --input _docs/experiments/pallet_observation_refiner_20261009_v1/REVIEW_LOSS_MASK_CHECK.json \
  --output /tmp/pallet-loss-mask-review.json
```

`--output`을 생략하면 별도 `REVIEW_LOSS_MASK_RECHECK.json`에 저장한다. 게시된 입력 snapshot·소스·manifest·기존 연구 산출물의 덮어쓰기는 loss 계산 전에 거부한다.

## 학습 관측에서 실제 자세로

query는66개 logits의 argmax 하나 또는 none을 선택한다. 다른 mode/basin의 평균을 만들지 않는다. 같은 edge에 선택 query가2개 이상이면 TLS 선을 만들고, 같은 코너의 incident edge 두 개 중 가장 독립적인 normal 쌍의 교점을 선택한다. 교점은 extrapolation일 수 있다. 7 query를 서로 독립인 7개 3D 코너처럼 세지 않는다.

저장 필드 `all_no_match`의 실제 정의는 `len(lines)==0`이다. 선택 query가0이라는 뜻은 아니다. 957개 관측 중15행은 고립된 query 하나를 선택했지만 선을 만들지 못해 이 flag가 true다. zero-query, zero-line, zero-corner, 부족한 pose observation을 각각 구분한다.

실사 edge 이름/코너 ID가 실제 물리 경계를 맞게 지칭한다는 사실은 RGB 예측만으로 보장되지 않는다. source에서 vertical proxy edge를 ignore한 것도 실사 교점의 물리 유효성을 자동 보장하지 않는다. `corners[].edges/xy/source/extrapolation_possible`, `lines[].support_points/residual_rms_px`, `queries[].selected_xy/no_match`를 함께 검토한다.

세 모델은 원본 Base RGB 좌표·같은 초기 Base mask를 사용한다. query role용 초기 자세는 새 표준 해법이고, 후단 H와 fallback은 고정 기존 Base 자세이므로 둘을 혼동하지 않는다. 선택 교점이 없는 코너는 solver input을 NaN으로 두며 초기 좌표로 채우지 않았다. 유효 4교점이 부족한 상태를 가림 분류 불합격으로 대신 판정하지 않았다.

[CORRESPONDENCE_COUNTS.json](CORRESPONDENCE_COUNTS.json)의 raw 관측4개 이상은 GEO0/319, NO_ROLE6/319, ROLE0/319이며, mask·배치·합의 검사를 거친 새 point 자세는0/3/0이다. fallback319/316/319는 이 간극을 그대로 보존한다. ROLE의 평균이 BASE와 같다는 것은 같은 fallback이 반환되었기 때문이다.

`IMAGE_ROLE_NO_MASK_ROBUST/IMAGE_ROLE_STANDARD`는 **같은 sealed ROLE 관측**에서 mask/solver만 바꾼 절제다. `IMAGE_ROLE_POINT_LINE`도 같은 관측이며 교점이 사용한 edge의 선 factor를 제거하여 같은 edge를 point+line으로 중복 세지 않는다. 이 경로는 초기 자세 주변의 지역 least-squares(`soft_l1`, f_scale8, max_nfev50, rank6, pose prior residual 없음)이며 독립4점 PnP 산출로 주장하지 않는다. 319경로의 checked start317 / 실제 optimizer38 / 전처리 거절279이고 SciPy reported nfev468은 finite-difference Jacobian 평가를 포함한 전체 residual 호출 수가 아니다(NA). [POINT_LINE_EXECUTION_AUDIT.json](POINT_LINE_EXECUTION_AUDIT.json)을 우선 읽는다.

## 원행, 통계, failure 분모

| 질문 | 파일/정확한 key |
|---|---|
| 319장 유지·산출률 | `METRICS.json → methods[method].total_frames/pose_available/new_pose_estimated/fallback_used/no_pose` |
| 전체 운용 오차 | `methods[method].metrics.operational.translation_cm/rotation_deg/ADDsym_cm` |
| 신규 조건부 오차 | 같은 위치의 `metrics.new_pose`, `new_pose_ids` |
| 표본분산·P90 | metric의 `mean/sample_variance/sample_std/median/P90/max/n/unit/ddof` 및 `METRICS.csv` |
| 짝차이 | `PAIRED_COMPARISONS.json → contrasts[new_minus_base].operational/new_pose_common.metrics` |
| 짝비교 분모 | `common_frames/common_ids/excluded_ids/denominator`, 양쪽 `new_marginals/base_marginals` |
| 한 프레임의 실제 자세 | `PREDICTIONS` 또는 `LEARNED_PREDICTIONS → actual_pose.R_cf/R_physical/centroid/cf_extents` |
| 새 자세와 반환 구분 | `new_pose_estimated/fallback_used/no_pose/output_status`, 새 solver `reason` |
| 마스크 맞음/틀림 | `mask_audit.mask_wrong_on_known/false_excluded_visible/false_retained_self` |
| 정확한 pool/최종 inlier | `REAL_CORRESPONDENCE_ROWS → correct_pool_ids/correct_final_inlier_ids/wrong_final_inlier_ids/unknown_final_inlier_ids` |
| 직접 가시 손상/숨은 오차 | `VISIBILITY_DAMAGE.json`의 category별 before/after와 damage |
| 비용 | `EXECUTION_LEDGER.json`, `SOURCE_EXECUTION_COUNTS.json`, 실제 `RUNTIME_ROWS.jsonl.gz` |

T는 기존 참조 centroid까지의 Euclidean 거리×100cm이고 R은 허용 proper symmetry에서의 geodesic 회전 오차 degree다. ADDsym은 허용 대칭에서 대응8코너의 평균 거리 최솟값이다. 원행 `pose.ADDsym_m`은 meter이므로 집계 ADDsym_cm의 평균/SD는100배, 분산은10,000배다. point coordinate 오차 px를 pose 위치 cm와 섞지 않는다.

평균·분산·SD·중앙값·P90은 저장된 **프레임 원행**에서 계산한다. SD는 n−1 표본 산포이며 CI/seed variation이 아니다. paired mean delta는 new−comparator로 음수가 개선이다. 13세션의 기존10,000 bootstrap draw(seed20260917)를 그대로 파일에서 읽었고 uint16 values hash `63e288…`를 저장했다. 오래된 draw 배열의 int64 byte hash와 다른 것은 dtype 표현 때문이다. CI는 반복 개발한 DEV319·단일 seed·다중비교 보정 없음이라는 한계를 갖는다. 공개 검산이 새 CI 생성을 원참조까지 인증하는 것은 아니다.

일반 T/R 평균은 pose가 있는 운영 row의 평균이며 무한 실패오차를 삽입한 평균이 아니다. 이번23방법은 fallback을 포함하면319/319 모두 pose가 있다. 일반화 시 complete failure는 수와 제외 ID를 따로 제시하고, 전체 ADDsym AUC는 실패를 infinity로 처리하면서 전체319분모를 유지한다. 신규 자세만 골라 실패를 숨기는 평균과 구분한다.

## 한 프레임을 끝까지 추적하기

예를 들어 `eval_noapril:1775201415399297536`에서 `N3_SUBPIX_GEOM_NOSELF_ROBUST`를 검토한다. 먼저 public verifier `--frame-id`로 numerical drilldown을 받는다. 원행을 직접 보고 싶으면 아래 표준 라이브러리 예제를 저장소 루트에서 실행한다.

```bash
python3 - <<'PY'
import gzip, json
from pathlib import Path
base = Path('_docs/experiments/pallet_observation_refiner_20261009_v1')
frame_id = 'eval_noapril:1775201415399297536'
method = 'N3_SUBPIX_GEOM_NOSELF_ROBUST'
for filename in ['PREDICTIONS.jsonl.gz', 'REAL_CORRESPONDENCE_ROWS.jsonl.gz']:
    with gzip.open(base / filename, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row['id'] == frame_id and row['method'] == method:
                print(filename, json.dumps(row, ensure_ascii=False, indent=2))
PY
```

`input_points → eligible → hidden_initial/excluded → used → generator_ids/fit_input_ids → final inliers → actual_pose → native_points[hidden] → pose` 순으로 읽는다. 마스크와 자세의 사건은 [REAL_CORRESPONDENCE_ROWS.jsonl.gz](REAL_CORRESPONDENCE_ROWS.jsonl.gz)의 `mask_pose_outcome`을 비교한다. 이 예는 wrong mask지만 T/R 둘 다 개선이고 참조상 accurate pool7/inlier7이다. 반대 유형의 예는 `eval_noapril:1775201443822140928`이다. 두 예는 accurate pool4개 이상인 해당 유형 중 ID 순서 첫 행이라는 사후 선택이다. 전체 효과의 증거는 전체319의 통계다.

‘accurate’는 봉인 후 고정 native phase에서 기존 참조까지8px 이하라는 oracle 진단 정의다. 직접 가시와 정확한 좌표는 서로 다르다. 참조가 없는 점은 unknown이며 숨은 일부 참조는 기존 PnP에서 유래했다. `remaining_human_direct_visible`는 used pool의 사람 가시 수로, GT8px 정확한 수나 최종 inlier 수가 아니다.

## 그림과 검토 checklist

[VISUAL_REVIEW_CASES.json](VISUAL_REVIEW_CASES.json)은 그림 입력 binding·frame/source ID·표시 좌표·선택 기준·파생 그림 해시를 기록한다. 그래프01–05는 저장된 raw/statistics의 파생 plot이다. 실사06과 source07은 기존 RGB 위에 저장 결과/감독을 표시한 검토용 montage이고, 원본 RGB 파일을 Git에 추가하지 않았다. 실사 reference overlay는 평가를 위해 봉인 뒤 그린 것이며 solver 입력이었다는 뜻이 아니다.

그림06의 여섯 사례는 mask/pose 유형별 위치오차 또는 위치차이 극값과 ID tie-break로 선택한 사후 진단이다. 원 영상의 고정 detector ROI crop을 표시하며 ROI 밖 최종 투영은 clip/marker로 구별한다. 본문의 ID-첫행 예시 두 개와 그림의 극값 선택은 별개다. 그림07의 실제 메쉬 silhouette는3개 source 프레임에 대해 표시 수리 전후 총6 auxiliary CPU ray pass를 수행했다. 이 작업은 검토 자료 작성량이며 원 실험의 감독/학습량에 합치지 않는다.

그림06의 주황 wireframe은 저장된 **최종 자세의8코너 투영**이지 `native_points` 전체를 뜻하지 않는다. 새 point 자세에서 H만 투영으로 바꾸고 nonhidden의 해당 관측 좌표는 투영으로 덮어쓰지 않는다. learned 관측에서 선택되지 않은 초기 좌표를 fit 입력으로 채우지도 않는다. cyan 초기 입력은 method에 따라 Base 또는 N3→SubPix이며, magenta 점선은 review-only 재구성 참조다. yellow 관측/line support와 green 최종 point consensus를 구별하며 point+line에는 point consensus가 적용되지 않는다.

| 그림06 panel | frame ID | method | pool 총/8px 정확 | 합의 총/8px 정확 |
|---|---|---|---:|---:|
| 1, wrong mask·T/R 개선 | `eval_pallet09:1778653832794714368` | `N3_SUBPIX_GEOM_NOSELF_ROBUST` | 7/2 | 5/2 |
| 2, correct mask·T/R 악화 | `plastic_night_01:038630` | `N3_SUBPIX_GEOM_NOSELF_ROBUST` | 4/4 | 4/4 |
| 3, 저정확 pool·큰 오차 | `eval_pallet09:1778653806958839552` | `N3_SUBPIX_GEOM_NOSELF_ROBUST` | 6/2 | 6/2 |
| 4, 선은 있으나 코너0·fallback | `eval_cad:1778653033056483584` | `IMAGE_ROLE` | 0/0 | 0/0 |
| 5, 같은 관측의 지역 정제 악화 | `eval_night09:1779449604823769344` | `IMAGE_ROLE_POINT_LINE` | 0/0 | 해당 없음; line rank6 |
| 6, 사람 가시 oracle 개선 | `wood_night_01:031671` | `N3_SUBPIX_ORACLE_VISIBLE_ROBUST` | 6/6 | 6/6 |

panel1은 참조8px보다 부정확한 inlier3개가 남아도 T/R가 감소한 사례다. panel2는 참조상 정확한4점과 맞는 mask로도 큰 오차를 낸 약한 합의 사례다. 정확한 입력 수만으로 성공을 보장하지 않는 이유다. panel4는 selected query14개/선2개가 있지만 관측 코너는0개다. panel5는 point inlier0을 성공 합의로 세지 않고 선만의 지역 정제로 표시한다.

| 그림07 panel | source ID·split | positive/no-match/ignore | actual mesh 대 delivered amodal IoU |
|---|---|---:|---:|
| 1, ID 순서 첫 eligible | `P0__shard_00_f0136`, source_test | 7/45/32 | 0.999927 |
| 2, positive query 최다 | `P0__shard_04_f0743`, source_test | 36/20/28 | 1.000000 |
| 3, no-match query 최다 | `P0__shard_06_f0072`, source_test | 1/55/28 | 0.999981 |

source 그림의 stripe는 기존 RGB의 display sampling이다. target은 retained cache의 `lo/hi/weight/valid`를 복사했다. 새 정답을 선택하거나 head를 재실행하지 않았다. source-test 극값을 설명용으로 고르는 것은 학습 checkpoint/설정을 선택하는 것과 구별되며 평균 source 성능은 전체128 source-test로 계산했다.

실제 `source_audit.EDGES` 순서에서 물리 horizontal edge ID는 `[0,2,4,6,8,9,10,11]`, unsupported vertical은 `[1,3,5,7]`이다. 그림07의 green/red edge는 이 순서를 사용한다. 코너ID나 임의의 상자 edge 순서를 대신 적용하면 비물리 경계를 positive로 오해하게 된다.

1. 그림01의 평균과 산출률을 `METRICS`에서 읽고, 그림02의 비교 이름·운영/신규 scope·분모·CI 부호를 `PAIRED_COMPARISONS`에서 확인한다.
2. 그림03의 analytic 오류 주입과 실제 사람 mask 오염을 구별한다. 공선 근처의 큰 오차도 삭제되지 않았는지 원행을 본다.
3. 그림04의 source-test 분류/후보 점수와 real 교점 수·새 자세 수0/3/0를 함께 본다. 그림05의 ROLE17.61ms는 fallback 경로라는 점을 확인한다.
4. 그림06의 case ID를 원행과 join한다. excluded/fit ID 교집합0, hidden 투영, final inlier·참조 accurate count, T/R delta, fallback 표시를 확인한다. 색상상 좋은 모습만으로 성능을 판정하지 않는다.
5. 그림07의 실제 source mask와 projected virtual edge를 구별한다. 비물리 vertical bounding edge가 ignore였는지 source target 정의를 읽는다.
6. `CHECKS.json`의 PASS는 무결성 PASS다. item12의 `subparts.four_controlled_variants`는 **NOT_EXECUTED/E6**다. 네 통제 변형까지 수행했다는 뜻이 아니다.
7. [EXECUTION_LEDGER.json](EXECUTION_LEDGER.json)의 실제9000 formal update, failed/retried inference, timing inclusion을 검토한다. source-preparation 내부 OpenCV generic/LM 호출 수와 전체 finite-difference residual 호출 수의 NA를0으로 더하지 않는다.
8. [PUBLICATION_PRECHECK.json](PUBLICATION_PRECHECK.json)과 [PUBLICATION.json](PUBLICATION.json)은 앞서 수행한 원 실험 게시의 commit/remote 스냅샷이다. 이번 상세 문서·그림 확장은 별도의 [REVIEW_MANIFEST.json](REVIEW_MANIFEST.json)이 현재 bundle의 파일 binding을 기록한다. 변경되지 않은 수치 원행과 새 문서/renderer를 구별한다. main 자동 병합이나 force push는 수행하지 않았다.

## 끝까지 남은 한계

고정 대조1,276행 중586행에는 과거 actual_pose R/t가 저장되지 않았다. 그 행의 원래 metric과 좌표를 그대로 보존하고 source binding/원행 parity를 확인했으며 R/t를 새로 만든 것처럼 채우지 않았다. 저장 metric 재집계와 해당 프레임의 독립 pose metric 재계산 가능 범위를 분리해야 한다.

네 통제 변형(원본/외부가림/방해물/저대비)은 선택한 기존 유효 source에 없었으므로 E6는 미실행이다. 없는 감독이나 변형을 실행한 것으로 보고하지 않았다. 유효 기존 RGB에서 정확한 감독을 만들 수 있으면 새 RGB를 만들지 않는 사용자 우선순위를 적용했다.

source-preparation의 logical feature pose1,024회는 측정했지만 내부 generic/LM primitive별 카운터는 저장되지 않아 NA다. point+line의 nfev468도 전체 residual 평가 수가 아니다. 단계 wall time은 일부 겹치므로 합산해 전체 경과시간이라고 부르지 않는다.

실사 reference의 독립 물리 정확성, source-to-real 관측 유효성, 반복 개발 DEV319, 단일 seed, 약한4점 합의와 일관된 오대응의 실패는 그대로 남아 있다. 이번 문서와 그림은 그 제한을 보여주는 자료이며 추가 촬영·수동 주석·재학습·추가 PnP 최적화를 수행한 결과가 아니다.

이번 보고서 보강의 실제 실행량과 원 실험 보존 여부는 [REVIEW_BUILD.json](REVIEW_BUILD.json)에, 공개 파일만 복사한 격리 검사와 의도적 오류 검출 결과는 [REVIEW_VALIDATION_TESTS.json](REVIEW_VALIDATION_TESTS.json)에 기록했다. [REVIEW_MANIFEST.json](REVIEW_MANIFEST.json)은 현재 코드·원행·문서·그림의 SHA256 목록이며 자체 파일과 재생성 가능한 REVIEW_CHECKS만 순환을 피하려고 제외한다.
