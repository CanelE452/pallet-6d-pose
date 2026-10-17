# P1/P6 합성자료 계약과 새 head의 자료 분리

신규 renderer·YOLO·N3 forward 없이 기존 Blender 자료와 정확히 저장된 upstream 출력을 재사용했다. 새 작업은 SubPix·기존 F·기하 조건부 상태 패킷과 기존 고정 MobileSAM의 자동 마스크 생산이다. renderer GT 마스크는 특징이나 프롬프트에 사용하지 않는다.

## 실제 원자료

- `data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json`:60,000합성 영상의 원 RGB/label SHA, reflection100px, 원/준비 해상도, scenario/source/partition.
- `.../cache/{points,point_valid,input_shape,gain,record_index}.npy`:실제 고정 YOLO의 입력 좌표 캐시. 점은 letterbox 좌표이며 N3의 저장 결과와 혼용하지 않는다. 거대한 P3/P4 특징 배열을 새로 복제하거나 전체 forward하지 않았다.
- `data/pallet/results/pallet_dim_conditioned_p_v1/source_baseline/{row:05d}.json`:원 R0의 전체 후보·원 선택. **이미 reflection100px가 제거된 raw 좌표**다.
- `.../predictions/SYNTH_HELDOUT/N3_DIM_SYM_seed1.json`:같은 원 RGB·선택 후보의 고정 N3 출력. **이 파일도 raw 좌표**다. 두 번100px를 빼지 않는다.
- `challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz`:원 renderer K·고정 XYZ 크기·R/t·camera_dynamic native8채널의 sourceXYZ 좌표. `Xcf`와 GT R/t는 감독 파일에만 남는다.
- `data/pallet/training_data/paper_release/v2_prod40k_clean_merged/{rgb,labels}`:실제 기존 원 RGB와 renderer 주석. `records.jsonl`은 Eevee engine, scene/cargo/context/occlusion 절차를 기록한다. 기존 Blender 생성기 `scripts/data_prep/blender/`도 확인했지만 추가 렌더는 필요하지 않았다.

원자료의 bbox-cuboid 좌표는 renderer 등록 모델의 외접 직육면체 특징이다. 8점 모두 실제 재료 표면의 육안 교점으로 독립 확인됐다고 주장하지 않는다. 보조 의미는 `virtual_cuboid_feature`이며 sourceXYZ owner와 native 순서의 기하 연결을 정확히 검산한다.

## 사전 분할

새 후보 성공 라벨이나 오차를 계산하기 전에, 기존 G38 heldout1,025장만 대상으로 source asset을 분리했다. 기존 heldout은 이번 head의 **개발 자료**로 재사용되므로 새 독립 test가 아니다.

- 검증 asset:사전 고정 사전식 마지막 이름 `woodpallet_block_jtoastie_ccby.glb` 전체.
- 학습 assets:`eur_pallet_bk_cc0.glb`, `scene.usd`, `scene_1.usd`.
- 각 split 내 ID의 `SHA256('P6_ASSET_SPLIT_V1:'+id)` 순서.
- 파일럿512장:train384+validation128.
- 확장1,003장:train768+validation235. 파일럿은 확장의 부분집합이다.

처음 제안한1,024장의 validation256은 실제 해당 asset235장보다 많았다. **결과를 보기 전에 자료 가용성으로1,003장으로 고정**했고1,024실험은 실행하지 않았다. 상한4,096장은 의무 표본 수가 아니다. 같은 asset의 다른 camera/light/연속 장면이 학습과 검증으로 갈라지는 일을 asset 전체 보류로 막았다. 반면 고정 backbone/N3가 과거 이 assets를 이미 보았다는 학습 계보는 남는다.

원 split·ID·source asset은 provisioning/분할 장부의 메타데이터다. 학습 특징에는 ID, source asset, material, session, GT permutation, 원래 heldout 이름, 성공 분류를 공급하지 않는다.

## 실제 검산

1,003장 모두에서 원 prepared RGB의 가운데를100px씩 제거한 픽셀과 원 renderer RGB가 완전히 같았다. 실제 K는 side table의 cx/cy에서100을 제거하여 renderer intrinsics와 정확히 같았다. native 채널과 source R/t의 투영은 renderer의 저장 cuboid와 최대0.0003728px 차이였다. 고정 sourceXYZ 크기도 기존 dimension sidecar와 정확히 같았다.

캐시 float32 점을 letterbox 역변환한 값과 원 R0 저장 double 좌표의 최대차이는4.9591e-5px였다. 이 값은 진단용이며 **추론에는 원 R0 저장 좌표를 그대로 사용**했다. N3도 기존 저장 결과를 사용했다. 중심8의 불변을 확인했다. renderer R의 미세 float 오차는 원값 그대로 보존하며 새로운 GT 회전을 맞춰 넣지 않는다.

첫 asset별 SHA순 영상4장에 원 RGB, RAW YOLO(파랑), N3+SubPix(주황), renderer cuboid 목표(녹색)를 겹친 그림을 실제로 확인했다. 화물 가림, 작은 물체, 어두운 장면, 서로 다른 배경이 포함됐다. 이 그림은 감독 계약 확인용이며 녹색 좌표는 후보 생성 입력이 아니다.

## 원 source별 대칭과 물리 phase의 한계

기존 source sidecar의 대칭은 asset별로 다르다. 확장1,003장 중 `scene.usd`243장과 `scene_1.usd`259장은 C2, `eur_pallet_bk_cc0.glb`266장과 validation의 `woodpallet_block_jtoastie_ccby.glb`235장은 C1이다. 즉 train은 C2 502장+C1 266장, validation은 C1 235장이다. 원 `symmetry_order`를 모든 target 행에 그대로 보존하며, 학습 라벨도 각 행의 원값으로 계산한다. 초기 문서의 “모든 synthetic이 C1”이라는 요약은 잘못됐으며 수치 target은 처음부터 이 혼합 계약을 유지했다. PRIMARY192는 기존 C2다.

camera_dynamic native 채널은 고정 renderer의 앞/뒤 signed phase를 소거할 수 있다. C1 source의 완전한 GT 2D 점을 원 F에 넣은 기술 예시에서도 고정 sourceXYZ에 대해180° 차이가 남는 경우가 있다. 이것을 관측 위치 오류라고 부르거나 synthetic GT를180° 뒤집어 좋은 후보를 만들지 않는다. 원 대칭 기준에서 bank에 좋은 후보가 없는 장면은 `no_reliable_candidate` 감독으로 보존한다. 전체 validation이 C1인 반면 train은 혼합 대칭이라는 차이도 실사 전이와 학습 가능한 정보의 제한이다.

## 입력·감독 분리

`UPSTREAM_INPUTS_NO_GT.jsonl.gz`와 이후 ROOT frames는 실제 q0/N3/SubPix, 등록 치수/K, 원 RGB, F의 pose, prediction-only 기하·관측만 포함한다. `SYNTH_GT_LABELS.jsonl.gz`에는 R/t, GT-native vertices/projection, 원 대칭 계약을 따로 저장한다. 후보 생성·자동 마스크 프로세스는 후자를 읽지 못한다. 후보와 특징을 봉인한 뒤에만 별도 학습 라벨 생산기가 GT를 읽는다.

Synthetic 상태는 원 F의 registered geometry에서 S와 B/I 역할을 만들고 q의 canvas 유효성을 사용한다. 외부 O classifier는 실행하지 않았다. 따라서 B/I는 보임을 독립 인증한 라벨이 아니며, 실제 외부 가림의 오분류도 이 automatic 입력에 남는다. 관측 covariance는 기존 fallback B2px/I3px와 동일 고정 CAL adapter를 사용하되 새 fit을 하지 않았다. T는 기존 parser의 broad outside-canvas union이며 유한 GT 위치/범위를 만들지 않는다. 모든 S 좌표는 영상 관측에 들어가지 않는다.

## 자동 마스크

기존 MobileSAM checkpoint SHA `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`, 기존 `GeometryPromptSAMCapture`와 component cleaner를 재사용한다. coarse projection 및 유효 non-S query의 union+기존19px padding으로 box만 만들고, GT pixel/좌표·renderer visibility·point prompt는 사용하지 않는다. 원본 raw/clean 마스크를 모두 보존한다. 좋은 마스크가 나왔다는 이유로 물리 팔레트 membership 또는 실제 가림을 인증하지 않는다.

Tiny/state classifier가 필요 없는 B의 target-mask 생산에만 사용하므로 Tiny forward는0이다. 원 provider의 사용하지 않는 state-mask carry-through 인자는0 placeholder로만 전달했으며, 원 cache가 있었다는 inherited metadata 문자열은 명시적으로 `NOT_RUN_ZERO_PLACEHOLDER_CARRY_ONLY_GEOMETRIC_SYNTHETIC_STATES`로 바로잡아 기록했다. 이 placeholder는 B histogram이나 head 특징으로 사용하지 않는다.

파일럿512장의 실제 MobileSAM 생산은28.82초 wall, CUDA event18.59초였다. parser의 누락 T-domain을 고친 뒤 같은512mask를 그대로 재사용했다. 실패한 schema adapter와 원 영수증을 보존했다. 이 생산비용은 기존 real mask cache 읽기 시간과 다르며 end-to-end 비용 표에 별도로 포함한다.

학습 입력의 추가 비대칭도 보존한다. synthetic 패킷은 기존 N3/SubPix fallback 점에서 출발하며 과거 real E0처럼 다단 RGB B/I 교점 회복과 archived partial-line 관측이 이미 들어 있는 자료가 아니다. 새 A/B/P5 계산은 공통 코드를 사용하지만 synthetic의 원 partial 수는0이다. head는 결측 validity bit를 받고 train에서 전혀 관측되지 않은 scalar를 임의의 real 통계로 교정하지 않는다. 이 지원 범위 차이는 real 전이 실패의 가능한 원인이며, 학습을 실행했다는 사실만으로 해소되지 않는다.
