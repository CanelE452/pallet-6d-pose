# 최신 사용자 목표와 인계

**초기 추론과 예상 기하 구조로 키포인트 중심 crop을 만들고, 그 위치/해당 키포인트에서 대상 팔레트가 보이는지 또는 가려졌는지 판별한다. 실사 키포인트별 가림 annotation을 평가에 활용한다. 현재 목표는 자세 개선이 아니다.**

학습은 합성으로만 한다. 실사 annotation은 평가 전용이다. 실제 이미지의 파생 특징·레이블로 fit/calibration/normalizer/threshold/checkpoint를 맞추지 않는다. 범용 frozen 사전학습은 허용한다. 기존 원고/원본/모델/부정 결과를 보존한다. 이 새 폴더는 파일 인계용이며 새 학습을 시작하지 않았다.

초기 추정 기하 구조·알려진 물체 치수와 K로 crop 크기를 정규화하는 방안도 논의했다. 단순 예측2D 크기와 기하 투영 크기의 비교는 처음 Pro 설계의 후보였다. v61은 예측2D span 기반 crop을 썼고, 그 두 crop을 새로 matched 학습 비교한 것은 아니다. GT 위치로 crop을 중앙 정렬하거나 실제 성능에 맞춰 crop 여유폭을 바꾸지 않는다.

## 역사적 OCPR/BI 배경 — 현재 목표와 분리

처음에는 이미 잘 맞는 pose를 보존하면서 큰 방향·W/D·대응 오류를 줄이는 연구였다. 원 OCPR은 N0→B/I 등의 관측과 G10/G13 별도 기하 producer→방문 후보은행→H1/LPCS 선택 경로다. G10과 G13을 항상 직렬 호출하는 연쇄로 오인하지 않는다. 당시 특정 PRIMARY192에서 실패72장 중31장은 성공 후보를 미선택했고41장은 bank에 성공 후보가 없었다는 사후 진단이 있었다. 이 수는 특정 버전/분모의 역사적 결과이며 현재 presence의 실패 수가 아니다.

B는 투영 외곽의 물체–배경 경계 관측, I는 내부 RGB gradient/색 전환 관측이다. **I를 intersection이라고 풀지 않는다.** 초기 pose에 따라 B/I/S·query gate가 달라져 오류가 순환할 수 있다. B/I 독립 기여는 같은 관측·solver·검색 예산의 COMMON 비교 없이 증명됐다고 하지 않는다. 과거 JOB30 COMMON/CAL blocker를 현재 해소했다고 하지 않는다.

v8은 최종 후보를 평균33.95→4.92(max5)로 줄이고 성공 수를 높였으나 T>50이 seed별7→8로 증가해 HOLD였다. upstream 관측/producer 의존성을 모두 제거한 방법이 아니다. 이후 v47 최소관측 단순화, v48 관측보존 시도, v49 PF+SubPix self-occ 제외는 별도 구성이고 원 OCPR과 혼합하지 않는다. self-occ만 제외한 PnP는 이미 해본 처방이며 현재 새 presence 아이디어로 재포장하지 않는다.

목재 직사각형과 정사각형은 유형별로 나눠 보고한다. 현재 PRIMARY 목재 등록 치수는 .8/.14/.59m(X/Y/Z), 정사각형은1.1/.15/1.1m이며 개별 실물 실측 완전 검증은 별개다. 사용자는 긴 직사각형 목재의 방향 오류도 보길 요청했지만, 현재 판별 목표를 pose 개선으로 다시 덮어쓰지 않는다. 형상 차이가 조명·재질·촬영 세션·크기와 함께 달라지므로 형상만의 인과 효과라고 하지 않는다.

## Pro 대화 순서와 접근 한계

주대화 제목은 **실험 감사와 실패 원인 진단**이며 다음 링크들은 부모가 전달한 실제 답변 위치다. 이 Ubuntu 작업은 모든 링크의 전체 원문을 이번에 직접 열어 읽은 것이 아니다. 아래 내용은 부모가 전달한 요약과 최소설계 원문에 근거한다. 새 대화가 링크를 읽지 못하면 전체 원문을 읽었다고 가정하지 않는다.

1. [가림·합성·frozen 모델 및 SAM2 비판](https://chatgpt.com/c/6ad599a5-f7c4-83ee-8f28-a9146700ff0a?messageId=a0218ef2-1a83-573d-88a0-539136852bae): 범용 frozen 모델을 쓰는 것과 대상 실사 특징/레이블로 학습하는 것은 다르다. 더 큰 SAM2 등의 도입부터 하지 말자는 검토였다.
2. [가시성·위치 loss와 B/I](https://chatgpt.com/c/6ad599a5-f7c4-83ee-8f28-a9146700ff0a?messageId=2248af1a-f428-5f44-8e1c-7e79fe28f833): 위치와 가시성 목표를 분리하고 공동 head가 실제로 필요한지 대조한다. B/I 독립 효과와는 별개의 질문이다.
3. [크기 정규화 crop](https://chatgpt.com/c/6ad599a5-f7c4-83ee-8f28-a9146700ff0a?messageId=cea1b746-b2b8-5f2d-a746-3c1dfa3529d3): 같은 초기 query 중심에서 예측2D 크기와 초기pose/K/치수 기반 크기를 비교한다. GT 중심 crop은 위치 정답을 제공하므로 학습 성능 증거와 구분한다.
4. [최소 실험 G0–G4 설계](https://chatgpt.com/c/6ad599a5-f7c4-83ee-8f28-a9146700ff0a?messageId=74037ab2-5b79-5adb-a63b-b7ab60e7f63f): 아래 단계 설계의 출처다.
5. [목재·정사각형 별도 평가](https://chatgpt.com/c/6ad599a5-f7c4-83ee-8f28-a9146700ff0a?messageId=c0d5dbe9-2a2e-572b-8396-b431f06eaa17): 두 유형을 같은 고정 설정·분모 규칙으로 비교하고 canonical90도와 symmetry-aware 오류·semantic 대응을 구분한다.

원 Pro 설계는 G0 출처/라벨 감사 → G1 좌표·crop·동일 후단·oracle → G2 위치전용2crop 학습 후 합성VAL로 하나 선택, 같은 crop의 가시성전용/joint로 첫 seed 최대4설정 → G3 설정동결 → G4 실사평가였다. physical DIRECT 위치 감독과 pose 효용을 다뤘던 설계이므로 최신 사용자 presence 목표에 그 전제를 그대로 강제하지 않는다. SAM2/DINO/PatchCore는 검토했을 뿐 새로 도입하지 않았다. 현재 완료한 v60/v61 두 head는 그4설정 완료가 아니다.

## v50–v61 실제 진행 목록

| 버전 | 실제 완료한 내용 | 남은 한계·구분 |
|---|---|---|
|v50|기존 동결 원 OCPR의 목재 재사용 및 새 정사각형 적용, 동일 평가·그림|historical real-CAL 의존 모델. 목재PRIMARY105 N0/PF/OCPR 성공57/63/65, 정사각형118 N0/OCPR30/41. 새 synthetic-only 방법 결과 아님|
|v51|PF 기준선의 엄격 GPU cache 재현 검사|약.001332px 차이로 원 exact-replay gate 실패. 사후 tolerance 완화하지 않음|
|v52|별도 CPU 실행 계약으로 PF+SubPix baseline118장|정사각형 PF-anchor37성공, N0/OCPR30/41. GPU exact 동일성 PASS 아님. 기존 모델 추론, 새 학습 없음|
|v53|실사 학습 의존 제거 설계 및 모듈별 출처 조사|read-only 설계. 실사 state head의 단순 frozen이 실사 학습 이력을 지우지 않음|
|v54|G0와 G1 좌표/crop·고정8장 및 원좌표 공통 후단1,003장 검사|처음 physical vertex의 약55% crop 밖 결과를 native query처럼 해석한 오류를 정정. 실제 native query miss는 TRAIN254/5693=4.46%, VAL60/1750=3.43%. 새 학습0|
|v55|합성 생성기·label provenance 추적|Windows 현재 소스/당시 snapshot 동일성 미확정. proxy를 DIRECT로 이름만 바꿀 수 없음|
|v56|고정8장 M0/M4 값·면적·subset·f_total 검산 및 RGB 대조|전체 실루엣은 3D point owner/physical DIRECT의 증명이 아님. 현재 pixel-presence에는 그 마스크 자체가 관련 정답임|
|v57|convex-cuboid self 및 external-ray aggregate를 source 발췌로 재현|physical DIRECT와 다른 oracle. mask를 NaN좌표로 전달한 API 문제를 다음 버전에서 수정|
|v58|finite9×2 좌표+별도valid mask, 2D/3D 동일 index 적용, min6/fallback 검증|1,003장 all-valid 원N0 회귀검사 PASS. 6점 미달을 억지로 통과시키지 않음. PnP 성능은 다른 목표|
|v59|실제 mesh fixture에서 native bbox32점의 표면support 진단 및 역사적 방향 그림|0/32 exact support. 다른 vertex로 점을 옮기지 않음. 이 한계를 presence 분류의 학습 차단으로 다시 사용하지 않음|
|v60|external-ray-only head1개·480update, 합성VAL, 실사223 frozenforward, 별도 pose 검사|target이 virtual-ray blockage. pose 부정 결과는 현재 presence 실패 근거가 아님|
|v61|M4 query-pixel presence head1개·480update, epoch9동결, 합성VAL AP.808/P.788/R.621, clear896 vs uncertain846, 실사223 frozenforward|실사 pixel mask reference 미확인. semantic visibility 주석은 발견했고 이제 mapping 후 별도 평가해야 함|

각 버전 폴더의 report/JSON/CSV를 PATH_INDEX에서 직접 찾을 수 있다. 이 문서를 위해 재실험하지 않았다. v60+v61에서 새로운 head fit은 합계2회/960update이며, v61 이후 추가 fit은0회다. 원본 실사 이미지로 새 모델을 학습하지 않았다. 기존 모델들의 real-CAL 학습 provenance는 별도로 보존한다.

## Windows 생성기 증거 — Ubuntu 경로와 혼동하지 말 것

부모의 Windows 원본 조사 루트는 `E:\CODING\GitHub\FoundationPose`다. 해당 경로를 Ubuntu 로컬 파일처럼 열 수 있다고 가정하지 않는다. 배포 ZIP은 `data/pallet/release/datasets/pallet6d_v2_clean_10k_part1~4.zip`, 원본 예는 `data/pallet/runs/diagnostics/v2_prod10k2_s8101_public`이다. 원본/ZIP 첫 label SHA256은 `552efe64d4bddc32809744132949e7503efe11bc4e4a72309fb3558ac79e8f1f`로 일치했다. 전체 대용량 ZIP SHA는 계산하지 않았다.

ZIP part1~4의 확인된 크기는 각각7,213,271,841 /7,218,357,212 /7,220,756,475 /7,229,164,566byte다. 각 part에 RGB/label/amodal/visible10,000개와 전체40,000행 records가 있다. records의 `_src_shard=v2_prod10k2_s8101_public`, `_src_id=0`가 원본 첫 label에 연결된다. 예시 label의 source_asset는 `eur_pallet_bk_cc0.glb`, dimensions_m=[1.433516,.168516,.975148], perm_v4=[1,5,6,2,0,4,7,3]이다. 이것을 Ubuntu의 다른 asset/프레임 치수로 그대로 이식하지 않는다. GLB 크기1,277,996byte, blend358,898,838byte이며 원본 run.log가 그 blend 경로를 기록한다. 원 실행 README는 `run_v2_scene_logic.py --completion-mode usable --n 2000`을 명시했다.

현재 소스의 주요 hash:

- `scripts/data_prep/blender/v2_realize.py`:119bf2372162b391b45b67bedc09b1d8f3fa4f880963dcb6161c9af5424052cf
- `run_v2_scene_logic.py`:ce955fe1f8f78d22f52fbb3107bd647d401c3a90144955deaec8351a6dc2e427
- `scene_placement_v2.py`:0c040f36f55b6be34db2d7c5528e2d31e1b8d905f7d630c4a8d71e029116bf38
- `synth_data_scene_portable_stage2c2.blend`:8cb4109adc6d3213920babda2a8e5845831aa359a2be8d017520661e8150907e
- `eur_pallet_bk_cc0.glb`:26c0ded54a55d50d3844d4a13e1faf0f6cf64835daa7717895e56147df7cafe8

현재 writer의 순서는 native8점→`perm_v4[dynamic_id]=native_index`→dynamic8점→centroid(index8)이다. 저장된 `cuboid/projected_cuboid/occlusion_fraction`은 이미 dynamic 순서이므로 perm을 이중 적용하지 않는다. `occlusion_fraction`은 중앙+주변8ray(약6px)의 blocked/valid이고, 첫 hit가 pallet 계층 밖이면서 query보다 앞이면 blocked다. 공중의 virtual cuboid corner까지의 시선도 측정하므로 물리 표면의 존재를 증명하지 않는다.

`M0=target-only binary silhouette`, `M4=final visible silhouette`, `f_total=1-area(M4)/area(M0)`이다. public profile은 M1–M3를 만들지 않으며, 확인한 writer에는 depth/instance ID 출력이 없었다. EEVEE는 holdout engine 증거이고 모든 RGB가 EEVEE라는 증거는 아니다. self 판정은 convex-cuboid 인접면의 front-facing 규칙이다. 현재 source와 생성 당시 snapshot의 byte 동일성, 전체 asset/import axis/scale binding은 미확정이며 이를 더 강한 물리 감독의 근거로 확대하지 않는다. Windows 조사에서 생성기를 이미 찾았으므로 사용자에게 같은 탐색을 다시 요청하지 않는다.

이 Windows 증거는 부모가 전달한 source 발췌·조사 보고에 근거한다. Ubuntu1003 cache 전체의 renderer/asset lineage를 새로 완전히 인증했다는 뜻은 아니다.

## 이번 작업을 잘못 확대하지 말 것

- 물리 mesh 코너 support를 얻거나 PnP 성공을 높이는 것을 현재 가림/presence 분류의 필수 조건으로 두지 않는다. 그 조건으로 오래 막혔던 과거 방향을 자동으로 이어가지 않는다.
- semantic keypoint visibility(그 의미의 코너가 가려졌는가)와 query-pixel presence(현재 예측 좌표의 픽셀이 대상인가)는 다르다. 초기점이 빗나가면 visible 코너 annotation이어도 query는 배경일 수 있고, 숨은 코너의 투영 위치에 다른 팔레트 면이 있을 수도 있다.
- 실제 사용자 주석이 없다고 단정하지 않는다. **사용자는 각 실사 키포인트 가림을 주석했다고 명시했고, 로컬에서 실제 사람 주석 sidecar를 찾았다.** 이전에 없음을 확인한 것은 pixel segmentation reference이며 semantic visibility annotation까지 없는 것은 아니다.
- semantic 주석으로 v61을 평가할 수 있지만 결과 이름을 semantic-visibility/crop-level 전이 평가라고 쓴다. 그 결과를 한 픽셀 presence 정확도라고 재명명하지 않는다. 초기 위치 오차·경계 모호성은 별도 집계한다.
- 새 모델에 유리하게 GT-best 순열·crop·threshold를 선택하지 않는다. 보존된 N3 baseline 전체-object mapping이 있으면 그것을 사용하고, 없는 경우 fixed mapping을 확인하기 전 해당 항목을 UNKNOWN으로 남긴다.

## 실제 완료한 두 head

### v60: external-ray blockage proxy

위치: `../external_ray_proxy_head_20261022_v60`.

native query8개(centroid 제외)의 저장 `occlusion_fraction >= .5`를 분류했다. 이 값은 가상 3D query까지9ray 중 외부 첫-hit 차단 비율이다. 물리 DIRECT나 M4 pixel presence가 아니다. 기존 frozen YOLO26 P3 9×9 ROI+valid channel, 새28,033-parameter head, TRAIN768/VAL235, seed20261018,480update/20epoch/fit1회. 합성 VAL BCE로 epoch15를 고정했다.

체크포인트: `BEST_HEAD.pt`, SHA256 `b7b6e548ee54c4b128abcb732def1e99fc9fe13e2a536d5b8e5e75a92e8cf68f`.

합성 VAL proxy AP.8690/precision.8454/recall.7675. 당시 다른 목표인 PnP 제외 실험도 시행했고 자세 gate는 실패했다. **이는 현재 presence 목표 실패가 아니다.** 추가 pose 평가를 현재 기준으로 요구하지 않는다. v60은 원본 보존.

### v61: M4 visible-target pixel presence

위치: `../target_presence_classifier_20261022_v61`.

초기 예측 query 좌표를 원영상 pixel center에 `floor(x+.5),floor(y+.5)`로 대응시켜 visible target mask M4>127을 정답으로 학습했다. GT를 query/crop 중앙으로 쓰지 않았다. 중심 한 픽셀이 label이고 3×3 이웃은 boundary uncertainty 속성만 만든다. broad patch vote가 아니다. 1,003장 binary mask의 값·면적·M4⊆M0를 검산했다. 유효 TRAIN5,673query(존재2,342)/VAL1,742query(존재701), 기존 split768/235. 기존 동결 P3 ROI cache, BN 동결, 새 head1개,480update/20epoch/seed20261018. VAL BCE 최저 epoch9, threshold.5로 고정했다. v60 head를 재명명하거나 불러오지 않았다.

체크포인트: `BEST_PRESENCE_HEAD.pt`, SHA256 `c75e04c41b8097ff63c2f134f76e12ec560603e7bcfe585446fe6042e31e4387`.

합성 VAL AP.80814/precision.78804/recall.62054, TP435/FP117/FN266/TN924. 명확한 내부·외부896query에서는 precision.91218/recall.87738, 경계·영상 끝846query에서는.56784/.33832였다. 전체 query의 거의 절반이 경계 층이며 좋은 subset만 보여주면 안 된다. 합성 feasibility screening은 통과했지만 독립 실사 성능 확보를 뜻하지 않는다. PIL center-pixel label7,415개와 sklearn AP/AUROC/confusion을 독립 검산했다. 새 PnP·pose평가는0회다.

v60/v61 공통으로 목재 PRIMARY105+정사각형118=실사223장에 frozenforward를 실행했다. v61의 .788/.621은 **합성 VAL** 수치다. 이 실사223장의 pixel presence GT를 새로 만든 것은 아니다. 기존 ROI와 query를 재사용했으며 실사 fitting은0회다.

## 기존 real-CAL 가림 모델: 정확한 버전 주의

실제 v50 original-OCPR은 `state_visibility/bounded_ownership_spatial_v3/MODEL.json`을 사용한다. 일반 `state_visibility/MODEL.json`과 다르다. descriptor SHA32f1683e... / arrays SHAa23a573b...가 v50 MODEL_BINDINGS와 일치한다.

target은 human native DIRECT_VISIBLE vs EXTERNAL_OCCLUDED, coordinate quality 제외. 실사 CAL46 nonS eligible DIRECT263/EXTERNAL7로 classifier·feature statistics를 적합했다. ImageNet ResNet18 frozen 특징과 raw/projected RGB·geometry·mask 단서를 사용한다. threshold.97/margin.01, visible p≥.98, O p<.96, 중간U이며 S는 RGB 전에 기하 gate로 억제한다. 단순히 p_VISIBLE>.5로 기존 결과를 재해석하면 안 된다. 정확한 기존 descriptor의 CAL LOGO AUC는.9397이며, 초기 다른 descriptor의.514와 섞지 않는다. 이 수치도 v61의 합성 AP와 직접 우열 비교할 수 없다.

v62 저장출력 감사에서223장×8=1,784개 공통 query를 연결했다. v60/v61은 같은좌표/같은P3 cache다. 이전 real-CAL RGB score가 같은 위치에 존재하는 valid query는 목재673개/정사각형796개였다. S/T/U 때문에 예측이 없는 점도 있다. 기존 head의 crop/다중 RGB guidance는 v61과 같지 않다. 따라서 세 모델이 동일 crop 입력이라는 주장은 하지 않는다. 같은 좌표에서 저장 결과를 나란히 보는 비교다.

v62의4개 실제 RGB crop을 직접 열었다: wood_183705:000000/k1, wood_night_01:030410/k6, SQUARE_YOLO:024276/k1, SQUARE_YOLO:027938/k5. 첫째와 셋째는 query가 물체 외곽에 걸쳐 단일 픽셀 존재와 의미상 visible 판단이 달라질 수 있어 보인다. 둘째는 판재와 홈이 섞이고 기존 모델은 기하S로 RGB score가 없다. 넷째는 cone과 팔레트/바닥 경계가 인접한다. 이는 보조 시각 관찰이며 사람 주석을 대신하는 새 GT·우열 판정이 아니다. 파일·좌표·동일crop·실제저장확률은 v62 FIGURE_MANIFEST에 있다. Figure의 세 열은 같은 RGB crop으로 보여준 것이며, 각 모델이 실제로 받은 입력 crop까지 동일하다는 뜻은 아니다.

## 발견한 실사 가시성 annotation

정확한 핵심 파일:

`/home/minjae/Documents/github/pallet-pose/_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json`

438frame, reference-corner3,101행. categories는 DIRECT_VISIBLE2313/EXTERNAL_OCCLUDED281/OUT_OF_FRAME45/SELF_OCCLUDED462다. source는 사람이 keyboard로 입력한 visibility sidecar이며 일부 geometry proposal을 사람이 확인했다. prior model exposure가 NOT_CONFIRMED이고, reference overlay exposure가 있으므로 완전히 독립 블라인드 주석이라고 부르지 않는다. 로컬 actor ID는 실명 확인이 아니며 과학 평가에 필요하지 않다.

현재223장의 ID+**image SHA256**으로 확인한 reference-corner 기록:

| 집단 | 주석이 연결된 frame | D | E | S | out-of-frame | 8slot 대비 기록 없는 slot |
|---|---:|---:|---:|---:|---:|---:|
| 목재105 |105/105|637|31|136|23|13/840|
| 정사각형118 |118/118|535|62|0|1|346/944|

이것은 reference slot inventory이며 아직 전부 native query에 안전하게 join한 최종 평가표가 아니다. source audit의 `missing_states=0`은 등록된 reference_corner_total3,101의 범위이며, 모든 frame의8slot이 다 채워졌다는 뜻은 아니다. 미기록 slot은 UNKNOWN으로 남겨야 한다. 사용자 annotation이 다른 sidecar에서 추가로 존재할 수도 있으므로 필요한 관련 경로만 더 확인한다.

기존 `.../pallet_visibility_curriculum_20261013_v1/SCORING_METADATA.json`은173frame(CAL/TEST)의 `human_states_native`, `native_to_reference_corner_ids`를 이미 고정했다. 현재223장 중 목재25장에 이 mapping이 있다. 목재 REMAINDER80 및 GREEN118의 mapping은 이번 인계 전에 최종 검증하지 않았다.

목재 REMAINDER80의 예전 N3_SUBPIX journal에는 `corner.branch`와 `qFinal/native_points`가 저장돼 있다. 동일 기존 permutation 계약을 복원할 후보 증거지만, 새 GT-best branch를 계산하여 고르면 안 된다. 기존 TARGETS.json에는 wood_day_01:002010 등 rect의 고정 permutations가 있다. GREEN0918의 직접 annotation은 camera_dynamic_0123_v4인데 saved native query와 semantic 대응을 전체적으로 검증하지 않았다.

정사각형 JSON의 `visibility:1`, `source:pnp_projected`, `reason:unknown`을 임의로 외부가림으로 바꾸지 않는다. raw JSON 옆의 explicit human sidecar가 그보다 구체적인 가림 상태를 기록한다. `(id, reference corner_id, image SHA)`를 먼저 묶고 실제 native mapping을 확인한다.

## 남은 최소 작업 — 학습 전에 저장 결과로 끝낼 것

1. 위 human visibility sidecar와 fixed native mapping을 결합하여 현재223장 전체 frame/slot inventory를 확정한다. unknown·mapping불명·query OOB·기존S gate로 RGB score없음 등을 각각 표시한다.
2. 사용자의 semantic 가림 판별 평가를 DIRECT vs EXTERNAL, DIRECT vs(EXTERNAL+SELF)로 별도 명칭·분모에 고정한다. SELF는 이전 모델에서 기하 gate인지 classifier score인지 구분한다. out-of-frame/unknown은 visible/occluded로 임의 합치지 않는다. CAL 학습 사용 영상/반복 개발자료/미사용 자료를 나누며 현재223장은 반복 개발자료다.
3. 공통 query에서 v61 presence score와 기존 native visibility prediction을 고정threshold로 평가한다. 서로 다른 학습 목표에서 이 semantic 평가로 전이시킨 비교라는 점을 공개한다. ray score도 포함하려면 방향(높음=rayblocked)과 별도 threshold.5를 그대로 표시한다. 1−rayblocked를 pixel presence 정답으로 재명명하지 않는다. AP는 정해진 semantic positive 방향의 ranking 측정일 뿐 목표 동일성 증명이 아니다.
4. reference 좌표와 초기 query 차이는 평가에만 이용하고, crop 중심/크기에 GT를 공급하지 않는다. 초기 오차 큰 집단, 가림 종류, 목재/정사각형, raw query와 GT점이 다른 표면을 가리키는 사례를 구분한다. 공통분모 paired 개선/악화와 실제 동일 crop을 본다.

**실제 human annotation을 활용한 새 semantic confusion matrix는 아직 계산하지 않았다.** 사용자 새 프로젝트 인계 요청으로 검색과 저장출력 비교를 안전한 지점에서 보존했다. 새학습0회/새forward0회/새pose0회다. 다음 대화는 이 준비 상태에서 시작하면 된다.

## 현재 Ubuntu Codex에서 직접 이어가기

현재 공식 tool inventory에는 로컬 프로젝트 등록/생성 도구가 없고 native CUA inventory는 apps=[]다. list_projects에는 이 새 로컬 폴더가 등록돼 있지 않았다. 따라서 폴더 파일 준비만 완료했으며 앱 프로젝트 ID는 없다. DB·숨겨진API·세션로그를 편집하거나 비공식 등록을 하지 않았다.

사용자는 현재 스크린샷의 **Ubuntu Codex GPT-6.1 Sol Ultra**와 직접 이어서 대화하려고 한다고 명시했다. ChatGPT Pro 선택·연결이나 모델 전환은 현재 요청 범위가 아니다. 그 지원 여부를 더 조사하지 않는다. 해당 화면의 모델명은 부모가 사용자 스크린샷에서 확인한 사실이며, 이 실행자가 모델을 바꾼 것은 아니다. 번개 아이콘만으로 Fast가 켜졌다고 단정하지 않는다.

사용자 지정 Pro 연구 대화: https://chatgpt.com/c/6ad599a5-f7c4-83ee-8f28-a9146700ff0a

위 Pro 링크는 과거 설계 참고자료다. 그 대화에 새 prompt를 보내거나 API 키를 만들지 않았다. 사용자는 현재 Ubuntu Codex에서 이 폴더를 열어 START_HERE_KO.md를 읽도록 하면 된다. 앱 등록·모델 전환은 이 작업에서 완료한 것으로 표시하지 않는다.
