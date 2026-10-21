> Public edition of an existing development audit. Identifiers and local paths are redacted. Publication adds no new experiment or improvement claim. Historical no-publication statements describe the original audit date; this edition is separately authorized.



# 실제 치수 입력 계약 확인

**점검한 사례에는 각 등록 물체에 맞는 치수가 실제로 입력돼 있다.** 단일 기본 치수를 모두에게 쓰는 경로가 아니다. 실사 CAL12와84° 목재1장에서는 N3의 실제 DEV_cache metadata, D1 upstream·E_RUN family 모델, 평가 참조를 연결했다. TRAIN8은 N3 sidecar array, upstream·family·synthetic GT 치수를 대조했다. 새 추론·학습·평가 실행은 없다.

| 점검 사례 | 실제 object_type | canonical W×D×H | 내부 XYZ(=W,H,D), m |
|---|---|---|---|
| CAL 목재 C02/04/06/09/12 및 E84 |wood_small_80x59x14|80×59×14cm|[0.8,0.14,0.59]|
| CAL 플라스틱 C01/03/05/07/08/10/11 |plastic_standard_110x130x11|110×130×11cm|[1.1,0.11,1.3]|

N3의 실제 캐시13개를 읽어 저장된 [W,D,H] 입력을 확인했고, DEV_CACHE_COMPLETE의 각 cache SHA256과 일치했다. 목재는[0.8,0.59,0.14], 플라스틱은[1.1,1.3,0.11]m였다. N3는 외부로 주어진 object_type을 registry에 연결해 얻은 canonical 치수의5개 log/ratio context를 사용한다. 사진에서 재질과 치수를 자동 추정한 결과를 입력한 것이 아니다. registry의 square plastic 기본값110×110×15cm는 이번13사례의 실제 입력이 아니다.

PnP와 D1 후단은 같은 치수의 cuboid를 사용한다. 실사 평가 참조의 xyz 치수와 모든13 upstream 값은 약4.44e-16m 이내로 같다. 이는 실수 반올림 차이이며 cm/m 오류가 아니다.

## 84° 목재의 W/D 순서

E84의 canonical 물리 입력은W0.8/H0.14/D0.59m다. N0가 고른 camera-facing extents는[0.59,0.14,0.8]m이고 ORIGINAL_WD는 그 표현을 유지한다. SWAPPED_WD는[0.8,0.14,0.59]m다. family 이름의 ORIGINAL은 등록 물체의 물리 W/D 순서와 동의어가 아니라 N0 표현 기준이다.

두 family는 치수뿐 아니라 native cuboid, R, physical_registration_Q와 대응을 함께 변환한다. 코드상 R_s=R R90^T, Q_s=R90 Q로 초기 물리 회전 R_s Q_s=R Q를 유지한다. 따라서 [0.59,0.14,0.8] 자체가 목재에 잘못된 치수를 준 증거는 아니다. 이후 다른 관측 대응과 자세 후보가 다른 물리 R/t로 발전할 수 있어84° 오류는 별도로 남는다.

## TRAIN8

합성은 실사 목재/플라스틱 두 고정값이 아니라, 각 frame의 등록 asset/scale metadata에서 얻은 canonical 치수를 사용한다. N3의 보존 DIMENSION_SIDECAR.npz와 JSON8개 값, D1 upstream, 저장 synthetic 참조 치수는 일치했다.

선정 순서의 W×D×H(cm):

1. 77.361×112.940×12.807 (scene.usd)
2. 114.237×120.712×15.658 (scene.usd)
3. 96.966×109.481×15.737 (scene.usd)
4. 105.937×127.102×15.306 (scene_1.usd)
5. 96.560×114.515×13.440 (scene.usd)
6. 89.241×111.567×15.876 (scene.usd)
7. 103.730×115.693×15.946 (scene_1.usd)
8. 79.480×119.081×14.884 (eur_pallet_bk_cc0.glb)

N3 synthetic input 준비 코드는 raw renderer의 camera-facing dimensions_m를 직접 쓰지 않고, 감사된 GEOMETRY_SIDETABLE의 canonical fixed dimensions를 XYZ→WDH로 바꾼다. 프레임 GT pose를 보고 W/D를 골라 넣는 경로와 구분한다.

## 값의 출처와 제한

실사는 dataset의 object_type→CHALLENGE_OBJECT_GEOMETRY_REGISTRY 물리 치수다. registry는 source_measurement_cm와 width_depth_height 순서를 명시하지만, 이번 감사는 개별 실물마다 줄자로 다시 측정한 증빙이나 촬영 물체와의 실측 대조까지 확인한 것은 아니다. 따라서 ‘등록 치수를 사례에 맞게 입력함’과 ‘각 실물 치수가 독립 실측으로 검증됨’을 구분한다. 알려진 object_type/치수 metadata가 외부에서 제공된다는 가정은 유지된다.

원 OCPR과 v5의 차이는 선택기 학습 목표의 derived C2 변경이며 치수 metadata 변경이 아니다. v8의 후보 축소나 v19–v37 후보/특징 실험도 보존 upstream 치수를 사용한다. v30/31의 ROI N3 재추론 코드는 known_physical_dimensions_xyz_m[[0,2,1]]을 N3에 전달한다. v38 이후 감사는 관측/pose/치수를 변경하지 않았다. H1 자체는 WDH를 새로 추정하는 모델이 아니라, 이 기하로 계산된 후보/잔차/문맥 특징으로 선택한다.

## 근거 파일

- REAL13_N3_INPUT_METADATA.json: 실제13개 N3 cache 입력·object_type·cache hash
- REAL13_ACTUAL_INPUTS.json: D1/PnP family 치수·Q·평가 치수
- TRAIN8_N3_INPUT_METADATA_PRIVATE.json / TRAIN8_ACTUAL_INPUTS_PRIVATE.json
- source: scripts/research/pallet_dim_conditioned_p_v1/dev_evaluate.py:18–29, inference.py:30–56, refiner.py:10–19
- source: scripts/research/pallet_boundary_action_repair_v1/pipeline_contract.py:62–85
- source: scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_two_wd_family_v4_9.py
- source: frozen_roi_reinference_20261021_v30/experiment.py:43

원 자료·원고·기존 결과는 수정하지 않았다. 마스크 의미 감사는 별도로 이어간다.
