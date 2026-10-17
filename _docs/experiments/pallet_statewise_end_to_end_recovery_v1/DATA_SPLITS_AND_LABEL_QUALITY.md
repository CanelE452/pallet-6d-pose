# 자료 분할과 감독 품질

새 학습은 기존 Blender RGB의 실제 고정 YOLO/N3 출력과 원 renderer 감독을 사용한다. PRIMARY192·CAL46의 정답은 새 head의 학습 표적이나 특징에 들어가지 않는다. 새 후보와 특징을 먼저 봉인하고 합성 감독을 별도 프로세스에서 결합한다. 아래 자료 계약은 학습 성공 여부와 관계없이 유지한다. 세부 upstream·마스크 절차는 [합성자료 계약](SYNTHETIC_DATA_QUALITY.md), 유한 후보 학습 정의는 [방법 L](METHOD_L.md)에 있다.

## 집합의 역할

| 집합 | 수량 | 이번 역할 | 독립성의 한계 |
|---|---:|---|---|
| 기존 합성 G38 heldout 가용 집합 | 1,025 | 기존 캐시와 원 GT 계약을 확인한 모집단 | 이번 head 개발에 재사용하므로 새 독립 test가 아니다 |
| 합성 파일럿 | train384 + validation128 = 512 | 첫 처리량·후보·감독 계약 확인 | 확장 집합의 부분집합이며 별도 시험 집합이 아니다 |
| 합성 확장 | train768 + validation235 = 1,003 | 고정된 최대 개발 범위 | 같은 assets를 고정 backbone/N3가 과거 보았던 계보가 남는다 |
| 기존 CAL46 | 46 | 허용된 교정·개발 진단, 기존 고정 covariance adapter | 새 head 학습자료가 아니며 독립 평가도 아니다 |
| 고정 PRIMARY | 192 | 모든 실제 완료 arm의 짝지은 비교 | 과거 분석에 반복 사용된 집합이며 새 독립 세션 시험이 아니다 |
| 설명용 추가 영상 | 본 평가와 별도 | 프로세스·실패 설명 | PRIMARY 분모나 학습 표본으로 합치지 않는다 |
| 새 독립 실사 시험 집합 | 0 | 이번 실험에서 새로 확보하지 않음 | 독립 일반화 성능을 주장하지 않는다 |

원 G38 heldout만 사용하고 source asset 전체를 분리했다. validation asset은 사전 고정한 `woodpallet_block_jtoastie_ccby.glb`이고, 나머지 세 asset이 train이다. 각 split 안의 순서는 `SHA256('P6_ASSET_SPLIT_V1:'+id)`로 고정했다. 성능·오차·후보 성공 여부로 영상을 고르거나 제외하지 않는다. 처음 제안한1,024장 중 validation256장은 실제235장보다 많아, 결과 전에 가용성만으로1,003장으로 고정했다. 제안한1,024장은 실행하지 않았고4,096장은 의무 표본 수가 아니다.

`DATA_SPLIT_LOCK.json` SHA256은 `bd77dae8cd52a30e5a36ade4fbcb2a393421e574f00fd1a7967ba98a0d76ff00`이다. 파일럿·확장 ID와 원 partition/source asset은 이 파일에 남긴다. ID·asset·material·session·원 실패 분류는 학습 특징에 전달하지 않는다.

## 원 대칭을 보존한 분포

| 원 asset | 원 대칭 | 파일럿 train | 파일럿 validation | 확장 train | 확장 validation |
|---|---|---:|---:|---:|---:|
| `scene.usd` | C2 | 123 | 0 | 243 | 0 |
| `scene_1.usd` | C2 | 130 | 0 | 259 | 0 |
| `eur_pallet_bk_cc0.glb` | C1 | 131 | 0 | 266 | 0 |
| `woodpallet_block_jtoastie_ccby.glb` | C1 | 0 | 128 | 0 | 235 |
| 합계 | 원 sidecar의 행별 값 | 384 | 128 | 768 | 235 |

확장 전체는 C2 502장과 C1 501장이다. train은 C2 502장+C1 266장, validation은 C1 235장이다. 원 C1에는 부여되지 않은180° 대칭을 새로 추가하지 않는다. PRIMARY192의 기존 C2 계약도 유지한다. 따라서 validation과 PRIMARY의 signed-phase 요구가 같다고 가정할 수 없다.

초기 설명에 모든 합성을 C1이라고 적은 부분은 잘못된 요약이었다. 수치 GT 파일은 처음부터 기존 sidecar의 각 `symmetry_order`를 복사했고, 학습 표적 함수도 각 행의 값을 읽는다. GT·대칭·오차 함수는 바꾸지 않고 설명 문자열만 교정했다. 교정 전 head 소스와 정확한 한 줄 diff를 보존했다. `CONFIG_FROZEN.json`과 PLAN의 원 바이트는 유지하고 별도 prose-correction 영수증으로 연결한다.

## RGB·카메라·GT의 실제 바이트 계약

원 출처는 기존 `pallet_line_pose_v1/SOURCE_MANIFEST.json`, `pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.json`, `GEOMETRY_SIDETABLE.npz`, G38 원 renderer의 RGB/label이다. 새 렌더링이나 새 정답 작성은0회다.

1,003장 모두에서 prepared RGB의 중앙 crop이 원 RGB와 픽셀 단위로 같았다. side table 카메라의 cx/cy에서 기존 reflection100px를 제거하면 원 renderer intrinsics와 정확히 같다. 같은 원 K, 원 R/t와 원 native XYZ의 투영은 renderer의 저장 cuboid와 최대0.00037280025px 차이였다. 이 차이는 원 geometry/table 표현의 검산값이며 GT를 이동시킨 양이 아니다.

GT R/t는 `synthetic_data.provision`에서 기존 table을 읽고 원 renderer `pose_transform`과 `np.array_equal`을1,003장 모두 확인한 뒤 그대로 직렬화했다. **R에 SVD·nearest-proper 보정을 적용하지 않았다.** table dtype은 float64지만 원 Blender 행렬의 float32 수준 수치 오차가 남아 있다. 최대 `|RᵀR−I|`는2.3841856378e-7이고 원값 그대로 유지한다. 적용한 GT 회전 보정각은0°, R/t 복사에 의한 투영 변화도0이다. 정확한 직교행렬을 가정하는 trace-angle 식은 이런 원 행렬의 자기 비교에서도 미세한 각도를 만들 수 있다. 이를 GT phase 변경이나 실제 자세 차이로 해석하지 않는다.

감독 파일은 원 R/t, native GT XYZ, 원 annotation projected corners, 원 대칭, annotation SHA와 table 행을 보존한다. `SYNTH_GT_LABELS.jsonl.gz`의 원 provisioning SHA256은 `578a24d67db0959a0b014ece9802279c803ee05c9845c2fbe38e8e8c389bdeb6`이며 설명 교정 뒤에도 같다. `PROVISION_SEAL.json`은 원 label/RGB/상위 캐시와 소스 파일 SHA, 전수 검산 결과, 실패와 비용을 묶는다.

이 GT는 renderer 등록 cuboid의 기하 감독이다. 모든 코너가 실제 재료의 육안 교점이라는 독립 물리 owner 정답을 뜻하지 않는다. `virtual_cuboid_feature`와 실제 팔레트 물질 경계를 혼동하지 않는다. camera_dynamic native 순서가 signed phase를 소거하는 C1 사례는 GT를 뒤집어 고치지 않는다.

카메라 K와 고정 objectXYZ 치수는 외부에서 알려진 PnP 입력으로 공급하는 조건이다. 원 dimension sidecar는 offline renderer index metadata로 camera-facing W/D를 고정 모델 축으로 복원해 전수 검산했지만, 그 permutation·GT pose·GT 채널좌표를 추론에 전달하지 않는다. 미지 물체의 실제 크기를 영상에서 새로 알아낸 실험은 아니다. `synthetic_data.py`의 inference whitelist는 원 RGB/K/해상도·알려진 치수·실제 저장 q0/qN/support만 내보낸다. 이후 nativeXYZ는 예측 F의 `cf_extents`로 새 cuboid를 만들며 GT `Xcf`를 사용하지 않는다. SAM prompt 역시 그 예측 cuboid 투영과 non-S 예측 점에서 만들고 renderer projected_GT를 읽지 않는다. 이 소스 자료흐름 감사의 기록은 `SYNTHETIC_REFERENCE_FIREWALL_SOURCE_AUDIT.json`이다.

## 실제 upstream 오차와 자동 관측

추론에는 원 R0 저장 double 좌표와 고정 N3의 저장 좌표를 쓴다. 두 파일은 이미 raw 좌표이므로 reflection100px를 다시 빼지 않는다. letterbox float32 캐시를 역변환한 값과 R0 저장값의 최대차이4.9591e-5px는 진단에만 사용한다. 실제 RAW 코너8,024개의 원 투영 대비 오차는 중앙값2.11245px, P90 8.41605px다. 합성 GT에 jitter만 넣어 backbone 오차를 대신하지 않았다.

새 YOLO/N3 forward는0회다. 기존 SubPix와 F를 실제 실행하고, 같은 registered geometry helper로 S 및 B/I의 기하 역할을 계산했다. synthetic에서는 외부 O classifier를 실행하지 않았다. B/I는 실제 보임의 인증이 아니며 실제 화물 가림의 오분류도 입력에 남는다. S는 영상 관측으로 쓰지 않는다. 결측은 U, 화면 밖 예측 query는 T이며 기존 parser의 broad outside-canvas domain을 사용한다. GT에서 얻은 유한 T 위치나 영역은 넣지 않았다.

기존 fallback B2px/I3px와 고정 CAL adapter를 적용했고 새 covariance fit은 하지 않았다. 과거 real E0의 다단 B/I 회복·archived partial 관측과 달리 synthetic 원 partial 수는0이다. 두 자료의 공통 solver·특징 코드는 같지만 관측 지원 범위는 같지 않다. validity bit와 train-only normalization이 이 차이를 표시하며, 그것만으로 전이 문제가 해결됐다고 주장하지 않는다.

### 합성 초기 pose·bank와 실사 carried bank의 차이

합성 N0의 실제 경로는 **저장 RAW YOLO → 저장 N3 → 기존 SubPix → RAW YOLO 기준 총 이동1% 원 영상 대각선 cap → 기존 F**다. F의 고정 source-basis 변환을 포함한 이 pose가 `initial_pose`, coarse geometry, fallback N0로 함께 쓰인다. 합성에 기존 전체 e를 별도로 실행한 E0 출력은 없다. `original_e_stage1_family_seed_bank`라는 이름은 공통 API에 맞추어 같은 F pose에서 만든 두 W/D basis를 담는 어댑터다. 이를 합성 E0 초기화라고 지칭하더라도 독립적인 e 결과를 뜻하지 않으며 기준 physical pose는 N0의 같은 F pose다. 두 family는 처음에 같은 물리 pose·투영을 서로 다른 native basis로 표현한다.

| 지원 항목 | 합성 입력 | 실사 입력 |
|---|---|---|
| 원 초기 후보 | N0 F pose와 같은 예측 점으로 실제 푼 두 W/D F hypothesis; 가용성 검사·공통 중복 제거 전 최대3개 | 원 G10/G13의 저장 feasible samples, selected/published poses, 가용 outcomes·logged endpoints, 별도 N0/E0를 보존하여 연결 |
| coarse와 E0 | F에서 만든 coarse; 독립 E0 추론·과거 G10/G13 bank 없음 | 원 family/coarse와 실제 과거 E0를 그대로 보존 |
| 외부 가림 O | 외부 O classifier를 실행하지 않아 O 상태를 생성하지 않음 | 과거 prediction-only 상태·O/T domain을 원 typed packet에서 보존; 물리 GT로 인증한 것은 아님 |
| 완전 B/I 점의 covariance | 모든 source가 `VISIBLE_BACKBONE_FALLBACK`; B는4I, I는9I px²에서 시작해 고정 CAL fallback group의 scale/floor/reliability 적용 | 과거 실제 보정·fallback별 원 covariance와 calibrated metadata를 보존; 합성처럼 모든 점을 동일 fallback recipe로 다시 만들지 않음 |
| 원 partial·영역 관측 | partial B/I와 U region은0; T는 예측 query에서 얻은 broad outside-canvas domain | 보존된 typed partial B/I·U region·O/T domain을 해당 원 지원 범위에서 사용 |
| 원 outline 및 새 B 마스크 | 원 legacy boundary는 빈 목록, 원 outline은 unsupported; 이후 고정 MobileSAM의 새 B support를 추가 | 기존 archived outline/관측과 기존 자동 마스크를 보존한 입력에서 공통 새 B 모듈 실행 |

실사의 N0/E0는 수치적으로 같다고 가정하지 않는 별도 기존 출력이다. 원 coarse/prior를 E0로 바꾸지 않으며, N0/E0는 추가 실제 후보로 보존한다. 실사에서도 R/t가 없는 종점을 새로 복원하지 않았으며, 비정상 회전·비양수 model depth의 pose는 원장에 남기고 시작점 admission에서 제외한다. 위 “원 후보”는 저장돼 실제 사용 가능한 범위이지, 가능한 모든 연속 자세나 미기록 root를 뜻하지 않는다. 양쪽의 P5 solver, 새 시작점 예산과 특징 계산 코드는 같지만 출발 bank의 다양성과 원 관측 패키지가 다르다.

합성에서 O가0이라는 것은 화물 가림이 없는 데이터라는 뜻이 아니다. 실제로 가린 코너도 기하 역할상 B/I로 남을 수 있고, 그 예측 점이 fallback 관측으로 사용되는 오류가 학습 입력에 남는다. 새 SAM 마스크가 이 상태들을 O로 다시 분류하지도 않는다. 또한 같은 고정 CAL adapter 숫자를 적용했다는 사실은 합성 fallback covariance의 정밀도·물리 owner가 교정됐다는 증거가 아니다. 따라서 이 합성 head의 validation은 **더 작은 초기 bank·외부 O 미지원·fallback 중심 불확실성**에서의 개발 결과이며, 과거 G10/G13 bank와 보정/partial 관측을 가진 실사 패키지의 동등한 복제 실험으로 해석하지 않는다.

소스 연결: `synthetic_data.py:207–227`의 N3/SubPix/F·기하 상태·covariance, `:257–272`의 F 기반 family/초기 bank/fallback, `inputs.py:72–140`의 실사 원 bank/관측 연결, `solver.py:97–121`의 동일 context 준비와 `:355–406`의 공통 P5 생성이다. 이번 범위 설명 때문에 코드·후보·학습 설정을 바꾸지 않았다.

MobileSAM은 기존 고정 checkpoint·기하 예측 box prompt만 사용했다. renderer mask/가림 label/GT point prompt는 사용하지 않았다. 실제 신규 forward는1,003회, 합계 CUDA event35.76944초다. raw/clean mask, source/hash, 시간·메모리를 보존한다. 마스크는 자동 예측이며 참 팔레트 membership·물리 owner GT로 인증하지 않는다. Tiny/state classifier forward는0회이고 사용하지 않는 carry 인자는 명시한0 placeholder다.

## 학습 라벨·전처리·기록의 범위

공통 실제 후보/특징을 봉인한 뒤 별도 `prepare-labels`가 원 합성 감독을 읽는다. 같은 후보 pose가 위치≤5cm AND 원 대칭 기준 회전≤5°일 때만 공동성공이다. eligible 공동성공 후보가 있으면 그 집합에 균등 확률을 주고, 없으면 보류가 유일한 목표다. 좋은 후보가 없는 영상도 남기며 GT pose를 추론 bank에 추가하지 않는다. 중복 제거·eligibility·baseline alias 정책은 real과 synthetic에 같은 코드로 적용한다.

지원 gate는 사전 고정한 train32장 이상·validation16장 이상의 eligible 공동성공 영상이다. 일반화 충분성 인증이 아니며 결과를 보고 바꾸지 않는다. gate에 못 미치면 학습0회/N/A다. 통과 시 선형189parameter와2-hidden-layer head16,321parameter를 각각 고정3seed로 최대6회 학습한다. 대표 seed는20261018이다. 다른 두 seed도 모두 보고하며 PRIMARY 최고 seed를 고르지 않는다.

31개 scalar의 normalization은 train의 유효한 eligible 후보에서만 fit한다. validation·CAL·PRIMARY 통계로 다시 fit하지 않는다. ID·session·source asset·GT 오차·oracle rank·수동 경계는 모델 입력이 아니다. checkpoint의 모델 선택은 고정 validation loss와 patience5, 최대50epoch다. 원 renderer GT는 이 감독/평가 프로세스에만 존재한다.

실제 라벨 통계와 학습 실행 수의 권위 기록은 실행 뒤 생성한 `LABEL_SEAL.json`, `TRAINING_PROTOCOL_LOCK.json`, `TRAINING_MANIFEST.json`, 각 seed의 `HISTORY.json`과 checkpoint SHA다. 최종 공개 manifest는 입력 feature/target/split/normalizer SHA, 실제 train/validation 수, support gate와 후보부재율, 모든 실행 seed·epoch·optimizer step·비용·실패/미실행 이유를 포함한다. 사전 protocol 작성 당시의 “학습 전 코드를 준비한 사실을 실제6회 학습 완료로 기록하지 않는다”는 문장은 당시의 기록 원칙이며 현재 미실행 상태를 뜻하지 않는다. 이번 실행은 고정6회 학습을 실제 완료했으며 [TRAINING_MANIFEST.json](TRAINING_MANIFEST.json)과 [SYNTHETIC_LEARNED_EVALUATION_KO.md](SYNTHETIC_LEARNED_EVALUATION_KO.md)에 실제 비용·전체 seed·validation 공동성공·보류를 연결한다. 모든 seed 예측은 별도 reference-read guard 프로세스에서 먼저 봉인한 후 평가했다.

## 보존한 기술적 시도

ROOT adapter의 `protection_px` 누락과 T-domain 누락은 원 실패 시도를 보존하고 형식만 고쳤다. 점·상태·GT를 바꾸지 않았다. 같은 mask를 재사용했고 최종1,003장/2,006family 입력 compile이 실패0으로 통과했다.

추가 원 GT 감사의 첫 시도는 NPZ 배열을 행마다 재압축해제하는 감사 스크립트 오류로 약1분 후 중단했다. root의4개 solver worker가 실행 중일 때 잠깐5번째 보조 CPU를 쓴 자원 사건도 숨기지 않는다. 최적화한 중복 감사는 “모두 C1”이라는 과도한 추가 assertion 때문에 실패했다. 원 GT는 혼합 대칭이었고 처음부터 그대로 보존돼 있었다. 이는 RGB·원 pose·GT 변조 증거가 아니다. 이 중복 감사를 통과했다고 보고하지 않으며 원 전수 provisioning 검사를 권위 기록으로 사용한다. `SYNTH_GT_AUDIT_ATTEMPTS.json`에 중단·실패 원인과 측정하지 못한 비용을 명시했다. 원자료·후보·GT 파일은 이 감사들로 변경되지 않았다.
