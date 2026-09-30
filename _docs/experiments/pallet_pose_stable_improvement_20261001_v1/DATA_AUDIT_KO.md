# 안정적인 T/R 개선을 위한 데이터 감사

2026-10-01. 이번 감사는 실제 manifest, 이미지 SHA256, recording 별칭, 모델 감독 계보를 확인했다. 추론·학습·평가 결과에 따른 방법 선택은 하지 않았다. 기계 판독 결과와 입력 해시는 [DATA_AUDIT.json](DATA_AUDIT.json)에 있다.

**현재 자료로 할 수 있는 것은 recording을 분리한 재사용 DEV 검증이다. 처음 보는 독립 확인 TEST는 확인하지 못했다.** 반면 학습 recording 다양성을 늘릴 수 있는 기존 무라벨 1,000장과 원본 8,031장은 존재한다. FULL125의 직접 학습은 한 recording 253장에 집중되어 있다. 이 중 한 장은 PAPER194 평가 이미지와 정확히 중복되어, 새 비교의 공통 학습 예산을 252장으로 맞추고 해당 SHA를 제외해야 한다. 현재 주 평가인 full128에는 이 중복이 없다.

## 실제 평가 모집단

| 모집단 | 이미지 | merged recording 수 | 현재 사용 가능 범위 |
|---|---:|---:|---|
| full128 | 128 | 7 | 주 평가. FULL125 및 Replay 교사 fitting recording과 분리된 재사용 DEV |
| natural99 | 99 | 6 | Moderate21+Severe78. 주 T/R 공동 개선 판정 |
| clean29 | 29 | 3 | 보존 확인. natural99와 같은 전체128의 별도 부분집합 |
| 추가 plastic66 | 66 | 2 | REC_001 44장+REC_002 22장. 학습 또는 교사 노출 recording의 보조 진단 |
| PAPER194 | 194 | 9 | full128+plastic66. 이 전체를 FULL125의 recording 독립 평가로 부를 수 없음 |
| 기존 wood45 | 45 | 2 | REC_039 25장+REC_042 20장. 별도 형태 전이 stress, 재사용 DEV |
| 추가 wood80 | 80 | 2 | REC_001 24장+REC_002 56장. 교사 노출 recording |
| PAPER319 | 319 | 11 | plastic194+wood125. 재사용 DEV 전체 목록 |

실제 확인한 [PAPER_EVAL_ALL_POS.json](../../../challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json)은 `role=DEV`, `held_out_final=false`다. 원본 canonical140은 GT-QA 이후의 다른 계보이며, paper128은 여기서 FT 중복12장을 제외한 모집단이다. 194/319와 섞어 분모를 바꾸지 않는다.

full128의 recording은 REC_007 33장, REC_021 18장, REC_022 16장, REC_025 27장, REC_027 12장, REC_041 10장, REC_044 12장이다. natural99는 각각 33, 2, 16, 27, 12, 9장으로 마지막 REC_044는 포함하지 않는다. clean29는 REC_021 16장, REC_041 1장, REC_044 12장이다.

`SOURCE_RECORDING_GROUPS.json`의 폴더 별칭만 읽지 않고 기존 [SPLIT_LOCK.json](../pallet_existing_data_transfer_v1/SPLIT_LOCK.json)의 partial-overlap 병합도 적용했다. 예를 들어 `eval_outside`의 최종 그룹은 REC_041이다. 파일명 또는 폴더명 차이를 독립 촬영으로 세지 않았다.

## 새 학습에서 제외할 정확한 중복

| 항목 | 값 |
|---|---|
| 기존 FULL125 학습 ID | `DAY264__005516` |
| PAPER194 평가 ID | `plastic_day_01:005516` |
| SHA256 | `862d175bd2ce92045aa62edbe34c76f20e5d28c8f9729ee481a9e4d0930ef1cd` |
| 기존 학습 원본 | `data/evaluation/pallet_eval_v1/incoming/sessions/real_unlabeled_day_20260830/rgb/005516.png` |

PAPER319의 이미지319개를 다시 해시했고 기존 FULL125 실제 학습 목록과 대조했다. 직접 이미지 중복은 위 1장, 직접 recording 노출은 plastic44+wood24=68장이다. 이 발견은 기존 full128/natural99의 이미지·recording 분리 결과를 뒤집지 않는다. 새 비교에서는 이 이미지를 두 arm 모두에서 제외하고, **SINGLE252 대 DIVERSE252**처럼 고유 영상 수를 동일하게 한다. 기존 FULL125는 과거 고정 모델로만 표시한다.

추가 plastic66은 전체가 Replay 교사 학습 recording REC_001/REC_002에 속한다. 중복 한 장을 제외하더라도 독립 확인 집합으로 승격되지 않는다. 예측을 먼저 고정하는 절차는 필요하지만 과거 노출 이력을 없애지는 않는다.

## 바로 사용할 수 있는 기존 학습 자료

기존 [MAIN_UNLABELED_BALANCED.csv](../../../data/evaluation/pallet_eval_v1/adaptation/MAIN_UNLABELED_BALANCED.csv)는 daytime500+nighttime500의 고정 1,000장이다. 1,000장의 실제 이미지 SHA와 manifest 저장 SHA가 모두 일치했다. PAPER319와 이미지 SHA 교집합0, 병합 recording 교집합0이며 manifest 자체 해시도 [ADAPTATION_POOL_LOCK.json](../../../data/evaluation/pallet_eval_v1/adaptation/ADAPTATION_POOL_LOCK.json)과 일치했다.

| 세션 | merged recording | 고정 1,000장 내 수 | 원본 PNG 수 |
|---|---|---:|---:|
| capturepallet01 | REC_049 | 9 | 42 |
| capturepallet10 | REC_028 | 133 | 613 |
| capturepallet11 | REC_011 | 358 | 1,572 |
| capturenight01 | REC_019 | 110 | 1,254 |
| capturenight02 | REC_024 | 82 | 782 |
| capturenight03 | REC_020 | 108 | 1,219 |
| capturenight04 | REC_023 | 80 | 1,075 |
| capturenight10 | REC_017 | 120 | 1,474 |
| 합계 | 8 recording | 1,000 | 8,031 |

원본8,031장은 실제 파일 수를 확인했다. 이번 감사가 전체8,031장의 개별 해시까지 재검증한 것은 아니다. 고정1,000장 전체는 재해시했다. 학습 입력 선별·의사 타깃 생성을 진행할 때는 이 구분을 유지한다. 하루/야간 수가 500씩이라는 사실은 recording 균형을 뜻하지 않는다. 특히 REC_011이358장이고 REC_049가9장이므로 recording 균형이 실험 변인이라면 고정 max-min 배정 등 결과를 읽지 않는 규칙이 필요하다.

기존 ST clean78도 파일 해시78/78이 일치한다. 구성은 REC_020 29장, REC_017 27장, REC_028 16장, REC_023 6장이다. 외부 가림이 관찰되지 않았다는 RGB 검토이며 독립 물리 자세 정답이나 동일 자세의 자연 가림 전후 pair가 아니다. OLD_REF217의 부분집합이라는 과거 사용 이력도 남긴다.

미검토 incoming day29,028/night13,583장은 각각 REC_001/REC_002의 같은 연속 촬영 자료다. 이미지가 많아도 독립 recording 수를 늘리지 못한다. 평가로 이미 활성화된 이미지·세션과의 교집합 검증 없이 새 무라벨 pool로 합치지 않는다.

## 감독 계보: 새 GT 미사용과 실사 GT 전무는 다르다

| 모델/타깃 | 확인한 계보 |
|---|---|
| R0 G38 | 팔레트 추가 fitting은 합성 전용. 상위 COCO-pose 사람 라벨 사전학습 존재 |
| PRIOR1 | 합성 PoseFix prior. 모델 선택에 재사용 DEV 이력 존재 |
| Replay 교사 | 기존 실사9장·직접 확인 manual38코너와 합성 source로 학습 |
| FULL125 | PRIOR1부터 직접 real253 REC_001 학습. 고정 Replay+self-occlusion PnP 의사 타깃이므로 교사의 실사 감독을 간접 상속 |
| OLD_REF217 | 별도의 pose-only 학생. 7 recording의217장, full128과 분리 |
| REALFT_A | 실사157장+negative259장+합성12,000장. 초기 모델도 R0와 다른 실사 지도 대조 모델 |

Replay 체크포인트 `fc9b3d7b…`는 [E2 의사 타깃 프로토콜](../pallet_occlusion_refiner_transfer_v2/PSEUDO_PROTOCOL.json)에 묶여 있고, 실사38코너/9장 감독은 [Replay 프로토콜](../pallet_posefix_replay_v1/PROTOCOL.json)에 명시되어 있다. 9장은 REC_001의 wood6장과 REC_002의 plastic3장이다. 따라서 “새 평가 GT를 학습에 사용하지 않음” 또는 “이번 적응에서 새 수동 좌표 없음”은 정확하지만, FULL 계보를 “실사 팔레트 GT를 전혀 쓰지 않은 방법”이라고 쓰면 부정확하다.

이번 해결 경로가 기존 Replay 의사 타깃을 고정해 사용한다면 이를 명시한 약한 실사 감독 계보의 방법 개선이다. 엄격한 실사 팔레트 GT 전무를 요구한다면 R0/PRIOR1와 합성·무라벨만 사용하는 별도 계보가 필요하다. REALFT_A의 개선을 FULL 보정 또는 무라벨 방법의 개선으로 대체하지 않는다.

## wood45의 별도 전이 검증 계약

기존 [wood EVAL_METADATA 공개 사본](WOOD45_METADATA.json)는45장 전체의 ID, K, 치수, 해상도, recording, severity를 포함한다. 입력 해시와 전체 metadata는 `DATA_AUDIT.json`의 `wood45_cache_audit`에도 보존했다.

- 고정 구성: Clean38, Moderate7, Severe0. REC_039 25장/REC_042 20장. 각 모델·교사·새 무라벨1,000장과 recording 분리.
- 물리 치수: xyz=(0.8,0.14,0.59)m. native 해상도1280×720. K의 fx=908.8597333, fy=908.9547333, cx=636.3943333, cy=384.4384667.
- 인덱스: `camera_dynamic_0123_v4`. 대칭: 기존 paper C2 계약. 물리적 외형 대칭을 독립 검수했다는 뜻은 아님.
- 참조 한계:45장 모두 migration=`MANUAL_REVIEW_REQUIRED`, 360코너 source=`unknown`; intrinsics quality=`SENSOR_PROFILE_SCALED`. 기존 좌표에서 유도한 geometry 참조로 해석.
- R0 cache: `data/pallet/results/pallet_material_selftrain_closure_v1/PREDICTIONS.json`의 `R0`45장. 체크포인트 SHA=`970a0913…`, 현재 G38과 일치. [WOOD_PREDICTIONS_LOCK](../pallet_material_selftrain_closure_v1/WOOD_PREDICTIONS_LOCK.json)의29개 파일 해시 모두 일치.
- 기존 `POSE_PREDICTIONS.json`은 D9 선택이다. 현재 GEO_LINEAR 결과와 같다고 가정하지 말고, R0 좌표 cache로 현재 `pallet_pose_diagnosis_20260930_v1.run.candidates`를 다시 실행해 같은 계약으로 평가한다.
- 기존 FULL125 및 진단용 PRIOR1 frozen cache에는 wood45가0장이다. 고정 모델45장씩의 새 forward가 필요하다. 캐시가 있는 것처럼 과거 다른 teacher 출력을 대체하지 않는다.

wood45는 원래 목표인 plastic natural99의 보조 형태 전이 검증이다. 두 recording, Severe0, unknown 참조 출처라는 한계가 있어 이것만으로 자연 가림 T/R의 안정적 개선을 주장할 수 없다. plastic과 wood를 합쳐 좋아진 평균으로 판정하지 않는다.

## 확인되지 않은 자료와 권장 검증 경로

검토한 기존 pair manifest와 연구 코드에서 카메라·팔레트 고정 및 움직임 parity가 검증된 **자연 clean→가림→clean pair는0개**다. E3의 동일 이미지 인공 mask pair와 E/H의 같은 recording 예산 대응은 이러한 물리 pair를 대신하지 않는다. 전체 raw 영상의 모든 가능한 pair를 시각적으로 검사한 것은 아니므로 “디스크 어디에도 존재하지 않는다”는 주장도 하지 않는다.

파일 이름의 `FINAL`은 독립성 증거가 아니다. `FINAL_EVAL`은 계약상 DEV 재사용 alias다. physical FINAL229장 중146장은 이미 PAPER319에 포함되어 반복 사용되었고, 나머지83장은 `plastic_night_01`의 미주석 이미지다. 이83장도 REC_002이므로 기존 Replay 교사와 recording을 분리한 확인 자료가 아니다.

현재 자료로 가장 정직한 경로는 다음과 같다.

1. 고정 full128/natural99/clean29 및 wood45를 평가 전용으로 보존하고 새 모델의 학습 입력·의사 타깃·샘플링·checkpoint 결정에 그 결과를 사용하지 않는다. 원래 봉인된 canonical final-test4세션으로 새로운 threshold 탐색을 하지 않는다. 이미 노출된 paper DEV를 재봉인된 TEST라고 부르지 않는다.
2. 누수1장을 제외한 SINGLE252와 recording이 다양한 DIVERSE252의 의사 타깃·수치 예산·source replay·증강·최적화 설정을 맞춘다. 차이를 recording 다양성으로 좁히되 recording별 검출 성공/필터 통과율도 남겨 선택 효과를 숨기지 않는다.
3. 학습 전에 T와 R의 공동 개선, 전체 분모/실패, Clean 보존, recording별 손익, recording bootstrap/LORO와 seed 반복의 판정 규칙을 고정한다. 모든 seed를 보고하고 가장 좋은 seed 하나로 성공을 대체하지 않는다.
4. 실험이 통과하면 **“이 고정된 재사용 DEV에서 여러 seed와 recording에 걸친 안정적 개선”**이라는 한정된 증거를 얻는다. 새로운 촬영 조건까지의 독립 일반화는 별도 미노출 recording과 신뢰 가능한 참조가 확보되기 전까지 미입증이다.

독립 확인 데이터와 물리 자연 pair가 없다는 사실은 지금 할 수 있는 구현·통제 학습·기존 자료 검증을 중단할 이유는 아니다. 다만 이 자료로 증명할 수 있는 범위를 넘어서는 성공 선언을 막는 한계다.
