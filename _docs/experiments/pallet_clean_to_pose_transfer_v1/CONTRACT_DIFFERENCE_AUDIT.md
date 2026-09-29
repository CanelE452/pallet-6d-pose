# Clean19 양성 신호와 현재 main의 계약 차이 감사

감사 기준 HEAD: `bdd9f84515921aae47367cc4507ba62ca8373bb9`. 신규 fit/optimizer update/GPU 추론 **0**. 기존 코드·잠금·학습 trace 집계·보고서를 읽어 비교했다. 현재 작업의 결과가 이미 완료돼 있다는 근거는 없고, 재사용 가능한 것은 아래 과거 계약과 frozen 결과다. 출처 경로와 실제 SHA256은 [CONTRACT_DIFFERENCE.json](CONTRACT_DIFFERENCE.json)에 기록했다.

## 결론

**“같은 방법의 clean→occlusion 효과가 재현되지 않았다”는 비교가 아니다.** 과거 Plastic S1은 사람이 Clean으로 분류한 **10장, 직접 클릭 출처 48개 채널, 종류별 교사, 넓은 학습 범위, LR 1e-4**였다. 현재는 **217장, 공통 confidence support, Replay9/38 교사, pose/flow-only, LR 1e-5**다. 기존 main 실행에는 217장 전부가 external-clean이라는 근거가 없었다. 이번 RGB 재검토와 사용자 요청인 마커판 제외 규칙은 별도 `CLEAN_POOL_AUDIT.md` 및 membership lock의 판단으로 구분한다.

과거 S0→S1의 matched 대조는 그 과거 조건 안에서 유효하다. 하지만 과거 S1과 현재 REF를 직접 빼서 LR·teacher·clean 여부 중 어느 하나가 원인이라고 확정할 수 없다. 또한 과거 Severe81/78과 현재 Moderate+Severe99는 다른 모집단이다. 같은 128장/99장, 같은 최종 metric으로 frozen 결과를 재집계한 뒤 비교해야 한다.

## 1. 주 계약 대조

| 항목 | 과거 Clean19 / Plastic S0·S1·S2 | 현재 Plastic main RAW_LR5·REF_LR5 | 비교 영향 |
|---|---|---|---|
| 출발 모델 | 동일 synthetic-pallet R0, upstream COCO-pose 사전학습 있음 | 동일 R0 SHA `970a0913…` | 공통 초기값은 확인됨 |
| teacher/refiner | Plastic 전용 Replay `pallet_replay_by_type_v1/plastic/step300.pt`; TYPE_REPLAY_PIPELINE의 visible refinement + hidden PnP | frozen PoseFix-derived RGB ResNet152 Replay `pallet_posefix_replay_v1/last300.pt` | 다른 checkpoint와 감독 계약; 교사 차이와 material 차이를 섞으면 안 됨 |
| teacher 수동 예산 | Plastic10장/48코너; Wood9장/39코너. 각각 synthetic-only PRIOR1에서300step, TFAdam 1e-4, 실사/합성 각2400 exposure | 총9장/38코너 = Plastic3/15 + Wood6/23 | 87은 old 두 재료 합계이며 현재38과 같은 supervision budget 아님 |
| real TRAIN membership | Plastic10, day5/night5, REC_001/002; Wood9 별도 | candidate1000→raw통과259→corrected통과249→replacement로 실제고유217 | 이미지 수뿐 아니라 촬영·구성·반복 가중치가 다름 |
| Clean 근거 | 기존 사람의 `COMBINED319` 난도 tag=CLEAN, manual≥3, session quota, RGB-thumbnail 다양성·near-duplicate 배제; 오차로 선택 안 함 | confidence/flip/LOO 통과는 정확도·external-clean 증거가 아님 | 현재 clean 상태는 별도 CLEAN_POOL_AUDIT로 판정해야 함 |
| supervised support | Plastic48코너: 4점 이미지4, 5점4, 6점2. 두 재료 합87. 비수동65코너(PnP대체19 포함)+center19 제외 | 고유217에서 코너1627 + center214. 이미지별 총support 9:169장 /8:5 /7:34 /5:6 /4:3 | old는 manual provenance로 좁힌 support; current는 pseudo support. 더 많은 support가 더 정확하다는 뜻 아님 |
| 타깃 좌표 source | yT frozen teacher 좌표 그대로, 수동좌표로 치환하지 않음. mask만 직접클릭 근거이며 물리적 visible 보장은 아님 | raw=R0, ref=Replay; 공통 finite/confidence mask 교집합. center 좌표는 RAW/REF 동일 | old에 RAW target 학생이 없으므로 old S0/S1은 corrected-vs-raw 인과 대조 아님 |
| 숨은 점 / PnP | teacher pipeline에 hidden PnP가 있으나 새 S0/S1/S2 좌표감독에서 PnP대체19 전부 제외 | 현재 REF에는 별도 hidden PnP completion 추가 없음 | PnP보완을 직접 manual visible 감독으로 승격하지 않음 |
| 학생 trainable | 저장 FIT inventory 498 parameter tensors / 3,043,704 scalars. TrueIgnorePoseTrainer stock 경로, 새로운 freeze 없음 | 132 parameter tensors /539,514 scalars, pose branches+flow. backbone/neck/detector 및 모든 buffer 고정 | 표현학습·검출·BN 변화 가능성이 다름. “더 넓게 학습하면 해결”은 미검증 |
| optimizer / LR | AdamW, lr0=1e-4, lrf=.1 cosine, weight_decay=1e-4, warmup0 | 동일 optimizer/스케줄, **arm override lr0=1e-5** | protocol.args.lr0=1e-4는 공통 template; arms.REF_LR5.lr가 실제값 |
| 학습량 / 선택 | 5epoch×64batch=320updates, batch=nbs16, seed42, last만 | 동일 | 같은 update 수가 같은 이미지당 exposure는 아님 |
| real/source exposure | epoch당 real512+source512, 전체각2560. Plastic10 평균256회/영상 | 전체각2560, Plastic217 평균11.80회/영상 | old는 아주 적은 clean 영상을 강하게 반복 |
| synthetic replay | 512 unique, 별도 extra input occlusion 없음 | 같은512 unique, negative0 | 두 train.txt의 resolved source file 512/512 및 source 슬롯 순서 동일을 직접 확인. 실제 증강 tensor는 별개 |
| base augmentation | HSV(.015,.5,.35), translate.1, scale.25; rotation/shear/perspective/flip/mosaic/mixup/copy-paste/erasing0 | 설정은 같음 | old는 occurrence별 RNG로 만든640tensor를 캐시, current는 loader runtime. old-current bit-exact tensor 동일은 아님 |
| 인공 가림 | S1 random placement, S2 graph인접점 placement; 같은 크기·fill·bbox-overlap tolerance의 **paired feasibility**를 만족한 경우만 둘 다 적용 | main 없음; v2 A=random REF-conditioned full-canvas32proposal, schedule.5; C=schedule1로만 변경 | S1은 unconditional random erasing 아님. old/new placement domain과 RNG 소비순서 다름 |
| RAW/REF 공통성 | S0/S1/S2는 좌표/support/box/base RGB/cache/순서 전수5120 parity | export support/box/order 공통. v2 실제 affine clipping 후 RAW/REF support 차이가3batch에 존재 | export parity와 전수 augmented support parity를 구분 |
| 추론 계약 | reflect100 BORDER_REFLECT_101 →640 rect=True, conf.001, no augment/half, argmax confidence →padding inverse | 동일 구조; frozen detector이므로 boxes/scores 공통 기대 | 학습-assigned output, GT closest candidate, oracle로 바꾸지 않음 |
| pose 후보 생성 | common production D9, corner0..7로 solve, W/D 두 extents 후보, 기존 SQPnP/LM | 같은 deployable D9와 등록치수/K | point 오류·후보 품질·최종선택을 따로 봐야 함 |
| pose selector | S0/S1 원 비교는 D9. 후속 old GEO_LINEAR는 별도 실험 | current main D9; GEO 미적용 | S1+GEO 결과를 원 S1/D9나 main REF로 섞지 않음 |
| 평가 population | 최초 ALL300=Plastic184+Wood116, same-recording 포함; 후속 Plastic HELDOUT128=29Clean+21Moderate+78Severe | 같은 HELDOUT128과 primary Mod+Sev99. Wood extension45는 별도 | Severe81↔78↔Mod+Sev99 직접 비율 비교 금지 |
| recording 분리 | old Plastic TRAIN REC_001/002;128평가는 REC_007/021/022/025/027/041/044. overlap/SHA0 | TRAIN REC_011/017/019/020/023/024/028, 같은 평가7recording. 교사 REC_001/002 | recording-disjoint여도 이미 반복 열람한 DEV이며 independent TEST 아님 |
| T/R reference | geometry-derived reference, C2 대칭 full rotation, W/D90°는 동치 아님; independent physical6D 아님 | 동일 원리. v2에centroid L2 cm와object→camera C2 geodesic, failure+∞/coverage를 명시 | yaw와full R, conditional median과실패포함 quantile 구분; camera x/z는차량좌표아님 |

학생 loss의 true-ignore 값 1, supervised 2, invisible 0 의미를 보존한다. 가림으로 좌표·support를 바꾸지 않는 amodal target 복원이 핵심이며 hidden 코너를 새 manual 정답으로 만들지 않는다.

## 2. 실제 가림 노출 — 이름만 같은 augmentation이 아님

아래는 **Plastic, 각 실사 2560 occurrence** 기준이다. supervised-point count는 같은 한 코너가 반복 노출되면 반복해서 센다. old S1/S2는 48개 unique corner, current는 훨씬 넓은 pseudo support이므로 point 총량의 분모도 다르다.

| 실행 | scheduled | actual applied | 적용률 | 가려진 감독코너 | 적용된 영상에서 남은 감독코너 | placement 실패 |
|---|---:|---:|---:|---:|---:|---|
| old S1 random |1241|730|28.52%|998|2510|508 paired-position 없음 +3 감독4점 미만 |
| old S2 structured |1241|730|28.52%|1477|2031|동일 paired skip |
| current main |0|0|0%|0|해당없음|해당없음 |
| v2 A RAW |1302|542|21.17%|781|이 감사에서 미재집계|760 random-valid 위치 없음 |
| v2 A REF |1302|542|21.17%|824|이 감사에서 미재집계|동일 계획 |
| v2 C RAW |2560|1054|41.17%|1470|이 감사에서 미재집계|1506 미적용 |
| v2 C REF |2560|1054|41.17%|1559|이 감사에서 미재집계|동일 계획 |

old S1/S2는 크기·fill·frequency를 맞췄지만 S2 masked target 1477 vs S1 998이라 구조 효과와 감독점 가림량을 완벽히 분리하지 못했다. old 전체 Plastic+Wood를 합친 1312회·1695/2653점 수치를 Plastic 단독 값으로 쓰지 않는다.

old S1은 native canvas를 reflect100 입력에 affine한 허용 영역에 배치하며 target≥1을 덮고 ≥2를 남긴다. S2와의 bbox-overlap 차이 ≤max(1px², bbox 면적 1%)를 만족해야 적용한다. v2 A는 full640 canvas, 최대 32 proposal, REF 변환 타깃으로 공통 mask 위치를 고르며 S2 가능 조건은 없다. size는 bbox 면적 fraction {.1,.2,.3}, aspect {.5,1,2}; fill은 8×8 random RGB를 bilinear 확대하는 원리를 공유한다. 입력 영역·support·RNG가 달라 old 이미지와 bit-exact라고 하지 않는다.

v2 A 실제 감독 수는 RAW 21819 / REF 21823, true-ignore 1185는 동일하며 source 감독 22531도 동일이다. REF 824/21823=3.78%는 center 포함 감독 분모에 가려진 8-corner 수를 나눈 기존 기술값이므로 정확히 그 정의로만 해석한다. C도 support 차이 3 batch가 남았다. 새 2×2는 실제 변환 후 support 차이를 preflight에서 검사하고, 이를 숨기지 않아야 한다.

## 3. 기존 양성·음성 신호의 올바른 역할

- **과거 matched S0→S1:** 같은 Clean10/R0/teacher/support/cache/최적화에서 S1 RGB 가림만 다르므로 해당 정책의 효과를 검사했다. recording-disjoint Severe78에서 T 15.080→13.407cm, R 12.542→5.630°였다. Clean은 T 4.297→4.707cm, R 1.513→1.573°로 손상; Moderate는 T 7.037→6.904cm이나 R 1.816→1.817°였다. “Clean·Moderate·Severe 전반 동시 성공”이 아니다.
- **현재 A/C:** primary99에서 A REF는 OLD_REF 대비 T 개선/R 악화; C REF는 T −.035730cm/R −.012101°의 작은 부호만 있고 R0보다 두 중앙값이 높다. C−A는 T 악화/R 개선이다. 따라서 빈도 증가만으로 현재 문제가 해결됐다고 하지 않는다.
- **current TRAIN-following:** 217영상/1627코너에서 REF 학생은 corrected 타깃을 RAW 학생보다 더 따른다. 이것은 타깃 추종이며 그 타깃의 물리 정확도 증거가 아니다. 높은 잔차만으로 LR/표현/노출량 중 한 원인을 확정할 수 없다.
- **seed 경고:** v2 명목 seed43의 loader stream/state가 seed42와 같은 재실행이 확인됐다. 새로운 독립 반복으로 세지 않고 CPU preflight에서 실제 stream 차이를 확인해야 한다.

## 4. hard/selector 과거 결과는 보조 근거

`pallet_min_hard_ab_v1`은 Clean448+hard64+source512, hard8 직접36코너의 **추가 감독**이다. H_PSEUDO/H_MANUAL는 hard 좌표 source만 다르나 current217 질문과 같지 않다. 기존 S1+GEO에 비해 hard PCK/oracle은 좋아지고 최종 AUC는 나빠졌다. 새 학생으로 old GEO를 그대로 쓰면 선택 호환성이 깨질 수 있다는 반례이지 모든 오류가 selector 때문이라는 증명이 아니다.

`pallet_selector_recovery_v1`의 old GEO는 S0+S1 출력을 모두 사용한 synthetic-only scorer, 4096/1024/1024 renderer-group 분리이다. S1 Moderate D9 AUC .435690→GEO .475333, Severe .195397→.211308; 반면 S0 Moderate는 악화했다. `pallet_model_conditioned_selector_v1`은 동일 94 features/Linear/고정 optimizer를 S1 전용과 H_MANUAL 전용으로 각각 학습해 교차 비교했다. H_MANUAL own GEO에서 일부 gap 회복이 있지만 합성 TEST는 old GEO보다 낮고 S1 전용도 old보다 항상 좋지 않았다. 현재 새 학생에 GEO가 도움될지는 frozen zero-fit compatibility부터 확인할 조건부 질문이다.

## 5. 원인 가설과 이번 결정에 주는 제약

| 가설 | 확인한 차이 | 아직 반박되지 않은 대안 / 다음검사 |
|---|---|---|
| clean membership 차이 | old 사람 CLEAN10; 기존 main은 current217 전수 external-clean 미검증 | RGB/기존 태그로만 membership 고정. pseudo 통과나 작은 오차로 clean 선택 금지 |
| target/support 신뢰성 | old manual-provenance48 vs current confidence 코너1627 | old teacher와 학습 budget도 다르므로 support만의 효과 미확정 |
| 최적화/표현 범위 | LR 10배, trainable 3,043,704 vs 539,514 | 새 clean 2×2의 TRAIN-fit을 본 후 한 축 bridge만; 둘 동시에 변경 금지 |
| 가림 노출/placement | applied 비율·masked point·canvas 조건 다름 | 단순 C 빈도 증가의 joint gain은 작고 R0 미달. 크기/probability 후향 sweep 금지 |
| recording/data diversity | old 2촬영 10장 강한 반복 vs current 7촬영 217장 | 영상 수/다양성/target 품질 효과가 얽혀 원인 확정 불가 |
| 후보 선택 | 과거 좋은 후보를 놓치는 사례와 model-dependent GEO 호환성 | 새 localization과 candidate가 좋아질 때만 selector 분기; oracle 배포 성능 금지 |

이 감사는 원인 분리를 위한 사전 계약 복원이지 성능 결과를 보고 한 변수를 승자로 고른 문서가 아니다. 모델 출력 없는 clean 판정을 잠글 수 없으면 fit0으로 그 제약을 보고해야 하며, 근거 없이 217장을 clean이라고 부른 2×2를 실행하면 안 된다. 이번 실제 membership은 별도 clean 감사 산출물에서 잠근다.
