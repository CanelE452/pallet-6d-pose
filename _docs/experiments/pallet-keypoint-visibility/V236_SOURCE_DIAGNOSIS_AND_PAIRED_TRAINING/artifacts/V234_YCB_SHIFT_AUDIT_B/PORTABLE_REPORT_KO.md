# YCB SOURCE 내부 FIT/DEV 분포 진단

저장값 검산은 PASS이며, 양팔이 놓친 DEV E 57행에는 뚜렷한 **물체×aux slot 구성 차이**가 있습니다. 이를 실제 촬영조건이나 모델 실패의 인과 설명으로 확정할 수는 없습니다. 새 분리·학습·모델 선택은 하지 않았습니다.

| 분모 | FIT | DEV |
|---|---:|---:|
| 기존 TRAIN instance / frame | 700 / 500 | 300 / 200 |
| 원 associated-r slot | 5,600 | 2,400 |
| 원 D / E / S / U | 841 / 283 / 1,412 / 3,064 | 340 / 57 / 505 / 1,498 |
| available slot | 5,456 | 2,265 |
| 실제 known D/E·available 점수 행 | 1,124 | 397 |
| 위 점수 행의 frame / instance | 480 / 659 | 176 / 267 |
| 점수 행 원 L 중앙값 (px) | 46.4597 | 45.0060 |

FIT는 scene 000000–000004, DEV는 000006/000007입니다. 원 TRAIN 역할, q/L, slot 순서와 D/E/S/U를 바꾸지 않았습니다. 입력 1,000×8, 저장 전체행의 YCB 8,000, 진단 1,521, 두 팔 CSV의 YCB 각 1,521행을 조인했고 nonpixel samples metadata의 ID/role/target/availability도 일치했습니다. 나머지 S/U와 unavailable은 원 전체행에 보존되며 이 known D/E 점수 분모에 합치지 않았습니다.

두 팔의 YCB DEV CM은 모두 `[[333,7,0],[57,0,0]]` (행 D/E, 열 D/E/U)로, E 57/57을 D로 예측했습니다. E 57행은 서로 다른 57개 frame·instance에 속하지만 물리적으로 독립인 57개 가림 사건이라는 인증은 아닙니다.

| DEV E 그룹 | DEV E | FIT 같은 object의 E | FIT 같은 object×slot의 D / E |
|---|---:|---:|---:|
| scene000006, object7, aux3 | 29 | 5 | 79 / 0 |
| scene000006, object7, aux7 | 24 | 5 | 20 / 0 |
| scene000007, object2, aux2 | 4 | 127 | 46 / 0 |

따라서 53/57(92.98%)은 scene000006/object7에 모입니다. 해당 object7의 FIT E는 전체 283개 중 5개(1.77%)뿐입니다. FIT E의 나머지는 object2 127개, object8 151개입니다. DEV E 물체가 FIT에 아예 없었던 것은 아닙니다. 같은 object×aux slot의 FIT **known D/E** 지원은 57/57에 있으나, 그 조합의 FIT **E** 지원은 0/57입니다. aux slot만 따로 보면 33/57에 FIT E 지원이 있어도, object와 slot의 결합 지원은 없다는 차이가 중요합니다.

`object_id`와 `logical_aux_slot`은 categorical SOURCE metadata입니다. native는 모두 UNKNOWN이며 이 여덟 slot을 pallet native8이나 물리 부위 대응으로 격상하지 않습니다. 같은 object×slot의 서로 다른 frame/instance에서 D와 E가 달라지는 것은 곧 동일 tensor에 반대 라벨이 있다는 뜻도 아닙니다. 작은 순수 counterexample에서도 object·slot 각각의 marginal 지원과 결합 지원이 다를 수 있음을 확인했습니다.

원 해상도는 기록상 모두 480×640입니다. known D/E 행의 q y 중앙값은 FIT 245.64, DEV 333.77px이며 object7에서는 FIT 267.77, DEV 374.13px입니다. 이러한 q/L·scene 구성 차이는 기술 통계입니다. 허용 metadata에는 lighting·camera pose·occluder 종류·실제 촬영조건을 인증하는 필드가 없으므로 그 집중 여부는 UNKNOWN입니다. associated-r 좌표와 q–r 거리는 이번 target/진단 export에 없어 추적하지 않았습니다.

감독은 기존 **associated-r surface teacher D/E/S/U**이며 literal q-owner를 새 target으로 쓰지 않았습니다. q는 원 SOURCE GT-jitter emulation 계보를 유지하고 실제 initializer 분포라고 주장하지 않습니다. 이미 사용된 TRAIN1000의 내부 DEV이므로 fresh blind 일반화·현재 실사 개선·native human gold를 뜻하지 않습니다. 이 진단은 SOURCE 분포 차이를 보존하며 새 학습 조건이나 재선정 근거를 제안하지 않습니다.

검산 범위는 승인된 saved JSON/JSONL/CSV와 samples NPZ의 비영상 metadata 다섯 member입니다. raw RGB·원 GT·prefix·모델/가중치·current/held를 읽지 않았고 metadata 안의 image binding도 따라가지 않았습니다.

- [분포 전체표](DISTRIBUTIONS.csv)
- [놓친 DEV E의 scene/object 분모](MISSED_DEV_E_BY_OBJECT_AND_SCENE.csv)
- [object×aux slot 라벨 지원표](OBJECT_AUX_LABEL_COVERAGE.csv)
- [독립 검산 영수증](REVIEW.json)
- [허용 범위](SCOPE_BEFORE_SAVED_METADATA.json)
- [검산 코드](verify_ycb_shift.py)
