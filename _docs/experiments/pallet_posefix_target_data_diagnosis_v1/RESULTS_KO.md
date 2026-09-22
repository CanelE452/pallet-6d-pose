# TARGET DATA / TRAIN FIT — 원인 분해 결과

**주 병목: HARD_SUPERVISION_SHORTAGE. 보조: TRANSFER_MORPHOLOGY_GAP_SIGNAL.** 현재 TRAIN에 큰 실사 입력 오류가 드물고, 학습한 TRAIN에서는 복구하지만 DEV로 같은 후보 생성이 전이되지 않는다. target/reference 신뢰도 차이와 한 recording의 memorization 가능성은 분리되지 않았다. 모델 용량이 충분하다거나, 데이터만 늘리면 해결된다는 결론은 아니다.

신규 학습 0 / optimizer step 0 / checkpoint 갱신 0 / GT·pseudo target 변경 0. 기존 모든 결과와 최종 모델 유지.

## 핵심 수치

|항목|결과|
|---|---|
|TRAIN 감독 코너|2024 / 253장|
|CLEAN 입력 >20px|0 (전부 ≤5px)|
|OCC 입력 >20px|50 / 2024 = 2.47%|
|OCC 20–40 / >40|50 / 0|
|hard 코너 보유 프레임|25; 2개 이상 23, 4개 이상 1|
|반복 노출|hard 444 / 전체 19200; 고유 hard는 50개|
|STRICT|253장 전체, hard 50개 유지|
|DEV matched R0 >20|196/659 = 29.74%|
|TRAIN/DEV hard 비율|0.0831 (약 1/12)|
|easy→hard20 / hard40|50 / 0|
|자연 hard TRAIN FULL 복구|49/50 = 98.00%|
|top5 후보 TRAIN / DEV|100.00% / 9.69%|

TRAIN error는 frozen pseudo와의 거리, DEV error는 기존 reference와의 거리다. 독립 물리 GT가 같은 두 집단의 성능 비교로 해석하지 않는다. DEV 주 모집단93장 중 matched85장/659코너만 분포 비교하며, 기존 official 전체713코너의 실패 penalty는 변경하지 않는다.

## Controlled basin: 원영상 좌표 오류를 복구하는가

TRAIN-only로 고른 128개 코너, 반경당 네 방향 총512회. 전체 TRAIN census와 구분. P1/P2는 같은 CLEAN bbox·다른 점·선택점 교란을 유지하고 RGB만 바꿨다. P3 자연 OCC는 기존 OCC R0의 다른 점과 bbox도 다를 수 있어 P2−P3를 순수 point morphology 효과로 단정하지 않는다.

|모델|RGB|r20 ≤10|r30 ≤10|r40 ≤10|자연20–40 ≤10|
|---|---|---:|---:|---:|---:|
|PRIOR1|P1|10.16%|2.15%|0.78%|0.00%|
|PRIOR1|P2|9.96%|2.34%|0.98%|0.00%|
|FULL125|P1|97.85%|97.27%|98.24%|98.00%|
|FULL125|P2|95.12%|94.14%|94.53%|98.00%|
|FULL_PRESERVE|P1|91.41%|91.41%|90.04%|98.00%|
|FULL_PRESERVE|P2|89.26%|88.67%|88.28%|98.00%|
|FULL150|P1|97.85%|98.83%|98.83%|100.00%|
|FULL150|P2|94.34%|92.97%|92.38%|100.00%|

>40 자연 hard TRAIN은 n=0이므로 성공률을 0%로 쓰지 않는다. 통제 40px 진단은 자연 >40px 사례가 아니다. 높은 TRAIN 복구는 이 recording/pseudo 기준의 결과이며 generalization 또는 physical correctness 보장이 아니다.

## Morphology / 시간적 다양성

한 recording에서 hard 프레임 25개, 연속 display-index run 20개, 최장 2개. 이 run 수를 독립 scene 수라고 부르지 않는다.

TRAIN hard는 두 코너 동시 오류가 22/25장. DEV hard는 4개 이상 동시 오류가 26/56장이다. TRAIN도 coherent 방향 오류가 있으므로 “TRAIN은 random single-corner뿐”이라고 하지 않는다. 코너쌍·평균 이동·이동 제거 잔차·similarity 잔차는 별도 JSON에 저장했다.

## Target consistency / excluded11 / reference 한계

accepted253 프레임 평균 OCC 오류와 stage1 LOO / flip / stage2 LOO의 Spearman ρ는 각각 −0.097 / −0.063 / −0.079였다. 이 선택된 집합 안에서는 hard일수록 consistency가 악화되는 강한 양의 상관이 관찰되지 않았다. 선택 편향 때문에 후보 전체의 인과관계로 일반화하지 않는다. clean_error와 teacher movement는 정의상 같은 변수다.

제외11장을 개별 확인한 raw score 범위는 0.6124–0.8473으로, 모두 0.85 미만이다. valid kp는 center 포함9개였으므로 이 11장은 낮은 검출 score에서 탈락한 것으로 확인된다. 그 사진의 코너 정답이 어렵거나 잘못됐다는 뜻은 아니다.

STRICT 0.025에서 accepted253 전부 유지되어 이번 기준으로 hard target만 consistency가 약하다는 증거는 없다. 제외11장의 기록 reason은 모두 raw_confidence다. 이는 raw score/valid-keypoint 공통 초기 gate의 이름으로, 실제 값은 비공개 EXCLUDED11 표에서 따로 확인했다. 제외된 점의 정답 정확도나 final target은 추정하지 않았다.

PRIMARY hard 196개 중 출처 unknown 196, higher-confidence manual-source 0. 비교할 독립 확인된 hard subset이 없으므로 unknown에 실패가 집중됐다는 사실만으로 GT가 주 원인이라고 할 수 없다. GREEN manual-only는 기존 별도 population 유지. 물리/가상 identity와 가시성 독립 검토는 미완료.

## 19개 필수 질문에 대한 답

1. 감독 2024코너.
2. CLEAN >20: 0.
3. OCC >20: 50.
4. 20–40 / >40: 50/0.
5. hard 프레임 25.
6. 300step hard 반복 노출 444, 고유 50.
7. STRICT hard 50, 25장.
8. TRAIN 2.47% vs DEV 29.74%.
9. easy→hard20/40=50/0.
10. FULL controlled20/30/40=97.85%, 97.27%, 98.24%.
11. OCC RGB controlled20/30/40=95.12%, 94.14%, 94.53%.
12. FULL natural20–40=98.00%; >40 n=0.
13. TRAIN/DEV top5=100.00% / 9.69%.
14. TRAIN은 주로 두 코너의 동반 오류, DEV는 더 많은 코너 동시 오류와 큰 tail; 단순 방향 coherence는 양쪽 모두 존재.
15. 현재 STRICT는 모두 통과. 제외11은 초기 gate; 정답 난도에 대한 인과 근거 없음.
16. unknown reference source=196/196; 오류라고 확정하지 않음.
17. HARD_SUPERVISION_SHORTAGE.
18. TRANSFER_MORPHOLOGY_GAP_SIGNAL.
19. 같은 frozen target·TRAIN 안에서 coupled hard-input dose를 바꾸는 통제 실험 한 개를 설계만 남김. 실행하지 않음.

## 분포·전이·후보 그래프


![train_clean](figures/train_clean.png)

![train_occ](figures/train_occ.png)

![dev_R0](figures/dev_R0.png)

![normalized_CDF](figures/normalized_CDF.png)

![clean_OCC_transition](figures/clean_OCC_transition.png)

![hard_timeline](figures/hard_timeline.png)

![corner_ids](figures/corner_ids.png)

![consistency_hardness](figures/consistency_hardness.png)

![controlled_basin](figures/controlled_basin.png)

![candidate_gap](figures/candidate_gap.png)

![morphology](figures/morphology.png)

## 사례 갤러리

[이미지 연속 보기](GALLERY.html). 성공/실패 예시 수가 요청보다 적으면 있는 사례만 표시한다. natural hard failure는 1개뿐이다. 블라인드 검토 화면과 별도이며 사람 응답을 생성하지 않았다.

### strict_TRAIN_hard

![strict_TRAIN_hard](figures/case_001.png)

### strict_TRAIN_hard

![strict_TRAIN_hard](figures/case_002.png)

### strict_TRAIN_hard

![strict_TRAIN_hard](figures/case_003.png)

### strict_TRAIN_hard

![strict_TRAIN_hard](figures/case_004.png)

### strict_TRAIN_hard

![strict_TRAIN_hard](figures/case_005.png)

### strict_TRAIN_hard

![strict_TRAIN_hard](figures/case_006.png)

### strict_TRAIN_hard

![strict_TRAIN_hard](figures/case_007.png)

### strict_TRAIN_hard

![strict_TRAIN_hard](figures/case_008.png)

### strict_TRAIN_hard

![strict_TRAIN_hard](figures/case_009.png)

### strict_TRAIN_hard

![strict_TRAIN_hard](figures/case_010.png)

### TRAIN_easy

![TRAIN_easy](figures/case_011.png)

### TRAIN_easy

![TRAIN_easy](figures/case_012.png)

### TRAIN_easy

![TRAIN_easy](figures/case_013.png)

### TRAIN_easy

![TRAIN_easy](figures/case_014.png)

### TRAIN_easy

![TRAIN_easy](figures/case_015.png)

### controlled_r30_success

![controlled_r30_success](figures/case_016.png)

### controlled_r30_success

![controlled_r30_success](figures/case_017.png)

### controlled_r30_success

![controlled_r30_success](figures/case_018.png)

### controlled_r30_success

![controlled_r30_success](figures/case_019.png)

### controlled_r30_success

![controlled_r30_success](figures/case_020.png)

### controlled_r30_failure

![controlled_r30_failure](figures/case_021.png)

### controlled_r30_failure

![controlled_r30_failure](figures/case_022.png)

### controlled_r30_failure

![controlled_r30_failure](figures/case_023.png)

### controlled_r30_failure

![controlled_r30_failure](figures/case_024.png)

### controlled_r30_failure

![controlled_r30_failure](figures/case_025.png)

### natural_hard_recovered

![natural_hard_recovered](figures/case_026.png)

### natural_hard_recovered

![natural_hard_recovered](figures/case_027.png)

### natural_hard_recovered

![natural_hard_recovered](figures/case_028.png)

### natural_hard_recovered

![natural_hard_recovered](figures/case_029.png)

### natural_hard_recovered

![natural_hard_recovered](figures/case_030.png)

### natural_hard_failed

![natural_hard_failed](figures/case_031.png)

### deterministic_random10

![deterministic_random10](figures/case_032.png)

### deterministic_random10

![deterministic_random10](figures/case_033.png)

### deterministic_random10

![deterministic_random10](figures/case_034.png)

### deterministic_random10

![deterministic_random10](figures/case_035.png)

### deterministic_random10

![deterministic_random10](figures/case_036.png)

### deterministic_random10

![deterministic_random10](figures/case_037.png)

### deterministic_random10

![deterministic_random10](figures/case_038.png)

### deterministic_random10

![deterministic_random10](figures/case_039.png)

### deterministic_random10

![deterministic_random10](figures/case_040.png)

### deterministic_random10

![deterministic_random10](figures/case_041.png)

### DEV_no_candidate

![DEV_no_candidate](figures/case_042.png)

### DEV_no_candidate

![DEV_no_candidate](figures/case_043.png)

### DEV_no_candidate

![DEV_no_candidate](figures/case_044.png)

### DEV_no_candidate

![DEV_no_candidate](figures/case_045.png)

### DEV_no_candidate

![DEV_no_candidate](figures/case_046.png)

## 감사·실행 주의

후속 설계는 보고서 단계에서 단순 단일점 jitter 대신 **동반 코너 오류를 포함하는 20/40/60px 고정 mixture**로 구체화했다. 60px는 이번 통제 probe에서 시험하지 않은 영역이다. 이 편집은 다음 설계만 바꾼 것으로, 현재 예측·수치·분기 임계값은 바꾸지 않았고 실행하지 않았다. 최종 설계는 NEXT_STAGE_PLAN.md와 FINAL_OUTPUT.json을 따른다.

초기 로더가 accepted-only253 manifest를 후보264로 잘못 가정한 것과 과거300장 role pool을 현재278장 집합과 직접 비교한 가정은 실제 저장 schema를 확인해 바로잡았다. 입력/프로토콜 변경 없음. GPU 진단 후 집계에서는 기존 evaluator의8-corner mask 및 null error 저장 형식을 처리하도록 분석 어댑터만 고쳤다. 완료된 TRAIN 집계는 exact 비교 재사용, 예측 재생성·추가 학습 없음. 실패 시도 메타데이터 보존.

코드는 기존 로컬 연구 모듈·동결 체크포인트·비공개 데이터에 의존하므로 공개본만으로 standalone 재현 패키지는 아니다. 원본 prediction/좌표/검토매핑은 공개하지 않는다.
