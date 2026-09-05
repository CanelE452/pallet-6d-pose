# 데이터 population 감사 — 세 역할을 섞지 않는다

## SOURCE_TRAIN — 합성, 정확한 라벨

```
yaml     challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k/data.yaml
train    images/train 55,980  (labels 동수)
val      images/val    4,020  ← 이 트랙에서 SOURCE_DEV 로 쓴다
계약     nc 1 · kpt_shape [9,3] · flip_idx [1,0,3,2,5,4,7,6,8] · names {0: pallet}
replay   1,440장 고정 부분집합 SYNTHETIC_REPLAY_SUBSET.txt
         membership_sha256 9dbd9c92…  (sha256(파일명) 정렬 앞 1,440)
```

`images/val` 4,020 은 기존 self-training 트랙이 **model selection 에 쓰지 않는다**
(checkpoint 규칙이 last.pt 고정). 이 트랙은 이것을 SOURCE_DEV 로 삼아
Gate B 후보생성 파라미터 sanity check · Gate C 의 coarse residual 분포 추정 ·
Gate E 의 domain classifier source 쪽에 쓴다. **real GT 로는 어떤 파라미터도 고르지 않는다.**

## TARGET_UNLABELED — 실제, 라벨 없음

```
manifest  data/evaluation/pallet_eval_v1/adaptation/MAIN_UNLABELED_BALANCED.csv   1,000행
          (DAYTIME_UNLABELED.csv 500 + NIGHTTIME_UNLABELED.csv 500)
          컬럼 image_path, image_sha256, capture_session, paper_condition
lock      data/evaluation/pallet_eval_v1/adaptation/ADAPTATION_POOL_LOCK.json
          manifest_sha256(MAIN) afb581a0…d303b
이미지     data/pallet/raw_data/outside/{capturepallet01,10,11}/rgb          실측 2,227
          data/pallet/raw_data/night/{capturenight01,02,03,04,10}/rgb       실측 5,804
                                                              적격 합 8,031, 사용 1,000
세션선정   metric_split_lock.md §1.6 에서 옮긴 목록. 여기서 새로 고르지 않는다.
```

### 누수 감사 — 이미 존재하고 통과 상태다 `[확인]`

`ADAPTATION_POOL_LOCK.json`
```
leakage_gate.adapt_session_intersect_eval_session  []
leakage_gate.adapt_sha_intersect_eval_sha           0
daytime   images_found 2,227  eval_sha_overlap 0  eval_session_overlap []
nighttime images_found 5,804  eval_sha_overlap 0  eval_session_overlap []
passed true
```

**이 트랙의 처리**: TARGET_UNLABELED = 동결된 1,000장 그대로 쓴다. 8,031 로 늘리지 않는다.
늘리면 기존 exposure contract 와 비교가 깨지고, 이 트랙의 질문은 "양" 이 아니라
"교사 합의의 질" 이기 때문이다.

## DEV_EVAL — PAPER_EVAL positive 319

```
manifest  challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json
          expected_count 319 · membership_sha256 8f5c28fe…5b0d · role DEV
          held_out_final false
구성      plastic 194 / wood 125 · 13 세션 · 이미지 크기 640x480 274 · 1280x720 45
프레임순서 INFERENCE_REPLAY_LOCK frame_order_sha256 72f83f6f…d8c15f
keypoint  supervised 코너 2,499/2,552 (98.0%) · visible(visibility==2) 1,594 (62.5%)
          visible 개수가 index 별로 크게 다르다:
            0:279 1:269 2:224 3:209 4:252 5:247 6:50 7:64   (6,7 = 먼 아래쪽, 거의 안 보임)
          centroid 8 은 항상 supervised, 항상 visible 아님 (내부점)
프레임당   supervised 코너 중앙값 8 (최소 4) · visible 코너 중앙값 5 (2~7)
```

역할은 DEV 다. 이미 반복 사용됐다. 이 트랙의 결과는 좋아져도 held-out 이 아니다.

## INDEPENDENT_CONFIRMATION_AVAILABLE = **NO**

METHOD_LOCK 시점 기준으로, "논문 트랙과 같은 기하 + 같은 camera-facing 9점 규약 +
train/pseudo pool 과 SHA 겹침 0 + 기존 method selection 미사용" 을 모두 만족하는
real manual-GT population 은 저장소에 **없다**. 근거:

```
후보                                판정
────────────────────────────────────────────────────────────────────────────
live_capture_gt  28 세션 851장      기하 불일치 — plastic_standard_110x110x15 (정사각)
                                    논문 트랙은 110x130x11. 정사각이면 W/D 축 질문 자체가 성립 안 함
                                    라벨 규약 감사에서 402 중 106 좌우순서 위반 · 187 90도 stale
                                    EXPERIMENT_STOP_LOCK 이 REAL_FT_V1 을 NOT_RUN 으로 봉인
                                    게다가 challenge FT 트랙(ft_live_gt_v1..v4)이 이미 학습에 소비
capturenight05/06/07 잔여 1,581장   기하 일치 · 세션 독립 · 그러나 **어노테이션 0장**
legacy_unverified 121장             axis_assignment_confirmed 부재 · DATASET_CONTRACT 가
                                    "not paper eligible" 로 규정 · n=121 은 95% CI ±3.9pp
FINAL_TEST 4 세션 105장             2026-08-20 에 개봉됨. 재봉인 불가
POSE_FINAL                          inventory 0
```

따라서 이 트랙의 최종 상태는 어떤 결과가 나와도
**DEVELOPMENT_METHOD_SIGNAL** 이다. held-out / confirmed / final / SOTA 표현 금지.

## 재사용할 exposure contract (Gate D2)

`data/pallet/results/paper_selftrain_v1/SELFTRAIN_EXPOSURE_LOCK.json`
```
batch 32 · lr0 0.002 · epochs 10 · updates/epoch 90 · total_optimizer_updates 900
pseudo 1,440/epoch + synthetic replay 1,440/epoch  (= 50/50)
작은 pool 은 복원추출로 슬롯을 채운다
```
V2/V3 는 같은 900 updates 에서 비율만 pseudo 0.25 / synthetic 0.75 로 바꿨다.
이 트랙은 **v1 의 50/50 · 900 updates 를 그대로 쓴다** — 비율을 새로 고르면
새 방법의 효과와 비율 변경의 효과가 섞인다.

## 미확인으로 남긴 것

```
1  live_capture_gt 신규 449장(09-03/09-04)의 PAPER_EVAL SHA 겹침 감사 — 402장분만 존재
   → 이 트랙은 live_capture_gt 를 아예 쓰지 않으므로 영향 없음
2  EXPERIMENT_STOP_LOCK 이 인용한 convention audit(106/402·187/402)의 원본 산출물 파일
   → 저장소에서 발견되지 않음. 결론은 lock 문서의 선언으로만 존재
3  88a87d1 4-fold 정규화 이후 그 위반이 해소됐는지 재감사 아티팩트 없음
```
