# In-house Dataset Validity

작성 2026-08-27. 수치는 전부 실측(`DATASET_AUDIT.json`).

이 문서의 목적은 **"왜 in-house dataset 을 논문 평가에 쓸 수 있는가"** 를 과장 없이
쓰는 것이다. 결론부터:

```
DATASET QUALITY AUDIT = PARTIAL
```

coverage 와 split integrity 는 상당 부분 확인됐고, **annotation reliability 는 아직
수치화되지 않았다.** 그래서 "dataset quality validated" 라고 주장하지 않는다.

---

## A. COVERAGE

metadata 에 실제로 있는 항목만 계산했다. 없는 항목은 만들지 않았다.

### capture sessions

```
7 세션 · 140 프레임
eval_cad 18 · eval_night08 12 · eval_night09 16 · eval_noapril 12
eval_outside 22 · eval_pallet07 27 · eval_pallet09 33
```

### DAY / NIGHT

```
DAY 112 (80.0%)   NIGHT 28 (20.0%)          (140 기준)
DAY 100 (78.1%)   NIGHT 28 (21.9%)          (128 기준)
```

NIGHT 이 28장뿐이다. 야간 지표의 신뢰구간이 넓다는 뜻이고, 야간 관련 주장에는
CI 를 병기해야 한다.

### image resolution

```
640 x 480   140/140  (단일 해상도)
```

### projected size (bbox 면적비 = cuboid 투영 bbox 넓이 / 이미지 넓이)

```
median 0.0896   p10 0.0284   p90 0.6546
```

p90 이 0.65 로 매우 크다 — 화면을 거의 채우는 근접 프레임이 상당수 있다.
작은 것부터 화면 가득까지 넓게 퍼져 있다.

### distance (GT translation z)

```
median 2.41 m   p10 1.69   p90 3.72   min 1.36   max 5.93
```

### viewpoint / elevation (GT R 에서 유도)

```
median 0.3 deg   p10 -2.6   p90 35.4
```

중앙값이 0 도 근처다 — **대부분이 거의 수평 시점(edge-on)** 이다. 이는 memory
`stage22-coord-loss-small-rear-signal-not-fix` 의 "real 94% 가 <8° edge-on" 과 같은 방향이다.

### occlusion / truncation

```
NOT RECORDED — 라벨에 occlusion/truncation 필드가 없다.
```

per-keypoint 가시성 flag 도 없다(§C 참조). 따라서 **가림·잘림 커버리지는 주장하지 않는다.**

---

## B. SPLIT INTEGRITY

### 실측 결과

```
positive 140 내부 동일 해시 중복        0
positive 140 stem 중복                  0
positive x negative 해시 교집합         0
positive x negative stem 교집합         0
negative 2,689 고유 해시                2,688  ★중복 1건
perceptual near-duplicate (pHash)       NOT MEASURED
```

★ **negative 셋에 완전 동일 해시 1쌍이 있다.** 2,689 중 고유 해시가 2,688 이다.
FP/image 같은 프레임 평균 지표에 미치는 영향은 1/2689 로 미미하지만, 기록해 둔다.

★ **pHash near-duplicate 는 재지 않았다.** 연속 촬영 프레임이 많은 셋이라
"해시가 다르지만 사실상 같은 장면" 이 존재할 개연성이 높다. 미측정이므로 주장하지 않는다.

### FT training overlap — 숨기지 않고 기록한다

```
발견   12장 (전부 eval_outside, 전부 DAY, NIGHT 0)
사유   eval_outside 는 별도 촬영이 아니라 capturepallet02/03/04/05/08 에서
       뽑아 모은 셋이다. 디렉토리명만 다르고 같은 프레임이라
       ★세션 디렉토리 비교만으로는 안 잡힌다.
영향   FT/adaptation 으로 real 을 학습한 모델의 DAY 지표는 낙관 편향.
       NIGHT 지표는 영향 없음.
대응   neg_eval_one.py 가 이 12장을 POS 에서 제외해 128 을 만든다.
       cf_real_eval.py 는 제외하지 않는다(140).
```

이것은 **provenance correction** 이지 데이터 결함이 아니다. 다만 이 사건이 드러낸 것은
**"디렉토리 = 세션" 가정이 이 데이터셋에서 성립하지 않는다**는 점이고, 그건 아래
capture-session overlap 판정에 그대로 영향을 준다.

### capture-session overlap

```
판정   PARTIAL
근거   eval_outside 가 다른 5개 capture 세션에서 뽑아 모은 셋임이 확인됐다.
       나머지 6개 set 이 서로 독립 세션인지는 **디렉토리명 외의 근거가 없다.**
       capture 일시·장비 metadata 가 라벨에 없어 세션 동일성을 독립 검증할 수 없다.
```

즉 **session-level separation 은 현재 보장되지 않는다.** final test 에서는 이 점이
반드시 해결돼야 한다(`FINAL_TEST_REQUIREMENTS.md`).

---

## C. ANNOTATION RELIABILITY

### 현재 확인된 것

```
전부 수동 annotation        gt_source = "manual"  140/140
prediction-assisted 아님    annotate.py 에 모델/torch/.pt 참조 0건
도구                        scripts/annotate/annotate.py (+ draw/pnp/io)
convention                  camera_dynamic_0123_v4, 9 keypoint
PnP 자기일관성              reproj_error_px  median 1.245 · p90 2.530 · max 4.481 px
```

### ★ reproj_error_px 를 annotation 정확도로 읽지 말 것

이 값은 **찍은 9점으로 PnP 를 풀고 다시 투영했을 때의 자기일관성**이다. annotator 가
일관되게 틀린 위치에 찍으면 이 값은 작게 나온다. **독립적인 annotation noise 측정이
아니다.**

### ★ 아직 수치화되지 않은 것

```
annotation error / noise floor      NOT MEASURED
inter-annotator agreement           NOT MEASURED
intra-annotator (blind re-anno)     NOT MEASURED
annotator 수 · 식별자               NOT RECORDED
annotation 작성/수정 이력           NOT RECORDED
GT-QA 판정 사유                     NOT RECORDED (frame_id 목록만 존재)
```

GT-QA 폴더의 `reviewed_gt` / `fixed_gt` / `logs` 는 **모두 0 files** 다. 21장을 어떤
기준으로 뺐는지 재구성할 수 있는 중간 산출물이 남아 있지 않다.

### 알려진 라벨 구조상 한계 2건

1. **per-keypoint 가시성 flag 가 없다.** `visibility` 는 object-level 스칼라이고
   140/140 전부 1 이다. 가려진 코너와 보이는 코너를 라벨에서 구분할 수 없다.
   → 가림 조건별 분석이 불가능하고, "보이는 점만 평가" 같은 프로토콜을 쓸 수 없다.

2. **`dimensions_m` 이 프레임마다 두 변종이다.** (1.1, 0.11, 1.3) 81장 /
   (1.3, 0.11, 1.1) 59장. 같은 물리 팔레트인데 W/D 가 프레임별로 스왑돼 라벨돼 있다.
   memory `evaluator-receives-gt-per-frame-axis-assignment` 가 이 문제를 이미
   `GT_DEPENDENT_AXIS_LEAK_PRESENT` 로 판정했다 — 평가가 GT 로부터 90° yaw 구분을
   넘겨받는다. ADD / translation / rotation 계열 지표가 이 영향권에 있다.

### 그래서 지금 할 수 있는 주장과 할 수 없는 주장

```
할 수 있다   "전부 사람이 직접 찍은 GT 이고, 예측 보조를 쓰지 않았다"
             "PnP 자기일관성이 median 1.2px 로 기하학적으로 모순이 없다"
             "positive/negative 사이 중복이 없다"

할 수 없다   "annotation 이 정확하다" / "dataset quality 가 검증됐다"
             "세션이 분리돼 있다"
             "가림·잘림 조건을 커버한다"
```

`ANNOTATION_RELIABILITY_PLAN.md` 가 위 빈칸을 메우는 최소 계획이다.
