# Manual review requirements

작성 2026-09-02.  **사람이 실제로 해야 하는 일만** 적는다.
자동으로 닫을 수 있는 blocker 는 먼저 닫거나, 닫을 수 없음을 측정으로 확인한 뒤에만
사람에게 넘긴다.

새 이미지는 수집하지 않는다.  319 장의 keypoint 좌표를 다시 찍지도 않는다.

## 요약

```text
Task                          Frames   Human action           Blocks what              판정
──────────────────────────────────────────────────────────────────────────────────────────────
Daytime visibility               70    119 kp 확인만           M2 strict keypoint      REQUIRED
Pose signed-axis                146    axis 확인               6D pose metrics         NOT_NEEDED_NOW
Wood symmetry                     —    geometry convention     Wood 6D pose            OPTIONAL
Annotation reliability           TBD   blind reannotation      noise-floor claim       RECOMMENDED
New image collection               0   none                    아무것도 막지 않음        NOT_NEEDED
```

---

## 1. Daytime visibility — REQUIRED

가장 값이 큰 작업이고 범위가 좁다.

```text
frames                       70
keypoints                   630
visibility unknown          630
  그중 화면 안 (실제 작업)   609
  그중 화면 밖                21   visibility 0 이 맞다 — 건드리지 않는다
원인 분류                     A 70 / B 0 / C 0 / D 0
```

`A = xy 는 있고 visibility 만 unknown`.  **좌표를 다시 찍을 일이 전혀 없다.**
B(좌표 없음)·C(schema)·D(파일 없음)는 0 건이다 — 추측이 아니라 세었다.

### 자동 분류로 630 → 119 로 줄였다 (2026-09-02)

630 개를 전부 사람이 보게 하지 않는다.  기하로 결정되는 것은 기하로 정했다.

```text
AUTO_TRUNCATED                  21   in_frame == False.  GT v2 기존 규칙
                                     기존 outside_keypoints 선언과 불일치 0
AUTO_SELF_OCCLUDED              95   back-face culling, signed-axis 두 후보 일치
AUTO_CENTROID_OCCLUDED          70   repo 규약 (신규 어노 146 장 전부 visibility=1)
SELF_VISIBLE_CANDIDATE         325   두 후보 모두 visible + depth 이상 없음
────────────────────────────────────────────────────────────────────────
자동 확정                      511
EXTERNAL_OCCLUSION_CANDIDATE   119   ← 사람이 볼 것 (18.9%)
```

`SELF_VISIBILITY_DISAGREES` 는 0 이다 — signed-axis 미해결이 self-visibility 판정을
흔들지 않았다.  camera-facing permutation 이 두 후보를 같은 인덱스로 맞춰 주기 때문이다.

depth 신호는 사람이 매긴 프레임 태그와 잘 맞는다.  ext 후보가 있는 64 프레임 중
62 개가 `occlusion=medium` 이다 (Daytime 70 중 medium 65).  임계값이 헛돌지 않는다는
교차 검증이다.  그래도 자동 확정하지 않는다 — depth 노이즈와 실제 가림을 센서만으로
가르지 않는다.

```text
queue   data/evaluation/pallet_eval_v1/review/DAYTIME_OCCLUSION_REVIEW_QUEUE.csv
report  _docs/paper/DAYTIME_OCCLUSION_AUTO_CLASSIFICATION.md
```

이게 막고 있는 것: M2 의 Daytime strict keypoint 열.  현재는 all-annotated 진단으로
대신 채워져 있고, Nighttime 과 같은 정의로 비교할 수 없다.

```text
queue   data/evaluation/pallet_eval_v1/review/DAYTIME_VISIBILITY_REVIEW_QUEUE.csv
audit   _docs/paper/DAYTIME_VISIBILITY_AUDIT.md
lock    _docs/paper/DAYTIME_VISIBILITY_REVIEW_LOCK.json
```

허용 편집은 visibility 뿐이고, xy·pose·bbox·intrinsics·object type 변경은 저장 단계에서
막는다.  모델 예측은 화면에 띄우지 않는다.

---

## 2. Pose signed-axis — NOT_NEEDED_NOW (하지 말 것)

146 장을 열어 axis 를 확인하는 작업은 **지금 하면 낭비다.**  해도 pose 가 열리지 않는다.

이유는 prediction-only W/D selector 가 게이트를 한참 못 넘기 때문이다.
저장된 진단이 옛 checkpoint 기준이라, 현재 모델로 **다시 재서** 확인했다.

```text
population DEV_POS140 (사전등록, 바꿀 수 없음)   gate  overall >= 0.95, night >= 0.90

                        overall     day     night
OLD_ROOT_G38 (기록)      0.5929   0.6250   0.4643
R0                       0.6500   0.6429   0.6786
R5_PROPOSED              0.5929   0.5357   0.8214
```

최고가 0.65 다.  0.95 와 격차가 크고, 이건 프레임 라벨이 아니라 **알고리즘 문제**다.
signed-axis 를 사람이 확인해도 selector 가 여전히 틀리므로 pose 는 BLOCKED 로 남는다.

```text
DEFER_MANUAL_REVIEW = true
```

부수 관찰: R5 는 night selector 를 0.679 -> 0.821 로 올리고 day 를 0.643 -> 0.536 으로
내린다.  self-training 이 야간 쪽에 치우쳐 있다는 다른 지표들과 방향이 같다.

먼저 해야 할 것은 selector 알고리즘이다.  그게 게이트를 넘은 뒤에 사람 확인이
의미를 갖는다.

---

## 3. Wood symmetry — OPTIONAL

```text
registry symmetry_status        UNREVIEWED
symmetry_contract               없음
selector                        NOT_RUN
canonical migration             NOT_PASS
```

물체 하나에 대한 **convention 결정**이지 프레임 작업이 아니다.  사람이 한 번 정하면
125 장 전체에 적용된다.  다만 plastic selector 가 막혀 있는 한 wood 를 풀어도
ALL pose 는 열리지 않으므로 우선순위가 낮다.

3d-expert 위임 후보다.

---

## 4. Annotation reliability — RECOMMENDED

noise floor 를 모르면 "corner 4.420 -> 4.180" 이 의미 있는 차이인지 말할 수 없다.
현재 Proposed 대 baseline 의 corner 차이가 0.24 px 수준이라 더욱 그렇다.

primary GT 를 덮어쓰지 않고 별도 blind 사본으로 받는다.

```text
data/evaluation/pallet_eval_v1/reliability/blind_A/
data/evaluation/pallet_eval_v1/reliability/blind_B/
```

sampling 규칙은 새로 만들지 않는다 — 기존 계획서를 먼저 따른다.

---

## 5. 새 이미지 수집 — NOT_NEEDED

```text
PAPER_EVAL positive        319   target 300 이상
Daytime / Nighttime        70 / 50   두 gate 모두 READY
Plastic / Wood             194 / 125  target 180 / 120 이상
NEW_IMAGE_COLLECTION_REQUIRED = false
```

새 300 장 촬영을 제안하지 않는다.  독립 confirmation set 은
`OPTIONAL_STRONGER_VALIDATION` 으로만 남긴다 — 지금 논문을 막지 않는다.

---

## 순서

```text
1. Daytime visibility review        사람 1회, 119 keypoint (630 중 511 은 자동 확정)
2. (자동) selector 알고리즘          pose 를 여는 유일한 길
3. wood symmetry convention          2 가 풀린 뒤에 의미
4. annotation reliability            언제든 병행 가능
```

2 가 열리기 전에는 146 장 pose review 를 요청하지 않는다.
