# DATASET MASTER INVENTORY

training 0 / render 0 / architecture 변경 0. filesystem 전수 검색으로 작성했고
이름이 아니라 라벨의 실제 contract 로 역할을 정했다.

```
dataset                 N       corner  line   gate      역할                       main
BROAD_40K              40,000    yes    yes   100%      MAIN_TRAIN_POSITIVE        YES
CORNER_LA_Y30_PLUS      2,500    yes    yes   100%      CORNER_TARGET_ABLATION     NO
CORNER_LA_Y15_30        2,500    yes    yes   100%      EASY_CONTROL_ABLATION      NO
CORNER_LA_FRONTAL           0     -      -      -       PRESERVE_UNUSED            NO
EDGE_HARD_TRAIN        10,000    NO     yes   FALSE     LINE_HARD_ABLATION         NO
EDGE_HARD_DEV           1,000    NO     yes   FALSE     EVAL_ONLY                  NO
EDGE_HARD_TRUNC_UNTOUCHED 1,000  NO     yes   FALSE     EVAL_ONLY                  NO
EDGE_HARD_CLEAN_UNTOUCHED 1,000  yes    yes   TRUE      EVAL_ONLY                  NO
NEGATIVE_SYNTH_V1           0   negative              NEGATIVE_TRAIN_CANDIDATE   PENDING
LEGACY (v8/aug/pl 계열)     -     -      -      -       ARCHIVE                    NO
```

## 이름이 아니라 contract 로 확인한 것

### EDGE_HARD 는 corner 학습 대상이 아니다

라벨을 직접 열어보면:

```
safety_gates  G1_Vvis>=4: False   all_pass: False
V_actual/V_vis 3 / 3
pnp_conditioning  {}   (비어 있음)
```

**설계상 그렇다.** README: "기존 4만장이 키포인트 부족(visible_kp < 4)으로 버렸던
프레임만 모은 여집합". TRUNC 은 `visible_kp` 가 1~3 이다. 즉 corner 감독을 붙이면
4개 미만 코너로 pose 를 가르치게 된다. **line branch 전용**이 맞다.

코너 손실 원인도 문서화돼 있다 — truncation 72.7% / self-occlusion 22.6% / 둘 다 4.7%.
"오직 truncation" 이라고 인용하면 부정확하다.

### EDGE_HARD 의 untouched holdout — 도착 완료 (2026-08-19/20)

> **정정 (2026-08-20).** 아래 절은 08-18 시점 기록이다. untouched 는 clean/trunc
> 양쪽 모두 도착했고 `DELIVERY_MANIFEST_20260820.json` 의 sha256 과 일치한다
> (`data_audit/HANDOFF_VERIFY_20260820.txt`). 원문은 이력으로 남긴다.

README 는 `train 10,000 / dev 1,000 / untouched 1,000` 쌍을 선언한다. 전달된 zip 두 개를
파일명 범위로 확인하면 **전역 인덱스가 0~10999 에서 끊긴다.**

```
edge_complement_v1_trunc_train.zip   f0000  ~ f9999    10,000   pair_index idx 0~9999      split=train
edge_complement_v1_trunc_dev.zip     f10000 ~ f10999    1,000   pair_index idx 10000~10999 split=dev
                                     f11000 ~ f11999    없음    ← untouched 1,000
```

세 갈래로 확인했다 — 각 zip 의 `pair_index.csv` 가 자기 split 행만 담고(untouched 행 0),
`records.jsonl` 의 `split` 필드도 `train`/`dev` 뿐이며, `edge/` 폴더에 세 번째 zip 이 없다.

### CLEAN train/dev 는 **의도적 미배포**다 (전달 실패 아님)

> **정정 (2026-08-20).** `edge/README_edge_complement_v1.txt` 원문:
> `TRUNC ... <- 학습 대상` / `CLEAN ... <- 대조군 (감사·figure 용)`.
> 배포 대상은 TRUNC 3 split + untouched 쌍의 CLEAN 쪽뿐이며,
> clean_train 10,000 / clean_dev 1,000 은 처음부터 배포 계획에 없다.
> 이전 판의 '전달되지 않았다' 표현은 이 문서에서 폐기한다.

#### (이력) 08-18 시점 서술

합성 머신에는 존재한다(사용자 확인):

```
data/pallet/release/datasets/edge_complement_v1_clean_untouched.zip   655 MB   1,000 프레임
data/pallet/release/datasets/edge_complement_v1_trunc_untouched.zip   649 MB   1,000 프레임
                                                          합 1,000 쌍 / 2,000 프레임
```

이 학습 머신에는 `data/pallet/release/` 아래 `attribution/` 만 있고 `datasets/` 가 없다.
전체 파일시스템에서 `*untouched*.zip` 과 `edge_complement*` 를 찾아도 0 건이다.

⚠ **초기 census 의 구멍**: `data/pallet/training_data/` 만 훑고 `data/pallet/release/` 를
보지 않았다. 경로를 지적받고 확인했다.

⚠ **규모 정정**: 배포된 train/dev zip 이 쌍의 TRUNC 쪽만 담고 있어 인덱스 산술로는
1,000 장으로 보였으나, 실제 untouched 는 **clean/trunc 두 벌 = 2,000 프레임**이다.

### clean 쪽은 corner 평가에 쓸 수 있다

이전 판에서 "EDGE untouched 는 line-hard 용" 이라고 썼는데 그것은 TRUNC 쪽 이야기다.

```
trunc_untouched   visible_kp 1~3   게이트 탈락   line-hard 평가용
clean_untouched   visible_kp >=4   게이트 통과   corner PnP 평가 가능 ★
```

현재 corner 쪽 평가 병목은 `LA_HARD n=51` 이다. `clean_untouched` 1,000 장의
elevation × canonical yaw × V_vis 분포에 따라 그 병목을 실제로 완화할 수 있다 —
받은 뒤 분포를 재서 판정한다.

### 배포된 것은 쌍의 TRUNC 쪽뿐이다

`pair_index.csv` 에 `clean_visible_kp` 와 `source_usable_id` 열이 있다. 즉 CLEAN 짝은
이미지로 배포되지 않고 원본 pool 참조로만 들어 있다. 그래서 라벨 수가 쌍 수와 같다.

### CORNER_LA 는 canonical 로 재분류해도 버킷이 그대로다

```
              canonical yaw
old bucket   <15   15~30   >=30
Y15_30         0    2500      0
Y30_PLUS       0       0   2500      누출 0
```

zip 이름을 믿지 않고 전수 재계산한 결과다. 다만 canonical failure map 기준으로
`Y15_30` 이 겨냥한 cell 은 near-baseline(9.62px, x1.14)이라 **control 성격**이다.

## collision

```
BROAD × CORNER_LA     rgb size+64KB blake2b, 40,000 × 5,000     collision 0
CORNER_LA 내부        고유 서명 5,000 / 5,000                    중복 0
BROAD × EDGE_HARD     생성 provenance                            collision 0
                      BROAD = v2_prod40k,  EDGE = v2_edgecomp_s9601~9608
                      EDGE 는 BROAD 게이트가 버린 프레임의 여집합이라 구조적으로 분리
train × eval          frame id                                   collision 0
```

## 정리해야 할 것 (기능 영향 없음)

```
paper_release/1, 2, 3, 4        빈 디렉터리
paper_release/data/pallet/...   빈 중첩 경로 (파일 0, 24K) — 과거 잘못된 unzip 잔재
```
