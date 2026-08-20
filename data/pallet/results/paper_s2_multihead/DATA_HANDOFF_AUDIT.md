# DATA HANDOFF AUDIT — CORNER_LA_OBLIQUE_V1 / EDGE_HARD

training 0. 생성 머신 figure 를 믿지 않고 학습 머신에서 라벨을 다시 읽어 재현했다.

## 경로 (filesystem 에서 resolve)

```
BROAD 40K     data/pallet/training_data/paper_release/v2_prod40k_clean_merged   40,000
CORNER_LA     .../paper_release/oblique/corner_la_oblique_v1_y15_30.zip          2,500
              .../paper_release/oblique/corner_la_oblique_v1_y30_plus.zip        2,500
EDGE_HARD     .../paper_release/edge/edge_complement_v1_trunc_train.zip         10,000
              .../paper_release/edge/edge_complement_v1_trunc_dev.zip            1,000  ← frozen split 이미 존재
E3 ckpt       weights/paper_s2/paper_s2_multihead/screen_A1_CORNER_LINE_e3confirm25k_seed{1,2}/
              step_{06000,12000,18000,25000}.pth
source lock   E3 @18k
```

`paper_release/1..4` 는 빈 디렉터리다.

## CORNER_LA QA 재현

```
                     labels  rgb  mask_visible  mask_amodal  중복  결손
y15_30                2500  2500     2500          2500       0     0
y30_plus              2500  2500     2500          2500       0     0

elevation <8도 위반   0 / 0
|yaw| >=15 위반       0 / 0        범위 [15.00,29.99] / [30.00,45.00]
V_vis   y15_30  {4:257, 5:897, 6:584, 7:762}      README 와 일치
        y30_plus{4:224, 5:956, 6:669, 7:651}      README 와 일치
해상도  README quota 와 정확히 일치 (640x480 / 960x540 / 720x480 / 560x560)
truncation  42.3% / 43.2%   (README 42.7%, BROAD 같은 영역 43.2/46.8%)
pallet type 4종 균형 (564~678)
records.jsonl / index.csv 존재
```

## collision — HARD_BLOCK 없음

RGB 파일 서명(크기 + 앞 64KB blake2b)으로 대조:

```
NEW 5,000장 고유 서명 5,000 (내부 중복 0)
BROAD 40,000장 대조 → collision 0건
```

## ★ yaw 규약 — README 경고가 맞았다

README 가 `DIAGNOSTIC_YAW_CONVENTION_MATCH = UNVERIFIED` 라고 명시했고, 실제로
어긋났다.

```
데이터셋 정의   abs_frontal_yaw = 45 - facing_margin      범위 [0, 45]
이전 진단 유도  pose + camera-facing 0123 에서 유도        범위 [0, 165]
bin(<15 / 15-30 / >=30) 일치율                             51.25%
중앙 절대차 15.53도, 상관 +0.50
```

**데이터셋 정의를 정본으로 채택했다.** 근거: 그 정의로 BROAD 를 재계산하면 release
note 가 명시한 개수를 **정확히 재현**한다.

```
Y15_30    재계산 1120   README 1120   차이 0
Y30_PLUS  재계산 1116   README 1116   차이 0
```

→ 이전 `CORNER_LA_RENDER_TARGETS.md` 의 cell 경계는 폐기된 yaw 로 그어진 것이다.
그 문서의 target 장수는 이 규약으로 다시 계산해야 유효하다.

## 알려진 분포 차이 (결과 보기 전에 lock)

크기·거리·앙각·truncation 은 BROAD 의 같은 영역과 사실상 일치한다. **V_vis 만 다르다.**

```
            V_vis=4    5     6     7
NEW  Y15_30    257   897   584   762      V_vis=4 가 10.3%
40k  Y15_30    359   343   194   224      V_vis=4 가 32.1%
NEW  Y30_PLUS  224   956   669   651      V_vis=4 가  9.0%
40k  Y30_PLUS  400   334   208   174      V_vis=4 가 35.8%
```

(40k 수치는 내 index 에서 독립 재계산한 것이고 README 와 일치한다.)

따라서 C1 은 **engineering effect** 이고 low-angle/oblique 노출 자체의 인과는
`C1_VCTRL` 에서만 분리 가능하다 — 이번에는 C1 이 gate 를 통과하지 못해 실행하지 않았다.

## EDGE_HARD

`train` / `dev` zip 이 이미 분리돼 있어 별도 holdout 을 만들 필요가 없다.
이번 screen 에서는 **개봉하지 않았다**(C1 PASS 조건 미충족). 압축 상태로 둔다.
