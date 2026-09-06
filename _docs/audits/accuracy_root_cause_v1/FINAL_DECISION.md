# FINAL_DECISION — 실제 정확도를 막는 것은 무엇이고, 다음 예산을 어디에 쓰는가

```
CURRENT_HEAD = 2e5ec0e559b1c8a5e9b3a8f425944ae90520d4dc

PRIMARY_ROOT_CAUSE   = 저앙각(edge-on) 레짐에서 보이는 코너의 위치추정 실패
                       (Layer D. R0 on PAPER_EVAL 319 = MISLOCATED 83 / 319, 26.0%.
                        앙각 <8도 실패 48~52% 대 >=30도 14%, 투영 크기와 교란 아님)
SECONDARY_ROOT_CAUSE = 배포 물체의 대칭 계약 부재 (과제 트랙 한정, GPU 예산 0)

GT_STATUS                  = 2D 는 신뢰 가능 / 6D yaw 90도 분기는 절반이 근거 부족
PHYSICAL_REFERENCE_STATUS  = 독립 계측 없음. depth 로 거리 스케일만 교차 확인됨
POINT_MODEL_STATUS         = 병목. 단 검출·랭킹은 이미 천장
RANKING_STATUS             = 병목 아님 (IoU50 매칭 97.5%, AUROC 0.9921)
PNP_STATUS                 = PNP_NOT_PRIMARY_LEVER (평가측·학습측 양쪽 닫힘)
LINE_EVIDENCE_STATUS       = 측정 불가 (target visibility 미정의) → 학습 STOP
EXISTING_DIRECT_HOUGH_STATUS = 누적 가설을 시험한 적 없음 (구현 family 만 반증됨)
SOURCE_DATA_STATUS         = 비율은 맞고 결합 다양성이 없다
                             (elev<8 AND thin 셀 유효자산수 1.01 / 4)
REAL_DATA_STATUS           = 가장 큰 실증된 레버. unseen 세션 코너 -33.5%,
                             이득이 저앙각에 집중. new-shape 는 미측정

NEXT_EXPERIMENT = real supervision 의 **구성(composition) ablation** —
                  같은 라벨 예산으로 저앙각(<8도)만 / 중앙각(8-15도)만 학습해
                  두 층 held-out 세션에 각각 전이시키는 2x2 행렬.
                  촬영단위(session-level) split, 정사각 배포 물체 단일.
                  ★rev2 — 라벨 감사로 3-arm 설계가 불가능함이 확정돼 2-arm 으로 축소했다.
WHY             = 저장소 93개 실험 중 unseen 세션의 위치추정을 움직인 유일한 개입이
                  real supervision 이고(-33.5%), 그 이득이 실패 레짐(<15도)에 정확히
                  몰린다(<8도 -36.4%, >=30도에서는 오히려 악화). 그러나 그것이
                  **레짐 커버리지** 때문인지 **일반적인 sim2real appearance** 때문인지
                  구분된 적이 없다. 이 구분이 다음 두 갈래를 동시에 판정한다 —
                  갈리면 "무엇을 더 찍어야 하는가" 가 확정되고,
                  안 갈리면 저앙각 자산 다양성 재렌더(F)의 전제도 함께 무너진다.
EXPECTED_RESULT = 저앙각 arm 이 고앙각 arm 을 held-out 세션 앙각<15도 층에서
                  코너 중앙값 기준 20% 이상 앞선다 (관측된 효과는 36%)
STOP_RULE       = 세 arm 이 held-out 세션에서 코너 중앙값 10% 이내로 붙으면
                  레짐 가설 기각. 그 즉시 F(저앙각 재렌더)를 후보에서 내리고
                  D(capacity)로 넘어간다. 추가 arm 을 만들지 않는다.

DO_NOT_RUN = line / Hough / edge-voting 학습 · PnP solver 탐색 ·
             self-training 재실행 · hard negative 추가 · synthetic **장수** 증가 ·
             threshold sweep · 새 loss 항
```

---

## 1. 어떻게 여기까지 왔는가 — 배제의 순서

지시문 §24 결정 트리를 그대로 밟았다. 모든 수치는 **DEVELOPMENT / POST-HOC DIAGNOSTIC**
이다. 저장소에 held-out 모집단이 남아 있지 않다(`LIMITATIONS.md` §1·§6·§8b).

**1. GT semantics 문제인가 → 2D 는 아니다, 6D 는 그렇다** [확인]

`manual_kps` 의 45.8% 가 사람 클릭이 아니라 채택 pose 의 투영이다
(`annotate_io.py:518-521`). 그럼에도 오차를 설명하지 못한다 —
외삽 p50 7.38 px vs 클릭 6.38 px 이고, 가림 코너 **안에서는 클릭이 더 나쁘다**
(18.91 vs 14.73 px). 프레임 최악 오차의 75% 가 보이는 코너에서 난다.
가림을 통째로 빼도 실패율은 28.6% → 22.5% 로 6.1 %p 만 준다.
→ `GT_REPAIR_FIRST` 는 2D 정확도에 대해 **기각**.

단 6D 는 다르다. `dimensions_m` 의 W/D(=yaw 90도) 배정을 클릭 점만으로 검증하면
margin p50 1.48 px, 130장 중 51장이 1 px 미만 차이다. 외삽 점을 넣으면 9.14 px 로
부풀지만 그건 순환이다.

★**2026-09-06 사람 리뷰로 판정됐다** (`GT_REVIEW_RESULT.md`, 54 프레임 전수 응답):
축 배정은 **옳다** — 판정한 39건 중 30건(76.9%) 저장본 일치, 이항검정 p = 5.3e-4,
확신도 4~5 에서 **17/17 = 100%**. 순환 우려는 "GT 가 틀렸다" 가 아니라
**"증거량이 부족하다"** 였음이 확인됐다(클릭 margin > 5px 인 10건은 9/9 = 100%).
다만 **20.4%(11/54)는 사람도 못 가른다** — 그 구간의 6D rotation 은 인용 금지.

★**2·3차 리뷰로 두 가지가 더 확정됐다** (`GT_REVIEW_RESULT_PHASE23.md`):
외삽 코너 71개 중 **"허공"(가상) 판정은 0개**다(물리 24 · 애매 47) — PnP 로 채운 점이
물체를 벗어난 경우는 관측되지 않았다. 단 66%가 검증 불가라 "틀렸다"를 배제할 뿐이다.
그리고 **재현성** — 20장 재리뷰에서 일치 17/20(85%), 양 회차 모두 A/B 를 고른 12건은
**12/12 = 100%**, **A→B 뒤집힘 0건**. 불일치는 전부 "판정 → 모르겠음" 이다.
(같은 날 같은 세션이라 85%는 상한이고 N=20 이라 구간이 넓다.)

★그리고 **새 결함이 나왔다** — 4건에서 리뷰어가 "두 가설이 **모두** 틀렸다" 고 적었다.
전부 `n_click = 4` 이고 저장 `reproj_error_px` 는 0.87~2.24 px 로 정상 범위 안이다.
PnP 는 4 점이면 정확결정이라 잔차가 **pose 를 정한 그 점들로 계산되어 검증력이 없다.**
정본 140장 중 `n_click <= 4` 는 23장(16.4%)이다.
→ `reproj_error_px` 를 `n_click` 없이 품질 지표로 쓰면 안 되고, 그 23장은 재클릭 대상이다.

**2. 기하 계약 문제인가 → 과제 트랙은 그렇다, 논문 트랙은 아니다** [확인]

배포 물체 `plastic_standard_110x110x15` 는 `symmetry_status: UNREVIEWED`,
`symmetry_contract: null` 인데 identity-only 열이 보고되고 있다. 유일하게 존재하는
계약은 다른 물체(110x130)의 것이고, 그 계약이 스스로 밝힌 90도 배제 근거는
"canonical X and Z extents differ" 다 — 정사각 배포 물체엔 성립하지 않는다.
그 결과 F1 모델의 fixed-index 실패 62/155 가 전부 270도 순열이고 C4 오차 median 1.97 px,
true collapse 0장이다. **위치는 맞는데 지표가 위치 실패로 셌다.**

그런데 논문 트랙은 정반대다 — 계약 대칭 {0,180}이 구제하는 프레임이 **0장**이고,
8! 자유 배정도 98장 중 13장만 구제하며 p50 을 4% 개선한다.
→ `GEOMETRY_CONTRACT_FIRST` 는 **과제/배포 트랙에만** 적용된다. GPU 예산이 들지 않으므로
다음 실험과 경쟁하지 않는다. 먼저 고친다.

**3. 랭킹 문제인가 → 아니다** [확인]
`candidate_count == 0` 인 프레임 0장, IoU50 매칭 311/319 (97.5%), AUROC 0.9921.

**4. 위치가 gross 하게 틀리는가 → 그렇다. 여기다** [확인]
MISLOCATED 83/319 (26.0%). 어떤 순열로도 구제되지 않는다.

**5. PnP 에서 증폭되는가 → 아니다** [확인]
평가측 solver 교체(+1.48% 악화), 학습측 DiffPnP(real 7~23% 악화), subset PnP 계산 완료.

**6. line evidence 가 hard case 에 강하게 존재하는가 → 측정 불가** [확인]
GT line 이 PHYSICAL_VISIBLE / OCCLUDED / VIRTUAL 을 구분하지 못한다.
게다가 예측 코너로 만든 기하 자기일관성 신호의 실패 AUROC 가 0.44~0.65 다 —
**R0 가 틀릴 때도 코너들끼리는 정합한 육면체를 이룬다.** 구조 붕괴형 실패가 아니므로
구조를 더 잘 보는 표현이 이 실패를 고칠 것이라는 전제 자체가 약하다.

**6a. 기존 hybrid 가 병목 층에 닿기는 하는가 → 아니다** [확인]
`HYBRID_POINT_LINE_PER_FRAME.csv` 의 B1(point) 대 P1(+Direct-Hough) 짝지은 비교에서
**corner 오차가 93/93 프레임 전부 소수점까지 동일**하다(R -0.70도, t -0.0017 m, IoU +0.032 는 다르다).
즉 이 계열은 pose 해만 바꾸고 2D keypoint 층은 입력 그대로 통과시킨다.
병목이 2D 코너인 이상 **구조적으로 닿을 수 없다.**

**6b. 해상도/관측성 한계인가 → 아니다** [확인]
저앙각은 "얇게 투영된다" 와 같은 변수다(앙각 vs bbox 단축 Spearman 0.850, AUROC 0.699 vs 0.692).
가장 얇은 4분위는 단축 중앙값 34.9 px 이고 최대 코너 오차가 그 **0.546 배**다.
그럼에도 같은 해상도·같은 레시피에서 real FT 가 그 구간을 36% 개선했다 —
**해상도가 한계라면 데이터만 바꿔 36% 가 줄 수 없다.** 학습되지 않은 것이지 못 보는 것이 아니다.

**6c. 오검출은 무엇에 붙는가 → 반복 슬랫 텍스처** [확인, 79건 전수 육안]
배포 임계(conf 0.4) 이상 오검출 79건을 시트로 만들어 **전부 눈으로 봤다**
(`HARD_NEGATIVE_REVIEW.md`). 카테고리는 보고 나서 붙였다.

```
SLATTED_WOODEN_BENCH        49  62.0%   RIBBED_TRANSLUCENT_LID   23  29.1%
LOUVRE_VENT_GRILLE           3   3.8%   그 외(표지판·화면·키보드·바닥) 4   5.1%
```

**91%(72/79)가 "평면 위에 나란한 요소가 반복되는 것"** 이고, 최고 오검출은
**컵뚜껑 conf 0.905** 다. 팔레트를 닮은 물건(랙·상자·지게차)은 **하나도 없다** —
크기·맥락·3D 구조가 전부 다르고 슬랫 텍스처만 같다.

[추정] R0 는 팔레트를 3D 구조가 아니라 **상판의 반복 슬랫 텍스처**로 찾는 것으로 보인다.
그렇다면 저앙각에서 상판이 얇은 띠로 압축될 때(단축 34.9 px) 주 단서가 무너진다 —
저앙각 실패와 이 오검출 패턴이 **같은 표현을 두 방향에서** 보여주는 셈이다.
★단 이건 가설이고, ranking 축 판정(`NEGATIVE_INTERVENTION_TOUCHES_LOCALISATION = NO`)은
바뀌지 않는다. "이 벤치들을 negative 로 더 넣자" 는 결론이 **아니다** — 이미 해봤고 해로웠다.

**7. 필요한 hard condition 이 synthetic 에 없는가 → 비율이 아니라 결합이 없다** [확인]
저앙각 marginal 은 이미 맞다 (real `<8도` 28.2% vs 학습셋 29.0%). 그런데 그 16,253장의
87% 가 `scene.usd` 한 mesh 이고, `elev<8 AND 두께<=0.0923` 셀의 유효자산수는
**1.01 / 4** 다. 4자산 균등 노출은 앙각 >=30도 구간에만 있다.
→ `TARGETED_DATA_INTERVENTION` 후보는 **장수가 아니라 저앙각 셀의 자산 다양성**이다.

---

## 2. 무엇이 병목인지 — 한 장의 표

R0, `PAPER_EVAL_ALL_POS` 319, role = DEV. [확인]

```
Layer                     크기        근거
A 검출                    8 / 319     IoU50 미매칭. 후보 0인 프레임 0장
B 후보 선택               작음         후보>1 105장, AUROC 0.9921
C role / 축 순열         15 / 319     계약 대칭 구제 0장, 8! 구제 13/98
D 위치추정               83 / 319     ★ 어떤 순열로도 구제 안 됨
E PnP 증폭               없음         solver 양쪽 REJECT
F GT 모호성              2D 0 / 6D 큼  외삽-클릭 차이 1 px, W/D margin 51/130 < 1px
```

그 83장이 어디에 있는가 — 앙각이 지배한다 (메인 세션 독립 재현, 교란 분리) [확인]:

```
앙각(도)     N    NME 실패   px>25 실패   같은 크기 층 안에서도 유지되는가
<3          27     48.1%      40.7%      소 58.8% / 대 30.0%
3-8         63     52.4%      49.2%      소 53.8% / 대 50.0%
8-15        61     32.8%      29.5%      소 40.0% / 대 23.1%
15-30       83     16.9%      18.1%      소 17.0% / 대 16.7%
>=30        85     14.1%      27.1%      소  9.5% / 대 15.6%
```

야간은 앙각을 통제한 뒤에도 남는 **독립 축**이다 —
DAY/<15도 37.4% vs NIGHT/<15도 55.8%, DAY/>=15도 11.6% vs NIGHT/>=15도 18.5%.

★**같은 구조가 6D 에서도 재현된다** (`POSE_BY_CONDITION.md`, 새 추론 0회,
기존 MAIN.ALL 을 기계정밀도로 재현해 집계 경로 동일 확인) [확인]:

```
R0        n    axis↑   R med↓   t cm↓    IoU3D↑
Clean    184   0.853    1.741    4.643   0.6670
Occlusion 135  0.607    3.122   11.436   0.5244
High      57   0.895    2.685    4.204   0.6186
Low      122   0.623    3.002   10.811   0.5593
Far       59   0.644    2.685   15.928   0.4350
```

가림에서 axis 0.853 → 0.607 · translation 2.5배, 저앙각에서 axis 0.895 → 0.623 ·
translation 2.6배다. `LIMITATIONS.md` §3 의 "축 선택기 0.59~0.65" 가
**저앙각·가림 구간에 몰려 있다**는 것이 여기서 처음 드러난다.
그리고 Far 는 **2D 로는 최고(3.8 px)인데 6D 로는 최악(15.9 cm)** 이다 — 투영 크기 때문이다.

★단 Occlusion·Low 행의 axis 하락은 **모델 오차와 GT 불확실성이 섞인 값**이다.
사람 리뷰에서 축 가설을 못 가른 20.4% 가 주로 그 구간이다(`GT_REVIEW_RESULT.md`).

real 앙각 분포: p50 16.5도, `<8도` 28.2%, `<15도` 47.3%.
즉 **배포 프레임의 절반이 실패율 3배인 구간에 있다.**

---

## 3. 왜 real supervision 인가 — 결정적 근거

[확인] `_docs/history/2026-08-20.md` 원문. FT 학습 세션(night01~07 · pallet02/03/04/05/08 ·
forklift_20260528)과 겹치지 않고 인접 non-eval 53장까지 제외해 프레임 겹침 0 을 문서화한
SEALED 105:

```
                 det    corner  R med  ADD-S   IoU3D
yolo26n_synth  0.838    10.51   2.90   0.12    0.57
yolo26n_ft     0.971     7.63   3.08   0.12    0.57
yolo26m_ft     0.952     6.80   2.78   0.10    0.66
```

[확인] 메인 세션 직접 계산(`ft_by_elevation.py`) — 이득이 실패 레짐에 정확히 꽂힌다.
정본 140장 중 세 모델 공통 검출, **SEALED 세션만** (73장):

```
앙각     N   synth   n_ft   m_ft   n_ft개선   synth>25   n_ft>25
<8      49   9.22   5.86   5.49    36.4%     53.1%     28.6%
8-15    24  11.49   8.49   7.87    26.1%     50.0%     37.5%
전체     73   9.62   6.40   6.53    33.5%
```

그리고 암기가 아니다 — 학습 세션과 겹칠 수 있는 나머지 51장에서는 이득이 **더 작다**(25.8%).

[확인] 짝지은 비교로 다시 재고 구간을 붙였다(`UNCERTAINTY.md` §2) — 세션 클러스터
95% CI 가 **0을 배제한다**: <8도 +3.33 px [+2.50, +4.69], 전체 +3.23 px [+2.72, +3.71],
프레임의 83.6% 가 개선, **4개 세션 전부 같은 방향**(+0.97 ~ +4.19).

★ **철회** — 이 문서의 초판은 ">=30도에서 이득이 뒤집힌다(gross 11.8% → 29.4%)" 고 썼다.
지지되지 않는다. 그 표본은 N=17 이고 **전부 한 세션(eval_cad)** 이며, 짝지은 중앙차는
오히려 +0.81 px 로 방향이 반대이고 프레임 CI [-1.54, +1.81] 이 0 을 포함한다.
정정된 진술: **고앙각에서 real FT 의 이득은 확인되지 않는다** — "없다" 도 "악화된다" 도
이 데이터로는 말할 수 없다. 상세 `UNCERTAINTY.md` §3.

→ real supervision 이 채운 것은 일반적인 성능이 아니라 **저앙각 레짐 그 자체**로 보인다.
   ★ 이 문장이 [추정]이라는 것이 다음 실험의 존재 이유다.

---

## 4. 후보별 점수 (§26)

증거·헤드룸·비용·중복위험·배포/논문 관련성. 5점 만점.

```
후보                        증거  헤드룸  구현  학습  중복위험  DEV과적합  배포  논문  종합
A GT/annotation 수정          1     1      3     -      1        2        1     2   기각
B 기하 계약 수정 (과제)        5     -      5     0      1        1        5     2   ★선행(GPU 0)
C inference/PnP/selection     2     2      4     0      2        3        2     2   약함
D capacity (matched medium)   3     3      3     1      1        3        4     3   2순위
E line/edge/voting            1     ?      1     1      3        2        1     4   STOP
F 저앙각 x 자산 다양성 재렌더   3     3      1     1      4        2        3     5   조건부
G hard negative               1     1      3     2      5        2        2     1   기각
H real supervision 구성 ablation 5   4      4     4      1        2        5     3   ★1순위
I 중단                        -     -      -     -      -         -        -     -   해당없음
```

- **B 는 GPU 0** 이라 "다음 실험" 과 경쟁하지 않는다. 먼저 한다.
- **F 의 중복위험 4** 는 근거가 있다 — "synthetic 목적함수는 내려가는데 real 전이 실패" 가
  PAPER_S2 6연속을 포함해 반복됐고, 게다가 F 를 지지하던 유일한 정량 근거
  (`GENERIC_SCALE_EFFECT = STRONG`, A42 10K vs G38 38K)가 이번 감사에서 **무효화**됐다.
  A42 는 패딩 안 된 이미지에 패딩 기준 라벨을 붙인 run 이고, 그 결함만으로 코너가
  중앙값 40.1 px 어긋난다(보고된 53.46 px 와 같은 자릿수). 망가진 기준선과의 비교였다.
- **H 의 학습비용 4** — real FT 는 짧고 데이터가 이미 라벨돼 있다.
- **I 는 해당 없음.** 개선 근거가 있다(unseen 세션 -33.5%).

---

## 5. 다음 실험 — 사전등록 초안

정식 `METHOD_LOCK` 은 착수 직전에 별도로 얼린다. 여기 있는 것은 그 초안이다.

```
question        real supervision 의 이득은 (a) 저앙각 레짐 커버리지 때문인가
                (b) 일반적인 sim2real appearance 때문인가
single changed axis   학습에 넣는 real 프레임의 **앙각 구성**. 장수·recipe·seed 고정
baseline        R0 (yolo26n, 합성만) + FT 없음
arms            L  : 저앙각(<15도) real 프레임만
                H  : 고앙각(>=15도) real 프레임만
                M  : real 분포대로 (층화 표집)
                세 arm 의 프레임 **수가 같아야 한다**. 부족한 쪽에 맞춰 절단한다.
training pop    라벨된 real. **촬영단위 split** (interleave 금지 — 그게 0.98 을 만들었다)
eval pop        held-out 세션. 앙각 층(<8 / 8-15 / >=15)별로 나눠 보고
seed            3 (line 효과가 아니라 데이터 효과이므로 seed 3 이면 충분.
                단 ultralytics seed 가 dataloader 에 안 가는 함정 확인 필요)
primary metric  held-out 세션 앙각<15도 층의 코너 오차 중앙값 (원본 픽셀)
secondary       같은 층의 gross(>25px) 비율 · NME · 검출 coverage · 앙각>=15 층(악화 감시)
success         L − H 의 **짝지은** 차이가 세션 클러스터 95% CI 로 0 을 배제하고
                부호가 L 우세  `[추정][미검증]`
failure         CI 가 0 을 포함하고 점추정 차이가 GT repeatability 이하
                → 레짐 가설 기각
stop rule       failure 판정 시 F 를 후보에서 내리고 D 로 이동. arm 추가 금지
held-out 세션 수 **최소 6개** 를 split 단계에서 확보한다 `[추정][미검증]`
                — 세션 4개면 클러스터 CI 가 겨우 서고 2개면 서지 않는다(`UNCERTAINTY.md` §2)
★임계 근거의 한계  20% 는 `synth -> FT` 의 효과(36%)에서 왔는데, 이 실험이 비교하는 것은
                **두 arm 모두 real 라벨을 받는** L vs H 다. 그 대비의 효과 크기는 알려진 바가
                없고 상식적으로 더 작다. 그래서 고정 % 가 아니라 위의 **구간 기준**을 쓴다.
expected result L > M > H
```

착수 전 반드시 확인할 것 (이번 감사가 찾은 함정):
- `_prepare_live_gt.json` 의 `split_mode` 를 **촬영단위로 바꾼다.** 기본값이 interleave 다.
- augmentation 은 base 계약을 쓴다. ultralytics 기본과 교락시키지 않는다.
- `paper_real_ft_v1` 이 중단된 이유가 라벨 결함(402장 중 106장 LR 순서 위반, 187장
  90도 stale)이다. **쓸 라벨을 먼저 감사한다.** 그 사이 live_capture_gt 는 402 → 851 로
  늘었고 추가 449장 감사 기록이 저장소에 없다.
- 드라이버는 train → eval → verdict → notify 를 한 스크립트에서 끝낸다.
  완료 판정은 결과 JSON 의 최종 마크로만.

---

## 5b. ★라벨 감사가 설계를 바꿨다 (2026-09-06 rev2)

`REAL_LABEL_AUDIT.md` 와 메인 세션 재현으로 세 가지가 확정됐다.

**(1) 3-arm 앙각 설계는 불가능하다** [확인]. LABEL_OK 프레임의 앙각 x 물체 실측:

```
앙각      정사각 110x110   논문 110x130   wood      합
<8            282            116          0       398
8-15          165              6          0       171
>=15            2              8         43        53
```

`>=15도` 는 어느 물체로도 arm 을 못 만든다 — 정사각 2장, 논문 물체 8장, 나머지는 wood 다.
그 층을 쓰면 **앙각축이 아니라 물체축을 재게 된다.** 쓸 수 있는 대비는
정사각 물체의 `<8`(282) 대 `8-15`(165) 하나다. → 2-arm 2x2 전이 행렬로 축소.
두 arm 모두 "어려운 절반" 안이라 대비가 약해졌지만, 실패율은 51.1% 대 32.8% 로 갈리므로
전이 비대칭은 측정 가능하다.

**(2) ★같은 파일 안에 keypoint 규약이 두 벌 있다** [확인, 메인 세션 독립 재현].
`live_capture_gt` 844 프레임 중 **392(46.4%)** 에서 `keypoint_annotations` 와
`manual_kps` 가 **정확히 90도 순열**만큼 어긋난다 — 잔차 중앙값 **0.000 px**,
즉 잡음이 아니라 정확한 재라벨이다. 어느 쪽이 규약인지는 갈린다:

```
camera-facing 0123 검사 (0 왼쪽/1 오른쪽, 0·1 위/3·2 아래), 847 프레임 전수
  keypoint_annotations   LR 위반   0 / 847  (0.0%)
  manual_kps             LR 위반 197 / 847  (23.3%)
```

학습 라벨 변환기는 `keypoint_annotations` 를 읽고, pose·평가 경로는
`manual_kps`/`projected_cuboid`/`pose_transform` 을 읽는다. 고치지 않으면
**학습은 A 규약, 평가는 B 규약**이 되어 46% 프레임에서 90도 어긋난 채로 채점된다.
→ 정본을 `keypoint_annotations` 로 못 박는 것이 **학습 착수의 선결 조건**이다.

**(3) 직전 중단 사유가 해소된다** [확인]. `paper_real_ft_v1` 을 멈춘
"402장 중 106장(26.4%) 좌우 코너 순서 위반" 은 `manual_kps` 기준 수치다.
`keypoint_annotations` 기준으로는 위반이 **0/847** 이다. 재어노테이션이 아니라
**필드 선택으로 해소된다.** 그리고 거울(mirror) 순열은 0장이다 —
`det(R)=1`·직교성이 1,701/1,701 정상이라 반사는 발생할 수 없다.

[추정] (2)의 근인은 §2 의 대칭 계약 부재다. 851장 전부
`pose_status="UNCONFIRMED_SIGNED_AXIS"` · `axis_assignment_confirmed=false` ·
`migration_status="MANUAL_REVIEW_REQUIRED"` 이고, 물체가 정사각(1.1 x 1.1)이라
yaw phase 를 정할 근거가 없다. **계약 부재가 문서 문제가 아니라 46% 라벨 불일치로 나타났다.**

---

## 6. GPU 를 쓰지 않는 선행 작업 (먼저 한다)

0. ~~정본 keypoint 필드 확정~~ — **완료 (2026-09-06).**
   `prepare_yolo_pose.load_kps` 가 `keypoint_annotations` 를 우선하도록 고쳤고,
   `gen_flip_noise_aug.flip` 이 그 필드도 함께 뒤집도록 고쳤다(안 그러면 뒤집힌
   이미지에 안 뒤집힌 라벨이 붙는다). `challenge/tests/test_keypoint_field_contract.py`
   5 tests 로 잠갔다. 합성 GT 에는 그 필드가 없어 합성 재빌드 결과는 불변이다.
   부수 효과 — 쓸 수 있는 `live_capture_gt` 프레임이 **449 → 833 장**.
1. ~~정사각 물체의 대칭 계약을 정의한다~~ — **완료 (2026-09-06).**
   소유자가 실물에서 **네 면 모두 포크 구멍 있음(4-way)** 을 확인했다. 정사각 footprint
   (x = z = 1.100 m)와 합쳐 **C4 {0, 90, 180, 270}** 으로 얼렸다:
   `challenge/config/SQUARE_PALLET_SYMMETRY_CONTRACT.json`,
   registry `symmetry_status` UNREVIEWED → FROZEN.
   ★계약이 밝히지 **않는** 것 — 상판 무늬가 90도 대칭인지는 확인하지 않았다.
   이 계약은 pose 지표와 포크 진입에 대한 것이지 "시각적으로 구분 불가" 라는 주장이 아니다.
2. `symmetry_aware_pose_metrics.py` docstring 의 "entering the pockets vs hitting the deck"
   문장을 고친다. 두 팔레트 모두 4-way 이므로 90도 회전 시 만나는 것은 deck 이 아니다.
3. ~~`infer_fps.py` 의 `PALLET_DIMS`~~ — **해소.** 그 스크립트는 Jetson FPS 측정용이고
   문서화된 입력 영상(`forklift_raw_20260528`)의 팔레트가 실제로 110x130x11 이다
   (그 세션 GT 25장 전수 확인). 값이 맞다.
4. **CLAUDE.md 와 memory 의 "161장" 을 140 으로 고친다.** 디스크 전수 = 140
   (22/12/18/27/33/12/16), `data_paths.EVAL_CANONICAL_TOTAL` 과 일치.
5. memory 2건 정정 — live-gt FT 의 0.98 은 same-session,
   medium 의 "명확히 낫지 않다" 는 OPEN 56 기준(SEALED 105 에서는 낫다).
6. ~~held-out 모집단을 하나 연다~~ — **부분 완료 (2026-09-06).**
   `live_capture_gt` 를 폴더 단위로 갈라 9 폴더 297 프레임(<8 155 / 8-15 141)을
   SHA256 으로 봉인했다: `next_experiment/HOLDOUT_SEAL.json`.
   ★**그러나 confirmatory 가 아니다** — C4 track 이 28 세션 전수를 평가했고
   `live_gt_v4/v5` 가 851장 전량으로 학습했다. 이 봉인이 사는 것은 "결과를 보고 split 을
   고르지 않는다" 와 "촬영 단위 분리" 뿐이다. 진짜 held-out 은 아래가 유일한 길이다. `data/evaluation/pallet_eval_v1/incoming/` 에
   검수 대기 중인 신규 촬영이 DAY 27,279 / NIGHT 12,459 프레임 있다. 결과를 보기 전에
   프로토콜을 얼려 세션 단위로 봉인하면, 이 프로젝트에서 처음으로 confirmatory 주장이
   가능해진다. 지금은 어떤 수치도 held-out 이 아니다.

---

## 7. Phase 0 역주행 표 (§25)

```
후보 action              예상 결과                    목적지지  최상위도달  독자의 첫 질문 / 답             판정
────────────────────────────────────────────────────────────────────────────────────────────────
GT repair               2D 중앙값 ~1px 이동           불가      안 닿음    "외삽이 원인 아닌가?" /       삭제
                                                                          가림 안에서 클릭이 더 나쁨
PnP / geometry solver   변화 없음                     불가      안 닿음    "이미 해봤나?" / 양쪽 REJECT  삭제
selective rejection     30% 버려 실패 1/3             부분      부분       "coverage 는?" / 0.70        보류
larger model            코너 -11% (7.63->6.80)        지지      닿음       "matched 인가?" / 아니다      2순위
line representation     ?                             불가      안 닿음    "target 이 뭔가?" / 미정의    STOP
targeted synthetic      ?                             부분      닿음       "전에도 실패하지 않았나?" /   조건부
 (저앙각 x 자산다양성)                                                       6연속 + 근거 run 무효
hard-negative data      ranking 만 이동               불가      안 닿음    "localisation 은?" / 불변     삭제
real-supervised         저앙각 코너 -36%              지지      닿음       "암기 아닌가?" / unseen 에서   ★1순위
 adaptation (구성 ablation)                                                 이득이 더 크다
기하 계약 수정 (과제)     계상 오차 제거                지지      닿음       "정확도가 오르나?" /          선행
                                                                          아니라 잘못 센 것을 고침       (GPU 0)
```

---

## 8. 반드시 답해야 하는 최종 질문 (§33)

**Q1. 현재 성능이 GT noise 때문에 실제보다 나빠 보이는가** — **NO** (2D) / **부분 YES** (6D).
2D: 외삽 코너와 클릭 코너의 오차 차이가 1 px 이고 가림 안에서는 방향이 반대다.
6D: W/D 배정이 클릭만으로는 130장 중 51장에서 갈리지 않는다.

**Q2. 몇 픽셀 click noise 보다 semantic/index/physical-target mismatch 가 더 심각한가**
— **논문 트랙 NO / 과제 트랙 YES.** 논문 트랙은 8! 자유 배정이 p50 을 4% 만 개선한다.
과제 트랙은 F1 의 fixed-index 실패 62장이 **전부** 270도 순열이다.

**Q3. R0 가 정답을 후보로 갖고 있는데 selector 가 못 고르는 것이 주병목인가** — **NO.**
후보 0인 프레임 0장, IoU50 매칭 97.5%, misrank 5.6%(다른 모델·모집단).

**Q4. 현재 keypoint 로 PnP 만 바꿔 의미 있는 pose 개선 헤드룸이 있는가** — **NO.**
solver 교체 +1.48% 악화, D3 는 3.67e-08 변화, 학습측 DiffPnP real 7~23% 악화.

**Q5. 지금보다 큰 model 이 matched condition 에서 기각된 적 있는가** — **NO.**
Wilcoxon p=0.1433 은 NOT_ESTABLISHED. `runs_arch_baseline` 은 전부 nano(family 비교).
medium 은 PAPER_EVAL 319 에서 채점된 적이 없고, SEALED 105 에서는 오히려 이긴다.

**Q6. 실제 RGB 의 긴 pallet edge 가 안정적으로 관측 가능한가** — **UNKNOWN.**
측정하지 않았고, 측정할 수 없다. GT line 이 physical/virtual/occluded 를 구분하지 못한다.
`EXTERNAL_SURVEY_NOT_RUN` — §14 는 게이트 통과 시에만 적용되는 조건부 절이고 통과하지 못했다.

**Q7. 기존 Direct-Hough 가 그 pixel evidence 를 line 을 따라 aggregate 하는가** — **NO.**
점수가 `role @ hypothesis_embedding.T` 한 줄이고 embedding 은 `(theta, rho)` 만의
해석적 함수다. 파일 docstring 이 raster 를 의도적으로 제거했다고 적는다.

**Q8. 기존 Hough 실패가 "line evidence 무가치" 를 반증했는가** — **NO, 구현 family 만.**
누적하는 계열(B, D)은 `@torch.no_grad()` 밖으로 못 나가거나 numpy optimizer 라 학습된 적이 없다.
다만 다시 열려면 세 조건(target 정의 / 누적을 학습 그래프에 / 저앙각 population 평가)이
동시에 필요하고, 계열 A 가 각도 요건 1도에 3.8~3.9도로 4배 모자란다.

**Q9. source synthetic 에 real hard failure 를 재현하는 조건이 있는가** — **부분적으로 NO.**
저앙각 **비율**은 맞다(29.0% vs 28.2%). 그러나 그 구간의 **자산 다양성**이 없다
(유효자산수 1.01 / 4, 87% 가 `scene.usd`). 거리 <1.5 m 는 렌더러 하드 바닥이라 아예 없다.

**Q9b. real 오검출은 어떤 구조에 몰리는가** — **반복 슬랫 텍스처.**
79건 전수 육안: 벤치 슬랫 49 · 컵뚜껑 리브 23 · 환기 그릴 3 = 91%가 같은 종류다.
단 `DEV_NEG2689` 는 사실상 6개 장면에서 나온 편향 표본이라 **비율이 아니라 종류만** 읽는다.

**Q10. 데이터를 바꿔야 한다면 정확히 무엇을 얼마나** —
(i) real: 저앙각(<15도) 라벨 프레임. 지금 851 + 391 장이 있고 앙각 층화가 안 돼 있다.
(ii) synthetic: 장수가 아니라 `elev<8 AND thin AND 대형투영` 셀의 자산 다양성
(현재 유효 1.01 / 4). 이 셀은 비-`scene.usd` 프레임이 8장/22장/1장이라 **재가중으로 불가**.
(iii) 야간은 휘도가 아니라 **대비** — real 야간은 휘도 37 인데 p99 124.5, source 야간은
p50 53 으로 `scene_preset` 이 휘도를 전혀 가르지 않는다.

**Q11. 데이터 변경 없이 가능한 가장 유망한 lever 는** — **capacity (D).**
medium 이 SEALED 105 에서 코너 7.63 → 6.80, IoU3D 0.57 → 0.66 이고 기각된 적이 없다.
그 다음이 selective rejection(kp_conf_min AUROC 0.744)이지만 coverage 대가가 크다.

**Q12. 다음 GPU 장시간 실험 하나를 고른다면** — **§5 의 real supervision 구성 ablation.**

---

## 9. 이 판정의 한계 (봉합하지 않는다)

- **held-out 이 없다.** 여기 모든 수치가 DEVELOPMENT 다. `PAPER_EVAL 319` 의
  `population_contract.role = DEV`, `held_out_final = false`. SEALED 105 도 2026-08-20 에
  열려 재봉인 불가다. 이 감사는 그 사실을 바꾸지 못한다.
- **앙각 기울기는 상관이다.** 세션 단위로 앙각·물체·조명·거리가 얽혀 있다. 투영 크기와의
  교란은 층화로 분리했지만 세션 효과는 분리하지 못했다. 다음 실험의 설계가 그 분리를 시도한다.
- **"real supervision 이 채운 것이 저앙각 레짐" 은 [추정]** 이다. 그것이 다음 실험이
  판정할 명제다. 지금 확정된 것은 "이득이 저앙각에 몰려 있다" 는 관측뿐이다.
- **"클릭 = 정수 좌표" 는 휴리스틱** 이다(`source` 필드가 전 파일 `unknown`).
  `annotate_io.py` 저장 경로와 정합하지만 필드가 아니다.
- SEALED 105 · 140 정본 비교는 **단일 seed** 이고, medium base 학습 로그가 이 머신에 없어
  완전 matched 가 아니다. 부트스트랩은 표집 불확실성을 재지 seed 불확실성을 재지 않는다.
- **앙각 구간 중 `<3` 과 `8-15` 는 해상도가 부족하다** — 전자는 세션 2개, 후자는 세션
  클러스터 CI 폭이 48 %p 다. 지지되는 진술은 "저앙각(<8) > 고앙각(>=15)" 하나이고,
  저앙각 **안에서의 순서는 확인되지 않았다** (`UNCERTAINTY.md` §1).
- 과제 트랙(155 프레임, F0/F1/F2/v4)과 논문 트랙(319, R0/R2/R5)의 수치를 같은 줄에
  놓지 않았다. 두 트랙은 **다른 물체**를 평가한다.
- ★**실패 레짐에서는 GT 의 축 배정도 사람이 못 가른다.** 리뷰 패킷의 가장 모호한 프레임
  (`eval_night08__1779449470423201536`, 클릭 5개, 클릭만 RMS 1.4393 vs 1.4443)은
  두 W/D 가설이 육안으로도 구분되지 않는다 — 팔레트가 야간·원거리·edge-on 이라
  얇은 띠로 투영되기 때문이다. 즉 **모델이 가장 많이 틀리는 구간에서 GT 의 6D 회전도
  가장 약하다.** 2D keypoint 지표는 W/D 배정에 의존하지 않으므로 Layer D 판정은
  영향받지 않지만, 같은 구간의 **6D rotation 수치는 인용하면 안 된다.**
