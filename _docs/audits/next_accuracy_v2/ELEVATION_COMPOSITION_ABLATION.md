# ELEVATION_COMPOSITION_ABLATION — 이득이 레짐 커버리지 때문인가

지시문 §13 · §14 · §15.  사전등록 `METHOD_LOCK.json` stage_2 (학습 전 동결).
드라이버 `challenge/yolo_pose_one_model/next_accuracy_v2/run_elevation_ablation.py`
결과 `.../RESULT_STAGE2.json`.  10.1 분.

## 1. 설계 — rev5 의 arm 을 왜 바꿨는가

지시문의 초안(균형 arm 178)은 **성립하지 않는다** [확인]:

```text
train pool   L(<8) 352장   handheld 42.9%
             M(8-15) 178장  handheld 96.1%
```

8-15 층에서 178장을 뽑으면 171장(96%)이 handheld 다.  두 arm 이 앙각만큼이나
촬영방식으로 갈려 `single changed axis` 가 성립하지 않는다.

→ 두 arm 을 **handheld 2 세션 안에서만** 뽑았다.  세션별 `min(L, M)` 이라
arm 의 세션 구성이 글자 그대로 같다.

```text
세션                            L     M    할당
capture_20260902_kimjihoon    142   128     128
capture_20260902                9    43       9
                                        arm = 137
```

복제는 seed 가 아니라 **membership draw** 로 만들었다 — stage_1 에서 ultralytics
seed 가 텐서를 비트 동일하게 만드는 것을 실측했기 때문이다.
draw 간 Jaccard: L 0.815~0.839 / M 0.903~0.930 (1.0 이 아니므로 실제 복제다).

## 2. 2x2 transfer matrix

held-out 297장, 적격 keypoint 중앙값 (px, 낮을수록 좋음), draw 3개 평균:

```text
                eval <8     eval 8-15
train L            2.18         2.74
train M            2.37         3.43
```

M − L 짝지은 세션클러스터 95% CI (양수 = L 우세):

```text
  <8     Δ+0.21 px  [−0.24, +0.82]   0 포함
  8-15   Δ+0.55 px  [+0.50, +1.24]   0 배제
```

세션별 부호:

```text
  <8    L 우세 8/9 세션   (M 우세는 forklift_v4_20260904_142318 하나, −0.38 px)
  8-15  L 우세 7/7 세션
```


### §9 병기 — 이 결과의 모집단 수

```text
[모집단 — arm 과 무관, GT 만으로 센다]
N_total                297   held-out 프레임
N_metric_eligible    1,251   manual_click keypoint
N_excluded_ambiguous 1,422   pnp_projected + centroid_auto (버리지 않고 따로 보고)
N_excluded_suspect       0   xy=None 프레임은 봉인 split 이 이미 제외했다

[실제 채점된 keypoint — arm 이 검출한 프레임만이라 arm 마다 다르다]
적격 kp   R0 1,243   legacy 1,239   contract 1,251
모호 kp   R0 1,412   legacy 1,407   contract 1,422
```
★두 수를 섞지 말 것.  1,251 은 모집단이고, 1,243 은 R0 가 검출한 297장 중 295장의
적격 keypoint 다.  산출물 `FULL_STRATA_RESCORE.json` 의
`population_counts_arm_independent` 대 `per_arm_by_stratum[*].N_metric_eligible`.

## 3. 판정

```text
REGIME_SPECIFICITY = NOT_SUPPORTED
```

§14 가 요구한 대각 우세(`A<C` 이고 `D<B`)의 **뒷부분이 성립하지 않는다**:
`train M / eval 8-15` 가 3.43 인데 `train L / eval 8-15` 는 2.74 다.
**8-15 로 학습해도 8-15 평가에서 더 낫지 않다.**

즉 "같은 레짐 학습이 같은 레짐 평가에 유리하다" 는 기각된다.
관측된 것은 다른 것이다 — **저앙각 프레임이 두 층 모두에서 더 나은 학습 데이터다.**
16개 (세션 x 층) 칸 중 15칸에서 L 이 우세하고, CI 가 0 을 배제하는 것은
`8-15` 층 하나다(그 층은 regime-matching 가설의 **반대** 방향이다).

`<8` 층은 `INCONCLUSIVE` 다 (+0.21 [−0.24, +0.82]).
§15 대로 **동등하다고 쓰지 않는다** — 동등성 주장에는 별도 margin 이 필요하다.

## 4. 적용범위 — 인용 시 필수

1. **두 arm 모두 handheld 2 세션에서 나왔고 held-out 은 100% forklift 다.**
   두 arm 이 같은 도메인 이동을 겪으므로 L-vs-M 대비는 유효하지만,
   절대값(2.18 px 등)은 배포 도메인 성능이 아니다.
2. **arm 137장은 전체 536장보다 나쁘다** — 전체 학습(FT_CONTRACT)은
   `<8` 1.87 px / `8-15` 2.69 px 였다.  arm 은 데이터를 4분의 1로 줄인 조건이다.
3. `M` arm 의 draw 간 Jaccard 가 0.90~0.93 으로 `L`(0.82~0.84)보다 높다 —
   `kimjihoon` 의 M 128장이 매 draw 에 전부 들어가기 때문이다(pool 이 128).
   M 쪽 복제 산포가 구조적으로 작다.
4. held-out 이 강한 일반화 시험이 아니다 — `CORRECTED_REAL_FT.md` §4.2 참조.

## 5. §35 의 어느 갈래인가

**Case B 의 변형**이다.  corrected real GT 에서 개선은 났고(§11),
low/mid composition 의 **regime-matched** 차이는 없다.

다만 지시문의 Case B 문구("low/mid composition 차이 없음")와 정확히 같지는 않다 —
차이는 있고 방향이 한쪽이다.  그래서 다음 두 문장을 구분해서 쓴다.

```text
지지되지 않음   "저앙각을 겨냥해 학습하면 저앙각에서 특별히 좋아진다"
관측됨          "저앙각 프레임이 (두 층 모두에서) 더 나은 학습 데이터다"
```

두 번째는 저앙각 커버리지를 늘릴 근거가 되지만, 첫 번째가 아니므로
**"저앙각 전용 synthetic 을 렌더한다" 의 근거는 지시문이 예상한 것보다 약하다.**
