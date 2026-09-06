# FINAL_DECISION — 다음 GPU 예산을 어디에 쓸 것인가

지시문 §34.  기준 커밋 `ddc0770`.
이번 작업에서 돌린 학습은 **12 run(총 31분)** 이고 새 렌더·새 self-training·
새 loss·새 solver 는 0 이다.

## 결정표

```text
CORRECTED_GT_PIPELINE          = FIXED_AND_LOCKED
REAL_GT_LOW_ANGLE_LEVER        = REPRODUCED
REGIME_SPECIFICITY             = NOT_SUPPORTED
LOW_ANGLE_PSEUDOLABEL_QUALITY  = FILTER_WORKS_BUT_RESIDUAL_TOO_HIGH
LOW_ANGLE_SELFTRAIN            = NOT_JUSTIFIED
TARGETED_SYNTHETIC_NEEDED      = NOT_YET (렌더 전에 공짜 pool 로 시험)
NEW_GT_NEEDED                  = 학습용은 아니다 / 평가용은 그렇다 (아래 §4)
AMBIGUOUS_GT_POLICY            = 지표별 적격성 (프레임 삭제 0건)
CAPACITY_EXPERIMENT_NEEDED     = 미해결 — matched 비교가 존재하지 않는다

NEXT_GPU_EXPERIMENT = D-cheap — 안 쓰인 저앙각 oblique pool 5,000장을 source 에
                      섞어 재학습하고, 같은 held-out 297장으로 채점한다.
                      새 렌더 0.  1 run.
WHY                 = 저앙각 실패 레짐에서 합성 asset 다양성만 무너져 있고
                      (effective 1.693 대 >=15 의 3.998), 그 구멍을 정확히 메우는
                      5,000장(100% <8도, effective 3.996)이 이미 디스크에 있는데
                      저장소 어디에서도 안 쓰인다.  이것이 "저앙각 asset 다양성" 가설을
                      렌더 비용 0 으로 시험하는 유일한 방법이다.
SUCCESS             = held-out <8 층의 짝지은 차이가 세션클러스터 95% CI 로 0 을
                      배제하고 새 source 우세
STOP_RULE           = CI 가 0 을 포함하면 D/F(저앙각 targeted synthetic) 계열을
                      전부 내리고 E(matched capacity)로 이동한다.
                      렌더를 추가하지 않는다.
```

## 1. 무엇이 확정됐나

### 데이터 계약 (§3~§7)

- `load_kps` 가 상태를 좌표 값으로 표현하던 결함을 재현하고 고쳤다.
  `xy=None -> (-0.5,-0.5) -> +PAD -> (99.5,99.5) -> v=2`, bbox 8x8 px 가 308x248 px 로.
  **반경은 작다** — 학습 소스 851장 중 2 프레임.  합성 60,000장과 정본 평가셋 140장에는 0건.
- 계약 정본은 새로 만든 것이 아니라 `real_gt_v2_schema.py` 에 이미 있었다.
- end-to-end 테스트 7개 + **산출물 불변식 테스트 2개**를 추가했다.
  후자는 낡은 `flip_noise_aug_livegt` 에서 851/851 을 잡는 것으로 이빨을 확인했다.

### 실험 (§11 · §13)

```text
§11  held-out 297장(9 폴더), 적격 keypoint 1,251점, 2D 중앙값
       arm             <8       8-15     검출
       R0            4.37 px   6.83 px   295/297
       FT_LEGACY     2.74 px   3.33 px   294/297
       FT_CONTRACT   1.87 px   2.69 px   297/297
     contract − R0 <8  Δ+2.70 [+1.73, +3.04]  0 배제  -> gate 통과
     contract − legacy <8  Δ+0.71 [+0.31, +1.46]  0 배제
                    8-15  Δ+0.63 [−0.37, +1.32]  0 포함 -> INCONCLUSIVE

§13  2x2            eval <8   eval 8-15
     train L(<8)      2.18       2.74
     train M(8-15)    2.37       3.43
     대각 우세 없음 — 8-15 로 학습해도 8-15 평가에서 낫지 않다.
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

## 2. 지시문 전제 중 뒤집힌 것

1. 정본 평가셋 140장의 `projected_cuboid` 규약 위반 **0/140** — 평가 수치 재해석 불필요.
2. 합성 GT 의 `[-1,-1]` sentinel **0개** — fallback 경로는 건드리지 않았다.
3. 기존 `live_gt_v4/v5` 의 **flip 라벨은 맞다**(696/696 정확한 미러).
   틀린 것은 base 프레임(321/696 = 46.1% 가 위반본에서).
4. §13 의 균형 arm 178 은 성립 안 함 — 8-15 층이 handheld 96.1%.
5. 저앙각 **× large** cell 은 실데이터와 어긋남 — corr(앙각, 투영크기) 실 +0.923.
   실데이터의 저앙각은 원거리·작은 투영이다.
6. §11 의 baseline 은 이미 한 번 돌았으나(clean_label) 평가가 interleave 라 무효였다.

## 3. 왜 다른 후보가 아닌가

```text
A  corrected real-GT FT        이미 했다.  성공.  더 할 것이 없다.
B  저앙각 real-GT 구성 개입    이미 했다.  regime-specificity 기각.
C  저앙각 self-training        NOT_JUSTIFIED — 통과 저앙각 PL 의 gross 20.3% 가
                               필터 안 건 고앙각 9.4% 의 2.2배 + coverage 48.8% +
                               같은 필터 6 arm 이 이미 6/6 R0 미달.
D  targeted synthetic          ★선택.  단 렌더가 아니라 **이미 있는 5,000장** 으로.
E  matched capacity            살아 있다.  nano vs medium 이 matched 로 기각된 적이 없고
                               unseen 세션(SEALED 105)에서는 medium 이 위치추정 전 축에서
                               낫다(코너 6.80 대 7.63 px, IoU3D 0.66 대 0.57,
                               R med 2.78 대 3.08, ADD-S 0.10 대 0.12; 검출만 0.952 대 0.971).
                               출처: 코너 6.80/7.63 은 `accuracy_root_cause_v1/
                               CAPACITY_AND_REAL_SUPERVISION_AUDIT.md`,
                               IoU3D·R·ADD-S 는 `accuracy_root_cause_v1/
                               MODEL_HEADROOM_AUDIT.md`.
                               n=153 의 p=0.14 는 NOT_ESTABLISHED 이지 기각이 아니다.
                               D 가 실패하면 여기로 간다.
                               ★단 진짜 matched 는 medium 을 55,980장 synthetic 으로
                               새로 학습해야 해서 D 보다 훨씬 비싸다.
F  중단                        아니다 — D 가 1 run 이고 렌더가 0 이라 중단할 이유가 없다.
```

## 4. 데이터에 관한 결론 (구체적으로 적는다)

**학습용 real GT 를 더 만들 근거는 약하다.**
137장 arm 이 `<8` 에서 2.18 px 인데 536장 전체가 1.87 px 다 — 4배 늘려 0.31 px 다.
포화에 가깝다.

**평가용은 다르다.**  지금 held-out 은 강한 시험이 아니다 — 촬영그룹 4개가
train·held-out 양쪽에 모두 있고, held-out `…_142318` 과 train `…_142958` 은
몇 분 차이다.  §30 이 요구하는 confirmatory 모집단이 저장소에 **없다**.

필요한 것을 정확히 적으면:

```text
무엇   같은 정사각 팔레트(1.1 x 0.15 x 1.1), 새 날짜 · 새 현장 또는 새 조명
얼마   100장 내외.  앙각 <8 과 8-15 를 각각 40장 이상
어떻게 클릭 6점 이상(현재 held-out 은 293/297 이 4~5점이다)
언제   모델이 보기 **전에** membership 을 얼린다 — 그래야 confirmatory 다
왜     현재 1.87 px 는 "처음 본 촬영" 값이지 "처음 본 현장" 값이 아니다.
       배포 판단에 필요한 수가 저장소에 존재하지 않는다.
```

이것은 성능을 올리는 작업이 아니라 **지금 수를 믿어도 되는지** 를 정하는 작업이다.

## 4.5 §35 의 어느 갈래인가

**Case B 의 변형**이다.

```text
Case A   corrected real GT 에서 저앙각 개선 + low training 이 low eval 에 특별히 강함
Case B   corrected real GT 에서 개선 + low/mid composition 차이 없음
```

§11 이 개선을 보였으므로 앞부분은 A·B 공통이고, §13 이 regime-matched 우세를
보이지 않았으므로 **A 가 아니다**.  다만 지시문의 Case B 문구("차이 없음")와도
정확히 같지 않다 — 차이는 있고 방향이 한쪽(L 우세)이다.
그래서 결론을 두 문장으로 나눠 쓴다:

```text
지지되지 않음   "저앙각을 겨냥해 학습하면 저앙각에서 특별히 좋아진다"
관측됨          "저앙각 프레임이 두 층 모두에서 더 나은 학습 데이터다"
```

Case B 이므로 지시문대로 **저앙각 전용 synthetic 을 추가할 근거가 약해진다.**
그것이 `TARGETED_SYNTHETIC_NEEDED = NOT_YET` 의 이유다.

## 5. 이 문서의 수를 인용할 때

- **seed 3개는 독립 복제가 아니다** — 텐서가 비트 동일(파라미터 753개, 최대차 0.000e+00).
  arm 당 유효 run 은 1.  CI 는 학습 난수가 아니라 held-out 모집단 불확실성이다.
- **axis / yaw / full 6D 를 이 결과로 보고하지 않는다** — 851장 전부
  `n_pose_candidates = 2` 라 축이 결정되지 않는다.
- **가림 층화는 불가** — `occlusion_level` 이 851장 전부 unknown.
- **DAY/NIGHT 과 휘도 삼분위는 사후(post-hoc) 층이다** — `METHOD_LOCK.json` 의
  `strata` 는 `["<8", "8-15"]` 뿐이고, 이 층들은 §11 요구를 채우려고 결과를 본 뒤
  만들었다.  사전등록 gate 판정에는 쓰지 않았고 **기술 통계로만** 본다.
- **`NIGHT` 층은 공허하다** — 851장 평균 휘도 최솟값이 64.6 이라 야간 프레임이 0장이다
  (`GT_PARTITION.md` §9.1).  대신 측정 휘도 삼분위를 쓴다.
- **`>=15` 층은 held-out 1장**이라 판정 불가다.  이 층 수치를 인용하지 말 것.
- **`live_capture_gt` 는 전부 `population_role = DEV`** 다.  confirmatory / held-out final /
  SOTA 로 부르지 않는다 (§30).
- §17·§18 은 **논문 트랙**(직사각 194장)이고 §11·§13 은 **과제 트랙**(정사각)이다.
  같은 줄에 놓고 빼지 않는다.
