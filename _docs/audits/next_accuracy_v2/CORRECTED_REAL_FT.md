# CORRECTED_REAL_FT — 정정된 라벨에서 real supervision 이 처음 본 촬영을 움직이는가

지시문 §11 · §12.  사전등록 `data/pallet/results/next_accuracy_v2/METHOD_LOCK.json`
(sha256 `95fe48bef176b804`, 학습 착수 **전** 동결).
드라이버 `challenge/yolo_pose_one_model/next_accuracy_v2/run_corrected_real_ft.py`
결과 `.../RESULT.json`.  20.6 분.

## 1. 설계

```text
base            R0 (합성만) sha256 970a0913b38ed4c9e3662837…
바뀌는 축       학습 라벨을 읽는 필드 하나
                  legacy   = projected_cuboid      (규약 위반 198/851)
                  contract = keypoint_annotations  (규약 위반   0/851)
고정            프레임 집합(536) · recipe · augmentation · 스텝 · 평가 GT
split           촬영단위(폴더).  train 536 / held-out 297 (9 폴더). 프레임 중복 0.
평가            held-out 적격 keypoint = source == manual_click 1,251점
                  (<8 681 / 8-15 566).  pnp_projected 는 저장된 pose 의 재투영이라
                  거기에 맞추는 것은 순환이므로 primary 에서 뺀다.
```

## 2. 결과

index-wise 2D 오차 중앙값 (원본 픽셀):

```text
arm             <8       8-15     검출
R0            4.37 px   6.83 px   295/297
FT_LEGACY     2.74 px   3.33 px   294/297
FT_CONTRACT   1.87 px   2.69 px   297/297
```

짝지은 세션클러스터 95% CI (양수 = 앞쪽이 우세):

```text
contract − R0        <8     Δ+2.70 px  [+1.73, +3.04]   0 배제
contract − R0        8-15   Δ+4.80 px  [+3.46, +5.22]   0 배제
legacy   − R0        <8     Δ+1.42 px  [+0.88, +1.76]   0 배제
legacy   − R0        8-15   Δ+4.54 px  [+3.03, +5.19]   0 배제
contract − legacy    ALL    Δ+0.68 px  [+0.16, +1.21]   0 배제
contract − legacy    <8     Δ+0.71 px  [+0.31, +1.46]   0 배제
contract − legacy    8-15   Δ+0.63 px  [−0.37, +1.32]   0 포함
```

★`contract − legacy` 는 처음에 문서에만 있고 산출물에 없었다(드라이버가 두 arm 대 R0
만 계산했다).  2026-09-06 `rescore_full_strata.py` 로 다시 계산해
`FULL_STRATA_RESCORE.json` 의 `paired_deltas_...` 에 넣었다 — 값은 위와 동일하다.

## 2.2 §11 이 요구한 전체 층 (2026-09-06 보강)

처음 보고에는 `<8`·`8-15` 두 층만 있었다.  지시문이 요구한 `ALL`·`>=15`·`DAY`·`NIGHT`
까지 **기존 체크포인트를 다시 채점해서**(새 학습 0) 채웠다.
산출물 `data/pallet/results/next_accuracy_v2/FULL_STRATA_RESCORE.json`,
스크립트 `scripts/research/next_accuracy_v2/rescore_full_strata.py`.

```text
층      N_total  N_eligible  N_ambiguous  N_suspect     R0   legacy  contract
ALL         297       1,243        1,412          0   5.27     3.03      2.25
<8          156         673          713          0   4.37     2.74      1.87
8-15        140         566          694          0   6.83     3.33      2.69
>=15          1           4            5          0  10.15     7.24      4.58
DAY         297       1,243        1,412          0   5.27     3.03      2.25
NIGHT         0           —            —          —      —        —         —
LOW         103         414          513          0   5.07     2.97      2.27
MID          96         400          464          0   5.58     3.07      2.18
HIGH         98         429          435          0   5.23     3.11      2.25
```

- **`NIGHT` 은 0장이다 — 공허하다.**  세션명 시각이 아니라 851장의 평균 휘도를 재서
  정했고, 가장 어두운 세션(19:22 촬영)도 66.4 다.  근거 `GT_PARTITION.md` §9.1.
  대신 측정 휘도 삼분위(`LOW`/`MID`/`HIGH`, 경계 104.6 / 111.2)를 층으로 쓴다.
- **`>=15` 은 held-out 1장이라 판정 불가**다.  수치를 인용하지 말 것.
- `N_ambiguous` 는 `pnp_projected` + `centroid_auto` 점 수다(§9 병기 규칙).

### secondary metric — 처음에 "미측정" 이라고 썼던 것들

```text
층      arm         p90 px   gross25(kp)      NME    파생점 median px
ALL     R0           10.96        0.00%   0.0205              2.88
        legacy        5.38        0.00%   0.0110              3.14
        contract      4.94        0.00%   0.0089              2.03
<8      R0            7.66        0.00%   0.0226              2.94
        legacy        5.41        0.00%   0.0154              3.15
        contract      3.95        0.00%   0.0105              1.85
8-15    R0           12.41        0.00%   0.0194              2.83
        legacy        5.32        0.00%   0.0088              3.10
        contract      5.37        0.00%   0.0076              2.25
```

`gross25` 는 **keypoint 단위로 다시 재서 실제로 0.00%** 다(p90 최대 12.41 px).
처음에 "프레임 중앙값이라 robust 해져 0% 로 보였다" 고 썼던 것은 정정한다 —
keypoint 단위로도 0 이다.

★**파생점(pnp_projected) 지표가 갈린다**: `legacy` 는 R0 보다 **나쁘고**(3.14 대 2.88)
`contract` 만 낫다(2.03).  적격점과 파생점이 같은 방향으로 움직이는 것은 contract 뿐이다.

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
REAL_SUPERVISION_LEVER = REPRODUCED
```

사전등록 gate("contract − R0 의 `<8` 층 CI 가 0 을 배제하고 FT 우세")를 만족한다.

부차 질문(§11 의 single changed axis)의 답은 **층마다 다르다**:
**라벨 계약 수정 자체의 효과는 저앙각에서만 확정된다** (+0.71 px, 0 배제).
`8-15` 에서는 구분되지 않는다 (+0.63 px, 0 포함) → 그 층은 `INCONCLUSIVE` 로 쓴다.
동등하다는 뜻이 아니다 — 동등성 주장에는 별도 margin 이 필요하다.

검출도 정본 쪽이 낫다(297/297 대 294/297 대 295/297).

## 4. ★ 이 수를 인용할 때 반드시 함께 적을 것

### 4.1 seed 3개는 독립 복제가 아니다

[확인] `legacy_s0/s1/s2` 의 **모델 텐서가 비트 동일**하다
(파라미터 753개, 최대 절대차 0.000e+00).  ultralytics `seed` 가 dataloader 표집에
도달하지 않는다 — memory `ultralytics-seed-does-not-reach-dataloader` 의 재현이다.
METHOD_LOCK 에 "먼저 확인할 것" 이라고 적어 두고 먼저 확인하지 않았다.

→ **arm 당 유효 run 은 3이 아니라 1이다.**  위 CI 는 학습 난수 산포가 아니라
held-out **모집단** 불확실성(세션 클러스터)이고, 사전등록 gate 도 그것으로 정의돼
있으므로 판정 자체는 유효하다.  그러나 "seed 3개에서 재현됐다" 고 쓰면 틀린다.

### 4.2 held-out 은 강한 일반화 시험이 아니다

[확인] 촬영그룹 4개가 **train 과 held-out 양쪽에 모두** 있다:

```text
group                          train 세션   held-out 세션
forklift_v4_20260901                1            3
forklift_v4_20260903                3            1
forklift_v4_20260904               13            5
handheld_20260902                   2            0
```

즉 held-out 세션 대부분이 train 세션과 **같은 날 · 같은 현장 · 같은 지게차**다
(예: held-out `…_142318` 과 train `…_142958` 은 몇 분 차이).
이것은 "처음 본 촬영" 이지 "처음 본 현장/팔레트" 가 아니다.
1.87 px 라는 절대값은 그 조건에서 읽어야 한다.

### 4.3 보고 금지 지표

이 모집단은 851장 전부 `n_pose_candidates = 2` 라 축이 결정되지 않는다.
**axis / yaw / full 6D 를 이 결과로 보고하지 않는다.**
가림 층화도 불가(`occlusion_level` 전부 unknown).
근거 `GT_PARTITION.md`.

### 4.4 gross 지표 — 정정됨 (처음엔 미측정이었다)

처음 보고에서 "프레임 중앙값이라 robust 해져 0% 로 보인다, 미측정으로 둔다" 고 썼다.
2026-09-06 **keypoint 단위로 다시 쟀고 그래도 0.00%** 다 — p90 이 최대 12.41 px 라
25 px 를 넘는 적격 keypoint 가 실제로 없다.  §2.2 표 참조.
NME 와 파생점 지표도 함께 채웠다.

## 5. §12 의 갈래 중 어디인가

§11 만 놓고 보면 Case A 와 B 의 **공통 앞부분**이다 — corrected real GT 에서
저앙각이 개선됐다.  둘을 가르는 뒷부분("low training 이 low eval 에 특별히 강한가")은
§13 이 답했고, 답은 **아니오**다.

→ 최종 갈래는 **Case B 의 변형**이다.  `FINAL_DECISION.md` §4.5 를 따른다.
   (이 문서가 한때 "Case A 의 앞부분" 으로만 적어 FINAL_DECISION 과 어긋났다 — 정정함.)
