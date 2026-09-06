# LOW_ANGLE_PL_QUALITY — 저앙각 pseudo-label 은 학습에 쓸 만한가

지시문 §17 · §18.  **새 추론 0회.**
teacher(R0, sha256 `970a0913…`) 의 프레임별 기록이
`data/pallet/results/paper_selftrain_v1/M4_FRAME_RECORDS.json` 에 이미 있어서
거기에 앙각만 붙여 층화했다.

산출물: `data/pallet/results/next_accuracy_v2/PL_QUALITY_BY_ELEVATION.json`
스크립트: `scripts/research/next_accuracy_v2/pl_quality_by_elevation.py`

## ★ 모집단 주의 — 두 트랙을 섞지 말 것

여기는 **논문 트랙**이다: `PAPER_EVAL_PLASTIC_POS` 194장, 직사각 물체
(`plastic_standard_110x130x11`).  self-training 의 적응 pool 이 이 트랙에 있다.

§11 의 corrected real-FT 는 **과제 트랙**(정사각 `110x110x15`,
`live_capture_gt`)이다.  두 결과를 같은 줄에 놓고 빼지 않는다.
앙각 못 구한 프레임 0/194.

## 1. teacher 품질은 앙각과 단조로 갈린다

```text
층      프레임  검출   kp    med px   p90 px   >20px   >40px
<8         80    80   710     9.55    85.98   31.3%   19.9%
8-15       51    51   434     6.49    51.60   18.4%   10.8%
>=15       63    63   543     5.64    18.80    9.4%    5.2%
```

[확인] 저앙각 실패 레짐이 **논문 트랙에서도 독립적으로 재현된다.**
`<8` 의 p90 이 85.98 px 인데 `>=15` 는 18.80 px 다 — 중앙값 차이(1.7배)보다
꼬리 차이(4.6배)가 훨씬 크다.

## 2. 필터는 저앙각에서 실제로 작동한다

`F4_PROPOSED` (confidence + reprojection + removal + flip, LOCK sha `57c9b939…`):

```text
층      coverage   med px    >20px           >40px            통과-기각 분리
<8        48.8%   9.55->8.11  31.3%->20.3%   19.9%-> 7.0%      +5.36 px
8-15      84.3%   6.49->5.96  18.4%->13.9%   10.8%-> 5.2%      +7.72 px
>=15      95.2%   5.64->5.67   9.4%-> 8.7%    5.2%-> 4.2%      -0.97 px
```

[확인] 저앙각에서 catastrophic(>40px)이 19.9% → 7.0% 로 3분의 1이 된다.
통과/기각 중앙값 분리도 +5.36 px 로 실재한다.
**"필터가 아무 것도 안 한다" 는 틀렸다.**

[확인] `>=15` 에서는 분리가 **−0.97 px** 로 부호가 뒤집힌다 — 고앙각에서는
필터가 오히려 약간 나쁜 쪽을 남긴다(차이는 작다).  필터의 값어치는 저앙각 전용이다.

## 3. 그런데 두 가지가 동시에 참이다

```text
필터 통과한 <8  의 gross(>20px)   20.3%
필터를 안 건 >=15 의 gross          9.4%
```

[확인] **필터를 통과한 저앙각 pseudo-label 이, 필터를 전혀 안 건 고앙각보다
여전히 두 배 이상 나쁘다.**  즉 필터는 저앙각을 "쓸 만하게" 만들지 못하고
"덜 나쁘게" 만든다.

[확인] 그리고 coverage 가 `<8` 에서 48.8% 로 반토막난다.  실패 레짐을 대표하려고
가져온 프레임의 절반이 버려진다.

## 4. §18 의 STOP 조건 대조 — 정직하게

지시문 §18 은 숫자 threshold 를 **결과를 보기 전에** 얼리라고 했다.
저장소에 저앙각 전용 purity/coverage 기준은 없었고, 나는 위 표를 이미 봤다.
따라서 **사후 threshold 를 만들어 "gate 통과/실패" 라고 쓰지 않는다.**
대신 조건별로 관측값만 적는다.

```text
A. 필터 후에도 gross tail 이 충분히 줄지 않음
   -> 줄기는 한다(31.3%->20.3%, >40px 는 19.9%->7.0%).
      그러나 잔여 20.3% 는 필터 안 건 고앙각(9.4%)의 2.2배다.  [관측]

B. coverage 가 너무 낮아 실패 레짐을 대표하지 못함
   -> <8 coverage 48.8%.  절반이 버려진다.  [관측]

C. 통과분 대부분이 R0 가 이미 잘 맞히는 프레임
   -> 통과 median 8.11 px 대 기각 median 13.47 px.  쉬운 쪽을 고르는 것은 맞다.
      다만 통과분의 20.3% 가 여전히 gross 라 "이미 맞히는 것만" 은 아니다.  [관측]

D. filtered target 이 raw teacher 보다 실질적으로 안 깨끗함
   -> 아니다.  더 깨끗하다.  [관측]
```

## 5. 결정적인 것은 이미 돌아간 실험이다

[확인] `paper_selftrain_v1` 은 이 필터들로 **이미 6개 arm 을 돌렸고, 6/6 이
2D localisation 에서 R0 보다 나빴다** (R0 6.6157 px, 최고 adapted arm 6.9987 px).
6D 는 24개 metric block 중 개선 방향으로 세션클러스터 CI 가 0 을 배제한 것이 0개다.
근거: `_docs/advising/2026-09-professor-consult/02_MAIN_RESULTS_SUMMARY.md`,
memory `filter-selftrain-fails-strong-base-2026-07`.

§19 의 `ST_LOW`(저앙각 강조)는 **그 실패한 계열에서 구성만 바꾼 변형**이다.
그리고 위 §3 이 말하는 것은, 강조하려는 그 층이 pseudo-label 품질이 가장 나쁜
층이라는 것이다 — 노출을 그쪽으로 옮기면 gross 노출이 늘어난다.


### §9 병기 — 이 결과의 모집단 수

```text
N_total                194   PAPER_EVAL plastic positive 프레임
N_metric_eligible    1,687   감독 keypoint (M4_FRAME_RECORDS.json 의
                             gt_supervised True 합 = errors_px 합, 실측)
N_excluded_ambiguous     —   이 모집단은 keypoint 출처를 나누지 않는다
N_excluded_suspect       0
```
★위 수는 **논문 트랙**이다.  과제 트랙(297 / 1,243)과 같은 줄에 놓지 말 것.

## 판정

```text
LOW_ANGLE_PSEUDOLABEL_QUALITY = FILTER_WORKS_BUT_RESIDUAL_TOO_HIGH
LOW_ANGLE_SELFTRAIN           = NOT_JUSTIFIED
```

근거 세 줄:
1. 필터 통과 저앙각의 gross 20.3% 가 필터 안 건 고앙각 9.4% 보다 2.2배 높다.
2. `<8` coverage 48.8% 로 실패 레짐의 절반을 못 담는다.
3. 같은 필터로 이미 돌린 6개 arm 이 6/6 R0 미달이다 — 구성만 바꾼 재시도는
   지시문 §26 의 중복 금지에 해당한다.

**이것은 "필터가 쓸모없다" 가 아니다.** 필터는 저앙각 catastrophic 을 3분의 1로
줄인다.  쓸모가 있는 곳은 pseudo-label 학습이 아니라 **배포 시 신뢰도 게이팅**일
가능성이 있고, 그건 다른 질문이다.
