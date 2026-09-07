# UTILITY SELF-TRAINING AUDIT

지시문 `INSTRUCTION_AS_RECEIVED.txt` §19 형식. 학습 21 run 완료 후 작성.

```
HEAD            = 903f1b3bb4e473908c7ed52335c834029e206afa
TEACHER_SHA     = 970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7
CURRENT_FILTER  = F4_PROPOSED = box_conf>=0.85 AND s_remove<=0.05 AND s_flip<=0.05
                  (reprojection 은 쓰지 않는다 — build_pseudo_manifests.py:173-174)
POLICY_DEV      =  91 frames / 3 sessions  ['plastic_day_01', 'plastic_night_01', 'wood_183705']
POLICY_VAL      =  96 frames / 4 sessions  ['eval_cad', 'eval_noapril', 'eval_outside', 'wood_night_01']
STUDENT_EVAL    = 132 frames / 6 sessions  ['eval_night08', 'eval_night09', 'eval_pallet07', 'eval_pallet09', 'wood_184309', 'wood_day_01']
STUDENT_EVAL 은 이번 실행에서 한 번도 열지 않았다 (§21 STOP 발동으로 §12 미실행).
```

## BASELINE FILTER (POLICY_DEV, plastic 66 with teacher records)

```
coverage         = 0.864
accepted wrong   = 8
rejected accurate= 5
>20px after filter = 0.072
>40px after filter = 0.060
```

## UTILITY MICRO-FT (POLICY_VAL 96 frames, matched N=43, 3 draws each)

판정 기준은 결과 보기 전 `METHOD_LOCK_UTILITY_ST.json` 에 얼렸다 —
corner median 이 U0·U1 **둘 다** 대비 15.0% 이상 개선
(측정된 동일규칙 draw 산포 7.5% 의 2배), p90 악화 17.5% 이내.

```
subset                         N cornerMed   draw폭%     p90    det   dU0%   dF4%  verdict
R0 (self-training 안 함)        --     5.183       --   17.64  0.990     --     --  REFERENCE
U0_RANDOM_MATCHED             43     5.444      4.8   18.86  0.976     --     --  BASELINE
U1_CURRENT_F4                 43     5.380      3.1   20.12  0.979     --     --  BASELINE
U2_SMALL_SCALE                43     5.133      2.4   18.52  0.969   +5.7   +4.6  NOT_UTILITY_POSITIVE
U3_LARGE_SCALE                43     5.119      2.9   18.25  0.990   +6.0   +4.8  NOT_UTILITY_POSITIVE
U4_LOW_ELEV                   43     5.460      3.5   18.51  0.979   -0.3   -1.5  NOT_UTILITY_POSITIVE
U5_HIGH_ELEV                  43     5.262      7.2   19.70  0.983   +3.4   +2.2  NOT_UTILITY_POSITIVE
U6_LOWCONF_STRONGGEOM         43     5.393      1.3   20.67  0.986   +0.9   -0.2  NOT_UTILITY_POSITIVE
U7_HIGHCONF_BORDERLINEGEOM    --        --       --      --     --     --     --  NOT_CONSTRUCTIBLE (pool 7장)
```

- arm **간** 평균 폭 6.7% vs arm **내부** draw 폭 중앙값 3.1% / 최대 7.2%. 두 수가 같은 크기다 — 조건 효과가 표집 운과 구분되지 않는다.
- 어떤 arm 도 R0(5.183px)를 의미 있게 넘지 못한다. 가장 나은 U3_LARGE_SCALE 이 +1.2% 로 draw 폭 안이다.
- p90 은 R0 17.64px 가 전 arm(18.25~20.67) 보다 낫고, 검출도 R0 0.990 가 최고다.
- 세션별 방향이 갈린다: U2 는 eval_outside 에서 −12.0%, U4 는 −22.6%, U6 는 eval_cad −8.2%.

## FINAL POLICY
```
features   = 없음 (Stage D 미진입)
rule       = 없음
accepted N = 없음
policy frozen before STUDENT_EVAL = N/A — 정책을 만들지 않았다
```
지시문 §10 은 "Stage C 에서 positive utility subset 이 실제로 확인됐을 때만 진행" 이라고
규정한다. 확인되지 않았으므로 진입하지 않았다.

## FINAL STUDENT

실행하지 않았다. §21 STOP RULE 의 "positive utility subset 이 하나도 없음" 이 발동해
§12 S0/S1/S2/S3 비교와 STUDENT_EVAL 개봉을 중단했다. STUDENT_EVAL 132 프레임은 미개봉 상태로 남는다.

## FINAL VERDICT
```
§13 판정  = NO_USEFUL_PSEUDOLABEL_REGIME_FOUND
§21 종료  = SELFTRAIN_UTILITY_TRACK_CLOSED
RESULT_ROLE = DEVELOPMENT_ONLY
```

**WHAT THIS SUPPORTS**
- 조건별 pseudo-label subset(투영크기·예측앙각·신뢰도x기하)으로 matched-N fine-tuning 했을 때,
  held-out 세션에서 학생 corner localisation 이 개선되지 않는다. 7 arm x 3 draw 전부.
- pseudo-label 순도는 기하 일관성으로 실제로 조절된다 (POLICY_DEV 에서 tau 0.05 → 0.01 로
  순도 0.709 → 1.000). 그런데 그 순도가 학생 성능으로 전이되지 않는다.
- 이 pool 에서 confidence 와 기하 일관성은 강하게 정렬돼 있다 (Spearman −0.714).
  "고신뢰인데 기하가 경계" 라는 위험군은 926장 중 7장으로, 가설을 시험할 표본이 없다.

**WHAT THIS DOES NOT SUPPORT**
- "utility-aware selection 이 원리적으로 불가능하다" 는 주장. 여기서 시험한 것은 matched N=43,
  10 epoch, 900 update, 단일 pool 이다. 더 큰 N 이나 다른 pool 에서의 결론이 아니다.
- "self-training 이 이 문제에서 해롭다" 는 주장. arm 들은 R0 와 draw 폭 안에서 겹친다 —
  개선도 악화도 분리되지 않는다.
- 어떤 논문 주장. `RESULT_ROLE = DEVELOPMENT_ONLY` 이고 PAPER_EVAL 은 이미 개발에 소진된 모집단이다.
- U7 가설의 기각. 표본이 없어 **미검증**이지 반증이 아니다.

**다음 트랙** — §21 이 지정한 대로 데이터 다양성 / capacity / representation 으로 이동한다.
이 실험이 남기는 구체적 단서: pool 의 1초 간격 유효 프레임이 648장뿐이고 예측 앙각 중앙값이
1.63도(91.5%가 8도 미만)라, 병목이 선별 규칙이 아니라 **adaptation pool 의 시점 다양성**일 수 있다.