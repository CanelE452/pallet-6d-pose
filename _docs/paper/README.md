# 논문 문서 — 정본

이 폴더가 **논문에 들어갈 내용의 정본**이다. 2026-08-17 개설.

> ⚠️ `_docs/` 의 다른 폴더(`method/`, `models/`, `experiments/`, `filter/`)는
> **2026-03~06 의 옛 세대**다. v8(object-frame) 시절 전제, 폐기된 평가셋 수치,
> 지금과 반대되는 판정이 그대로 남아 있다. **논문 작성 시 참조하지 않는다.**
> 필요한 사실은 여기에 다시 적는다(중복 허용 — 섞이는 것보다 낫다).

예외로 계속 유효한 것 두 가지:

```
루트 paper_strategy_master.md   전략 정본 (2026-08-15). 여기 복사하지 않고 가리킨다
루트 metric_split_lock.md       split·metric 봉인 정본
```

이 둘만 루트에 두는 이유는 CLAUDE.md·memory 가 이미 "정본은 루트" 로 박아두었기
때문이다. 복사하면 두 벌이 갈라진다.

---

## 목차

```
architecture.md    2-head 아키텍처와 그 선택의 근거 (E0~E4, 25k×2seed)   ★확정
```

아직 안 쓴 절(쓸 때 이 폴더에 만든다):

```
evaluation.md      평가 프로토콜 — 정본 161장, metric 정의, split 봉인
data.md            합성 데이터(v2_prod40k_clean_merged)와 real capture 의 분포 차이
selftraining.md    필터·PL — 현 시점 판정은 "2D 기하 dims-free 필터는 원리적 불가"
results.md         최종 수치표
```

---

## 지금 유효한 사실 (옛 문서와 충돌하면 이쪽이 맞다)

### 평가셋

`objects[0].split == "eval"` 인 manual GT **161장**(7폴더). 경로는
`challenge/data_paths.py` 의 `EVAL_CANONICAL` 을 import 한다 — 문자열로 다시 쓰지 않는다.

```
01_real/eval_canonical/_outside_eval_manual_gt         22
01_real/eval_canonical/capture0403noapril_manual_gt    12
01_real/eval_canonical/capturepalletcad_manual_gt      22
01_real/manual_gt/capturepallet07_manual_gt            27   ★final-test
01_real/manual_gt/capturepallet09_manual_gt            36   ★final-test
01_real/manual_gt/capturenight08_manual_gt             17   ★final-test
01_real/manual_gt/capturenight09_manual_gt             25   ★final-test
                                                      합 161
```

- 옛 문서의 **"56장" 은 폐기된 수치**다(07-3x 기준). 08-07/08-08 에 final-test 4세션을
  봉인 해제해 정본에 편입했다.
- ★final-test 4개는 **threshold 튜닝·모델 선택 금지**(봉인 소진, 재봉인 불가).
- `data/_eval_sets/*combined` 는 구본이라 **평가·진단·probe 모두 금지**.
  여기서 subset 을 뽑아 판정 4건이 뒤집힌 이력이 있다.
- 상세: `_docs/EVAL_SET_CANONICAL.md`, 테스트 `challenge/tests/test_eval_set_canonical.py`.

### Keypoint convention

**camera-facing 0123** (`camera_dynamic_0123_v4`, 2026-05-22 결정).
0~3 = 앞면, {0,1,4,5} = 위 / {2,3,6,7} = 아래, 8 = centroid.

옛 문서의 **"Y=UP object-frame" 표기는 폐기**됐다. object-frame 점을 0123 으로 잘못
구성한 v8 계열(`v8_ablation_A_coord`, `mixed_v8`, `pl_*_r0_*`)은 **실패작이며 사용 금지**다.

### 두 트랙

```
논문용     v1/v2(내 팔레트) 제외. 인터넷 무료 모델로 학습 → 처음 본 팔레트 일반화.
           비율 강건성(squash + JSON 꼭짓점 동기화) + truncation padding.
과제용     challenge. v1/v2 과적합, forklift 실배포.
```

### 아키텍처 판정 (2026-08-17 확정)

```
채택   E3_SPLIT_LATE — corner 가 line 의 128ch 병목 이전에 뿌리내린 전용 late 경로를 갖는다
기각   A1 fully-shared / A2 +mask / E2 stop-grad / E4 병목 이후 capacity
```

근거는 `architecture.md`. 진단 서사는 `_docs/audits/MULTIHEAD_FAILURE_DIAG.md`,
수치 원본은 `data/pallet/results/paper_s2_multihead/` 의 JSON 이다.

### line branch 가 pose 에 주는 것 = orientation, 그리고 그것뿐 (2026-08-17)

full `(theta,rho)` 제약은 두 번 다 사전등록 기준(two-of-two)을 못 넘었다
→ `TWO_HEAD_POSE_QUALIFIED = False`. **그러나 이걸 "line 은 쓸모없다" 로 번역하면 안 된다.**

세 번째 시도에서 pose objective 에서 `rho` 만 대수적으로 제거했다
(`(da−db)/2 = (L/2)·sin(delta)`, offset 이 정확히 소거). 결과:

```
회전    20개 subset×seed×population 조합 전부 CI 가 0 배제 (ALL +16~25%, V<8 +19~39%)
        미접촉 confirmation set(D3)이 dev(D2) 판정을 그대로 재현
translation  8개 중 7개에서 CI 가 0 포함 = 도움도 손상도 미확립
rho 가 범인  seed2 에서 full-line 은 5cm5deg 를 0.1367→0.0684 로 반토막,
             theta-only 는 0.1367→0.1504 로 올림
```

사전등록 gate 는 여전히 FAIL(`THETA_ONLY_LINE_USEFUL = False`) — seed1 이 t −3.9%,
5cm5deg −1.37pp. 원인은 추적됐다: selection 에 안전 필터는 넣었는데 통과자 중
**R median 최소**로 고르는 규칙이 남아 seed1 을 grid 끝(λ=3.0)까지 밀었다.

→ 논문 주장은 이렇게 좁혀서 쓴다: **line branch 는 rotation 으로 견고하게 전이되는
orientation 정보를 담고 있고(특히 truncation 에서 가장 크다), translation 을 반복해서
망가뜨려온 것은 offset(rho) 채널이다.**
상세: `data/pallet/results/paper_s2_multihead/THETA_ONLY_SOLVER_RESULT.md`

### 열린 축 — per-frame scale (닫히지 않았고, 예측으로는 못 닫는다)

corner 배치의 **per-frame isotropic scale** 이 translation 최대 레버다
(GT 복원 시 t +31~33%, 5cm5deg +3.3~3.5pp).

그런데 **모델 자기 출력으로는 예측되지 않는다** — Ridge D2 R² 0.13~0.17(기준 0.30),
best block 이 seed 간 뒤집힘. 예측 보정을 적용하면 pose 가 **나빠진다**(5cm5deg
−7.0pp / −4.3pp, 상수 보정도 손해). 기전: translation 은 scale 에 거의 비례하므로
곱셈 보정은 bias 를 variance 와 맞바꾸는데, E3 가 이미 bias 를 2% 로 줄여놔서
R²=0.13 짜리 추정은 넣는 게 더 많다.

→ **표현·loss 도 아니었다** (2026-08-17 추가 확인). E3 위에 pose-aware corner 감독
(DiffPnP3D)을 3,000 step × 2 seed 로 붙여 보았다. pose loss 는 scale gap 을 λ 에 대해
**단조로 닫는다**(10.7→44.4→57.8→76.0%). 그런데 **잘 닫는 λ 는 전부 translation 을
해치고**, t 를 안 해치는 유일한 λ 는 3,000 step 에서 **28개 비교 전부 CI 가 0 을 포함**한다.

같은 구조가 세 번 독립적으로 나왔다:

```
Ridge 로 per-frame scale 예측 보정  → scale 잔차 −15% 인데 pose 악화
pose loss 로 scale 통계 개선        → gap 76% 닫아도 t −7.1%
t 를 안 해치는 유일한 λ             → 효과 0
```

**oracle 의 +31~33% 는 per-frame 정확성에서 나오지, 집계 scale 통계를 1.0 쪽으로
옮기는 것으로 재현되지 않는다.** 중앙값을 옮기며 per-frame 분산을 키우면 얻는 게 없다.

→ 판정 `CURRENT_CORNER_REPRESENTATION_POSE_BIAS_REMAINS`.

### solver/loss 축 종료 (2026-08-17) — `SOLVER_LOSS_TRACK = CLOSED`

두 cheap gate 를 마지막으로 닫았다.

```
PARTIAL_DIFFPNP_SUPPORTED         False   mask 가 (B,) frame-level, GN solve 이후에만 적용
                                          → V<8 감독은 부정 증거가 아니라 **측정 불가**
THETA_ONLY_POSE_ALIGNED_CONFIRMED False   새 pose-aligned rule 이 옛 rule 과 **같은 λ** 선택
                                          → 결함은 selection 이 아니라 D0→held-out 일반화 격차
```

### 데이터 축이 유일하게 움직인 레버다

risk map(dev 1,536 = D2+D3+D4)에서 **결정적 반전**이 나왔다. 가장 나쁜 regime 인
`V<=6`(5cm5deg 0.000~0.023)은 **학습셋의 22.58%** 로 희소하지 않다 — coverage 문제가
아니라 표현·과제 정책 문제다. 실제로 결핍인 축은 저앙각뿐이다:
**synthetic `<8° = 7.69%` vs real `94%`** (둘 다 source 실측).

기존 40k 안에서 저앙각 노출만 7.66%→30% 로 올린 결과(새 렌더 없음, V·size 분포 유지):

```
              front_rear_shift    R          t         full-view 안전
seed1            +16.8%        +29.8%     +7.7%       +4.2% / +3.0%
seed2            +23.0%        +12.7%    +23.3%       +2.9% / +9.4%
```

주효과 세 절이 **두 seed 모두 통과**한다. 사전등록 gate 는 seed2 의 line offset
−6.4%(허용 −5%) 때문에 `False` 이지만 — 그건 개입이 line branch 의 학습 분포까지
바꾼 결과이고 guard 가 제 역할을 한 것이다.

이번 세션의 네 개입을 나란히 두면 결론이 분명하다.

```
Ridge scale 보정      pose 악화
pose-aware corner     28개 비교 전부 CI 가 0 포함
theta-only solver     rotation 은 강하나 translation gate 미통과
저앙각 resampling      기하·R·t 가 두 seed 같은 방향, P 0.90~0.99, full-view 무해
```

→ 다음 레버는 loss·head 가 아니라 **viewpoint coverage** 다. 다만 gate 가 FALSE 이므로
fresh synthetic 생성은 아직 승인되지 않았고, 재시험은 새 사전등록이 필요하다.
상세: `FINAL_2HEAD_POSE_QUALIFICATION.md`, `POSE_AWARE_CORNER_RESULT.md`,
`PARTIAL_DIFFPNP_AUDIT.md`, `THETA_POSE_ALIGNED_SELECTION.md`, `DATA_RISK_MAP.md`,
`DATA_RESAMPLING_RESULT.md`

### 사용하지 않는 것 (재시도 금지)

```
dense vector voting (PVNet 계열)   전 셋 패배, 투자 종료
CIGM 을 fusion 경로로               direct corner 가 74~76% 우세, oracle headroom 7~9%
2D 기하만으로 하는 dims-free PL 필터  6회 반복 확인, 원리적 불가
mixup / cut-paste negative          HONEST NEGATIVE
```
