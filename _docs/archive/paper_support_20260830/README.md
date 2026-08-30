# 논문 문서 — 정본

> **2026-08-30 클린 스타트 60k 아키텍처 확장:** 대응 조건을 맞춘 고정 캐시
> 행렬에 치수 전용, 키포인트 전용, 정확한 중심점, 치수 미사용, 단순 결합,
> FiLM, 교차 어텐션 헤드를 각각 3개 시드로 포함했다. 어텐션의 단순 점수는
> 조금 높았지만 고정된 `+2%p` 복잡도 기준을 충족하지 못했고 대응 표본 구간도
> 0을 포함했다. 따라서 단순 공간 특징 결합을 탐색 실험의 선택 헤드로 유지한다.
> 이 도전 과제 전용 혼합 DEV에는 독립 TEST와 실제 parity/PnP 결과가 없다. 근거는
> [`ARCHITECTURE_TRAINING_QUEUE.md`](ARCHITECTURE_TRAINING_QUEUE.md)와
> [확장 실험 요약](../../challenge/yolo_pose_one_model/spatial_concat_scratch/architecture_extension/ARCHITECTURE_EXTENSION_SUMMARY.json)을 따른다.

> **2026-08-29 Phase-E 아키텍처 진단:** 고정된 YOLO 위에서 단일
> 64-D 원천 셀 대신 `7×7×64` 국소 공간 패치를 유지한 parity 진단기를
> 비교했다. 합성 DEV 기반 선택에서는 후단 치수 결합이 FiLM·교차 어텐션보다
> 우선됐고, 재사용 COMMON128에서 이전 단일 셀 헤드보다 개선됐지만 고정된
> 실제 데이터 및 오라클 통과 기준은 넘지 못했다.
> `PAPER_METHOD_STATUS = NOT_LOCKED`이며 YOLO 갱신, 전체 학습, 래퍼,
> 배포 산출물은 모두 0이다. 정본 서술은 [`architecture.md`](architecture.md),
> 수치와 보고 한계는
> [`SPATIAL_FUSION_SUMMARY.json`](../../challenge/yolo_pose_one_model/dimension_conditioning_probe/phase_e_spatial_fusion/SPATIAL_FUSION_SUMMARY.json)과
> [`POST_RUN_AUDIT.md`](../../challenge/yolo_pose_one_model/dimension_conditioning_probe/phase_e_spatial_fusion/POST_RUN_AUDIT.md)를 따른다.

> **2026-08-28 현행 다중 형상 계약:** 논문 처리 흐름은 YOLO 기반이며,
> DOPE는 비교용 및 과거 코드다. 현재 DEV는 plastic 140/128장, wood 45장,
> 공정한 양성 합집합 173장, 음성 2,689장으로 구성된다. 매니페스트의
> `object_type`에 따라 객체 레지스트리에서 plastic `(1.10,0.11,1.30)` m 또는
> wood `(0.80,0.14,0.59)` m를 선택하며, 두 형상의 기하 정보는 절대 서로
> 바꾸지 않는다. Wood45는 이전에 평가된 `CROSS_SHAPE_DEV`이며 FINAL이 아니다.
>
> Plastic 대칭은 `{I,Ry(180°)}`로 고정됐지만 선택기는 전체 83/140,
> NIGHT 13/28, 최약 세션 4/12로 실패했고 각 자세 오차 꼬리의 14장 중 13장이
> 선택기 실패였다. Wood 대칭은 `UNREVIEWED`, 선택기는 `NOT_RUN`이며 plastic
> 대칭을 물려받지 않는다. PLASTIC, WOOD, ALL의 자세 필드는 모두 null이다.
> 다중 형상 FINAL 모집단 4개는 모두 확보되지 않았다.

> **2026-08-27 GT 주석 품질검사:** 무효로 확인된 JSON 라벨 23개를 복구 가능한
> 격리 위치로 옮겼고 원본 이미지는 모두 보존했다. 이 중 21개는 과거 평가 원본
> 스캔에서 제외해 161장을 정제된 140장으로 만들었고, 나머지 2개는 올바른 정본
> 사본이 남아 있는 오래된 중복이었다. 정확한 경로, SHA-256, 클릭 가능한 이미지
> 목록은 `real_gt_v2/GT_QA_STATUS.md`에 있다.

이 폴더가 **논문에 들어갈 내용의 정본**이다. 2026-08-17 개설.

## 표 형식 규칙

이 정본에서 새로 작성하거나 수정하는 상태·결과 표는 Markdown 파이프 표를
사용하지 않는다. 고정폭 코드 블록 안에서 열을 공백으로 정렬하고, 헤더 아래에
`─` 구분선을 넣는다. 값이 없는 FINAL 셀은 `—`, 차단 상태는
`BLOCKED_*`처럼 명시한다. 재현성에 필요한 모델명·지표명·상태 코드·파일명은
원래 식별자를 유지하되, 제목·헤더·상태 설명·해석은 한국어로 작성한다. 예:

```text
결과 항목           상태                           FINAL 값
─────────────────────────────────────────────────────────────
상자 AP50:95        DEV 진단 완료                  —
PLASTIC 자세        BLOCKED_SELECTOR               —
```

> ⚠️ `_docs/` 의 다른 폴더(`method/`, `models/`, `experiments/`, `filter/`)는
> **2026-03~06 의 옛 세대**다. v8(object-frame) 시절 전제, 폐기된 평가셋 수치,
> 지금과 반대되는 판정이 그대로 남아 있다. **논문 작성 시 참조하지 않는다.**
> 필요한 사실은 여기에 다시 적는다(중복 허용 — 섞이는 것보다 낫다).

현행 정본 진입점:

```
PAPER_EVALUATION_STATUS_20260828.md  모집단·지표·평가 통과 기준 상태 (현행)
PAPER_CLAIM_LOCK.md                  방법 주장·근거 통과 기준
method.md / data.md / results.md / limitations.md
evaluation.md                        현행 주요 지표 정의
baselines.md / baselines/            기준 방법 준비·변환기·어댑터 계약
TABLE_COMPLETION_STATUS.md           표 셀별 산출물·차단 사유
ARCHITECTURE_TRAINING_QUEUE.md        새 60k 헤드 행렬과 논문 기준 방법 준비 순서
real_gt_v2/                          GT-v2·프레임·대칭 정본
current_real_dataset/                DEV/FINAL 데이터 계약
```

루트 `paper_strategy_master.md`와 `metric_split_lock.md`는 역사적 판단을 보존한다.
두 파일 상단의 2026-08-28 ACTIVE CONTRACT만 현행이며, 그 아래 옛 `[LOCKED]`
표시는 현재 방법·모집단·지표의 고정 상태로 읽지 않는다.

---

## 목차

```
PAPER_EVALUATION_STATUS_20260828.md  현재 평가 상태
DOC_STATUS_AUDIT.md                 현행/과거 문서 충돌 감사
PAPER_CLAIM_LOCK.md                 제안 방법 미확정 상태
method.md / data.md                 현행 처리 흐름·데이터 계약
evaluation.md / baselines.md        지표·기준 방법 계약
results.md / limitations.md         빈 FINAL 결과·현재 한계
TABLE_COMPLETION_STATUS.md           표 셀별 상태와 차단 사유
ARCHITECTURE_TRAINING_QUEUE.md        추가 아키텍처 학습의 READY/BLOCKED 구분
data_source_overlap_audit.md         G38/실제 데이터 구성 주장 경계
architecture.md                     현행 Phase-E 진단 + 과거 DOPE
current_real_dataset/               현재 실제 DEV 데이터 계약
real_gt_v2/                         정본 프레임·GT v2·자세 gate
evaluation_tables/                  비교 절차·빈 논문 표
```

---

## 지금 유효한 사실 (옛 문서와 충돌하면 이쪽이 맞다)

### 현재 실제 평가 모집단 계약 (2026-08-28)

현재 계약은 `challenge/real_gt_v2/manifests/`에 명시된 구성 목록만 사용한다.

```text
DEV_PLASTIC_POS140          140  DEV_POS140과 별칭 호환
COMMON_DEV_PLASTIC_POS128   128  COMMON_DEV_POS128과 별칭 호환
DEV_WOOD_POS45               45  CROSS_SHAPE_DEV; 이전에 사용한 세션 2개
COMMON_DEV_MULTISHAPE_POS   173  plastic 128 + wood 45
DEV_NEG2689                2689  공통 팔레트 부재 DEV
FINAL_PLASTIC_POS             —  UNAVAILABLE / 미고정
FINAL_WOOD_POS                —  UNAVAILABLE / 미고정
FINAL_ALL_POS                 —  UNAVAILABLE / 미고정
FINAL_NEG                     —  UNAVAILABLE / 미고정
```

DEV 비교는 PLASTIC, WOOD, ALL 모집단을 별도로 명시한다. wood45를
본 뒤 방법을 바꾸고 같은 45장을 FINAL로 보고하지 않는다.

**과거 기록 / 폐기됨:** 이전 README가 정본이라고 부르던 161장 구성과 네
세션의 `★final-test` 표기는 폐기된 계약이다. 그 수치와 `challenge/data_paths.py`의
과거 집계를 현재 모집단 또는 FINAL 구성 목록으로 사용하지 않는다.

### 현재 선택기 통과 기준 (2026-08-28 반영)

```text
전체                    83/140 = 0.592857   실패 (요구값 >= 0.95)
NIGHT                    13/28 = 0.464286   실패 (요구값 >= 0.90)
최저 세션            4/12 = 1/3 = 0.333333  실패 (요구값 >= 0.85)
자세 오차 꼬리            네 꼬리 모두 선택기 실패 13/14
상태                      POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR
```

DEV를 본 뒤 선택기 점수나 임계값을 조정하지 않는다. 원천 데이터 및 합성 데이터만
사용한 재설계와 새 사전등록 진단 전까지 논문용 자세 값은 빈칸/null이다.

### Phase-E 공간 parity 아키텍처 진단 (2026-08-29)

이 절은 위 공식 기준 선택기의 상태를 바꾸지 않는다. 고정된 G38 YOLO의 활성
one2one 분류 출력 위치 주변 `7×7×64` 패치를 토큰으로 인코딩한 뒤, mask를
적용한 평균/최댓값 128-D, 키포인트 기하 26-D, 합성 메타데이터 또는 실제 객체
레지스트리 XYZ로 만든 TRAIN 기준 정규화 4-D 크기 불변 치수 로그 비율을 후단에서
결합하는 작은 사후 parity 헤드를 평가했다. 원본의 정확한 XYZ는 parity 헤드가
아니라 PnP 기하 계산에 직접 사용한다.

```text
합성 DEV 평균 BA         S0 치수 미사용 84.857%   S1 결합 86.795%
                        S2 FiLM 86.361%          S3 어텐션 86.722%
COMMON128               S0 66/128                S1 82/128
NIGHT                    S0 21/28                 S1 14/28
oracle 회복률            S1 49.87%                요구값 70%
상태                     DIAGNOSTIC_WINNER_NOT_ADOPTED
```

FiLM과 어텐션은 고정된 3개 시드 합성 DEV 선택 지표에서 S1을 넘지 못했고 실제
데이터에는 평가하지 않았다. S1도 전체/NIGHT/세션/치수 섞기/oracle 회복률 통과 기준을
통과하지 못했으므로 현행 방법이나 제안 방법 행으로 승격하지 않는다. G38 TEST와
COMMON128은 이미 열렸고 고정 YOLO도 TEST 이미지에 앞단 노출이 있으므로, 이 결과는
`ADAPTIVELY_REUSED_DEV_DIAGNOSTIC`이다. 방법 고정 전에 캐시와 전체 배열을 미리
불러온 범위에 관한 사후 정정까지 포함한 해석은
[`POST_RUN_AUDIT.md`](../../challenge/yolo_pose_one_model/dimension_conditioning_probe/phase_e_spatial_fusion/POST_RUN_AUDIT.md)를 따른다.
사전등록 범위와 원래 실행 보고서는 각각
[`PRE_REGISTERED_PROTOCOL.md`](../../challenge/yolo_pose_one_model/dimension_conditioning_probe/phase_e_spatial_fusion/PRE_REGISTERED_PROTOCOL.md),
[`SPATIAL_FUSION_REPORT.md`](../../challenge/yolo_pose_one_model/dimension_conditioning_probe/phase_e_spatial_fusion/SPATIAL_FUSION_REPORT.md)에 보존한다.

### 키포인트 규칙

**camera-facing 0123** (`camera_dynamic_0123_v4`, 2026-05-22 결정).
0~3 = 앞면, {0,1,4,5} = 위 / {2,3,6,7} = 아래, 8 = centroid.

옛 문서의 **"Y=UP object-frame" 표기는 폐기**됐다. object-frame 점을 0123 으로 잘못
구성한 v8 계열(`v8_ablation_A_coord`, `mixed_v8`, `pl_*_r0_*`)은 **실패작이며 사용 금지**다.

### 현행 통제 데이터 노출 계약

```
논문용     정확한 G38 38,002/1,998, plastic/wood 실제 데이터 감독 0.
           G38 구성이 wood와 엄밀하게 비중복인지는 아직 미확정.
배포용     챌린지 트랙은 별도다. 논문의 통제된 주장과 섞지 않는다.
```

### 과거 기록 / 폐기됨 — DOPE 아키텍처 판정 (2026-08-17 당시 확정)

> 아래 E3/line/5cm5deg 서술은 DOPE 세대 실험 기록이다. 현재 YOLO 논문
> 처리 흐름에서 채택된 아키텍처나 제안 방법 주장이 아니다.

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
