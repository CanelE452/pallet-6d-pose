# FINAL DATA ROLE LOCK (2026-08-20)

> ## ⚠ V2 로 일부 대체됨 (2026-08-20 후반)
> **EDGE_HARD_TRUNC_TRAIN 10,000 의 학습 제외가 해제됐다.** 아래 EDGE 절은 이력이다.
> 정본은 `DATA_ROLE_LOCK_V2.json` / `FINAL_SYNTH_TRAIN_V2.json`(50,000).
>
> 해제 사유는 '코너가 4개 이하라 못 쓴다' 가 **범주 오류**였기 때문이다 —
> G1 게이트 `V_vis>=4` 는 평가에서 PnP 성립 여부를 보는 것인데 학습 배제에 썼다.
> corner loss 가 학습하는 것은 화면 안 채널(n_supervised)이고 EDGE 의 **99.2%**
> 가 `n_supervised>=4` 다(중앙값 5). line_valid_roles>=6 도 99.1%.
>
> CORNER_LA · NEGATIVE · dev/untouched 항목은 **그대로 유효**하다.

architecture 는 lock 됐다. 이 문서는 **최종 학습에 무엇이 들어가고 무엇이 안 들어가는지**를
고정한다. 각 제외는 이유를 함께 적는다 — "데이터가 나빠서" 가 아닌 경우가 대부분이다.

## FINAL_NEURAL_TRAIN

```
Corner  =  BROAD_40K  40,000
Line    =  BROAD_40K  40,000
manifest = dataset_release/FINAL_SYNTH_TRAIN_V1.json
sha256   = FINAL_SYNTH_TRAIN_V1.sha256
```

BROAD 40,000 = historical MH_TRAIN 33,758 + historical MH_DEV 6,242.

**MH_DEV 를 되접어 넣는 것이 정당한 이유는 하나뿐이다** — 최종 성능 주장이
REAL IN-HOUSE DEV/TEST 로 옮겨갔기 때문이다. 그래서 synthetic holdout 을 지킬 이유가
없어졌다. 대가는 명시한다: **이 checkpoint 로 MH_DEV 를 unseen 이라 부를 수 없다.**

## NOT IN FINAL NEURAL TRAIN

```
CORNER_LA_Y15_30      2,500   ABLATION_ONLY
CORNER_LA_Y30_PLUS    2,500   ABLATION_ONLY
CORNER_LA_FRONTAL         0   PRESERVE_UNUSED
EDGE_HARD_TRUNC      10,000   LINE_HARD_ABLATION   (sampling weight = 0)
EDGE clean untouched  1,000   QA / CONTROL
NEGATIVE_SYNTH_V1    10,000   CALIBRATION / EVALUATION
```

### CORNER_LA (Y15_30 / Y30_PLUS / FRONTAL)

targeted enrichment 가 일관된 training gain 을 보이지 않았다 — C1 / C1_RESCUE 에서
두 seed 의 방향이 충돌하고 CI 가 0 을 포함한다. Y15_30 은 canonical 재분류 시 겨냥한
cell 이 near-baseline(9.62px, ×1.14)이라 개입이 아니라 control 이다. FRONTAL 은
렌더 0장이고 `FRONTAL_DATA_DECISION.md` 가 `RENDER 0 frames` 로 종결했다.

부가 근거(독립): 누수 감사 L5 에서 CORNER_LA 는 **내부 near-duplicate 가 기대값의
6.06배**다. 2,500장의 실효 다양성이 장수보다 낮다.

향후 REAL_DEV 에서 특정 geometry regime 의 명확한 failure 가 확인될 때만 다시 연다.

### (이력 · V2 에서 해제됨) EDGE_HARD_TRUNC — 데이터가 나빠서가 아니다

```
1  V_vis < 4 라 point-valid support 밖이다 (G1 게이트 0% 통과, 설계상)
2  최종 F3 pose route 는 Point PnP initialisation 을 먼저 요구한다
3  EDGE -> line 개선 -> 최종 pose 이득 은 아직 미검증 (NOT_TESTED)
4  따라서 main final model 에 자동 포함하지 않고 line-hard ablation 으로 보존
```

**V2 정정**: 1번이 틀렸다. `V_vis` 는 가시 코너 수이고 corner loss 가 감독하는 것은
화면 안 채널 수(`n_supervised`)다. EDGE 의 99.2% 가 `n_supervised>=4` 이므로
"4개 미만 코너로 가르친다" 는 서술 자체가 성립하지 않았다. corner stream 투입 금지도
함께 해제한다.

2·3번은 **여전히 유효하다** — F3 는 추론에서 Point PnP 초기화를 요구하고, EDGE 를
넣어 성능이 좋아진다는 증거는 아직 없다(NOT_TESTED).

보존 manifest: `EDGE_HARD_LINE_ABLATION.json` (sampling weight 0 으로 명시).

### NEGATIVE_SYNTH_V1

dense negative corner training 은 pose 를 깨뜨려 REJECTED (seed2 pose safety 대실패).
최종 rejection 은 `score_4kp` 이고 **inference 전용**이다 — pose network 를 건드리지
않는다.

역할: score_4kp rejection calibration / synthetic hard-negative diagnostic / ablation.

**최종 threshold 는 REAL_DEV 에서 정한다.** `presence_threshold.json` 의 합성 값은
initial range/reference artifact 이며 "FINAL threshold" 라고 부르지 않는다
(`_delivery/README_DELIVERY_20260820.txt` 계약).

## 이름 구분 (혼동 금지)

```
PAPER_CORE_V1          33,758   architecture-development contract. 과거 실험 재현용
FINAL_SYNTH_TRAIN_V1   40,000   real-evaluation 용 final training contract
```

## 전달 상태 (2026-08-20 기준)

```
negative train / dev              sha256 일치 OK
edge clean / trunc untouched      sha256 일치 OK
edge clean train/dev 11,000       ★의도적 미배포 (README 가 CLEAN 을 '대조군' 으로 규정)
                                   — delivery failure 아님
```
근거: `_delivery/DELIVERY_MANIFEST_20260820.json`,
`data_audit/HANDOFF_VERIFY_20260820.txt`, `edge/README_edge_complement_v1.txt`.
