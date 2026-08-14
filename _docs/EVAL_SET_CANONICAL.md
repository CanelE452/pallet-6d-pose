# 평가셋 정본 (canonical eval set) — 반드시 이것만 사용

> **2026-08-04 사용자 확정.**  평가는 사용자가 어노테이션 툴에서 직접 `v` 키로
> `eval` 표시한 프레임만 사용한다.  다른 셋을 쓰면 결과의 적용범위가 틀린다.

## 정의

평가용 = `objects[0].split == "eval"` 인 manual GT 프레임.

경로는 `challenge/data_paths.py` 의 `EVAL_CANONICAL` 이 유일한 출처다. 코드에서
문자열로 다시 쓰지 말고 그쪽을 import 한다 (2026-08-14 폴더 재편에서 두 곳이
어긋난 전례).

```
폴더                                                             eval  png  cuboid  gt_source  lock §1.6 역할
──────────────────────────────────────────────────────────────────────────────────────────────────────────
challenge/data/01_real/eval_canonical/_outside_eval_manual_gt      22   22      22  manual     filter-val 세션
challenge/data/01_real/eval_canonical/capture0403noapril_manual_gt 12   12      12  manual     (lock 미지정)
challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt   22   22      22  manual     (lock 미지정)
challenge/data/01_real/manual_gt/capturepallet07_manual_gt         27   27      27  manual     ★final-test
challenge/data/01_real/manual_gt/capturepallet09_manual_gt         36   36      36  manual     ★final-test
challenge/data/01_real/manual_gt/capturenight08_manual_gt          17   17      17  manual     ★final-test
challenge/data/01_real/manual_gt/capturenight09_manual_gt          25   25      25  manual     ★final-test
──────────────────────────────────────────────────────────────────────────────────────────────────────────
합계                                                              161
```

2026-08-14 재편 전 경로(`challenge/data/_outside_eval_manual_gt` 등)는 더 이상
존재하지 않는다. 호환용 symlink 를 잠시 뒀다가 참조를 모두 전환한 뒤 제거했다.

## 2026-08-07 변경 — outside final-test 편입 (봉인 1회 소진)

`metric_split_lock.md` §1.6 은 outside final-test 를 **p09, p07** 로 동결했는데, 그때까지
정본 eval 22 장은 전부 **filter-val 세션**(p02~p05, p08)이었다. 즉 정본과 lock 이 어긋나 있어
"final-test 에서 평가했다"고 쓸 수 없는 상태였다.

- 조치: 이미 어노테이션된 `capturepallet07_manual_gt` 27 + `capturepallet09_manual_gt` 36
  = **63 장에 `split="eval"` 부여**(전부 기존 `(none)`, `train` 표시 0 → 덮어쓴 것 없음).
- **PL 풀 미포함 확인 [확인]**: `data/pallet/real_unlabeled_ralph*` 어디에도 p07/p09 프레임이 없다
  (풀 = p02,p03,p04,p05,p08 + cad + night + noapril). 따라서 p07/p09 는 **inductive-clean**.
  테스트 `test_final_test_sessions_are_not_in_any_pseudo_label_pool` 로 강제.
- **⚠️ 봉인 소진**: 이 문서 "절대 하지 말 것" 의 `capturepallet07/09` 봉인은 이 시점에 해제됐다.
  **한 번 본 셋은 더 이상 미접근 holdout 이 아니다** — 이후 이 셋으로 threshold 튜닝·모델 선택을
  하면 안 되고, 새 봉인이 필요하면 새 세션을 촬영해야 한다.
- **보고 규칙**: outside 를 한 덩어리로 합치지 말 것.
  - `outside_finaltest` = p07,p09 (63) → **주 결과**
  - `outside_filterval` = p02~p05,p08 (22) → threshold 캘리브 세션이므로 참고용
- night 은 이 시점(08-07)에는 부여하지 않았다. → **2026-08-08 에 아래와 같이 변경**.

## 2026-08-08 변경 — night final-test 편입 (봉인 1회 소진)

outside 와 같은 어긋남이 night 에도 있었다. lock §1.6 의 night final-test 는 **n09, n08** 인데
`_night_eval_manual_gt`(43 장, 세션 n05/n06/n07)는 **filter-val 세션**이고 그 세션들이 그대로
night PL 풀(n05 176 / n06 160 / n07 164)에 들어가 있다 = transductive.

- 조치: 이미 어노테이션된 `capturenight08_manual_gt` 17 + `capturenight09_manual_gt` 25
  = **42 장에 `split="eval"` 부여**(전부 기존 `(none)`, `train` 0 → 덮어쓴 것 없음).
- 사전 품질 확인: 전부 `gt_source=manual`, reproj med 1.48 / 2.20 px, >5px 는 0 / 3 장(평가 시 제외),
  `(-1,-1)` 코너가 있는 프레임 n08 4 / n09 5 (PnP 는 ≥6 점이면 성립).
- **PL 풀 미포함 확인 [확인]**: night 풀 = n05, n06, n07 뿐 → n08/n09 는 **inductive-clean**.
  테스트 `test_final_test_sessions_are_not_in_any_pseudo_label_pool` 에 두 폴더 추가.
- **⚠️ 봉인 소진**: 이 문서 "절대 하지 말 것" 의 `capturenight08/09` 봉인은 이 시점에 해제됐다.
  outside(p07/p09)와 마찬가지로 **재봉인 불가**이며, 이후 이 셋으로 threshold 튜닝·모델 선택 금지.
  프로젝트에 남은 미접근 holdout 은 `testset_full8_manifest`, `handannot17` 뿐이다.
- **보고 규칙**: night 도 한 덩어리로 합치지 말 것.
  - `night_ft` = n08, n09 (42) → **주 결과**
  - `night_fv` = n05~n07 (43, `_night_eval_manual_gt`) → PL 풀과 같은 세션이므로 참고용
  - 정본 eval 은 `night_ft` 만이다(`_night_eval_manual_gt` 는 여전히 split 미부여 = 정본 아님).

`split` 은 **최상위가 아니라 `objects[0].split`** 에 있다.  최상위 `split` 을 읽으면
전부 "없음" 으로 보인다 (2026-08-04 실제 발생한 오류).

## night 이 0 장인 이유 (의도된 제외)

사용자 확인(2026-08-04): **night 시퀀스는 가림(occlusion)이 심해 추론 자체가 성립하지
않는 프레임이 많아** eval 로 표시하지 않았다.  `_night_eval_manual_gt` 43 장은
train 10 / 미표시 33 이며 평가에 넣지 않는다.

따라서 **야간 성능은 이 정본으로 측정하지 않는다.**  과거 night 를 포함해 얻은 결론
(예: PPD real 에서 outside 0.023 = night 0.023 이므로 조도 가설 기각)은 그 시점의
셋에 한정된 것으로 읽어야 한다.

## 선정 방법 (툴)

`challenge/scripts/annotate.py`

```
v 키   이 frame eval/train 토글 (annotate.py:29, 225-228)
       -> objects[0].split 에 "eval" / "train" 저장
--default_split {eval,train}   새 frame 기본값 (기본 train)
```

대량 pool 은 기본 `train` 으로 두고 평가에 쓸 것만 `v` 로 `eval` 표시한다.

## 절대 하지 말 것

- `_eval_sets/outside_combined` 를 평가에 쓰지 말 것 (경로 `data/_eval_sets/outside_combined`).
- `_eval_sets/night_combined` 를 평가에 쓰지 말 것 (경로 `data/_eval_sets/night_combined`).
  둘 다 05-27 에 만든 구본 통합본이며 split 정보가 없다.
- `split` 이 없는 JSON 을 "eval 기본" 으로 간주해 포함하지 말 것.
  구 memory 규칙이 그랬으나 **2026-08-04 로 폐기**한다.  명시 `eval` 만 쓴다.
- `split == "train"` 프레임을 평가에 넣지 말 것.
- final-test 봉인 유지: `testset_full8_manifest`, `handannot17`.
  (`capturepallet07/09` = 2026-08-07, `capturenight08/09` = 2026-08-08 에 **봉인 해제 후 정본 편입**
  — 위 변경 절 참조. 네 세션 모두 threshold 튜닝·모델 선택 금지는 계속 유효하다.)

## 과거에 어떻게 평가했었나 (이력)

```
2026-05-27  data/_eval_sets/outside_combined (129) / night_combined (90) 생성.
            per-session manual_gt 를 도메인별로 합친 통합본.  split 개념 없음.
2026-07-14  annotate.py 에 eval/train 토글(v 키) 추가.  outside 는 A안 채택 —
            per-session 폴더 대신 통합 폴더 challenge/data/01_real/eval_canonical/_outside_eval_manual_gt
            하나로 모으고, 대량 pool 에서 v 로 eval 을 골라 표시.
            ★"향후 eval 시 outside 도메인 경로를 이 통합 폴더로 지정해야
              새로 고른 eval 이 반영된다" 고 그때 명시됨.
2026-07-20  심링크 소실/복구 이력.  GT 출처는 전 도메인 gt_source="manual".
2026-07-29  paper_s2 mechanism_val_manifest.json 생성.  출처를 _eval_sets/
            outside_combined + night_combined 로 잡음 (구본).  이후 모든
            PAPER_S2 스크린이 이 manifest 를 물려받음.
2026-08-04  사용자 지적으로 발견.  아래 "무엇이 잘못됐나" 참조.
```

## 무엇이 잘못됐나 (2026-08-04 확인)

PAPER_S2 의 N87(strict mechanism-val 87장)은 07-14 이후의 eval 표시를 반영하지 못했다.

```
N87 87장의 실제 split:   split 없음 62 / eval 12 / train 13
  - 사용자가 eval 로 표시한 22장 중 10장이 평가에서 빠짐 (전부 outside)
  - 사용자가 train 으로 표시한 13장이 평가에 포함됨 (night 10, outside 3)
  - noapril eval 12장, cad eval 22장은 한 장도 포함 안 됨
```

즉 N87 은 **eval 셋이 아니라 05-27 구본의 부분집합**이다.

영향: 07-29 ~ 08-04 의 PAPER_S2 스크린(mechanism diagnostic, micro arch, PalletGraph,
SAI, PPD, PGBC, corner replacement, proposal router, stagewise loss, predseed DiffPnP,
VCR, PCR) 전부가 이 셋 위에서 판정됐다.  대부분 큰 폭의 실패라 결론이 뒤집힐 가능성은
낮지만, **적용범위 표기가 틀렸고 경계에 가까웠던 판정은 재평가 대상**이다.

경계에 가까웠던 것:
- predseed DiffPnP: GT reproj +4.5% (기준 -5%)
- VCR Gate 0 일부 항목

## 새 평가를 만들 때

1. 위 3개 폴더에서 `objects[0].split == "eval"` 인 것만 모은다 (56장).
2. manifest 에 `split_source: "objects[0].split"`, `eval_frame_count: 56`,
   폴더별 내역, membership hash 를 기록한다.
3. final-test 토큰 가드를 그대로 유지한다.
4. `data/_eval_sets/*` 를 참조하지 않는다.

검증: `challenge/tests/test_eval_set_canonical.py`
