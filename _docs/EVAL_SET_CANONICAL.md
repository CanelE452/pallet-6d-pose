# 평가셋 정본 (canonical eval set) — 반드시 이것만 사용

> **2026-08-04 사용자 확정.**  평가는 사용자가 어노테이션 툴에서 직접 `v` 키로
> `eval` 표시한 프레임만 사용한다.  다른 셋을 쓰면 결과의 적용범위가 틀린다.

## 정의

평가용 = `objects[0].split == "eval"` 인 manual GT 프레임.

```
폴더                                              eval  png  cuboid  gt_source
──────────────────────────────────────────────────────────────────────────────
challenge/data/_outside_eval_manual_gt              22   22      22  manual
challenge/data/capture0403noapril_manual_gt         12   12      12  manual
challenge/data/capturepalletcad_manual_gt           22   22      22  manual
──────────────────────────────────────────────────────────────────────────────
합계                                                56
```

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
- final-test 봉인 유지: `capturenight08/09`, `capturepallet07/09`,
  `testset_full8_manifest`, `handannot17`.

## 과거에 어떻게 평가했었나 (이력)

```
2026-05-27  data/_eval_sets/outside_combined (129) / night_combined (90) 생성.
            per-session manual_gt 를 도메인별로 합친 통합본.  split 개념 없음.
2026-07-14  annotate.py 에 eval/train 토글(v 키) 추가.  outside 는 A안 채택 —
            per-session 폴더 대신 통합 폴더 challenge/data/_outside_eval_manual_gt
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
