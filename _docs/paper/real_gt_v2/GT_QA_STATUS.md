# Real GT annotation QA status

Status: **PASS — confirmed-invalid labels quarantined**

2026-08-27에 확정된 오류 annotation JSON 23개만 원본 데이터 경로에서
`_archive/real_gt_invalid_20260827/` 아래로 이동했다. 이미지는 0개 삭제/이동했고,
격리 JSON은 원래 repo-relative 경로 구조와 SHA-256을 그대로 보존한다.

## 영향

| source | 격리 | clean eval |
|---|---:|---:|
| `capturepalletcad_manual_gt` | 4 | 18 |
| `capturepallet09_manual_gt` | 3 | 33 |
| `capturenight08_manual_gt` | 5 | 12 |
| `capturenight09_manual_gt` | 9 | 16 |
| `capturepallet08_manual_gt` stale duplicate | 2 | eval 변화 없음 |
| **합계** | **23** | **raw 161 → clean 140** |

평가에서 제외된 것은 21개다. 나머지 2개는 같은 frame ID/이미지의
오래된 잘못된 `capturepallet08` 복제본이다. 두 frame의 정상 canonical label은
`eval_canonical/_outside_eval_manual_gt` 아래에 남겼고 `DEV_POS140`도 그 정상
소스만 가리킨다. 따라서 두 duplicate는 frame ID가 아니라 잘못된 label
경로/SHA로만 차단한다.

기계 계약: [INVALID_GT_QUARANTINE.json](../../../challenge/real_gt_v2/INVALID_GT_QUARANTINE.json)

## 평가 경로 정리

- `DEV_POS140`, `COMMON_DEV_POS128`: 격리 source SHA 교집 0.
- legacy `paper_generic_pipeline/eval_manifest.json`: 161 → clean 140으로 재생성.
- 원래 DEV161 manifest/generator: `paper_generic_pipeline/historical/DEV161/`에
  기존 SHA와 byte-exact로 보존하고 실행 불가 historical로 표시.
- `manifests/all_samples.csv`: 잘못된 exact path 23행 제거, 정상 duplicate 2행 보존.
- generator/evaluator: quarantine path, source SHA, official exclusion frame ID를 fail-closed로 강제.
- 구형 `mc_geom.py`는 clean 140장을 과거 `ALL_161` 이름으로 잘못 기록하지 않도록
  실행을 차단하고 현재 `evaluation_v2/paper_real_eval.py`로 안내한다.

## 이미지 바로 열기

아래는 격리한 JSON의 원본 이미지다. VS Code Markdown preview에서 링크를
클릭하면 이미지를 바로 열 수 있다.
잘못된 wireframe을 그려 놓은 대표 overlay는
[capturepallet08/1778653498432396288](../../audits/gt_data_audit/overlays_suspect/capturepallet08_manual_gt__1778653498432396288.png),
[capturepallet09/1778653619758178816](../../audits/gt_data_audit/overlays_suspect/capturepallet09_manual_gt__1778653619758178816.png)이다.

### capturepalletcad (4)

- [1778653020759897344.png](../../../challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt/1778653020759897344.png)
- [1778653072734532096.png](../../../challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt/1778653072734532096.png)
- [1778653072835089920.png](../../../challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt/1778653072835089920.png)
- [1778653072868820224.png](../../../challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt/1778653072868820224.png)

### capturenight08 (5)

- [1779449490370609408.png](../../../challenge/data/01_real/manual_gt/capturenight08_manual_gt/1779449490370609408.png)
- [1779449494740471040.png](../../../challenge/data/01_real/manual_gt/capturenight08_manual_gt/1779449494740471040.png)
- [1779449503746678784.png](../../../challenge/data/01_real/manual_gt/capturenight08_manual_gt/1779449503746678784.png)
- [1779449506181779968.png](../../../challenge/data/01_real/manual_gt/capturenight08_manual_gt/1779449506181779968.png)
- [1779449508483286784.png](../../../challenge/data/01_real/manual_gt/capturenight08_manual_gt/1779449508483286784.png)

### capturenight09 (9)

- [1779449584709824000.png](../../../challenge/data/01_real/manual_gt/capturenight09_manual_gt/1779449584709824000.png)
- [1779449589380006912.png](../../../challenge/data/01_real/manual_gt/capturenight09_manual_gt/1779449589380006912.png)
- [1779449600421037056.png](../../../challenge/data/01_real/manual_gt/capturenight09_manual_gt/1779449600421037056.png)
- [1779449641015972352.png](../../../challenge/data/01_real/manual_gt/capturenight09_manual_gt/1779449641015972352.png)
- [1779449651923662336.png](../../../challenge/data/01_real/manual_gt/capturenight09_manual_gt/1779449651923662336.png)
- [1779449654325364736.png](../../../challenge/data/01_real/manual_gt/capturenight09_manual_gt/1779449654325364736.png)
- [1779449658861846528.png](../../../challenge/data/01_real/manual_gt/capturenight09_manual_gt/1779449658861846528.png)
- [1779449663431979264.png](../../../challenge/data/01_real/manual_gt/capturenight09_manual_gt/1779449663431979264.png)
- [1779449665933553920.png](../../../challenge/data/01_real/manual_gt/capturenight09_manual_gt/1779449665933553920.png)

### capturepallet09 (3)

- [1778653619758178816.png](../../../challenge/data/01_real/manual_gt/capturepallet09_manual_gt/1778653619758178816.png)
- [1778653793553994496.png](../../../challenge/data/01_real/manual_gt/capturepallet09_manual_gt/1778653793553994496.png)
- [1778653907581160960.png](../../../challenge/data/01_real/manual_gt/capturepallet09_manual_gt/1778653907581160960.png)

### capturepallet08 stale duplicate (2)

- [1778653345465966336.png](../../../challenge/data/01_real/manual_gt/capturepallet08_manual_gt/1778653345465966336.png)
- [1778653498432396288.png](../../../challenge/data/01_real/manual_gt/capturepallet08_manual_gt/1778653498432396288.png)

## 제거하지 않은 review-only 6개

다음 6개는 자동 지표에서 육안 재검토 후보로 드러났지만 오류로 확정될
건이 없어 삭제하지 않았다: `1778652126319943168`, `1778652144496057088`,
`1778652156557165568`, `1778652170735118080`, `1778652172717607680`,
`1778653804674198784`.

## 재현/복구 정보

```bash
# read-only verification
python scripts/annotate/quarantine_invalid_real_gt.py

# registry의 exact SHA가 맞을 때만 격리 이동
python scripts/annotate/quarantine_invalid_real_gt.py --apply
```

스크립트는 source/archive 중 하나만 존재해야 통과하며, 둘 다 있거나 둘 다
없거나 SHA가 다르면 파일을 하나도 이동하기 전에 차단한다.

> `_archive/`는 Git ignore 대상이므로 현재 워크스페이스에서만 복구 가능하다.
> 영구 보관이 필요하면 별도 백업을 만들어야 하며 `git clean -fdX`를 실행하면 안 된다.
