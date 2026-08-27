# GT DATA AUDIT — 저장된 어노테이션 결과물 감사

도구: `scripts/annotate/audit_gt_data.py`
실행 2026-08-27, 수동 GT **702장** (28 폴더).

```
python scripts/annotate/audit_gt_data.py \
  --glob "challenge/data/01_real/**/*_manual_gt" \
  --glob "challenge/data/01_real/**/pallet11_gt" \
  --out _docs/audits/gt_data_audit
```

## 왜 만들었나

`_audit_annotate.py` 는 어노 **코드**를 검사한다. `qa_risk.py`(repo 밖, 일회성)는 클릭점에서
PnP 를 다시 풀어 본다. **둘 다 저장된 `pose_transform` 자체는 검사하지 않는다** —
"PnP 가 제대로 안 된 채로 저장된" 경우를 놓친다.

이 도구의 핵심 검사(T2+T3):

```
저장된 pose_transform 을 투영한 결과  vs  저장된 projected_cuboid
                                     vs  같은 점에서 다시 푼 PnP
```

세 값이 어긋나면 **파일 안의 두 값이 서로 다른 것을 말하고 있다**는 뜻이다.

## 결과

```
severity      RED 311   AMBER 48   GREEN 343
```

RED 의 대부분(243)은 `pallet11_gt` 이고, 이건 이미 폐기된 셋이다
(memory `pallet11-gt-apriltag-broken-do-not-use`). `manual_kps` 와 `reproj_error_px`
필드 자체가 없다.

**pallet11 을 뺀 459장 기준 실제 flag:**

```
MANUAL_KPS_NONE_x*        49    manual_kps 에 None 이 섞임 (클릭 안 된 점)
SENTINEL_x*               30    projected_cuboid 에 [-1,-1] 이 남아 있음
STORED_POSE_MISMATCH       8    ★저장 pose 가 점에서 푼 해와 어긋남
```

## ★ STORED_POSE_MISMATCH — 사용자가 말한 그 문제

8건이지만 고유 프레임은 **6개**다. `_outside_eval_manual_gt` / `_night_eval_manual_gt` 가
원본 capture 폴더의 프레임을 **복사해 모은 셋**이라 같은 frame_id 가 두 번 잡힌다
(이 사실 자체가 `FT_EVAL_LEAK.json` 이 지적한 구조와 같다).

```
folder                      frame                    T2      T3   excess    dR°     dt m
──────────────────────────────────────────────────────────────────────────────────────────
capturepallet08_manual_gt   1778653498432396288    8.758   6.513   +2.245  10.079   0.2464
capturepallet09_manual_gt   1778653619758178816    8.869   7.569   +1.300   5.911   0.0618
_outside/_pallet08          1778653524402450688    2.640   2.930   -0.290   0.790   0.0016
_night/capturenight07       1779449325409244928    2.929   3.922   -0.993   0.626   0.0017
capturenight09_manual_gt    1779449584709824000    5.131   5.168   -0.037   0.566   0.0017
_night_eval_manual_gt       1779449343555269120    1.475   2.783   -1.308   0.541   0.0019
```

**두 부류로 갈린다.**

```
심각 2건   dR 5.9 ~ 10.1도, excess > 0
           저장 pose 가 점이 허용하는 최선보다 **나쁘다** — 내부 불일치가 실재한다
경미 4건   dR 0.5 ~ 0.8도, excess < 0
           저장 pose 가 오히려 재풀이보다 좋다.  저장 시점에 9점(centroid 포함)이나
           약간 다른 점 집합으로 풀었을 가능성이 크다 `[추정]`.
           현재 임계(0.5도)가 빡빡해서 잡힌 쪽이다.
```

## ★ 도구 검증 — 사람 리뷰와 대조

```
frame                   GT-QA 제외    정본 140 포함
1778653619758178816       ★YES           -        도구·사람 일치
1779449584709824000       ★YES           -        도구·사람 일치
1778653498432396288        no          ★포함       ★사람 리뷰가 놓친 것
```

6건 중 2건은 사람 GT-QA 가 이미 뺐고, **가장 심한 1건은 아직 정본 140 안에 살아 있다.**
오버레이: `overlays_suspect/capturepallet08_manual_gt__1778653498432396288.png`
(빨강 = 저장 pose / 파랑 = 재풀이 / 초록 = 저장된 점)

원거리·소형 프레임이라 클릭 오차가 pose 로 크게 증폭된 경우다. 자동 수정은 하지 않았다.

## ★ 개발 중 잡은 내 오탐 (기록)

첫 판은 T2 를 **절대 잔차 2px** 로 판정해 `capturepallet07` 27장 중 **15장을 RED 로 오탐**했다.
확인해 보니 그 15장은 전부 `dR = 0.0도 / dt = 0.0m` — 저장 pose 가 점에서 푼 해와
**정확히 같았다.** 잔차 2~6px 은 저장 버그가 아니라 클릭 정밀도였다.

→ 판정을 고쳤다:

```
STORED_POSE_MISMATCH   저장 pose vs 재풀이 pose 의 dR/dt, 그리고 excess(T2-T3)
                       즉 "저장 pose 가 점이 허용하는 최선보다 나쁜가"
GROSS_RESIDUAL         재풀이 잔차의 절대 임계는 10px 로 올림 (gross 만 hard)
중간 구간              T5 robust z 가 셋 안에서 상대적으로 판정
```

수정 후 같은 폴더는 27/27 GREEN 이 됐다.

**교훈: 절대 임계로 어노테이션 품질을 판정하면 정상을 대량 오탐한다.**
같은 데이터로 푼 두 해를 비교하는 상대 판정이 맞다.

## 남은 것 / 다음

```
□ 심각 2건 처리        정본에 남은 1778653498432396288 을 어떻게 할지 결정 필요
                       (제외 / 재어노테이션 / 그대로 두고 한계로 기록)
□ SENTINEL 30건        화면 밖 코너가 [-1,-1] 로 저장된 것.  memory
                       `annotate-tool-audit-and-sentinel-gt-damage` 의 그 문제.
                       ★복구는 "이미 맞는 값" 검증 후에만.
□ MANUAL_KPS_NONE 49   클릭 안 된 점이 있는 프레임.  외삽 의존도가 높다는 뜻
□ 경미 4건 재검토       임계 0.5도가 적절한지 — 저장 시점 점 집합을 확인하면 갈린다
□ 어노 직후 자동 실행   이 도구를 어노테이션 저장 직후에 돌리면 그 자리에서 잡힌다
```

## 산출물

```
GT_DATA_AUDIT.csv              프레임별 전체 지표 + severity + 사유
GT_DATA_AUDIT_SUMMARY.json     집계 · 임계값 · 검사 정의
overlays_suspect/              심각 2건 오버레이
```
