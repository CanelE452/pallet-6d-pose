# COMPARISON — F0 (indexed) vs F1 (C4)

같은 INIT · 같은 데이터 · 같은 recipe.  **다른 것은 keypoint symmetry loss 하나뿐**이다.

```
INIT      runs_paper/yolo26n_paper_generic_v1_seed42/weights/last.pt
          sha256 6a40a4d430fd205a427e38a1927aad2a0a0bef20b984484e0b77318e7bd355ea
데이터     datasets/live_gt_v4   train 3,087 / val 155
recipe    40 epoch · batch 32 · SGD lr0 0.01/lrf 0.01 · cos_lr · seed 42 · patience 0
학습시간   F0 14.7분   F1 14.8분

F0 best.pt  sha256 a4606fc5c39717e9…
F1 best.pt  sha256 42dbfc01ef181d56…
```

---

## A. Detection (val 155)

```
              box mAP50-95   pose mAP50-95   precision   recall
F0               0.9615         0.9297        0.9860     1.0000
F1               0.9551         0.4112        0.9792     0.9871
```

`pose mAP50-95` 는 **indexed 기준**이라 270도 프레임을 통째로 오답 처리한다.  아래 D 를
보기 전에 이 숫자만으로 판단하면 안 된다.

## B. fixed-index keypoint (기존 번호 그대로)

```
        검출      median    p90      <5px     >20px
F0     152/155    2.00px    5.23     88.8%     4.6%
F1     155/155    3.54px  139.73     57.4%    40.0%
```

## C. C4-equivalent keypoint (네 permutation 중 최소)

```
        median    p90      <5px     >20px
F0      1.96px    4.32     91.4%     1.3%
F1      2.08px    3.60     96.8%     0.0%
```

## D. symmetry rescue / true collapse

```
                              F0      F1
symmetry-rescued (fixed>20 & C4<5)     4      61
true collapse    (C4 > 20px)           2       0
```

**F1 의 fixed-index 실패 62 프레임을 분해하면 전부 270도다.**

```
best_deg 분포        {270: 62}
그 프레임의 C4 오차   median 1.97px   max 5.92px
```

위치는 정확하고 **번호만 270도 돌았다**.  §12 가 말한 "fixed-index 가 나빠졌지만 그
차이가 전부 equivalent permutation 인 경우" 에 해당하므로, localization 실패로 세지
않는다.

## E. 선택된 permutation (val)

```
        0도    90도   180도   270도
F0     147      2      0       3
F1      93      0      0      62
```

F1 은 세션에 따라 다른 면을 앞면으로 고른다.  val 은 여러 세션에서 6 장마다 뽑은
것이라 이 분포가 이봉으로 보이지만, **한 세션 안에서는 그렇지 않다** — 아래 F.

---

## F. downstream phase 안정성 (GT 미사용, 5 세션)

이웃한 두 프레임의 **예측끼리만** 비교해 "다음 프레임이 이전 대비 몇 도 돌아
보이는가" 를 셌다.  연속 촬영이면 정답은 항상 0도다.

```
세션                          F0 pairs / 안정률      F1 pairs / 안정률
forklift_v4_20260904_142318      199   100.0%           197   100.0%
forklift_v4_20260904_103429       82    92.7%            94    98.9%
forklift_v4_20260904_105615      119   100.0%           119   100.0%
forklift_v4_174925               119   100.0%           119   100.0%
capture_20260902                 119   100.0%            98    99.0%
─────────────────────────────────────────────────────────────────────
합계                             638    99.06%          627    99.68%
```

전면 중심(0~3 평균) 프레임 간 이동도 F1 이 같거나 작다 (예: 142318 p90 2.6 vs 1.8px,
103429 p90 2.9 vs 9.2px).

**즉 F1 도 한 접근 시퀀스 안에서는 face phase 가 튀지 않는다.**  E 의 이봉 분포는
세션 사이의 차이지 세션 내부의 흔들림이 아니다.

---

## 판정 (§12 우선순위)

```
1  진짜 localization collapse       F0 2  →  F1 0            F1 우위
2  newauto downstream 호환          F0 99.06% / F1 99.68%     F1 근소 우위, 둘 다 안전
3  C4-equivalent p90 / gross20      F1 3.60 / 0.0%            F1 우위
                                    F0 4.32 / 1.3%
4  fixed-index diagnostics          F0 우위 (차이는 전부 270도 permutation)
5  Ultralytics pose mAP             F0 우위 (같은 이유)
```

## ★ 이 결과를 과신하면 안 되는 지점

* **F1 의 우위는 소표본에 걸려 있다.**  true collapse 2 → 0, 미검출 3 → 0,
  phase 불안정 6 → 2.  전부 한 자릿수 사건이라 "C4 가 확실히 낫다" 고 말하기에는
  근거가 얇다.  seed 하나짜리 실험이다.
* **val 은 interleave** 라 같은 세션이 train 에 들어 있다.  과제 트랙 목적에는 맞는
  평가지만 처음 보는 현장 성능이 아니다.
* `capture_20260902` 에서 F1 이 14 프레임을 놓쳤다 (F0 는 0).  검출 자체는
  전체적으로 F1 이 나은데 이 세션만 반대다 — 원인 미확인.
* 동료가 "많이 흔들렸다" 고 한 세션(`..._190700`)은 이 저장소에 없다.  그 세션에서
  재보지 못했다.
