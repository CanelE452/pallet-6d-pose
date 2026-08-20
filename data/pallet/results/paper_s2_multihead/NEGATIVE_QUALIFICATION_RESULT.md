# NEGATIVE QUALIFICATION RESULT — N0 vs N1

```
NEGATIVE_HANDLING_SUPPORTED = False
실패 절                      pose safety 하나 (negative·detection 은 두 seed 모두 통과)
LINE_ISOLATION_EXACT         True  (3,000 step 후 line_late max|diff| = 0.000e+00, 두 seed)
```

## 설정

```
q* = 0.20 (gradient-normalized)   lambda = q*/r_s → seed1 0.003245 / seed2 0.035178
단일 backward / 단일 step          negative 전용 step 없음
positive  Corner=Line=MH_TRAIN     negative  NEGATIVE_SYNTH_V1 train 9,000, B_neg=2 (별도 batch)
평가      positive D2_MH_DEV512 512 / negative synth-dev 1,000
solver    F3 (rotation-only fusion + point-only t-refit)
```

## 결과

```
seed1        R med   t med   5cm5    ADD-S   AUROC   AUPRC   recall  FP/img
  N0         6.317  0.2188  0.1289  0.2175  0.9546  0.9010  0.951   0.1670
  N1         6.447  0.1943  0.1270  0.1928  0.9764  0.9547  0.951   0.0990
seed2
  N0         6.086  0.1770  0.1582  0.1871  0.9446  0.8833  0.951   0.1850
  N1         7.326  0.2361  0.1309  0.2434  0.9894  0.9853  0.951   0.0220
```

## gate

```
                        seed1                        seed2
negative FP 감소        +40.7%   OK                  +88.1%   OK
AUPRC                   +0.0538  OK                  +0.1021  OK
detection recall drop    0.00pp  OK                   0.00pp  OK
pose R                  -2.06%   OK                 -20.38%   NG
pose t                 +11.19%   OK                 -33.39%   NG
pose ADD-S             +11.36%   OK                 -30.10%   NG
5cm5deg                 -0.19pp  NG                  -2.73pp  NG
→                       FAIL                         FAIL
```

## 읽기

### negative 쪽은 확실히 작동한다

FP/image 가 seed1 0.167→0.099 (**-40.7%**), seed2 0.185→0.022 (**-88.1%**) 이고 AUPRC 가
두 seed 모두 크게 오른다(+0.054, +0.102). **recall 은 0.951 로 정확히 유지**된다
(drop 0.00pp, 두 seed). gate 의 negative·detection 절은 전부 통과다.

### 떨어뜨린 것은 pose 뿐이고, seed 가 정반대다

```
          R        t       ADD-S
seed1  -2.06%  +11.19%   +11.36%    ← translation·ADD-S 는 오히려 개선
seed2 -20.38%  -33.39%   -30.10%    ← 전면 악화
```

seed1 은 5cm5deg 만 -0.19pp(사실상 0)로 걸렸고 나머지는 통과이거나 개선이다.
seed2 는 pose 가 광범위하게 무너졌다. **같은 개입 강도 `q`=0.20 인데 방향이 반대다.**

주목할 점: seed2 는 negative 억제가 가장 강했던 쪽이다(FP -88.1%, AUPRC 0.985).
즉 이 데이터에서는 **negative 억제가 강할수록 pose 가 상한다**는 관계가 보인다
[추정 — seed 2개, 인과 미확인].

가능한 기전 하나: corner belief 를 dense 하게 0 으로 누르면 positive 프레임에서도
peak 가 낮아지고, softargmax·PnP 가 쓰는 국소 형상이 함께 눌린다. seed2 의
`lambda` 가 seed1 보다 10.8배 큰 것(0.0352 vs 0.0032)도 이 방향과 정합한다 —
`q` 는 gradient 비를 맞췄지만 belief 절대 크기에 주는 영향까지 맞추지는 못한다.

## 잔여 FP 의 소재 (ADDENDUM 4)

N1 의 negative score 상위 100장 구성:

```
seed1   N1_STRUCTURAL_HARD 84 · N2_PALLET_LIKE_HARD 16
seed2   N1_STRUCTURAL_HARD 77 · N2_PALLET_LIKE_HARD 18 · N0_MATCHED_EMPTY 5
```

**잔여 FP 는 팔레트를 닮은 N2 가 아니라 구조물 N1 에 몰린다.** corner head 가
"팔레트처럼 생긴 물체" 보다 **평행 rail·교차 topology 자체**에 반응한다는 뜻이다.
빈 장면(N0)은 거의 완전히 억제된다.

contact sheet: `fig_hard_negative_top100_seed1.png`,
목록: `negative_hard_top100.json` (frame_id · negative_type · score_4kp · rank)

## 산출 그림 (ADDENDUM 5)

```
fig_score_pos_neg_seed{1,2}.png                positive / negative score_4kp ECDF (N0 vs N1)
fig_score_negative_categories_seed{1,2}.png    category 3종 ECDF
fig_hard_negative_top100_seed1.png             상위 100장 contact sheet
```

## 판정과 다음

```
NEGATIVE_HANDLING_SUPPORTED = False
ARCHITECTURE                = NOT_LOCKED   (negative path 미확정)
```

브리프 규정대로 **여기서 멈춘다** — λ 재튜닝·새 head·hard-negative 재생성을 자동으로
하지 않는다. 실패 절은 pose safety 하나로 특정되었고, negative·detection 절은
두 seed 모두 통과했다는 것이 이번 결과의 내용이다.

## 한계

- **이번 negative 는 synthetic 이다.** "real-world false positive 해결" 로 인용 금지.
  현재 결과는 synthetic negative suppression qualification 까지다.
- seed 2개. pose 방향이 정반대라 평균으로 말할 수 없다.
- positive 평가는 D2_MH_DEV512 512장이다.
- 3D IoU 는 구현하지 않았다. oriented-box intersection 을 검증된 형태로 만들지 못했고,
  axis-aligned 근사를 "3D IoU" 라는 이름으로 보고하지 않기로 했다. ADD/ADD-S 로 대체했다.
