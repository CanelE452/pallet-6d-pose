# Stage B best on wood pallet WITH manual GT (selectivity)

- weights: `weights/paper_s2_stageB/net_epoch_0057.pth` (PAPER_S2 Stage B best)
- data: wood RealSense D435I manual-GT, N=45 (2 sessions), 1280x720
- dims (unseen, user-given): (0.8, 0.59, 0.14) m | K: 1920x1080 에서 16:9 선형유도 → 1280x720
        fx=908.9 fy=909.0 cx=636.4 cy=384.4
- preprocess: **SQUASH** (Stage B training parity, eval_frame_squash)
- GT-good = corner_med(order-free Hungarian) < 10px ; GT-gross = corner_med > 20px or no-det
- ★ N=45 소표본, wood=UNSEEN(치수·텍스처), 2세션, 필터 임계 heuristic.

## Stage B wood 정확도 (GT 45 기준)
```
det (n_det>=6)     : 34/45  (76%)
corner_med  median : 13.9 px  (over det frames)
honest8     median : 12.7 px  (PnP reproj vs GT)
front       median : 13.6 px
rear        median : 13.8 px
GT-good (<10px)   : 6/45  (13%)
GT-gross(>20px)  : 12/45  (27%)
```

## ★ 필터 선택성 (accept vs GT-good confusion, N=45)
```
filter/combo                     n_acc  TP  FP  FN  TN   Prec  Recall
-------------------------------------------------------------------------
f1_peak                             34   6  28   0  11  0.176   1.000
f2_peak_ratio                       33   6  27   0  12  0.182   1.000
f3_flip                              0   0   0   6  39    nan   0.000
f4_tta_stab                         33   6  27   0  12  0.182   1.000
f5_rear_conf                        33   6  27   0  12  0.182   1.000
f6_frsep                            34   6  28   0  11  0.176   1.000
f7_posdepth                         34   6  28   0  11  0.176   1.000
f8_size_env                         28   5  23   1  16  0.179   0.833
f9_bbox_iou                         34   6  28   0  11  0.176   1.000
reproj_selfcons                     20   4  16   2  23  0.200   0.667
-------------------------------------------------------------------------
COMBO strong(f1&f4&f5&reproj)       20   4  16   2  23  0.200   0.667
COMBO all10                          0   0   0   6  39    nan   0.000
reproj-gate only                    20   4  16   2  23  0.200   0.667
f9_bbox_iou only                    34   6  28   0  11  0.176   1.000
```
(TP=accept&good FP=accept&bad FN=reject&good TN=reject&bad ; total GT-good=6)

## per-frame
```
frame                det  cmed  hon8 frnt rear peak flip  tta reproj f9iou #p acc GTg
-------------------------------------------------------------------------------------
183705/000000          8  11.7  11.9 13.6 10.8 0.78   18  0.4    7.5  0.94  9  Y  n
183705/000088          6  17.5  18.1 18.1 17.5 0.83   25  0.4   12.9  0.53  8  n  n
183705/000149          8  15.9  17.3 15.9 17.2 0.84   21  0.3    6.9  0.91  9  Y  n
183705/000213          8  13.4  13.4 10.7 13.9 0.87   22  0.4    8.2  0.92  9  Y  n
183705/000275          8   9.3  12.6  6.2 15.0 0.92   17  0.7    7.1  0.92  9  Y  Y
183705/000335          8  11.1   9.8  7.0 11.7 0.81   16  0.5    6.1  0.92  9  Y  n
183705/000400          6   5.8   9.7  4.4  9.6 0.92   17  0.4    8.4  0.67  9  Y  Y
183705/000460          3   -     -    -    -    -    -    -      -     -    0  n  n
183705/000521          8  14.0  60.6 15.2 11.7 0.62  143  0.5   58.7  0.93  8  n  n
183705/000586          8  15.7  17.5 15.7 15.7 0.79   24  0.6    9.8  0.90  9  Y  n
183705/000646          7  16.3  18.6 19.9 13.8 0.77   23  0.5    6.1  0.80  9  Y  n
183705/000706          5   -     -    -    -    -    -    -      -     -    0  n  n
183705/000768          6  14.7 135.2 15.3 10.4 0.59  147  0.5   35.4  0.52  5  n  n
183705/000828          4   -     -    -    -    -    -    -      -     -    0  n  n
183705/000889          5   -     -    -    -    -    -    -      -     -    0  n  n
183705/000949          8  14.0   9.9 12.1 14.3 0.88   17  0.6   15.2  0.93  8  n  n
183705/001010          7  14.4  12.4 15.3  9.2 0.89   19  0.6    7.8  0.83  9  Y  n
183705/001070          3   -     -    -    -    -    -    -      -     -    0  n  n
183705/001130          7  20.7  12.9 13.7 24.0 0.62   18  2.5   21.8  0.92  8  n  n
183705/001193          6  14.8  13.4 20.0 10.2 0.87   16  0.5    6.4  0.65  9  Y  n
183705/001263          4   -     -    -    -    -    -    -      -     -    0  n  n
183705/001327          8  12.8  10.9 14.6  5.7 0.81   69  0.3    5.0  0.92  8  Y  n
183705/001388          2   -     -    -    -    -    -    -      -     -    0  n  n
183705/001463          4   -     -    -    -    -    -    -      -     -    0  n  n
183705/001523          8  16.4   9.7 12.4 16.7 0.87   19  0.7    9.1  0.93  8  Y  n
184309/000012          8   9.9   9.1 13.9  7.9 0.89   18  1.5    4.7  0.91  8  Y  Y
184309/000072          8  13.1  10.2 10.5 13.3 0.81   21  0.6    4.3  0.89  8  Y  n
184309/000132          8   9.5   4.7 10.8  7.4 0.86   19  0.6    4.2  0.95  9  Y  Y
184309/000192          7  11.9  15.4  7.9 16.2 0.74   19  0.4    8.7  0.82  9  Y  n
184309/000252          8  13.1  11.8 14.2 13.0 0.76   23  1.5    7.9  0.95  9  Y  n
184309/000312          8  10.8   9.7 11.4 10.8 0.88   19  0.4    6.8  0.94  9  Y  n
184309/000372          1   -     -    -    -    -    -    -      -     -    0  n  n
184309/000432          6  15.1 119.2 15.1 15.0 0.80   24  1.3   19.7  0.83  8  n  n
184309/000492          3   -     -    -    -    -    -    -      -     -    0  n  n
184309/000552          0   -     -    -    -    -    -    -      -     -    0  n  n
184309/000612          8   9.3  13.7  9.1 17.1 0.91   17  0.5   11.3  0.98  8  n  Y
184309/000674          8  13.9   6.2 11.8 15.5 0.87   17  1.5   12.9  0.93  8  n  n
184309/000734          8  15.0   6.0 11.4 18.6 0.85   13  1.9   10.8  0.92  8  n  n
184309/000804          6  13.9  14.2 10.9 16.3 0.74   15  1.2    9.1  0.80  9  Y  n
184309/000864          6  14.5  82.8 14.8 12.6 0.73   22  0.5   46.2  0.80  8  n  n
184309/000925          8   9.5   7.9 13.8  8.6 0.90   18  1.0   11.8  0.94  8  n  Y
184309/000987          6  14.0  11.0 13.5 14.0 0.88   19  1.4   11.5  0.69  8  n  n
184309/001081          6  13.4  19.1  9.4 21.1 0.91   15  0.8    8.8  0.69  9  Y  n
184309/001141          8  15.0  13.4 17.8 13.6 0.81   15  1.1   10.0  0.88  7  n  n
184309/001201          7  11.6  36.2 10.1 50.8 0.63   43  5.9   15.3  0.72  7  n  n
```

overlays: TP(accept&good)=4 FP(accept&bad)=16 FN(reject&good)=2, saved 12 -> `wood_gt_overlays/`
f8 GT-envelope: size[0.314,0.709] asp[1.12,3.05]

★ caveat: N=45 소표본, wood UNSEEN, 2세션, 필터 임계 heuristic. 절대수치 과해석 금지.

---

## ★ f3_flip 좌표 조사 (2026-07-10) — "좌표 버그" 가설 검증 결과: 기각

배경: GT-good 프레임까지 f3=16~18px 로 부풀어 f3 통과 0장. δ-sweep 에서 δ*≈+18px
subtract 시 f3 18→4~6px 급감(전 프레임 일정)이라 "un-flip 좌표 offset 버그" 로 의심.
진단 스크립트: `scripts/stage0/_wood_flip_diag.py` (GT 45장, per-corner residual + 3 변형).

측정 (per-corner residual  r = x_n + x_f - (W-1),  N=232 corners):
```
                        median r    std      해석
전체                     18.37px    39.47    (collapse 프레임이 std 키움)
GT-good corners만        16.95px     6.30    ← tight 상수 (systematic)
y-residual (세로축)       0.02px    39.35*   ← 완벽 (flip 세로엔 offset 無)
  r_med / sx(=25.6) = 0.718 cell,  함의 model 수평 bias β = r/2 ≈ 9.2px
```
(*y std 는 collapse 프레임 큰 오차 포함; median 0.02 가 핵심)

핵심 판정 — **좌표 버그 아님, model L-R non-equivariance**:
- extract_keypoints_from_belief 는 x·y 를 완전 대칭 처리(+OFFSET 동일)인데 residual 은
  x=+18px / y=0 → 비대칭은 decode/extract 가 만들 수 없음 = **model belief 출력 자체**.
- 사용자 제안 "belief-space mirror 후 동일 decode"(OFFSET 원리적 상쇄)를 구현·측정 →
  residual **+20.5px** (오히려 +2px 악화). 즉 가장 원리적인 좌표-정합 un-flip 으로도
  offset 안 사라짐 = 좌표 규약 문제 아님.
- 세 un-flip 규약 residual: `W-x` 17.4 / `(W-1)-x` 18.4 / belief-mirror 20.5. 스프레드 ~3px
  (좌표 자유도) 아래에 ~18px model bias 가 공통으로 깔림.
- 원인: 학습에 horizontal-flip aug 미사용(프로젝트 규칙, L↔R corner 순서 파괴 방지) →
  model 이 flip-equivariant 하지 않음(절대 L-R 위치 단서 학습). flip-consistency 필터의
  전제(근사 flip 대칭)를 이 model 이 구조적으로 위반.

적용한 수정 (원칙적, 매직상수 無): `flip_infer_squash` un-flip 을 `W - x` →
**`(W-1) - x`** (cv2.flip 정확한 역변환 = pixel-flip-axis 규약). 1px 교정. **f3 는
사실상 불변**(~17→~18px, 오히려 +1). ★ +18 하드코딩(δ-sweep 값)은 = model bias 빼기 =
fitted 상수(금지) 이고, 아래처럼 빼도 선택성 안 생김.

de-bias(+18 subtract) 로 f3 를 억지로 낮춰도 선택성 無 (GT-good vs GT-bad 완전 중첩):
```
frame                cmed   GTgood   f3_debias    (낮을수록 accept)
000400                5.8      Y        10.7
000132                9.5      Y         4.5
000312               10.8      n         4.1   ← bad 인데 good 보다 낮음
000000               11.7      n         4.2   ← bad
001193               14.8      n         4.1   ← bad
000149               15.9      n         4.3   ← bad
000521(collapse)     14.0      n       132.5   ← collapse 만 튐(reproj 이미 잡음)
```
GT-good f3_debias 범위 [4.5,10.7] 가 GT-bad [4.1,14.5] 안에 완전히 포함. f3 는 L-R 대칭
검사라 wood 의 지배적 실패모드(depth/rear collapse = 앞뒤 축)를 원리적으로 못 봄
(프로젝트 memory: "rear 가 병목", "depth collapse" 와 정합). collapse(앞뒤 붕괴)만 f3 폭증
하는데 그건 reproj_selfcons 가 이미 잡음.

### 결론 (정직 판정 — 이전 결론 유지)
- f3 의 16~18px 은 **좌표 버그가 아니라 model 의 좌우 비대칭(no-flip-aug)**. 원리적 좌표
  수정(belief-mirror/W-1)으로 **4~6px 로 못 내림**(belief-mirror 오히려 20px).
- 4~6px 은 model bias(+18)를 빼야 나오는데 이는 fitted 상수(금지)이며, **빼도 all10 통과
  프레임 0장·선택성 개선 0** (GT-good/GT-bad f3 중첩). 
- 따라서 **"f3 선택성 없음 / all10 통과 0장" 결론은 바뀌지 않음.** 다만 근본 원인을 교정:
  (구) "un-flip 좌표 offset 버그" → (신) "model L-R non-equivariance + flip 이 depth/rear
  실패모드에 blind". 유효 게이트는 reproj_selfcons(collapse 포착)이며 f3 는 이 model 에서
  배포 필터로 부적합.
- caveat: N=45 소표본·unseen·heuristic 임계. β·residual 은 이 weight(Stage B ep0057)
  고유값(다른 weight 재측정 필요).
