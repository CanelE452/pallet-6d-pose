# STAGE22 — rear 붕괴 3각 진단 종합 (coord-diag chain)

3-파트 무정지 체인. 목적: DOPE B2(`stage11_16k_B2_maskaux/ep0084`)의 rear(4-7) 코너 붕괴 원인을 3 각도로 진단.
convention=camera-facing 0123. real inference=reflect-pad(pad=100, PART B pass1만 aspect-only). final-test(p07/p09,n08/n09) 봉인 유지.

---

## PART A — "윗면 보임" 앙각 실측 (기록 재사용, 학습·재추론 X) [완료]
질문: "윗면 잘 보이는데 왜 rear 안 되나"의 데이터 답.
스크립트 `scripts/stage0/stage22_partA_elevation_band.py`, 산출 `partA/`.

```
elev bin   n   band_med(px)  front_med  rear_med  rear_spk
────────────────────────────────────────────────────────
  <3       34    1.0          11.9       19.2      0.77
  3-8      58   17.2          11.1       18.5      0.72
  8-15      1   37.5          13.4       53.1      1.0
  25+       5  301.8          14.1        9.8      0.4   (전부 CAD)
```
- real 92/98(94%)이 <8° edge-on. 저앙각 band_med=12.8px(윗면=얇은 sliver), <3°는 band 1px(윗면=선).
- 윗면이 진짜 보이는(band 300px, ≥25°) 5장은 전부 CAD → 거기선 rear 정확(9.8px). ★단 CAD=appearance confound.
- spearman elev↔band=0.895(강), band↔rear_err=0.124, elev↔rear_err=0.075.
- 눈검증: elev5.9° outside=얇은 sliver rear43px / elev35° CAD=윗면 완전보임 rear6.8px [확인].

**판정**: "윗면 보인다" 인상은 실측과 불일치. real은 대부분 저앙각 edge-on → 윗면 실제로 안 보임 → rear depth collapse.
즉 "윗면 보이는데 rear 실패"가 아니라 "저앙각이라 윗면 안 보임 → rear 붕괴". flat-view depth collapse 재확인.

## PART B — crop-and-refine 유효해상도 진단 (추론 X) [완료, CONFOUNDED negative]
질문: 유효 해상도(short-side 400→belief 50)가 rear 병목인가?
스크립트 `scripts/stage0/stage22_partB_crop_refine.py`, 산출 `partB/`.

- REAL same-frame 페어(N=14 rear): refine이 rear **+78.5px 대폭 악화**(front +63.5px). rear gross율 0.33→0.93.
- ★기전 [확인]: 저앙각 flat 팔레트 tight bbox=얇은 strip(aspect~5, 예 458×83). short-side→400 resize시 pass2 입력이 1104×400·2200×400 극단 wide strip=OOD → pass2 붕괴. 좌표역매핑은 정상(reproj 페어 검증, 버그 아님).

**판정**: crop-refine(지정 프로토콜)은 flat edge-on 물체엔 원리적 부적합 → 유효해상도 가설 clean test 실패(CONFOUNDED, 스케일/aspect 분포 이탈). "해상도 기각" 단정 금지. 병목 질문 OPEN. 부수확인: flat PnP reproj는 rear 126px인데도 0.4px(degenerate)=honest 아님.

## PART C — coord loss paired ablation (2-arm 파일럿, 학습 ~1h) [완료]
질문: coord loss가 rear를 고치나? 슬라이드(CoordDOPE) 방어숫자.
런처 `partC/run_partC_arm.sh`, eval `scripts/stage0/stage22_partC_eval.py`, weights `weights/stage_screens/stage22_coord_pilot/{control,coord}/`.
2-arm: control=B2 recipe(2:1:1, trunc제외) +8ep / coord=+vis_coord_loss(soft-argmax coord Huber/GT-2D-diag, vis-weighted, λ0.24 ramp800). λ 캘리 검증: vis/bel 0.072~0.079(~7.5%, loss-ratio proxy).
ckpt=syn val로 둘 다 ep0085(val이 ckpt 거의 구분못함). real 145(pad100), 페어 N=97.

```
metric   control  coord   delta    imp/wrs    유의성
────────────────────────────────────────────────────
rear     19.86    17.64   -0.81    45/23      sign p=0.010 ★
front    11.71    13.63   +0.78    20/42      (front 악화)
full8    16.34    15.68   -0.07    25/18      ~무변화
rear good(<10)  0.04→0.12(3x)   gross(>20) 0.49→0.45
V=8 rear 19.71→17.64(무회귀)    front 11.51→13.46
elev bin rear: <3 -0.75 / 3-8 -1.28 / 8+ -0.77 (전 구간 개선)
μ↔argmax(coord rear): rear 0.95 vs front 0.73 belief px (둘 다 sub-pixel=병리 없음)
```

**판정 (task 3-way 중)**: (i)/(ii) 경계 — rear 페어는 **방향-유의(sign p=0.010)**하게 개선(signal)이나 매그니튜드 작고(-0.8px) front 비용 있어 full8 flat(screening "weak" 정합). 병리진단상 soft-argmax μ 병리 없음(둘 다 <1px)→"국소 5×5 변형" 후속 미동기. novelty=screening에 없던 front/rear 분리로 coord=**targeted rear regularizer**(front→rear precision 재배분) 규명.
슬라이드 방어숫자: rear -0.8px(p=0.01)+good-rate 3x+V=8 무회귀 → "표준 coord(Integral Pose Regression 계열)의 소폭 rear 기여"로 정직 하향(단독 rear 해결책 아님).

---

## 종합 결론 (3-part 정합)
세 각도 모두 **같은 결론**을 가리킨다:
1. **PART A**: real rear 붕괴의 근본 = 저앙각 edge-on(윗면 안 보임)→depth collapse. "윗면 보임" 인상은 착각.
2. **PART B**: 유효해상도는 clean test 불가(flat 물체 crop=OOD). 해상도가 병목이란 증거 없음(OPEN).
3. **PART C**: coord loss는 rear를 방향-유의하게(작게) 개선하나 근본 해결 아님. soft-argmax 병리도 없음.

→ rear 천장의 진짜 레버는 **저앙각 flat-view 데이터/appearance**(PART A·기존 trackb 정합)이지, 해상도(B)나 loss 표현(C)이 아니다. coord는 보조 정규화자로 소폭 기여(슬라이드 방어용 숫자 확보).
★ 모든 real 수치 소표본(rear 페어 N=14~97)=예비. CAD 고앙각 appearance confound. λ=loss-ratio proxy(≠gradient).
