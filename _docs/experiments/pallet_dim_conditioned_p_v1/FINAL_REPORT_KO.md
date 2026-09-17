# Dimension-Conditioned Symmetry-Aware P v1 — 완료 보고

R0는 완전 고정했다. 기존 P의222 후보·반경0.08 bbox diagonal·center8·검출 후보/box/score/order/index를 유지했다. 새 refiner30회 ×6000 =180,000 updates. 유효 smoke200, 보존된 무효 smoke100, 폐기한 CPU adapter18 one-step updates는 별도다(총 optimizer 호출180,318).

중요한 계약 정정: 참고 exporter의 G38 dimensions_m은 canonical이 아니라 camera-facing였다. 치수 없는 N0/N1 6회는 유지하고, DIM 본학습은 한 step도 실행하기 전에 고정 renderer XYZ로 정정·전수 검증했다. 과거 입력·normalization·smoke와 중단 경위는 CANONICAL_DIMENSION_CORRECTION.json 및 history에 보존했다. GT pose를 보고 보이는 W/D를 입력하는 방식은 사용하지 않는다.

치수 입력은 canonical [W,D,H]에서 만든 [logW,logD,logH,log(W/D),log(H/sqrt(WD))], paper TRAIN55,980행 mean/std로 표준화했다. META는 C1/C2/C4 one-hot을 추가한다. G38/legacy는 고정 renderer XYZ를 사용했고 G38은 원본 builder와 40,000행 전수 대조했다. raw dimensions_m은 camera-facing라 입력에서 배제했다. 실사는 registry만 사용했다. **실사 object_type이 외부에서 알려져 있다는 조건부 결과**이며 unknown-type deployment는 지원/측정하지 않았다. GT pose 기반 W/D swap·추론 GT branch는 없다.

Loss target은 frozen R0 phase에 가장 가까운 허용 whole-object GT tuple을 TRAIN에서만 선택한다. 같은 tuple의 좌표와 mask가 함께 움직이며, corner별 자유매칭은 없다. Calibration은 모든 arm에 동일한 기존 fixed-index Gaussian target CE를 사용했다. 원본 P lambda/cap 고정, temperature만 synthetic calibration에서 선택했다.

## SYNTH_HELDOUT

3-seed 지표 평균. median/P90은 matched corner8, E_sym/PCK는 missing penalty를 포함한 전체 GT 분모.

| arm | median px | P90 px | E_sym | PCK10 | translation median cm |
|---|---:|---:|---:|---:|---:|
| OLD_P | 1.7952 | 6.8105 | 0.00890344 | 0.9372 | 2.4213 |
| N0_BASE_REPLAY | 1.7849 | 6.7833 | 0.00889908 | 0.9372 | 2.4134 |
| N1_SYM_ONLY | 1.7832 | 6.8119 | 0.00889597 | 0.9371 | 2.4238 |
| N2_DIM_ONLY | 1.7734 | 6.7964 | 0.00887695 | 0.9375 | 2.4113 |
| N3_DIM_SYM | 1.7714 | 6.7886 | 0.00887716 | 0.9378 | 2.4199 |
| N4_META_SYM | 1.7629 | 6.7568 | 0.00886965 | 0.9376 | 2.4565 |

| contrast | delta | CI95 | improving seeds | verdict |
|---|---:|---|---:|---|
| N1_SYM_ONLY__minus__N0_BASE_REPLAY | -3.1072741e-06 | [-6.962953776531673e-06, 7.482193572471144e-07] | 2/3 | UNRESOLVED |
| N2_DIM_ONLY__minus__N0_BASE_REPLAY | -2.2120869e-05 | [-3.024464900586725e-05, -1.3881243031148183e-05] | 3/3 | UNRESOLVED |
| N3_DIM_SYM__minus__N2_DIM_ONLY | 2.0935122e-07 | [-3.3738931151000656e-06, 3.8216287952282105e-06] | 1/3 | UNRESOLVED |
| N4_META_SYM__minus__N3_DIM_SYM | -7.5157204e-06 | [-1.0720883251301456e-05, -4.2549119174331045e-06] | 3/3 | SUPPORTED |
| N4_META_SYM__minus__N0_BASE_REPLAY | -2.9427239e-05 | [-3.829898721387071e-05, -2.019075345554412e-05] | 3/3 | UNRESOLVED |
| N0_BASE_REPLAY__minus__OLD_P | -4.3660935e-06 | [-7.148534771572897e-06, -1.5876263724452065e-06] | 3/3 | UNRESOLVED |
| N4_META_SYM__minus__OLD_P | -3.3793332e-05 | [-4.266737650768706e-05, -2.448639201029309e-05] | 3/3 | UNRESOLVED |

OLD_P 재현 사전 tolerance 통과: True. CUDA grid_sample backward 비결정성 때문에 bit-exact 재학습을 주장하지 않는다.

## REAL_DEV

3-seed 지표 평균. median/P90은 matched corner8, E_sym/PCK는 missing penalty를 포함한 전체 GT 분모.

| arm | median px | P90 px | E_sym | PCK10 | translation median cm |
|---|---:|---:|---:|---:|---:|
| OLD_P | 5.9381 | 42.6314 | 0.04868357 | 0.6749 | 7.1530 |
| N0_BASE_REPLAY | 5.9439 | 42.8723 | 0.04868298 | 0.6753 | 7.2604 |
| N1_SYM_ONLY | 5.9210 | 42.5135 | 0.04867357 | 0.6756 | 7.2742 |
| N2_DIM_ONLY | 5.7777 | 42.4595 | 0.04842188 | 0.6859 | 7.0106 |
| N3_DIM_SYM | 5.7782 | 42.1338 | 0.04841909 | 0.6859 | 7.0676 |
| N4_META_SYM | 5.7621 | 41.8581 | 0.04842552 | 0.6891 | 6.9824 |

| contrast | delta | CI95 | improving seeds | verdict |
|---|---:|---|---:|---|
| N1_SYM_ONLY__minus__N0_BASE_REPLAY | -9.4077897e-06 | [-2.378753060040831e-05, 4.7305494490299545e-06] | 3/3 | UNRESOLVED |
| N2_DIM_ONLY__minus__N0_BASE_REPLAY | -0.00026109594 | [-0.00039284778302907885, -0.00016566375963459573] | 3/3 | UNRESOLVED |
| N3_DIM_SYM__minus__N2_DIM_ONLY | -2.7939156e-06 | [-2.9129826018249672e-05, 1.5664908027579418e-05] | 1/3 | UNRESOLVED |
| N4_META_SYM__minus__N3_DIM_SYM | 6.4366033e-06 | [-3.3424666805809696e-05, 4.112319408436988e-05] | 1/3 | UNRESOLVED |
| N4_META_SYM__minus__N0_BASE_REPLAY | -0.00025745325 | [-0.0004267878892985505, -0.00013572174480976556] | 3/3 | SUPPORTED |
| N0_BASE_REPLAY__minus__OLD_P | -5.9691019e-07 | [-1.3038824819410525e-05, 1.3799765381329505e-05] | 2/3 | UNRESOLVED |
| N4_META_SYM__minus__OLD_P | -0.00025805016 | [-0.0004285313653776234, -0.00013520546788037494] | 3/3 | SUPPORTED |

OLD_P 재현 사전 tolerance 통과: True. CUDA grid_sample backward 비결정성 때문에 bit-exact 재학습을 주장하지 않는다.

## 판정 요약과 UNRESOLVED의 뜻

실사 N4−N0: SUPPORTED, ΔE_sym=-0.00025745325, CI95=[-0.0004267878892985505, -0.00013572174480976556], 3/3 seed 개선. 알려진 object_type 조건에서의 작고 제한적인 개선이며 재사용 DEV 결과다.
실사 N4−N3: UNRESOLVED. 따라서 explicit symmetry code의 필수성이나 보편적 우월성을 주장하지 않는다. Target-only 증분도 각 contrast를 따로 해석한다.
UNRESOLVED는 실행 실패가 아니다. CI가0을 포함하거나, CI가 개선 방향이어도 사전 안전 기준을 통과하지 못한 경우다.
실사 N2−N0의 canonical-aligned good<5→bad>10 합계는 1건, 역방향은 0건이다. 평균 개선과 별개로 이 안전 조건을 확인해야 한다.
합성 N4−N0 gross20 변화는 +0.02335 percentage points다. 따라서 평균/CI만으로 전체 성공을 선언하지 않는다.
6D 표는 기술통계이며 paired superiority 검정을 추가로 주장하지 않는다. Translation/rotation/ADD와 IoU3D가 같은 방향인지 각 지표를 확인해야 한다.

## Metadata 민감도 — 실사 3-seed 평균

| 진단 | ΔE_sym vs correct | 평균 좌표 변화 px | top1 변화율 |
|---|---:|---:|---:|
| TRAIN mean dims | +0.00027276 | 0.8690 | 0.3357 |
| frozen shuffled dims | +0.00029684 | 0.8094 | 0.2271 |
| wrong group | -0.00002270 | 0.0907 | 0.0664 |
| zero standardized metadata | +0.00024381 | 0.8493 | 0.3246 |

치수 perturbation은 실제 출력 변화 진단이지 인과 증명은 아니다. Wrong-G가 나빠지지 않는 결과도 그대로 보존했다.

## TRAIN target locality

Paper TRAIN 계약 수: C1=19268, C2=36712, C4=0. 사용 가능 matched C2 중 nonidentity phase는 125/36664.
반경 내 코너 비율: C1 96.899%, C2 98.598%. Paper에서는 전후 비율이 동일하다.
Square C4 matched TRAIN=695; branch counts={'0': 657, '3': 33, '1': 5}. 반경 내 비율 94.374%→99.639%. 반경은 확대하지 않았다.
Square 세부 결과: [SQUARE_SECONDARY_REPORT.md](SQUARE_SECONDARY_REPORT.md). Mixed 세 집단별 결과: [MIXED_ROUTING_DIAGNOSTIC.md](MIXED_ROUTING_DIAGNOSTIC.md).

## 비용·해석 제한

P 기본18,962 params; DIM20,259(+1,297,6.84%); META20,307(+1,345,7.09%).
동일 RTX3080 full pipeline median: N0 12.638ms, N4 13.277ms. 메모리는 두 head와R0가 함께 상주한 process scope이며 arm별 독립 측정이라고 부르지 않는다.
Metadata perturbation 수치는 METADATA_SENSITIVITY.json: correct 대비 mean/shuffle/wrong-G/zero 변화는 민감도 진단이지 인과적 증명이 아니다. N4−N3가 unresolved이면 explicit code가 필수라고 주장하지 않는다.
합성 frame bootstrap 의존성 한계와 scenario-cluster 부 분석을 같이 보존했다. 실사319는 reused DEV13세션, 다중 비교 보정 confirmatory claim은 없다. Geometry-reconstructed pose GT는 독립 물리 계측이 아니다. Source C1의 physical front phase 모호성은 남는다.
Square와mixed는 별도 보고서다. Square 안에서는 치수가 상수이므로 치수 효용을 검증하지 않는다. Mixed에서는 synthetic/real domain과 C2/C4가 얽혀 있으므로 기능적 routing 진단으로만 해석한다.
원본 R0/P/D/L/PoseFix·데이터·cache·논문/배포를 수정하지 않았고 FINAL/sealed 열람, 재부팅/드라이버 변경, 새로운 sweep이나 seed 추가는 없다.
최종 회귀검사 26/26 PASS. 성능 성공과 무결성 PASS는 별개다.
