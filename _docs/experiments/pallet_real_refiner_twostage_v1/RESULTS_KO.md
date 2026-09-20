# N2 실사 2단계 필터 적응 — 단일 seed 예비 실험

상태: MIXED_OR_NO_POSITIVE_SCREEN. 기존 최종 모델/논문 표는 변경하지 않음.

R0 1000장 → 원본 confidence 필터 272장 → N2 보정 후 geometry/flip 필터 259장.

최종 학습 풀: {'daytime': 120, 'nighttime': 139}; 세션: {'capturepallet10': 21, 'capturepallet11': 99, 'capturenight04': 7, 'capturenight10': 29, 'capturenight01': 53, 'capturenight03': 38, 'capturenight02': 12}.

R0와 pseudo teacher는 고정. 기존 N2 seed1에서 시작한 복사본 3개, 각 1,500 step / batch16. 실사 두 arm은 같은 이미지/마스크/증강/순서이며 target 좌표만 다름.

실사는 기존 직사각 플라스틱 팔레트(1.1×1.3×0.11 m)이며, 초록 정사각 실사 학습은 아님. 초록 150장은 평가에만 사용.

메트릭: 매칭된 코너 median/P90(px), 전체 eligible 코너 PCK10, 누락 패널티 포함 평균 E_sym. 작을수록 좋음(PCK10은 클수록 좋음).

## DEV319

| 모델 | median px | P90 px | PCK10 % | E_sym |
|---|---:|---:|---:|---:|
| R0 | 6.7207 | 43.8900 | 63.425 | 0.0495239 |
| N2_ORIGINAL | 5.6984 | 42.7914 | 68.948 | 0.0483718 |
| SYN_ONLY | 5.6937 | 42.6964 | 68.707 | 0.0484007 |
| REAL_RAW | 5.9733 | 42.6161 | 67.507 | 0.0486956 |
| REAL_REFINED | 5.7873 | 42.7909 | 68.347 | 0.0484645 |

REAL_REFINED 비교:

- vs N2_ORIGINAL: ΔE_sym=+0.00009263, 개선/악화 프레임 113/198, good<5→bad>10 코너 0개.
- vs SYN_ONLY: ΔE_sym=+0.00006382, 개선/악화 프레임 128/183, good<5→bad>10 코너 0개.
- vs REAL_RAW: ΔE_sym=-0.00023111, 개선/악화 프레임 231/80, good<5→bad>10 코너 0개.

## GREEN150_MANUAL

| 모델 | median px | P90 px | PCK10 % | E_sym |
|---|---:|---:|---:|---:|
| R0 | 4.5798 | 9.2132 | 82.232 | 0.1127083 |
| N2_ORIGINAL | 4.1221 | 9.7495 | 81.351 | 0.1123264 |
| SYN_ONLY | 4.0511 | 9.9172 | 80.764 | 0.1123767 |
| REAL_RAW | 4.3647 | 10.1245 | 80.176 | 0.1126696 |
| REAL_REFINED | 4.0986 | 10.0816 | 80.323 | 0.1124460 |

REAL_REFINED 비교:

- vs N2_ORIGINAL: ΔE_sym=+0.00011962, 개선/악화 프레임 35/99, good<5→bad>10 코너 0개.
- vs SYN_ONLY: ΔE_sym=+0.00006927, 개선/악화 프레임 41/93, good<5→bad>10 코너 0개.
- vs REAL_RAW: ΔE_sym=-0.00022358, 개선/악화 프레임 114/20, good<5→bad>10 코너 0개.

## GREEN150_ALL_KNOWN_PROXY

| 모델 | median px | P90 px | PCK10 % | E_sym |
|---|---:|---:|---:|---:|
| R0 | 4.2368 | 10.1133 | 80.333 | 0.1134939 |
| N2_ORIGINAL | 3.7784 | 10.0098 | 80.333 | 0.1131069 |
| SYN_ONLY | 3.7858 | 10.2019 | 80.000 | 0.1131470 |
| REAL_RAW | 4.0388 | 10.3517 | 79.583 | 0.1134165 |
| REAL_REFINED | 3.8718 | 10.5387 | 79.750 | 0.1132181 |

REAL_REFINED 비교:

- vs N2_ORIGINAL: ΔE_sym=+0.00011120, 개선/악화 프레임 39/95, good<5→bad>10 코너 0개.
- vs SYN_ONLY: ΔE_sym=+0.00007103, 개선/악화 프레임 41/93, good<5→bad>10 코너 0개.
- vs REAL_RAW: ΔE_sym=-0.00019840, 개선/악화 프레임 110/24, good<5→bad>10 코너 0개.

## 해석 범위

- 단일 seed, 재사용 DEV에 대한 예비 결과. 독립 검증·통계적 유의성 주장 없음.
- GREEN150 manual-only가 초록 주 지표, all-known은 PnP 생성점 포함 proxy. 평가 라벨의 기존 QA/카메라 불일치 한계 유지.
- Teacher 예측은 정답이 아님. consistency와 학습 loss 개선만으로 좌표 정확도 개선을 주장하지 않음.
- 기존 candidate CE를 보정 좌표로 학습하므로 soft distribution을 좌표로 압축한 데 따른 self-sharpening 효과도 포함됨.
- 실사 arm은 photometric 증강+좌표 jitter, 합성 대조군은 기존 특징+동일 좌표 jitter. 합성 대비 차이는 이 적응 패키지 전체이며, 보정 좌표 고유 효과는 REAL_RAW 대비로 해석.
- R0/박스/score/centroid는 불변. negative 추가 추론은 하지 않았으며 박스 confidence 기준 오검출 개선 주장은 불가.
- 1차 필터 탈락 이미지에는 N2를 호출하지 않았음. 2차 필터를 완화하거나 결과를 보고 샘플을 교체하지 않았음.
