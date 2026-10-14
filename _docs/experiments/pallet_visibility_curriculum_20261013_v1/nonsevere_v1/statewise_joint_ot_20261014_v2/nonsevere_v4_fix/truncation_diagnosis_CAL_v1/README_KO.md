# CAL-only T 근거 진단

CAL46 중 호환44의 초기 nonS 코너 287개를 검사했다. 3D 대응 불일치 CAL2는 별도로 기록했으며, S 좌표·영상 query와 TEST 자료는 사용하지 않았다. 모델·솔버·봉인 예측은 수정하지 않았다.

T 13개/7프레임은 모두 initial_pose_projection에 의한 조건부 판정이며 독립 증명은 0개다. raw query와 초기 투영이 함께 영상 밖이어도, 초기 pose는 같은 검출기 관측과 모델로 만든 값이므로 독립 증거 두 개가 아니다.

사람 상태는 {'UNANNOTATED': 9, 'DIRECT_VISIBLE': 2, 'EXTERNAL_OCCLUDED': 2}. 참조 재투영이 화면 안인 T는 3개, 유효 annotation이 화면 안인 T는 4개다. UNANNOTATED는 물리적 T 정답 상태로 간주하지 않는다.

모든13개에 대해 원 좌표·화면 거리·사람 상태·annotation/reference·조건부 상태 레코드·인접 B/I 영상 선분 근거를 [ALL_13_T_AUDIT.json](ALL_13_T_AUDIT.json)에 기록했다. 실제 B/I에서 T 쪽으로 향한 supported ray가 있는 T는 7개, 비평행 supported선 교점이 계산되는 T는 4개다. 선 방향·대응이 초기 모델을 사용하고 외삽 오차가 교정되지 않았으므로 T certificate로 승격하지 않는다.

## CAL에서 확인한 가능한 규칙

주된 raw localization 가설은 native-compatible44, initial nonS, 유효 human DIRECT/EXTERNAL annotation에서 raw query 오차의 정규화 최댓값을 구한다. raw query를 중심으로 r=이 CAL 최댓값×현재 nonS scale인 원 Ω를 놓는다. 닫힌 화면과 Ω가 완전히 분리되는 경우에만 조건부 T를 유지한다. Ω가 화면과 겹치면 U로 유보한다. 독립적으로 증명된 T는 우선하며, producer가 유지한 O 및 T는 공동 hard constraint로 남긴다. 여기서는 어떤 규칙도 채택하거나 pose solve하지 않았다.

초기 투영(또는 CAL median을 적용한 투영)을 중심으로 한 다음 범위들은 보조 진단이다. raw localization 범위의 대체물로 선택하지 않았다.

1. 전체 CAL frame-max의 최댓값을 반지름으로 고정: 관측한 모든 호환 CAL 오차를 포함하지만 미래 오차를 보장하지 않는다.
2. frame-max 95% rank 보정 분위수 ceil((n+1)×.95): 더 작은 범위의 가설이며 LOGO 누락을 그대로 공개한다. 같은 session의 프레임 상관 및 CAL 4개 session 때문에 95% 물리적/미래 coverage라고 부를 수 없다.
3. CAL prior covariance6를 mean 위치에서 2D Jacobian으로 전파: 국소 선형 민감도 보조 자료다. P3P/phase 다중성·모델/semantic 오차를 모두 포괄하는 범위라고 볼 수 없어 독립 채택하지 않는다.

- initial 중심: full-CAL max=0.54916882, rank95=0.09614411 (rank 43/44).
- mean 중심: full-CAL max=0.53685092, rank95=0.08970929 (rank 43/44).

최대 오차 프레임은 wood_184309:000372이며 normalized=0.54916882다. 같은 CAL에서 참조/model 오차가 큰 사례를 제거하여 범위를 줄이지 않았다. 초기 중심 frame-max 중앙값 0.01507383, 보간95% 0.08583268다.

| held-out session | train-max initial frame coverage | rank95 initial frame coverage |
|---|---:|---:|
| eval_cad | 18/18 | 18/18 |
| eval_night08 | 3/3 | 3/3 |
| eval_outside | 5/5 | 5/5 |
| wood_184309 | 15/18 | 15/18 |

## Raw query localization 범위

유효 DIRECT/EXTERNAL 264점/44프레임의 full pooled max=0.54475075. 최대점은 wood_184309:000372 native0 (445.236px)이다. 풀 CAL max 원을 쓰면 기존 T13 중 0개 유지, 나머지는 U 가설이다. held-out session별 train-only max로는 0개 유지한다. O/domain/scale 숫자는 바꾸지 않았다.

| human class | 점 수 | normalized max |
|---|---:|---:|
| DIRECT_VISIBLE | 255 | 0.54475075 |
| EXTERNAL_OCCLUDED | 9 | 0.08645593 |

| held-out session | train max | point coverage | frame coverage |
|---|---:|---:|---:|
| eval_cad | 0.54475075 | 100/100 | 18/18 |
| eval_night08 | 0.54475075 | 18/18 | 3/3 |
| eval_outside | 0.54475075 | 32/32 | 5/5 |
| wood_184309 | 0.08645593 | 105/114 | 15/18 |

CAL full pooled max는 관측 CAL 안에서 유효 점들을 포함한다는 경험적 규칙이다. 미래 native point coverage 또는 독립적 physical T를 보장하지 않는다. EXTERNAL annotation 또한 가림 뒤 좌표의 인간 추정 오차를 가진다.

## Native phase와 C2 구별

calibrator는 xyz/Q/native-to-reference의 고정 3D 대응으로 phase를 정하고 2D 오차나 추정 pose로 phase를 고르지 않는다. 물리 C2 pose 점수의 가장 가까운 phase는 native KP ID 대응과 별개다. 두 phase가 다른 CAL 사례를 [NATIVE_PHASE_AUDIT.json](NATIVE_PHASE_AUDIT.json)에 기록했다. 가까운 C2 phase로 바꾸어 큰 native 오차를 줄이지 않았다.

- wood_184309:000372: fixed native phase 1 / closest coarse C2 phase 0.

LOGO는 envelope를 held-out session 없이 fit했다. mean/Jacobian은 이미 모든 CAL로 fit된 μ/Σ를 사용하므로 별도 조건부 기술 통계이며 full pipeline OOF가 아니다. 초기 중심의 envelope LOGO도 upstream state/geometry가 all-CAL fit인 조건부 입력이라는 제한을 유지한다.

## 실제 crop 메타데이터

CAL3 실제 crop의 RGB·K 이동은 알려져 있으나 metadata의 physical_T_confirmed=false, source_inframe_native_ids=[]이고 source_RGB_validated=false다. source corner 위치가 독립적으로 확인되지 않아 특정 T 코너의 증명이 되지 않는다. 이 진단은 crop 좌표의 사후 정답을 로드하여 certificate를 만들어 내지 않았다.

## 경계 사례

- eval_cad:1778653041187027200 T2: raw/coarse는 오른쪽 밖, DIRECT annotation 및 reference는 안.
- eval_night08:1779449501478488320 T0/T3: raw/coarse는 왼쪽 밖, EXTERNAL annotation 및 reference는 안.
- eval_cad:1778653055734035712 T0: DIRECT annotation x=1은 안, derived reference x≈−0.417은 밖. 1픽셀 근처의 reference/annotation 차이는 물리적 정확도의 한계를 드러낸다.

[전체 수치·LOGO·crop evidence](DIAGNOSIS.json) · [CAL 오차 CDF](CAL_ERROR_CDF.png) · [변경·입력 검증](CHECKS.json)
