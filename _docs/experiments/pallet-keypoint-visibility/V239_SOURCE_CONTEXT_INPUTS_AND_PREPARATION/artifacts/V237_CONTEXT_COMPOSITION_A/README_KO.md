# V237: 같은 native 장면에서 원 local96와 q8 context224 읽기

CODE-only 순수 NumPy 준비다. 실제 SOURCE/RGB/NPZ/GT/모델/feature/평가 payload는 읽지 않았다. 새 학습·forward·데이터 생성·라벨 채택은 없다.

## 판정과 Root의 다음 범위

기존 V235의 5,367개 **post-RAW96 합성쌍을 그대로 clean full-context와 결합하면 안 된다.** 원본 배경의 target 픽셀이 문맥에 남고 두 입력이 다른 장면을 가리킨다. old96 합성 결과를 원 해상도에 역붙이는 것만으로도 기존 local 결과와 동일하다고 할 수 없다. 아래의 보간 비가환 반례가 있다.

Root는 우선 새 V235 CF 없이 원 natural SOURCE와 기존 SYN actual BEFORE/COVER/SHAM/pose RGB를 읽는 input-only 대조를 검토 중이라고 밝혔다. 이 문서는 그 대조·모델·recipe를 선택하거나 실행하지 않는다. 기존 actual scene마다 원 anchor q8/q/L을 유지하고 원 local96 parity를 별도 실제 범위에서 검증해야 한다.

후속 새 CF에는 먼저 **한 공통 native composite**를 만든 뒤 두 원 좌표 그리드로 읽는 방법이 명확하다. 이 경우 old donor/seed/placement는 제안값으로 재사용할 수 있으나 old5367 성공·byte·teacher 증명은 새 뷰의 증명이 아니다. 해당 native 합성/재검증 단계는 Root의 후속 범위가 필요하다. 이 구현은 2D opaque-layer rasterization이며 물리 렌더·자연 외부가림·native gold가 아니다.

## 수학

- 원 local grid: `p_x=q_x+(i-47.5)*L/96`, y도 동일하다. 역주소는 `(p-q)*96/L+47.5`다. q/L을 바꾸지 않는다.
- 원 V107 q8 box: finite이고 `[-1,-1]`이 아닌 원 q가 2개 이상이며 두 축 span이 양수일 때 coordinate-wise min/max다. enlargement/clipping 또는 availability 필터를 추가하지 않는다. finite하지만 unavailable인 q가 있을 수 있다는 historical caveat를 유지한다.
- V107 context224: `p_x=x0+(j+.5)*(x1-x0)/224`. F64 NW→NE→SW→SE bilinear sum, native OOB 이웃 기여0, 재정규화 없음, F32 출력이다. valid는 양의 weight 이웃이 모두 존재함이다.
- 두 RGB가 같은 장면을 보려면 `local = S_local(F)`와 `context = S_context(F)`의 **동일한 F**를 사용해야 한다. 일반적으로 `S_local(compose_native(B,A,D)) != compose_local(S_local(B),a,d)`다. 반올림, mask 경계와 두 격자가 이 비가환성을 만든다.
- 공통 native 합성은 `F[n]=D[n]` if alpha[n], else `B[n]`으로 저장한다. F만 샘플링하고 clean B를 context branch에 따로 주지 않는다. sampled native 이웃4개가 모두 alpha이면 그 좌표 RGB에 clean B의 기여는0이다. 경계에서 layer coverage가1 미만이면 부분 clean 혼합이므로 opaque라고 주장하지 않는다.

## 최소 API

`context_scene_math.py`의 `compose_native_then_sample(base_rgb, query8, q, L, layer_rgb96, alpha96, reference=None)`를 사용한다.

- `base_rgb`: 승인된 actual-image 원 HWC uint8. `query8`: 원 F64[8,2]. `q`: 원 F64[2]. `L`: 원 F64 scalar.
- `layer_rgb96`: 제안된 donor-pixel texture HWC uint8[96,96,3], `alpha96`: bool[96,96]. old pair RGB를 넘기더라도 alpha 안의 donor 값만 사용한다.
- 이 API의 **새로 정의한** lift는 native p를 RAW96 주소 u로 옮겨 half-open cell `[i-.5,i+.5)`의 nearest 색과 alpha를 택한다. 이를 원 데이터 생성법이라고 부르지 않는다. native grid 밖은 합성 대상이 아니다.
- 반환: `native_composite`, `native_alpha`, `local_rgb_F32`, `context_rgb_F32`, 두 geometry-valid, 두 layer-coverage 진단, 원 q/q8/L, q8 box. `reference`는 그리드·layer 위치를 정하지 않고 마지막 opacity 진단에만 쓰며 배포 입력에 추가되지 않는다.
- 모든 결과 `new_label=None`이다. unsupported q8는 명시적 unsupported이며 E가 아니다. native/available/roles/condition은 caller가 원 sidecar 그대로 유지한다.

**중요한 인터페이스 한계:** local 출력은 명시 F32 sampler 결과다. 기존 RGB96의 uint8 양자화·invalid-zero·제작별 grid parity를 이 코드가 인증하지 않는다. 원 crop sampler와 exact payload 비교가 다음 actual scope의 별도 요구사항이다. 원 valid bitmap도 읽지 않았으므로 여기 valid는 V107 geometry 진단뿐이다. alpha coverage 역시 teacher 진단이며 학습 입력 feature가 아니다.

## 인공 배열 검사

`toy_scene_composition.py` 7그룹, `727fbe exit0` PASS다.

1. pinned V107 query-box/grid/RGB/validity와 bitexact.
2. 임의 q=[64.2,63.7], L=80.6에서 동일 round-half-up을 적용해도 post-crop 합성과 native-first 합성 local96의 120픽셀이 다름. old alpha 안59픽셀은 native-first layer coverage가 부분적임.
3. 두 뷰에서 native 좌표가 정확히 겹치는 1,024지점 RGB/validity bitexact.
4. reference native4tap을 모두 덮으면 clean base0↔255 교체에도 reference RGB가 같음.
5. nearest pixel 하나의 alpha hit만으로 four-tap opacity를 증명할 수 없음.
6. 원 q/q8/L/배열 불변; unsupported q8→라벨 없음.
7. fractional OOB에서도 literal V107 zero-neighbor RGB/valid parity.

120/59는 **인공 반례 하나의 수량**이다. 실제 5,367쌍 오류율·교사 오류·실패 원인의 추정이 아니다. 실행은 인공 NumPy 배열과 허용된 V107 순수 함수뿐이며 SOURCE pixels·모델·optimizer는0이다.

재현: `python -B toy_scene_composition.py` (TOY_RESULT는 새 경로에서 생성하거나 기존 결과를 보존해야 한다.)

원본 prior 파일은 수정하지 않았고 추가 inert pointer를 따르지 않았다. 실제 실행·새 target/모델 선택은 이 READY가 승인하지 않는다.
