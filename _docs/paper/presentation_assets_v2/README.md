# N3 DIM+SYM — 교수님 보고/PPT 시각자료

현재 고정된 **N3_DIM_SYM**의 설명 자료다. 새 학습·추론·하이퍼파라미터 탐색은 하지 않았다. `optimizer_updates = 0`. 기존 논문 그림, checkpoint, prediction, 표는 수정하지 않는다.

기준 원격 `main`: `a8a7c73ce50ac522502be61a47e082e3933d3cb8` (fetch 및 ls-remote 확인). 제작 시작 로컬 HEAD는 `2fb7f0206f09a0722025c7c8ead7489c1b0e5cb6`; 새 그림 생성은 두 경로에만 한정한다. 전체 source binding과 파일별 SHA-256은 `ASSET_MANIFEST.json`에 있다.

## 산출물

| 파일 (PNG / 같은 이름 PDF) | 내용 | 읽을 때 주의 |
|---|---|---|
| [01_method_pipeline.png](01_method_pipeline.png) | frozen R0 → N3 국소 보정 → prediction-only PnP | 치수 입력과 training-only 대칭 대응은 별개 |
| [02_dimension_conditioning_examples.png](02_dimension_conditioning_examples.png) | 일반 플라스틱·목재·초록 정사각형의 실제 RGB와 W/D/H | 초록색은 입력/계약 설명용이며 N3 DEV319 결과가 아님 |
| [03_residual_refinement_examples.png](03_residual_refinement_examples.png) | 실제 R0 / N3 seed1 / GT와 원본 배율 이동 화살표 | 3-seed 평균으로 사례를 정하고 seed1만 고정 표시 |
| [04_symmetry_contract_C2_C4.png](04_symmetry_contract_C2_C4.png) | 실제 whole-object 회전 permutation | 치수로 대칭을 추정하지 않음; center8 고정 |
| [05_local_candidate_refinement.png](05_local_candidate_refinement.png) | 실제 222개 후보 lattice, expectation 및 cap | probability/mean 예시는 schematic이며 측정 logits가 아님 |

모든 PNG는 3200×1800, 16:9다. PDF는 도식·텍스트·마커를 벡터로 저장하고 실사 사진만 raster로 포함한다. 기존 설치된 DejaVu Sans를 사용한다. 큰 그림 제목은 넣지 않았으므로 PPT의 슬라이드 제목을 별도로 사용한다.

## 기존 그림 감사

기존 `_docs/paper/sensors_submission_v1/figures/pipeline.pdf`와 `graphical_abstract.png`는 **P-only** 흐름이다. 치수는 PnP로만 전달되고, 현재 N3의 dimension-conditioned scorer와 symmetry-aware supervision이 표현되지 않는다. 따라서 N3 그림으로 재사용하지 않았다.

`candidate_sampling.pdf`의 13방향×17반경+null 구조는 현 generic refiner와 일치한다. 새 Figure 05는 같은 lattice에 N3의 치수 score residual과 실제 decode/cap 구분을 더한다. `qualitative_coordinates.pdf`는 과거 P1 등 다른 arm의 사례이므로 현재 N3의 실제 보정 사례로 바꾸어 인용하지 않았다. 기존 그림은 전부 보존한다.

## 구현과 과장하지 않는 설명

- Frozen YOLO26n-Pose의 초기 8개 코너와 P3/P4 특징을 재사용한다. N3의 trainable refiner는 20,259 parameters이며, 이 작업에서는 학습하지 않는다.
- canonical `[W,D,H]`는 알려진 object type의 registry에서 읽는 외부 메타데이터다. 미지 object type 분류나 RGB에서 크기를 추정하는 기능이 아니다. GT pose로 입력 W/D를 바꾸지 않는다.
- N3에는 **명시적 C2/C4 one-hot code가 추론 입력으로 들어가지 않는다**. 5차원 dimension feature는 `logW, logD, logH, logW−logD, logH−0.5(logW+logD)`이며 고정된 training normalization을 사용한다.
- GT 대응은 학습 때만 한 개의 유효한 whole-object rotation을 고른다. 독립 코너별 최적 배치나 Hungarian matching이 아니다. 평가지표도 GT에 대해 허용된 whole-object permutation 하나로 오차를 계산한다.
- bbox, confidence, selected instance, nonselected instances, center8은 유지한다. 선택된 인스턴스의 corner0..7 xy만 변경한다. 이 그림은 detector AP가 향상되었다는 주장이 아니다.
- 후보 반경은 `0.08 × bbox diagonal`, 실제 출력 제한은 **`0.01 × original-image diagonal`** 이다. 서로 다른 기준이다. 221 displacement + null의 temperature-softmax expectation을 계산하고 λ=1 및 radial cap을 적용한다. Argmax나 큰 코너 복구용 coarse-to-fine 모델이 아니다.
- downstream PnP는 예측점·알려진 geometry·카메라 K만 사용한다. GT 대응으로 pose를 고르는 흐름이 아니다. 이 패키지에서는 6D pose를 새로 계산하지 않는다. 기존 실사 6D reference는 geometry-reconstructed reference이며 독립적인 물리 계측 GT가 아니다.
- DEV319는 일반 플라스틱 194 + 목재 125의 재사용 DEV population이다. 초록 정사각형의 별도 실험이나 C4 학습 범위를 N3 DEV319와 합쳐 주장하지 않는다. 사각형이라고 자동 C2, 정사각형이라고 자동 C4가 되는 것이 아니라 **object/task에서 미리 승인한 equivalence**다. 목재 C2는 특히 physical inspection이 아니라 기존 논문의 benchmark convention이다.

## Figure 02: 치수 및 RGB 선택

Registry: `challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json`.
대조 계약: `_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json`.
물리 XYZ를 `[x,z,y] = [W,D,H]`로 표시하며 두 source의 값이 exact 일치함을 검사한다.

| 타입 | canonical [W,D,H] m | 프레임 |
|---|---|---|
| rectangular plastic | [1.10, 1.30, 0.11] | `eval_cad:1778653029965134592` |
| rectangular wood | [0.80, 0.59, 0.14] | `wood_183705:001463` |
| square plastic | [1.10, 1.10, 0.15] | `capture_20260902_kimjihoon__006169` |

성능이 아니라 도해 가독성으로 선택했다. 8개 anchor가 이미지 안쪽 15px margin에 있고, 표시 높이 edge가 12px 이상이며, annotation의 camera-facing W/D가 canonical 길이와 이미 같은 후보 중 top-face 면적/이미지 면적이 가장 큰 것을 선택한다. 동률은 frame ID 사전순이다. 카메라 방향을 위해 입력 치수를 swap하지 않는다. W:0↔1, D:0↔4, H:0↔3의 실제 annotation anchor를 사용한다. edge 화살표는 방향/길이 항목 설명이지 별도 물리 계측 결과가 아니다. 일부 anchor는 기존 PnP-derived annotation이므로 독립적인 수작업 GT로 과장하지 않는다.

## Figure 03: 사례 선택과 수치

`scripts/research/pallet_dim_conditioned_p_v1/eval_math.py::measure`로 DEV319 × R0/N3 seed1/2/3의 **1,276 frame metrics**를 다시 계산하여 `_docs/experiments/pallet_final_paper_tables_v1/RESCORED_2D.json`과 일치함을 검증한다. 8 corners 중 supervision-valid GT만 사용하고 center8은 제외한다. missing/mismatch 처리는 원래 evaluator를 그대로 따른다. 이는 표의 pooled corner median과 다른 **frame mean pixel error**다.

`Δ = mean(seed1/2/3 frame errors) − R0 frame error`; 음수가 개선이다. 319장 중 평균 개선 268장, 악화 43장, 동일 8장이다. 세 seed 모두 개선은 238장이다. 8개의 동일 사례는 detector IoU 미매칭에 동일 penalty를 받는 사례일 수 있으므로 보정 성공으로 해석하지 않는다.

선택 규칙은 출력 전 고정했다: A는 모든 seed가 개선한 후보 중 전체 negative delta 분포의 22.5 percentile 값에 가장 가까운 프레임, B는 전체 negative delta의 median에 가장 가까운 프레임, C는 positive delta의 median에 가장 가까운 프레임이다. 동률은 frame ID 사전순이다. 모델/seed/학습량을 선택하지 않는다. overlay seed는 항상 **1**이다.

| 사례 | 프레임 | R0 px | N3 seed1 px | seed1 Δ px | 3-seed mean Δ px |
|---|---|---:|---:|---:|---:|
| A 개선 대표 | `eval_pallet07:1778652125245035520` | 27.2133 | 25.4195 | −1.7937 | −2.0272 |
| B 일반 개선 | `eval_pallet09:1778653706706429696` | 8.4776 | 7.4457 | −1.0319 | −0.8604 |
| C 악화 대표 | `wood_night_01:029919` | 2.9031 | 2.9877 | +0.0845 | +0.1511 |

왼쪽은 full RGB, 오른쪽은 valid GT corner 중 R0→N3 이동이 가장 큰 코너의 원본 좌표 crop이다. R0·N3 좌표, GT 좌표, 실제 이동 화살표를 확대/이동해 꾸미지 않는다. crop의 화면 확대만 있으며 화살표 배율은 **1×**다. GT 표시와 예측은 native index 좌표를 보존하고, 오차 계산만 whole-object symmetry correspondence를 적용한다. GT가 모델에 입력되었다는 의미가 아니다.

이 그림의 적절한 메시지는 “작은 잔차 보정은 가능하지만 큰 오차가 남을 수 있고, 일부 프레임은 악화된다”이다. **큰 코너 오차 복구가 해결되었다고 주장하면 안 된다.**

## Source / 저작권 / 무결성

Figure 02·03의 RGB는 repository 연구 이미지이며 **내부 교수님 보고용**으로 구분한다. raw RGB의 외부 재배포/논문 공개 권한은 **UNVERIFIED**다. 사용자 요청으로 기존 repository source를 사용했지만 이것이 출판 권한 확인을 대신하지는 않는다. 외부 웹 이미지, 새 font, 외부 디자인 asset은 사용하지 않았다. 공개 제출 전 권한을 확인하거나 RGB 없는 좌표 도식으로 대체해야 한다. Figure 01·04·05는 code-native 도식이다.

`ASSET_MANIFEST.json`은 그림별 source image, prediction, checkpoint, seed, registry, GT, metric, selection rule, script, limitations를 기록한다. 표시한 scatter 좌표는 matplotlib artist에서 원본 배열과 exact 비교한다. verifier는 다시 source에서 좌표를 읽어 manifest의 표시 좌표와 exact 비교한다. 모든 DEV319에 대해 selected instance·bbox·confidence·center 보존, 이동 cap과 frame metric parity를 검사한다. checkpoint·prediction·annotation·기존 그림·원본 code의 SHA-256을 생성 전/후 및 별도 검증에서 확인한다.

## 재현 / 검증

Repository root에서 실행:

```bash
MPLCONFIGDIR=/tmp/pallet-agreement-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/paper/presentation_assets_v2/build_assets.py
MPLCONFIGDIR=/tmp/pallet-agreement-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/paper/presentation_assets_v2/verify_assets.py
```

Builder는 기존 PNG/PDF/manifest를 덮어쓰지 않는다. 이미 생성된 package에는 verifier만 실행한다. 원본 예측/이미지는 저장소의 기존 local data 경로에 있어야 한다. GPU와 추가 패키지 설치는 필요 없다. `--probe`는 `/tmp/pallet_presentation_v2_probe.png`에 사전 확인용 contact sheet만 만든다.

Verifier는 이미지 decode 및 ≥1920×1080 해상도, PDF parser open/1-page, source SHA, 좌표 parity, frame ID, 치수, C2/C4 proper rotation/permutation, 동일 evaluator 및 no-training 선언을 검사한다. 모델 학습 코드 경로는 호출하지 않는다.

## PPT 배치 추천

1. **Table 1 앞:** Figure 01 — 기존 YOLO 점을 치수로 조건화하여 국소 수정하는 전체 흐름.
2. **Table 1 직전/직후:** Figure 02 — “치수 입력”의 실제 의미. Figure 05 — 넓은 후보를 보고 최종 작은 보정을 만드는 동작.
3. **Table 2 (6D pose) 앞:** Figure 03 — 실제 점 이동·개선·악화를 먼저 보고 downstream pose 표로 이동.
4. **Table 3 (symmetry ablation) 앞:** Figure 04 — C2/C4 whole-object 정답 동치와 고정 코너 순서를 설명.

상담 흐름: 문제 → 방법이 하는 일 → 실제 점 이동과 한계 → symmetry 계약 → 표 결과.
