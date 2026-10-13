# 상태별 영상 보정과 모든 O/T의 공동 자세 제약

현재 구현은 **B/I의 영상 관측을 먼저 보정·고정하고, 모든 O/T 영역을 동시에 적용해 자세를 구하며, S는 그 자세의 모델 투영으로만 완성하는 방법**이다. 이전 `anchored_completion_v1/region_completion_v1`은 S를 영상 영역에 포함하고 기존 유한 자세 후보를 검사했던 방법이다. 그 결과를 이 방법의 완성·검증으로 보고했던 주장은 철회한다. 이전 기록은 비교와 감사용으로 보존했다.

현재 기본 실행 설정은 [RUNTIME_CONFIG_V3.json](RUNTIME_CONFIG_V3.json)이다. 연구 폴더와 호환 API의 `v2` 이름은 유지하지만, 현재 솔버는 `immutable_BI_joint_OT_v3`이다.

## 정의와 처리 순서

| 기호 | 이름 | 처리 |
|---|---|---|
| S | Self-occluded keypoint | 초기 등록 모델의 자기 가림 판정 후 영상 관측에서 제거한다. 최종 PnP의 모델 투영만 사용한다. |
| B | Visible boundary keypoint | 실제 파렛트/배경 경계선의 교점으로 보정한다. |
| I | Visible interior keypoint | 실제 파렛트 내부 RGB 선의 교점으로 보정한다. |
| O | Externally occluded keypoint | 외부 가림 판정 후 거친 위치 영역을 자세의 개별 제약으로 사용한다. |
| T | Truncated keypoint | 화면 밖 공간 제약을 사용하며, 독립 근거와 초기 자세에 따른 조건부 판정을 구분한다. |

U는 내부 판정 보류이며 여섯 번째 물리 상태가 아니다. B/I는 초기 자세에 따른 역할이고, 실제 보임 여부는 별도 DIRECT/EXTERNAL RGB 판정기로 확인한다.

1. 동결 backbone의 native KP, 원 카메라 K, 등록된 3D 모서리·치수, 초기 PnP를 받는다. 초기 모델로 S와 B/I 역할을 정한다.
2. S 좌표를 마스크 프롬프트, 상태 특징, 영상 보정, 영역 생성, PnP 점 대응에서 제거한다. 비-S의 실제 RGB로 보임/외부 가림을 판정한다.
3. 보이는 B/I를 서로 다른 영상 근거로 보정하고, 관측 좌표와 공분산을 고정한다. 이전 seed의 강체 투영을 고정하지 않는다.
4. 회전과 이동의 6자유도를 연속 최적화하며 **모든 O/T의 각 영역 제약을 동시에** 만족시킨다. 영역 중심을 정밀 대응점으로 사용하지 않는다. 최종 B/I는 가정한 공분산 오차 한도, 등록 모델의 면 가시성 기준, 화면 내부 조건도 만족해야 한다.
5. B/I가 두 점뿐이면 O/T가 있는 경우 공동 가능성을 탐색하되 항상 모호한 자세 집합으로 보고한다. 발견된 대표 자세를 유일한 정밀 해로 표현하지 않는다.
6. S는 선택된 PnP 자세에서만 투영한다. 공동 해를 찾지 못하면 O/T는 미해결로 두며 S의 fallback 자세 출처를 명시한다.

수식과 허용 오차, 초기 자세의 조건부 역할, 실패 처리의 상세는 [METHOD_DETAILS_KO.md](METHOD_DETAILS_KO.md)에 기록했다. 모델 가시성의 2° grazing 유예와 3σ 관측 한도는 물리 가시성이나 보정된 확률의 보장이 아니다. 유한 다중 시작점 탐색도 전역 유일성·불가능성을 증명하지 않는다.

## 실제 검증 결과

학습과 주 평가는 CLEAN/MODERATE만 사용하며 severe는 0장이다. CAL46/TEST49 및 별도 실제 RGB crop 3건의 예측을 먼저 봉인하고 평가했다. TEST는 이미 재사용된 탐색 평가이고, 자세 기준은 기존 영상에서 파생된 reference이므로 새 독립 평가로 표현하지 않는다.

| TEST49, 실패 시 fallback 포함 | 평균 위치 오차 | 평균 회전 오차 | 5cm·5° 동시 성공 |
|---|---:|---:|---:|
| 원래 PnP | 3.363 cm | 6.527° | 40/49 |
| 같은 보정 B/I만 사용하는 PnP | 4.692 cm | 8.285° | 38/49 |
| 모든 O/T 공동 제약, 현재 v3 | 3.514 cm | 6.568° | 39/49 |

**현재 방법이 원래 PnP보다 좋다는 결과는 나오지 않았다.** 공동 O/T는 같은 B/I만 사용한 비교군보다 오차를 줄였지만 원래 PnP의 전체 성공률을 넘지 못했다. 현재 구현 완료와 성능 우수성 주장은 구분한다.

- TEST의 실제 외부 가림 12점 중 11점을 O로 찾았고 1점을 B로 놓쳤다. 실제 보이는 286점 중 11점은 O로 오분류했다. 상태 판정은 완벽하지 않다.
- 물리 native 대응을 비교할 수 있고 실제 DIRECT이며 B/I로 판정된 TEST224점의 평균 픽셀 오차는 원 KP 6.159px에서 보정 관측 5.449px로 줄었다. 이 부분집합의 픽셀 개선이 전체 자세 성능 개선을 뜻하지 않는다.
- 현재 솔버는 CAL18/46, TEST29/49에서 공동 가능 자세를 찾았다. 나머지는 미해결 또는 출처가 명시된 fallback이다. 같은 B/I만 사용한 자세와 수치적으로 달라진 accepted 사례는 CAL2건, TEST7건이다(회전 행렬 norm >1e-6 또는 이동 >1e-7m). TEST 한 건은 이동 차이가 3.74e-7m인 작은 수치 차이다. 이 개수를 자세 개선 건수로 해석하지 않는다.
- 자연 T 주석은 0개다. 별도 crop은 실제 RGB를 잘랐지만 upstream KP와 K를 좌표 변환해 사용했다. cropped backbone을 새로 추론한 실험이 아니다. 첫 crop은 실제 T4/T5 중 T4를 놓쳤으며, 세 번째 crop은 T0/T3을 잘못 조건부 판정하고 실제 T5를 놓쳤다. 수치적으로 가능한 자세를 물리 T 성공으로 보고하지 않는다.
- B/I 두 점의 실제 night/crop 결과는 모호한 자세 집합으로 표시한다. `OT_solved=False`, `OT_unique_solution_claimed=False`이며 가능한 모델 투영 후보를 제공한다.

전체 지표와 봉인된 프레임별 결과는 [evaluation_v3/POSTSEAL_SCORING.json](evaluation_v3/POSTSEAL_SCORING.json), 간결한 결과와 근거 해시는 [RESULTS_V3.json](RESULTS_V3.json), 현재 실제 사진 시각화는 [current_figures_v3/index.html](current_figures_v3/index.html)에 있다. [figures/index.html](figures/index.html)의 이전 솔버·첫 판정기 그림은 역사적 비교 자료다.

## 검증과 재현

현재 전체 98건 평가는 [evaluation_v2](evaluation_v2/PROTOCOL.json)의 봉인된 RGB 상태·B/I 관측·O/T 영역을 그대로 사용해 솔버만 수정한 [evaluation_v3](evaluation_v3/PROTOCOL.json) 재실행이다. 판정기·영역을 TEST 성능에 맞춰 다시 학습하지 않았다. [runtime_smoke_current_v3](runtime_smoke_current_v3/CONTRACT_CHECKS.json)에서는 실제 RGB 3사례에 Tiny/SAM과 현재 전체 API를 새로 실행했다. 상태·보정·영역은 캐시와 일치했고, S의 원 좌표를 극단적으로 바꿔도 마스크·보정·자세·readout은 같았다.

[TEST_RECEIPT_V3.json](TEST_RECEIPT_V3.json)의 관련 95개 검사를 통과했다(기본 suite 90개와 물리 상태 회귀 suite 5개). S 제외와 좌표 교란 불변성, 서로 다른 B/I 픽셀 근거, 모든 O/T 개별 제약, 2점 모호성, 잘못된 자기 가림 자세 거부, native 순서·등록·9번째 중심 보존, 실패 시 숨은 점 미해결 처리를 검사했다.

[독립 완료 감사](INDEPENDENT_COMPLETION_AUDIT_V3_KO.md)는 33묶음·4,137개 검산에서 구현·출력 불일치 0건을 확인했다. S 제외는 **초기 기하가 S로 정한 ID**에 대한 보장이다. 실제 사람 SELF 주석 중 CAL13/TEST13점은 B/I로 오분류돼 영상 관측에 사용됐다. 초기 상태 오류까지 해결했다는 의미는 아니다.

실제 이미지 한 장 실행은 저장소 루트에서 아래 명령을 사용한다. 출력 파일은 새 경로를 지정한다.

```bash
python -m scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_joint_cli_v2 \
  --image data/evaluation/pallet_eval_v1/dev_existing/sessions/eval_outside/rgb/1778651519235162880.png \
  --input _docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/EXAMPLE_INPUT.json \
  --config _docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/RUNTIME_CONFIG_V3.json \
  --output /tmp/pallet-statewise-current-v3-example.json \
  --device cuda
```

입력 JSON에는 `raw_points`, `K`, `initial_pose`와 선택적인 `native_xyz`, `independent_truncation`, `crop_transform`을 넣는다. 기본 estimator 교체는 하지 않았다. CLI는 이 상태별 후처리를 실제 backbone 출력에 적용하는 별도 실행 진입점이다. 원 RGB 데이터, MobileSAM/ImageNet 대형 체크포인트와 기존 로컬 assets가 필요하며 공개 묶음의 포함·제외와 재현 조건은 [PACKAGING_REPRODUCIBILITY_REVIEW_V3_KO.md](PACKAGING_REPRODUCIBILITY_REVIEW_V3_KO.md)에 기록한다.
