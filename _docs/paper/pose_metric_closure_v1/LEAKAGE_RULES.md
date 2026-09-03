# Leakage rules

이 트랙이 지키지 못하면 결과 전체가 무효가 되는 규칙.

## 1. GT 는 hypothesis 선택에 들어가지 않는다

```text
selector 입력으로 허용   predicted 9 keypoints
                         camera intrinsics
                         fixed physical dimensions (registry, 배포 시에도 알려진 값)
                         frozen selector config

selector 입력으로 금지   GT dimensions_m
                         GT pose / GT R / GT yaw
                         GT axis assignment
                         GT keypoint error
                         test-set ADD / IoU3D
                         session prior (세션마다 정답이 한쪽이라는 사전지식)
```

GT 는 **모든 선택이 끝난 뒤** parity 비교에만 읽는다.

근거: 기존 `PLASTIC_SELECTOR_DIAGNOSTIC.json` 의 `gt_leakage_contract` 가 이미 이
계약을 담고 있고, 재현된 0.5929 / 0.6500 은 그 계약 아래의 값이다.  즉 지금 수치는
누수를 걷어내면 떨어질 값이 아니라 **이미 정직한 값**이다.

## 2. deployed selector 를 real target pose GT 로 학습하지 않는다

논문이 유지하려는 문장은

> without manually annotated target-domain pose labels during training

이다.  real target 의 yaw/axis GT 로 배포 selector 를 학습하면 이 문장을 못 쓴다.

```text
selector 학습 라벨    synthetic 만
real pose GT 용도     selector DEV 평가 · final pose evaluation 만
```

`SELECTOR_DEV` 의 real GT 를 method tuning 에 쓰면 엄밀한 의미의 target-label-free
개발이 아니다.  따라서 **hyperparameter 도 synthetic dev 에서 동결**하고, real
SELECTOR_DEV 는 sanity check 로만 쓴다.  이 규칙을 어기면 그 사실을 논문에 적는다.

## 3. PAPER_EVAL 을 보며 selector 를 튜닝하지 않는다

PAPER_EVAL 319 는 `role = DEV`, `held_out_final = false` 이고 V1~V5 와 모든 진단이
이미 소진했다.  selector 진단 population `DEV_POS140` 은 그 안의 7개 정본 세션에서
나온 부분집합이다.

```text
금지   DEV_POS140 / PAPER_EVAL 정확도를 보며 0.95 가 될 때까지 selector 를 고치는 것
허용   원인 분해(왜 실패하는가)를 위해 읽는 것 — 이미 이 트랙이 한 일
```

selector 를 실제로 개발한다면 **PAPER_EVAL 과 image SHA256 overlap 0** 인
별도 population 에서 한다.

## 4. unresolved 프레임을 조용히 버리지 않는다

selector 가 확신하지 못하면 `POSE_UNRESOLVED` 를 반환할 수 있다.  그러나 최종 표에서
그 프레임을 제거하면 성능이 부풀려진다.

```text
반드시 함께 보고   pose coverage
                   conditional pose error (해결된 프레임만)
                   failure-aware score (전 프레임 기준)
```

coverage gate 0.95 가 사전 고정된 이유가 이것이다.

## 5-0. 현재 selector 는 실제로 GT 를 안 쓴다 — 코드로 확인함

선언이 아니라 구현을 읽었다.

```text
challenge/evaluation_v2/pnp_selector.py:495

def select_pnp_hypotheses(
    predicted_keypoints,      <- 예측
    camera_intrinsics,        <- 카메라
    physical_dimensions,      <- registry, 배포 시에도 아는 값
    config,                   <- 동결된 설정
) -> PnPSelectionResult:
    """Solve and score both physical W/D parities without evaluation labels."""
```

인자가 넷뿐이고 GT 유래 인자가 없다.  결과 객체의 `axis_assignment` 필드는
**출력**이지 입력이 아니다.  파일 전체에서 `gt_` / `ground_truth` / `target_pose` /
`expected_hypothesis` 참조가 0건이다.

따라서 재현된 **0.5929 / 0.6500 은 누수를 걷어내면 떨어질 값이 아니라 이미 정직한
값**이다.  이 트랙이 개선해야 할 출발점이 저 수치다.

## 5. 확인 절차

이 트랙의 어떤 산출물이든 아래를 만족해야 병합한다.

```text
[ ] selector 입력 목록에 GT 유래 필드가 없다 (코드로 확인, 선언 말고)
[ ] selector 학습 라벨에 real target pose 가 없다
[ ] gate 수치는 PAPER_EVAL 이 아닌 population 에서 나왔다
[ ] coverage 와 accuracy 를 함께 보고했다
```
