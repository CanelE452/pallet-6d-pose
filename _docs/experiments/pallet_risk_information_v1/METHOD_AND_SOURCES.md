# 방법과 근거 범위

이 캠페인은 이미 실행해 보관한 D1 실제 행동 중 하나를 고르는 실험이다. 새 pose solve·PnP·수치 자세 최적화·후보 생성·본체 forward를 추가하지 않는다. [고정 모델·감독 계약](MODEL_AND_SUPERVISION.md), [F1/B 상세 계약](FEATURE_SUPPORT_AND_B_DIAGNOSIS.md), [B 정답 대입 결과](B_REFERENCE_RESULTS.md), [평가 보고서](REPORT_KO.md)가 정의와 결과를 분리한다. 채택 판단은 사전 고정 대표 seed와 모든 세 seed를 함께 보며 실사 최고 seed를 골라 바꾸지 않는다.

## 구현이 기준인 부분

원 F0 188차원, 행동 순서·alias·수치 중복·eligibility·normalizer는 이전 [action_data.py](../../../scripts/research/pallet_boundary_action_repair_v1/action_data.py)를 그대로 쓴다. 새 [registry](../../../scripts/research/pallet_risk_information_v1/registry.py)는 실제 V gate에 따른 군과192/49/143/CAL46 분모를 정의한다. [예측기](../../../scripts/research/pallet_risk_information_v1/predict.py)가 정답 입력 없이 전량 예측을 봉인한 뒤, [평가기](../../../scripts/research/pallet_risk_information_v1/evaluate.py)가 원 물리 C2/수치식을 적용한다. [독립 평가기](../../../scripts/research/pallet_risk_information_v1/independent_evaluation.py)는 숫자·짝지은 변화·cluster CI를 별도로 검산한다. 정답 선택 oracle은 후보 정보의 진단이며 새 선택기의 검증 성능이 아니다.

R_CONT는 두 축 연속 오차를 회귀하고 예측 log 오차의 최대값이 작은 행동을 고른다. R_ORDER는 성공 우선과 실패 간 두 축 Pareto 지배 관계에만 선호 쌍을 만든다. 두 성공, 실패 tradeoff, 수치 모호 관계를 임의 순서로 만들지 않는다. [supervision.py](../../../scripts/research/pallet_risk_information_v1/supervision.py)와 [risk_models.py](../../../scripts/research/pallet_risk_information_v1/risk_models.py)가 pair 구성·frame/category 평균·loss를 정의한다. 이는 실패 구간에서 연속 위험이 좋아지는지 실험하는 설계이며 안전 보장은 아니다.

F1은 같은 실제 물리 자세를 두 family의 native Q/origin으로 표현해 원 full/partial/outline 비용·제약·finite profile·출처·유효성·관측 부재를 추가한다. 값144와 validity144를 train768에서만 정상화한다. F1_ZERO는 같은 차원·초기화에서 추가 block만0으로 만드는 대조다. 원 numeric B 비용을 무지지라는 이유로 지우지 않고,0 profile을 신뢰 지지로 해석하지 않는다. [feature_support.py](../../../scripts/research/pallet_risk_information_v1/feature_support.py)가 고정 scalar 평가와 출처를 담당한다.

V_ORDER는 같은 행동의8코너·12edge중점에 작은 RGB/mask query를 배치한다. NULL/RGB는 같은 architecture·초기화·frame/pair schedule을 쓰고 NULL만 RGB를0으로 만든다. 이 구현은 CAD render나 RGB-D 검증기를 재현한 것이 아니다. 합성 validation 외의 실사 정답으로 checkpoint·feature·threshold를 바꾸지 않는다. 실제 patch mapping은 [visual_patches.py](../../../scripts/research/pallet_risk_information_v1/visual_patches.py)와 모델 문서를 따른다.

## B 진단의 의미와 제한

[reference_support.py](../../../scripts/research/pallet_risk_information_v1/reference_support.py)는 별도 프로세스에서만 합성 reference를 대입한다. 원 phase0과 행별 C1/C2의 어느 허용 phase가 가능한지를 분리하고, 같은 family·G13 mode·관측 package 안에서만 E0와 비용을 비교한다. 다른 목적함수의 절대 숫자를 빼거나 정답 대입을 production candidate로 추가하지 않는다.

원 float32 GT R의 precision 때문에 원1e−10 roundtrip assertion은 실패했다. 그 원 FAIL을 남긴 채 raw GT의 projection/finite와 원 backend의 rotvec 표현 비용·제약을 별도 기록한다. 원 R 수정·정규직교화·tolerance 확대는 하지 않는다. geometry 검산의 통과는 원 모든 비용·관측의 물리적 정확성을 인증하지 않는다.

원 annotation에는 `v2_labels.material_variant_target/actual`, `front_visibility_cos`, `scene_placement_v2`의 face/opening visibility와 mask 비율이 실제 존재한다. 합성 scene/variant metadata를 별도 진단에 보존하며, 이를 per-edge/per-pixel material owner 정답으로 바꾸지 않는다. F1·학습 입력에는 이 GT metadata를 전달하지 않는다. cuboid 투영변이 실제 재료 외곽인지와 독립 실측 pose 정답은 여전히 UNVERIFIED다. 공개 B 그림은 합성 RGB 권한을 가정하지 않는 geometry-only 그림이다.

## 외부 원문과 실제 재사용 범위

원문 읽기와 scope는 [EXTERNAL_SOURCE_READING.json](EXTERNAL_SOURCE_READING.json)에 연결한다. 공식 문헌은 설계 근거이고 이번 데이터에서의 이득은 봉인 평가로만 판단한다.

| 근거 | 읽은 부분과 이번 구현의 범위 |
|---|---|
| [RankNet, Burges 등(2005)](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf) | §3의 within-query score difference logistic preference를 `softplus(score_j−score_i)`로 사용한다. 성공 우선·Pareto·frame/category 균형과 안전성은 이 문헌의 보장이 아니다. |
| [Regression with Deferral, Mao 등(2024)](https://proceedings.mlr.press/v235/mao24d.html) | 오차 회귀와 선택 문제를 분리하는 동기다. 논문의 bounded loss·consistency 조건을 여기서 검증하거나 그대로 적용했다는 주장은 하지 않는다. |
| [PyTorch SmoothL1Loss 공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.SmoothL1Loss.html) | R_CONT의 beta1 기본 연산이다. 실제 정의·버전은 설치 Torch2.1.1 docstring/source 및 환경 receipt와 대조했다. 축/행동/frame 평균은 이번 source 계약이다. |
| [MegaPose, Labbé 등(2023)](https://proceedings.mlr.press/v205/labbe23a.html) | 후보 pose를 이미지 근거로 검증한다는 관련 접근이다. CAD render·refiner·학습 체계를 가져오지 않았고20개 local query만 새로 실험한다. |
| [FoundationPose, Wen 등(2024)](https://arxiv.org/html/2312.08344v2) | 후보별 관측 검증의 관련 접근이다. RGB-D/mesh/transformer와 전체 시스템 성능·보장을 본 구현에 귀속하지 않는다. |
| [Switchable Constraints, Sünderhauf·Protzel](https://nikosuenderhauf.github.io/projects/switchableConstraints/) | 관측과 추정의 불일치를 구분해야 한다는 관련 근거다. switch 최적화를 새로 실행하지 않는다. |
| [Graduated Non-Convexity, Yang 등](https://web.mit.edu/sparklab/2020/11/10/Graduated_Non-Convexity_for_Robust_Spatial_Perception__From_Non-Minimal_Solvers_to_Global_Outlier_Rejection.html) | robust spatial estimation의 관련 문헌이다. GNC solver나 새로운 outlier threshold를 넣지 않는다. |

## 검증과 공개 범위

[VERIFICATION.json](VERIFICATION.json)의 입력·실행·숫자·감독·기하·모델 픽셀 검토·물리 진실·원격 게시 상태를 각각 읽는다. PNG decode와 실제 모델 픽셀 검토, owner 권한과 독립 물리 정답은 다른 근거다. [RIGHTS_AND_PRIVACY.json](RIGHTS_AND_PRIVACY.json)은 PRIMARY192 파생 RGB 표시 권한과 원 display-only redaction만 연결한다. 원본 RGB·합성 RGB·전체 bank·logits·가중치는 공개하지 않는다. 실행·실패·미계측 비용과 과거 원본 보존 검사의 최초 FAIL/재검사 PASS는 함께 남긴다.

실행과 공개 패키지 검사는 [REPRODUCE.md](REPRODUCE.md)를 따른다. 운영 기본값·논문·과거 실험은 자동 교체하지 않고, 독립 신규 실사 test는0장이다.
