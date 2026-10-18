# 방법의 근거와 적용 범위

이번 구현은 기존 팔레트 상태별 경로를 이용한 별도의 개발 실험이다. 아래 논문 전체를 재현한 방법이나 독립 실사 성능의 이론적 보장으로 부르지 않는다. 원문·공식 문서의 열람과 적용 범위는 다음과 같다.

| 원자료 | 실제 확인한 내용 | 이번 적용과 다른 전제 |
|---|---|---|
| [SRT3D, IJCV 2022](https://link.springer.com/article/10.1007/s11263-022-01579-8) | §3.4 Eq.(30)의 회전·이동 Tikhonov 항, §5.4의 제한. B 담당 모델이 원문과 설치 SciPy 소스를 열람했다. | 단일영상 cuboid의 N0 주변 누적 pose prior다. tracker의 프레임 간 갱신, 정확한 물체 메시, foreground/background 가정을 재현하지 않았다. 원 λ는 복사하지 않았다. [상세](METHODS_AND_SOURCES_B.md). |
| [Mozannar & Sontag, ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html) | §3–4, Eq.(1)–(6): 위임 대상의 실제 결과와 비용을 학습에 포함하는 원리. | 인간 expert 대신 고정 N0/E0와 실제 유한 pose 행동을 사용한다. 논문의 비용민감 surrogate 전체나 일치성 증명을 현재 가변 SE(3) 집합에 적용했다고 주장하지 않는다. |
| [Verma et al., AISTATS 2023](https://proceedings.mlr.press/v206/verma23a.html) | 공식 PDF에서 softmax와 OvA, confidence calibration 구분을 확인했다. | H1은 frame-balanced 다중 양성 BCE와 argmax다. 논문의 전체 surrogate·conformal ensemble을 구현하지 않았다. sigmoid는 검증된 안전 확률이 아니다. [L 상세](METHOD_L.md). |
| [Mao et al., ICML 2024](https://proceedings.mlr.press/v235/mao24d.html) | 공식 PDF §2의 bounded regression loss와 expert cost, Eq.(1)의 deferral loss, §4의 고정 predictor two-stage Eq.(3)을 읽었다. | 이번 목표는 공동 5cm·5° 성공의 이진값이다. 논문의 연속 회귀 비용 surrogate와 다르다. BCE의 모두실패 레이블은 6cm와 100cm를 구분하지 않는다. 연속 오차·꼬리는 별도로 전량 평가한다. |
| [SciPy 1.12 SLSQP](https://docs.scipy.org/doc/scipy-1.12.0/reference/optimize.minimize-slsqp.html) | 실제 설치 `_minimize_slsqp` 소스와 equality/inequality, Jacobian, 종료 조건을 확인했다. | analytic finite 제약과 최대 20-step restoration을 같은 80-iteration 시도 예산에 넣었다. optimizer 종료 성공, feasibility, finite support, 원 참조 성공은 별도 필드다. |
| [PyTorch BCEWithLogitsLoss](https://pytorch.org/docs/2.1/generated/torch.nn.BCEWithLogitsLoss.html) | 공식 손실 정의와 로짓 입력을 확인했다. 실제 환경은 `ENVIRONMENT.json`에 연결한다. | padding과 ineligible 항을 제외하고 행동별 평균 후 frame 평균을 취한다. 새 class weight·temperature·실사 gate는 탐색하지 않는다. |
| [Git worktree](https://git-scm.com/docs/git-worktree), [Git push](https://git-scm.com/docs/git-push) | 별도 작업트리/index와 일반 fast-forward 게시 규칙을 확인했다. | 사용자 원 작업트리·기존 기록을 보존한다. 운영 기본값·논문 변경 및 강제 push는 범위 밖이다. |

S4 PDF는 PMLR가 연결한 [공식 보관본](https://raw.githubusercontent.com/mlresearch/v235/main/assets/mao24d/mao24d.pdf)을 내려받아 텍스트로 확인했다. SHA256은 `06d3b0b82937a157b23084ba8ba6c7110420edec0ecd7e545f57fee18a9a1ed5`다. 원 PDF는 공개 결과에 복제하지 않는다. 외부 모델이나 외부 연구 코드를 새로 가져오지 않았으며, 기존 저장소 코드와 설치된 수치·시각화 라이브러리를 사용한다.

수학적 적용의 핵심 한계는 별개다. 유한 교차는 해당 profile와 모델 모서리의 대응 가정 아래 기하를 검사한다. 그 profile의 실제 material owner를 인증하지 않는다. H1의 실제 행동 레이블은 none→N0의 의미 혼동을 없애지만, 합성 영상 차이·참조 오차·모두실패 행동의 큰 연속 오차를 자동으로 해결하지 않는다. 실제 효과는 봉인한 결과표에서 판단한다.
