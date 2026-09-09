# 2026-09-07: HTML 자동 열기, 측면 범위, 패딩과 정면 실패

사용자 선호: HTML 결과를 만들면 바로 브라우저로 연다. `visualize.py`, `experiment.py`, `explain_followup.py`가 기본 자동 열기를 지원하며 `--no-open`으로 끌 수 있다. 통합 실험은 검증 후 한 번만 연다. 기존 갤러리와 추가 진단 화면을 실제 데스크톱 브라우저로 열었다.

[대화형 진단 화면](../../../data/pallet/results/deep_hough_side_v1/diagnosis_followup/viewpoint_padding_diagnosis.html)에서 좌우 측면 8개와 이번 출력에서 제외한 앞·뒤 폭 방향 가로선 4개를 켜고 끌 수 있다. 이 도식은 원본 GT로 학습 대상의 범위를 설명하며 예측 결과가 아니다. 정면의 긴 앞면 가로선 2개가 빠진 점은 이번 8개 측면 실험의 한계다.

[시점 진단](../../../data/pallet/results/deep_hough_side_v1/diagnosis_viewpoint/VIEWPOINT_FINDINGS.md): 측면/앞면 면적비≤0.1은 학습 1,211/2,048장으로 흔하다. 윗면/앞면 면적비≤1은 학습 5/2,048장, 실사 22/52장이다. 합성 카메라 고도는 15° 미만이 0장이며 world 좌표로 확인했다. 낮은 윗면 노출의 실사 22장은 야외 22장과 정확히 겹쳐 시점·배경·촬영 그룹의 효과를 분리할 수 없다. 학습 보강의 인과적 효과는 아직 검증하지 않았다.

[패딩 진단](../../../data/pallet/results/deep_hough_side_v1/diagnosis_padding/PAD_FINDINGS.md): 고정 seed 1 모델과 같은 100px 패딩에서 반사/검정/평균색을 비교했다. 실사 DHT 위치 중앙값은 7.18/9.03/9.59px, P90은 76.02/145.03/145.99px다. 반사로 학습한 모델의 추론 입력 변경 진단이며 패딩별 재학습 비교가 아니다. 기존 반사 입력의 선택된 모든 Direct/DHT 오차를 이전 CSV와 정확히 재현했다. 원본 학습 가중치·설정·수치 결과는 유지했다.

반사 선택의 근거는 앞선 Direct 실험의 전처리 조건 유지였다. 반사 우위를 사전 검증한 선택은 아니었다. 현재 사방 100px 패딩은 전체 입력 면적의 46.2%이며, 50×50 특징에서 원본은 약 38.1×35.3칸이다. 색만 공백으로 바꾸어도 이 공간 해상도 감소는 남는다.

다음 비교 후보는 앞·뒤 가로선을 포함한 12개 선, 낮은 고도·정면·원거리 조건 보강, 검출 ROI 확대/높은 특징 해상도, 유효 영역을 반영하는 DHT, 알려진 치수와 보정 카메라를 이용한 기하적 전체 구조 추론이다. 이 후속 학습은 아직 실행하지 않았다.

추가 진단 HTML은 브라우저에서 8→12개 전환, 면 선택, 두 실제 이미지 전환과 이미지 로드를 검증했고 JS 오류는 0건이다. 기록은 `diagnosis_followup/browser_qa/QA.json`과 `diagnosis_followup/COMPLETION.json`에 있다.

재현: `pallet-pose` 환경에서 `viewpoint_diagnosis.py`, `padding_probe.py`, `explain_followup.py`를 사용한다. 코드 위치는 `scripts/research/deep_hough_side_v1/`이며 각 CLI의 `--help`에 입력·출력 경로가 있다.
