# Deep Hough Transform 팔레트 측면 외곽선 실험 프로토콜

[실험 결과 보고서](../../../data/pallet/results/deep_hough_side_v1/REPORT.md) · [Direct/DHT 시각화 갤러리](../../../data/pallet/results/deep_hough_side_v1/deep_hough_gallery.html) · [최종 step 검증](../../../data/pallet/results/deep_hough_side_v1/COMPLETION.json)

실험 ID: `deep_hough_side_v1`. 아래 설계는 본학습 전에 정했다. 결과는 이 문서가 아니라 `data/pallet/results/deep_hough_side_v1/`에 별도로 기록한다.

## 질문과 비교 범위

합성 이미지로 학습한 팔레트 측면 외곽선 모델에서, **후보 직선을 따라 영상 특징을 모으는 Deep Hough Transform(DHT)** 구조가 기존의 전역 attention 요약을 사용하는 Direct Hough보다 실제 이미지의 선 위치와 방향을 더 정확히 예측하는지 확인한다. 사용자가 선택한 대상은 측면의 긴 위·아래 선과 짧은 높이 선을 모두 포함한 측면 외곽이다.

두 모델은 같은 고정 VGG 특징, 같은 학습/평가 이미지, 같은 선 정답, 같은 Hough 후보 격자와 손실, 같은 seed·배치 순서·학습량을 사용한다. 구조와 파라미터 수는 다르므로 **동일 조건의 구조 비교**이며, Hough 연산 하나만 바꾼 단일 요인 ablation이라고 해석하지 않는다.

Han et al.의 DHT는 후보 선을 따라 특징을 합쳐 Hough 공간에서 선을 검출한다. 이번 실험은 그 특징 집계 원리를 팔레트의 역할별 선 예측에 적용한다. 원 논문의 다중 해상도 ResNet/FPN 및 전체 학습 절차를 그대로 재현하는 실험은 아니다. [ECCV 2020 논문](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/779_ECCV_2020_paper.php), [저자 공개 구현](https://github.com/Hanqer/deep-hough-transform).

## 정답: 좌우 측면 외곽 8개

`camera_dynamic_0123_v4`: 0–3은 가까운 앞면, 4–7은 먼 뒷면이다. 각 면의 순서는 위 왼쪽, 위 오른쪽, 아래 오른쪽, 아래 왼쪽이다. 왼쪽 측면은 `{0,3,7,4}`, 오른쪽 측면은 `{1,2,6,5}`이다.

| 새 head 번호 | 기존 12선 role | 끝점 | 의미 |
|---:|---:|---|---|
| 0 | 1 | 1–2 | 오른쪽 앞 높이 |
| 1 | 3 | 3–0 | 왼쪽 앞 높이 |
| 2 | 5 | 5–6 | 오른쪽 뒤 높이 |
| 3 | 7 | 7–4 | 왼쪽 뒤 높이 |
| 4 | 8 | 0–4 | 왼쪽 위 깊이 |
| 5 | 9 | 1–5 | 오른쪽 위 깊이 |
| 6 | 10 | 2–6 | 오른쪽 아래 깊이 |
| 7 | 11 | 3–7 | 왼쪽 아래 깊이 |

기존 cuboid 코너 정답을 그대로 사용하며 정답을 새로 옮기거나 GT pose로 다시 투영하지 않는다. 정답은 **직육면체의 구조적 지지 직선**이다. 팔레트 내부의 판재·격자 선은 대상이 아니다. 가려진 경계와 이미지에서 물리적 엣지가 없을 수 있는 경계도 포함된다.

주 평가 `support`는 양 끝점이 유효하고 원영상에서 길이가 2px 이상이며 선분이 원영상 사각형과 교차하는 경우다. 이는 물리적으로 보인다는 뜻이 아니다. off-frame 끝점은 실제 좌표를 유지한다. sentinel `(-1,-1)`와 비유한 좌표를 유효 GT로 취급하지 않는다.

별도 진단은 카메라를 향한 측면의 경계를 대상으로 한다. 바깥쪽 방향으로 정렬한 왼쪽 `(0,4,7,3)`, 오른쪽 `(1,2,6,5)` 면의 투영 부호 면적이 `-1px²` 미만일 때 해당 면이 카메라를 향한다고 판단한다. 네 코너가 모두 유효해야 한다. 이 조건은 물체 자체의 면 방향에 대한 기하 조건이며, 적재물·다른 물체·팔레트 구멍에 의한 실제 가시성을 판정하지 않는다. GT 면 방향은 평가 부분집합을 정하는 데만 사용한다.

본학습 전 `targets.py` 감사에서 5,374개 합성 면의 투영 방향이 world cuboid와 camera 위치로 구한 방향과 모두 일치했다. 거의 edge-on인 면 2개는 면적 기준에 따라 이 비교에서 제외했다. 실사 DEV52에서 주 평가 대상은 414개 선이며, 카메라를 향한 측면 부분집합은 25장·100개 선이다. [대상 생성 코드](../../../scripts/research/deep_hough_side_v1/targets.py), [감사 결과](../../../data/pallet/results/deep_hough_side_v1/target_audit/TARGET_AUDIT.json).

## 데이터와 입력 고정

이전 `hough_attention_transfer_v1/manifest.json` 및 특징 캐시를 그대로 재사용한다. 새 실험에서 실사 이미지를 학습에 넣지 않는다.

| 모집단 | 장수 | 사용 |
|---|---:|---|
| synth_train | 2,048 | 새 head 학습 |
| synth_val | 256 | 최종 checkpoint의 합성 검증 성능 확인 |
| synth_test | 256 | 같은 생성 계열의 미사용 합성 평가 |
| cross_v4 | 128 | 다른 합성 생성 계열 평가 |
| real_dev | 52 | 기존 실사 개발셋 전이 평가 |

모든 입력은 원영상에 100px 반사 패딩을 적용하고 400×400으로 리사이즈한다. 동일한 고정 VGG가 만든 `128×50×50` 특징을 사용한다. 백본은 `weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth`이며, 이전 실험에 기록된 SHA-256은 `0de80490cb3b4f9b11565db7a4aea6338f64edb8f9614910bfb52bf03ce0dc3f`이다. 백본은 새 실험에서 갱신하지 않는다. 모델 추론에는 GT crop, GT mask, GT 코너, GT 면 방향을 입력하지 않는다.

백본은 합성 사전학습에서 상속했다. 사전학습 원본 manifest가 삭제되어 새 합성 평가 이미지와 백본 사전학습 이미지의 완전한 중복 여부는 확인하지 못했다. 현재 head 학습 분리만으로 전체 파이프라인의 완전한 미사용 합성 평가를 주장하지 않는다.

## 두 구조

**Direct8:** 기존 Direct Hough의 역할 query 수를 8로 바꾼다. 고정 영상 특징과 위치 정보에서 역할별 attention으로 전역 descriptor를 만든 뒤, 각 Hough 후보의 특징과 결합해 역할별 후보 점수를 계산한다.

**DHT8:** 고정 영상 특징에 XY를 붙이고 학습 가능한 영상 convolution을 적용해 16채널 특징을 만든다. 고정 sparse DHT 연산이 각 특징 픽셀을 모든 theta에서 이웃 rho 두 bin에 선형 보간 투표한다. 누적 보간 가중치로 나누어 선을 따른 가중 평균을 사용한다. 이는 합산 기반 DHT를 이 실험에 맞게 정규화한 선택이다. Hough 공간의 convolution에 연속적인 선 좌표와 후보의 집계 질량을 함께 제공하고, 마지막 8개 출력 채널이 역할별 선 후보를 예측한다. DHT 앞뒤 convolution은 학습하며 고정 격자 기하를 학습한다고 주장하지 않는다.

두 모델의 차이는 특징 집계와 readout 전체를 포함한다. 결과가 좋아져도 DHT 집계, 영상 convolution, Hough convolution, 파라미터 수 중 하나만의 인과적 효과로 단정할 수 없다.

## Hough 좌표와 손실

50×50 특징 좌표 중심은 `(24.5,24.5)`이며 식은 `(x-24.5)cos(theta)+(y-24.5)sin(theta)=rho`이다. theta는 선 방향이 아니라 법선 방향이다. 수직선은 theta=0°, 수평선은 theta=90°이다.

원영상 픽셀 중심에서 특징 픽셀 중심으로의 변환은 `(original_xy+pad+0.5)*50/padded_image_size-0.5`이며 역변환은 `(feature_xy+0.5)*padded_image_size/50-0.5-pad`이다. OpenCV resize의 반 픽셀 중심 규칙과 실제 VGG의 stride 8·첫 receptive-field 중심 3.5를 함께 반영한다. 이전 실험의 캐시된 코너 격자는 `(original_xy+pad)*50/padded_image_size`라는 연속 좌표를 사용하므로, `make_targets` 안에서 이 차이를 보정한다. 실제 640×480 입력과 100px 패딩에서는 캐시 좌표에 x=-0.470238095, y=-0.463235294 특징셀을 더한다. 호출부가 중복 보정하지 않는다. 보정 후 특징 중심 `(24.5,24.5)`는 원영상 픽셀 중심 `(319.5,239.5)`와 정확히 일치한다.

- theta: 0°부터 179°까지 1° 간격, 180 bin.
- rho: -35부터 +35까지 0.5 특징셀 간격, 141 bin.
- 후보 수: 역할별 25,380개. 특징 픽셀에서 투표를 받지 않는 후보는 손실과 decode에서 제외한다.
- `(theta+180°,rho)`와 `(theta,-rho)`의 동일성을 target과 Hough convolution 경계에서 처리한다. theta 축만 단순 원형 padding하지 않는다.
- Gaussian soft target: theta sigma=1°, rho sigma=0.5 특징셀. 유효 후보에 정규화한 cross entropy를 supported role에만 적용한다.
- 최종 예측: 각 역할에서 유효 후보 중 최고 점수인 단일 무한직선. 실제 선분의 끝점을 검출한 것으로 해석하지 않는다.

`targets.py`의 정답→Hough→원영상 직선 왕복 최대 오차는 감사한 2,740장에서 0.0000357px, 0.00000947°였다. 특징 원점과 실제 영상 중심의 오차는 0px였다. 이는 좌표 구현 감사이며 학습 성능이 아니다.

## 학습 예산과 선택 규칙

| 항목 | 고정 값 |
|---|---|
| 본학습 seed | 1, 2, 3 |
| 각 구조·seed의 업데이트 | 6,000 step |
| batch | 12 |
| optimizer | AdamW |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| 선택 checkpoint | 고정 최종 step |
| 실사 학습·튜닝·checkpoint 선택 | 없음 |
| sanity | 합성 학습 32장, DHT seed 101, 1,500 step |

sanity는 정답을 학습할 수 있는지와 구현 오류를 확인하는 별도 실행이다. 본학습 결과에 포함하거나 초기 가중치로 사용하지 않는다. 본학습의 두 구조에는 seed별 같은 배치 순서를 준다. 서로 다른 구조에 동일한 수치 가중치를 공유했다고 주장하지 않는다. 학습 loss 곡선은 기록한다. `synth_val`은 본학습 도중 평가하지 않고 최종 checkpoint에서 평가하며, 중간 checkpoint 선택에 사용하지 않는다.

본학습 전 검토에서 위 반 픽셀 좌표 차이를 발견해 정답 생성과 직선 역변환을 수정했다. 수정 전 sanity·검증·설정은 결과 폴더의 `preliminary_continuous_grid_sanity/`에 보관한다. 수정된 좌표로 sanity와 검증을 다시 실행하고 본학습에 들어간다. 수정 전 수치를 최종 성능으로 섞지 않는다.

## 평가와 시각화

각 seed·모집단에 대해 원영상 좌표의 역할별 예측 직선을 평가한다. 주 평가 집합은 supported 8개 구조적 측면 선 전체이며, 카메라를 향한 측면 집합과 긴 깊이 선/짧은 높이 선별 결과는 보조 분석이다.

- 위치: GT 선분의 두 끝점에서 예측 무한직선까지의 수직거리 평균. 중앙값과 90백분위를 px 단위로 보고한다.
- 방향: 원영상에서 무방향 직선 사이의 작은 각도. 중앙값과 90백분위를 도 단위로 보고한다.
- 역할·실사 촬영 그룹별 표를 함께 확인해 전체 중앙값에 가려진 실패를 확인한다.
- seed별 수치와 세 seed 통계의 평균을 구분한다. 세 중앙값의 평균을 전체 선의 pooled 중앙값이나 신뢰구간이라고 부르지 않는다.

고정 seed 1의 원영상·GT·Direct8 예측·DHT8 예측, 역할별 DHT 후보 확률, 공간 특징 민감도를 함께 시각화한다. 공간 지도는 예측 최고 후보의 pre-softmax logit에 대한 고정 VGG 특징의 gradient×feature 절댓값을 채널별로 더한 값이다. 이는 **특징 민감도**이며 attention 가중치나 픽셀의 인과적 중요도와 같지 않다. 정답 위치로 gradient 후보를 고르지 않는다. 반사 패딩 영역과 원영상 경계를 표시한다.

구조적 선의 위치와 방향 오차가 실제로 줄었는지로 성능을 판단한다. 팔레트 주변에 밝은 지도가 생겼다는 사실만으로 전이 성공을 주장하지 않는다. 위치 중앙값 개선과 큰 실패 증가가 함께 나타나면 둘을 모두 보고한다.

## 해석의 한계

실사 DEV52는 이전 실험과 진단에 사용된 개발셋이며 새 독립 최종 테스트가 아니다. 동일한 실제 촬영 장면의 상관성도 남는다. 백본 사전학습과 합성 평가 이미지의 중복은 완전히 검증되지 않았다. 50×50 특징 해상도는 얇고 짧은 높이 선의 구분에 불리하다. 실제 물리적 엣지와 가려진 cuboid 경계를 분리한 수동 가시성 정답이 없으므로, 이 실험만으로 실제 visible-edge 검출 정확도를 확정하지 않는다. 전체 선·단일 팔레트 역할별 구조 추론이며 여러 팔레트의 instance 분리나 내부 판재 선 검출은 평가하지 않는다.

재현 코드: `scripts/research/deep_hough_side_v1/`. 결과·checkpoint·시각화: `data/pallet/results/deep_hough_side_v1/`. 대상 감사는 `/home/minjae/anaconda3/envs/pallet-pose/bin/python scripts/research/deep_hough_side_v1/targets.py`로 재실행할 수 있다.

## 재현·재개 명령

아래 driver는 모든 하위 작업의 cwd를 결과 디렉터리로 고정한다. 새 결과 폴더에는 간단한 `PURPOSE.md`와 정답 기하 감사를 먼저 만든다. 현재 진행 중인 학습 세션과 중복 실행하지 않는다. 이번 최초 실행은 driver 추가 전에 저장소 루트에서 시작했고 입력·결과는 절대 경로로 지정되었다. 기존 checkpoint가 있으면 runner는 설정 SHA를 검증하고 완료된 실행을 재사용한다.

```bash
cd /home/minjae/Documents/github/pallet-pose/data/pallet/results/deep_hough_side_v1

/home/minjae/anaconda3/envs/pallet-pose/bin/python /home/minjae/Documents/github/pallet-pose/scripts/research/deep_hough_side_v1/experiment.py --source /home/minjae/Documents/github/pallet-pose/data/pallet/results/hough_attention_transfer_v1 --run-dir /home/minjae/Documents/github/pallet-pose/data/pallet/results/deep_hough_side_v1 --steps 6000 --seeds 1 2 3 --batch 12
```

`experiment.py`는 CPU 정답 감사 후 `runner --phase all`의 검증→sanity→6회 본학습→모집단 평가→시각화 데이터 export를 실행한다. 이어 `analyze.py`가 분석·완료 검사·보고서를 작성하고 `visualize.py`가 PNG와 HTML gallery를 렌더링한다. 모든 하위 명령의 성공, `COMPLETION.json`의 최종 step·설정 SHA, 분석 입력 SHA와 gallery 존재를 확인한 뒤 실사 지표·판정을 stdout과 `DRIVER_COMPLETION.json`에 기록한다. 외부 알림은 보내지 않는다. 자동 완료는 사람이 시각화 이미지를 확인한 것을 의미하지 않는다. 필요 시 각 하위 스크립트는 기존 `--run-dir` 옵션으로 별도 실행할 수 있다.

사용자 선호: 이 프로젝트에서 HTML 시각화 결과를 만들면 작업 완료 후 로컬 기본 브라우저로 자동으로 열고, 결과 파일 링크도 제공한다. `visualize.py`는 렌더링 직후, `experiment.py`는 전체 완료 검증 직후 한 번 연다. Driver가 부르는 visualizer에는 `--no-open`을 전달해 중복 실행을 막는다. 두 명령 모두 `--no-open`을 지정하면 파일만 생성한다. `GALLERY_OPEN.json`과 `GALLERY_OPEN.log`는 로컬 브라우저 실행 요청 결과를 기록하며, launcher의 성공을 실제 창 표시 확인으로 해석하지 않는다.
