# N3 입력·노출 감사

이 문서는 전달 번들의 실행 지시와 `source_context`를 결과 증거로 오인하지 않도록, 실제로 읽은 데이터와 모델 입력을 분리해 기록한다. 모든 해시는 `PROTOCOL.json`, `PREFLIGHT.json`, 각 cache/training receipt에서 다시 검산한다.

## 학습·선택 집단

| 집단 | 고정 행 수 | 역할 | 실제 성능 선택에 사용 |
|---|---:|---|---|
| synthetic train | 55,980 | N3 최적화 순서의 모집단 | 예 |
| synthetic calibration | 1,004 | seed별 온도 `{0.5,1,2,4}` 선택 | 온도에만 사용 |
| synthetic selection | 1,031 | source 구성 감사용 | DEV·square 선택에 사용하지 않음 |
| synthetic held-out | 1,985 | 온도 고정 뒤 source-only 기술 결과 | 온도 선택에 사용하지 않음 |
| DEV319 | 319 | 고정 실사 개발 평가 | 학습·온도 선택에 사용하지 않음 |
| GREEN0918_119 | 119 | 별도 정사각형 2D 전이 감사 | 학습·온도 선택에 사용하지 않음 |
| recorded lifter | 4 usable sessions, 846 fixed samples | 1 s sensor-time 오프라인 재생 | 학습·선택에 사용하지 않음 |

synthetic source manifest는 60,000행, SHA-256 `feaa24075d31c4227e3397b7450a19a0f1d3a73dc0203d12a8ca5b927eb59789`이다. 치수·대칭 sidecar는 행 번호 `0..59,999`와 정확히 대응하며 SHA-256은 `4746bbf40db027fbf34d8b33d3d6e8b0b9dbf5ddc69a73ba6a3f8934d01ace82`다.

## 기반별 synthetic 사용 가능 행

| Backbone | Train usable | Calibration usable | Selection usable | Held-out usable |
|---|---:|---:|---:|---:|
| DOPE | 44,063 | 774 | 820 | 1,574 |
| ResNet-18 CONSTANT-fold | 55,806 | 989 | 1,018 | 1,950 |

이 차이는 각 고정 기반의 검출·코너 유효성 차이이다. DOPE나 ResNet 분모를 YOLO의 2,445 observed corners에 강제로 맞추지 않는다. 각 fit은 usable train 행에서 seed별 고정 순서를 만들고 정확히 6,000 batch × 16 = 96,000 표본 노출을 사용한다.

## 실제 입력 경계

| 단계 | 허용 입력 | 금지한 입력 |
|---|---|---|
| frozen base | RGB와 고정 전처리 | W,D,H, GT corner/pose/box, 가림 등급 |
| N3 | frozen 특징, 초기 9점, 예측 box, point-valid mask, 등록 `[W,D,H]` | GT corner/pose, 평가용 가림·가시성, 최적 GT branch |
| PnP, base와 N3 공통 | 예측 corner, K, 같은 물리 치수 | GT pose로 W/D 교환, GT 대응 재선택 |
| evaluation | 고정 GT corner/pose, material/occlusion label | inference 결과나 온도·checkpoint 선택 |

대칭 branch 선택은 synthetic 학습·calibration target 생성에만 사용한다. DEV319, GREEN0918, 리프터 추론 경로에는 GT branch 선택 함수가 없다.

## ResNet-18 기반의 증거 경계

기존 60-epoch pure-RGB receipt의 decoder는 144점 중 유효점 0개였으므로 주 평가 기반으로 사용할 수 없었다. 실제 사용한 기반은 완료된 10-epoch `CONSTANT` checkpoint `a94e55f...d8e1c4`이며, 항상 0인 context에서 FiLM을 convolution weight/bias에 대수적으로 접어 표준 RGB-only forward로 변환했다. 접기 전 0-context logits와 접기 후 RGB-only logits의 parity를 시험했고, 런타임 base API는 이미지만 받는다.

따라서 이 결과는 **10-epoch CONSTANT-fold ResNet-18 기반의 N3 결과**이다. 60-epoch pure-RGB ResNet-18 결과로 이름을 바꾸지 않는다.

## 실사와 self-training 노출

- DEV319는 과거 개발·분석에 이미 노출된 집단이며 새 봉인 TEST로 주장하지 않는다.
- GREEN0918_119는 고정 치수 `[1.10,1.10,0.15] m`인 개발 2D 감사 집단이다. 집단 내 치수 변화가 없으므로 치수 입력의 인과 효과를 단독으로 식별하지 않는다.
- 기존 self-training 비교는 DEV319 전체에 대해 공정한 공통 노출 계약이 확인되지 않아 전체 표는 `x`다. hash가 연결된 별도 HELDOUT128 안전 공통집단만 재집계한다.
- 리프터 기록에는 독립 GT corner/pose가 없다. coverage·결측 연속 길이·정지 구간 변동만 기술하고 위치·yaw 정확도는 `x`로 둔다.

