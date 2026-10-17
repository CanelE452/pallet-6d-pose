# 실제 가시 팔레트 경계 입력보조 시험: 사전 계약

새 입력의 후보 비용이나 평가 결과를 보기 전에 고정한 계약이다. 시작 main은 `f87a3d4fcfbf5c2bc638b3fbe5613a83f707bb72`, 실행 지시문 SHA256은 `1ab73138285e7f9e13fcd4c42a1cafa8023753236e63e8f1da875df5072be574`이다. 이번은 사후 선정된 여섯 장의 입력보조 진단이다. 자동 시스템 성능이나 독립 시험으로 해석하지 않는다.

## 모집단·입력 경계

원 순서 `plastic_night_01:038630`, `wood_day_01:005286`, `wood_night_01:030698`, `plastic_day_01:020955`, `plastic_day_01:014704`, `wood_night_01:030898`와 전 retained 후보 수22/94/42/130/41/22(총351)를 유지한다. PRIMARY192나 CAL46로 확장하지 않는다. 이전 privileged whitelist exporter의 neutral package를 바이트 그대로 재사용한다. package SHA256은 `ec62902cf6e42ccabaa6a66a65339957d08c17abc445a2900a7735b9a806c7dc`이다. stage/family별 `outlines.configuration`, 원 weighted components, 원 방향별 dense 표본/normalizer와 각 후보의 native XYZ/R/t/physical registration Q를 사용한다.

별도 문맥의 모델이 원 full RGB640×480과 같은 RGB 확대만 보고 여섯 장 모두에 열린 바깥 경계를 한 번 작성한다. 입력 작성자에게 후보/기존 결과/참조/감사 그림/비용/성공 정보는 제공하지 않는다. 모델 작성과 모델 시각 검토이며 사람 검수나 자동 추출로 쓰지 않는다. 사례 선정의 사후성은 남아 있다. 콘 접촉 경계, 내부 무늬/구멍, 화면 잘림, 식별 불가능 구간과 가려진 연결은 배제·미확정으로 기록한다. 둥근 실제 외곽은 펴지 않는다. 끝점은 보이는 구간의 끝이며 코너 입력이 아니다. 한 번의 기술/시각 검사 후 입력 SHA를 고정하고 결과를 보고 바꾸지 않는다.

## 고정 점수

`C = full_BI + partial_normal + active prior`는 원래 가중된 값이다. `old_observed_weighted_cost`와 `old_model_weighted_cost`에는 각각 원 lambda/2가 이미 포함된다. `BASELINE_E`는 원 objective scalar를 그대로 반환한다.

열린 각 raw 선분 길이 l, 해당 원 설정의 간격 s에 대해 n=max(1,ceil(l/s)) midpoint를 만들고 각각 질량(l/s)/n을 부여한다. 새 질량 Z_E=sum(l/s), 원 방향별 observed 정규화 질량 Z_H를 연결해 r=min(1,Z_E/Z_H)를 고정한다. r은 후보 독립적인 길이 비율이며 독립 증거 수나 교정된 신뢰도가 아니다. 같은 원 패키지 내의 모든 후보·두 신규 질량 경로는 같은 r을 쓴다.

원 candidate 자체의 nativeXYZ/R/t/K로 투영한 직육면체 convex hull의 닫힌 **유한** 선분까지 새 관측 표본의 최소 유클리드 거리를 구한다. 관측 E는 열린 polyline 그대로이고 새 관측 hull이나 가림 구간 연결을 만들지 않는다. 원 outline sigma·delta·pseudo-Huber 계수를 그대로 사용해 질량 평균 A_E를 얻는다. 모델 target hull은 원 observed→model 관례대로 canvas clip하지 않는다. 방향 loss·각도 threshold·새 대응·fit은 없다.

| 경로 | 고정 scalar |
|---|---|
| BASELINE_E | 원 J_E |
| HULL_ONEWAY | C + old_observed_weighted_cost |
| HULL_ONEWAY_MASS | C + r × old_observed_weighted_cost |
| VISIBLE_SUPPORT | C + lambda/2 × r × A_E |

이는 원 outline 항의 교체다. full/partial/prior·O/T/U 유한구간·depth·원 eligibility/dedup을 보존한다. 입력 없음 또는 원 outline 비활성/부모 위임이면 세 신규 경로 모두 원 scalar/rank/pose와 정확히 같다. 활성 처치와 무입력 항등성을 구분한다. 실제 입력이 있는 성공 반례5·6도 같은 점수 변경을 받는다. 중복 RGB 출처 가능성은 보고하되 별도 파일이나 가까운 픽셀만으로 독립/동일 물리 변이라고 단정하지 않는다.

## 선택·봉인·평가

G10 각 family 전체 retained 후보를 원 comparator(원 initialisation 및 outcome 순서 포함)로 고른 뒤 원 accepted-family 순서를 적용한다. 그 family의 G13을 같은 방법으로 선택한다. G10과 G13 objective를 직접 비교하지 않는다. 후보 생성·refine·가중치 탐색 없이 네 경로×6=24개 출력과 자기 family R/t/Q를 봉인한다. 봉인 후 별도 프로세스에서만 원 fixed reference를 결합한다. 같은 후보의 중심위치≤5cm AND physical/C2 회전≤5°가 성공이며90° 오류를 지우지 않는다. 원 참조는 독립 물리 정답이 아니고 미확인 K/등록 부호·일부 geometry ambiguity의 한계를 유지한다.

전체6장의 평균±표본SD·최대, baseline 및 질량 대조 대비 paired 회복/손상/양쪽 성공/양쪽 실패와 연속차를 보고한다. 봉인 후 모든351개 후보에서 성공 가능한 후보를 설명용으로 찾고 A_E 단독·C 결합·G10 family·G13 선택 중 어디에서 잃는지 구분한다. 설명용 성공 후보를 selector에 되돌려 넣지 않는다. 이전 부모 제외/질량 대조/N0/PoseFix RAW는 저장된 동일6장 결과만 읽는다.

## 검산·비용·보존

원 e 후보/family/pose의1e-9,rtol0 의미별 재현, no-input scalar/rank/pose 항등성, 원 dense 잔차와 native hull 투영 일치, 새 유한 거리/질량/모든351개 네 비용/24개 선택/24개 오차의 별도 구현 검산을 수행한다. 입력 원좌표·hash·zero-length/중복 검사와 실제 읽은 RGB/검토 범위를 남긴다. 새 공개 namespace만 commit·원격 main 반영하고 실제 원격 바이트를 확인한다. 사용자 dirty 파일·과거 브랜치·기존 관측/참조/weights·논문·운영 기본값은 보존한다.

새 학습/NN forward/SAM·콘 입력/PnP·수치 최적화/후보 생성/seed/데이터/참조 수정은 각각0이다. CPU 동시 작업 최대2, 점수·갤러리·독립 검산 합계 wall≤60분. 각 실제 프로세스 wall/CPU/peakRSS를 기록하며 RGB 시각 작성·복원 시간은 분리한다. 런타임의 정상 임시파일·폰트 캐시와 scientific reference/audit 입력 차단을 구분한다. 구현·패키징 오류는 이력을 남겨 수정할 수 있으나 새 관측·점수의 의미 버전은 탐색하지 않는다. 나쁜 결과도 전체 분모에 남긴다.

기존 `image_supported_edge_chains_v4_17c.py`에는 이미 RGB 경계점·chain·불확실성 추출이 있다. 초기 projection strip/native edge 대응에 조건화되므로 이번 RGB-only 입력 작성에 호출하지 않는다. 이번은 최초의 영상선 사용이나 아래 논문의 재현이라는 주장이 아니다. SRT3D(DLR 기관 설명), FDCM(MERL 초록), uncertainty-aware PnPL(CVF 공식 초록)의 실제 열람 범위와 차이는 별도 source map에 기록한다.
