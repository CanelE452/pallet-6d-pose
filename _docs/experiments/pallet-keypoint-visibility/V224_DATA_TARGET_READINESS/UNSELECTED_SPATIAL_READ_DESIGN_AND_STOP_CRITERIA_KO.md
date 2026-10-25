# V224 공간 read 제안과 현재 중단 기준

**현재 접근의 실사 성공 가능성은 아직 판단할 근거가 부족하다. 새 attention 구현·모델 생성·학습은 하지 않았다.** V223의 출력 분리는 CF DEV17/27 및 natural DEV9/12를 개선하지 못했고, triplet 전체 정답은2/9→1/9였다. 이 결과는 공간 표현이 원인이라는 증거도, 불가능하다는 증거도 아니다. 현재는 더 작은 학습을 이어가는 대신 이 갈래의 결론과 다음 판단 조건을 고정해야 한다.

## 무엇을 바꾸는 안인가

미선택·미구현 예시: C69의 동일121561-parameter 모델을 두 팔 공통 warm start로 두고, encoder의64×12×12 feature를 기존3×3 pooling 전에 읽는 PART residual을 추가한다. 기존 후기 native/other-mask/XY32 조건을 새 정보라고 부르지 않는다.

- query: 기존 condition32→16 무편향 선형,512 parameters.
- key: feature64+고정 XY2→16 무편향1×1,1056 parameters. 공간적으로 상수인 key bias는 softmax에서 상쇄되므로 넣지 않는다.
- attention으로 읽은64-vector→PART scalar,65 parameters. 마지막 weight/bias를0으로 초기화하면 최초 출력은 C69와 같다.
- 추가1633, 전체123194 parameters라는 정적 계산이다. 실제 module/forward/속도/성능 검증은0이다. 12×12 이전에 CNN이 잃은 미세 정보를 이 read가 복원하지는 못한다.

두 팔 모두 같은 조건부 read를 쓰고 treatment만 training-only ROI bag KL을 추가하면, 검증되는 것은 **약한 공간 target auxiliary의 효과**다. conditioning 효과 자체의 대조가 아니다. classification만으로 attention을 배우는 control도 같은 구조·초기화다. zero residual 때문에 초기 classification gradient는 attention 내부까지 바로 흐르지 않는 반면, ROI auxiliary는 직접 흐를 수 있다는 차이도 처리의 일부다. auxiliary가 shared encoder도 바꾸므로 효과가 나도 정확한 부위 localization 하나의 인과 효과라고 단정할 수 없다.

ROI teacher는 기존 FIT39 D부모의 두 관찰 ROI exact-OR bag을 coarse cell mass로 정규화하는 훈련 target일 뿐이다. bbox를 segmentation mask로 바꾸지 않는다. COVER에는 가려진 동일 목표의 위치 sidecar로만 유지할 수 있으며 visible part가 보인다는 teacher가 아니다. natural E6의 foreign-evidence ROI를 visible-part teacher로 자동 전용하면 안 된다. DEV는 teacher·fit·normalization에서 제외한다. ROI/pose/GT reference는 deployment feature가 아니고 원q/L/RAW96도 바꾸지 않는다. coarse cell convention은 물리점 투영 정답이 아니다.

대안인 조건부 read 대 비조건부 read(ROI auxiliary 모두0)는 질문 주소를 pooling 전에 쓰는 효과를 더 직접 검사한다. 다만 condition을0으로 했을 때 query까지0이면 uniform pooling이 되어 학습되는 비조건부 attention과 다른 대조가 된다. learned global query를 유지해야 하고 control의 condition projection은 unused capacity가 된다. 현재처럼 FIT 분류가 이미 완전 정답인 작은 자료에서는 새 read가 학습되지 않은 부정 결과를 정보 부족과 구분하기 어렵다. 지금 어느 안도 실행 대상으로 선택하지 않았다.

## 설계만으로 해결되지 않는 부분

SOURCE FIT39 CF 묶음은39개의 독립 실사 장면이 아니다. DEV9묶음도2 frame/component와 그 jitter이며 이미 관찰된 자료다. natural45/12와 CF의 일부 입력은 중복된다. 따라서 이 자료의 완전 FIT 또는 작은 DEV 변동으로 실사 일반화를 예측할 수 없다.

이번6297행의 공통 입력 fingerprint에서 반대 target 그룹0은 확인됐지만, synthetic own-image point0 상태와 공개 finite-part weak target이 같은 질문이라는 뜻은 아니다. GT-derived training q8와 실제 production initializer가 같은 부위/instance를 주소화한다는 증거도 아니다. local wood/관찰 ROI는 physical native gold가 아니다. 실제 부위 대응·자연 외부/자기 가림·원query 오류/지원불가의 경계를 분리한 자연 SOURCE 표본과 독립 장면 coverage가 더 필요하다. 이는 완벽한 mesh gold를 요구하는 조건이 아니며, 기존 공개 TRAIN 이미지에서 실용적인 약한 대응 관찰로도 확인할 수 있다. 더 많은 데이터를 확보해도 성공이 보장되지는 않는다.

## 지금의 종료·재개 판단

1. **현재 추가 구현·학습을 멈춘다.** C68/C69 scalar 분리 갈래는 이번 고정 조건에서 공동 개선 없음으로 마감한다. 작은 변형·epoch·seed를 계속 늘리며 성공을 기다리지 않는다.
2. 다음 비용을 쓰기 전 intended instance/part, D/E/SELF, 위치 오류·지원불가의 질문과 허용되는 약감독 권한부터 합의해야 한다. 원 initializer 주소가 불명확한 행은 UNKNOWN으로 남긴다.
3. 새 실행을 결정한다면 한 번의 고정 대조에 대해 독립 SOURCE 장면 분모와 D 보존/E 검출의 최소 실익·허용 손상, 예산, 종료 조건을 먼저 수치로 정해야 한다. 정하지 못하면 실행하지 않는다. 그 대조에서 양축의 실익이 없으면 같은 갈래의 반복을 종료한다. 보호 평가로 arm·문턱을 고르지 않는다.
4. 실사 성공 또는 불가능을 현재 자료로 장담하지 않는다. 사용자가 비용을 더 들일 판단을 할 수 있도록 현재 미해결과 예상 검증 비용을 제시해야 하며, 이 문서는 후속 실행 권한이 아니다.

근거는 앞서 직접 읽은 V223 root protocol5b38c720·최종 ROOT_RESULT0ca2bc40, B saved QA c9b6e50e, V181 base14a40c4d, V202 spatial a03f2f4f, V223 split head1d305d80/runner0e9714dc 및 A의 schema-only 논리 확인이다. 이번 V224에서 새 source 파일·array·PNG·GT·checkpoint·보호 평가를 읽거나 모델을 실행하지 않았다. 기존 frozen 폴더는 수정하지 않았다. 위 parameter 수는 코드 실행 없이 차원 곱셈으로만 계산했다.
