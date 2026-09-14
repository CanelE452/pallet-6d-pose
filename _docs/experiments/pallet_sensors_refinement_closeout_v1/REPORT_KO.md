# 고정 P 원고 검증 결과

실행 상태 PARTIAL; 개발 근거 P_DEV_SUPPORTED; 논문 상태 NEEDS_BOTH.

새 본 학습은 D만3seed×6000=18000 updates, disposable smoke1회, 재개0회입니다. R0/P/L 재학습은0입니다. 모든 D 최종 checkpoint를 사용했고 성능 기반 재시도는 없습니다.

R0 pooled9 median 6.6157px → P seed평균 5.9049px. 차이 -0.7108px, 세션95% [-1.1737,-0.4111]. 같은311 matched frames/2756 points이며 전체GT PCK 분모는2818점입니다. R0 PCK10 63.733% → P 67.341%.13개 재사용 개발 세션의 관측입니다.

P−D median 차이 -0.5254px, 세션95% [-0.9800,-0.2101]. P has a lower development median than this fixed direct control. 같은 샘플링/optimizer/노출이지만 loss·출력범위·readout이 달라 softmax 하나의 인과분해가 아닙니다. D 고정train/cal probe는 결과와 함께 공개하며 일반 회귀 방식의 원리적 열세를 주장하지 않습니다.

P−D P90은0.200px로 P가 관측상 약간 크며, 세션95% [-1.855,1.247]는0을 포함합니다. D가 런타임도 더 낮아 P의 모든 지표 우세라는 결론은 내리지 않습니다.

P의 geometry-reference translation median 감소는 seed평균7.439mm입니다. 독립 장치의 실측6D GT나 실제 삽입 성공 개선이 아닙니다. 모든2D/6D/시간 수치는 새 원고 RESULTS_DRAFT.md에 자동 연결했습니다.

GPU는 실제 RTX3080. 같은26영상×5repeat, 모델별20warmup, 균형순서로 BGR→2D와 PnP 포함 시간을 측정했습니다. Display/RustDesk 유지; Jetson 미측정. 기존 timing은 보존했습니다.

PoseFix 공식 소스/라이선스를 확인하고 실제 합성train8장 adapter를 검사했습니다. 공식 네트워크의TF1환경·9점연결·GPU자원 실측은 미완료이며 formal prior 성능 비교는 하지 않았습니다. CRT-6D 공식 저장소에는 실행 코드가 없습니다. 독립FINAL manifest는 모두 UNAVAILABLE이며 새 촬영·사람의 blinded annotation/QA가 필요합니다. 이 두 항목 때문에 PAPER_READY라고 하지 않습니다.

원고에 쓸 수 있는 범위: 고정 estimator의 특징을 재사용하는 소형 정제기, 검사된 wrapper 보존 계약, 현재DEV의 대응 정확도 변화와 실제 desktop 비용. 아직 쓸 수 없는 범위: SOTA/최초/독립일반화/정식선행우월/Jetson실시간/지게차안전보장.

시작 main bb952ad4d81be39a3c1a88d97f16b741b521e632. 최종 commit/push SHA는 Git 게시 확인 메시지와 기록을 따릅니다. 기존 scientific verdict와 _docs/paper/final은 보존되었습니다.
