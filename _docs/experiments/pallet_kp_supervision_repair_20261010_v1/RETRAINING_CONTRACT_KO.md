# 추가 학습 전 고정 계약과 검토 경로

현재 상태는 **감독 준비와 무업데이트 CPU 검사 완료, 추가 학습 미승인·미실행**이다. 원래 세 조건의 3,000 업데이트, 합계 9,000회는 이미 사용했다. 이 문서는 수정한 감독으로 같은 실험을 한 번 재실행하기 위한 별도 9,000회 요청을 구체화한다. 실제 자세가 개선됐다는 결과는 아직 없다.

## 변경하는 것과 고정하는 것

바꾸는 입력은 `READY_PREPARED_TARGETS.npz`의 감독뿐이다. 실제 mesh의 소유 경계에 맞춘 POS와 유한한 앞 표면 증거를 가진 NONE을 사용하고, 불명확하거나 물리적으로 지원되지 않는 질의는 IGNORE로 둔다. 이 판정은 기존 근사 깊이 허용폭과 제공 마스크 정책에 한정된다. 영상 대비나 독립적인 실제 가시성 정답까지 증명하지 않는다.

원래 FP16 `[1024,84,28,65]` 특징 캐시와 3,000×16 배치 순서를 읽기 전용으로 재사용한다. 기존 RGB, 초기 키포인트 추정기, 특징 추출, 역할 정의, 5,890개 파라미터의 head, 초기화 seed 1, loss, optimizer, schedule, decoder와 후단 자세 경로를 유지한다. 새 RGB·feature detector 호출·학습 seed·하이퍼파라미터 탐색을 추가하지 않는다.

| 항목 | 고정 값과 검산 근거 |
|---|---|
| 세 조건 | `GEOMETRY_ONLY`, `IMAGE_NO_ROLE`, `IMAGE_ROLE` |
| 조건별 입력 | G는 image 0–18을 0으로 만들고 geometry 19–24 및 기하 역할 25–27을 유지한다. NR은 role 25–27만 0으로 만든다. IR은 28개 채널 모두 사용한다. 원래 구현의 정확한 의미다. |
| 초기 가중치 | 세 조건 공통 SHA `d40067b3ca409a216ba6f6e874ff64c29cf16b90f444c76196b422997e26d231` |
| 배치 순서 | 원래 raw tensor SHA `016ad8829614a9b9691954fff030b1d63648a1fd1bd149bf55558188d2df0b31`, train index 0–767만 포함 |
| 학습량 | 각 3,000회, batch 16, 총 9,000 업데이트·144,000 formal image exposures |
| optimizer | 기존 `AdamW(lr=.001, weight_decay=.0001)`, 100회 warmup, 기존 cosine schedule, gradient clip 10 |
| checkpoint | 3,000회 마지막 checkpoint만 사용. source/실사 점수로 선택하지 않음 |
| loss | 이미지별 유효 질의 평균 후 유효 이미지 평균. POS는 인접 bin 0–64 soft CE, NONE은 65번, IGNORE는 gradient 0 |
| source probe | 원래 초기 test128 및 1,000/2,000/3,000회 train128·test128. 168 head calls·2,688 exposures, 선택용 아님 |
| throwaway | 0회. 기존 100회 smoke optimizer 반복 대신 이번 CPU 무업데이트 검사 사용 |

## 실제 감독 준비 상태

| 분할 | 이미지 | POS | NONE | IGNORE |
|---|---:|---:|---:|---:|
| train | 768 | 26,049 | 11,780 | 26,683 |
| calibration | 128 | 4,465 | 1,974 | 4,313 |
| source test | 128 | 4,497 | 1,966 | 4,289 |

타깃 배열은 원래 feature index·family split에 그대로 대응한다. train/calibration의 13,757개 고정 질의에 한 번씩 광선을 계산해 13,754개 유효한 앞 표면 NONE을 확보했고, 무한대 3개는 IGNORE로 유지했다. 기존 POS·미복구 타깃·test·index·partition·dtype의 바이트 보존을 별도 감독 복구 단계가 검사했다. source test는 감독 버그 진단에 이미 사용됐음을 공개하며, fitting과 모델 선택에 사용하지 않는다. 정확한 감독의 ideal graph 상한은 학습된 정확도나 실제 자세 성능을 뜻하지 않는다.

## 이번에 실행한 CPU 검사

원래 첫 배치 `[363,393,579,729,26,110,632,728,191,239,667,325,209,635,197,314]`의 실제 특징과 새 감독을 사용했다. 16장·1,344개 질의는 POS 534개, NONE 291개, IGNORE 519개다.

세 모델 모두 519개 IGNORE의 logit gradient가 정확히 0이고, 825개 유효 질의에는 gradient가 생겼다. soft CE 해석적 미분과 실제 backward의 최대 차이는 `2.3283064365386963e-10`이다. G의 image 채널 gradient와 NR의 role 채널 gradient는 0이며 IR의 두 gradient는 모두 0보다 크다. 검사 전후 모델 state SHA가 동일하다.

실행량은 CPU head forward 3회·48 image exposures, `autograd.grad` 10회, full backward 3회, 별도 uniform-logit loss 평가 1회다. optimizer 생성·step·학습 업데이트·weight 파일 쓰기·detector·PnP·광선·새 RGB는 모두 0회다. 이 검사는 소스 fitting에 사용할 loss와 gradient가 작동함을 보여준다. 수렴이나 실제 자세 향상을 보여주지는 않는다.

## 승인 후 동일하게 실행할 실제 평가

새 세 마지막 checkpoint로 원래 66-way MAP, 동일 TLS와 교점 decoder를 사용해 319장의 관측을 먼저 봉인한다. 원래 Base 초기 자세와 같은 예측 self mask를 사용한다. 세 core 조건과 IR no-mask robust·IR standard·IR point-line의 여섯 경로를 각각 319장, 총 1,914행으로 평가한다.

point-only는 원래 8px 유한 4점 부분집합 합의와 top3 refit을 사용한다. self로 제외한 초기 좌표는 fit에 넣지 않는다. 새 R,t가 구해지면 해당 self 좌표를 최종 재투영으로 교체하고, 재투영점을 다시 독립 관측으로 fit하지 않는다. correspondence가 없을 때 초기 좌표로 관측을 자동 채우지 않는다. 기본 출력 반환은 `BASELINE_FALLBACK`으로 기록하며 새 자세 산출과 구분한다.

IR point-line은 같은 IR 관측의 원래 별도 경로를 유지한다. 교점에 소비한 선을 제거하여 같은 선에서 나온 point와 line을 중복 관측으로 세지 않는다. 이번 replay는 match-mass 또는 새 all-lines 경로를 섞지 않는다.

geometry 1,914행을 모두 봉인한 다음에만 reference를 읽어 position cm·rotation degree·ADDsym cm의 평균·표본 분산·표본 SD·중앙값·선형 P90·최대를 계산한다. 전체 319장 운용 결과, 공통 산출집합, 두 방법 모두 새 자세를 낸 집합, 직접 가시점 손상과 숨은점 재투영을 각각 보고한다. 기존 고정 13-session bootstrap draws를 그대로 재사용한다. 일차 비교는 repaired IR 대 N3→cornerSubPix이며, 세 모델의 모든 결과를 보고한다. mask 분류가 일부 틀렸거나 source 점수가 낮아도 실제 후단 평가를 생략하지 않는다. 실제 시간은 자원 경합이 없는 별도 구간에서 detector부터 최종 solver까지 전체 경로를 측정한다.

## 직접 확인할 파일

| 파일 | 확인할 내용 |
|---|---|
| [RETRAINING_PROTOCOL.json](RETRAINING_PROTOCOL.json) | 고정 설정·입력 SHA·3×3,000·1,914행 평가 계약 |
| [ZERO_UPDATE_CPU_CHECKS.json](ZERO_UPDATE_CPU_CHECKS.json) | 첫 배치 타깃 배열, 세 모델 loss/gradient, 초기화 SHA와 실제 무업데이트 실행량 |
| [AUTHORIZATION_GUARD_CHECKS.json](AUTHORIZATION_GUARD_CHECKS.json) | 승인 없음이 optimizer 이전에 차단됨, 출력 경로 10건 검사(보호 8·허용 2) |
| [READINESS_EXECUTION_RECEIPT.json](READINESS_EXECUTION_RECEIPT.json) | 이번 준비 단계의 바인딩·실행량·미실행 한계 |
| [AUTHORIZATION_PENDING.json](AUTHORIZATION_PENDING.json) | 현재 PENDING 요청. 승인으로 사용할 수 없음 |
| [retrain.py](../../../scripts/research/pallet_kp_supervision_repair_20261010_v1/retrain.py) | 원래 모델·loss 그대로, 새 target 배열·별도 출력·단일 승인 실행 claim |
| [downstream.py](../../../scripts/research/pallet_kp_supervision_repair_20261010_v1/downstream.py) | 원래 infer/evaluate 구현의 checkpoint와 출력 경로만 연결하는 adapter |

학습 실행에는 `status="APPROVED"`, `additional_formal_updates=9000`, 정확한 protocol/checks SHA와 명시적 사용자 승인 근거를 가진 별도 기록이 필요하다. 같은 승인으로 두 번째 실행을 시작하는 것은 exclusive `EXECUTION_CLAIM`이 차단한다. 현재 이 기록과 추가 업데이트는 없다. 승인을 받은 경우에도 실제 결과가 나쁘면 설정이나 seed를 바꾸어 반복하지 않는다.
