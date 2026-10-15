# V32 clean fresh B/I 연결 제안

설계만 작성했으며 CAL·PRIMARY 수치, 영상, 정답, 모델을 읽거나 실행하지 않았습니다. 기존 소스·관측·모델은 변경하지 않습니다.

기존 V31은 V29 `_run`을 복제하므로 `task.proposals`가 contender/refiner에 들어가는 경로를 상속합니다. V32는 이 `_run`을 재사용하지 않고, 금지 필드를 받지 않는 새 clean worker를 제안합니다. baseline, cached proposals, 과거 초기값/endpoint, 임의 pose guides는 worker 입력에서 제외합니다. e 결과는 별도 비교 스트림에 두고 새 결과 계산 후 부모 출력기에서만 합칩니다.

독립 first-input 출처를 검증한 R/t/K/Q/known-size로 `make_wd_families(..., initializations=None)`를 호출합니다. 각 가족에서 새 helper의 full B/I 및 고정 3/4점 SQPnP 모든 유효해를 구합니다. `Rphysical=Rnative@Q`, `center=Rnative@mean(xyz)+native_t`를 유지합니다. helper의 독립성 플래그는 caller 진술이며 upstream 증명을 대신하지 않습니다.

관측은 새 F32가 허용하는 초기17c/실측 B/I endpoint query만 활성화합니다. 원 raw archive는 증거로 보존하되, worker에는 exact selected-ref로 인증한 새 typed bundle을 전달해 임의 cached guide bank를 수치 입력에서 빼는 방식을 권합니다. 새 집단의 별도 CAL noise가 필요합니다. S 관측좌표와 U 점좌표는 읽지 않고 기존 모델 투영·제한 regularizer 및 범위 계약을 유지합니다.

시작점은 새32 native core로 평가한 fresh 후보에서 기존 최대6 diverse+독립 coarse 정책으로 고릅니다. 같은 source-policy의 native/C는 같은 시작점을 사용하며 C가 보존하는 endpoint도 같은 실행의 native 결과만 허용합니다. 과거 cached 시작점 이름과의 일치는 주장하지 않습니다. 현재 제약으로 모든 후보를 다시 검사하고, 해가 없으면 accepted=False인 자기 ORIGINAL coarse만 남깁니다.

구현 전 root 승인과 초기 입력·중앙선언·관측·새 CAL 모델 closure가 필요합니다. 금지 필드 poison, S/U/O/T gate, PnP/Q/중심, native/C 시작점, e 비교전용 분리, 전체238/zero/fallback 및 독립 비용·guard 검사를 제안합니다. 자세한 API와 필드는 JSON에 있습니다.
