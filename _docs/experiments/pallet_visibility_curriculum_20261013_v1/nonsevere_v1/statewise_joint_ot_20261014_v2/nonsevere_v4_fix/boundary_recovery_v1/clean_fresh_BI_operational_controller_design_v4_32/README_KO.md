V32 새 실행기는 원본 작업을 검증하는 부모와 수치 계산 worker를 분리합니다. 부모는 원본 초기 형상과 B/I 관측을 정확히 연결하고 AUTH32가 선택한 요인만 새 입력으로 저장합니다. 기존 e 자세·후보 은행·초기화·다른 guide의 원시 관측은 worker 입력에 없습니다. e는 별도 비교 저널에 보존하고 네 새 방법의 계산이 끝난 뒤 부모에서 합칩니다.

worker 입력은 `schema`, `identifiers`, `clean_input`, `physics`, `typed_bundles`, `regularizers`, `source_contract`의 일곱 항목입니다. 독립 초기 자세와 원래 B/I 관측에서 새 SQPNP 후보를 만들고, 각 새 정책의 native 요인으로 고른 여섯 시작점과 자체 초기 자세를 native/C가 공유합니다. C에는 같은 실행에서 나온 같은 정책의 native 끝점도 남깁니다. 기존 O/S의 제한된 벌점과 형상 제약은 유지합니다.

잘못된 초기 자세는 정직한 `pose_available=False`로 남깁니다. 유효한 초기 자세에서 가능한 해가 없으면 자체 ORIGINAL 초기 자세만 실패 반환합니다. 두 경우 모두 e를 대체 자세로 사용하지 않습니다. S 영상 좌표·국소 질의·2D 요인은 없고 모델 투영과 기존 coarse-derived S 벌점만 유지합니다. U는 범위만 보고합니다.

새 관측 238행과 새 CAL 모델 20개가 봉인된 뒤에만 실행 준비를 허용합니다. 실제 준비·PnP·자세 최적화는 각각 후속 ROOT 승인이 필요합니다. 이 문서는 소스 연결 설계이며 새 방법의 정확도나 목표 달성을 주장하지 않습니다. 구체 API와 검증 범위는 [CONTROLLER_DESIGN_V1.json](CONTROLLER_DESIGN_V1.json)에 있습니다.
