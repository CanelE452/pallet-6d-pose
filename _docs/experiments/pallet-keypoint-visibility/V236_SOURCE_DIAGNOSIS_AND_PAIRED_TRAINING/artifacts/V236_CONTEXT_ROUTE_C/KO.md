# V236 원 q8 기반 full-context 입력 경로

기존 SOURCE metadata만으로 SYN7040의 원 q8와 실제 영상 raw buffer를 연결하는 정확 경로가 있다. 이는 CODE/선언 binding 경로 확인이며 실제 RGB 복원·성능 효과 확인이 아니다. 새 모델·데이터·분할을 선택하지 않았다.

|항목|확인 수|
|---|---:|
|selected 원 TRAIN/NOMINAL/lineage-bound 행|7,040|
|원 available / U|6,976 / 64|
|base / component / 실제 scene|110 / 110 / 440|
|선언 raw buffer path+SHA|440|
|실제 영상과 query anchor가 같음 / 다름|4,400 / 2,640|

각 행의 q/L/native/availability는 원 context_q8/L8/available8의 해당 slot과 같다. 실제 scene/native는 source_image_original_global_index와 V211 source_teacher_key에 일치한다. [7040행 연결표](SYN7040_CONTEXT_JOIN.csv)는 원 metadata/mapping row index와 q8 SHA, q8-only box를 보존한다. [440개 inert 원 RGB 표](SYN440_RAW_RGB_INERT.csv)는 실제 scene별 선언 NPZ와 member를 닫는다. source pointer 파일을 stat/hash/decode한 것은 아니다.

BEFORE_ANCHOR4는 원 BEFORE q8를 유지하면서 실제 COVER/SHAM 영상을 읽는 행이 있다. 따라서 context box는 그 입력행의 원 context_q8로 만들고 RGB는 actual_image_source_binding_inert로 연결해야 한다. anchor 영상을 대신 읽거나 실제 영상의 다른 q8/GT projected reference로 교체하면 입력 권위가 달라진다. query/native/availability·원component는 변경하지 않았다.

V107 target_context_math.py(aa7cd8e…)의 query_extent는 finite이고 [-1,-1]이 아닌 원 q8의 min/max만 사용한다. availability8 mask는 이 함수의 입력이 아니다. 이번 metadata에는 unavailable지만 finite/non-sentinel인 q8 slot이 행별 반복을 포함해 512회 존재하므로 이 차이를 기록했다. helper 정의를 바꾸거나 이 좌표를 새로 선택하지 않았다. 모든7040 box는 수학적으로 양수이나 실제 frame 지원을 인증한 것은 아니다.

224 grid는 x0+(j+.5)·width/224, y도 동일한 F64 좌표이며, 네 bilinear tap 중 영상 밖 tap은0·재정규화 없음, F32 HWC RGB와 모든 양수 가중 tap이 안쪽인 validity를 반환한다. box clipping/확장·GT reference/pose/ROI·q 보정은 없다. 잘못된 box나 실제 영상 밖 지원은 입력 지원 상태와 구분하며 semantic E로 만들지 않는다.

원 selected RGB96 packet만으로 crop 밖 RGB는 복원할 수 없다. 앞서 봉인한 renderer CODE 감사에서 full-image *_BUFFERS.npz의 RGB_uint8 member를 확인했고, 이번 saved110 pointer set과 V211 mapping의440 경로가 일치한다. 이번에는 renderer·raw GT110·NPZ를 재열람하지 않았다. filtered metadata에는 raw_hw가 없어 실제 RGB shape는 UNKNOWN이다. 후속 Root 실제 entry가 이440 container/member만 허용하면 opaque SHA와 uint8 HWC3 shape를 확인한 뒤 원 q8 box sampler를 적용하는 경로가 닫힌다. 여기서는 실행하거나 새 접근권한을 부여하지 않는다.

YCB는 기존 input1000 schema의 image{path,sha256}, raw_hw, initial_points, query_box_xyxy 경로와 같은 V107 순수함수가 있다. 이번에는700 RGB 경로나 input1000 body를 열거하지 않았다. YCB q는 GT-associated r+jitter의 SOURCE emulation origin이므로 GT 영향0으로 표현하지 않지만, r/pose/mask를 추가 배포 feature로 넣는 경로와는 다르다. 원 full-context/fixed-Q C11 SOURCE 시도는 이미 존재하며 신규성·최초 도입 주장은 없다. mixed family를 읽는 옛 producer 전체 run을 새 실행기로 재사용하지 않는다.

[정확 CODE·metadata·미허용 후속 읽기 목록](ROUTE.json)에 증거와 한계를 보존했다. context 누락이 이전 실패의 원인이라는 결론, 새 성능·모델 채택·current admission은 없다. 원 SOURCE internal FIT/DEV는 이후 별도 Root binding으로 보존해야 하며 이 감사는 새 split을 만들지 않았다.
