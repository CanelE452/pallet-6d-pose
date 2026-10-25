코드에서 output indexing·loss mask·full/centre 행 복원 오류는 발견하지 못했습니다. family0→output0(SYN), family1→output1(YCB), target0=D/1=E와 저장 decode가 일치합니다. FIT·available·known만 loss에 들어가며 각 family 8D+8E입니다. 기존 B QA의 7617행 정합 검산은 인용했고, 이번에 배열을 다시 열어 검산하지 않았습니다.

YCB condition32는 전부0입니다. 따라서 native/8-point index·GT instance·associated reference r·r−q는 head의 명시적 입력이 아닙니다. 원 q를 중심에 둔 crop의 모습이 암묵적인 target 주소입니다. 입력 필드의 부재만으로 식별 불가능을 입증할 수는 없습니다. 동일한 전체 입력에 반대 target이 있는 실제 충돌 증거는 이번에 확인하지 않았습니다.

full96 CLS는 원 frame 전체가 아니라 해당96 crop의 문맥입니다. centre32 CLS와 중앙4token 평균 Q도 같은 q에 정렬됩니다. Q는 RGB4pixel 판정이 아니라 transformer 이후 patch descriptor이므로 crop 안의 넓은 attention 문맥을 포함할 수 있습니다. Crop 밖 RGB와 r 위치는 공급되지 않습니다. SYN native/q8 조건은 descriptor 뒤에 붙고 YCB에는 그 조건도0입니다.

서로 다른 SYN witness0와 YCB associated-r target은 출력 두 개로 분리했지만, backbone과 head의 첫 hidden layer는 공유합니다. 이것은 코드 사실이며 교사 간 간섭이 실패 원인이라는 증거는 아닙니다. YCB FIT E283/283 대 DEV E0/57, 양팔 DEV 순개선0은 저장 결과입니다. 주소·문맥·분포·최적화 중 어느 원인인지 이 숫자만으로 확정하지 않습니다.

과거 full-context/fixed-Q DINO·associated-r supervision·native/finite-part head 사용 이력은 그대로 인정합니다. 이번 코드는 새 아이디어의 최초 시도가 아닙니다. 실제 weight/prefix/NPZ/RGB/GT/보호 본문, 새 forward·학습·문턱 선택은 모두0입니다. 검증 가능한 가설만 JSON에 남겼으며 해결 불가능·라벨 오류·특정 구조의 인과를 단정하지 않습니다.
