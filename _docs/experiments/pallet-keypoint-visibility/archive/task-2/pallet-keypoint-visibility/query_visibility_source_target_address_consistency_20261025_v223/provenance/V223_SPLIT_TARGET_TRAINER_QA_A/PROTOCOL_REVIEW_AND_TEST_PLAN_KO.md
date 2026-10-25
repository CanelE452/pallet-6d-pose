# V223 split-target protocol 사전 검토·최종 검사 계획

현재 판정: **프로토콜상 중대 차단 없음, 구현 검토는 B FINAL 뒤**. 읽은 문서는 root의 `V223_ROOT_PREPARATION/PROTOCOL_BEFORE_SPLIT_TARGET_MODEL_OR_PAYLOAD.json` 하나이며 SHA는 `5b38c7206197d1ba5158d619d306146b7108dc91d2572aec998ddbd09ea01c44`다. B 부분 파일·실제 model/weights/NPZ/PNG/GT는 읽지 않았다.

대조는 동일 architecture·C65 continuity init·공통252행/4 RNG·1100 update에서 **strong loss의 scalar routing**만 바꾼다. C68은 strong/natural/CF를 PART에, C69는 strong을 POINT에/natural·CF를 PART에 둔다. 두 팔에 같은 POINT/PART capacity를 두되 C68 POINT는 nuisance-unused라는 사실을 숨기지 않는다. shared encoder는 모든 loss의 gradient를 계속 받으므로 전체 encoder 간섭까지 제거한 실험으로 부르지 않는다.

FINAL 뒤 작게 확인할 항목은 다음과 같다.

1. **초기값·용량**: 원 C65 118359 state를 strict load한 뒤 direct65+spatial3137을 deepcopy한다. 총121561 및 예상42 state tensors는 실제 pure fixture에서 확인한다. 두 팔의 초기 전체state와 POINT/PART 출력이 같고 cloned storage는 공유하지 않아야 한다. 새 GT/ROI/reference feature는 없다.
2. **실제 gradient routing**: strong만 줄 때 C68 PART만/C69 POINT만 direct-head gradient를 받고, natural·CF만 줄 때 양팔 PART만 받아야 한다. shared encoder gradient는 유지하고 feature detach 또는 추가 독립 encoder는 없어야 한다. features를 한 번만 계산한다.
3. **unused optimizer 경계**: C68의 POINT direct/spatial6 tensors는 grad=None/AdamW state 부재가 정상이다. 모든42 parameter에 step1100을 요구하면 잘못된 gate다. 사용된 tensors의 state와 step1100을 검사하고 C69의 양 branch와 구분한다. unused parameter capacity는 결과 한계로 남긴다.
4. **loss·batch·RNG**: old192/point32/natural16/CF4×3의 동일252행과 RNG1810/1811/1812/1813을 양팔에 재현한다. 기존5loss·balanced strong·natural.25·parent-balanced CF.25는 불변이며 배열 slice192/224/240/252가 맞아야 한다. CF39 FIT 부모만 쓰고 모든 DEV descendant는 모든 loss/pool/normalization에서 제외한다.
5. **고정1100·freeze·출력 권위**: 새 probe나 intermediate selector는 없다. 양 final 뒤에만 DEV를 열고 primary natural/CF는 PART, strong apparent-fit은 각 실제 routed scalar로 집계한다. optional POINT weak 출력을 저장해도 finite-part 약라벨을 point gold로 채점하지 않는다.

이 실험은 ROI 주소/위치 추정의 실험이 아니다. 동일 condition에 다른 ROI가 있어도 RGB 차이가 남으며, exact identical-input 충돌은 별개다. 또한 warm C65는 이미 앞선 target이 섞인 state이므로 scratch target-pure 모델이라고 부를 수 없다. 어느 팔도 physical/native gold·원 initializer 정합·실사 개선·채택을 증명하지 않는다.

접근 한계: A의 이번 검토는 formal protocol 텍스트뿐이다. root가 C의 held360 metadata eager-parse incident 존재를 통지했지만 그 값은 A에게 전달되지 않았고 모델 판단에 사용하지 않았다. V223 전체의 protected metadata access가0이라는 주장은 하지 않는다.
