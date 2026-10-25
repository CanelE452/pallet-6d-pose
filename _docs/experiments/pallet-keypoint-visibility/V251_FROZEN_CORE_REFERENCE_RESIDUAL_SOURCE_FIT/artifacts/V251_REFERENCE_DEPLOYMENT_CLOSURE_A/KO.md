# V251 배포 reference 입력 연결

현재 samples/features에서 바로 만들 수 있는 것은 SOURCE FIT-D의 **원 native 주소0..7별** reference bank입니다. class·material·CAD bank는 가정하지 않습니다. condition32 첫8은 기존 native 슬롯의 onehot이며, 원 q8/L/U를 바꾸지 않고 SOURCE family0/fit/available/targetD와 연결할 수 있습니다. 실제 주소별 지원수와 descriptor 유효성은 아직 배열을 읽지 않아 UNKNOWN입니다. 빈 bank를 다른 주소로 대체할 권한은 없습니다.

features1186 = full96 CLS384 + 고정중앙 Q384 + centre32 CLS384 + valid fractions2 + 기존 condition32입니다. 원 cached1152는 이 앞 세 appearance view입니다. 이 native 주소가 정확한 canonical anatomical part·깨끗한 part template·실제 pallet class를 뜻한다는 증거는 아닙니다. SYN 라벨은 saved intended-native witness0의 SOURCE 감독이고 YCB는 associated-r 표면 감독/nativeUNKNOWN입니다. 두 namespace를 섞지 않습니다.

상류 TRAIN scene metadata에는 factors(material/aspect/geometry_family), CF→canonical, base/scene 주소가 있으나 V219 조립 cache는 META6+lineage/context만 남겼습니다. 수정된 native-address bank에는 이 상류 factor join이 필요하지 않습니다. 실제 배포 initializer가 pallet class ID를 준다는 CODE 계약은 이번 범위에서 확인하지 못했습니다. q8·기하만으로 class를 만들어 넣지 않습니다.

READINESS.json에 exact CODE SHA/line과 미확인 권한을, dependency_map.json에 기존 공개 feature/node/context/map/geometry/samples 및 YCB bank의 SHA/size/상대 경로를 기록했습니다. 공개 경로는 이미 봉인된 metadata map을 재사용했으며 새 Git/blob·원본 배열 검산은 하지 않았습니다. SOURCE row/array/CSV·RGB/GT/weight·현재/held 자료 접근, 모델·학습 실행은 모두 0입니다.
