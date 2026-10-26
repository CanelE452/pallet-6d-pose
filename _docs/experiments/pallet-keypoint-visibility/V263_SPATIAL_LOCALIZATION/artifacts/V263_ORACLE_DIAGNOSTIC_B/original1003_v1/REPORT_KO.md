# V263 고정 후단 SYN 진단

원 공통 1,003장(원 TRAIN768/반복 노출 VAL235)을 전수 재생했다. 원 finite9×2, native 첫8 대응 index, K/치수, W/D 선택, SQPnP와 기본 LM, source basis를 유지했다. 전체 결과의 저장 N0 일치를 검사했으며 설정·문턱·초기값을 최적화하지 않았다.

GT visibility-only / 직접 관측 GT location-only / both는 실행 가능한 물리 oracle 계약이 없다. 기존 native8은 원 camera_dynamic_0123_v4의 cuboid 투영 채널이고, 원 render의 점별 물리 표면 지원·depth/instance-owner가 인증되지 않았다. V57의 convex-self/virtual-ray 결과를 DIRECT로 승격하지 않았다. 모든 1,003장×3 미지원 cell을 원 분모 그대로 보존했다. 이것은 oracle 실패 성능 0점이 아니라 oracle 입력의 부재다.

V59의 별도 로컬 mesh fixture에서 native32점 표면 지원0/32였다는 기존 metadata는 경계 근거다. 그 fixture를 과거 원 render 전체의 인증으로 확대하지 않는다. 필요한 것은 원 source asset/import transform과 native3D order를 결합한 same-render 표면/depth/owner 증거이며, 최근접 표면 snap이나 GT-best permutation으로 대체하면 목표가 바뀐다.

ORIGINAL 통계는 asset 및 split별 평균·중앙값·P90/P95·최대 T/R, canonical near90와 symmetry-aware R>45를 별도 기록한다. asset 이름을 목재/정사각형으로 임의 매핑하지 않았다. 세 oracle의 correct→wrong/회복·damage는 비교 자체가 미지원이므로 계산하지 않았다. near90를 width/depth 또는 native correspondence의 확정 원인으로 부르지 않는다.

기존 real-CAL 의존성은 G0 metadata inventory를 그대로 보존했다. state classifier·feature statistics, residual covariance/reliability와 pose prior·bias, O/T empirical domains, H1의 transitive bank/cost 의존성이 있다. 이 실행은 해당 artifact/모델을 불러오거나 적합하지 않았다. raw RGB·annotation archive·mesh·실사/current/held case·모델/encoder/학습·PnP 변경은0이다.
