# V217 · 후보 mask/depth/query 진단 V2 준비

V1 원문은 그대로 보존하고, root가 추가 요청한 **고정12 derived context PNG + 후보15 full/visible overlay**만 V2에 확장했습니다. 원 mask/depth/query 수치 함수는 AST 그대로입니다. 새 write는 이 V2 준비 폴더뿐이며 V216 및 기존 준비·GT 산출물은 수정하지 않았습니다.

12프레임·고유96 native/U11/FIT64DEV32를 유지합니다. obj1 후보15×native8=120행을 모두 기록하며 selected instance는 UNKNOWN입니다. 기존42 known-SHA PNG(full15/visible15/depth12)와 원 V215 derived context PNG12개만 향후 별도 root entry로 읽습니다. 원 JPEG/RGB NPY/CAD/원NPZ/weights는 열지 않습니다. [정확12 context 바인딩](EXACT12_ORIGINAL_DERIVED_CONTEXT_PNG_BINDINGS.json)은 같은8슬롯이 같은 원 fullcontext를 쓰는지 메타데이터만 검증했습니다.

## 질문별 수치

full/visible 면적·부분집합·비율은 **실제 in-frame mask** 진단입니다. GT_info의 global visib_fract와 다른 측정이며 native 가림 라벨이 아닙니다. 원 GT bbox와 count/fraction은 문맥 열에 그대로 남기고 OOB·깊이 결측을 분리합니다. `[-1,-1,-1,-1]` 등 음수 폭/높이 bbox sentinel은 기하 support UNKNOWN이며 pxvis가 pxvalid보다 커도 강제로 고치거나 거부하지 않습니다. mask 음성/다른 물체@q/쿼리 OFF/낮은 global fraction을 E로 만들지 않습니다.

원 q의 `floor(q+0.5)` 최근접 픽셀과 잘린 3×3 창만 계산합니다. RAW96 bilinear sampler나 원 ROI support를 재현한 값이 아니며, A/C의 원 ROI와 rootweak 상태는 역사적 옆 열로만 보존합니다. 원 U는 mask가 양성이어도 U입니다. metric depth는 실제 selected camera의 유한 양수 depth_scale이 있을 때만 계산하고 없으면 UNKNOWN입니다.

full/visible/depth 차원이 다르면 수치 resize 없이 해당 후보와8행을 실패/UNKNOWN으로 남깁니다. context 차원만 불일치하면 overlay를 만들지 않고 독립적으로 유효한 mask/depth 수치는 유지합니다. 원12 context는 byteexact 복사하며, 후보 panel은 고정 alpha96/255의 표시용 mask tint와 원q/native/U를 표시합니다. 영상 입력·좌표·라벨을 바꾸지 않습니다.

모든 신규 native D/E/SELF label·학습 admission은 UNKNOWN/false입니다. 선택인스턴스 자동 GT-best/첫후보/최근접 할당·nominal projection·PnP·fit·모델·보호점수 접근은 없습니다. 현재 실제 source payload stat/hash/read/decode는 0입니다.

[프로토콜](PROTOCOL_BEFORE_MASK_DEPTH_PIXEL_READS.json) · [false entry](ROOT_ACTUAL_MASK_DEPTH_ENTRY_TEMPLATE_APPROVED_FALSE.json) · [20개 기능 검사](INVENTED_FUNCTIONAL_CHECKS_BOUNDARIES_FINAL.json) · [V1 보존·변경 근거](V1_PRESERVATION_AND_CONTEXT_EXTENSION.json)

실제 GT metadata reader는 root가 이미 실행한 COMPLETE를 미래 entry에서 바인딩합니다. 이 준비 중에는 해당 원 GT나 새 PNG를 열지 않았으며, 새 root pixel scope/entry가 오기 전 실제 실행하지 않습니다.
