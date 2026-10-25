# V241 검토 문서와 실제 V242 상태 대조

사용자 제공 검토 문서: PALLET_VISIBILITY_REVIEW_AND_NEXT_STEPS_V241_KO.md, SHA256 `95ae15b142756c599f9ac47b5e173472309f345e5f9769c59c2046b9207dd66a`, 59,166 bytes. 문서의 분석·제안과 사용자의 직접 지시를 구분한다. 문서 안 외부 링크 및 과거 current 수치는 새 모델/데이터/문턱 선택에 사용하지 않았다.

검토 문서는 main `8c2ae5c47a831339f626e5d502ef9092a9a6e474`의 V241 준비 상태를 기준으로 한다. 그 뒤 V242에서는 SOURCE FIT D 5,406개 부모를 모두 시도해 5,367개의 native-scene COVER/ADJACENT_SHAM 쌍을 생성했다. 미생성 39개는 UNKNOWN으로 유지하고 원 natural D를 보존했다. 독립 검산은 전체 부모의 원 BEFORE SHA, 실제 생성된 모든 쌍의 native 합성·q8 crop·context·teacher footprint 조건을 확인했다. 고정된 네 사례도 Root가 열어 보았다. 이는 데이터 검증 결과이며 새 모델 성능이 아니다.

## 반영할 핵심 진단

1. V240 TREAT 내부의 base와 residual을 분리한다. 별도로 학습한 CONTROL은 TREAT.base가 아니다. 기존 logit0/logit1은 두 family의 TOTAL binary logits이며 두 class logits가 아니다. 저장된 SOURCE 특징과 동결 head만 CPU에서 읽어 base/residual/total을 구하고 기존 TOTAL CSV의 재현을 먼저 확인한다.
2. V235의 최종 head가 자신의 학습 COVER/SHAM 쌍을 구분하는지 확인한다. 기존 CSV는 원 natural 7,617행만 포함하며 CF 특징은 학습 중 메모리에만 존재했다. 정확한 구 CF crop과 같은 frozen encoder로 특징을 다시 계산해야 한다. 이를 새 학습 또는 V242 입력 평가라고 부르지 않는다. SHAM의 D 유지, COVER가 SHAM보다 E 방향으로 이동하는지, 최종 문턱을 넘는지를 따로 보고한다.
3. q8+context 자체는 V240에서 이미 시험했고 SOURCE gate에 실패했다. 새 제안으로 반복하지 않는다. 동일 native 장면에서 CF q8+context를 함께 재구성한 V242는 그 입력 일관성 문제를 다루는 제한된 데이터 비교다. 두 변경의 상호작용이나 실사 전이 개선을 사전에 입증한 것이 아니다.

## 라벨·해석 경계

COVER/SHAM은 supervision reference r의 가림 조건이다. 최초 입력 q/L/native 및 UNKNOWN을 보존하고 r를 모델에 넣지 않는다. r와 q의 footprint 차이는 별도 진단으로 보존하며 q의 문자 그대로 한 픽셀 presence 정답으로 승격하지 않는다. 2D alpha proof도 3D/native 물리 정답이 아니다. YCB-V는 공개 보조 물체이며 pallet native 라벨은 UNKNOWN이다.

V240 SOURCE YCB DEV E 1/57은 전이 문제가 남아 있음을 뜻한다. 동일 조합의 natural FIT E가 없다는 관측만으로 원인을 확정하지 않는다. V235 CF는 그 조합에도 실제 draw되었으므로 동작 확인이 필요하다. 반복 노출된 SOURCE DEV는 새로운 blind test가 아니다.

## 현재 실행 경계

V242 실제 데이터·독립 QA·prefit 시각화는 완료했다. V242 CF 특징 생산기와 V243 trainer는 CODE/가상 fixture 검토만 끝났으며 실제 새 특징과 optimizer update는 0이다. 이 두 행동 진단 뒤 고정 SOURCE 실험의 타당성을 판단한다. 현재223 및 held 데이터는 fitting/정규화/문턱/모델/데이터/구조 선택에 사용하지 않고 새로 읽지 않는다. PnP를 변경하지 않는다.

사용자의 공개 실사 학습 허용 및 계속 main에 시도·결과를 올리라는 직접 지시는 계속 적용된다. 첨부 검토의 재승인 제안 자체를 새 사용자 금지 지시로 취급하지 않는다. main 공개는 모델 성공과 구분한다. 전체 목표는 active/unsolved다.
