V248 진전은 최소 handoff와 exact actual COMPLETE35c169e9…/trainer54da2204…로 확인했습니다. main c0e8c63db3ad5cecd6d57ee620ca9ad39dc35d71, 새 SOURCE update는 양팔1,000씩 총2,000이며 finalfreeze-before-DEV가 기록되어 있습니다. 실제 QA/수치 재집계나 model 재실행은 하지 않았습니다.

| 입력·교사 경로 | 기존 CODE로 확인한 범위 | 권한 한계 |
|---|---|---|
| 원 native/availability/relative condition32 | V237+/V239/V248 main에 이미 있음 | index는 target instance/해부학 정답이 아님; YCB condition은0 |
| q8 local8+q8minmax context224+관계 read | 원 RGB/원q/L, input-only support로 구현됨 | 문맥 자체가 새로운 미시도 해법은 아님 |
| V153 T/Q/F/C/P와 cosine | T는 같은 이미지의 initializer-q8 contextCLS; Q는 그 context의 원q bilinear read | 정답 object template/reference RGB나 GTowner crop이 아님 |
| V155 role D/E+context, V165/166 intended-part CF | 기존 봉인 CODE provenance가 직접 role BCE와 part weak CF를 확인 | 이 조합도 미사용으로 포장하면 안 됨; 과거 실제 결과/행별 update는 재인증하지 않음 |
| V163 owner/part/nominal reference | intended instance·part·q관계를 분리하는 SOURCE witness CODE | renderer/join은 새 native/quality target=-1·trainingfalse; 배포 owner/part branch 학습 근거 아님 |
| V248 public pixel AUX | 공유 visual stem의 별도 scalar 경로가 구현·실행됨 | conditional pixel을 native/instance 정답으로 승격하지 않음 |

V111은 cosine(T,C), cosine(T,F), cosine(Q,C), cosine(Q,P)도 이미 입력에 넣습니다. 이는 same-image 입력 벡터 사이의 특징이며, teacher object(+logicalslot)별 SOURCEFIT-D prototype을 평균하고 correct/wrong reference cosine/AUROC를 비교하는 집계와 다릅니다. 후자의 실행 근거는 이번 제한된 CODE/provenance 집합에서 찾지 못했습니다. 전체 과거 미시도·신규성 또는 feature로 식별 불가능하다는 단정은 하지 않습니다.

새 identity 진단에 연결할 기존 경로는 input-only GROUPS_INPUT_ONLY, ALL15040_GROUP_JOIN, canonical feature COMPLETE/samples/geometry입니다. 정확 path/SHA와 schema를 JSON에 보존했고 본문 배열/행은 열지 않았습니다. teacher object/logicalaux와 native anatomy/physical instance는 서로 다릅니다. 이후 별도 Root scope 없이는 teacher identity를 forward input으로 넣거나 SOURCEDEV로 prototype/normalizer를 맞추지 않습니다.

V163 ROOT_RESULTS/ROOT_PHASE/actual QA/joins/visual metadata, 모든 current·held score와 이미지/GT/NPZ/PT를 읽지 않았습니다. 새 학습·sweep·모형 선택·테스트·Git 동작도0입니다.
