# V230: YCB SOURCE의 입력 q 관측과 기존 r 교사 권위를 분리한 점검

기존 YCB SOURCE 1,000 instance·700 frame의 **8,000개 logical auxiliary slot**을 모두 보존하고, 입력 query **q** 위치의 mask/depth 관측을 기존 surface reference **r** 기반 D/E/S/U 교사와 나란히 기록했습니다. 두 값은 같은 질문이 아닙니다. 이 결과는 **semantic classifier 혼동행렬이나 모델 성능 평가가 아니며**, 새 라벨·학습·admission·현재 실사 추론·PnP는 모두 0입니다. V229의 실사 실패·미채택 상태도 바뀌지 않습니다.

이번 실행은 V229 결과를 보고 만든 새 설계가 아니라, 그 전에 고정한 V228 protocol(`9f3d2657…`)의 관측 계획을 재개한 것입니다. 기존 `V228_PREP_ONLY` 기록과 downstream UNKNOWN을 남긴 과거 보고서는 역사적 원본으로 보존합니다.

## 원 8,000행을 유지한 교차표

기존 교사는 **D 1,181 / E 340 / S 1,917 / U 4,562**로 그대로입니다. 아래 열은 `floor(q+0.5)`로 정한 q 중심 pixel의 owner 관측입니다. OWN은 선택한 source instance, FOREIGN은 유일한 다른 instance, OVERLAP은 visible-mask owner가 둘 이상, UNASSIGNED는 owner가 없는 경우입니다. OOB도 삭제하지 않습니다.

| 기존 associated-r 상태 | 행 분모 | q 연속 OOB | q nearest OOB만 | 유일 OWN | 유일 FOREIGN | OVERLAP | UNASSIGNED |
|---|---:|---:|---:|---:|---:|---:|---:|
| D | 1181 | 0 | 0 | 972 | 13 | 16 | 180 |
| E | 340 | 0 | 0 | 1 | 338 | 0 | 1 |
| S | 1917 | 0 | 0 | 1275 | 255 | 86 | 301 |
| U | 4562 | 275 | 2 | 1382 | 588 | 358 | 1957 |

D 1,181행 중 q 중심은 OWN 972행, FOREIGN 13행, OVERLAP 16행, UNASSIGNED 180행입니다. E 340행에서는 FOREIGN 338행, OWN 1행, UNASSIGNED 1행입니다. 이는 **기존 r 상태와 별도 q 관측의 교차 빈도**이며 D/E 판정 accuracy·precision·recall로 부르지 않습니다. S의 q가 OWN인 1,275행과 U의 q가 OWN인 1,382행도 원 S/U로 남깁니다. q의 owner 관측으로 기존 상태를 자동 수리하지 않았습니다.

상태 enum은 연속 OOB를 먼저 판정합니다. 별도 validity flag의 연속 OOB 275행과 nearest-invalid 273행은 서로 배타적이지 않습니다. 표의 두 OOB 열과 이 flag 수를 혼동하거나 더하지 않습니다.

[전체 교차표 CSV](results/OLD_STATE_BY_Q_OWNER.csv) · [SUMMARY](results/SUMMARY.json) · [완료·원문 SHA bindings](results/COMPLETE.json)

## 거리 구간은 구성한 jitter의 진단

입력 q는 고정 surface reference r에 반경 4px 이내 jitter를 더한 SOURCE emulation입니다. 이것을 실제 initializer 오차 분포나 현재 실사 reference 오차라고 해석하지 않습니다. 사전 고정 구간의 행 수는 `[0,1]` 490, `(1,2]` 1,467, `(2,3]` 2,515, `(3,4]` 3,528이며 합계는 8,000입니다. 거리 ≤3px 부분집합은 4,472행입니다.

| 거리 ≤3px의 기존 상태 | 해당 상태 분모 | q OWN | q FOREIGN | OVERLAP | UNASSIGNED |
|---|---:|---:|---:|---:|---:|
| D | 680 | 563 | 12 | 10 | 95 |
| E | 186 | 0 | 185 | 0 | 1 |

따라서 ≤3px D 680행 중 OWN은 563행, 다른 owner/overlap/unassigned 관측은 117행입니다. ≤3px E 186행은 FOREIGN 185행과 UNASSIGNED 1행입니다. 가깝다는 사실이 같은 surface/semantic authority를 자동 보장하지는 않지만, 이 차이만으로 원 교사가 손상되었거나 라벨이 틀렸다고 판정하지도 않습니다.

## 무엇을 관측했고 무엇을 바꾸지 않았는가

기존 r 교사는 고정 surface point의 ray와 r 주변 depth/owner 근거에서 계산됩니다. 이번에는 동일 source의 **q 중심 및 3×3** 관측을 추가했습니다. Mask-only OWN/FOREIGN과 양성 depth의 교집합 count는 별도이며, depth나 mask가 없다고 곧바로 E가 되지 않습니다. q에서 다른 표면이 보이거나 아무 owner가 없어도 associated-r 상태의 의미가 자동 무효가 되는 것은 아닙니다.

모든 slot의 `native_mapping=UNKNOWN`을 유지합니다. YCB object의 logical auxiliary slot을 pallet native point, 인간이 확인한 intended part, 또는 물리적 corner gold로 바꾸어 부르지 않습니다. GT reference·mask·depth는 teacher 측 근거이며 배포 입력에 추가하지 않았습니다. 원 q/L·원 상태·이유를 유지한 채 `new_native_visibility_target=null`, `training_admission=false`로 저장했습니다.

코드의 최초 목록에는 depth/mask와 함께 raw scene-GT/camera JSON binding 14개가 있었지만, 실행 전 v2에서 payload 대상을 **선언한 PNG 4,700개로만** 분리했습니다. 실제 실행 기록은 이 4,700개 depth/mask PNG의 SHA 확인·decode와 원 8,000개 reference evidence/state/reason exact를 보고합니다. Raw scene-GT/camera JSON 14개는 inert binding으로 남았고 그 payload의 추가 stat/hash/read/값 parse는 0입니다. 관측 builder의 RGB decode·새 ray/CAD·모델 forward·fit도 0입니다. 후속 RGB 사례 gallery는 별도 표시 자료이며 이 표를 만든 입력이나 새 라벨이 아닙니다.

## 기존 학습 소비 경로에서 확인한 사실

이번 bounded 코드·실행 metadata 추적은 이전 UNKNOWN을 좁게 해결했습니다. V109 corpus seal `934a9a3a…`를 V110/C10 design이 동일 SHA로 pin하고, 실제 source-fit entry·RUN_STARTED·MODEL_FREEZE·FIT_COMPLETE가 trainer `3baff03b…`를 연결합니다. C10은 **D0/E1, S·U−1**을 검사하고 known D/E·원 TRAIN·available만 단일 external-logit **BCE 직접 감독**에 사용했습니다. S/U를 E로 합치는 경로는 이 코드에 없습니다. 공통 matched hinge는 old/synthetic triples에 적용되며 자연 YCB 자체가 matched triple이라는 뜻은 아닙니다.

V111/C11 trainer `f34ce3b4…`도 같은 C10 label join·sampling·loss를 재사용합니다. C10은 frozen ResNet18 영상 descriptor, C11은 frozen DINO 영상 descriptor와 initial-q 공간 특징을 받았으며 teacher/reference를 forward에 넣는 계약이 아닙니다. 그러므로 YCB SOURCE를 실제 head loss에 쓰는 것이 처음이거나, 기존에는 간접 AUX에만 저장했다고 말하지 않습니다. 이 연결은 코드·actual-entry·완료 metadata의 증거이며 원 GT 배열, 실제 gradient, 과거 성능을 독립 replay했다는 주장은 아닙니다. [소비 경로 근거 JSON](CONSUMER_LOSS_AUTHORITY.json)

## 저장 자료의 재현 범위와 현재 결론

8,000행 원장은 `results/QUERY_OBSERVATIONS8000.jsonl` 또는 같은 원문을 보존한 `.jsonl.gz`입니다. 원문 SHA256은 `c20440d6add5969a5d2655aeba5a0abe8009737ade2171049fbf217491c4496f`, 원문 크기는 **22,637,109B**입니다. 완료 영수증 SHA256은 `4c3ca28102149d773ea870bd9c6f118078d62cf6fa3477d1ed3e6447b8d346d3`입니다.

[표준 Python 검산기](VERIFY_SAVED_QUERY_OBSERVATIONS.py)는 package 상대경로만 사용해 gzip 해제 원문 SHA/size, 8,000행 identity, 네 거리 구간과 ≤3 subset, nearest-floor 계산, 저장 owner bits/count/depth·3×3 evidence, SUMMARY·CSV 교차표를 다시 셉니다. 원 GT/NPZ/PNG를 열지 않으므로 원 teacher의 권위나 mask 정확성을 재인증하는 절차가 아닙니다. 이 보고서는 봉인 SUMMARY/CSV의 24개 cell과 구간 합계를 대조했으며, 새 모델 forward나 학습을 하지 않았습니다.

현재 확인한 것은 **associated-r 상태와 literal-q 관측이 서로 다른 축**이며, 그 associated-r D/E가 기존 C10/C11에서 실제 감독으로 소비된 계보입니다. 라벨 오염, V229 실사 실패의 전이 원인, 전체 GT 오류를 확정하지 않습니다. 새 라벨 채택·추가 학습·모델 선택·문턱 조정은 하지 않았고, 실제 모델 성능 개선을 새로 측정한 결과도 아닙니다.

## 원 q/r crop 사례 6개

[고정 갤러리](figures/PORTABLE_GALLERY_KO.html) · [RGB 읽기 전 고정 ID](figures/FIXED6_CASE_IDS_BEFORE_RGB.json) · [선정표](figures/CASES.csv) · [원 q/L·teacher r·저장3×3 근거](figures/CASE_DATA.json)

서로 다른6프레임에서 사전6bucket을 모두 채웠습니다. D×q-own/foreign/unassigned와 E×q-foreign/unassigned는 기존 구성 jitter≤3px에서 선정됐습니다. E×q-own은 해당≤3px 사례가 없어서 사전 규칙대로3.7354px 사례를 사용했으며, 이를≤3px 사례로 해석하지 않습니다. 원 q/L crop에 q(청록+)와 teacher r(자홍×)를 함께 표시했고, r로 다시 자르거나 좌표를 보정하지 않았습니다. 저장 mask/depth는 교사 측 진단 그림이며 모델 입력이 아닙니다.

실제 원본 PNG/NPZ/라벨은 변경하지 않았습니다. 게시용8000행 JSONL은 lossless gzip으로 담았으며 검산기가 압축 해제 후 원 JSONL SHA와23MB 원문 크기를 확인합니다. 원 depth/mask4700개 전체와 RGB700장 전체는 복사하지 않았습니다. 새 분류기나 새 성능 개선 결과는 없습니다.
