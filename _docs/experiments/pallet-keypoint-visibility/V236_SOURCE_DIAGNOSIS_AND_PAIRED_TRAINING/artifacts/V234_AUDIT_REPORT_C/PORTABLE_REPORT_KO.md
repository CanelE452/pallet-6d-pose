# V234 SOURCE 라벨·입력 주소·분포·생성 계보 감사

저장 자료에서 YCB DEV의 E57을 두 모델 모두 D로 판정한 사실을 확인했다. 같은 object×aux 조합의 FIT에는 E가 없고 D가 있으며, DEV E57 중 56은 q 위치에서도 foreign이다. 따라서 좌표 차이 하나로 이 실패를 설명할 수 없다. 이 감사는 원인을 확정하거나 라벨을 바꾸지 않으며, 새 실사 성능 개선·모델 채택을 뜻하지 않는다.

## 분모와 관측 결과

YCB의 교사는 associated-r surface 상태이고 SYN의 교사는 own-image intended-native witness0 상태다. 두 referent의 혼동행렬을 하나의 과업으로 합치지 않는다. YCB의 aux slot은 physical native part ID가 아니다.

| YCB 원 상태 | FIT | DEV | 전체 |
|---|---:|---:|---:|
| D | 841 | 340 | 1,181 |
| E | 283 | 57 | 340 |
| S | 1,412 | 505 | 1,917 |
| U | 3,064 | 1,498 | 4,562 |
| 전체 | 5,600 | 2,400 | 8,000 |

D/E score 분모는 FIT 1,124, DEV 397이다. S/U 6,479는 이진 score 분모와 구분한다. 원 8,000행 q/L/state 정합과 known 1,521행의 reference가 원 crop 안에 있다는 것은 [Root 저장 진단](artifacts/V234_ROOT_LABEL_DIAG/SUMMARY.json)의 검증 범위다. 본 보고서는 [1,521행 CSV](artifacts/V234_ROOT_LABEL_DIAG/YCB_KNOWN1521_LABEL_QUERY_MODEL_DIAGNOSTIC.csv)에서 역할·상태·고정 logit 부호와 아래 YCB CM을 독립 재집계했다.

| SOURCE DEV / 두 arm 각각 동일 | CM: D행→D/E/U; E행→D/E/U | D recall | E recall |
|---|---|---:|---:|
| SYN main, N=1,344 | [1273,15,0]; [5,51,0] | 98.84% | 91.07% |
| YCB AUX, N=397 | [333,7,0]; [57,0,0] | 97.94% | 0.00% |

YCB FIT CM도 두 arm 모두 [840,1,0]; [0,283,0]이다. SYN 수치와 사전 gate는 [A의 봉인 SOURCE 요약 인용](artifacts/V234_MODEL_INPUT_AUDIT_A/MODEL_INPUT_AUDIT.json)이며, 이 통합 단계에서 SYN 원 score를 다시 열지 않았다. macro DEV balanced-recall gain은 0pp, 사전 수치 gate는 FAIL, current admission은 false다. 같은 집계 CM이 모든 개별 예측까지 동일함을 뜻하지는 않는다.

## DEV E57의 지원 분포와 q 위치

| DEV scene | object | aux | DEV E | 같은 object×aux FIT D | 같은 object×aux FIT E |
|---|---:|---:|---:|---:|---:|
| 000006 | 7 | 3 | 29 | 79 | 0 |
| 000006 | 7 | 7 | 24 | 20 | 0 |
| 000007 | 2 | 2 | 4 | 46 | 0 |

이 표는 [object×aux 전체 CSV](artifacts/V234_YCB_SHIFT_AUDIT_B/OBJECT_AUX_LABEL_COVERAGE.csv)와 1,521행 표를 교차 검산했다. DEV E57 모두 같은 조합의 known FIT 지원은 있지만 그 지원은 D뿐이다. object만 보면 FIT E 지원이 57/57에 있으므로 “해당 object의 E를 전혀 학습하지 않았다”는 표현도 맞지 않는다. 53/57은 scene6/object7에 집중된다. 57행은 서로 다른 frame/instance이지만 독립 물리 사건 57개로 인증한 것은 아니다. 같은 object×aux라도 다른 frame·instance의 입력이므로 동일 tensor에 반대 라벨이 붙었다는 증거는 아니다. [B 분포 감사](artifacts/V234_YCB_SHIFT_AUDIT_B/PORTABLE_REPORT_KO.md)

| DEV E 거리 진단 | N | q UNIQUE_FOREIGN | q UNIQUE_OWN |
|---|---:|---:|---:|
| 전체 | 57 | 56 | 1 |
| q-r ≤3px | 30 | 30 | 0 |
| q-r >3px | 27 | 26 | 1 |

여기서 q-r은 구성된 public emulation jitter이며 실제 detector 오차가 아니다. ≤3px E30도 q에서 전부 foreign이므로 “교사는 r을 보고 모델은 q를 보아서 생긴 좌표 차이”만으로 E recall 0을 설명할 수 없다. 다만 nearest q owner는 교사 측 위치 진단이다. 모델의 semantic confusion, 새 D/E 정답, 라벨 오염 판정이나 전이 실패의 인과 증거로 바꾸지 않는다.

## 저장 교사 근거를 어디까지 재검산했는가

[Root 코드](artifacts/V234_ROOT_LABEL_DIAG/audit_source_labels.py)는 원 8,000행과 기존 q/L/state를 맞추고, 저장된 mask/depth/ray 요약으로 D 1,181과 E 340을 점검했다. D는 centre unique-own, 3×3의 own/depth 지원, 저장 centre ray Z와 depth의 15mm 일치를 확인한다. E는 3×3 foreign 단일 owner 일관성·own-visible 부재·양수 depth 최소7·amodal own-full 지원 등을 확인한다.

이것은 저장 export의 재검산이다. 이 통합 보고서는 원 GT·mask·depth·geometry를 다시 decode하거나 ray를 재실행하지 않았다. 특히 E의 referenceZ보다 15mm 앞이라는 부등식은 봉인 V228에서 점검된 것으로 인용되지만 현재 export에 referenceZ가 없어 여기서 독립 재계산할 수 없다. 이를 물리 native gold 재인증이나 모든 E의 새 기하 증명으로 확대하지 않는다. [프로토콜과 한계](artifacts/V234_ROOT_LABEL_DIAG/PROTOCOL_BEFORE_SOURCE_DIAG.json)

## 입력·loss 경로에서 확인한 것과 미확인인 것

[A 코드 감사](artifacts/V234_MODEL_INPUT_AUDIT_A/KO.md)에서 SYN family0/output0, YCB family1/output1, D=0/E=1, logit≥0→E와 FIT·available·known loss mask가 일치한다. full/centre row 복원이나 label index 반전의 뚜렷한 코드 오류는 발견하지 않았다. 이는 모든 runtime 오류가 없다는 증명은 아니다. A는 해당 구현 작성자이며, 별도 B의 봉인 검산을 인용한 코드 readback이다.

YCB condition32는 0이므로 explicit native/aux ID·GT instance·r·r-q를 전달하지 않는다. 그러나 q 중심 crop 자체가 암묵적 주소를 주므로, 이 필드 부재만으로 문제의 식별 불가능성이나 실패 원인을 확정할 수 없다. 정확히 같은 전체 입력/descriptor에 반대 라벨이 붙은 충돌도 입증하지 않았다. full96 CLS는 원 frame 전체가 아니라 crop 전체의 문맥이며, fixed-centre Q token도 attention으로 crop의 정보를 포함할 수 있어 literal 네 RGB pixel의 가시성 시험과 같지 않다. 두 출력은 referent를 구분하지만 backbone/hidden layer는 공유한다. 공유 gradient가 실패를 일으켰다는 인과 결론은 없다.

## SYN 외부 사진 계보와 기존 counterfactual 이력

처음의 [폐쇄 metadata 경로 감사](artifacts/V234_SOURCE_PROVENANCE_C/ROUTE.json)는 SYN background/donor 계보를 UNKNOWN으로 남겼다. 이후 승인된 TRAIN GT110 파일 감사에서도 실제 asset lineage field는 없었다. `nearest_background` 3,520개는 pixel owner 진단이며 배경 asset 기록이 아니다. 원 keyword 집계는 보존하고 이 의미를 [분류 addendum](artifacts/V234_SELECTED_GT110_LINEAGE_C/ASSET_LINEAGE_CLASSIFICATION_ADDENDUM.json)에서 바로잡았다. [110 parent 표](artifacts/V234_SELECTED_GT110_LINEAGE_C/ALL110_PARENT_ASSET_LINEAGE.csv)

더 구체적인 [pinned renderer CODE 감사](artifacts/V234_RENDERER_LINEAGE_C/REVIEW.json)는 RGB가 seed·좌표 sine·고정 palette·geometry normal/light로 생성되고, 검토한 RGB writer에 외부 YCB 사진 background/donor를 읽어 붙이는 입력 경로가 없음을 확인했다. `BACKGROUND_LIKE`도 사진이 아닌 계산된 배경색이다. 이 근거는 단순한 GT field 부재보다 강하지만 imported helper 전체, geometry template의 원 제작 계보, selected parent별 실행 영수증과 실제 RGB byte를 전부 재검증한 것은 아니다. fresh-blind·전체 데이터 독립성 인증이나 current admission 변경으로 확대하지 않는다.

[기존 CF metadata 감사](artifacts/V234_PRIOR_CF_AUDIT_A/AUDIT.json)에는 원 public initializer 계보의 BEFORE/COVER/ADJACENT_SHAM 각1,272행, 같은 q/L/native/q8/condition을 유지한 1,272 triplet이 이미 있다. 원 parent46frame·선언 component1의 반복이며 자연 YCB associated-r 계보와는 구별된다. stripped input metadata만으로 COVER를 native E, SHAM을 native D로 인증할 수 없고, row extractor는 CF generator가 아니다. 같은-query COVER/SHAM 자체를 새 아이디어나 첫 시도로 표현하지 않는다.

## 이번 감사와 후속 작업의 경계

이번 V234에서는 새 학습·모델 추론·문턱 변경·PnP·target 변경을 하지 않았다. 통합 작성 중 원 RGB/GT/NPZ/weights 및 current/held 자료를 추가로 열지 않았다. 앞선 GT110 ancestry 읽기는 별도 승인·봉인된 실제 metadata 감사이며, 이 보고서에서는 그 저장 영수증만 인용했다. 과거 보호 metadata 노출 이력까지 없었다는 포괄 주장은 하지 않는다.

Root가 별도로 정한 V235 data-only pair 시험은 이 진단과 분리된 후속 작업이다. 이 보고서 시점에는 그 시험의 학습·성능 결과가 없으며, 본 보고서는 새 모델·데이터·architecture를 선택하거나 추천하지 않는다. 합성 SOURCE의 높은 수치가 실사 문제 해결을 뜻하지 않으며, 기존 V229 실사 실패 상태를 바꾸지 않는다. 원 코드·표·실패/교정 기록은 보존한다.
