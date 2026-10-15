# 전역 AMG도 단독 스툴 후보를 확보하지 못했다

CAL46 자동 detector→box-SAM 진단에서 파란 스툴을 놓친 뒤, 같은 `eval_outside:1778651519235162880` 한 장에 class-agnostic MobileSAM AMG를 별도로 실행했다. 이 사례는 해당 누락을 본 **진단 후 선택**이며, 성능을 평가하기 위한 사전 선정 표본이 아니다. 원 관측·포즈·CAL 모델·기존 보고서는 바꾸지 않았다.

기존 로컬 vit_t checkpoint를 CPU 한 개로 사용했다. 전역 균일 16×16 grid, crop 0, 그 외 pinned vendor 기본값을 고정했다. detector box·수동 ROI·키포인트·S/U/O 좌표 prompt는 사용하지 않았다. 원영상 전체의 균일 point prompt는 특정 S/U 위치를 겨냥한 국소 질의와 구분한다.

실행은 44.292초에 오류 없이 끝났다. vendor의 기존 quality/stability/crop-edge 필터 및 box NMS를 지난 42개 mask를 모두 저장했다. 원래 256×3 raw multimask 슬롯 전체를 저장한 것은 아니다. encoder 1회·64 point 단위 decoder 4 batch는 고정 소스에서 예상한 호출 수이며 별도 계측값은 아니다.

| 결과 | 수량 | 의미 |
|---|---:|---|
| `CANDIDATE` | 40 | 기존 target mask와 겹치지 않는 후보. 다른 물체 또는 앞가림의 확정 아님 |
| `suspected_target` | 2 | 기존 target mask와 겹치는 후보. target 정체의 확정 아님 |
| 단독 스툴 mask | 확인 못함 | 전체 갤러리의 시각 확인이며 수동 pixel 정답/정확도 점수는 아님 |

[전체 결과와 균일 grid](../class_agnostic_AMG_CAL_case_v1/overview_candidates.png), [AMG0–11](../class_agnostic_AMG_CAL_case_v1/gallery_00.png), [AMG12–23](../class_agnostic_AMG_CAL_case_v1/gallery_01.png), [AMG24–35](../class_agnostic_AMG_CAL_case_v1/gallery_02.png), [AMG36–41](../class_agnostic_AMG_CAL_case_v1/gallery_03.png)을 모두 직접 확인했다.

AMG5는 스툴과 팔레트를 함께 담은 mask이고, AMG6는 주로 팔레트를 담은 mask다. 각각 기존 target mask와의 IoU는 0.69804와 0.97654다. 이 값은 저장 target과의 overlap이며 물체 정체·깊이·분할 정답에 대한 정확도가 아니다. 두 후보 모두 `suspected_target`으로 보존했다. 나머지는 주로 지면·건물·창문·배경 사람 등의 세그먼트다. 일부 `predicted_iou` 값은 1을 넘으며, uncalibrated model score 그대로 보존했다.

AMG5−AMG6은 향후 **파생 object proposal**로 조사할 수는 있어도 독립적인 다른 물체 mask나 물리적 앞가림의 증거가 아니다. 이번에는 차집합·target complement·grid 확대·threshold 변경·추가 추론을 하지 않았으며 consumer 공급도 0이다.

두 진단의 한계는 구분된다. generic detector→SAM은 필요한 box 자체를 놓쳤고, 이번 전역 AMG는 detector 없이 스툴이 포함된 영역까지 제안했으나 스툴을 단독 instance로 분리하지 못했다. 따라서 현재 두 방식의 결과를 바로 가림 mask로 연결하지 않는다.

[저장 결과 QA](ACTUAL_EXECUTION_QA_V1.json)는 **568개 검사 / 실패 0**이다. source 20·input 9·보호 파일 6개, 50개 봉인 asset, 전체 42개 mask의 file/pixel SHA·면적·bbox·전역 prompt·고정 필터·overlap 산술과 5개 그림 링크를 검산했다. QA 중 새 NN·RGB 재추론·GT/primary 참조·pose/noise fit은 0이다. [원 실행 봉인](../class_agnostic_AMG_CAL_case_v1/EXECUTION_SEAL.json)과 원 보고서·source·PLAN은 그대로 보존했다.
