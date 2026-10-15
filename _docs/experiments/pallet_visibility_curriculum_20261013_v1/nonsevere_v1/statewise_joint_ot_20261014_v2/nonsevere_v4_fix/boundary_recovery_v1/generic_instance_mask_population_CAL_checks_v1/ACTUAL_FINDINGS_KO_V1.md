# CAL46 자동 instance 후보 실제 확인

같은 로컬 COCO80 YOLO26n → box-prompt MobileSAM 설정으로 CAL46 전부를 한 번 처리했다. 49.623초, 실행 오류 0이며 빈 결과도 보존했다. 원래 봉인한 공급기·PLAN·결과 보고서는 수정하지 않았다.

| 범위 | 영상 | 검출 | 후보 있음 / 없음 |
|---|---:|---:|---:|
| CAD | 18 | 2 | 2 / 16 |
| night08 | 3 | 0 | 0 / 3 |
| outside | 5 | 28 | 5 / 0 |
| wood | 20 | 37 | 20 / 0 |
| 전체 CAL | 46 | 67 | 27 / 19 |

전체 detector 호출 46회, SAM encoder 27회, box decoder 67회와 SAM 대안 201개를 저장했다. 선택 마스크 중 `suspected_target`은 22개, `CANDIDATE`는 45개다. 뒤의 45개는 target mask와 겹치지 않았다는 뜻이며, 다른 물체의 정체나 앞뒤 관계를 증명하지 않는다. 팔레트 자체를 `bench`로 분류한 검출이 있고 같은 팔레트의 중복 검출도 있었다.

## 핵심 누락

[`eval_outside:1778651519235162880` 실제 그림](../generic_instance_mask_candidates_CAL46_v1/case21_candidates.png)에는 팔레트 위 파란 스툴이 명백히 보인다. 그러나 검출기는 배경의 사람 8개와 우산 1개만 반환했고 스툴 box를 만들지 않았다. SAM은 받은 box만 처리하므로 스툴 마스크를 생성할 기회가 없었다. 다음 outside 두 장에서도 파란 스툴은 후보에 없었다. night08 세 장의 팔레트 앞/위 원뿔도 검출 0이었다([예시](../generic_instance_mask_candidates_CAL46_v1/case18_candidates.png)).

따라서 이 공급기를 현재 가림 mask로 바로 연결할 근거는 없다. 빈 결과는 가림 없음의 증거가 아니며, 배경 후보가 많아도 필요한 가림 물체의 재현율을 보장하지 않는다. COCO class를 모두 non-target으로 취급하면 팔레트 자체를 제외할 위험도 실제로 확인됐다.

전역 class-agnostic SAM grid 진단은 detector box 누락과 SAM 분할 능력을 분리할 다음 검사 후보다. 아직 실행하지 않았고, 그 마스크도 instance 정체·target 여부·앞뒤 관계의 정답으로 간주하지 않는다.

## 검산 범위

[저장 결과 보고서](../generic_instance_mask_candidates_CAL46_v1/REPORT_KO.md), [실행 봉인](../generic_instance_mask_candidates_CAL46_v1/EXECUTION_SEAL.json), [QA 1849개 / 실패 0](ACTUAL_EXECUTION_QA_V1.json)을 보존했다. QA는 164개 봉인 asset, source 32 / input 101 / 보호 파일 6개, 전체 마스크 pixel SHA·quality 최고값·target overlap 산술, 기존 CAL3와 동일한 세 영상의 저장 예측 일치, 46개 그림 링크를 확인했다. 8개 contact sheet에서 46장 전체 그림을 직접 확인했다. 정답 마스크를 새로 만들거나 가림 정확도 점수를 계산한 검사는 아니다.

키포인트·S/U/O 좌표 prompt, GT/primary 참조, pose/noise fit, 기존 pipeline 공급은 모두 0이다. 새 runner는 Ultralytics 설정 디렉터리를 output 안에 먼저 만들어 CAL3에서 관찰된 `/tmp` 설정 fallback을 피했다. 모델 다운로드나 기존 runtime source 변경은 요청하지 않았다.
