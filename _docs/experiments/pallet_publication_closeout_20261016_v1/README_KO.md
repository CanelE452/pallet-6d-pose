# 2026-10-16 최근 연구 자료 공개 마무리

이번 대화에서 만든 코드·실험 결과·실제 이미지·갤러리와, 이전 작업 중 main에 없던 작은 근거를 모아 확인했다. 기존 원자료·실패 기록·논문·원본 로컬 수정은 덮어쓰지 않았다. 새 학습·seed·데이터·추론·PnP·정답 수정·운영 기본값 변경은 없다.

## 최신 결과와 실제 이미지

| 내용 | 자료 |
|---|---|
| RAW/N3 초기화와 최소 B/I 네 경로 | [결과](../pallet_minimal_bi_ablation_v1/RESULTS_KO.md) · [재현](../pallet_minimal_bi_ablation_v1/REPRODUCE.md) |
| guarded 후속 실험과 채택 권고 | [결과](../pallet_guarded_bi_final_v1/RESULTS_KO.md) · [최종 권고](../pallet_guarded_bi_final_v1/FINAL_RECOMMENDATION_KO.md) |
| N3만 / SubPix만 분리 | [결과](../pallet_single_component_bi_v1/RESULTS_KO.md) |
| 전체192장·별도CAL46·평균·분산·지시문 마무리 | [결과](../pallet_instruction_closeout_v1/RESULTS_KO.md) · [완료 점검](../pallet_instruction_closeout_v1/CLOSEOUT_KO.md) |
| PoseFix 차이와 아웃라이어 | [종합 보고서](../pallet_posefix_gap_audit_v1/REPORT_KO.md) · [공통 제외 분석](../pallet_posefix_gap_audit_v1/outliers/OUTLIER_RESULTS_KO.md) |
| 제외71장 실제 이미지 | [갤러리](../pallet_posefix_gap_audit_v1/excluded_gallery/index.html) |
| 제외71장·다섯 방법·전체 실제 과정 | [병렬 갤러리](../pallet_posefix_gap_audit_v1/parallel_gallery/index.html) |
| 이미지 순번별 위치·회전 | [다섯 방법 산점도](../pallet_posefix_gap_audit_v1/scatter/index.html) · [우리 e와 비교 방법](../pallet_posefix_gap_audit_v1/comparison_scatter/index.html) |
| x=위치, y=회전 | [대화형 산점도](../pallet_posefix_gap_audit_v1/joint_scatter/index.html) · [PNG](../pallet_posefix_gap_audit_v1/joint_scatter/position_rotation_e_vs_n0.png) |
| 사용자가 지목한54/208/260cm 세 사례 | [실제 이미지](../pallet_posefix_gap_audit_v1/pointed_cases/index.html) · [두 방법의 전체 과정](../pallet_posefix_gap_audit_v1/pointed_cases/process/index.html) |
| 011640 폭·깊이 대응 비교 | [물리 축 설명](../pallet_posefix_gap_audit_v1/axis_011640/CASE_REPORT_KO.md) |

HTML은 해당 폴더의 관련 파일과 함께 내려받아 브라우저로 연다. GitHub 파일 화면은 HTML 소스를 표시한다. 갤러리의 모든 상자는 저장된 예측이며 GT를 새로 만든 것이 아니다. 전체192장 분모와 모든 실패를 보존했다. 공통 제외121장은 사후 보조 분석이며 전체 평가를 대체하지 않는다.

## 이번에 보충한 누락 근거

- 기하학 선 탐색·렌더·진단의 기존 소스와 테스트12개를 원본 바이트로 추가했다. 기존 parent 모듈을 포함한 검사73개가 통과했다. 코드 동작을 새로 바꾸지 않았다.
- V31/V32 준비·검산·리뷰·예측 기록, 최초 렌더 이미지와 문서를 추가했다. 이 중 준비·설계 문서는 새 성능 실험의 성공을 뜻하지 않는다. 최초 렌더는 과거 기록이며 이미 공개된 최신 `display_v2`를 대체하지 않는다. [파일 목록](LEGACY_PUBLIC_ADDITIONS.json)
- [초기14손상 분석](../pallet_minimal_bi_ablation_v1/historical_forensics_pre_guarded/README_KO.md), [guarded 상세 재집계](../pallet_guarded_bi_final_v1/evidence/independent_details/README_KO.md), [N3/SubPix 분리 상세 재집계](../pallet_single_component_bi_v1/evidence/independent_details/README_KO.md), 브라우저 검증과 물리 축 진단 원본을 추가했다. 초기 분석의 제안은 guarded 실험 이전 기록이며 현재 권고와 구분한다.
- [옛 ResNet-18·세 기반 문서 정정과 실패 근거](legacy_corrections/README_KO.md)를 날짜가 있는 별도 스냅샷으로 보존했다. 당시 학습 종료·기능 실패·복구 진단 완료를 구분했다. 원본 사용자 파일6개는 수정하지 않았다.

압축 JSON은 압축 해제하면 원본 바이트와 SHA256이 복원된다. 기존 봉인은 바꾸지 않았으며 이번 공개 manifest에 원본 해시와 압축 파일 해시를 별도로 기록했다. 일부 옛 실행 코드·영수증에는 로컬 환경 경로가 있어 원본 그대로 보존했다. 토큰이나 인증정보를 추가하지 않았다.

## 이미 공개된 대형 원자료

[기본 원자료 Release](https://github.com/CanelE452/pallet-6d-pose/releases/tag/visibility-curriculum-snapshot-20261015-v1)와 [후속 원자료 Release](https://github.com/CanelE452/pallet-6d-pose/releases/tag/visibility-curriculum-followup-20261015-v11)에 기존 원자료와 코드 스냅샷이 있다. main에서 누락으로 보인 최근 파일29,495개 중29,493개는 이전의 바이트 검증된 Release 인벤토리에 동일 크기·수정 시각으로 존재했다. 이번에는 대형 파일을 모두 다시 해시하거나 내려받지 않았다. 나머지2개는 공개 payload 독립검사 코드여서 main에 추가했다. 현재 두 Release는 공개 상태이고 총85개 asset이 uploaded이며, 후속55개는 기존 검증 영수증과 서버 SHA256·크기가 모두 같았다.

[감사 요약](AUDIT_SUMMARY.json) · [Release 확인](LIVE_RELEASE_VERIFICATION.json) · [소스 목록](SOURCE_ADDITIONS.json) · [작은 상세 증거 목록](RECENT_EVIDENCE_ADDITIONS.json) · [검사73개](TEST_RESULTS.json)
