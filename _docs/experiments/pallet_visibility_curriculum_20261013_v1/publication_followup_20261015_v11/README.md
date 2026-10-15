# 이미지·정보·원시 자료 공개 완료

이번 B/I 비교와 V33·이전 후속 실험 자료를 원격 **main**과 GitHub Release에 올렸다. 기존 실험 결과와 사용자 수정 파일 6개는 그대로 두었다.

- [전체 결과 보고서](REPORT_KO.md)
- [양쪽 B/I 보정: 10사례·80개 단계별 그림](CASE_GALLERY_KO_V1.md)
- [전체 다운로드·복원 안내](repro_inventory_DOWNLOADS_KO_v1.md)
- [추가 원시 자료 Release](https://github.com/CanelE452/pallet-6d-pose/releases/tag/visibility-curriculum-followup-20261015-v11)

## 저장한 자료

main의 첫 공개 묶음은 509경로, 이미지 132개, HTML 12개와 코드·점수·보고서·입력·작은 관측/예측·마스크다. [파일 목록](CLOSED_FILES_TO_PUBLISH_V1.txt)과 [SHA manifest](CLOSED_PAYLOAD_MANIFEST_V1.json)를 함께 제공한다. 원래 YOLO→N3→SubPix 과정, 실제 B/I 보정과 최종 PnP, 191cm 사례, V33 확대 그림과 전체 평가를 포함한다. GitHub에서 그림은 Markdown 갤러리로 바로 볼 수 있고 HTML은 clone/다운로드 후 브라우저에서 실행할 수 있다.

큰 관측·추론 원장과 중간 기록은 원본 12,107경로·23,479,722,138바이트를 보존한 추가 아카이브에 있다. 37분할의 합은 19,525,309,856바이트이며, 분할과 목록·복원 안내를 합친 Release 자산은 55개다. 기존 snapshot과 합치면 검산 범위 38,450경로를 복원할 수 있다. [전체 원시 자료 안내](repro_inventory_DOWNLOADS_KO_v1.md)에는 번호 순서 연결, SHA 확인, 별도 디렉터리 복원 명령이 있다.

전체 아카이브 멤버의 원본 SHA/동일-content hardlink를 스트리밍 검산했고 실패는 0이다. 업로드한 55개 자산도 GitHub 서버 SHA256·크기·완료 상태가 모두 로컬 목록과 같다. 공개 Release 페이지와 작은 다운로드 파일도 인증 없이 열리는 것을 확인했다.

## 결과의 범위

같은 사후 선택 10장에서 공통 B/I 보정 후 성공 수는 YOLO 7/10, N3+SubPix 10/10이다. B/I-only 대조이며 전체49/192의 새 성능이나 V33 전체 재실행 결과가 아니다. V33 전체 실험은 목표를 달성하지 못해 기본 e를 유지한다. 실패·fallback·미채택 과거 후보와 초기 실행 오류 기록도 원형으로 보존했다.

## 확인 기록

- [독립 공개 묶음 검수](independent_payload_review_v1/REVIEW_V2.json)
- [main 509개 Git blob 및 공개 그림·문서 확인](ROOT_PUBLIC_PAYLOAD_VERIFY_V1.json)
- [Release 55개 서버 해시 검증](ROOT_RELEASE_REMOTE_ASSET_VERIFY_V1.json)
- [최종 게시 완료 기록](ROOT_PUBLICATION_COMPLETE_V1.json)

첫 자료 커밋은 `ae469179`, 원시 manifest·복원 안내·독립 검수 커밋은 `f5908f31`이다. 실험 식별자는 2026-10-15를 유지하고 최종 Release 게시 시각은 2026-10-16 00:02 KST다. 이전 봉인 보고서에 남아 있는 게시 전 상태 설명은 당시 기록이며 현재 공개 상태는 이 문서를 따른다.
