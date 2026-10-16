# 점·선 연구 종합의 그림

[종합 문서](../../notes/point_line_fusion/STATUS_AND_NEXT_STEPS.md)에 삽입하는 공유 그림이다. 기존 실험 수치를 집계하거나 기존 결과 이미지를 복사했으며 새 학습·추론은 수행하지 않았다.

| 파일 | 내용 |
|---|---|
| [performance.png](performance.png) | 동일 실사 대상점에서 7방법의 오차 및 V12 점 기준 대비 차이 |
| [learnability_and_transfer.png](learnability_and_transfer.png) | tiny 학습·별도 원본의 차이와 합성/실사 방향 AUROC |
| [wrong_direction_example.png](wrong_direction_example.png) | V23 시드1 사후 최대 개선 사례. 개선 후에도 기준점보다 악화 |
| [FIGURE_DATA.json](FIGURE_DATA.json) | 수치·원본 입력 경로·SHA256 |
| [FIGURES.json](FIGURES.json) | 그림 및 생성 코드 SHA256 |
| [ORGANIZATION_RECEIPT.json](ORGANIZATION_RECEIPT.json) | 문서 이동과 연결 경로 갱신 기록 |
| [VALIDATION.json](VALIDATION.json) | 원문 보존·문서 링크·표 수치·그림 출처 검사 결과 |

생성 코드: [build_point_line_overview.py](../../../scripts/docs/build_point_line_overview.py). 저장소 루트에서 `pallet-yolo26` 환경의 Python으로 실행한다. 기존 이미지 복사는 픽셀을 바꾸지 않는다. tiny 음영은 3시드 범위, AUROC 검은 점은 개별 시드이며 신뢰구간이 아니다.
