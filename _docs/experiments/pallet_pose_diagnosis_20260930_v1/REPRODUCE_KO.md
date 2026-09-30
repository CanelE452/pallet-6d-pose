# 재현 및 파일 안내

보고서와 비교 이미지는 GitHub만으로 확인할 수 있다. 원본 전체 RGB, GT/카메라 배열, 모델 가중치와 대용량 raw cache는 이 게시물에 포함하지 않는다. 새 clone만으로 1,027회의 모델 추론을 즉시 재현할 수 있다고 주장하지 않는다. 로컬 데이터 소유자는 RUN_MANIFEST.json의 입력 경로·hash와 MODEL_LINEAGE.json의 checkpoint를 먼저 맞춰야 한다.

## 검토 파일

| 목적 | 파일 |
|---|---|
| 주요 관찰·반론·다음 비교 | [REPORT_KO.md](REPORT_KO.md) |
| 입력·집합·모델·평가 규칙 | [INPUTS_AND_METHOD_KO.md](INPUTS_AND_METHOD_KO.md) |
| 자연99 전체 사진과 전후 코너 | [GALLERY_NATURAL99.md](GALLERY_NATURAL99.md) |
| 모든 비교의 CI·분모·LORO | [STATISTICS_KO.md](STATISTICS_KO.md), [JSON](E7_ALL_COMPARISONS.json) |
| 프레임별 결과 | [FRAME_RESULTS.csv](FRAME_RESULTS.csv), [E2](E2_FRAME_REVIEW.csv), [E5-B](E5B_FRAME_REVIEW.csv) |
| recording·metadata 결과 | [recording](E7_BY_RECORDING_ALL.csv), [metadata](E7_METADATA_STRATA_ALL.csv) |
| 실제 미완료·생략·차단 질문 | [QUESTION_STATUS.json](QUESTION_STATUS.json) |
| 발견한 누락과 보완 | [REVIEW_CORRECTIONS_KO.md](REVIEW_CORRECTIONS_KO.md) |
| Source 별도 기하 확인 | [SOURCE_CONTRACT_REVIEW.json](SOURCE_CONTRACT_REVIEW.json) |
| 원래 지시문 | [REQUEST_PLAN.txt](REQUEST_PLAN.txt) |

CSV의 `selected_hypothesis_for_pose`와 `GEO_name`은 그 행의 T/R에 실제 사용한 후보다. `GEO_free_name`은 자유 GEO 재선택 후보다. `held_identity` 및 `_held`에서는 서로 다를 수 있다. `twoD_value_scope=condition_output`은 개입 후 좌표의 2D 오차다. 매칭 실패/유효 감독점 부재는 `twoD_NA_reason`으로 표시하며 pose의 행을 제거하지 않는다. ΔT/ΔR은 조건의 baseline 대비 차이이고 E3/E4/E5의 baseline은 해당 단계 조건에 맞춘 identity다.

## 실행 환경과 명령

이번 환경은 Python(pallet-yolo26), torch2.1.1+cu118, ultralytics8.4.60, OpenCV4.9.0, numpy1.26.4이며 실제 CUDA 사용은 불가능하여 CPU에서 추론했다. 모델 추론량1,027회와 새 fit0회를 혼동하지 않는다. 최초 실험 명령은 [EXECUTION_COMMANDS.md](EXECUTION_COMMANDS.md)에 있다. 명령들은 repository root에서 실행한다.

```bash
export MPLCONFIGDIR=/tmp/pallet-mpl
PY=/home/minjae/anaconda3/envs/pallet-yolo26/bin/python
$PY -m scripts.research.pallet_pose_diagnosis_20260930_v1.review_analysis source
$PY -m scripts.research.pallet_pose_diagnosis_20260930_v1.review_analysis repair
$PY -m scripts.research.pallet_pose_diagnosis_20260930_v1.review_analysis statistics
$PY -m scripts.research.pallet_pose_diagnosis_20260930_v1.review_figures
$PY -m scripts.research.pallet_pose_diagnosis_20260930_v1.review_report
```

보완 명령은 기존 raw 결과가 있어야 하며 새 추론을 하지 않는다. 수정 전 산출물은 before_github_review에 보관한다. 최초 run/close/report 명령은 exclusive-create라 기존 결과를 덮어쓰지 않으며, 새 추론 재실행은 별도 namespace와 데이터 계약 확인이 필요하다. 역사적 report.py는 최초 초안을 만들기 위한 코드이고 최신 GitHub 검토판은 review_report.py가 생성한다.

시각화는 저장된 실제 사진과 예측/참조 좌표로 만들었다. AI 생성 이미지나 성능을 보이게 조작한 사진이 아니다. gallery는 natural99 전체이며 recording·ID 순이다. 본문의 4개 방향 사례는 각 집단의 ID 첫 프레임, E3는 recording별 ID 첫 프레임, T꼬리는 상위10장 전부다. [선택 기록](FIGURE_CASE_SELECTION.json).

## 검증 범위

원래 2,964행 독립 T/R 검산, 원본 입력 hash 보존, 이번 CSV 필드 수정과 E4 pose 재검산, Source256 기하 대조, E7 모집단/recording 수, 이미지 파일·Markdown 상대 링크 및 Git 게시 목록을 점검한다. PUBLICATION_VALIDATION.json은 게시 전 검증이고 GitHub 원격 반영 여부는 실제 push 뒤 확인한다. 최초 완료 manifest는 history에 보존하며 최신 공개 파일 SHA256은 PUBLICATION_MANIFEST.json에 있다.
