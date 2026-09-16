# 대칭 target / 세 incident edge — 분리된 개발 실험

2026-09-16. 기준 main `44fdd7b415c5ba8cdaaf41ff487217d59a639de1`.
사용자의 새 지시는 이 경로의 A/B 실험을 승인하며, Hough 재사용은 여기의 B에 한정한다. 과거 Hough 종료 메모, 원고, 배포 가중치는 수정하지 않는다.

## 순서와 현재 상태

입력/geometry 감사 → protocol lock → A0 재채점 → A loss 배선 검증 및 동일 예산 학습 → B frozen observation / calibration / evaluation → 별도 보고.
파일 보존, main-only, 각 복구 경계에서 scoped commit/push. 다른 작업의 dirty files는 포함하지 않는다.

A: R0에서 시작하는 전체 검출기 INDEXED/EQUIV, RECT/SQUARE 별도, seed 1/2/3, 각 2,000 update, batch 16, 총 24,000 update 상한.
FP32 AdamW 1e-4, betas .9/.999, decay 1e-4, constant LR, clip10, BN statistics fixed, affine trainable, EMA/AMP/augmentation 없음.
Smoke는 cohort별 2 arms × 3 update 이하. 실제 update 0에서 시작한다.
B: 원래 R0/P1-3/DHT corrected Hough1을 동결. 새 학습 0. 첨부 full-posterior 수식으로 기존 P 후보만 rerank.
주지표는 all-annotated-corner 분모 E_sym, 실패 prediction raw diagonal penalty. Center 별도; 9점 legacy 지표 보존.

## 설치 API에서 발견한 사전 수식 충돌

후속 정정: 사용자가 공동 선택을 승인했고 `A/JOINT_TRAINING_LOCK.json`으로 새 실행 계약을 고정했다. 아래 충돌 발견 기록은 보존하며, 현재의 실제 학습·평가 상태는 `A/training_audit.json`과 `A/results_and_intervals.json`을 따른다.

`PoseLoss26.calculate_keypoints_loss`는 visible point 전체에 걸친 RLE 평균을 **배치 전체에서 clamp(min=0)** 한다.
따라서 일반적으로 `max(0, r1+r2) != max(0,r1)+max(0,r2)`다. 예: r1=-2,r2=3일 때 1 != 3.
두 E2E head에는 서로 다른 assignment / visible-count reduction과 .8/.2 초기 head weight가 있다.
원래 stock 손실을 보존한 전역 대칭 조합 선택과 독립 object-level min은 일반적으로 동치가 아니다.
이것을 숨기고 RLE를 빼거나 object별 clamp를 추가하지 않는다. A 본학습 전 사용자에게 공동 선택 허용 여부를 질문했다.
답변 전 A0/입력 감사/B는 진행 가능. A identity parity만으로 EQUIV 배선 성공이라고 보고하지 않는다.
기존 c4.py는 위치비용만 사용하며 stock의 RLE/visibility 및 두 head 공동 선택을 반영하지 않는다.
또한 stock preprocess/_select_target_keypoints와 기존 _object_slot은 batch_idx 정렬을 가정한다. 새 loader는 정렬을 검증하거나 안정 정렬 adapter를 적용해야 한다.

## 출처 및 주장 경계

- [COPE (WACV 2023)](https://openaccess.thecvf.com/content/WACV2023/papers/Thalhammer_COPE_End-to-End_Trainable_Constant_Runtime_Object_Pose_Estimation_WACV_2023_paper.pdf): 공식 색인의 서지정보와 PDF 링크 확인; 개별 HTML 직접 접근은 403이었다. 본문 전체를 검토했다고 주장하지 않으며, 대칭 supervision 자체의 최초 제안 주장도 없다.
- [BOP task definitions](https://bop.felk.cvut.cz/tasks/): MSSD/MSPD는 사전 정의 object symmetries 사용. 여기의 sparse-corner 평균은 dense-mesh BOP 재현이 아니다.
- [Deep Hough-Transform Line Priors](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/4061_ECCV_2020_paper.php): Hough prior를 학습 feature에 결합하는 선행 맥락.
- [Deep Hough Transform for Semantic Line Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/779_ECCV_2020_paper.php): line parameter space voting의 선행 맥락.
- [HAWP](https://openaccess.thecvf.com/content_CVPR_2020/html/Xue_Holistically-Attracted_Wireframe_Parsing_CVPR_2020_paper.html): 공식 HTML 403 (2026-09-16); 본 실험은 HAWP 구현 재현이 아니다.

새 B의 uniform-lattice likelihood ratio / entropy weighting / fixed-three denominator 조합은 검증 대상이지 성능 보장이 아니다.
Entropy는 physical visibility나 calibrated uncertainty가 아니다. DEV 재사용이며 FINAL은 열지 않는다.
