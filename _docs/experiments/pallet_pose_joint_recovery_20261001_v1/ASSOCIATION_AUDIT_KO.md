# Natural99의 큰 T 오차와 검출 대상 연관 감사

**현재 큰 T 꼬리에는 실제 팔레트 대신 교통 콘의 일부를 선택한 출력이 포함된다.** 자연 가림 99장의 기존 R0 출력 222개 후보를 확인했다. 참조 박스와 IoU≥0.5인 최고 score 검출은 91장이고, 나머지 8장 중 3장은 저장 후보를 바꾸면 박스 매칭을 회복할 여지가 있다. 5장은 저장 후보 중 IoU≥0.5인 박스가 없다. 이것은 사후 박스 진단이며, 새 선택기를 적용하거나 평가에서 8장을 제외하지 않았다.

8장의 원본 RGB를 모두 직접 열어 확인했다. REC_027의 3장에서는 선택 박스가 앞쪽 교통 콘 받침에, REC_022의 5장에서는 뒤집힌 콘의 사각 스티커/패치에 놓여 있다. 여러 개의 정상 팔레트 중 어느 것을 원했는지 모르는 문제라는 증거는 이 8장에서 관찰되지 않는다. 이 시각 관찰은 **새 학습용 실사 라벨이 아니며**, 참조의 독립적인 물리적 정합성 검증도 아니다.

[전체 수치·99개 ID·8장 후보·입력 SHA](ASSOCIATION_AUDIT.json)에는 원래 예측을 보존했다. 새 이미지 추론·학습·후보 routing·GT 기반 출력 선택은 모두 0회다. 앞 단계의 동시 T/R 안정성 실패 판정은 유지한다.

## 1. 전체 8장 시각 증거

![기존 R0의 미매칭 8장 전체](ASSOCIATION_ALL8.jpg)

[원본 크기로 보기](ASSOCIATION_ALL8.jpg). 원본 640×480 전체 RGB 위에 **모든 저장 후보**를 그렸다. 주황은 원래 선택된 `#0`, 파랑은 비선택 후보, 초록 점선은 저장된 참조 코너 8개의 min/max 박스다. 초록 박스와 오른쪽 IoU는 사후 설명에만 사용했다. 화면 밖 참조 박스는 그림 경계에서만 잘려 보이며 계산상의 박스를 clip하거나 다시 정의하지 않았다.

이 그림은 기존 E2 진단에서 이미 고정된 미매칭 8장 전부다. 좋아진 사례를 선택한 그림이 아니다. 아래 표의 `best`는 참조 IoU가 가장 큰 기존 박스이며 **배포에서 사용할 수 없는 oracle**이다.

| 원래 ID의 session / timestamp | 후보 수 | 선택 score | 선택 IoU | 최대 IoU / index | 원래 T cm / R° | 관찰 |
|---|---:|---:|---:|---:|---:|---|
| night08 / 1779449483432542720 | 3 | 0.117387 | 0.0268 | 0.0268 / 0 | 487.70 / 71.78 | 콘 받침, 매칭 후보 없음 |
| night08 / 1779449485800670464 | 7 | 0.013868 | 0.0310 | 0.7686 / 3 | 352.45 / 27.35 | 콘 받침, 매칭 후보 있음 |
| night08 / 1779449496875356416 | 2 | 0.072303 | 0.0276 | 0.0432 / 1 | 291.68 / 76.20 | 콘 받침, 매칭 후보 없음 |
| night09 / 1779449575470221824 | 3 | 0.002912 | 0 | 0.5058 / 1 | 2563.84 / 98.57 | 콘 스티커, 매칭 후보 있음 |
| night09 / 1779449596017728000 | 8 | 0.612443 | 0 | 0.3690 / 7 | 1272.22 / 86.43 | 콘 스티커, 매칭 후보 없음 |
| night09 / 1779449602689248000 | 1 | 0.795900 | 0 | 0 / 0 | 1313.35 / 65.12 | 콘 스티커만 저장됨 |
| night09 / 1779449604823769344 | 5 | 0.759558 | 0 | 0.7573 / 2 | 1056.87 / 64.96 | 콘 스티커, 매칭 후보 있음 |
| night09 / 1779449661263803392 | 1 | 0.813226 | 0 | 0 / 0 | 632.40 / 75.04 | 콘 스티커만 저장됨 |

네 오선택의 score가 0.61~0.81이므로 단순히 낮은 confidence만 버리면 되는 문제가 아니다. 8개의 선택 박스 중심은 모두 원본 영상 안에 있다. 이전에 확인된 reflect-padding 영역 오검출만 제거하는 규칙으로는 이 8장을 고칠 수 없다.

## 2. 평가가 정한 대상과 실행 코드가 받는 정보

평가 함수 [`_legacy_forbidden_target`](../../../challenge/evaluation_v2/paper_real_eval.py#L1566)은 GT-v2 파일에 객체가 정확히 하나인지 검사하고 `objects[0]`의 9개 코너·가시성·카메라·등록 치수를 읽는다. 비교 박스는 [`L1643`](../../../challenge/evaluation_v2/paper_real_eval.py#L1643)의 코너 0~7 전체 min/max다. 가림·외삽·화면 밖 코너도 박스 구성에 포함되며, 가시점만 감싼 박스가 아니다. 따라서 IoU<0.5가 곧 별개의 물리적 instance를 검출했다는 증명은 아니다.

실제로 natural99의 annotation은 모두 객체 1개, `class=pallet`, `name=real_pallet`이다. 검사한 `target_id/instance_id/track_id/object_id/query/prompt` 필드는 없다. **주석 작성자가 지정한 단일 대상**은 명확하지만, 실행 중 특정 팔레트를 지목하는 별도 ID/클릭/작업 ROI가 이 계약에 들어 있지는 않다.

[`FrozenYoloFeatures.predict`](../../../scripts/research/pallet_line_pose_v1/features.py#L55)는 RGB만 받아 후보를 만들고 box score의 `np.argmax`로 하나를 고른다. 현재 pose 경로는 그 검출의 점과 외부 카메라 K·등록 치수를 사용한다. 관련 로컬 이미지/치수 API `scripts/research/pallet_direct_dimension_roi_v1/predict.py:12`의 `predict(image, checkpoint, dimensions)`에도 target ID 인자는 없다. 이 API 파일은 이번 GitHub 공개 범위에 포함하지 않았다. `challenge/robot/fork_target.py`는 이미 선택된 pose에서 포크 진입 위치를 계산하는 후단이며 객체 식별기가 아니다.

따라서 일반적인 **여러 동일 종류 팔레트 중 작업자가 원하는 대상 선택**까지 요구하면 작업 대상 지정 계약이 추가로 필요하다. 크기나 화면 중앙을 조용히 정답의 대용으로 쓸 수 없다. 그러나 이번 8장은 이 불명확성을 핑계로 해결 불가능하다고 할 사례가 아니다. 이미지상 콘의 부분을 팔레트로 보고한 검출 품질 문제가 우선 확인된다.

## 3. 현재 후보 풀은 정확히 무엇인가

현재 R0 SHA는 `970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7`이다. 기존 [추론 입력 lock](../pallet_pose_stable_improvement_20261001_v1/INFERENCE_INPUT_LOCK.json)과 진단의 원시 입력/metadata SHA를 다시 대조했다.

| 항목 | 현재 계약 |
|---|---|
| 입력 | 원본 RGB에 100 px `BORDER_REFLECT_101`, `imgsz=640`, `rect=True` |
| inference flags | `augment=False`, `half=False`, 원본 좌표로 padding 차감 |
| head | Pose26 end-to-end one2one, score 상위 top-k |
| 후보 수 제한 | `max_det=300` |
| score floor | **`score > 0.001`**, `>=`가 아님 |
| IoU NMS | 이 end-to-end 경로에서는 수행하지 않음 |
| 최종 선택 | 최고 box score, 동일 score면 `np.argmax`의 첫 후보 |
| natural99 실물 | 총 222개, 단일 후보 48장, 다중 후보 51장; 99/99가 실제 최고 score 선택 |

근거는 [현재 feature extractor](../../../scripts/research/pallet_line_pose_v1/features.py), [paper evaluator adapter](../../../challenge/evaluation_v2/paper_real_eval.py#L1721), 설치된 Ultralytics **8.4.60**의 `utils/nms.py`, `nn/tasks.py`, `nn/modules/head.py`, `cfg/default.yaml`이다. 설치 파일 SHA도 감사 JSON에 기록했다. 기본 `iou=0.7`은 존재하지만 `end2end=True` 분기는 IoU 억제 없이 score floor와 `max_det`만 적용한다. 따라서 **“NMS가 정답 박스를 없앴다”는 설명은 현재 저장 경로에 맞지 않는다.**

저장 풀은 dense head의 모든 pre-threshold anchor가 아니다. 여기서 “5장에 맞는 후보가 없음”은 **현재 전처리·head·top-k·floor를 통과해 저장된 후보 중 없음**을 뜻한다. 더 낮은 score의 dense 후보, 다른 해상도나 변환 입력에서의 후보 존재까지 부정하지 않는다. 반대로 이들의 존재도 현재 cache로 증명할 수 없다.

## 4. 박스 선택을 고치면 T/R이 함께 해결되는가

**그렇지 않다.** 기존 참조 best-IoU 박스 oracle조차 natural99 전체 T 중앙값/P90은 `12.403/120.471 → 12.118/69.766 cm`지만, R 중앙값/P90은 `5.218/88.922 → 4.704/88.924°`다. 이 oracle는 정상 매칭 사례를 포함해 99장 중 19장의 선택을 바꾸며, 3장만 고친 값이 아니다. GT 정보를 쓴 사후 상한 진단이고 실행 가능한 모델 결과가 아니다.

매칭 후보가 있었던 3장의 best-IoU 결과를 따로 보아도 다음과 같다.

| ID 끝부분 | 원래 T/R | best-IoU 기존 후보 T/R | 해석 |
|---|---:|---:|---|
| 1779449485800670464 | 352.449 cm / 27.354° | 49.448 cm / 89.618° | T 감소, R 악화 |
| 1779449575470221824 | 2563.837 cm / 98.574° | 151.771 cm / 87.941° | 둘 다 감소하지만 여전히 큰 오류 |
| 1779449604823769344 | 1056.871 cm / 64.960° | 3.462 cm / 89.486° | T 감소, R 악화 |

기존 [E2 진단](../pallet_pose_diagnosis_20260930_v1/E2_DETECTION_SUMMARY.json)을 재인용했으며 새 GT selector를 실행하지 않았다. 박스 매칭은 키포인트 역할·W/D·연속 좌표 정확도를 보장하지 않는다. 정상 매칭 91장에서도 앞 단계의 seed 전반 안정적 개선은 없었으므로, 검출만이 유일한 실패 원인이라고 하지 않는다.

## 5. 이미 수행한 관련 시도

| 선행 시도 | 실제 범위와 결과 | 현재 의사결정에서의 한계 |
|---|---|---|
| [기존 learned GEO](../../../scripts/research/pallet_selector_recovery_v1/features.py#L28) | 먼저 `selected(pred)`를 읽은 뒤 같은 검출의 long/short W/D pose 두 개 중 선택 | 객체/박스 재선택기가 아니다. 현재 source-only GEO 재보정과 검출 회복을 구분해야 함 |
| [PRE_V2 confidence/해상도](../../../challenge/yolo_pose_one_model/analysis_pre_v2/PRE_V2_DIAGNOSTIC_REPORT.md) | 640/960/1280, conf 0.001~0.4. 당시 960에서 availability↑여도 matching↓, 1280 악화 | **구161/별도 `paper_generic_v1` checkpoint**이며 cache는 top5 절단. 현재 G38 자연99의 효과로 이식 불가. 문서의 강한 원인·GT 신뢰성 표현도 여기서 재승인하지 않음 |
| [PRE_V2 박스 rerank](../../../challenge/yolo_pose_one_model/analysis_pre_v2/RERANK_REPORT.md) | box area/diagonal 등 단일 특징, 사후 session LOSO. 당시 box recall 0.708→0.814, 5cm5 0.304→0.311, R median 4.33° 동일 | 단순 최대 박스 선택은 새로운 방법이 아님. 구161·평가 GT에 의한 feature 선택이 포함된 개발 결과를 현재 no-new-GT 선택 규칙으로 채택 불가 |
| [P26 one2one/one2many/NMS](../../../challenge/yolo_pose_one_model/p26_inference_path_audit/FINAL_P26_INFERENCE_PATH_REPORT.md) | 무학습. `INFERENCE_PATH_NOT_FACTOR`, benefits 0/4. NIGHT any↑지만 top1 14→12/28 | 체크포인트는 Y0 SHA `37f904…`, 현재 G38 SHA가 아님. generic 경로 변경이 미시험이라고 주장할 수 없고 현재 완전한 불가능 증명도 아님 |
| [FAST-A/B/C](../../../scripts/self_training_yolo/fast_teacher_v1/fast_teacher.py) | A=640 orig/flip 평균, B=640+960 orig/flip median, C=다른 source 모델까지 median. **세 가지 모두 FAIL** | box confidence≥0.85 등의 teacher 지원 집합만 평균한 2D 실험이며 전체99 최종 pose나 모든 박스의 공동 재선택 실험이 아님. A/B/C 지원 frame은 230/221/235 of319 |
| [hard-negative 학습](../../../challenge/yolo_pose_one_model/hard_negative_v1/PHASE_A/METHOD_SPEC.md) | source negative9K에서1,900장 mining, stock/focal loss. HM/HF 모두 positive suppression으로 STOP | “hard negative를 처음 넣어보면 된다”는 제안은 선행 실패를 누락함. 당시 별도 Y0 계열/학습조건이며 현재 콘 혼동에 대한 독립 증명은 아님 |
| [현재 R0의 검출 분기 적응](../pallet_paper_contribution_screen_v1/C_geometry_preserving_da/RESUME_COMPLETE/REPORT.md) | 3arm×3seed×900 updates, dense geometry 보존 C2도 검출 회복 gate FAIL | 검출 score/box 학습도 이미 시도됨. 좌표만 고정하면 최종 pose가 보존되는 것이 아님 |
| [현재 R0/C2 score–box 교환](../pallet_paper_contribution_screen_v1/C_score_box_selection_v1/INTERPRETATION_KO.md) | 같은 dense grid에서 분리. 두 hybrid 모두 모든 seed에서 R0보다 AP 낮음. C2 score는 일부 matching↑이나 FPR95·T 평균 악화 | 기존 score만 바꾸거나 box만 섞는 것을 검증된 개선으로 재추천할 근거 없음 |

FAST-A/B/C의 동일 지원점 ALL NME P90은 각각 `0.07840→0.09933`, `0.08197→0.09964`, `0.07946→0.10538`이었다. 실제 JSON과 gate를 감사 JSON에 작게 포함했다. FAST-B의 `R0_TTA960_CACHE.json`은 319장의 top1/flip_top1 좌표·confidence만 남기며 box와 직접 checkpoint SHA 영수증이 없다. 코드가 현재 R0 경로를 가리킨다는 것만으로 bit-exact 재사용이 인증되지는 않는다.

또한 설치된 `DetectionModel._predict_augment`는 end-to-end 모델 또는 `DetectionModel`이 아닌 클래스에서 `augment=True`를 단일 추론으로 되돌린다. **flag만 켜는 것은 이 Pose26에서 실제 TTA가 아니다.** FAST 실험처럼 입력을 명시적으로 변환하고 좌표·역할을 복원해야 한다.

## 6. 가능한 다음 작업의 경계

이 감사만으로 안정적 T/R 개선이 입증된 신규 검출 수정은 찾지 못했다. **새 학습을 바로 추천하지 않는다.** 단순 score threshold, 최대 면적, PnP 잔차, NMS 변경, 좌표 평균 TTA는 이미 시도됐거나 현재 계약에 작동하지 않거나 다른 지표의 회귀가 확인됐다.

이전 시도와 구분 가능한 아직 미확인 질문은 **“정확한 현재 G38에서 변환 입력별 모든 객체 후보를 보존했을 때, 놓친 팔레트 후보가 생성되는가”**다. 기존 FAST-B는 top1 좌표 평균이므로 이 질문에 답하지 못한다. 이를 진행한다면 source TRAIN/VAL에서만 고정한 객체 단위 대응·선택 규칙, 명시적 변환 복원, 전체 full128/wood45 출력 lock, 동일 T/R·꼬리·실패·clean 기준이 먼저 필요하다. 현재 8장에 맞춰 해상도·박스 모양·위치·cone 색 threshold를 고르거나 recording별 분기를 만들 수 없다. 현 감사에서는 해당 추론을 실행하지 않았고 성공을 예상 수치로 제시하지 않는다.

source-only 객체/비객체 분별을 다시 검토하려면 기존 negative9K 및 source positive가 **팔레트가 존재하는 영상 안의 부분 구조 오검출**을 감독할 수 있는지 먼저 확인해야 한다. negative-only 영상에서 confidence를 낮추는 과거 학습과 다른 입력·타깃 증거가 없는 상태에서 새 classifier 학습을 시작하면 같은 실패를 반복할 위험이 있다. 현재 실사 8장의 콘 위치를 학습용 새 negative box로 지정하는 것은 no-new-real-GT 조건에 맞지 않는다.

5개의 저장 풀 누락은 재순위만으로 IoU≥0.5 매칭을 만들 수 없다. 후보를 새로 만들거나 검출 위치를 고치는 방법이 필요하며, 그것도 R 문제를 별도로 해결해야 한다. 반대로 다중 팔레트 작업에서 어떤 팔레트를 원하는지 지정하지 않은 문제는 모델 confidence만으로 복원할 수 없는 작업 계약 문제다. **이번 자료에서는 전자의 검출 문제를 확인했으며, 후자의 불가능성을 이유로 현재 실패를 정당화하지 않는다.**

## 7. 감사 범위와 보존

기존 natural99 전체 예측/박스/선택을 읽고 이전 진단의 수치를 요약했다. 99개 annotation SHA 및 8개 표시 RGB SHA를 기존 binding과 확인했다. 모든 8장을 원본으로, 완성된 contact sheet를 다시 이미지로 직접 확인했다. 실측 T/R은 K·수동2D로 구성한 geometry reference 기준이며 독립 장비의 physical truth가 아니다. 이 자료는 반복 사용한 DEV이다.

새로 쓴 감사 결과는 본 문서, [감사 JSON](ASSOCIATION_AUDIT.json), [8장 그림](ASSOCIATION_ALL8.jpg)이다. [그림 재현 코드](../../../scripts/research/pallet_pose_joint_recovery_20261001_v1/association_figure.py)도 함께 공개한다. 코드는 감사 JSON에 기록된 8개 ID·박스·수치와 SHA가 일치하는 기존 로컬 RGB만 읽으며, 새 출력 경로만 허용한다. 원본 RGB는 별도로 보유해야 하고, 검토한 그림을 덮어쓰지 않는다.

```bash
python -m scripts.research.pallet_pose_joint_recovery_20261001_v1.association_figure --output /tmp/association_all8_reproduced.jpg
```

기존 phase의 protocol·모델·예측·결과는 수정하지 않았다. 후보 교체·학습·추론·Git 변경은 수행하지 않았다. 공개 직전 추가한 생성 코드는 AST와 CLI help를 확인했고, 이미 검토한 그림은 재생성하지 않았다.
