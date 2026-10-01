# IEEE Sensors Journal 원고 마감: 현재 주장할 수 있는 결과

검토일: 2026-10-01. 이번 주 마감을 위한 저자 검토용 정리다. 추가 실험을 진행 중이며 완료 수치는 각 실험의 완료 영수증과 결과 보고서로 확인해야 한다. 원래 T/R 안정적 공동 개선 목표는 미달성으로 유지한다. 기존 봉인 결과는 유지하고 새 원고 revision에서 확장한다.

## 판단

논문에 쓸 수 있는 검증된 결과가 있다. 최근 선택기 실험의 실패가 기존의 2D 보정 성과나 별도의 corrected-pseudo-label 통제 비교를 무효화하지 않는다. 다만 현재 증거로 IEEE Sensors Journal 채택 가능성을 높다고 단정할 수는 없다. 새로운 방법의 차별성과 반복 DEV에 대한 일반화 한계가 남는다.

사용자가 전한 교수님 조건은 **YOLO, DOPE와 추가 모델 하나를 포함한 총 세 기반 추정기에서 보정 전후를 실험해야 한다**는 것이다. 아래의 작은 보정 모듈 P와 정밀도–계산비용 비교는 바로 그 보정기 논문 방향이다. 별개의 대안이 아니다. 기존 [Sensors 원고](sensors_submission_v1/manuscript.tex)는 출발점이지만, 현재 YOLO 결과만으로 교수님 조건까지 충족했다고 판단할 수 없다. 앞서 이 조건을 반영하지 않고 기존 결과 정리만 권한 판단을 수정한다.

**T/R의 안정적 공동 개선을 주장하지 않는 것과, 다른 추정기에서도 보정 효과를 검증하는 것은 별개의 문제다.** 전자는 현재 증거에 맞게 주장을 제한하는 것이며 후자의 실험을 대신하지 않는다. 기존 2D 결과는 유효한 근거로 남지만, 다른 추정기로의 적용 범위는 아직 확인되지 않았다.

## 교수님 조건에 따라 추가로 필요한 비교

| 기반 추정기 | 보정 전 | 보정 후 | 현재 확보 상태 |
|---|---|---|---|
| 기존 YOLO | 고정 R0 | 고정 R0 + P | 기존 Sensors 평가 있음 |
| DOPE (VGG) | 기존 최종 DOPE | DOPE + D/P 각3seed | adapter·CPU/GPU smoke 완료, source 예측 생성 진행 중. 본학습·실사 평가 미완료 |
| SimpleBaseline-derived ResNet-18 | 합성 TRAIN으로 새로 학습할 9점 추정기 | 동결 ResNet-18 + D/P 각3seed | 독립 RGB 입력·모델·9점 타깃 CPU 검산 완료. GPU 학습 및 평가 미완료 |

핵심은 각 기반 추정기 안에서 보정 전후의 차이를 확인하는 것이다. YOLO와 DOPE 자체의 성능 비교, 또는 P와 더 큰 PoseFix-derived PRIOR의 비교는 이 질문을 대신하지 않는다. 기존 DOPE+DHT 실험도 다른 보정 방법이므로 P의 적용 범위를 입증하는 결과로 사용할 수 없다.

2026-10-01 읽기 전용 점검에서 다음을 확인했다.

- DOPE 체크포인트 `weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth`가 실제 존재하고 SHA-256은 `0de80490cb3b4f9b11565db7a4aea6338f64edb8f9614910bfb52bf03ce0dc3f`이다. 합성 학습 데이터도 있으므로 DOPE 전체를 처음부터 재학습해야 하는 상황은 아니다.
- 현재 [P 구현](../../scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py)은 YOLO P3/P4의 채널과 stride 8/16을 사용한다. [DOPE 구현](../../Deep_Object_Pose/common/models.py)은 VGG 기반이며 마지막 공통 특징의 stride가 8이다. 특징·좌표 adapter가 필요하므로 기존 P 가중치를 그대로 꽂아 평가할 준비가 됐다고 말할 수 없다.
- DOPE의 서로 다른 기존 캐시는 square400/short-side400 등 전처리와 decoder 규약이 다르다. 한 규약을 먼저 고정하고 같은 DOPE 출력으로 보정 전후를 비교해야 한다. resize 역변환 수정 효과가 P 효과에 섞이면 안 된다.
- 기존 캐시에 현재 Plastic128/Wood45의 173장 identity는 모두 있지만, 빈 예측과 결측 코너가 있다. 경로 대응은 원영상 해시·실행 재현성 검증과 다르다. 성공한 이미지로 분모를 줄여서는 안 된다. 기존 Sensors 주 표와 직접 연결하려면 동일 319장 평가를 맞춰야 하며, 173장 실험만 수행하면 별도의 평가 범위로 표시해야 한다.

현실적인 검증 방향은 기반 추정기를 각각 고정하고, 동일한 보정 원리와 학습 예산으로 기반별 보정 head를 학습·평가하는 것이다. 이것은 **방법을 세 추정기에 각각 적용하는 실험**이며, **YOLO에서 학습한 동일 가중치의 무학습 전이**와 다르다. 채널 adapter, 좌표 규약, trainable parameter 수, 각 추정기의 결측·box 정의와 감독 예산을 공개한다. 같은 데이터에서 median/P90/PCK, 실패 수, T/R, 추가 지연시간을 함께 보고한다. 세 추정기에서 성공하더라도 모든 backbone에 일반화한다고 주장하지 않는다.

현재 DOPE의 합성 예측 캐시를 실제로 생성 중이다. 다음으로 동일 P/D 고정 학습 예산과 실사319장 평가를 실행한다. 세 번째 ResNet-18은 ImageNet 초기값에서 팔레트9점 기본 추정기를 합성55,980장으로60epoch 학습한 뒤 고정한다. YOLO crop이나 DOPE 예측을 입력으로 사용하지 않는다. 이는 full-image pallet 적응형 SimpleBaseline이며 원 논문의 human-pose benchmark 재현이라고 부르지 않는다. 기존 local PVNet 체크포인트는 DOPE/VGG에 다른 head를 붙인 구조여서 독립 세 번째 backbone으로 세지 않는다.

[DOPE 실행 코드](../../scripts/research/pallet_dope_refiner_20261001_v1/run.py) · [ResNet-18 학습 코드](../../scripts/research/pallet_resnet18_refiner_20261001_v1/baseline_train.py) · [확장 원고 초안](sensors_dope_extension_20261001_v1/README_KO.md). 성능 개선이나 이번 주 완료를 사전에 보장하지 않으며 새 모델의 학습 속도를 측정해 시간을 갱신한다.

## 주된 논문 근거: 기존 Sensors 보정 모듈 P

근거는 [기존 최종 보고서](../experiments/pallet_sensors_submission_v1/FINAL_REPORT_KO.md), [최종 상태와 paired 결과](../experiments/pallet_sensors_submission_v1/FINAL_STATUS.json), [전체 평가](../experiments/pallet_sensors_submission_v1/UNIFIED_DEV_RESULTS.json), [속도 패널](../experiments/pallet_sensors_submission_v1/RUNTIME_PANEL.json)이다.

| 주장 | 실제 근거 | 반드시 함께 적을 범위 |
|---|---|---|
| 작은 모듈로 기존 검출기 출력의 코너 정밀도를 개선했다 | P는18,962 parameters. R0 6.615678px → P 5.904910px | 319장/13세션의 반복 DEV. 기존 매칭311장/2,756점 조건부 pooled median의3seed 평균이며, 모든 프레임의 무조건 오차가 아님 |
| 차이는 통제 비교에서 관찰됐다 | P−R0 −0.710767px, session95% interval [−1.173744, −0.411083]; P−direct-control −0.525361px | 개발 데이터 내 비교이며 새 세션 독립 확인이나 모든 지표 우월성은 아님 |
| 더 큰 별도 보정기와 정확도–비용 차이를 측정했다 | P 대표 full-path15.352ms, PoseFix-derived PRIOR29.020ms. PRIOR median5.568679px로 P보다 정확 | 동일 desktop 측정 조건의 대표 seed1 속도. 정확도는3seed 평균. 임베디드/Jetson 실시간 보장·동등 정확도 주장은 불가 |
| 검출 결과를 유지하는 보정 인터페이스를 구현·확인했다 | 기존 boxes/scores/instance selection/center 보존, 지원되는 corner0–7만 수정 | 시험한 모델·wrapper 계약에 한정. 모든 모델에 적용 가능하거나 pose 악화가 없다는 보장은 아님 |

P가 PRIOR보다 정확하다고 쓰면 안 된다. P−PRIOR는 +0.336232px, session95% interval [+0.062853,+0.584639]로 오히려 PRIOR 쪽 오차가 작다. P의 기여는 측정된 국소 정밀도 개선과 계산비용, 기존 검출 출력 보존을 함께 보여주는 데 있다. 파라미터 수만으로 속도 향상을 추론하지 않고 실제 측정값을 쓴다.

원고의 중심 문장 초안:

> We evaluate a lightweight local keypoint refiner that reuses features of a frozen RGB pallet detector while preserving its detection outputs. On reused development data, the refiner reduces image-space landmark error relative to the unchanged detector and a matched direct-regression control. A larger PoseFix-derived comparator attains lower error at higher measured desktop latency. We report this accuracy–cost comparison together with downstream geometry errors and failure cases, without claiming independent confirmation of stable joint translation and rotation improvement.

추천 제목 방향: **Feature-Reusing Local Keypoint Refinement for Monocular Pallet Pose Estimation: Accuracy and Computational Cost**. “최초”, “SOTA”, “robust 6D improvement”, “안전한 포크 삽입”을 제목이나 결론에 추가할 근거는 확인하지 못했다.

## 별도의 양의 근거: corrected-target self-training

이 결과는 위 P 모듈과 다른 개입·교사·학습 예산·분모를 사용한 연구다. 두 방법의 좋은 수치를 같은 proposed-method 행으로 합치면 안 된다. 원래 의도한 원고가 self-training이라면 [별도 전체 초안](selftraining_submission_v1/manuscript.tex)을 기반으로 아래의 좁은 주장을 쓸 수 있다.

| 비교 | 관찰된 결과 | 제한 |
|---|---|---|
| Plastic raw-target vs corrected-target 학생 | 동일128장/985코너 PCK10 468/985→507/985, 47.51→51.47%; 동일 D9 AUC0.33472→0.35902 | LR5는 과거 DEV에서 선택. 기존 mixed-provenance reference에 대한 pooled 개선. 독립 재검수66점의 학생 PCK10은43/66→43/66 동률 |
| Wood raw-target vs corrected-target 학생 | 동일45장/346코너 PCK10 163/346→165/346; AUC0.65643→0.66503 | +2코너의 작은 변화. R0 167/346 및 AUC0.67050보다 낮고, P90과 한 recording의 PCK10은 악화 |
| 직접 검수 가시점에서 교사 좌표 보정 | Plastic66점 PCK10 44/66→50/66 | 학생 성과와 다른 질문. 무주석 학습217장의 정확도를 직접 측정한 것이 아니며 Wood trusted quality는 미확정 |

이 통제 비교의 질문은 “같은 학습 이미지·support·초깃값·augmentation·source replay·320updates에서 보정 좌표가 원래 좌표보다 유용했는가”이다. teacher의 실사9장/수동38코너 감독과 두 arm의 공통 teacher-filtered 선택을 공개해야 한다. 전체 시스템을 무실사정답이라고 표현하지 않는다.

근거: [Plastic 결과](../experiments/pallet_selftraining_paper_closure_v1/REPORT_KO.md), [공정 비교](../experiments/pallet_selftraining_paper_closure_v1/CORE_COMPARABILITY_AUDIT.json), [Wood 결과](../experiments/pallet_material_selftrain_closure_v1/REPORT_KO.md), [원고 주장–근거](selftraining_submission_v1/CLAIMS_KO.md). 기존 빌드 기록은13페이지 초안이며, 이것이 Sensors 권장 분량이나 독립 확인 요건을 자동으로 충족한다는 뜻은 아니다.

## 최근 T/R 실험의 논문 내 역할

최신 [이미지 특징 추가 실험](../experiments/pallet_pose_signed_axes_visual_20261001_v1/REPORT_KO.md)은 합성 VAL에서 T 중앙값이 개선되었으나43/45로 실패했고 실사는 평가하지 않았다. 이를 실사 성공으로 쓰지 않는다. 앞선 [RBF 실험](../experiments/pallet_pose_anchor_rbf_20261001_v1/REPORT_KO.md)은 합성45/45를 통과했지만 실사 안정성2/5였으므로, 최적화 수렴이나 합성 통과만으로 실사 개선을 보장할 수 없다는 한계 분석에 쓴다.

[초기 진단](../experiments/pallet_pose_diagnosis_20260930_v1/REPORT_KO.md)은 2D 개선과 T/R 개선이 일치하지 않는 사례, W/D 선택과 검출 실패의 영향을 구분했다. [후보 oracle](../experiments/pallet_pose_real_union_feasibility_20261001_v1/REPORT_KO.md)은 평가 참조를 사용하므로 배포 성능 표에서 제외하고 진단으로 명시한다. 학습법을 계속 바꾼 수십 가지 실험을 모두 본문의 기여로 나열하지 않는다. 원래 목표를 달성했다고 쓰거나 평가 기준을 뒤늦게 바꾸지 않는다.

## 이번 주 완성 순서

1. 먼저: 기존 P 보정기 연구를 기준으로 교수님이 요구한 YOLO/DOPE/ResNet-18 전후 비교의 입력·학습·평가 규약을 고정한다. DOPE adapter와 좌표·결측 처리를 검증하고, 실제 실행 가능 여부를 판단한다.
2. 추가 실험: 고정 DOPE와 새로 source 학습한 ResNet-18을 각각 D/P 보정 전후의 같은 이미지·분모에서 평가한다. 보정 효과와 전처리 수정 효과를 구별하고, 실패·악화 사례도 남긴다. 기존 YOLO 결과와 지표·집계 방식을 맞춘다.
3. 결과 확인 후: 주 표는 세 추정기 각각의 보정 전후와 정확도–속도 비교, 주요 그림은 파이프라인·실제 이미지와 치수·개선/악화 사례로 구성한다. T/R은 전 지표와 실패를 보존한다. 추가 모델의 결과가 없거나 개선되지 않았다면 그대로 밝히고 주장 범위를 다시 결정한다.
4. 제출 전: 모든 수치의 출처·분모·seed 집계·단위·DEV/독립 확인 구분과 PDF 레이아웃·그림·참고문헌·graphical abstract·저자 정보를 점검한다. P와 별도 self-training 결과를 하나의 방법처럼 합치지 않는다.

IEEE Sensors Journal 공식 범위에는 sensor data processing과 sensor systems/application이 포함된다. 따라서 카메라 기반 측정의 정확도와 비용을 중심에 두는 구성은 검토할 만하지만, 범위에 맞는 것만으로 방법의 독창성이나 채택 가능성이 입증되는 것은 아니다. [공식 범위](https://ieee-sensors.org/ieee-sensors-journal/)

2026-10-01 확인한 공식 저자 안내는 IEEE 두 단 형식, 통상8페이지 이내의 최초 원고, graphical abstract를 안내한다. 8페이지 초과를 자동 불가로 해석하지 않으며 초과 분량에 비용 규정이 있다. 이번 마감에는 본문 약8페이지와 필요한 보조자료로 정리하는 편집안을 권한다. [공식 저자 안내](https://ieee-sensors.org/ieee-sensors-journal/for-authors/)

현재 판단: **기존 P의 2D 정밀도–계산비용 결과는 논문 근거가 있다. 그러나 교수님이 요구한 다른 추정기 적용 검증은 아직 미완료이며, T/R의 안정적 공동 개선도 달성하지 않았다. 보정기 논문을 완성하려면 이 둘을 구별하고 DOPE와 독립된 세 번째 모델의 전후 비교를 추가해야 한다.**
