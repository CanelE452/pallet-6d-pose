# 여러 팔레트의 치수 입력 모델 — 진행 상태 확인

> 최신 실행: [네 군 ×3시드 출력층 비교 완료](dimension-symmetry-2x2-results.md). 치수 입력의 작은 효과는 관측됐으나 안정적인 기존 YOLO 개선 기준은 미통과했다. 아래는 이 실행 이전의 상태 확인 기록이다.

> **공유 대화 원문 확인 후 정정:** 사용자가 지칭한 것은 치수 입력과 점·선·자세의 공동 대칭 로스였다. 관련 구현·학습은 v5/v6에서 했지만, 치수 유무 × 대칭 학습 유무의 네 군 완료 기록은 찾지 못했다. [원문·실행 대조 기록](shared-dimension-symmetry-review.md)을 우선한다. 아래 S1과 translation/LC는 구분을 위한 별도 실험 기록이다.

2026-09-11 사용자 정정: “여러 파렛크 종류 치수 입력으로받기”. 이번 문서는 저장된 실행 산출물과 코드를 확인한 상태 정리다. 새 학습 결과가 아니다.

## 실제로 완료한 것

1. **치수로 앞면 W/D 방향을 고르는 분류기.** S1_SPATIAL_CONCAT은 동결 YOLO 공간 특징·키포인트 특징·치수 로그 비율 4차원을 입력받아 SHORT_FRONT/LONG_FRONT를 분류한다. YOLO 점 좌표 자체는 바꾸지 않는다. 클린 스타트 60k 계열의 재사용 합성 DEV에서 치수 없음 S0 balanced accuracy 85.5254%, 치수 포함 S1 90.4141%, 치수 섞기 74.5329%였다. 이 수치만으로 실사·새 팔레트 일반화를 주장할 수 없다.
2. **치수 입력을 쓰는 점·선·자세 공동 모델.** 로컬 v5와 수정 v6에 K와 (W,D,H)를 입력하는 특징 변조 경로, 점 head, 선 head, 자세 초기값과 공동 solver가 있다. v6는 점/직접 선/Hough 세 군·3시드·각2000업데이트, 총18000업데이트까지 완료했다. 합성 진입 기준 미통과로 실사 단계는 실행하지 않았다.
3. **로스도 구현돼 있다.** v6는 점 좌표 SmoothL1, 선 역할 분류, 치수로 생성한 3D 코너의 자세 오차, 초기 자세·후보 순위 항을 사용한다. 허용 대칭 하나를 샘플 전체에서 공유한다. 치수를 PnP에만 전달하는 방식과 달리 신경망 특징에도 넣는다. 다만 원래 YOLO는 동결이고 새 head를 학습한 구조다.

## 완료된 결과의 해석

v6에서 Hough 대 새 점 전용의 주 지표는 시드별 0.95% 개선 / 7.94% 악화 / 5.85% 악화였다. 번호를 면제한 공간 배치 평균에서도 원래 YOLO 4.18px에 비해 새 점+Hough는30.88/27.47/32.14px였다. 이는 배포 지표가 아닌 공통 일대일 대응 진단이다. 치수 입력 경로가 있다는 사실과 정확한 점·자세 예측을 달성했다는 주장은 다르다.

이 결과는 치수 유무만 분리한 대조가 아니다. 따라서 치수가 해롭다거나 무용하다는 인과적 결론은 낼 수 없다. C4에 대한 학습 표본도 없어서 모든 팔레트 대칭을 검증했다고 할 수 없다.

## 처음 찾았던 translation/LC 로스와의 구분

`pallet_translation_loss_v1`은 pose 민감도를 학습 손실에 더한 별도 실험이다. 문서의 “미착수” 상태와 달리 A2b/CONTROL 각각5epoch와319장 평가가 이미 완료됐다. 그것을 이번에 사용자가 지칭한 치수 입력 모델로 단정하지 않는다. translation 전용 L_TR은 그 실행에 포함되지 않았다.

## 재개 전에 구분할 목표

- 치수로 기존 점의 앞면/가로세로 방향만 선택하는 목표는 S1 계열이다.
- 치수로 점 좌표와 최종 자세를 함께 개선하는 목표는 공동 예측 계열이다.

사용자가 논문 기여를 위한 팔레트 기반 아키텍처를 목표로 명확히 했다. 재개 판단은 [원문 대조 기록](shared-dimension-symmetry-review.md)을 따른다. [기여 후보 설계](pallet-dimension-architecture-proposal.md)는 원문 확인 전에 작성한 별도 잠정안이다. 이번 확인 중 새 모델 학습은 실행하지 않았다.

## 근거

- [클린 스타트 치수 입력 결과](../../../challenge/yolo_pose_one_model/spatial_concat_scratch/SPATIAL_CONCAT_60K_REPORT.md)
- [앞면 방향 선택기 정의](../../../challenge/yolo_pose_one_model/dimension_conditioning_probe/METHOD_SPEC.md)
- [v6 공동 모델 실제 결과](../../../data/pallet/results/pallet_point_line_decisive_v6/INTERPRETATION_KO.md)
- [치수 입력 경로](../../../scripts/research/pallet_point_line_decisive_v6/plpose_local/heads.py)
- [공동 로스 구현](../../../scripts/research/pallet_point_line_decisive_v6/plpose_local/losses.py)

원래 실험 결과와 GT, 체크포인트는 수정하지 않았다.
