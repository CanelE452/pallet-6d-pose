# 191.31cm 오차 사례만 확대

`wood_night_01:031426`의 실제 사진이다. 왼쪽 주황색은 V32 SOURCE+C의 저장된 예측, 오른쪽 초록색은 이미 보관된 평가 참조다. 위치 오차는 191.3134cm, 회전 오차는 82.3232°다. V33 결과를 나타내는 그림은 아니다.

![191.31cm 예측과 평가 참조](191cm_prediction_and_reference.png)

화면상의 픽셀 거리가 아니라 카메라 좌표계의 3D 중심 위치 오차다. 저장 예측의 깊이는 약 5.51m, 참조의 깊이는 약 7.36m다.

사용자가 이 사례만 보여 달라고 요청해 사후 설명용으로 생성했다. 새 PnP·키포인트·선 탐색·참조 추정은 수행하지 않았다. 기존 관측/초기 모델로 정한 동일 ROI와 이미 저장된 투영 좌표만 사용했다. 예측이나 매개변수 선택에 사용하지 않는다. 생성 소스는 최초 표시 후 보존했으며, 재현 시 기존 그림을 덮어쓰지 않는다.

[원본과 저장 수치](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_full_result_process_v4_32_r2_v1/case1/PROCESS_DATA.json) · [재현 소스](../../../../scripts/research/pallet_visibility_curriculum_20261013_v1/render_single_191cm_case_v4_32.py)
