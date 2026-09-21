# Crop support / candidate generation 원인 분리

C0: R0 bbox center +1.25, 기존 FULL control. C1 PRIMARY: 같은 center +1.50만 변경, real/source/inference 공통값. C2 FOLLOW-UP: bbox와 valid point union, outlier 위험. C3 CREATIVE FOLLOW-UP: 기존 치수/PnP extent union, pose 실패·인과혼합 위험. C2/C3 실행 금지.

E0에서 기존 도달불가 중15개 이상 새 도달가능 또는 hard unreachable 비율50% 이상 감소할 때만 FULL_EXP150을 기존 FULL과 같은253장·순서·원영상 pseudo/가림·source·초기화·seed로300step 한 번 학습. 아니면 STOP. Decoder·loss·해상도·검출기·학생 학습·최종 모델 변경 없음.

GT/reference는 E0 기하 진단 및 frozen 출력 사후평가에만 사용. 사람검토 pending이므로 external occlusion 주장 금지. 최신 첨부의 관행적 push금지보다 사용자 상시 MD+이미지 push 방침 적용.
