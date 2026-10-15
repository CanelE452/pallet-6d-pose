# 표시 metadata에 관한 보충

최초 full-frame 패널은 부모 예측행의 W/D family를 표시했다. 확대본의 E `WD=None`은 legacy `actual_pose` 내부에 family 키가 없어 compact PROCESS_DATA에 그 항목이 보관되지 않았음을 뜻한다. 부모 예측에서 W/D가 없었다는 뜻이 아니다. 새로운 네 방법의 family는 `actual_pose.WD_family`에 보존돼 있다. 확대본은 저장된 물리 회전·모델 중심·기하로 그렸으며 이 family 표시 공백은 wire나 저장 오차를 바꾸지 않는다. 정확한 E family는 최초 full-frame 패널을 함께 보면 된다.

최초 전체 그림과 확대본의 참조 wire는 ROOT 승인 후 동일한 고정 평가 참조를 마지막 별도 패널에만 그린 것이다. 참조로 query·보정·PnP·optimizer를 실행하지 않았다. 전체 그림 뒤 확대 단계는 이미 저장된 PROCESS_DATA와 복사한 RGB만 사용했고 참조파일을 다시 열지 않았다.
