# 기존 annotation.py의 키포인트·PnP 흐름 복원

기본 리프터 창이 기존 annotation.py의 update_pose와 실제 카메라 K를 사용하도록 복원했다. 키포인트 4점 이상 입력 후 PnP 윤곽이 표시되며, 원래 G/F 자동 채움과 저장, M 자세 조작, t 선 교점, x 외삽, c 중심점을 사용할 수 있다. 8개 상태 검수나 C 확인을 먼저 요구하지 않는다. 공유 annotate.py 파일은 수정하지 않았다.

실행 중 입력된 0/1/2/4 네 직접 클릭을 초안에서 보존하고 재실행했다. 실제 사용자 화면에서 같은 좌표와 PnP 윤곽이 함께 표시되는 것을 확인했다. CLI는 실제 화면에 직접 클릭·가림 확인·G/F/M 조작 입력을 대신하지 않았다. 창 재실행에는 주석을 생성하지 않는 저장·재열기 키만 사용했다.

![실제 사용자 클릭이 보존된 native PnP 화면](../images/lifter_native_pnp.png)

G는 보조점을 채워 현재 프레임에 저장하고 F/S는 키포인트·PnP를 저장한 뒤 다음으로 이동한다. M의 기본 이동·회전 키를 원래 dispatcher로 전달한다. 수동 직접 참조는 별도 record에 보존하고, G/M으로 생성한 좌표는 그 참조를 덮어쓰지 않는다. source와 extrap_mask를 유지한다. 생성점의 위치를 직접 클릭해 수정했을 때만 그 점을 새로운 직접 클릭으로 기록한다. 복구 파일은 모든 9개 editor point, source, mask, mode와 locked_pose도 별도로 기억한다.

보조 출력은 원본을 보존하는 별도 `*.PNP_ASSISTED.json`에 native annotation 문서와 직접 참조를 함께 저장하며 evaluation_use=false와 independent_reference=false를 표시한다. 실제 가시성 검수·검수 이력 확인 전에는 보조 출력을 완성된 사람 참조로 승격하지 않는다. 키포인트·PnP 저장 진행은 NATIVE_PNP_PROGRESS.json, 가시성 승인 진행은 기존 annotations_in_progress.json에서 별도로 계산한다. 외부 평가 예측이나 박스는 표시하거나 읽지 않는다.

Shift+C는 이후 가림 제안 확인, Shift+T는 저장·재실행이다. 소문자 c/t는 원래 중심점/선 교점으로 유지한다. 사용자 제외 5장, 원래 120/24 계획, 기존 3,030점 정적 상태 입력과 독립 T/R 참조 부재를 보존한다.

검증은 임시 합성 자료에서 실제 PnP를 실행했다. 관련 48개 테스트가 통과했으며, 새 8개는 실제 K로 네 점에서 pose 생성, G/F/S 저장, M 조작, 직접/생성 좌표 분리, 외삽·선 교점, 생성점 직접 수정 및 재실행 복구를 검증한다. 실행 명령·해시는 VALIDATION.json에 있다. 새 학습·모델 추론·실제 제어·push·업로드·PDF는 0회다.
