병렬 실행 경로에서 구현을 막는 worker binding 문제는 발견하지 않았습니다. spawn process마다 모델과 mask provider를 따로 보유하며, 같은 process 안에서 SAM의 현재 영상 cache가 다른 frame과 동시에 공유되지 않습니다. Master가 future와 frame ID의 일치 및 중복을 검사하고, 원래 CAL46/TEST49와 별도 crop 세 장의 순서로 98개 결과를 다시 만듭니다. 일부 frame이 실패하면 전체 prediction seal을 만들지 않습니다.

Initializer/worker/predict는 scorer를 호출하거나 GT target을 열지 않습니다. Prediction 전용 field만 runtime에 들어가며 좌표 GT의 crop 변환은 별도 postseal score에 남아 있습니다. Source binding에는 새 parallel runner와 Jacobian module이 포함되고, 실행 source snapshot도 SHA를 확인하여 복사합니다. 현재 지시대로 v3 모델·parameter·checkpoint·원래 prediction 입력을 고정하는 조건에서 worker loader의 모델 binding 확인과 master의 source/crop 확인이 연결됩니다.

수학적 문제도 같습니다. 모든 P3P 시작점과 전역 회전, 6D 자세 자유도, 고정 B/I 관측, 각 B/I의 두 보호 조건과 모든 O/T 개별 제약을 유지합니다. Jacobian source는 원래 projection과 radial pseudo-Huber, depth 및 영역 제약의 값·순서에 대응합니다. 더 정확한 미분 때문에 지역 optimizer의 수렴 결과는 달라질 수 있으므로 중단된 numeric 실행을 새 실행의 결과로 바꾸면 안 됩니다.

기존 numeric evaluation_v1은 scoring 전에 중단한 상태로 보존하고 evaluation_v2를 별도로 seal해야 합니다. 부모 agent가 보고한 focused 테스트 31개와 source 검토는 정확도·O 일반화·physical T의 검증을 대신하지 않습니다. 전체 98개 prediction seal 뒤 TEST49 전체 지표와 실제 crop의 physical T 확인을 끝내야 성능 평가가 완료됩니다. 이 검토에서 TEST GT 좌표나 severe/FINAL 행을 읽지 않았습니다.
