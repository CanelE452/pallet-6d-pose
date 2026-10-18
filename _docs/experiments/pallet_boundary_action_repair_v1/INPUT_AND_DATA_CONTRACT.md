# 입력과 좌표 계약

기준은 main `c69f57e1fd26fcf268eb2f3cd0040e801b30f60b`의 원 결과다. 실제 입력 파일과 가중치의 동일성은 P0 inventory와 각 단계 seal로 연결한다. hash가 같다는 것은 파일 동일성이며 물리 정답 인증이 아니다.

- 실사 PRIMARY192는 원 TEST49 + REMAINDER143이다. CAL46은 별도 분석이다. 실패·미산출·복귀를 분모에서 제외하지 않는다. 새 독립 실사 test는 0장이다.
- 합성은 원 RGB1003장과 asset 분할 train768 / validation235를 그대로 사용한다. N3/SubPix와 카메라·등록 치수는 원 캐시에서 재사용한다. 새 렌더링·새 본체 학습·정답 수정은 하지 않는다.
- B는 원 real238 predictor packet과 원 RGB profile/mask bundle을 그대로 쓴다. profile의 처음부터 불가용인 상태와 후보 pose가 기존 지원 profile을 지지하지 못하는 상태를 구분한다. 원 group·reliability·분모는 고정한다.
- D1은 기존 N0 초기값·원 RGB·K·등록된 native XYZ만 내보내는 명시적 whitelist를 사용한다. 과거 실사 E0·history-only 후보·기존 상태·마스크·관측은 D1 upstream 입력에서 제외한다. 같은 자동 callable로 합성과 실사 관측·후보를 만든다. [상세 계약](INPUT_AND_DATA_CONTRACT_D1.md), [절차 비교](DATA_PIPELINE_PARITY.md).
- 합성 참조 pose·visibility·box·mask를 관측 생성에 넣지 않는다. 기존 sanitized provision의 asset별 축 등록은 재사용한 모델 좌표 규약이며 새 per-frame GT crop/pose 입력이 아니다. 이 이전 등록 규약과 backbone의 기존 asset 노출 한계는 남는다.
- 새 공통 경로의 실제 e 출력은 `E0_REGENERATED`로 부른다. 역사적 114/192의 `E0`를 덮어쓰지 않는다. N0↔E0 pair는 새 공통 bank의 실제 N0와 독립 실행한 E0_REGENERATED를 사용한다.
- 저장 `R_physical`은 물리 팔레트 축의 회전이다. `centroid`는 카메라 좌표의 물체 중심, 단위 m다. native 회전은 `R_physical @ Q.T`이고, projection은 `(native_XYZ - native_mean) @ R_native.T + centroid`에 원 K를 적용한다. 전체 위치오차는 중심 차이를 cm로 바꾼다.
- 원영상 raw pixel 좌표를 사용한다. 캐시의 letterbox/crop 해제는 재사용하며 새 GT 기반 crop을 하지 않는다. 표시용 비례 축소·RGB 비공개 사본은 측정값과 별개다. 그림의 crop/표시 변환은 gallery manifest에 기록한다.
- 실사 회전 평가는 원 C2의 I 및 Y(pi)만 사용한다. 직사각형 W/D의 90° 혼동을 정답 대칭으로 지우지 않는다. 합성은 원 행별 C1/C2 레이블 규약을 사용한다. GT 행렬을 유리하게 직교화하지 않는다.

L 입력은 기존 31개 signed-log scalar와 31개 validity bit, 같은 mean/max frame pool, 후보수, 명시한 행동역할 한 bit의 188차원이다. ID/session/material/참조 오차/oracle/사후 실패분류는 feature가 아니다. H0의 마지막 bit는 원 none 역할, H1에서는 is_N0다. 역할별 affine과 MLP는 같은 정보를 받는다. H1은 실제 수치 중복 pose를 한 행동으로 합치고 N0/E0 source alias를 보존하며, 90° 물리 차이를 중복으로 제거하지 않는다. 정규화는 해당 D의 train만 사용한다.

예측 producer는 실사 참조를 받지 않는다. 합성 참조는 별도 레이블·감사 프로세스에서만 읽는다. 모든 새 B/L 출력과 21개 모델 예측을 봉인한 뒤 evaluator가 실사 fixed reference를 처음 읽도록 gate를 둔다. 파일명 필터만으로 누출 차단을 주장하지 않고 whitelist·함수 입력·특징·분할·체크포인트를 함께 감사한다. 그림에서 참조를 쓰는 시점도 예측 봉인 뒤다.

원 실사 참조는 기존의 고정 기하 annotation이며 독립 실측된 물리 6D GT로 인증하지 않는다. 사람의 새 owner/경계/pose 판정은 학습과 추론에 추가하지 않는다. 모델의 그림 열람은 `model_review`이며 `human_review`가 아니다.
