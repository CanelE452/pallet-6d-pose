# 독립 산술 검산의 범위

`independent_check.py`는 이번 `kernel.py`·`replay.py`와 기존 selector를 import하지 않는다. 표준 라이브러리와 NumPy만 사용한다. 기존에 봉인된 neutral package, 한 번 고정한 원 RGB 관측, 실행 산출물을 읽고 아래를 별도 계산한다. 실제 통과 여부와 최대 차이는 실행 receipt 및 `VALIDATION.json`을 따른다.

- 원351개 후보의 native XYZ/R/t/K를 직접 투영하고, 별도 monotone-chain 구현으로 **모델 쪽** convex hull을 만든다. 후보의 Q와 중심/물리 회전 연결도 검사한다. 관측 쪽 hull은 만들지 않는다.
- 각 열린 구간에서 `ceil(length/spacing)` midpoint와 `(length/spacing)/n` 질량을 별도 생성한다. 끊긴 구간을 잇지 않는다. 원좌표/캔버스/중복 및 collinear 중복 길이/zero-length/닫힘을 검사한다.
- 각 표본에서 모든 **유한 모델 선분**으로의 거리를 loop로 계산한다. 선분 밖 수선은 끝점으로 제한한다. 원 observed 방향의 거리·강건 손실과도 대조해 원 sigma/delta/lambda/half 의미를 확인한다.
- 원 full B/I·partial grouped normal·활성 prior의 이미 가중된 합을 보존한다. 원 저장 방향 비용을 사용한 두 단방향 대조와 새 질량/평균 손실을 별도로 계산해351×4 점수 전량을 대조한다. r이 같은 frame/stage/family의 후보에 따라 변하지 않는지도 검사한다.
- 별도 안정 정렬로 원 insertion order와 `(objective, initialization)` 규약을 적용한다. G10의 accepted family winner를 비교한 뒤 선택 family의 G13만 읽는다. 24개 최종 pose뿐 아니라 전체 pool 순위와 stage winner를 대조하며, G10/G13 점수를 섞지 않는다.
- 새 입력을 제거했을 때 모든 후보의 scalar와 세 신규 경로의 전체 순위·출력이 원 e와 정확히 같은지 검사한다. 실제 입력이 들어간 성공 반례5·6의 후보 비용이 바뀌었는지도 검사한다. 비용 변화와 성공 여부는 별개의 검사다.
- 예측 artifact SHA를 먼저 확인한 뒤 참조를 연다. 별도 중심 오차와 physical/C2 회전을24개 동일 후보에서 계산하고, 원 성공 기준 `T≤5cm AND R≤5°`를 적용한다. 90도 동등 회전은 추가하지 않는다. 18개 원 e 대비 전이, visible 대 질량 대조의6개 전이, 전체6장의 평균·표본SD·최대를 검산한다.

원 숫자·새 거리·질량·점수·오차는 절대 허용치 `1e-9`, `rtol=0`으로 비교한다. ID/family/tie/상태/메타데이터는 정확 일치를 요구한다. 무입력 scalar는 근삿값 대신 정확 원값을 요구한다. hull의 cyclic vertex order가 달라도 같은 유한 기하 거리인지 검사하며, 다른 pose의 arc index를 같은 물리 표본으로 빼지 않는다.

합성 geometry 검사는 연구 후보 생성이 아니다. 내부 점을 제거하는 model hull, 무한 직선과 다른 끝점 거리, 열린 gap을 넘지 않는 midpoint/질량, 범위 밖·닫힘·zero-length·중복/겹침 거부를 확인한다. 실제 입력·후보 평가와 구분해 receipt에 기록한다.

이 검산은 **산술과 출처 계약**을 확인한다. 모델 제작자가 팔레트 외곽을 정확히 표시했다거나, 참조·K·등록 직육면체가 독립 물리 정답이라는 인증은 아니다. 별도 모델 문맥에서 원 RGB만 보고 작성했어도 기존 사례의 사후 선정 한계와 물리 모호성은 남는다. 표시점/보간 표본을 독립 데이터셋 분모로 세지 않는다.
