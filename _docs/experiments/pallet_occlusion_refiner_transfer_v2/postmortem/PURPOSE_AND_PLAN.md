# POST-E2 원인분해 계획

[확인] 시작 HEAD: `ceb323452a77ed8cebb152f6ab2626d018bf7e38` (`main`). 사용자 지시문 `528092bb-edba-416f-ba41-a9d962b95a28/pasted-text.txt`을 따른다. 새 학습·새 pseudo-label·threshold 변경·commit/push는 하지 않는다.

[확인] 기존 PRIMARY_OCC96 이름의 실제 93장/713코너를 고정한다. 저장 예측과 whole-object 대칭 분기를 재검산한 뒤 canonical GT 코너 identity로 D1 전이, D2 오류 구간, D3 가시성, D4 oracle, D5 그룹 분포를 계산한다. 실패 검출/매칭은 원래 대각선 벌점과 분모에 남긴다.

[확인] 기존 annotation의 visibility 숫자만으로 manual/external GT라고 주장하지 않는다. 기존 visibility amendment 및 auto queue를 함께 조사한다. 수동 `occluded`도 외부/자기 가림을 구분하지 않으면 UNKNOWN subtype으로 남긴다. 기존 geometry-derived self/visible은 별도 proxy 집계한다. 새 사람 판정이나 GT 생성은 없다.

[확인] D4 primary frame oracle은 평균 코너 오차 최소 후보를 프레임 단위로 선택한다. 과거 E1의 PCK 우선 oracle과 선택 목적이 다름을 표시한다. 코너 oracle은 동일 canonical GT identity에서만 secondary upper bound로 계산한다. ties는 고정 후보 순서로 결정하며 동률 빈도를 별도 보고한다.

[확인] D5는 통계만 작성한다. 사후 feature 차이는 selector 가능성의 가설이지, GT 없이 후보 선택이 검증됐다는 근거가 아니다. route 하나와 필요한 경우 보조 route 하나만 제안하며 실행하지 않는다.

[확인] 변경은 이 postmortem 디렉터리에만 추가한다. 기존 입력 파일과 checkpoint SHA를 전후 비교하고, 모든 보고 숫자를 JSON/CSV와 검산한다.
