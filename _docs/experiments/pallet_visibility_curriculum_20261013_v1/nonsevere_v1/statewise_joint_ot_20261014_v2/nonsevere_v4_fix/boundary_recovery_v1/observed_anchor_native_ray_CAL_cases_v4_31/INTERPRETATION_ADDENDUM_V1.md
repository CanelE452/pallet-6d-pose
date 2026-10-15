# v31 CAL3 재순위 해석 보충

기존 `REPORT_KO.md`, 소스, PLAN 및 `RERANK_SEAL.json`은 수정하지 않았다.

`previous_selected_rejections`는 기존 선택 선을 생성한 **원 ray/chain reference별** admission 실패를 뜻한다. 같은 실제 지지점이나 같은 TLS 계수가 다른 q 앵커 reference에도 보관돼 있으면 그 대안이 살아남을 수 있다. 따라서 이 장부를 해당 물리 edge 또는 실제 픽셀 전부가 제거됐다는 증거로 읽지 않는다. 여기서는 그러한 alias의 잔존 수를 별도로 측정하지 않았다.

표의 후보 수는 admission 후 dedup된 unique line 후보 수다. `rejected`는 dedup 전 raw node 검사 건수라 서로 분모가 다르며, 두 값을 직접 같은 탈락 비율로 나누지 않는다. 선택 선 수가 6→6, 9→9, 4→4라는 사실도 선의 identity·모든 실제 지지점·계수가 같다는 뜻은 아니다. 구체 선택 이름과 원 provenance는 각 graph와 RESULTS 원장에 있다.

세 사례는 원래 고정된 CAL 진단 집합이며 새로운 holdout이나 전체 성능 결과가 아니다. 이번 실행은 실제 cached 선만 사용했고 새 RGB·참조·라벨·posefit·정밀 KP 채택은 없다. 새 population 잡음 모델도 아직 없으므로 기존 v29/v30 noise를 그대로 검증된 것으로 전이하지 않는다.
