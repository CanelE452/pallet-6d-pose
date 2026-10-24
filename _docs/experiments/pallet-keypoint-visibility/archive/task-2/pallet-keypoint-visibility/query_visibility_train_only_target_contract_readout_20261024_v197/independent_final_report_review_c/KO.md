# V197 최종 보고서 독립 QA

완료/PASS. 생성된 TRAIN metadata·CSV·기존 A/B/root QA와 현재 보고서·바인딩을 대조했다. 원 NPZ/header/RGB/점수/모델/보호 평가 body를 열거나 새 이미지를 보지 않았다.

전체109=합성82+공개실사 weak27, known D218/E109와 native별35·26·13·22·6·7·0·0이 일치한다. 선택12=합성8+실사4이며288행 D24/E12/UNKNOWN252, 입력 U24(8.33%)와 원native8/고정qL을 보존했다. 입력 U24는 선택 실사96행 안의 값이며 전체109 가용성 비율로 확대하지 않았다.

도메인별 q→후보영역 중심 거리 및 enclosure 폭/높이의 최소·중앙·최대는 별도 scalar 계산으로 ROOT JSON·A 통계·표시표와 일치했다. canonical 제공82/실사UNKNOWN27, 실사12frames·1component와 native6/7 direct known0을 구분했다. 이 거리를 물리 키포인트 좌표 오차·보정 정답으로 부르지 않았고 원인을 확정하지 않았다.

실제 확인은 root00–07과 B08–11의12contact+12BEFORE문맥 union이다. 288개 crop 또는36개 문맥 전체를 개별 도구 확인했다고 하지 않는다. C는 저장된 view receipt의24개 PNG SHA binding만 metadata에서 대조했고 새 view/decode는0이다.

MD 상대링크 11개(두 MD 바이트동일), HTML 2개 링크와24개 고정 image 경로가 모두 존재하며 저장 manifest에 연결된다. HTML 본문 report text는 MD와 일치한다. Browser 렌더는 주장하지 않는다.

작은 변화·시각적 유사성·q와 부위 분리·3×3 표현은 미검증 가설로 남겼다. 기존 weak 감독을 native physical gold나 자연실사 성능으로 승격하지 않았고 새 fit/label/model/data/threshold 채택을 하지 않았다. 전체 목표는 미해결이다.
