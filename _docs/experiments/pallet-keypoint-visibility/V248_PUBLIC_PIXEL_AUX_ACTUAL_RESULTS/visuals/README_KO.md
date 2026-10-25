# V248 고정 공개 SOURCE AUX 시각화

[HTML gallery](GALLERY.html) · [16 slot overview](OVERVIEW_FIXED16.png) · [정확 좌표·SHA·저장 예측](CASE_DATA.json) · [표](CASES.csv)

새 head output 전에 Root가 고정한16개 slot/12개 record/빈4개 stratum을 그대로 보존했다. 선언된RGB8장만 SHA확인 후 열었고 V90 pure crop으로 재생성한12개 RGB/valid SHA가 feature 기록과 일치했다. raw96 PNG 저장·재해독 뒤 CHW SHA도12개 모두 동일하다. 원native352 crop은 읽거나 변경하지 않았다.

양final1000 동결 barrier metadata를 확인한 뒤 별도로 허가된 SOURCE AUX CSV2의 저장 logit/예측만 붙였다. checkpoint/log 수치나 모델은 읽지 않았고 신규forward0이다. 4개 pair board를 실제 도구로 열어 전체문맥·표식·raw crop·확대·빈 slot 및 예측 표시에 겹침이 없음을 확인했다. HTML 브라우저 렌더는 주장하지 않는다.

V/E는 conditional SAM+mesh+depth pixel teacher의 권한이다. native 역할·SELF는 UNKNOWN이며 static-camera/calibration/depth-grid 가정을 상속한다. 이 q는 새teacher-selected AUX 좌표다. 조건부 학습사례를 원initializer/native 성공이나 독립 일반화로 해석하면 안 된다. 새human라벨·학습admission·모델선택은 만들지 않았다.
