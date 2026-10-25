# V242/V243 게시 후보와 검산 순서

이 목록은 게시 실행 권한이 아닌 CODE/metadata inventory입니다. 원본이나 clone을 수정하지 않았습니다.

V242 데이터 5,406개는 전부 시도됐고 5,367개 proof PASS, 39개 미생성으로 보존됐습니다. 독립 all5406 pixel QA와 fixed4 gallery는 실제 완료됐습니다. gallery의 full/native·crop·card PNG는 COMPLETE에 선언된 원 byte binding으로만 후보에 넣었으며 이 inventory에서 영상·배열을 읽지 않았습니다. q/r diagnostic은 합성 footprint 관측이지 실사 literal-q GT 또는 detector error 측정이 아닙니다.

CF feature producer·packet QA와 V243 trainer는 CODE 준비 상태입니다. 실제 CF 특징 생성/새 학습/성능 개선/현재셋 검증으로 과장하지 않습니다. current admission과 model adoption은 false입니다. SOURCE 데이터의 prospective feature/trial admission과 실사 모델 admission을 구분합니다.

Root builder는 CANDIDATE_FILES.json의 exact leaf만 사용하고 source-map에 원 SHA/size를 보존합니다. PAIR_METADATA JSONL을 gzip으로 실으면 압축 byte SHA/size와 해제 원문 SHA/27,613,654B를 함께 기록합니다. small numeric 6개는 명시 Root copy admission 뒤에만 포함합니다. 나머지 큰 5개 cache는 EXCLUSIONS의 원 binding만 남기고 파일 자체 stat/read/hash/copy를 하지 않습니다. 미래 feature/model caches도 자동 포함하지 않습니다.

게시 검산은 manifest exact leaf SHA/size, source copy byte equality, 압축본 decoded SHA/size, README/HTML 내부 상대링크 closure 및 제3자고지를 확인합니다. 외부 literal bindings는 문서로만 보존하며 따라가지 않습니다. V1 실패/준비 수리/가상 harness 실패 및 원 code를 모두 함께 보존합니다. 기존 V241 39파일과 prior history prefix는 변경하지 않습니다. Root의 V244 결과 추가 뒤 최종 manifest를 별도로 닫아야 합니다.

Git/복사/네트워크·모델/데이터/architecture 선택은 이 작업에서 하지 않았습니다. main base는 Root가 준 8c2ae5c47이며 C는 clone을 열지 않았습니다.
