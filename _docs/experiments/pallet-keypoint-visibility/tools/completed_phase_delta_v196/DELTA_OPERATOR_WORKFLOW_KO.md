# V196 추가 게시 준비

현재는 코드와 가상 검사만 완료했다. V196 과학 폴더는 동결되어 있으며 이 준비 도구는 실제 root entry 전까지 원본 목록·해시·저장소·과거 manifest를 읽지 않는다.

1. Root가 정확한 phase result, 독립 최종 보고서 QA, 최종 freeze를 binding한 inventory entry를 작성한다. `prepare_completed_phase_delta.py --entry <absolute-entry>`는 V196 안의 원문·표·그림만 목록화하고 모든 배열·weights·입력 cache를 제외한다.
2. Root가 실제 main OID와 BASE 및 V187..V195 9개 delta/chain/index, BASE 포함10개 closing plan, 현재 README 해시와 기존 copier를 고정한 build entry를 작성한다. `build_completed_phase_delta.py --build-entry <absolute-entry>`는 새 V196 delta와 인덱스·chain을 만든다. 2MiB 미만 text는 원 경로를 유지하고 큰 text는 기존 lossless gzip descriptor 정책을 쓴다. 기존 게시 경로는 바꾸지 않는다.
3. 별도 독립 byte auditor는 원본 SHA/size·gzip 왕복·physical100MiB 제한·8개 portable 페이지 실제 상대링크·credential 형식과 과거 경로 불변을 검사한다. 이미지·tensor·array·모델·평가를 decode하지 않는다.
4. Root가 완료/PASS 독립 QA를 바인딩한 push-plan entry를 승인하면 builder의 `--push-plan-entry`가 새 physical 파일만 strict480MiB plan으로 봉인한다. 실제 Git/push는 별도 검토된 runner가 수행한다.

기대 main은 준비 단계에서 null이다. 새 학습·추론·라벨·문턱·모델/데이터 선택·채택 및 전체 목표 성공을 뜻하지 않는다. 실패와 첫 시도는 보존하며 자동 재시도하지 않는다.
