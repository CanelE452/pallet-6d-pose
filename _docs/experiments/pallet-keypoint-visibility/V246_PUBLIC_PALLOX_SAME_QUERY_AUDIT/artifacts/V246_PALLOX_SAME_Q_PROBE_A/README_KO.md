# Pallox 원 q의 조건부 SOURCE 진단

실제 실행은 Root가 두 개의 별도 true entry/scope를 작성한 후 합니다.

`python -B probe_same_q.py --entry <Root entry>`

1. `stage=select`: 원 TRAIN44/352를 확인하고 mixed pair manifest에서 사전22 pair_id만 선택합니다. 제외28의 나머지 값과 선택행의 count/stat 값은 JSON 해석하지 않습니다. 출력 NPZ binding은 문자열만이며 배열 stat/hash/read는 없습니다.
2. `stage=probe`: 별도 승인을 받은22 TRAIN NPZ의 view별 네 Boolean 필드만 읽습니다. 저장 class_A/B는 읽지 않습니다. `construct_certainty`로 canonical cert를 만든 뒤 원 q 그대로 `query_truth`를 호출합니다.

Root scope의 metadata_whitelist는 자기 scope를 제외합니다. Entry의 metadata_whitelist에는 그 동일 목록과 root_scope binding 한 개를 더합니다. Stage2에는 선택 COMPLETE 및 그 metadata/whitelist 파일 binding도 추가합니다. Certificate whitelist는 Stage1 빈 배열, Stage2 정확22개입니다. false template는 실행 승인 없이 실패합니다.

352행과 입력U55를 보존하고 exact1pixel·radius2·strictEcore를 별도 기술합니다. 원 native 의미는 UNKNOWN, SELF는 None, native 라벨·학습 admission은 없습니다. 빈 교사와 미확정 pair는 UNKNOWN이며, 잘못된 schema/hash/대응/모순은 실패합니다. 기존 conditional 교사는 물리 gold가 아닙니다. 원 RGB/depth/SAM/pose/mesh와 모델을 읽지 않습니다.
