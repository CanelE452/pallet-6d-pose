# V245 metadata entry·저장 결과 준비

현재 CODE/metadata 준비이며 새 특징·원 samples·배열·CSV·가중치·RGB·GT는 열지 않았습니다. 기존 V242 checker와 이전 C 준비물은 수정하지 않았습니다.

Root가 feature COMPLETE를 보낸 뒤 FEATURE_QA_ASSEMBLY_CONFIG_FALSE.json의 flag·future binding·QA true-scope를 채워 다음을 실행할 수 있습니다.

```bash
python -B assemble_entries.py --config ROOT_FILLED_FEATURE_QA_ASSEMBLY_CONFIG.json
```

helper는 metadata/CODE SHA만 확인하고 새 own assembly 하위 폴더에 실제 QA entry를 씁니다. payload16 leaf는 binding 문자열만 조립하며 stat/hash/header/body를 열지 않습니다. sharing_QA는 **새 CF COMPLETE의 CANONICAL_CF_SHARING_QA.json**이며 old input QA35d156 또는 old packet QA733dce로 대체하지 않습니다. 실제 특징 검산은 Root의 별도 숫자 접근 허가 후 기존 verify_features.py로 실행합니다.

SOURCE fit 완료 뒤 report mode는 COMPLETE·freeze·global feature QA·baseline reuse·CF feature QA와 사전 plan SHA를 metadata로 확인합니다. baseline은 과거 final1000의 byte 재사용으로 새update0이고 새 treatment만1000입니다. 결과 config는 saved-output reading=false로 남겨 별도 Root CSV scope를 요구합니다.

report_math.py는 고정 산술 모듈 d8eef669의 byteexact 복사본을 쓰며 arm/update metadata만 바꿉니다. natural7617과 CF10812를 섞지 않고 family0/1/FIT/DEV를 유지합니다. q/r sidecar는 이 준비에 없으므로 거리 strata를 새로 추정하지 않습니다. fixed +3pp/−2pp/YCB E50%/SYN D·E90%를 계산할 준비일 뿐, current admission·모델 채택·새 설계/선택 권한은 없습니다. 실제 metric read·실행은 Root 완료 통지와 명시 scope 이후입니다.
