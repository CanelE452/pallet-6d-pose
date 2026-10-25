# V245 SOURCE 저장 결과 report driver 준비

현재 CODE/false-config 준비이며 실제 CSV·feature·배열·모델·원영상·GT는 읽지 않았습니다. 기존 C helper와 순수 산술3개는 SHA/byteexact 복사했고 AST4개만 확인했습니다. 새 toy/workflow 실행은 없습니다.

Root가 두 final/freeze와 actual fit QA를 확인한 뒤 ROOT_ENTRY_FALSE_TEMPLATE.json을 복사해 `approved_saved_SOURCE_scene_CF_report=true`로 채웁니다. root_scope 역시 같은 true flag를 요구합니다. fit_QA는 completed/PASS와 exact fit_complete binding을 가져야 합니다. metadata_whitelist에는 fit COMPLETE의 freeze/global featureQA/baseline reuse/CF featureQA 및 사전 plan을 명시합니다. CSV_whitelist는 natural2개와 CF1개만 허용합니다. 배열·PT·모델 입력은 whitelist에 없습니다.

```bash
python -B report_saved_source.py --entry ROOT_ACTUAL_ENTRY.json
python -B report_saved_source.py --verify-saved results_v1
```

산출: family별 자연 CM/P/R/U·balanced recall CSV/JSON, paired 자연행, 별도 CF 진단행, 한글 Markdown·HTML, manifest·COMPLETE입니다. --verify-saved는 own 결과의 local leaf만 SHA/산술 재생하며 원 absolute bindings를 따르지 않습니다. HTML은 local table만 있고 브라우저 render를 수행했다는 주장은 없습니다.

자연7617/원masked7423/CF10812를 별도로 유지합니다. baseline은 과거 final1000의 byte 재사용으로 새update0, treatment는 새1000입니다. SYN intended-native witness0 / YCB associated-r AUX(native UNKNOWN)는 별도 과업입니다. SOURCE DEV는 노출·재사용되어 fresh blind가 아니며 고정 gate 산술은 현재 팔레트 성능·채택 결론이 아닙니다. current_admission=false/selected_arm=null을 유지합니다.
