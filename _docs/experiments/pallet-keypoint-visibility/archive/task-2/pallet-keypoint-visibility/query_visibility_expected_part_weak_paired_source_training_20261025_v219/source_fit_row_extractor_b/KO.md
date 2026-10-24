# SOURCE FIT 행 추출기 준비

현재는 코드 준비만 완료했습니다. 원 NPZ의 stat·hash·헤더·배열 접근은 0이며 실제 실행은 root entry 뒤에만 가능합니다.

원 A metadata에 C의 `fit_mask`를 적용한 인덱스와 A의 `domains[domain].fit_indices`가 완전히 같아야 합니다. TRAIN·nominal·계보 확인·nonDEV4만 남기며 U로 순위를 정하지 않습니다. SYNTH selected7040은 point7040 전체와 원 META6·own-image/anchor 계보가 같아야 합니다.

여섯 SOURCE ZIP의 SHA/size를 스트리밍으로 먼저 확인합니다. INPUT8/TARGET5의 NPY 헤더가 C-contiguous·nonobject·원 dtype/shape인지 확인한 뒤 선택 행만 `np.frombuffer`로 해석합니다. 제외 행은 seek/압축 해제 중 바이트로 통과·폐기될 수 있으므로 **제외 행 바이트를 전혀 읽지 않았다는 주장은 하지 않습니다.** 제외 행을 수치 배열로 만들거나 손실·정규화·forward에 쓰지 않습니다. `np.load(fullmember)`는 없습니다.

출력은 새 `source_fit_rows_v1`의 6 NPZ, 3 filtered metadata JSONL, hash proof/census/COMPLETE입니다. 기존 q/L/native/U·global index·targets·context는 바꾸지 않습니다. metadata에는 원 source 행 인덱스만 추가합니다. `COMPLETE.old_caches`는 기존 runner에 그대로 연결할 수 있습니다.

가상 ZIP에 HOLD/FIT/DEV/계보불명 sentinel을 넣어 선택 행만 해석, U 유지, object/Fortran 헤더 거부, false entry와 nofollow/V112 선검사 3그룹을 확인했습니다(`83d1e7`, exit0). 실제 소스·모델·GT·보호 평가자료 접근이나 학습은 없습니다.

[실행 entry 스키마](ACTUAL_ENTRY_INTERFACE.json) · [승인 false 템플릿](ROOT_ACTUAL_SELECTED_SOURCE_FIT_ROW_EXTRACTION_ENTRY_FALSE_TEMPLATE.json) · [가상 검사](GENERATED_INVENTED_SELECTED_ROW_STREAM_CHECKS.json)
