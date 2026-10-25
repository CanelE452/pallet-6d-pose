# V245 prospective matched plan 및 entry helper

새 특징·샘플 배열·PT·optimizer·DEV 출력은 읽지 않았습니다. 저장된 V239 draw1000과 V242 pair proof metadata만으로 계획을 실제 봉인했습니다.

- 원 draw SHA: `20a1fc74…`
- semantic plan SHA: `50e547b4c07afdee690727d895278ad7b82fe80eab52551dd00b9f5eb7a5d0e3`
- 예상 실제 ledger 바이트 SHA: `b2c1c3bff32981b2570f0330353c3c4f54fcfa4fb41d209b0dacfccfc1ac0dae`

SYN3974/YCB3940번씩 같은부모SHAM과COVER를 함께 치환합니다. 증명실패26/60번은 두 원자연D/E슬롯을 모두 유지합니다. 첫4D/E와 원q/L/U/native/split은 그대로이며 새RNG·성공pool선택은 없습니다. proofPASS pool은SYN4537/YCB830이고 실제draw고유부모는2630/815입니다.

Root가 late CF feature COMPLETE/독립QA를 받은 뒤 아래 helper를 실행할 수 있습니다.

```bash
python -B assemble_training_entry.py --config /absolute/Root_APPROVED_METADATA_CONFIG.json
```

helper는 metadata 및 SHA graph만 검사해 true training entry와 assembly receipt를 씁니다. payload를 stat/hash/decode하거나 trainer를 실행하지 않습니다. false config는 실행을 거부합니다. Root scope와 entry에는 봉인된 두 plan artifact와 semanticSHA가 들어갑니다.

원 V243V2 trainer는 수정하지 않습니다. Root는 실제 run의 source_plan_sha256 및 actual ledger 전체 바이트가 사전계획과 같은지 과학적 해석·모델 채택 전에 독립 비교합니다. 불일치는 무효이며 자동재시도하지 않습니다. 이는 SOURCE 사전계획이며 모델 성공이나 current 검증 결과가 아닙니다.

stdlib struct/list 계산을 frozen V243 plan_digest 함수와 full1000행에서 비교해 일치했습니다. JSON key순서/공백/newline까지 예상 actual ledger와 같습니다. 별도의 numeric feature/sample input 검사와 FIT-only optimizer 검사는 기존 runtime 계약을 따릅니다.
