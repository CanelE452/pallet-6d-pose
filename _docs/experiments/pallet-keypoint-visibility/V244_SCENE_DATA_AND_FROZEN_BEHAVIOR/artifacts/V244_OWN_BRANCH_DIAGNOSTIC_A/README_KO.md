# V244 frozen SOURCE 내부 분기 진단 준비

현재 actual checkpoint/feature/sample payload 읽기·forward·학습은0입니다. Root의 별도 exact true entry 뒤에만 실행합니다.

```bash
python -B diagnose_own_branches.py --entry /absolute/path/to/ROOT_TRUE_ENTRY.json
```

같은 frozen TREAT의 ownbase와 ownresidual을 더해 total을 만들고, 기존 all7617 TOTAL을 CPU replay atol=rtol1e-5 및 예측부호 exact로 재현해야 COMPLETE를 씁니다. logit0/logit1은 두 family의 별도 binary 출력이며 argmax하지 않습니다. 별도 학습한 CONTROL은 이 TREAT의 base가 아닙니다.

표에는 target-aligned residual, ownbase→total 문턱교차와 회복·악화를 기록합니다. family×원FIT/DEV×D/E를 유지하고 원SOURCE DEV는 이미 노출·재사용됐음을 밝힙니다. 잔차 부호만의 혼동표는 추가항 진단이며 독립 훈련·보정 모델이 아닙니다. 수치분해를 이미지/객체/가림의 원인이나 입력 식별불가능성의 증명으로 해석하지 않습니다.

가상4그룹 PASS: 실제자료 없이 random frozen head/가상logits로 additive sum, family단일출력, 회복·악화·문턱, 기존1e-5 재생조건 및 false-entry와 금지경로를 검산했습니다. V239 pure helper15개 AST동일입니다. 실제 optimizer/DINO/원RGB/GT/current/held 읽기0, 자동재시도0입니다.
