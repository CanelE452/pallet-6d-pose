# V243 SOURCE scene-CF data-only trainer 준비

실제 특징·가중치·이미지 읽기와 학습은 아직 0입니다. Root가 데이터·독립 QA·시각 검토·특징 QA 및 사전 fit protocol을 묶은 true entry를 발급한 뒤에만 실행합니다.

```bash
python -B train_scene_cf_data.py --entry /absolute/path/to/ROOT_TRUE_ENTRY.json
```

BASELINE_V239_CANONICAL_CONTEXT는 기존 V239 TREAT의 checkpoint/log/CSV 바이트를 재사용하며 새 optimizer update는 0입니다. TREAT_SCENE_CF_MATCHED_DATA는 같은86054구조를 원seed17 초기값에서1000회 새로 학습합니다. 원PCG17 draw와 family8D8E 순서는 고정입니다. 각 family 마지막4 D의 own parent가 새PROOF_PASS일 때만 같은부모SHAM을 D자리에, COVER를 대응 E자리에 넣습니다. 실패하면 두 자연표본 자리를 모두 유지합니다. 성공pool에서 새 부모를 뽑지 않습니다.

원 q/L/q8/native/available/condition32와 두 SOURCE 교사 권한을 보존합니다. SYN saved witness0와 YCB associated-r(surface/nativeUNKNOWN)는 출력0/1로 분리됩니다. 새 데이터는 whole-part/native physical gold 또는 literal query-pixel label이 아닙니다. UNKNOWN−1는 loss에서 사용하지 않습니다. CF feature계산은 proof를 gate로 쓰지 않았으므로 트레이너가 별도PROOF_PASS/PAIR_TARGETS를 검사합니다.

원모델을 다시 학습하거나 새DEV로 checkpoint/threshold를 고르지 않습니다. 기존CSV는 새 최종checkpoint+로그 및 combined freeze를 쓴 뒤에 처음 해석합니다. 새DEV 출력도 그 뒤에만 만듭니다. 원SOURCE DEV는 노출·재사용된 자료이며 freshblind가 아닙니다.

가상검사4그룹은 실제자료 없이 배치 치환, loss routing, original14helper AST, seed17초기SHA/모수86054, 가상1optimizerstep을 확인했습니다. 실제DINO instantiate/actualtraining0입니다. 실제 실패는FAILED_EXECUTION.json으로 보존하며 자동재시도하지 않습니다.
