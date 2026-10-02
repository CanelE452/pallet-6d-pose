# N3 v3 실행 작업표

상태는 코드 존재와 실제 결과 완료를 구분한다. 최종 수치는 `VERIFY_RESULTS.json`의 hash 검증을 통과한 경우에만 원고 표에 넣는다.

| 계약 | 목적 | 행동 | 현재 근거/출력 | 결과가 나쁠 때의 처리 |
|---|---|---|---|---|
| A | DEV319 회귀와 pose P90 | `RESCORE` | `REUSE_RESULTS.json` A, 고정 319/311/2499/2445 회귀 통과 | 그대로 보고 |
| B | N0/N1/N2/N3 절제 | `RESCORE` | 기존 seed별 원시 좌표를 현재 8-corner evaluator로 재집계 | 무차이·악화 그대로 보고 |
| C | 재질/형상 | `RESCORE` | plastic 194, wood 125; DEV319 rectangular | 각 실제 분모 유지 |
| D | 가림/가시성 | `RESCORE + NEED_REVIEW` | 검수 128장: clean 29, moderate 20, severe 79; 나머지 191 unclassified | 미분류를 임의 등급화하지 않음; visibility는 `x` |
| E | cap·손상·복구·2D/3D paired 변화 | `RESCORE` | 기존 seed별 좌표와 1% cap 무결성 재집계 | 손상 수도 포함 |
| F | GREEN0918_119 square | `REUSE + INFER` | R0/N2 raw 재사용; OLD_P/N3 및 DOPE/ResNet N3 고정 추론 | 2D 악화도 보고; 독립 6D 참조는 `x` |
| G | DOPE/ResNet 동일 N3 | `TRAIN + INFER` | 각 3 seed×6,000; fit·optimizer·RNG receipt | 성능과 무결성을 분리해 보고 |
| H | D/L/PoseFix | `RESCORE` | 기존 raw·checkpoint SHA 연결, 동일 evaluator | 없는 방법을 재학습하지 않음 |
| I | 기존 update/self-training 대안 | `RESCORE + BLOCKED_CONTRACT` | 안전 HELDOUT128만 재집계; DEV319 공통 노출 계약은 `x` | 새 self-training 금지 |
| J | runtime/parameters | `BENCHMARK` | fixed 26, warmup20, repeat5, seed1; 같은 RTX | historical YOLO 수치를 덮어쓰지 않음 |
| K | 기록 리프터 | `INFER` | 4 usable sessions, fixed 1 s sensor-time 846 frames | 독립 GT 부재로 정확도 `x`; coverage/jitter만 기술 |

## 실제 실행 진입점

```text
python -m scripts.research.pallet_n3_completion_v3.preflight
python -m scripts.research.pallet_n3_completion_v3.train smoke {dope,resnet18}
python -m scripts.research.pallet_n3_completion_v3.train train {dope,resnet18}
python -m scripts.research.pallet_n3_completion_v3.selection calibrate {dope,resnet18}
python -m scripts.research.pallet_n3_completion_v3.inference {dope,resnet18} {DEV319,GREEN0918_119}
python -m scripts.research.pallet_n3_completion_v3.evaluation {dope,resnet18} --predictions <path>
python -m scripts.research.pallet_n3_completion_v3.square {dope,resnet18}
python -m scripts.research.pallet_n3_completion_v3.square_yolo all
python -m scripts.research.pallet_n3_completion_v3.runtime measure all
python -m scripts.research.pallet_n3_completion_v3.lifter_run {plan,run,verify}
python -m scripts.research.pallet_n3_completion_v3.reuse
python -m scripts.research.pallet_n3_completion_v3.integrity verify
```

최종 `run.py --help`와 stage receipt가 검증되면 위 개별 명령과 함께 재시작 가능한 통합 진입점으로 기록한다.

