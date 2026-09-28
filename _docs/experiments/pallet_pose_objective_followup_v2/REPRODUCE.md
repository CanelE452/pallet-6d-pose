# 집계 보고서 재생성

상태: **FINAL**. EXECUTION=PARTIAL_BUDGET; POSE_OUTCOME=MIXED; REPEAT=NOT_RUN.

이 문서의 명령은 학습을 추가하지 않는다. 기존 lock이 불일치하면 원본을 고치지 말고 중단한다. 실제 fit 재실행·새 seed·미완료 run 덮어쓰기는 별도 승인 및 budget 검토가 필요하다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.final_report
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m unittest scripts.research.pallet_pose_objective_followup_v2.test_final_report
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.final_audit
```

branch 분해 CPU 재계산 명령(일반 inference 또는 학습을 추가하지 않음):

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.branch_diagnostic
```

모든 선언된 결과·최신 그림·원장 정산이 끝난 뒤에만 `final_report --final`을 사용한다. 이 명령은 CPU private pose metrics로 C−A 및 동일 반복 seed의 recipe−원 baseline 집계를 재계산한다. JSON pointer와 SHA는 REPORT_NUMERIC_TRACE에 기록된다. 원장과 STATE는 실행 조정자만 수정한다.

완료된 평가의 frozen cache 확인/재집계 명령:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.eval_student score --cycle A_INPUT_OCCLUSION --material PLASTIC --seed 42
```

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.eval_student score --cycle BASELINE_REPEAT --material PLASTIC --seed 43
```

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.eval_student score --cycle B_COORDINATE_SUPPLEMENT --material PLASTIC --seed 42
```

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.eval_student score --cycle C_EXPOSURE --material PLASTIC --seed 42
```

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.eval_student score --cycle RECIPE_REPEAT --material PLASTIC --seed 43
```

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.eval_student score --cycle WOOD_APPLICABILITY --material WOOD --seed 42
```

## 이미 완료된 학습의 정확한 기록 명령 — 지금 추가 실행하지 말 것

아래는 같은 원 recipe와 immutable protocol을 재현하기 위한 역사적 entrypoint 기록이다. 현 예산에서 새 fit을 허가하는 명령 목록이 아니다. 완료 checkpoint와 실패/부분 로그를 보존하며, 독립 재현이 필요하면 별도 namespace·예산·승인을 먼저 정한다. 명목 추가 seed 명령은 실제 worker stream을 바꾸지 못했던 원 실행 그대로이며, 유효한 확률적 반복 방법으로 제시하지 않는다.

실행 `A_INPUT_OCCLUSION`의 [잠긴 protocol](cycles/A_INPUT_OCCLUSION/PROTOCOL.json), [사전 SPEC](cycles/A_INPUT_OCCLUSION/SPEC.md), [결과](cycles/A_INPUT_OCCLUSION/RESULTS_PLASTIC_S42.json).

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.student train --cycle A_INPUT_OCCLUSION --material PLASTIC --target RAW --seed 42
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.student train --cycle A_INPUT_OCCLUSION --material PLASTIC --target REF --seed 42
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.student parity --cycle A_INPUT_OCCLUSION --material PLASTIC --seed 42
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.eval_student infer --cycle A_INPUT_OCCLUSION --material PLASTIC --seed 42
```

실행 `BASELINE_REPEAT`의 [잠긴 protocol](cycles/BASELINE_REPEAT/PROTOCOL.json), [사전 SPEC](cycles/BASELINE_REPEAT/SPEC.md), [결과](cycles/BASELINE_REPEAT/RESULTS_PLASTIC_S43.json).

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.student train --cycle BASELINE_REPEAT --material PLASTIC --target RAW --seed 43
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.student train --cycle BASELINE_REPEAT --material PLASTIC --target REF --seed 43
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.student parity --cycle BASELINE_REPEAT --material PLASTIC --seed 43
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.eval_student infer --cycle BASELINE_REPEAT --material PLASTIC --seed 43
```

실행 `B_COORDINATE_SUPPLEMENT`의 [잠긴 protocol](cycles/B_COORDINATE_SUPPLEMENT/PROTOCOL.json), [사전 SPEC](cycles/B_COORDINATE_SUPPLEMENT/SPEC.md), [결과](cycles/B_COORDINATE_SUPPLEMENT/RESULTS_PLASTIC_S42.json).

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.loss_student train --cycle B_COORDINATE_SUPPLEMENT --material PLASTIC --target RAW --seed 42
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.loss_student train --cycle B_COORDINATE_SUPPLEMENT --material PLASTIC --target REF --seed 42
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.student parity --cycle B_COORDINATE_SUPPLEMENT --material PLASTIC --seed 42
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.eval_student infer --cycle B_COORDINATE_SUPPLEMENT --material PLASTIC --seed 42
```

실행 `C_EXPOSURE`의 [잠긴 protocol](cycles/C_EXPOSURE/PROTOCOL.json), [사전 SPEC](cycles/C_EXPOSURE/SPEC.md), [결과](cycles/C_EXPOSURE/RESULTS_PLASTIC_S42.json).

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.exposure_student train --cycle C_EXPOSURE --material PLASTIC --target RAW --seed 42
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.exposure_student train --cycle C_EXPOSURE --material PLASTIC --target REF --seed 42
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.student parity --cycle C_EXPOSURE --material PLASTIC --seed 42
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.eval_student infer --cycle C_EXPOSURE --material PLASTIC --seed 42
```

실행 `RECIPE_REPEAT`의 [잠긴 protocol](cycles/RECIPE_REPEAT/PROTOCOL.json), [사전 SPEC](cycles/RECIPE_REPEAT/SPEC.md), [결과](cycles/RECIPE_REPEAT/RESULTS_PLASTIC_S43.json).

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.exposure_student train --cycle RECIPE_REPEAT --material PLASTIC --target RAW --seed 43
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.exposure_student train --cycle RECIPE_REPEAT --material PLASTIC --target REF --seed 43
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.student parity --cycle RECIPE_REPEAT --material PLASTIC --seed 43
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.eval_student infer --cycle RECIPE_REPEAT --material PLASTIC --seed 43
```

실행 `WOOD_APPLICABILITY`의 [잠긴 protocol](cycles/WOOD_APPLICABILITY/PROTOCOL.json), [사전 SPEC](cycles/WOOD_APPLICABILITY/SPEC.md), [결과](cycles/WOOD_APPLICABILITY/RESULTS_WOOD_S42.json).

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.exposure_student train --cycle WOOD_APPLICABILITY --material WOOD --target RAW --seed 42
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.exposure_student train --cycle WOOD_APPLICABILITY --material WOOD --target REF --seed 42
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.student parity --cycle WOOD_APPLICABILITY --material WOOD --seed 42
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_objective_followup_v2.eval_student infer --cycle WOOD_APPLICABILITY --material WOOD --seed 42
```

마지막 명목 seed 대조·recipe 재실행·Wood 실행은 [run_closure.py](../../../scripts/research/pallet_pose_objective_followup_v2/run_closure.py)가 순차 관리했다. 로그 및 각 FIT JSON은 `data/pallet/results/pallet_pose_objective_followup_v2/closure_logs/`와 `cycles/<cycle>/`에 있다. 전처리 preflight/lock의 입력·구현 SHA는 각 protocol에 잠겨 있다. 이 orchestrator도 지금 다시 실행하지 않는다.

실제 검사 결과: 55 tests PASS, 5.730000 sec. source-file SHA와 원 pytest 출력은 [TEST_RESULTS](TEST_RESULTS.json)에 보존했다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m pytest -q scripts/research/pallet_pose_objective_followup_v2 scripts/research/pallet_material_selftrain_closure_v1/test_pair.py scripts/research/pallet_type_selftrain_v1/test_contract.py
```
