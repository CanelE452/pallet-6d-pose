# V233 저장 SOURCE 결과 분석 코드

현재 CODEONLY. 미완료 V1 성능 출력·모델·가중치·prefix·실제 source packet을 읽지 않았다. 실행은 Root가 V2 실제 완료를 통지하고 새 exact binding을 넣은 entry를 승인한 뒤만 한다.

`python -B summarize_pair.py --entry /absolute/ROOT_ACTUAL_SUMMARY_ENTRY.json`

`ENTRY_TEMPLATE.json`의 `approved_completed_SOURCE_saved_summary`를 true로 하며 `fit_complete`, `all2_freeze`, `training_code`, `training_entry`, 두 `arm_outputs`를 실제 완료 binding으로 채운다. fresh V2 fit 디렉터리도 받는다. code/root_protocol은 그대로 pin한다. inputs는 COMPLETE/freeze/cropCOMPLETE/samples/sampleDIAGNOSTICS/fullROWS의6개다. output_dir은 이 폴더의 새 `results/` 고정이다. V1 RuntimeError와 CONTROL 재사용·V2 forward동등 수리의 root provenance는 `execution_provenance_note` 및 선택적 `runtime_repair_provenance`로 보존하되 그 pointer는 자동으로 따라가지 않는다.

양팔 final1000/freeze/초기head·draw·base metadata closure와 crop output SHA를 먼저 검증한다. 모델 본체나 prefix/log를 읽어 재학습/forward를 재현하는 검사는 아니다. samples.npz는 allow_pickle=False로 metadata만 읽으며 각7617행 id/target/family/role/availability/view/fraction과 원 sidecar·full15040을 대조한다. 원q/reference는 sidecar 동일성과 거리 대수만 검사한다. CSV selected-family logit/문턱0·가용여부·U를 검사하며 두팔을 모두 보존한다.

SYN main과 YCB AUX의 CM/P/R/U/balanced recall을 FIT/DEV별 따로 계산한다. DEV 회복/악화는 D/E별로도 나누며 SYN q-reference≤3/>3/unknown을 고정 서술 strata로 낸다. YCB r좌표는 없으므로 null 그대로다. macro gate(DEV balanced recall+3pp 및 각 family/class 감소≤2pp)는 계산만 한다. ancestry UNKNOWN이므로 current_admission=false이고 새 모델 선택은 없다.

출력: SUMMARY.json, METRICS.csv, PAIRED_ALL7617.csv, PORTABLE_REPORT_KO.md, COMPLETE.json. full15040 중7423 masked는 modelU가 아니라 forward대상이 아니었음을 명시한다.

`python -B summarize_pair.py --verify-saved`는 같은 파일 옆 `results/`의 SUMMARY/PAIRED/METRICS만 읽는 stdlib 검산이다(NumPy도 import하지 않음). 패키지의 절대경로 binding을 따라가지 않고 CM/P/R/U·gate·paired값·표·고정logitsign을 재집계한다. 원 packet/GT 권위나 신경망 forward의 독립재현을 주장하지 않는다. `--self-test`는 작은 가상 행만 사용한다.
