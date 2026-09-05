# GATE E 후속 — target adapter

```
상태 = NOT_RUN
사유 = TARGET_BIAS_SIGNAL 이 STRONG 이 아니다
       (domain AUROC 1.0000 통과 · gross 분리 AUC 0.4820 불통과, 임계 0.65)
```

METHOD_LOCK `gate_e.adapter_admission.runs_only_if` 에 "TARGET_BIAS_SIGNAL strong" 으로
잠겨 있다. E0 AdaBN 과 E1 target residual adapter 를 하나도 실행하지 않았다.

임계를 사후에 낮추지 않았다. domain AUROC 만 보고 "분리가 되니 adapter 가 먹을 것" 이라고
진행했다면, 오차와 무관한 축을 손대면서 개선을 기대하는 실험이 됐을 것이다.
그 연결을 요구하는 두 번째 조건이 사전등록돼 있었기 때문에 걸렀다.
