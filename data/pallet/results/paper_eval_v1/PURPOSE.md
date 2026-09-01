# PURPOSE — PAPER_EVAL 평가 산출물

[소비처]
논문 본문 M1 (main method comparison) · M2 · M3 · M5 의 모든 수치.
`_docs/paper/generated/TABLE_M*.md` 가 이 폴더의 JSON 을 읽어 표를 만든다.

[문장]
"PAPER_EVAL(positive 319 / negative 2,688 unique) 하나에서, 모든 비교 대상을
같은 evaluator·같은 population·같은 metric 정의로 채점한다 — 그래야 M1 의 행들이
서로 비교 가능한 값이 된다."

## 판단 지표

```
2D        supervised keypoint location median px (↓) · detection rate @IoU50 (↑)
ranking   box AP50-95 (↑) · AUROC (↑) · FPR95 (↓)
pose      POSE_METRICS_STATUS 가 READY 가 되기 전까지 보고하지 않는다
```

## 계약

```
population   challenge/real_gt_v2/manifests/PAPER_EVAL_*.json + DEV_NEG2689
evaluator    challenge/evaluation_v2/paper_real_eval.py
recipe       PAD 100 BORDER_REFLECT_101 / imgsz 640 / conf floor 0.001
동결         _docs/paper/PRE_RESULT_LOCK.json (commit 15f0cb5, 결과 보기 전)
```

Ultralytics 가 아닌 baseline(DOPE 등)은 예측을 미리 덤프해
`--predictions` 로 **같은 채점 코드**를 태운다.  별도 채점기를 만들지 않는다.
