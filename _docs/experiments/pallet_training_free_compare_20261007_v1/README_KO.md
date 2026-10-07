# 비학습 보정 비교

[RESULT_KO.md](RESULT_KO.md)의결론과[RESULTS.json](RESULTS.json)의전체/seed/가시성/세션/paired값을읽는다. 같은YOLO319장과두알고리즘네출력의고정평가다.

추가보정학습0. 현재F·기존N3/PoseFix3seed원행·원본YOLOcache·가시성상태를읽기전용재사용한다. a22baseline은별도불변worktree또는PALLET_BASELINE_ROOT이며개인원영상/가중치는공개물에포함하지않는다.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B -m scripts.research.pallet_training_free_compare_20261007_v1.reporting
```

reporting은완료된PREDICTIONS와기존원행만집계하며모델/F를실행하지않는다. 입력/방법/예산은PROTOCOL,수정좌표·F는PREDICTIONS,새동일환경시간은RUNTIME,검산·호출수는CHECKS. 보고전실제선행평가가필요하고원고/LaTeX/PDF/bib변경·빌드0.
