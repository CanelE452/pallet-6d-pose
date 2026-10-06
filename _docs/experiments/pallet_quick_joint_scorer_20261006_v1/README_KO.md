# LOCAL_CAP / JOINT8 고정 비교

[확인] LOCAL_CAP **NO_N3_GAIN**, JOINT8 **NO_N3_GAIN**. [RESULT_KO.md](RESULT_KO.md), [RESULTS.json](RESULTS.json), [PROTOCOL.json](PROTOCOL.json), [RUN_RECEIPTS.json](RUN_RECEIPTS.json)이 실제 원행·짝 비교·설정·실행량 근거다.

```bash
cd /home/minjae/Documents/github/pallet-pose
export PALLET_BASELINE_ROOT=/home/minjae/Documents/github/pallet-pose-handoff-20261006
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B -m scripts.research.pallet_quick_joint_scorer_20261006_v1.run --help
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B -m scripts.research.pallet_quick_joint_scorer_20261006_v1.reporting
```

[확인] source `/home/minjae/Documents/github/pallet-pose`, bank `/tmp/pallet-joint-action-cache`, cost `/tmp/pallet-pose-target-6d-cache`와 기존 LOSS_SETUP 저장 soft target을 읽기 전용으로 재사용했다. private 가중치는 `/tmp/pallet-quick-joint-scorer-20261006-cache/fits/{LOCAL_CAP,JOINT8}_seed1/last.pt`에 있으며 SHA·bytes는 RUN_RECEIPTS에 기록한다. 불변 a22 작업트리와 private 데이터/feature/cache가 필요하므로 공개 checkout만의 완전 재현을 주장하지 않는다.

[확인] 완료 영수증만 검증 후 재사용한다. 중단된 학습·평가를 quiet retry하지 않는다. reporting은 완료된 저장 원행만 읽고 새 경로에 결과를 쓴다. NN/F/optimizer 추가0, 원고/PDF/bib 작업0.
