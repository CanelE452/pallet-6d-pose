# 고정 seed1 quick 6D loss 비교

[확인] SOFT6D: **NO_CLEAR_GAIN**; EXPECT6D: **OLD_ONLY_GAIN**. 두 방법 각 6000 update를 실제 실행했다. 추가 seed/대규모 sweep은 없다.

[RESULT_KO.md](RESULT_KO.md)의 단위·분모·불확실성 설명과 [SUMMARY.json](SUMMARY.json)의 원행 재집계/공유 bootstrap/보존 flag를 함께 읽는다. [PROTOCOL.json](PROTOCOL.json), [LOSS_SETUP.json](LOSS_SETUP.json), [TRAIN_RECEIPTS.json](TRAIN_RECEIPTS.json), [VERIFICATION.json](VERIFICATION.json)이 실제 설정·학습·검증 근거다.

[확인] report 시점에 setup PASS, 두 train DONE, 두 방법의 SYNTH/REAL evaluate 및 TRAIN probe DONE을 확인했다. 최초 setup은 아래 direct setup CLI로 실행했다. 나머지는 고정 경로의 단계·완료 결과 재사용 CLI이다. verification은 보고서 생성 뒤 별도 실행하며 완료 근거는 VERIFICATION.json으로 확인한다.

```bash
cd /home/minjae/Documents/github/pallet-pose
export PALLET_BASELINE_ROOT=/home/minjae/Documents/github/pallet-pose-handoff-20261006
PALLET_QUICK_PYTHON=/home/minjae/anaconda3/envs/pallet-yolo26/bin/python
$PALLET_QUICK_PYTHON -B -m scripts.research.pallet_quick_pose_loss_20261006_v1.run --help
$PALLET_QUICK_PYTHON -B -u -m scripts.research.pallet_quick_pose_loss_20261006_v1.setup --source-root /home/minjae/Documents/github/pallet-pose --bank-cache /tmp/pallet-joint-action-cache --cost-cache /tmp/pallet-pose-target-6d-cache --output-cache /tmp/pallet-quick-pose-loss-20261006-cache
$PALLET_QUICK_PYTHON -B -m scripts.research.pallet_quick_pose_loss_20261006_v1.run --stage train
$PALLET_QUICK_PYTHON -B -m scripts.research.pallet_quick_pose_loss_20261006_v1.run --stage evaluate
$PALLET_QUICK_PYTHON -B -m scripts.research.pallet_quick_pose_loss_20261006_v1.run --stage report
$PALLET_QUICK_PYTHON -B -m scripts.research.pallet_quick_pose_loss_20261006_v1.verification
```

| 읽기 전용 의존성 / 새 외부 출력 | 위치 |
|---|---|
| source / 공유 detector feature·치수·seed1 순서 | /home/minjae/Documents/github/pallet-pose |
| native GEO bank | /tmp/pallet-joint-action-cache |
| 완성한 TRAIN 6D 비용 은행 | /tmp/pallet-pose-target-6d-cache |
| 불변 a22 원 구현·기존 대조군 | /home/minjae/Documents/github/pallet-pose-handoff-20261006 (`PALLET_BASELINE_ROOT`로 지정; 원 코드/근거 SHA 검증) |
| 신규 target·checkpoint·실행 ledger | /tmp/pallet-quick-pose-loss-20261006-cache |

| 새 private 최종 가중치 | SHA256 | bytes |
|---|---|---:|
| /tmp/pallet-quick-pose-loss-20261006-cache/fits/SOFT6D_seed1/last.pt | 9c705073be79e479d778d4f8b9f7961f079332cbce559de3f1f18535f1a39d78 | 339518 |
| /tmp/pallet-quick-pose-loss-20261006-cache/fits/EXPECT6D_seed1/last.pt | a2a0a4d54763802006975a416af3b1fd1bb14ca1c15a317f03d7e6efb27f70c0 | 339518 |

[확인] 저장소 출력은 `_docs/experiments/pallet_quick_pose_loss_20261006_v1/`의 PROTOCOL.json, LOSS_SETUP.json, TRAIN_RECEIPTS.json, SUMMARY.json, RESULT_KO.md, README_KO.md, VERIFICATION.json 및 `results/{SYNTH_HELDOUT,REAL_DEV}_{SOFT6D,EXPECT6D}_seed1.jsonl`/각 EXECUTION.json, `TRAIN_PROBE_{SOFT6D,EXPECT6D}_seed1.json`이다. 가중치는 `/tmp/pallet-quick-pose-loss-20261006-cache/fits/{SOFT6D,EXPECT6D}_seed1/last.pt`에 남기고 commit하지 않는다.

[확인] 완료된 학습·평가 영수증 재사용은 `run --stage train` 또는 `run --stage all`의 검증된 CLI만 사용한다. 완료 receipt 없는 기존 ATTEMPT_STATE는 RUN_FAILED로 닫히며 자동 재학습·quiet replay가 없다. 중단된 학습을 low-level training.py로 재개하지 않는다.

[확인] reporting은 완료된 저장 원행만 읽고 새 namespace에 보고서만 쓴다. F/NN/optimizer 추가 호출 0. private feature/weight/cache와 불변 a22 baseline은 별도 의존성이며 공개 저장소만의 완전 재현을 주장하지 않는다. REAL 독립 물리 계측은 0쌍/BLOCKED_DATA. 원고·참고문헌·PDF는 수정하지 않았다.
