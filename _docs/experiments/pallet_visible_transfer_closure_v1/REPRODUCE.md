# 재현 / 재개

저장소 root, 기존 `pallet-yolo26` Python/CUDA 환경을 사용한다. private RGB/좌표/checkpoint는 자동 공개되지 않으며 INPUT_BINDINGS의 일치 자료가 필요하다.

```bash
python -m scripts.research.pallet_visible_transfer_closure_v1.prepare
python -m scripts.research.pallet_visible_transfer_closure_v1.infer_train
python -m scripts.research.pallet_visible_transfer_closure_v1.diagnose
python -m scripts.research.pallet_visible_transfer_closure_v1.pre_fit
python -m scripts.research.pallet_visible_transfer_closure_v1.report_diagnosis
python -m scripts.research.pallet_visible_transfer_closure_v1.fit RAW_NEW
python -m scripts.research.pallet_visible_transfer_closure_v1.fit REF_NEW
python -m scripts.research.pallet_visible_transfer_closure_v1.evaluate_new freeze
python -m scripts.research.pallet_visible_transfer_closure_v1.evaluate_new score
python -m scripts.research.pallet_visible_transfer_closure_v1.evaluate_new source
python -m scripts.research.pallet_visible_transfer_closure_v1.close_report
python -m unittest scripts.research.pallet_visible_transfer_closure_v1.test_contract
```

이미완료된TRAIN추론/사전lock은 재실행하지말고해시검증한다(시각필드가있어새로잠그면기존lock과다르다). fit/평가완료파일이있으면비싼계산을재실행하지않는다. 중간fit실패는기존run/optimizer를보존하며 blind재시작하지않는다. 허용된총fit2/update1280상한을넘기지않는다. 평가점으로후보/문턱/seed를다시고르지않는다.

학습구현의effective lr0=1e-5다. INTERVENTION_LOCK.args는원래protocol template(lr0=1e-4)을보존한필드이며,실제lr0 override와10개epoch경로는learning_rates_by_epoch/fit.py/새run args.yaml·results.csv가명시한다. args.epochs=5는기존E2ELoss스케줄보존용,trainer.epochs=10이실제예산이다. 원래5epochcosine의epoch5 실제학습률을이후5epoch고정했다. 첫320 loss/LR/EMA parity를통과하지않으면실험전체중단하도록했다.

신규그림은실측좌표와이전공개예시RGB에서생성했다. 원본RGB·좌표·checkpoint와추론캐시는data하위private로유지하고공개문서에는집계·오차·출처해시만포함한다. CUDA no-opfallback없음/재부팅없음/환경변경없음/다른프로세스종료없음.

기존주결과는pallet_selftraining_paper_closure_v1에그대로있고본namespace는사후진단이다. 원고빌드는기존Tectonic을사용하고본namespace의로그/캐시/BUILD_RESULT에기록한다. 원래숫자표는덮어쓰지않는다.
