# Gradient transfer diagnostic v1

이미 승인된 no-fit 진단의 공개 보고서다. 새 fit, optimizer 생성/step, checkpoint write, DEV reference read는 허용하지 않는다. 원래 이미지·target 배열·frame ID는 private namespace에 유지한다.

## CPU 보고서 재현

저장된 private 결과가 있는 동일 환경에서 실행한다. GPU 측정은 다시 실행하지 않는다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_gradient_transfer_diagnostic_v1.report --self-test
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_gradient_transfer_diagnostic_v1.report
```

입력: `PROTOCOL.json`, `data/pallet/results/pallet_gradient_transfer_diagnostic_v1/RESULTS_PRIVATE.json`, 기존 source bindings. 모든 binding을 검증한 뒤 집계·그림을 재생성한다. Private 데이터가 없으면 재계측이나 데이터 교체로 우회하지 않고 중단한다.

출력: `RESULTS.json`, `REPORT_KO.md`, `figures/functional_response.png`, `figures/sign_transitions.png`. 그림은 실제 측정 데이터의 집계/응답이며 원래 이미지나 좌표를 그리지 않는다. 공개 JSON에는 frame ID·bbox·intrinsics·target/predicted coordinate 배열을 넣지 않는다.

## 이미 실행한 계측 경로

`probe.prepare`가 기존 TRAIN midpoint 입력과 사전 protocol을 잠갔고, `probe.run`이 functional ±epsilon으로 측정했다. 측정 실행·host GPU 접근은 부모가 별도로 관리했다. 이 README는 재실행 허가가 아니다.

`RESULTS.json`의 raw measurement hash·protocol/code bindings·state audit를 먼저 확인한다. 양수는 저장 타깃 방향이지 물리 정답 방향이 아니다. REAL 성분은 full mixed loss의 성분이며 standalone real-only training과 같지 않다.
