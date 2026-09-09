# Pallet DHT decoder probe v1 — 진행 중

사용자 승인: “그러면 이거를 이용해서 실험해줄수있어?” — 무학습 후보 진단과 작은 frozen-backbone 결합부 학습을 실제로 완료한다. 이전 큰 실험은 완료됐으며 재시작하지 않는다. HTML을 실제로 열고 검토한 후 Discord에 결과를 보낸다. 기존 알림 허가가 지속된다.

목적/판정/범위: [notes](../notes/pallet_dht_decoder_probe.md).

경로: scripts/research/pallet_dht_decoder_probe_v1, data/pallet/results/pallet_dht_decoder_probe_v1.

소유: root=protocol/driver/docs/finalize, side_edge_targets=geometry/tests/14frame diagnostic, dht_operator=cache/model/train(GPU 단독), dht_visualization=evaluate/report/visual_qa.

기존 v1/v2 bound 코드/결과 수정 금지. 기존 dirty 변경 보존, commit 없음. 원래camera_dynamic_0123_v4 및 같은 ID 공식평가 우선, skill의 낡은 object-frame/order-free 지침 적용하지 않음. 신규 FINAL 접근 없음. 실제캐시/학습/평가 산출물로 단계 완료 확인, 코드 구현 선언만으로 완료 처리 금지.

초기 계획: 기존 hough_joint seed1 EMA SHA0960fb32fd99fd0837a588792e07727298fc6f88f1e4ec3464efd69bd5574d37, synthetic2048train/512val, real319, decoder1000updates×2arms(single seed), wrong-line evaluation. 합성pool은 기존backbonetrain/val에서서로분리된부분집합이며 새로운backboneholdout아님. 무학습14장×3기존seed는3새학습seed와구분한다. 아직 본 실행 전.

## 실제 진행

- CPU기하19tests/실제GPU스모크12장+2arms×4updates PASS. 독립model smoke receipt `provenance/model_smoke/RESULTS.json`.
- TRAIN_PROTOCOL SHA97c5e8e2772e8da5e6987a6523147c6d04340d3b5a399689991d7d5d67dfb0e1. SOURCE_FREEZE에서geometry/model/cache/train/diagnostic및원래backbone소스고정. boundsource변경금지.
- driver unifiedexec session48772,18:52:26UTC시작. diagnostic→cache→train→evaluate→report→visual_qa→root실제스크린샷검토→finalize. main GPU는driver단독. 중복실행금지.
- DIAGNOSTIC.json 실제완료42records/14unique: 고정선택median3seed모두악화, oracle후보상한은더좋지만사용자case0번후보오류가남는다. notes/history실제수치기록완료. 현재cache진행.
- 최종root검토 전에 Herschel INDEPENDENT_AUDIT.json(complete/PASS/protocol_sha256)이필요. root는실제screenshots를view_image로본후 VISUAL_REVIEW.json(actual_screenshots_inspected,html_sha256,complete/PASS)을쓴다. driver가대기하다finalize로이어져실제브라우저window+DiscordHTTP204를확인하고COMPLETION기록.
