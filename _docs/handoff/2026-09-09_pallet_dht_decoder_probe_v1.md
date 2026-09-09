# Pallet DHT decoder probe v1 — COMPLETE

사용자가 요청한 빠른 실험과 보고 전달은 실제 완료됐다. 잔여 학습/평가/화면검토/알림이 없다. 기존 큰 실험도 완료 상태다. 새로운 사용자 지시 없이 driver를 재시작하거나 Discord를 재전송하지 않는다. unifiedexec session48772는 exit0으로종료됐다. Commit없음, 기존 unrelated dirty변경보존.

최종결과: [주제별 해석](../notes/pallet_dht_decoder_probe.md), [HTML](../../data/pallet/results/pallet_dht_decoder_probe_v1/index.html), [COMPLETION](../../data/pallet/results/pallet_dht_decoder_probe_v1/COMPLETION.json).

- 기존joint3seed×동일14장 무학습42records 및 고정jointseed1+동일55,118parameter2모듈×1000actualupdates 완료. 실제합성2048train/512val+실사319캐시2,879CNNforwards+5warmups. Cache76.035+463.715초, head학습11.280+11.110초. Syntheticloss매칭2047/2048,509/512,realGT학습0. 새학습1seed의2D검사이며6D/negative/독립FINAL미평가,centroid8고정.
- 실제512val+319real×3모드=2,493cachedheadforwards. 실사모든군309/319매칭,2738/2818유효GT점. median/P90(px): 원래6.897/41.487,점결합부6.514/44.928,선결합부6.661/41.856,wrongline7.372/42.996. 선−점frame평균차−.0231px,20k session95%CI[−.6419,.3051]. 등록후속기준false,안정적향상주장없음. 원래양호점1711개중10px초과전환점58/선68.
- 사용자case공식8점median270.065→점267.160/선265.213px로번호오류지속. 선군4/7의가중후보GT거리39.901/32.701px이나signedgate−.01289/−.00826,최종오차279.481/280.702px. 후보를전혀못찾는것만의문제로단정하지않는다. 추가사후gate=1단일component진단은전체median/P90/mean16.190/128.422/44.522px로악화,양호점806/1711이10px초과;사용자casemedian119.021px로줄지만미해결. GT로gate튜닝/새forward/재학습없으며원등록RESULTS보존,학습원인인과확증아님.
- CPU기하19tests/합성12장+2arms×4updates스모크/독립831프레임·16표·20kCI감사PASS. PASS는완료·일치성이지성능개선아님. 최종1276overlay/319원본이미지실제QA와root최종6PNG직접검토PASS. 어두운GT의흰halo/범례만렌더수정,원render/HTML/QA/sourcefreeze와수치를보존하고RENDER_VISIBILITY_AMENDMENT로명시.
- 전체완료2026-09-08T19:12:26.516809Z(2026-09-09 04:12 KST), Chrome창0x62003d3 viewable/nonhidden확인,DiscordsentHTTP204. 제목 `Point–Line Decoder Probe · 점·선 후보 결합`.
- COMPLETION SHA `014b4d290b6e9d14cfd73b334566fd2ce1f4166a0d125277a647f7d1b33d4e58`; HTML SHA `881a54a976c33844d6ee60be53e183f00962a33f2a264d46541ef721dc35b93c`; TRAIN_PROTOCOL SHA `97c5e8e2772e8da5e6987a6523147c6d04340d3b5a399689991d7d5d67dfb0e1`. 최종22artifact/14source/6PNG SHA root재검증. 원진행인계는output/provenance/handoff_history/pre_complete.md에보존.

## 아래는 실행 당시 진행 기록 — 현재 상태는 위 COMPLETE

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
