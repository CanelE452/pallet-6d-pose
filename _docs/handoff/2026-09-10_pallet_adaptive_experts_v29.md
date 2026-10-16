# V29 점별 적응형 점/선 전문가 검증

사용자: 점 우선 clean / 점 가이드 선 occlusion / 점 독립 선 truncation 가설을 실제로 학습하고 적응적으로 선택하라고 승인했다.

- 결과 폴더: `data/pallet/results/pallet_adaptive_experts_v29/`
- 코드: `scripts/research/pallet_adaptive_experts_v29/`
- 학습 전 계약: 결과 폴더 `PROTOCOL.json`, `PRECHECK.json`.
- 점 전문가: frozen stock YOLO 특징/좌표에서 새 보정. V12/V25 부모 출력 미사용.
- DHT 전문가 2개: 같은 V6 시작과 구조. guided는 예측 점 거리맵, independent는 항상 영맵. 새 structural-line CE 학습. 독립 분기는 점 좌표를 바꿔도 출력이 완전히 같아야 한다.
- 선 디코더: incident cuboid 3역할×3mode의 비평행 선쌍27개, abs(det)>=.1 및 ROI 좌표 절대값<=4. 전부 무효면 ridge1e-4 최소노름 후보. 점 좌표 없이 선 교차 후보만 학습 선택한다. 물리 가시 경계/끝점 GT는 새로 확보하지 않았다.
- 합성764원본을 해시80/20 분리. 새 전문가/선디코더와 선택기 원본 분리. V6/YOLO의 과거 사전학습 원본과는 겹칠 수 있음. router는 frozen expert out-of-finetune-original 잔차로 학습.
- 3시드, point/DHT2/decoder2 각각1200, router3 각각800 =24모델25,200업데이트. batch16 FP32 AdamW, DHT1e-4 나머지3e-4, clip5. 최종 체크포인트만 사용.
- router: 유지/점/가이드선/독립선 기대오차 argmin. state+quality / no-state / point-only 대조. 합성 인공가림·화면밖 kind 정답은 학습만 사용하며 실제 가시성 GT로 간주하지 않음.
- 기존 실사297/9세션 재사용DEV의 명시적 연속 비교. canonical140/논문 계보와 섞지 않음. 새 실사 봉인검증/자연가림 라벨/새 GT수정 없음.
- 평가: prospective6조건, synth_val/real8조건, 검출실패 포함 분모. GT최선전문가/GT상태고정선택은 별도 privileged 진단. 실제 forward는 GT/조건명 미사용.
- 안정성은 V27과 같은 각시드 쉬운조건1%/어려운조건V18이득90% 보존; crop20 adaptive vs point_router 각시드1% 및 세션95%CI상한<0 동시 요구.
- 기존 완료 V25/V27/V28 해시 파일 수정 금지. 새 노트/색인/STATUS만 갱신.
- 종료까지 report 이미지/HTML 생성, Chrome 실제창 확인, 기존 Discord helper로 완료문장만 전송. credentials 출력 금지.

현재 상태와 이어서 할 일은 결과 폴더 최종 마커와 `driver.log`를 확인한다. 드라이버: `run29.py`, 재시작 시 완료 마커 단계는 건너뛴다. 학습 중단 셀은 START를 보존하고 원인을 기록한 뒤 복구하며 결과 기반 재학습은 하지 않는다.
