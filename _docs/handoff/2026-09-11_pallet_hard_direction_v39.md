# V39 큰 점 오류의 보정 방향 식별

사용자승인. V38fixed12fit4cal8check,V34anchor/V36gridboth3seeds. Hard>=10pxGT는학습/진단만. BalancedL1/hardgeometry/hardappearance+normal/OOFhardgeometry/point및linehardnessclassifier48HGBfits60rounds. 기준사전PROTOCOL. Cal4에서만F1threshold선택,check8로조정금지. GTgateoracle과실제gate분리. 기존완료결과불변,0neural/newreal/manualGTedit. 완료후MD/시각화/Chrome/Discord.

## 실행 완료

48 HGB 모델 ×60라운드=2880라운드, 별도 감사 재학습3개. 새 신경망 업데이트/실사 평가/수동GT 수정 없음. 기존 V38 중심6개 재사용. 모든 주요 학습·평가·감사 마커 완료.

큰 오류(기존 점≥10px) 확인159/189/179개. Both 선 요약 분류 AUROC .932/.929/.931, precision .667/.709/.720, recall .717/.593/.475. 큰 오류 전용 법선 MAE 학습18.214→11.040px, 확인15.442→16.073px. OOF17.134px. 부호 정확도 학습약88%, 확인55~57%. 선 기하를 제외한16.670px보다16.073px는 좋지만 기존 중심과0보정을 넘지 못했다.

실제 점+선 선택10.484px 대 기준10.463px, 시드 개선율−3.59/−2.08/+5.41%, 차이95%구간[-.286,+.402]px. 점만 선택10.868px보다 좋으나 등록 안정성 실패(accepted=false). GT 큰 오류 선택10.204px도 차이구간0포함/피해점39/45/36개. GT 피해 거절9.606px, 후보oracle7.802px는 비배포 진단.

후속 기술 통계는 별도 PHENOTYPE_PROTOCOL/PHENOTYPES에 보존. 다른 GT점5px 이내 큰 오류9/9/1개; 독립2물리선 방향 없는 큰 오류약67~70%. 단순 번호 혼동이나 관측 부족 하나로 법선 예측 실패 전체를 설명하지 않는다.

48모델 예측 재현,60점 배열 독립 재계산,3개 비학습 영역 교란 재학습,calibration 기준/checkGT 분리,같은 점 입력 분류기 두 arm 동일성,이전 완료 바인딩 보존 모두 AUDIT PASS. README와 run39.py로 단계별 재현. REPORT_KO/노트/HTML과6그림 생성,3개 원본 사례 개별 확인. 최종 Chrome 실제 창·DOM 수치 검사는 DELIVERY.json, 완료 파일 해시는 COMPLETION.json, 알림 영수증은 DISCORD_NOTIFICATION.json에 기록한다.

현재 판정: 쉬운 점 비중이나 큰 오류 선택만으로 해결되지 않으며 현재 특징에서 보정 방향·크기의 그룹 간 일반화가 병목이다. 원천적 융합 불가능 판정은 아니다. 추가 대응/관측 정보와 새 그룹 대조는 미실행 제안이며 다음 턴에 이48모델을 반복하지 않는다.
