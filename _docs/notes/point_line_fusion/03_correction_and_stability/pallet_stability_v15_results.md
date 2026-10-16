# V15 결과: 악화 억제·소폭 개선 확인, 큰 복원 이득 유지 실패

사용자 요청에 따라 안정성에 필요한 조건을 나눠 실제 학습·평가했다. 결과는 `PARTIAL_SMALL_CONSISTENT_GAIN_FULL_STABILITY_FAIL`이다. 사전 전체 안정성 기준 통과 모델은 없고 기존 배포를 교체하지 않았다. 다만 balanced_old의 작은 개선과 악화 억제는 실제 관측된 결과로 별도 기록한다.

설계: [pallet_stability_v15_design.md](pallet_stability_v15_design.md).
코드: `scripts/research/pallet_stability_v15/`.
결과: `data/pallet/results/pallet_stability_v15/`의 HTML, REPORT_KO.md, SUMMARY.json, STABILITY_DIAGNOSIS.json, AUDIT.json, INTEGRATION_CHECK.json.

## 실행

4구조: median_old(tau.5), risk_old(tau.2), balanced_old(합성 부호별 역빈도 가중치 추가), balanced_rich(큰 선 후보3개 추가). 모든 구조8슬롯 head, 같은 초기화·batch순서,253합성 이미지/1462검출 관측, 각3시드×2000업데이트=총24000업데이트. 이전 YOLO/보정/복원/DHT 고정, 실제 학습/튜닝 없음.

새 선 후보는 점별 상위3 DHT 선의 법선 방향으로 이동하되 최대5%box diagonal로 제한한다. 후보에만 추가하고 최종점 강제 투영 없음. 모든 학습점 유지, 유효점 GT이득의 harm<-0.25 / neutral / benefit>0.25를 노출class·후보별 가중치(cap10)로 균형화. Pinball 손실만 사용해 median vs lower-quantile을 직접 비교, V14 CE 사용 안 함.

평가: 기존 synth505/real297·9sessions/manual1251점의6조건 + 사전등록 crop10/crop30. 실제 새 조건 모두293검출, outside489/601점. 같은 장면 재사용이므로 새현장 검증은 아니다.

## 실제 결과

3시드 평균px. 원본은 프레임 평균, 가림/잘림은 대상 점 평균.

| 모델 | clean | hidden corner20 | outside crop20 | outside crop40 | outside crop10 | outside crop30 |
|---|---:|---:|---:|---:|---:|---:|
| V12 base |5.537562|16.539330|12.372119|27.770936|10.605251|13.274195|
| V12 amodal |5.542445|13.554489|16.496415|19.921341|11.002902|13.554369|
| V14 fusion |5.527641|13.767414|16.258101|20.526354|11.022809|13.414562|
| median_old |5.489423|13.856347|15.875221|20.384082|11.131773|13.234063|
| risk_old |5.537493|15.577225|12.391689|26.774757|10.692656|13.133830|
| balanced_old |5.536086|16.288084|12.372986|27.536208|10.615789|13.271013|
| balanced_rich |5.547742|16.295015|12.575846|27.764765|10.632336|13.335329|

balanced_old는 clean/crop10/20/30의 base대비1% 상한을 모든 시드에서 통과하고, corner20/crop40도 base보다 모든 시드 좋아졌다. 대상 점 기준 paired9session95%CI:

- corner20: -0.251246px, CI[-0.611727,-0.039093].
- crop40: -0.234728px, CI[-0.482325,-0.115749].
- crop20: +0.000867px, CI[-0.001147,+0.002870].

작은 일관된 개선은 관찰되지만 기존 amodal의 큰 개선80% 유지 기준은 실패했다. 이것을 기준을 완화한 최종 합격으로 바꾸지 않는다. median_old는 큰 복원 이득80%를 유지했으나 약한 잘림에서 악화했다. risk_old는 약한 잘림을 보호하면서 어려운 조건에서 약1px 개선했지만80% 유지 실패, 새 crop10 seed3에서1% 상한도 실패했다. balanced_rich도 모든 조건 기준 실패.

## 선 후보와 선택의 병목

새 선 후보 평균 이동량: hidden corner20 5.676px, outside crop20 5.535px, outside crop40 6.164px.

| 대상 | old5 oracle | rich8 oracle | extra oracle gain |
|---|---:|---:|---:|
| corner20 hidden |13.273738|12.604179|0.669559|
| crop20 outside |11.549036|10.205025|1.344012|
| crop40 outside |17.063706|16.267319|0.796387|

이는 GT를 아는 사후 후보 선택 상한이며 정상 추론이나 모델선택에 쓰지 않았다. 후보의 개선 여지는 V14보다 커졌으나 학습한 평가기는 이를 활용하지 못했다. crop20에서 새 선 후보를 적용한9/78/43점 중4/67/35점이 악화. 큰 선 후보 추가의 사전1%/모든시드/CI기준은 주요6조건 모두 실패.

과도하게 보수적인 선택은 합성 검증에서도 큰 복원을 줄였으므로 실사 도메인 전이만이 원인은 아니다. 다음 검증 가능한 가설은 후보별 선 위/주변 영상 증거, 관측 가능한 유한 선 구간 및 파렛트 외곽선 역할을 직접 학습하는 것. 아직 실행하지 않은 가설이며 성능 보장 아님.

## 검산과 전달

독립 audit174,836예측 프레임 행 및106,920선택 프레임 행 PASS. 후보 재구성 최대0.000160721px, 지표 최대1.0658e-14.24000유한업데이트·초기화/배치일치·부모/소스해시·새GT이동/수동GT해시 확인.

StablePointLineNetwork는 V14의 사용하지 않는 점수head를 제거한 전체 추론 그래프.12모델×6조건×4프레임(288프레임) 통합/캐시출력 및argmax일치, 최대0.000061035px.12개 통합 실험 체크포인트를 export/에 저장, 실제 평가로 시드선택 없음.

보고서에는 안정성/복원이득 tradeoff plot, 모든8조건/시드/지표 표,12개 GT진단 예시, 후보별 점수와실제이득,세션별 대상오차를 포함했다. 이미지 시각화 예시·점 선택은 사후설명용이며 GT inference 사용 아님. 자연가림unknown·재사용DEV·수동GT 재클릭없음·기존reflect100·원본점ID유지.6D/이전41.49px과 다른 지표다.

재현(기존 학습 덮어쓰기 거부): prepare15.py --phase extract → prepare15.py --phase parents → train15.py → eval15.py --phase inference → eval15.py --phase score → network15.py → audit15.py → diagnose15.py → report15.py. Python `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`.

HTML 자동 열기와 완료-only Discord 알림은 결과 폴더의 전달 receipt에 기록한다.
