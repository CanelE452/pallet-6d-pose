# A/B 실행 완료 — 최종 판단

A의 12회 × 2,000-step 학습, 초기 R0 포함 평가·통계·6D 보조 분석까지 완료했다. B의 주지표·실사 pose·runtime은 불변이며, 원본 파일을 보존한 채 합성 회전 보조 지표에만 좌표계 정정 sidecar를 추가했다.

## A — 대칭 지도

- RECT 합성 val: **UNRESOLVED**, ΔE_sym=-0.00065538552, 95% CI [-0.0016709867, 0.00032045198].
- SQUARE 실사 DEV: **UNRESOLVED**, ΔE_sym=-0.010872931, 95% CI [-0.022119474, 3.5006556e-05].
- RECT 실사 DEV: **UNRESOLVED**, ΔE_sym=0.015942689, 95% CI [-0.0034990496, 0.038864134].

공동 선택 방식은 사용자 승인 후 실행했으며 기존 RLE를 제거하거나 물체별 clamp로 바꾸지 않았다. `definition_integrity=PASS`와 `geometry_gain`은 별개다. 성능 결론은 fixed checkpoint·data·budget 및 재사용 DEV에 한정한다. 자세한 tail/검출/pose 손해는 `A/REPORT_KO.md`를 따른다.

## B — 세 incident line 보조

**LINE_AUXILIARY_NOT_ESTABLISHED** 유지. 합성·실사 primary CI 모두 0을 포함했고, 같은 RTX3080 전체 파이프라인 median은 P12.73ms → B3 39.33ms(약3.09배)였다. 현재 고정 구성에서 최종 추론에 추가할 근거가 부족하다. Hough 원리의 불가능성으로 일반화하지 않는다.

## 실행량과 보존

- A: main24,000 + smoke12 optimizer updates; 학습384,000 images + 고정 probe768 + smoke192. 새 평가31,458 detector-image forward 및 동일 수 pose-frame evaluation.
- Pose 실행 중 ID 대응 정정으로 기존 캐시의 CPU 채점만 재수행해 전체 pose 채점 횟수는 59,598이다. 별도 합성 회전 좌표계 정정은 A/B 35,820 frame을 회전만 재평가했으며 새 PnP·neural forward·학습은 없다. 정정 전 수치와 실행 경위를 보존했다.
- CPU 회귀검사 43/43 PASS. 가중치·sampler·업데이트별 lr/gradient·BN·마지막 state 무결성 확인.
- 원본 R0/P/DHT·원본 데이터·기존 A0/B 결과·논문·배포는 보존. Square 2행 정정은 새 target sidecar만 변경.
- Git은 코드·표·가벼운 기록만 포함한다. 가중치/상세 raw 예측 및 ignored A/runs의 전체 JSON trace는 로컬 결과 경로에 유지; 24,000-step 요약 CSV와 run별 해시·검증은 Git에 포함한다.
- 재부팅·드라이버 변경·타인 GPU 작업 종료·FINAL 열람·외부 알림·새 탐색 없음.

추가 학습은 자동으로 시작하지 않는다. 다음 작업은 이 제한된 결과를 근거로 논문에 포함할 범위 결정이며, 원고는 이번 실행에서 자동 수정하지 않았다.
