# C3 독립 CPU 감사

핵심 실행·평가 계약은 PASS다. 이는 방법 성공 판정이 아니다. 새 fit/update/GPU 사용은 모두 0이며 실제 좌표 배열은 이 문서에 싣지 않았다.

- 두 last320 checkpoint 모두 보호 state **747개가 R0와 exact**다. 변경된 parameter tensor는 RAW9 116개, MANUAL9 126개이며 모두 허용 pose/flow 범위다. FIT의 744 표기는 보호 nonparameter buffer 3개를 제외한 간단한 집계다. 초기화는 같은 R0 binding과 실행 당시 전체 state assertion을 확인했으며 초기화를 다시 실행한 것은 아니다.
- 실제 **320개 batch**의 RGB·bbox·mask·순서가 짝에서 일치한다. arm당 source/real 각각 2,560회, real supervised 10,880·ignore 12,160·invisible 0이다.
- 원 주석과 export를 다시 확인해 **Plastic 3장/15점 + Wood 6장/23점**의 manual_click 좌표만 감독함을 확인했다. 중심·나머지 점은 true-ignore, bbox는 공통 R0 예측이다.
- TRAIN9와 DEV173의 ID/SHA/recording은 분리돼 있고 실제 RGB 182개 SHA를 확인했다.
- detector 364 frame 비교, 2D/fixed-ID frame metric 692개 재계산, cached pose/2D group summary 96개, verified 오류 396개가 일치했다. Plastic128/985·match120, Wood45/346·match45 및 실패 분모를 유지했다.
- 전체 raw/D9 잠금은 02:35:55 UTC, scoring 시작은 02:36:21 UTC다. 소스와 기록 순서를 확인했으며 시스템 전체 파일 접근 추적을 수행한 것은 아니다.
- 공개 표 **57행 전부** 원 RESULTS/CSV의 표시 정밀도와 일치한다. 초고의 Wood SYN IoU P90 한 셀은 0.8959→**0.8960**으로 수정됐으며, 검산했던 본문과의 차이가 이 한 셀뿐임을 역치환 SHA로 확인했다. 고정 실행 결과와 최종 보고서 해시를 연결했다.

C3 결과는 **부분 TRAIN 적합**이다. MANUAL9는 35/38 PCK10이며 한 Wood frame의 >20px 3점이 남는다. MANUAL9−RAW9는 Plastic +5/985 / AUC +0.012457, Wood −22/346 / −0.043267, verified66 −3/66이다. 완전한 TRAIN 적합 후 순수 일반화 실패라고 해석하지 않는다.

`camera_dynamic_0123_v4 / UNCONFIRMED_SIGNED_AXIS / MANUAL_REVIEW_REQUIRED`를 보존한다. **물리적 signed-axis와 독립 6D 정답은 여전히 미확정**이다. 기존 교사에 노출된 9/38 stored-index capability이며, 전체217/361 GT 학습 상한이나 일반화 보장이 아니다. 이번 감사는 새 6D solver를 돌리지 않고 고정 per-frame pose 지표의 집계를 재검산했다.

세부 카운트·체크포인트·핵심 입력 해시는 [AUDIT_C3_INDEPENDENT.json](AUDIT_C3_INDEPENDENT.json)에 있다.
