# 저장 깊이 → RGB 학생 거리 신호 pilot 결과

기존 깊이 교정 타깃은 RGB 학생에게 전달됐고, 같은 원래 의사 레이블 학습보다 재사용 DEV의 z/위치 **중앙값**을 줄였다. 그러나 평균·P90과 합성 보존 성능은 악화했고, GLOBAL 학생과의 비교는 혼합되어 **영상별 깊이의 추가 정확도 이득은 입증되지 않았다**. 판정은 `GLOBAL_COMPARISON_UNRESOLVED`이며 GLOBAL의 동등성·충분성도 확정하지 않는다.

## 고정 범위와 입력

- 직접 사용자 요청의 `main`/새 브랜치 금지가 첨부 계획의 새 브랜치 제안을 우선한다. 새 실험 namespace만 추가했다.
- 직사각형 plastic TRAIN217/원촬영7개를 재사용했고 교사 accepted119/7개 촬영을 세 학습 팔에 동일하게 사용했다. 원본 RGB SHA와 원촬영 기준 EVAL 중복0.
- 기존 clean-to-pose의 같은128 ID: clean29/자연가림99. 깊이116/9촬영; `capture0403noapril`12장은 실제 깊이 폴더가 없어 P0 fallback으로 전체128에 유지했다. 반복 DEV, 단일 seed42이며 독립 test가 아니다.
- SOURCE256/249 scenario는 등록 synthetic TRAIN과 ID/scenario/SHA 중복0. 고정 TRAIN probe32는 accepted ID 사전순으로 성능을 보기 전에 선정했다.
- 원 깊이 취득 당시 시간/FOV/optical-Z/registration·왜곡 기록은 미확인이다. 기존506 감사의 경험적 호환성과 NOT_READY_FOR_GATE1을 그대로 기록했다. TRAIN172는 같은 감사 촬영,45는 촬영별 호환성 미검증이며 accepted119 중26장은 이 미감사 촬영에 속한다. 정렬/스케일 fitting·새 자료/주석은0.

## 1. 교사 생성과 실제 자세 함수

예측 P0의 R_cf/치수/W-D 가설을 고정하고 중심 시선의 z 한 자유도만 바꿨다. 면15% 축소, cos≥0.3, 고정32×32 격자≤1024,4×4 checkerboard 공간 분할, 블록200회 재표집/seed20261007,0.8–1.2z0 범위를 먼저 봉인했다. 범위 밖 해는 clip하지 않고 abstain했다. fitting A의 거리와 별도 B 방향/잔차를 검사했다. 교사 입력에 사람 GT/GT ROI/GT branch를 사용하지 않았다.

qD는 q0에 투영 차이만 더하며 center8·기존 true-ignore를 보존한다. 새 타깃에 N3의1% 상한을 적용하지 않았다. 교사 실제 F 평가를 마친 뒤에만 학습을 허용했다.

| RGB-D 교사 구분 | z 중앙값 cm | T 중앙값 cm | 자세 실패 |
| --- | --- | --- | --- |
| R0 | 8.156 | 9.353 | 0 |
| TEACHER_PD_RGBD | 5.889 | 6.469 | 0 |
| TEACHER_F_QD_RGBD | 6.013 | 6.668 | 0 |

accepted EVAL59와 전체128 fallback을 별도로 보존했다. 교사 F(qD) z/위치 중앙값과 평균 z가 사전 기준을 만족해 `PILOT_GO`였다. 하지만 accepted59의 z P90은25.61→36.82cm, T P90은26.32→41.49cm로 악화했고 직접 가시점 손상은2/66이다. 원촬영 재표집의 z/T 차이 구간은0을 포함한다. 구성 PD와 달리 실제 F(qD)에서 W/D branch 변경은 TRAIN6/EVAL3회였으므로 전체6D/면 혼동 해결로 해석하지 않는다.

## 2. RGB 학생의 타깃 전달

RAW/GLOBAL/DEPTH는 같은 R0 SHA, accepted119, real512+source512/epoch, 동일 affine/HSV/RGB/순서/지원 마스크로 각각5epoch·320update를 완료했다. AdamW lr1e-5/lrf0.1/weight decay1e-4/cosine/warmup0, 추가 가림 증강·seed·설정 탐색0. 전체320-batch trace84,480개 검산과 각747개 보호 텐서의 저장 전후 동일성이 통과했다. pose/flow132개만 갱신 가능했다.

| TRAIN probe32 모델 | 같은 qD 잔차: frame-mean의 중앙값 px | 실제 F(qD) z까지 중앙값 cm |
| --- | --- | --- |
| R0 | 6.473 | 19.992 |
| RAW_TARGET | 6.900 | 20.108 |
| GLOBAL_TARGET | 3.386 | 7.925 |
| DEPTH_TARGET | 2.534 | 5.739 |

이는 TRAIN 타깃 추종이다. 숨은 코너의 측정 정답이나 정확도/일반화 증거로 부르지 않는다.

## 3. RGB 단독 최종 자세와 GLOBAL 대조

| RGB-only 모델 | z median cm | T median cm | z mean cm | T mean cm | T P90 cm | R median deg | ADDsym median m | 실패 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 8.156 | 9.353 | 73.260 | 77.081 | 79.122 | 3.530 | 0.119 | 0 |
| GLOBAL_POST_RGB_ONLY | 11.411 | 12.518 | 83.533 | 87.234 | 98.743 | 3.530 | 0.147 | 0 |
| RAW_TARGET | 7.970 | 9.098 | 72.723 | 76.566 | 77.759 | 3.970 | 0.128 | 0 |
| GLOBAL_TARGET | 6.168 | 8.412 | 73.736 | 77.533 | 86.519 | 4.002 | 0.112 | 0 |
| DEPTH_TARGET | 6.534 | 8.027 | 74.043 | 77.844 | 87.383 | 3.920 | 0.099 | 0 |

DEPTH-minus-RAW는 z 중앙값−1.437cm/T−1.071cm지만 평균은+1.320/+1.278cm, P90은+9.750/+9.624cm 악화했다. 원촬영10개 짝 재표집95% 산포 구간은 z `[-3.696, 3.908]`cm, T `[-4.152, 3.986]`cm다.

TRAIN에서만 구한 m=0.051136997, 배율=1.052467068를 사용했다. 학습 없는 GLOBAL_POST는 R0보다 악화했다. **GLOBAL_TARGET는 DEPTH보다 z 중앙값은 좋고 T 중앙값은 나쁘다.** DEPTH-minus-GLOBAL의 구간은 z `[-1.440, 1.525]`cm/T `[-1.318, 1.517]`cm라 영상별 신호의 추가 이득이나 동등성을 확정하지 않는다. 최초 sign-only 분류의 GLOBAL_SUFFICIENT는 혼합 방향을 충분성으로 과해석했으므로 `DECISION.json`이 그 분류를 바로잡는다. 원 수치행/학습/교사 규칙은 변경하지 않았다.

RGB 학생의 직접 가시점 good<5→bad>10은 모두0/66, 자세 실패도0/128이다. 레거시 전체2D의 median/P90/PCK10과 >20/>50px 오류, clean29/natural99, 자동 F와 R0-branch 고정 진단은 원행과 STUDENT_SUMMARY에 별도로 남겼다. N3 추가 연결은 이 pilot의 필수 대조가 아니며 수행하지 않았다.

## SOURCE 보존과 자원

| SOURCE256 모델 | z median cm | T median cm | T P90 cm | 실패 |
| --- | --- | --- | --- | --- |
| R0 | 1.954 | 2.181 | 16.677 | 0 |
| RAW_TARGET | 2.139 | 2.448 | 19.648 | 0 |
| GLOBAL_TARGET | 5.969 | 6.414 | 22.137 | 0 |
| DEPTH_TARGET | 5.159 | 5.569 | 24.604 | 0 |

DEPTH의 SOURCE T 중앙값2.181→5.569cm, P90 16.677→24.604cm로 악화했다. 보호 텐서 동일성이 원 정확도 보존을 뜻하지 않는다. 원 SOURCE reference는 등록 renderer의 physical R이며 source=True의 기존 Rx(pi)를 한 번만 적용했다. C1=62/C2=194, 원 YOLO label2028코너를 사용했으며 새 GT를 만들지 않았다.

학습3회/960updates/15,360 노출(실사7,680+source7,680), GPU 학습 wall합230.699초<1,200초. 원 trainer의 synthetic32 bookkeeping은64예제/fit, 총192로 최종 학생1248 평가와 분리했다. 정식 최종 학생1248, R0 추가288, EVAL R0캐시128 재사용; predictor 내부 warmup4도 별도다. 실제 F는 교사403+최종RGB1536+합성검사4=1,943회, R0-branch 고정384회는 별도이며 교사 기하333회다. 준비/입력·teacher/학습·최종추론 시간은 FINAL_SUMMARY에 분리했다. 추가 fit·교사 탐색·추론 재시도0.

## 원행, 보존, 재현

교사 `TEACHER_ROWS.csv`/`EVAL_ROWS.jsonl`, RGB 학생 `STUDENT_EVAL_ROWS.jsonl`, SOURCE `SOURCE_EVAL_ROWS.jsonl`, TRAIN `TRAIN_PROBE_ROWS.jsonl`, 학습CSV3개와 TRAIN_RECEIPTS를 보존했다. PROTOCOL/입력 SHA/추론 lock/검증 영수증과 코드가 함께 있다. 최종 가중치3개는 ignored 로컬 경로에 SHA 그대로 보존하고 Git에는 공개하지 않았다. RGB/depth/cache도 공개하지 않았다. 논문·PDF·LaTeX·참고문헌·기존 결과/자료/사용자 변경은 수정하지 않았다.

현재 환경 재현 명령은 `REPRODUCE.md`를 따른다. 기존 봉인 결과는 재사용하며 미완료 실행은 자동 재시도하지 않는다.
