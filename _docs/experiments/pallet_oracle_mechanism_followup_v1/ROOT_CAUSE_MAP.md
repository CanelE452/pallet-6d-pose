# 원인 지도 — 관찰과 확정을 분리

아래는 새 실험 전에 확인한 근거다. 개입 결과는 cycles와 최종 REPORT에 추가하며 과거 주 결과를 덮어쓰지 않는다.

| 원인 후보 | [확인] 관측 | 경쟁 설명 / 확정하지 못한 것 | 판정을 바꿀 대조·oracle 연결 | 우선순위 |
| --- | --- | --- | --- | --- |
| 후보 선택 손실 | Plastic REF AUC0.359016→동일후보oracle0.450094, Wood0.665033→0.667400 | GT로 고를 수 있다는 것과 현재입력으로구별할 수 있다는 것은 다름 | 같은9점/penalty/후보에서 Huber12 대조 C1; C1선택변경0으로 gap회수0 | 현재 단계의단순강건점수는닫음 |
| 교사좌표 오류 | 교사가 모든평가점을 개선하는 것은 아님; Plastic검수66 teacher50/66, Wood검수가시점출처0 | unlabeled217/361의 물리정확도는 모르며 teacher in-sample9/38은 독립calibration아님 | 같은4출력 whole/perpointoracle·교사/학생교차표, legacy/verified분리 | 계량, 새confidence필터자동도입보류 |
| 타깃 전달 부족 | Plastic RAW학생→ref4.078px vs REF학생3.070px; Wood3.689→3.325px. 보정방향양의비율82.64%/77.18%, 투영분율중앙값0.319/0.129 | 전달0이 아니며 native잔차는 실제정확도아님. 표현/증강/손실/모순타깃경쟁 | 기존640연장결과재사용, TRAIN기하증강대조C2 | 실행대상 |
| 기하증강 민감성 | 고정TRAIN32실사노출에서 REF bbox정규화잔차중앙값: Plastic .01234→기하OFF .01057, Wood .01089→.00940 | Plastic평균차이는 한frame의큰오류에지배됨; 최고score검출전환가능성. 증강을끄면 일반화손상가능 | real-onlytranslate/scaleOFF, source/HSV/목표/updates유지, RAW/REF둘다학습 | C2명세후실행 |
| source/real gradient 충돌 | 동일pose/flow공간step0에서 부호가batch/재료에따라혼합. Wood source/realnorm은일부1–3배 | 음의cosine단독은손상원인이아님, EMAfrozenprobe는원래online경로아님. source제거옛실험은real노출2배confound | 현재노출·loss scale을기록. PCGrad·replay변경을동시에추가하지않음 | 관찰, 새개입근거불충분 |
| 좌표·참조·대칭 정의 | 실사legacyxy→기존pose AUC1.0은같은기하경로순환성. 합성64중C1 6개는180도방향모호성, exactrenderer대응이면최대재투영8.13e-6px | 높은순환정합은물리6D보장아님. C1의삽입방향을C2로승격해지표를올리지않음 | known-correspondence sanity와production정확점sanity분리; 합성axis잘못된역할비교NA | 원인범위제한 |
| 검출·인스턴스 | Plastic8/128미매칭, Wood45/45매칭; detector는동결 | 검출부고정은병목0의증거아님 | 미매칭에한한저장후보box/keypointoracle, GTcrop재추론없음 | 조건부저비용진단 |

## 기존 결과가 실제로 배제한 범위

같은낮은LR의320추가만으로 검수66에서보정학생우위가생기지않았다. 이는 모든최적화의불가능성을뜻하지않는다. 과거source-off·agreementmask·multi-teacher융합·local snap의조건과무실행학생을 PRIOR_ATTEMPTS에서구분했다. 같은설정재실행대신현재검증가능한한축을시험한다.

## 현재 정보의 한계

평가오류·oracle선택을학습표적이나가중치로전달하지않는다. 새직접클릭·깊이·mesh·물리pose가없어판정못하는항목은 REFERENCE_LIMITED/UNRESOLVED로남긴다. 동일계산의낙관치를더해서총headroom을만들지않는다. 후보를바꾸는미래방법은현재fixed-set회수율로채점할수없다.
