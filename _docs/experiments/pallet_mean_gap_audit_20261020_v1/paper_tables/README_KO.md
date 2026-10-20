# 논문 대화용 검증 표 패키지

원고를 편집하지 않고 같은 검증 출력에서 표를 조립했다. **평균 우위 확보 실패**를 포함한 부정 결과를 보존한다. 모든 수치는 PRIMARY192와 별도CAL46/TEST49/REMAINDER143으로 나뉘며 이전245장 N3 결과를 섞지 않는다.

| 표 | 파일 | 범위 |
|---|---|---|
|1|[자료·계약](TABLE_01_DATA_AND_PROTOCOL.csv)|교정/학습/평가·고정 transfer·독립GT 공백|
|2|[전체 추정 비교](TABLE_02_PRIMARY_AND_PARTITIONS.csv)|실제 R0/SubPix_ONLY/N3_ONLY/PoseFix_RAW/PF+SubPix/N0/E0/D1|
|3|[현재 세 backbone](TABLE_03_ALL_POPULATION_METRICS.csv)|완료된 frozen original D1 재사용;60행/미산출 유지|
|4|[단계·고정 은행](TABLE_04_STAGE_AND_FIXED_BANK.csv)|N0/E0/D1-H0/D1-H1, N0대비회복손상|
|5|[COMMON/BI와 no-B 구별](TABLE_05_COMMON_BI_VS_NO_B.csv)|동일정보 matched 정책과 특징차단의 해석 분리|
|6|[전체분포](TABLE_06_ALL74_POPULATIONS.csv)|평균/중앙/P90/P95/미산출/성공; 별도seed/recording/session/전이표|
|7|[공정 비용 범위](TABLE_07_FAIR_COST_SCOPE.csv)|실제 typed CPU/벽시간과 미계측FPS·불일치학습예산 구별|

SubPix_ONLY는 실제 저장된 SUBPIX_F, N3_ONLY는 N3_F다. N0=N3+SubPix+F를 단독으로 바꿔 부르지 않았다. 원48/95 또는245장 수치를 누락 셀에 넣지 않았다. R0/SubPix/N3/PoseFix_RAW는 원봉인 반환 자세에서 독립 재계산하고 원 scalar와 검산했다. 세 backbone표는 오늘 완료된 출력3570행/60집계/96전이를 재집계하여 재사용했으며 추가 forward/fit/PnP0이다.

DOPE는PRIMARY192중146자세 산출/46미산출이다. 성공은192분모, 연속평균은146조건부이므로 YOLO/ResNet18의192평균과 단순 우열로 해석하지 않는다. 보조공통146은 기존 초기가용성으로 고정된 별도집합이며 전체192를 대체하지 않는다. PRIMARY46미산출은 검출실패3/<6지원코너13/현재F입력계약거부30이다. 마지막30을 수치PnP실패라고 이름 바꾸지 않는다. frozen F가 rectangular 입력에서9개슬롯 모두유한을 요구한 기존 진단이며 이번에 고치지 않았다.

COMMON/BI표의 F는 새 fullnormal bank/동일cue 정책 비교이며 모든B/I 제거군이 아니다. no-B는 원 D1 bank에서 B유래 특징72/188을0으로 차단한 작은선택기 ablation이다. 목적과 은행이 달라 같은 B효과라고 합치지 않는다. 각행의 same-seed counterpart와 원참조를 따른다.

원자료·모델·R/t는 넣지 않았고 source SHA는 [SOURCE_MANIFEST](SOURCE_MANIFEST.json), 누락은 [MISSING_CELLS](MISSING_CELLS.csv)다. 별도GPT에서 교수님 의견/기존 자료를 합칠 때는 다음 사실을 유지해야 한다. 문제는 가림/관측대응과 선택의 실패이며 성공률 우위가 평균정확도 우위를 뜻하지 않는다. 목표는 같은정보·참조·분모에서 한최종자세의 평균위치/회전·성공/tail을 함께 개선하는 것이다. 현재 입증된 기여는 상태별 관측/후단/고정후보 선택의 명시적 구현·진단과 검증 프로토콜 범위이며, 물리owner 해결/평균우월성/새실사일반화는 입증하지 않았다. 교수님 의견·논문 reference 원문은 이 실행에 제공되지 않아 임의로 반영했다고 쓰지 않았다. 논문구성과 문헌 수정은 별도GPT 대화에서 이어간다.

## 기본 비교의 PRIMARY192 요약

| 경로 | 성공/192 | T평균 cm | R평균 ° | T/R 산출 |
|---|---|---|---|---|
| R0 | 66/192 | 14.535790 | 14.281903 | 192/192 |
| SUBPIX_ONLY | 98/192 | 12.793573 | 12.951672 | 192/192 |
| N3_ONLY | 75/192 | 13.059080 | 11.847055 | 192/192 |
| POSEFIX_RAW | 80/192 | 13.835732 | 11.639767 | 192/192 |
| POSEFIX_SUBPIX_PF_ANCHOR | 112/192 | 9.848271 | 10.538719 | 192/192 |
| POSEFIX_SUBPIX_YOLO_ANCHOR | 111/192 | 9.377412 | 10.593944 | 192/192 |
| N0 | 108/192 | 11.305759 | 10.631096 | 192/192 |
| E0 | 114/192 | 14.112109 | 13.001138 | 192/192 |
| D1_H1_MLP_s20261018 | 120/192 | 12.072574 | 11.548887 | 192/192 |
| D1_H1_MLP_s20261019 | 117/192 | 12.251478 | 12.454362 | 192/192 |
| D1_H1_MLP_s20261020 | 121/192 | 11.406796 | 10.233225 | 192/192 |

## 세 backbone의 현재D1 대표seed18

| backbone | 성공/192 | 미산출 | T/R 가용 n | T평균 cm | R평균 ° |
|---|---|---|---|---|---|
| yolo | 120/192 | 0 | 192/192 | 12.072574 | 11.548887 |
| dope | 72/192 | 46 | 146/146 | 20.083423 | 16.716291 |
| resnet18 | 90/192 | 0 | 192/192 | 20.502290 | 20.351206 |

실행 주체·시각·기준commit와 미커밋 소스 overlay 구분은 [THREE_BACKBONE_PROVENANCE](THREE_BACKBONE_PROVENANCE.json)에 기록했다. 오늘03:01–04:56UTC 실행,05:18UTC 전체완료한 현재D1+H1이다. 이 작업은 기존 완료출력을 읽기 전용 재사용했다. 기준93c24d14만으로 새 pipeline소스가 모두포함된다고 주장하지 않는다.

독립 삽입용 LaTeX는 [latex/](latex/)의 tab01–tab07 파일이다. 표3의 DOPE 평균은146가용집합 조건부이며 본문/캡션에서 이 한계를 유지해야 한다. 이전 원고파일에는 삽입하지 않았다.

COMMON→BI와 원D1→no-B의 같은seed·같은영상 회복/손상·연속차는 [TABLE_05_PAIRED_CONTRASTS](TABLE_05_PAIRED_CONTRASTS.csv)에 따로 있다.

[IEEE Sensors 8쪽 목표: 본문핵심/보충자료 구분](IEEE_SENSORS_8PAGE_HANDOFF_KO.md). 이는 사용자 목표이며 공식 페이지제한 주장이 아니다.
