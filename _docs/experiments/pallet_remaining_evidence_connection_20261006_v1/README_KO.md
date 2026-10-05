# 잔여 근거 인계 연결·재집계·원고 반영 — 2026-10-06

**실제 원영상·기존 주석·저장 예측을 대조하고, 가능한 CPU 재집계와 LaTeX/Markdown 원고 반영까지 완료했습니다. 새로 클릭하거나 재학습하는 작업은 열지 않았습니다.** 전체 120/24 검수, 정지 잡음, 독립 실측 T/R은 근거가 없는 부분을 x로 유지합니다.

[최신 전체 원고](paper_updated/manuscript_ko.md) · [최신 보충 원고](paper_updated/supplement_ko.md) · [실제 삽입 receipt](PAPER_INTEGRATION_RECEIPT.json) · [39개 추가 수치 출처](PAPER_INSERTION_CELL_MAP.json) · [남은 칸과 이유](PAPER_GAP_MATRIX.json)

## 실제로 완료한 연결과 계산

| 항목 | 실제 확인·계산 | 원고 반영 |
| --- | --- | --- |
| 전달 ZIP | 631파일 크기·SHA-256·CRC 통과 | 출처 감사 기록 |
| 리프터 원영상·시간·예측 | 8,910프레임 픽셀·센서/host 시각 일치; 중복29행 보존 | 사례 본문·보충표 |
| 기존 full 6D 예측 | Base/N3 각각8,772개 유효; no-pose138 | 사례 본문·보충표; 참조0이므로 정확도x |
| 정사각형 기존 원사진·주석 | 119/119 해시 일치; 602/600점 모드 보존 | 사례 연결 설명·보충표 |
| 기존 12장·96점 결과 | 원시 예측에서 재계산하여 기존 결과와 정확히 일치 | 기존 표·12장 그림 보존 |
| 사용자 제외 5장 | 원계획 해시와 실제 제외 기록 연결 | 보충 설명·작업 원장 |
| 후속 72좌표 버전 | 동일12장의 별도 참조 버전 민감도 | 보충표; 미제출 초안, 공식 정확도 아님 |
| 중립 CAN 명령32구간 | 저장8,263프레임·각방법fresh8,125/no-pose138/held0·구간별분산 | 보충 가용성 표·그림; 정지 잡음 아님 |
| 독립 물리 참조 | 센서19사본/27논리출처 실제 파싱, 대상9,029행 연결0 | x 유지 이유를 본문·보충에 반영 |

[주석 연결·검산](annotations/FINDINGS_KO.md) · [원영상/119주석/기하/실측 감사](time_physical/REPORT_KO.md) · [중립 명령 전체32구간 계산과 원사진](neutral_intervals/README_KO.md)

## 12장 재사용과 원120/24 계획의 구분

고정 원계획은 120개 고유 사진에 주120건·반복24건의 총144작업입니다. 별도 완료12장의 원래66클릭+30PnP 좌표와 실제대상same12건을 재사용했습니다. 사용자 제외는 사진5장이고, 그중1장이 반복목록에도 있어 작업으로는6건입니다. 제외사진을 다시 넣지 않았습니다.

| 작업 상태 | 주120건 | 반복24건 | 합계 |
| --- | --- | --- | --- |
| 완료 탐색적 기하 자료 재사용 | 12 | 0 | 12 |
| 기존 사용자 제외 | 5 | 1 | 6 |
| 참조·실제 반복 기록 미확보 | 103 | 23 | 126 |
| 원계획 합계 | 120 | 24 | 144 |

반복목록의6사진은 완료12장과 겹치지만, 첫 좌표를 복사해 실제 반복 검수를 완료했다고 기록하지 않았습니다. 위126건은 원래 전체계약에서 아직 근거가 없는 범위이지, 이번에 사용자가126건을 새로 하라는 요청이 아닙니다. 기존12장과 정사각형119장을 다시 주석할 필요는 없습니다. 공식 승인 상태를 CLI가 새로 만들지 않았습니다.

과거312원JSON(같은세션67+다른촬영245)·Git blob67개·원ZIP2개·245JSON멤버·245원사진SHA를 실제 대조했습니다. 다른촬영245장은 원120장에 넣지 않았습니다. 173507:2910/3210 두 과거 후보도 원사진픽셀·전체물체코너대응의 근거가 없어 승인참조로 승격하지 않았습니다. [144작업별 원장](../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/annotations/FRAME_TASK_CONNECTION.csv).

## 후속72좌표 민감도: 기존96점과 분리

같은12장에 대해 후속 저장72좌표를 고정했습니다. 원래직접66좌표는 정확히 같고, 원래PnP였던6자리만 실제나중저장좌표로 바뀌었습니다. 자체가림24자리에는 새좌표를 만들지 않았습니다. 저장된 status=draft·evaluation_use=false를 유지하므로 미제출·미확인 저장본의 탐색적 민감도이며 공식 직접가시 정확도가 아닙니다.

| 방법 | 중앙값 px | P90 px | 전체72점 PCK≤10px | 실패 |
| --- | --- | --- | --- | --- |
| Base | 3.649195 | 8.878079 | 67/72 (93.0556%) | 0 |
| N3 | 3.990919 | 8.611929 | 69/72 (95.8333%) | 0 |

중앙값은3.649→3.991px로 악화, P90은8.878→8.612px·PCK는67→69점으로 개선되어 혼합 결과입니다. 개선33점·악화39점이며 좋은 참조 버전·프레임·seed를 고르지 않았습니다. 중앙값의차이+0.341724px와 짝지은차이의중앙값+0.169492px를 구분합니다. 기존96점 headline은 그대로입니다.

[72점 원시 오차 CSV](../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/annotations/LATER_DRAFT72_POINT_ERRORS.csv) · [세 버전·분모·해시·지표 JSON](../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/annotations/LATER_DRAFT72_SENSITIVITY.json) · [기존96점 비교12장](../pallet_paper_review_20261006_v1/LIFTER_12_ALL_IMAGES_KO.md)

## 중립 명령 자료와 독립 실측의 한계

CAN중립은 실제 상대정지의 정답이 아닙니다. 모든32개 구간을 그대로 계산했고 큰분산·실패를 제거하지 않았습니다. 전체9,029대상관측에 연결된 독립T/R참조는0입니다. 원시예측에는 full6D가 있어도 참조는 별도이며, 운용추정·같은점PnP·다른촬영의태그·synthetic값을 독립정답으로 사용하지 않았습니다.

![모든 중립 명령 후보의 출력 분산](neutral_intervals/images/all_32_command_spreads.png)

[모델표시 없는 원사진96장 미리보기](neutral_intervals/README_KO.md#원사진-확인용-미리보기)를 제공합니다. 시작·중간·끝사진이 전체구간정지를 증명하는 자료는 아닙니다. 이번에는 새 사람검수 화면을 열거나 사용자에게 재클릭을 요구하지 않았습니다.

## 원고·실행·검산 출처

[최신 본문LaTeX](paper_updated/main.tex)와 [보충LaTeX](paper_updated/supplement.tex)의 실제표·문장을 수정했습니다. 이전원고173파일은 원위치에서 해시를 보존하고 새복사본의8파일을 변경/추가했습니다. 신규연결수치39개를 [JSON포인터·단위·분모·seed](PAPER_INSERTION_CELL_MAP.json)로 연결했고 [교체patch](RECEIVER_EVIDENCE_PAPER.patch)도 제공합니다. 기존YOLO/DOPE/ResNet319장3seed·N0/N1·602/600·D/L/PoseFix319·학생128·시간표는 변경하지 않았습니다. PDF는 컴파일하지 않았습니다.

검증명령은 실제실행한 것과 같습니다. 아래후보재집계와 주석연결은 공개gzip/작은JSON 및 기존예측gzip만으로 clone에서도 검산됩니다. 원영상재디코딩 검사는 해시가맞는 원영상·기존데이터가 별도로 필요하며, 그 결과를 새모델실행으로 기록하지 않습니다.

```bash
python3 scripts/research/pallet_remaining_evidence_connection_20261006_v1/neutral_interval_analysis.py --verify-only
python3 -m unittest discover -s scripts/research/pallet_remaining_evidence_connection_20261006_v1 -p "test*.py" -v
python3 scripts/research/pallet_github_publication_20261006_v1/verify_published_evidence.py
```

[주석본실행 비용](../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/annotations/VALIDATION.json) · [시간/실측 본실행비용](../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/time_physical/VALIDATION.json) · [32구간실행 비용](../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/neutral_intervals/EXECUTION_COST.json) · [그림실행 비용](neutral_intervals/FIGURE_PROVENANCE.json)

본계산은 주석연결약1.3초, 원영상/정사각형/실측감사10.430초, 중립구간재집계1.35초, 원사진검산·그림5.54초였습니다. 병렬실행의wall시간을 합쳐 CPU사용시간·총대기시간이라고 부르지 않았습니다. 새학습0·optimizer update0·새모델추론0·새PnP0·장치제어0입니다.

## 남은x와 사람에게 필요한 최소 범위

원120/24가시코너·반복품질은 실제승인·반복기록, 정지잡음은 전체구간상대정지확인, 독립실측T/R은 같은대상·시각·좌표계의별도측정근거가 필요합니다. 이번 연결만으로 이 값을 숫자로 바꾸지 않았습니다. 현재원고에 필요한 완료된정적실험·12장기하보조·명령자료기술과 새정확도실험의추가근거를 구분했습니다. 사용자에게 새어노테이션을 요구한 수는0장입니다.

[원인계보고서](source_handoff/MISSING_INPUTS_KO.md)의 NOT_FOUND_LOCALLY는 송신Windows PC범위입니다. 여기서 찾아연결한12장·제외5ID·119주석을 다시미완료로 기록하지 않았습니다. [인계무결성과실제수신범위](HANDOFF_IMPORT_RECEIPT.json)를 확인할 수 있습니다.

기존 원고에는 세 도식의 LaTeX 원본만 있고 process_overview/refiner_architecture/symmetry_supervision PDF 그림은 아직 없습니다. 이는 이전 소스에도 있던 미생성 산출물이며 이번에 PDF를 생성하지 않았습니다. 원고의 페이지 모양·완성 PDF 검산은 수행하지 않았습니다.
