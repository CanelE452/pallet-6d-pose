# V233: 같은 local crop에서 고정 backbone과 마지막 qkv 적응 대조

두 referent를 분리한 SOURCE 결과다. 현재 실사 성공·모델 채택 결과가 아니다. 원 query/L/native와 문턱0을 유지했고 추가 평가 기반 조정은 하지 않았다.

원15040행은 SYN7040+YCB8000이다. known D/E7617행은 사용 불가 입력도 삭제하지 않고 양팔 출력에 보존했다. 나머지7423행은 원 S/U/제외 타깃−1이며 forward 대상이 아니다. 이7423은 모델 U가 아니다.

SYN main은 원영상 native witness0의 associated-point D/E다. literal q-center owner가 아니다. YCB AUX는 associated-r surface D/E이며 native UNKNOWN이다. 두 CM을 하나의 의미 과업으로 합치지 않는다.

|가족|역할|팔|N|CM [[DD,DE,DU],[ED,EE,EU]]|D P/R|E P/R|U율|균형 recall|
|---|---|---|---:|---|---|---|---|---|
|SYN_MAIN|FIT|control|4752|[[4525, 40, 0], [0, 187, 0]]|100.00%/99.12%|82.38%/100.00%|0.00%|99.56%|
|SYN_MAIN|FIT|treatment|4752|[[4531, 34, 0], [0, 187, 0]]|100.00%/99.26%|84.62%/100.00%|0.00%|99.63%|
|SYN_MAIN|DEV|control|1344|[[1273, 15, 0], [5, 51, 0]]|99.61%/98.84%|77.27%/91.07%|0.00%|94.95%|
|SYN_MAIN|DEV|treatment|1344|[[1273, 15, 0], [5, 51, 0]]|99.61%/98.84%|77.27%/91.07%|0.00%|94.95%|
|YCB_AUX|FIT|control|1124|[[840, 1, 0], [0, 283, 0]]|100.00%/99.88%|99.65%/100.00%|0.00%|99.94%|
|YCB_AUX|FIT|treatment|1124|[[840, 1, 0], [0, 283, 0]]|100.00%/99.88%|99.65%/100.00%|0.00%|99.94%|
|YCB_AUX|DEV|control|397|[[333, 7, 0], [57, 0, 0]]|85.38%/97.94%|0.00%/0.00%|0.00%|48.97%|
|YCB_AUX|DEV|treatment|397|[[333, 7, 0], [57, 0, 0]]|85.38%/97.94%|0.00%/0.00%|0.00%|48.97%|

|DEV 가족|범위|회복|악화|순증|양팔 모두정답|양팔 모두오답|
|---|---|---:|---:|---:|---:|---:|
|SYN_MAIN|ALL|1|1|0|1323|19|
|SYN_MAIN|D|1|1|0|1272|14|
|SYN_MAIN|E|0|0|0|51|5|
|YCB_AUX|ALL|1|1|0|332|63|
|YCB_AUX|D|1|1|0|332|6|
|YCB_AUX|E|0|0|0|0|57|

사전 수치 gate는 DEV macro balanced recall +3pp 이상이며 각 family/class recall 감소가2pp 이하여야 한다. 관측 gain=0.000pp, 수치 통과=False. 계산 결과와 무관하게 SYN donor/background와 YCB DEV의 교차 계보 독립성은 UNKNOWN이므로 current admission=false다. 새 팔·seed·epoch·문턱·데이터 선택은 하지 않았다.

SYN q–reference ≤3px/>3px 표는 target reference와 원 query의 거리에 따른 서술이며 좌표오차 보정이나 q-center ownership 증거가 아니다. YCB target sidecar에는 r 좌표가 없어 거리를 null로 보존한다. SOURCE DEV는 사전 노출/재사용됐고 fresh blind 평가가 아니다.

Root 봉인 metadata에 따르면 V1은 CONTROL1000 완료 후 TREAT 첫 backward에서 deterministic CUDA grid_sample backward 미지원으로 중단됐다(TREAT optimizer0/DEV0). V1 실패와 완료 CONTROL을 보존했다. SOURCEFIT5876의 old/new1186 특징과 CONTROL logits은 Tensor.equal·FP32 byteSHA 동일/maxabs0인 사전 검사를 통과했다. V2는 CONTROL checkpoint/log를 원 bytes 그대로 재사용해 newCONTROL0, TREAT1000을 실행했고 양 final1000 freeze 뒤 SOURCE 출력을 산출했다. 이는 실행 오류에 대한 동일 forward 수리이며 결과 기반 모델·문턱 선택이 아니다. 본 saved 분석은 해당 Root preflight/byte-copy metadata를 인용하고 모델 forward나 checkpoint bytes를 독립 재검산하지 않았다.

[전체 검산 수치](SUMMARY.json) · [strata CM/P/R/U](METRICS.csv) · [7617행 paired 결과](PAIRED_ALL7617.csv) · [원본 binding/검산 범위](COMPLETE.json)

같은 배포 입력에서 실제 새 모델 성능이 개선됐다는 결론은 이 SOURCE 결과로 내리지 않는다.
