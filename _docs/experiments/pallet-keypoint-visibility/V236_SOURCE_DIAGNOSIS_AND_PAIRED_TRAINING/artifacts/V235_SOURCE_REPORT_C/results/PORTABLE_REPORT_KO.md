# V235: 같은 local crop·고정 head에서 SOURCE counterfactual 데이터 대조

원 SOURCE 7,617행의 저장 결과다. SYN main은 intended-native witness0 associated-point D/E, YCB AUX는 associated-r surface D/E(native UNKNOWN)로 분리한다. 두 referent의 CM을 하나로 합치지 않는다. 원 q/L/native·문턱0은 그대로이며 새 실사 성공·모델 채택을 뜻하지 않는다.

원 전체 15,040행 중 known D/E 7,617행은 입력 불가도 삭제하지 않고 보존한다. 나머지 7,423행은 원 S/U/제외 target−1로 forward 분모가 아니며 모델 U와 다르다. 학습 증강 COVER/SHAM 행은 별도 FIT counterfactual 자료로서 아래 평가 분모에 더하지 않는다.

|가족|역할|팔|N|CM [[DD,DE,DU],[ED,EE,EU]]|D P/R|E P/R|U율|균형 recall|
|---|---|---|---:|---|---|---|---|---|
|SYN_MAIN|FIT|control|4752|[[4525, 40, 0], [0, 187, 0]]|100.00%/99.12%|82.38%/100.00%|0.00%|99.56%|
|SYN_MAIN|FIT|treatment|4752|[[4520, 45, 0], [0, 187, 0]]|100.00%/99.01%|80.60%/100.00%|0.00%|99.51%|
|SYN_MAIN|DEV|control|1344|[[1273, 15, 0], [5, 51, 0]]|99.61%/98.84%|77.27%/91.07%|0.00%|94.95%|
|SYN_MAIN|DEV|treatment|1344|[[1271, 17, 0], [4, 52, 0]]|99.69%/98.68%|75.36%/92.86%|0.00%|95.77%|
|YCB_AUX|FIT|control|1124|[[840, 1, 0], [0, 283, 0]]|100.00%/99.88%|99.65%/100.00%|0.00%|99.94%|
|YCB_AUX|FIT|treatment|1124|[[840, 1, 0], [1, 282, 0]]|99.88%/99.88%|99.65%/99.65%|0.00%|99.76%|
|YCB_AUX|DEV|control|397|[[333, 7, 0], [57, 0, 0]]|85.38%/97.94%|0.00%/0.00%|0.00%|48.97%|
|YCB_AUX|DEV|treatment|397|[[329, 11, 0], [57, 0, 0]]|85.23%/96.76%|0.00%/0.00%|0.00%|48.38%|

|DEV 가족|정답 범위|회복|악화|순증|모두 정답|모두 오답|
|---|---|---:|---:|---:|---:|---:|
|SYN_MAIN|ALL|3|4|-1|1320|17|
|SYN_MAIN|D|2|4|-2|1269|13|
|SYN_MAIN|E|1|0|1|51|4|
|YCB_AUX|ALL|2|6|-4|327|62|
|YCB_AUX|D|2|6|-4|327|5|
|YCB_AUX|E|0|0|0|0|57|

사전 gate는 두 family DEV balanced recall 평균 +3pp 이상, 각 family/class recall 감소 2pp 이하이다. gain=0.113pp, 수치 통과=False. 이 계산으로 모델·seed·문턱·데이터를 선택하지 않는다. current admission=false, selected arm=null을 유지한다.

학습 provenance: CONTROL 새 optimizer update=0, TREAT 새 update=1000. CONTROL 원본 재사용 검증은 Root의 봉인 QA binding을 인용한다. 두 checkpoint metadata와 새 head DEV-before-freeze=0을 확인한 뒤에만 저장 CSV를 읽었다. 고정 backbone/같은 head의 data-only 비교이며 이번 trial에 LoRA 실행을 가정하지 않는다. V233의 runtime failure/수리는 역사적 선행 기록이고 V235의 새 실패로 복제하지 않았다.

성공 counterfactual parent=5367, 실패 masked parent=39, COVER/SHAM 학습 증강 행=10734. 원 7,617행 평가와 구분한다. 합성 가림은 새 자연 관측 또는 human gold가 아니다.

SYN ≤3px/>3px 표는 원 sidecar의 q-reference 거리 진단이다. 원 V233 YCB sidecar의 r와 거리는 계속 null로 보존한다. 별도 승인된 YCB1521 export를 사용했으면 SUPPLEMENTAL strata에만 추가하며, GT 좌표를 모델 입력·q 보정·교사 교체로 사용하지 않는다. q-r 거리는 실제 detector 오차와 같지 않다.

SOURCE DEV는 이미 노출된 진단이며 fresh-blind가 아니다. 본 검산은 저장 CSV/JSONL과 완료·freeze metadata의 산술/정합 범위다. 원 RGB·GT·NPZ·weights·prefix·모델 forward·신경망 loss 전체를 재실행하지 않았다. 새 current/held 평가·PnP도 없다.

[수치 JSON](SUMMARY.json) · [CM/P/R/U·거리 strata](METRICS.csv) · [원7617 paired 출력](PAIRED_ALL7617.csv) · [binding·검산 범위](COMPLETE.json)
