# 저장 원행에 의한 v4 오류 분해

새 모델·PnP·학습·ray·RGB·GT 채점 없이 기존245장 원행을 한 번 조인했다. 
실제 반환 cf_extents로 W/D 분기를 비교했고 기본 반환의 거절 후보를 새 자세로 세지 않았다.

|v3→v4 상태|N3 대비 분기|경계 채택|n|Δ위치 cm|Δ회전 °|전체245 위치 기여 cm|전체245 회전 기여 °|
|---|---|---|---:|---:|---:|---:|---:|
|ADDITIONAL_NEW|SAME|ADOPTED_BOUNDARY|1|-1.385240|1.269925|-0.005654|0.005183|
|ADDITIONAL_NEW|SAME|NO_ADOPTED_BOUNDARY|18|2.523893|0.988695|0.185429|0.072639|
|ADDITIONAL_NEW|WD_SWITCHED|NO_ADOPTED_BOUNDARY|3|34.258770|83.369937|0.419495|1.020856|
|BOTH_BASELINE_RETURN|SAME|NO_ADOPTED_BOUNDARY|6|0.000000|0.000000|0.000000|0.000000|
|BOTH_NEW|SAME|ADOPTED_BOUNDARY|81|0.470340|-0.147994|0.155500|-0.048928|
|BOTH_NEW|SAME|NO_ADOPTED_BOUNDARY|127|0.391657|0.250277|0.203022|0.129735|
|BOTH_NEW|WD_SWITCHED|ADOPTED_BOUNDARY|3|-3.285387|-29.722436|-0.040229|-0.363948|
|BOTH_NEW|WD_SWITCHED|NO_ADOPTED_BOUNDARY|5|1.532957|50.953407|0.031285|1.039865|
|LOST_NEW|SAME|NO_ADOPTED_BOUNDARY|1|0.000000|0.000000|0.000000|0.000000|

모든 그룹의 기여도를 더하면 전체 원행 평균 차이와 일치한다. 음수는 N3보다 오차가 작다는 뜻이다.

경계 교체 없는 nativeH와의 같은 영상 차이, 같은 분기에서 양쪽 새 자세인 부분집합, 
채택/표시 코너의 상대 참조 정확도, 사용 대응점·최종 inlier의 정확/부정확/미상과 수·배치는 CHECKS와 ROWS에 보존했다.

W/D 분기 변화는 오차와의 연관이다. 실제 물리 분기가 틀렸다는 독립 인증이 아니며, 
경계 채택과 branch 선택은 같은 solver 안에서 상호작용한다. maskH·Base 제안 특징의 의존도 남아 있다.
기존 geometric proxy 점수와 알려진8px 기준을 재사용했으며 새로운 물리 소유권 정답·원인 비율을 만들지 않았다.
rotation≥45°는 이 표의 고정된 설명용 집계이며 설정 선택이나 실패 제외 기준이 아니다.

[프로토콜](FAULT_ISOLATION_PROTOCOL.json) · [원행245](FAULT_ISOLATION_ROWS.jsonl.gz) · 
[표 CSV](FAULT_ISOLATION_TABLE.csv) · [검산과 모든 그룹](FAULT_ISOLATION_CHECKS.json)
