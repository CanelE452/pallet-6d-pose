# V263 current223 고정 후단 utility

새4학습 설정과 SOURCE 선택이 모두 동결되고 현재1784 예측도 동결된 후에만 실행했다. 현재223장 입력 geometry와 원9점/K/등록치수의 동일성을 검산했고 원N0를 재생한 전223장의 pose차이는0이다. 원 centroid9 값과 true지원은 전구성에서 그대로 유지했다.

RGB classification/crop availability와 solver coordinate support는 다르다. 기존V60처럼 ORIGINAL은 유한9점 전부를 유지한다. LOC는 frozen accepted점만 final_xy로 바꾸며 원RGB U는 원q로 남는다. VIS/JOINT는 RGBavailable이고 frozen semanticE(logit>0)인 native만 제외한다. D/U는 원solver지원을 유지한다. 동일 native index를2D/3D에 적용하는 V58두 W/D가설+SQPnP+기본LM/min6를 바꾸지 않았다.

전223×4개의 후단pose를 먼저 저장·SHA봉인한 뒤 기존목재FIXED_REFERENCES와정사각형V50bundle을 읽었다. GT-best branch/축/심메트리 선택은 새로 하지 않았다. 기존목재C2/정사각형C4와 V60의 strict T<5cm∧R<5°를 유지하고 canonical C1 각도도 별도저장했다. 목재는 재구성 기하reference, 정사각형은 기존수동키포인트PnP reference이므로 독립물리6D GT 성능이라고 부르면 안 된다. 현재223장은 반복개발자료이며 blind test가 아니다.

|집합|구성|성공/n|평균T cm|P95T cm|평균R°|P95R°|fallback|
|---|---|---:|---:|---:|---:|---:|---:|
|ALL223|ORIGINAL|87/223|12.345|38.906|7.828|81.612|0|
|ALL223|LOC_ONLY_INITIAL_GEOMETRY|123/223|11.899|38.626|8.296|82.231|0|
|ALL223|VIS_ONLY_SELECTED|87/223|12.330|38.906|7.825|81.454|0|
|ALL223|JOINT_SELECTED|108/223|11.468|35.485|8.078|81.576|120|
|SQUARE118|ORIGINAL|30/118|9.378|23.233|3.344|5.417|0|
|SQUARE118|LOC_ONLY_INITIAL_GEOMETRY|72/118|6.934|25.543|3.421|5.208|0|
|SQUARE118|VIS_ONLY_SELECTED|30/118|9.371|23.233|3.351|5.417|0|
|SQUARE118|JOINT_SELECTED|50/118|8.264|23.233|3.332|5.417|90|
|WOOD_PRIMARY105|ORIGINAL|57/105|15.679|84.251|12.867|85.633|0|
|WOOD_PRIMARY105|LOC_ONLY_INITIAL_GEOMETRY|51/105|17.478|119.577|13.774|85.539|0|
|WOOD_PRIMARY105|VIS_ONLY_SELECTED|57/105|15.655|83.726|12.852|85.633|0|
|WOOD_PRIMARY105|JOINT_SELECTED|58/105|15.068|61.144|13.411|85.633|30|

ORIGINAL→LOC의 합산 회복/손상은50/14이나, 목재2/8과 정사각형48/6으로 방향이 다르다. 목재 성공57→51, 평균T15.679→17.478cm, P95T84.251→119.577cm로 악화됐다. 전체의평균R7.828→8.296°와 T>50cm 새진입1건도 기록했다. 합산성공123/223만으로 채택을 주장할 수 없다.

ORIGINAL→JOINT는25/4(목재5/4, 정사각형20/0)이고 성공108/223이다. 단, 최소6점 미달로120/223에서 기존N0를 그대로 유지했다. 예측masked rawpose와실제fallback pose를 모두保存했으며 GT fallback은 없다. 전체평균T/P95T는개선됐지만 평균R은7.828→8.078°로악화했고 큰회전오류 새진입1건이 있다.

VIS의성공수변화는0이다. 실제semanticE제외수와 RGBavailability·min6/fallback은 SUPPORT_COUNTS.csv에 보존했다. SOURCE에서동결된선택rule의이름을현재결과를보고바꾸지 않았고 C가내보낸semantic결과를그대로후단에사용했다.

모든집합의중앙값/P90/P95/최대T·symmetryR·canonicalR, near90, 큰오류수, 회복/손상은원CSV에있다. raw solver실패와fallback을삭제하지 않았고미선택crop/head를현재결과로재선택하지 않았다. 이작업에서원RGB·encoder/head추론·추가학습·CAL적합·문턱/solver탐색은0이다. 기존reference R/t만예측동결후늦게읽었다.
