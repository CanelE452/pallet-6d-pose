# Wood 짝지은 학습 완료

신규 fit은 WOOD_RAW_LR5·WOOD_REF_LR5 두 개뿐이며 각5epoch/320update를 실제 optimizer hook으로 확인했다. 고정 Replay9장/38코너 교사와 동일 R0 초기값을 사용했다. 동일 후보1,000장→공통통과676장→고정표본고유361장/실사512슬롯, 기존합성512슬롯이다. RAW/REF 이미지순서·박스·support·source replay 일치, 유효좌표만 다름을 검증했다. 744개 protected state가 저장checkpoint에서도exact 유지됐다. RAW46.7초·REF46.4초, final last만 사용했다.

기존Plastic에는 없던 공통감독최소6점 검사가 실행전에발견되어 제거했다. 기존Plastic도4/5점 이미지를사용한다는원자료로정정했으며 원판정은로컬에보존했다. 승인집합676장은전부유지했고 threshold·교사·학습량을바꾸지않았다. 성능결과를본후재시도한학습은0회다.

실제 checkpoint/입력목록/출처해시는 로컬의 상세재현파일에보존했다. 원본RGB·원시좌표·카메라행렬·모델가중치는공개하지않는다.
