# V217 실제 selected GT 독립 검산

PASS. 승인된 12 scene JSON의 원 SHA/size와 선택 TRAIN 값 36개를 독립 검산했다. 다른 프레임 값 519개는 구조 경계만 스캔하고 해석하지 않았다. 원 96행/U11/FIT64·DEV32, 15개 obj1 후보 및 12개 instance UNKNOWN을 보존했다.

GT/info와 mask 파일명 index 및 camera K/양의 depth_scale은 저장 결과와 일치한다. 실제 mask·depth·CAD·RGB는 열지 않았다. 전역 visible fraction이나 프레임 밖 bbox, valid-depth count 차이를 native E로 변환하지 않는다. instance/part와 픽셀 증거를 결합하는 다음 단계가 별도로 필요하며 새 라벨·학습 채택은 없다.
