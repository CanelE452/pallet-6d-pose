# V217 실제 selected GT 독립 QA

PASS. 승인된 원 scene JSON 12개를 독립 재해시하고 고정 TRAIN 값36개를 선택 파싱했다. 다른 값519개는 경계만 스캔해 의미 해석하지 않았다. 원96행/U11/FIT64·DEV32와 obj1 후보15개·instance UNKNOWN12개가 유지됐다.

첫 checker는 bbox sentinel `[-1,-1,-1,-1]`을 잘못 거부했다. 원코드·실패를 보존하고 root 승인 V2에서 zero valid/visible count를 동반한 정확한 sentinel만 UNKNOWN으로 수용했다. V2 actual은 exit0이다. 원 GT/생산자/라벨 수정 및 producer 재실행은 없다.

GT/info/mask index와 camera K/양의 depth_scale은 정합하다. 그러나 mask·depth·CAD·RGB 픽셀, calibration 정확도와 instance identity는 검증하지 않았다. sentinel2개·bbox 음수 원점5개·visible>valid-depth count2개 및 global visible fraction을 native E로 바꾸지 않는다. 새 라벨·학습 채택은 없다.
