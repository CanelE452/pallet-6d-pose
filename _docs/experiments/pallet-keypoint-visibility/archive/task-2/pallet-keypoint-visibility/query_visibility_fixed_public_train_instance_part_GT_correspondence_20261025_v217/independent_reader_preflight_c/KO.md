# V217 reader 독립 사전 검토

PASS. 원 12 TRAIN 프레임·96행·U11·FIT64/DEV32와 q/L/native/flags를 저장 원문과 대조했다. 선택 key만 JSON 값으로 해석하며, obj1 후보 15개를 보존하고 instance는 모두 UNKNOWN이다.

known scene GT 4개만의 실행 또는 root가 정확히 승인한 camera/info 8개의 first-opaque SHA intake 후 해석이 가능하다. 미확인 SHA는 파일 부재나 영구 진행 금지를 뜻하지 않는다. 누락·불일치는 대체 없이 중지한다.

실제 GT/이미지/배열/모델 파일은 열거나 stat/hash하지 않았다. mask/depth/CAD 진단과 nominal projection은 이번 reader 범위 밖이며 별도 실행 권한과 코드를 요구한다. 새 라벨·학습 채택은 없다.
