# V205 원 TRAIN 입력 프리뷰 코드 독립 사전검토

PASS. 최종 V2 코드·프로토콜·고정 선택 메타데이터만 확인했고 실제 원RGB/배열/header/GT/모델/점수는 읽지 않았다.

- 원 SOURCE185 입력과 원 component TRAIN을 META6로 조인한 뒤 기존 direct12프레임을 제외한다. source_id SHA256와 prefix2/2/1/1 규칙을 사용하며 점수·옛 weak label로 고르지 않는다.
- 고정 프레임17/67/150/155/187/205는 6개 서로 다른 graphcomponent, 48native슬롯, 입력U8을 보존한다. graphcomponent 차이는 독립 촬영 인증이 아니다.
- 실제 실행은 선택6 canonicalRGB NPY와 선택6 원 FRAME INPUT_ONLY NPZ만 연다. 원 shared1480 배열이나 held 이미지, SAM/GT/teacher/model 파일을 따라가지 않는다.
- 원 q/L/native/U를 그대로 표시하고 null/offframe을 다른 좌표로 대체하지 않는다. bbox는 명시적 deployed hint만 허용하며 이번 선택에는 없어 null이다. q8 context bbox나 GT bbox를 대신 쓰지 않는다.
- fullRGB decoded SHA, crop96/valid SHA, globalindex·constructed flag를 검증한다. 깨끗한 96패널과 표시용 확대한 십자 패널을 구분하며 observation template은 UNREVIEWED·training_admission=false이다.

독립 가상 기능검사18개 PASS: 단순 scalar SHA순서 대조, source185×8 가상 join, null/finite U, META 드리프트 거부, clean96 ROI 원바이트·native순서, GT/owner/SAM 필드 비노출 등을 실제 순수함수에서 확인했다. 프리뷰 전체 actual 실행이나 새 라벨 생성은 하지 않았다. 실제 저장 PNG 정합 검사는 root preview COMPLETE 뒤에 별도로 수행한다.
