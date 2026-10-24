# V220 동결 비교 준비

V213의 원 입력·F64 q8 문맥·decoder·평가 산술을 재사용한다. 원 current1784와 public360 입력, q/L/native0..7 및 U45/U108을 유지하며 두 새 모델 C64/C65만 forward한다. 기존14 모델은 저장 출력으로만 읽고 전체16 표를 남긴다. 주요 비교는 v61/legacy/C55/C63/C64/C65이다.

primary D/E1237(D1147/E90), WOOD/SQUARE, native8, SELF21/136, 기존 reference≤3px D461와 manualnear376은 서로 분리한다. 기존 annotation/reference는 양 모델의 모든 예측 seal 뒤 saved evaluation entry에서만 연다. 기존 좌표·location·owner 관련 열은 진단이며 새 primary 제외조건이나 physical gold 판정이 아니다. public owner/crop 약한 권위와 legacy missing을 그대로 둔다.

새 모델은 V219 내부 STRONG_ONLY_CONTROL→C64, STRONG_PLUS_EXPECTED_PART_WEAK→C65로 연결한다. 같은 RAW118359/native/q8 아키텍처이며 새 학습·SAM·initializer·crop·PnP·GT recenter·문턱선택은 없다. 공개 SOURCE weak의 GT-derived 초기 q8와 실제 평가 original initializer q8의 권위 차이를 숨기지 않는다. 이 준비에는 실제 전이 결과가 없다.

준비 폴더만 작성했다. V220 과학 phase, scope 및 실행승인은 root가 이후 별도로 만든다. false entry의 모든 SOURCE 최종/QA/barrier/terminal binding과 기존 saved seals를 root가 채우기 전 실행할 수 없다. source score에 따라 모델을 고르지 않고 두 모델을 모두 평가·보존한다.

실패 이력: 생성기의 마지막 .patch hash 출력 suffix guard만 중지했다. 생성된 코드/프로토콜은 이미 완성되어 있었으며 원 생성기와 실패영수증을 보존했다. 순수 기능10검사와 원19함수 AST 정합은 exit0이다. 보호 배열·라벨·점수·원 영상·weight 접근은 0이다.
