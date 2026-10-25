# 공개 Pallox TRAIN conditional pixel AUX 준비

관측한 것은 코드와 화이트리스트 메타데이터뿐이다. TRAIN22 pair/44 view의 certificate는 각 view별 `visible_positive`, `visible_negative`, `full_positive`, `full_negative` bool[480,640] 4개를 제공한다. V246 코드의 canonical helper는 V=0/E=1/U=−1을 구성하며 O/full-negative와 SELF/native 의미 인증은 제공하지 않는다. 실제 certificate, RGB, GT, 모델은 열지 않았다.

제안은 known V 픽셀과 완전한 5×5 E core에서 별도 AUX 좌표를 뽑는 순수 함수다. pair/view/class별 quota·seed·AUX L은 기본값 없이 Root가 읽기 전에 고정한다. quota가 부족하면 shortfall을 남기고, UNKNOWN을 보충하거나 결과를 보고 quota·문턱을 바꾸지 않는다. 원352 query의 q/L/U/native/index는 그대로이며 AUX 좌표는 teacher-conditioned 분포임을 기록한다.

이 teacher는 조건부 expected-object pixel support이며 original initializer q8의 native 대상과 대응한다는 증거가 아니다. AUX에 임의 native slot/condition32를 복사하면 안 된다. Root가 AUX query의 배포 가능한 RGB/좌표/context 입력 adapter를 별도로 고정해야 한다. pose/depth/mask/target ID/teacher box는 forward feature가 될 수 없다. 현재 순수 함수는 해당 adapter나 학습 recipe를 선택하지 않는다.

원샘플의 SELF·target-instance·좌표오차 권한이 없으면 UNKNOWN을 유지한다. 이는 conditional pixel AUX 학습 자체를 완벽 physical gold 부재 때문에 금지한다는 뜻은 아니다. 실제 추출은 별도 exact Root scope/entry 이후 허용된 eight Boolean members만 열고 수행한다. 이 준비는 실행·학습 admission이 아니다.
