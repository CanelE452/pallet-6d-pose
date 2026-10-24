고정 SOURCE HOLDOUT 평가 도구를 준비했다. V194 component 역할을 먼저 적용한 실제 metadata는 18부모·72장면·576 native이며, NOMINAL SELF 및 BEFORE-anchor 입력은 1,152행이다. 원 HOLDOUT augmentation 전체는 9,216행이다. 원 own-input U48, 매핑 입력 U80도 보존한다. 과거 source 문서의 split=TRAIN 표기는 별도 이력이며 현재 HOLDOUT 역할을 바꾸지 않는다.

저장 GT18은 path와 기록 SHA만 고정했다. 실제 GT·원 RGB/NPZ·예측 점수·가중치는 읽지 않았다. 이후 별도 root entry와 V212 두 모델 final4000 봉인 및 독립 fit PASS 뒤, 실제 RGB 장면의 records_CF point0를 원 image/anchor 관계로 조회한다. q/L/native/U는 유지하고 저장 reference가 원 frame 및 실제 crop canvas 안인지 같은 사전 조건으로 평가한다. S/U/누락/밖 reference는 E로 만들지 않는다.

원 전체 SYN 저장 예측 두 개만 row_id 및 META6로 연결한다. all576/all1152와 D/E 평가 가능 분모를 분리하고 native/canonical/재질/aspect/부모·variant별 표를 저장한다. TRAIN6096의 apparent fit과 혼합하지 않는다. C62·C63은 성적과 무관하게 모두 평가하며 C63은 사전 primary다.

이는 합성 SOURCE 내부 기하 point 일반화 진단이다. 기존 SOURCE HOLDOUT은 과거 노출되어 새 blind 평가가 아니며 실제 물체·사람 native gold가 아니다. HOLDOUT 결과로 모델/문턱/데이터를 선택하거나 학습에 편입하지 않는다. 12개 순수 가상 검사에서 역할 보존, image/anchor 분리, point/region 분리, 원 U 및 canvas 경계, 두 final4000 gate, decoder, lexical금지와 symlink 거부를 확인했다. 실제 평가는 아직 실행하지 않았다.
