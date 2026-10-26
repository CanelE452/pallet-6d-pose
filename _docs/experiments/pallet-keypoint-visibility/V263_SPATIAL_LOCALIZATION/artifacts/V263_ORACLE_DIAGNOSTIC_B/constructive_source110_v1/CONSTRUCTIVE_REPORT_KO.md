# V263 constructive SOURCE 고정 후단 oracle

원 SOURCE 110base(FIT86+내부DEV24), 880 logical query groups/7,040 native rows를 모두 보존했다. BEFORE/COVER/NEAR_SHAM/FAR_SHAM과 두 원 query family의 alias가 있으므로 880개를 독립 실사 사례로 세면 안 된다. V255 신규 증강은 포함하지 않았다. 원 availability 6,976/7,040이며 원 U64개를 GT로 복구하지 않았다.

네 구성은 동일 V58의 finite9×2+별도 mask, 동일 2D/3D native index, 두 W/D 가설, SQPnP+기본 LM, 최소6 native 규칙을 사용했다. 원 centroid9가 없어 모든 구성에서 (0,0) finite placeholder와 invalid9[8]=False를 고정했다. 새 GT_R_CF가 CF3D를 CV raster의 r로 재투영함을 전행 확인하여 source=False로 실행했다. GT pose/선택된 축/GT permutation은 solver 초기화에 넣지 않았다.

VIS/BOTH는 partial-certified 규칙이다: identity가 인증된 E/S만 제외하고 D 및 UNKNOWN은 원 availability를 유지한다. 실제 packet에서는 identity/point visibility가 7,040점 모두 인증됐다. LOC는 원 availability∧physicalD∧두 noncollinear 기하 edge∧원 검색창 내∧finite r 조건의 5,841점만 정확한 anchor 투영으로 치환했다. 이는 물리 anchor·ray·edge에 대한 기하 oracle이며 이미지의 contrast나 국소 photometric localizability를 검증한 학습 성능이 아니다.

## 고정 분모880의 실제 결과

|구성|raw pose available|post W/D-normalized 5cm·5° 성공|physical canonical C1 5cm·5° 성공|raw 유지 fallback sensitivity의 post 성공|
|---|---:|---:|---:|---:|
|ORIGINAL|872|625|242|625|
|GT visibility만|868|621|243|625|
|direct-geometric 위치만|872|850|313|850|
|둘 다|868|861|313|865|

raw unavailable은 primary 분모에서 삭제하지 않았다. 최소6 미달은 각각8/12/8/12그룹이다. 별도 sensitivity는 oracle이 pose를 내지 못할 때 계산된 ORIGINAL pose만 그대로 유지하며 GT fallback은 없다.

W/D-normalized raw primary의 회복/손상은 VIS4/8, LOC227/2, BOTH242/6이다. raw incumbent 유지 sensitivity에서는 VIS4/4, LOC227/2, BOTH242/2다. 따라서 '가시성만으로 개선됐다'고 요약할 수 없다.

ORIGINAL의 T 평균/P95/최대는5.009/14.612/65.161cm, LOC는0.933/2.960/59.518cm, BOTH raw는0.178/약2.19e-11/55.730cm다. 큰 꼬리와 실패를 보존했다. 모든 role/각도축의 평균·중앙값·P90/P95·최대 및 paired transition은 CSV에 있다.

## 방향·native 대응 한계

post W/D-normalized GT는 실제 CFdims에서 사전결정한 Q_truth를 적용하여 전4구성에 고정했다. 각 예측 W/D 선택에 따라 GT를 돌리지 않았다. 이 기준을 물리 canonical 성능으로 부르면 안 된다. 원 canonical GT와 CF GT도 모두 별도로 저장했고 최소오차의 축을 고르지 않았다.

physical canonical C1 평균 R은 ORIGINAL91.83°, LOC91.80°, BOTH91.28°이며 P95는 거의180°로 남는다. 따라서 W/D-normalized 성공861/880을 원 native/canonical 방향 문제가 해결됐다는 주장으로 확대할 수 없다. 원 SOURCE symmetry_order의 literal 인증이 없어 C2/C4를 발명하지 않았고 symmetry-min 수치도 만들지 않았다. canonical80..100°와 >45° count는 별도 CSV로 보존했다.

이 결과는 기하적으로 인증한 위치를 공급했을 때 동일 후단의 개선 여지가 있다는 진단이며, RGB만으로 그 위치를 학습·관측할 수 있다는 증명이나 실사 전이/채택 결과는 아니다. 원 SourceDEV는 반복 노출된 내부진단이며 설정·문턱·epoch 선택에 사용하지 않았다. 새 모델/encoder forward·학습·CAL·RGB/depth/mask/mesh 본문·private current/held 접근은0이다.
