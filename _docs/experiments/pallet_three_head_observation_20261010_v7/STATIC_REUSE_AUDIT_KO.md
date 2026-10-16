# v7 실사 실행 전 코드·입력 재사용 감사

현재 단계는 source CAL 준비 완료 및 실사 평가 전이다. 아래 통과 상태를 실사 성능 개선으로 해석하지 않는다.

- 기존 감독 수치 실패를 NONE으로 학습한 문제는 별도 연구 selector의 training.targets() 문제였다. main의 고정 N3→cornerSubPix를 수정하지 않는다. 실제 경계·깊이로 수정한 READY 감독, 원 동일 초기화·배치 순서·각3000/batch16의 완료된 세 checkpoint를 재사용한다. 이번 추가 학습·RGB·seed는0이다.
- corrected checkpoint SHA는 GEOM d188dcc68bd8795c88232d5bf1b85259d684695723b96809017abd47d6ac009b, NO_ROLE 9c52a2e2036ee8f65ba2e191bdcbebe93ac3ecde60a6aa31e71a4ffbadc0f835, ROLE 882a6eddc964258abfbc1ad3baeee380ab9fac42b3b90c7f64166bd98d777d5d다. 마지막 step3000, 파라미터5890의 metadata/완료 영수증을 모델 로드 전에 검증한다.
- 원 model.forward의 채널 처리를 그대로 쓴다. GEOMETRY_ONLY는 영상/neck0:19를0으로 하고 geometry19:25 및 role25:28을 유지한다. IMAGE_NO_ROLE는 role25:28을0으로 한다. IMAGE_ROLE는 모든 채널을 유지한다. 가중치만 바꾸고 ROLE 모드로 세 head를 부르는 방식은 사용하지 않는다.
- 기존 CAL128/family split/기존 fp16 feature cache/READY 타깃만 사용해 GEOM·NO_ROLE 각각8batch16=128exposure를 실제 실행했다. 합계16forward/256exposure, ROLE 신규0이며 이전9파일을 byte-exact 복사했다. source calibration protocol e1978ad4439e5937e943c2e017600bb535a9297d4206daea29b706dc1278131e는 모델 실행 전 고정했다. 이 준비는 기존 P0의 외부 가림·방해물·저대비 통제 변형 검증을 대신하지 않는다.
- source 계약9검사와 CPU PnP/공급13검사를 각1회 실행하여 통과했다. CPU 실제 OpenCV Generic1028/LM142/projectPoints48; detector·N3·학습 head·실사·GT·rays는0이다. dispatch stub과 실제 수치 검사를 영수증에서 구분했다.
- 이번 비교는 원 C1/E7의 미완료 세 head 절제를 같은 새 decoder와 후단에 적용한다. 기존 corrected66-way 직접 decoder 세 head245 결과와 구분한다. 또한 sparse 실제 경계 관측과 변경 없는 v4 hybrid 공급을 사전에 고정해 비교한다. 성능에 따라 head·threshold·seed·학습량을 고르거나 재시도하지 않는다.
- sparse BOUNDARY_ONLY는 admission을 통과한 실제 선 교점만 final fit pool에 넣는다. 없는 점을 N3로 채우지 않고 native 거리/LOO gate도 적용하지 않는다. 출력에 남는 미관측 non-H의 N3 좌표는 display이며 fit 관측으로 세지 않는다.
- CORNERWISE_HYBRID는 원 v4 native finite bank의 H∪{k} LOO로 후보를 평가한다. 변경 없는 ROLE hybrid를 원 v4 GT-free geometry와 완전 실사 봉인·cleanup 후, GT reference 전 비교한다. 과거 geometry는 검산 전용이다.
- finite4subset/SQPnP·IPPE/root/multisolution/refit/geometry 조건은 변경 없는 v4 solver다. 모든8arm에서 initial numeric R,t/projection/dimension prior를 사용하지 않는다. masked7arm은 초기 H를 생성·합의·refit에서 제외하고 새 pose가 나오면 H를 projection으로 대체하며 refit하지 않는다. no-mask native 대조만 H=[]이며 predicted H는 진단으로 분리한다.
- selected_corner_ids는 admission/공급 proposal ID다. 실제 eligible/used/fit_input_ids/final_inliers 및 NEW 여부를 따로 기록한다. 이미지 caption과 case 원행도 admitted 후보와 actual fit/inlier를 분리한다. 준비 중 발견한 새 statistics 변수 shadowing을 실사 실행 전에 수정하며, 이전 v4–v6에는 이 새 selected_corner_ids 반환 필드가 없으므로 이전 pose/통계 결과의 오류라고 주장하지 않는다.
- infer cleanup hook 예외도 독립적으로 수집하여 실패 영수증을 보존한다. complete seal만으로 GT score를 허용하지 않고 성공 cleanup이 기록된 완료 receipt가 함께 있어야 한다.
- 보호 snapshot은 연구 게시 commit424ae3b4d61e24b99662dbd28611b4c465852411 이후 tracked17017파일과 별도24감사 파일, 사용자 original main checkout의 기존 status/diff SHA를 포함한다. 원 영상·가중치·마감/기존319결과는 불변이다. 새 평가 cohort는 사용자 지정 clean153+moderate92=245이며 severe74는 제외한다.

소스 경계/깊이 감독 수정과 source CAL 통과는 독립 실사 physical ownership 또는 실제 위치·회전 개선의 증명이 아니다. 형상 좌표 source replay의 작은 오차, ROLE 채널의 초기 pose/projection replay witness 부족, 기존 GEOMETRIC_PROXY physical truth 미검증을 구분한다. 다음 단계는 사전 고정8arm의 실제245 fresh geometry·scoring·검산과 별도 quiet1500 full-path 비용 측정이다.
