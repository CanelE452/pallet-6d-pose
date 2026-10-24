# V218 부위·native 계약 독립 검토

현재9개 weak part 관찰을 native D9로 복사할수없다. V159의 native0/1/4/5는 **상판(deck-board) 끝**, 2/3/6/7은 **바닥 runner 끝**이다. f0n6의 “오른쪽 상판 끝 아래쪽코너/끝면”, f5n7의 “상판 바깥옆면/끝경계”는 bottom-runner라는 뜻이 아니다. 이두사례는 nominal ref가crop안이어도 부위일치를추가확인해야한다. f6n6은같은모호함과 standing규약 문제가겹치고, f6n4/5도원crop밖nominal역할과의대응이불명확하다. 나머지4개도deck계열과서술상모순이없다는정도이며native학습승인이아니다. 원약관찰과ROI는그대로보존했다.

**해부학적위·세계위·화면위는별개다.** 모델 +Z는상판방향이다. 수식은 `deck_W=R_WM e_z`, `up_C=R_CW up_W`, `c_M=solve(R_CM,-t)`다. 고정된물리부위ID는카메라가바뀌어도같지만, 기존native8은카메라동적역할이다. “camera-independent native mapping”보다는 camera-independent **canonical part ID**와 q에의존하지않는 **camera-dependent native alias**를나누는것이정확하다. 회전가상fixture는standing에서deck-up과world-up이직교할수있고camera roll에서world-up이화면오른쪽이될수있음을보인다. 실제SOURCE축을q에맞춰추정하거나perm을찾지않았다.

V159저장CAD검산은8개명목꼭짓점이재료vertex이고 +Z가상판/-Z가runner임을보여준다. 이것은실제standing팔레트의initializer채널정렬이나물리좌표정답증명이아니다. V153의중심모델좌표적용은명시적adaptation이며공식MRnative8규약복원이아니다. [기존정의](../../query_visibility_native_part_support_feasibility_20261024_v159/native_definition_audit/DEFINITION_AND_FEASIBILITY.json)

지금진행가능한감독은 **질문으로주소화되는같은instance의구체적finitepart 약관찰**이다. 원 q/native/q8와영상으로어느판끝·edge를묻는지식별가능하고양성근거가있으면, 정확한3Dcorner나완벽mesh없이도그부위의visible/covered 보조감독을전향정의할수있다. q가ROI를정확히포함할필요는없다. 반대로임의나뭇결ROI는모델이무엇을묻는지알수없으므로그라벨을조건없는nativehead에넣을수없다. 이런경우ROI-conditioned/proposal 보조문제로보존하고배포시ROI출처도별도정의해야한다.

BEFORE/COVER/SHAM은같은원q와같은부위를유지하고, COVER의그부위opacity·SHAM의그부위불변·matched donor/alpha/주변근거를실제proof로닫는다. 그때생기는finitepart상태는native point D/E·owner·quality와별개다. 마스크0/depth0/다른물체/queryOFF는externalE가아니다. 기존FIT7은review후보이며admission0, DEV2와모든파생은모든fit분기에서제외한다.

합성은object-fixed part ID와생성기가선언한원native mapping/up규약을보존해SOURCE타깃을만든다. 공개SOURCE의GT/part sidecar는타깃·진단전용이며배포feature는원initializer q8/native/q/L/availability뿐이다. C61에이미있던native+otherXY/L의재포장은새정보나성능진전으로설명하지않는다. 새영상·라벨·pairedcontrol·학습·forward·보호평가·GT/CAD원본접근은0이다.
