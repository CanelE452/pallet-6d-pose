# V217 독립 해석

현재 근거는 native 가시성 정답을 늘렸다는 결과가 아니라, **무엇을 묻는 q인지와 어느 인스턴스·부위가 보이는지를 분리해야 한다는 SOURCE 진단**이다. 고유입력은96행/U11/FIT64·DEV32이고, 모든GT후보를 보존한 반복표가120행/U17이다.

V216의 weak part D9는 유효한 약관찰로 남지만 native point D9나 새학습타깃9개를 뜻하지 않는다. 특히 frame6/candidate3의 native4/5/6 nominal reference는 원crop밖이다. 각각의 약315.710/324.801/256.914px 변위는 명목 reference 진단이며 실제물리좌표오차라는 뜻이 아니다. crop안의 판·블록이 보인다는 사실만으로 그멀리있는 nominal역할이 보인다고 할수없다. [독립기하검산](../independent_actual_nominal_query_review_c/REVIEW.json)

Root와 A의 봉인관찰에는 실제팔레트와 넓게정렬되는 mask도 있고, q가 다른상자·손잡이·배경을 묻거나 visible색이 전경/덮개외관에 걸린 경우도 있다. C는 이번해석에서 새이미지를 보지 않았다. visible-mask는 depth기반 proxy이고 물체/expectedpart/native D와 동치가 아니다. [Root15view](../ROOT_ACTUAL_ALL15_GT_OVERLAY_VISUAL_CORRESPONDENCE_QA.json), [A독립관찰](../actual_GT_overlay_correspondence_observations_a/ALL15_CANDIDATE_AND_FRAME12_OBSERVATIONS.json), [pixel검산](../independent_actual_pixel_review_b/actual_review_v1/REVIEW.json)

[BOP/MR6D정의 확인](../ROOT_PRIMARY_SOURCE_VISIBILITY_DEFINITION_CHECK.json)은 missingdepth 처리 등이 proxy차이의 가능한 설명임을 보여주지만 실제MR6D mode/delta나 각픽셀의원인을 확인하지 않았다. 이자료로 모든GT가 틀렸다거나 기존모델문제가 순수transfer라고 결론낼수없다. 새D/E·원q/L보정·fit·성능개선주장은 없다.

다음 고정4카드에서는 원crop의 feature와 후보별nominal역할을 함께보되 GTbest 선택이나 recenter를 하지않는다. instance·part·query-support의 양성약근거와 불확실성을 기록하는것이 실용적이다. 완벽한physicalgold를 모든weak관찰의 일괄선결조건으로 추가하지않는다.
