# Task-risk robustness audit

원래 판정 `TASK_RISK_AL_NO_SIGNAL`은 유지한다. 학습·risk score 변경·평가 모집단 변경은 하지 않았다.

평가 145장 전부에 task-risk 실험 전부터 `MANUAL_REVIEW_REQUIRED` migration metadata가 존재했다. 현재 annotation blob과 이전 `d653dce`의 blob이 일치한다. 분류는 QA_PREEXISTING_FLAGGED 145, QA_CLEAN 0, QA_UNKNOWN 0이다. 따라서 사전 고정한 QA_CLEAN-only S1은 `NOT_ESTIMABLE`이며, 문제 한 장만 빼서 clean 결과라고 부를 수 없다. 이 flag는 기존 정본 축/annotation provenance 미완료 상태이며 145장의 수치 GT가 모두 틀렸다는 뜻이 아니다.

기존 CVaR90 수치를 동일 정의로 재현했다. `CVAR_INFLUENCE_PER_FRAME.json`은 모든 프레임 leave-one-out, worst-tail contribution 및 QA flag를, `CVAR_LEAVE_ONE_SESSION_OUT.json`은 session 제거 민감도를 담는다. `CAT_FRAME_REPORT.md`와 JSON은 매우 큰 2D 예측 오차가 이미 PnP 앞에서 발생했고 축만 바꿔서는 회복되지 않는다는 제한된 원인 분해를 제시한다.

추가 좌표 감사에서 원본640×480 영상 밖 오른쪽 x≈698–740의 box가 제안 모델 세 seed의 최고 score 검출로 선택되었음을 확인했다. 모든9점도 원본 오른쪽 경계 밖이며 여러 점은 640+100에 있다. 기존100px reflect padding 영역의 검출을 선택한 것이 PnP 앞의 구체적 실패다. 원본 팔레트 위치의 다른 후보들도 있었지만 score가 더 낮았다. 초기 index involution의 C2 이름 오류는 `CANONICAL_C2_DIAGNOSTIC_CORRECTION.json`에 정정했다. 실제 정본 yaw180 순서에서도 286.436/286.115/285.838px 오류가 남는다. 후보 교체나 padding 필터는 적용하지 않았다.

감사 판정: `TASK_RISK_RESULT_QA_SENSITIVE_REQUIRES_CAUTION`. 이는 QA flag가 제안 방법의 실패 원인이라는 주장도, 원래 FAIL을 뒤집는 주장도 아니다. 다음 architecture 비교와 별개의 비학습 감사이다.
