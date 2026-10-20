# IEEE Sensors 계열 8쪽 목표에 맞춘 표·그림 사용 안내

사용자가 말한 **IEEE Sensors Journal, 참고문헌 포함8쪽 목표**를 분량 가정으로 삼는다. 공식 페이지 제한을 확인·인용한 문서가 아니다. 원고나 슬라이드는 수정하지 않았다.

본문은 다음 최소 핵심만 우선 검토하고, 전체 수치와 출처는 보충자료에 둔다.

| 용도 | 본문용 후보 | 보충자료 |
|---|---|---|
| 평가 계약 | [tab01](latex/tab01_protocol.tex): PRIMARY192/CAL46/합성768·235/신규독립test0 | TABLE_01 전체 계약 및 출처 |
| 동일영상의 기본 비교 | [tab02](latex/tab02_primary.tex): 실제 R0, SubPix 단독, N3 단독, PF RAW, 두 PF+SubPix, N0, E0, D1 세seed | 네 모집단 전체분포, 영상별paired, recording/seed별·회복/손상 |
| 현재D1 적용성 | [tab03](latex/tab03_current_D1_backbones.tex): 3backbone×3seed; 미산출·가용n 함께 표시 | 공통146 보조표,60집계,96paired,후보coverage 및 typed비용 |
| 기전·한계 | [전체192 산점도](../figures/all192_scatter.png)와 [좋은 후보 미선택 사례](../figures/cases/case_03.jpg) 중 핵심2개그림 | 23사례, signed기여, recording기여, 원인coverage/UNKNOWN |
| ablation과 비용 | 공간과 중심주장에 따라 tab04–07에서 꼭 필요한행만 선택 | [tab04 단계](latex/tab04_stages.tex), [tab05 서로다른ablation](latex/tab05_distinct_ablations.tex), [tab06 recording](latex/tab06_recording.tex), [tab07 비용범위](latex/tab07_cost_scope.tex) 전체 |

본문에서 유지할 사실은 다음과 같다. D1의 높은 성공수가 평균위치·회전 우위는 아니다. 성공 후보부재와 좋은 후보미선택은 다른 병목이다. 상태별 관측·후단 선택의 구현과 해당 개발자료의 성능·진단을 중심으로 설명할 수 있으나, 독립 물리owner 해결이나 평균우월성·새실사 일반화를 입증했다고 쓸 수 없다. DOPE 평균은146가용집합 조건부이며 전체192의 비교평균으로 쓰지 않는다. COMMON/BI 정책과 no-B특징차단은 다른은행·다른질문이다.

원고 구성을 별도GPT에서 정할 때 교수님 의견·사용자가 준 원자료·이전reference 목록을 함께 읽고 문제정의·목표·기여를 다시 연결해야 한다. 이 환경에는 그 별도 대화 자료가 없어 의견이나 문헌 수정을 임의로 만들어 반영하지 않았다. [누락셀](MISSING_CELLS.csv)의 NOT_RUN/BLOCKED를0이나 유리한 과거수치로 채우지 않는다. 현재 결과에 맞춰 주장범위를 좁히고 유효한 수치만 사용한다.
