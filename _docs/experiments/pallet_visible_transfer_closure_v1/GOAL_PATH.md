# 목적과 가지치기

| 행동 | 왜 필요한가 | 기존으로 대체 |
| --- | --- | --- |
| 66점 paired/난도/연속오차 | 동률이 같은 점 유지인지 진입·이탈 상쇄인지 구분 | 기존 frozen예측/참조 재계산 |
| 같은66점 legacy↔verified | 표본 구성과 참조 변경 효과 분리 | 같은prediction/fixedID/no-match-gate로 통일 |
| TRAIN217 native 타깃 추종 | 원래 감독도 못 따라간 것과 학습밖 전이 구분 | 일치하는 TRAIN예측 cache 없음:3frozen모델 추론 |
| 학습곡선·loss reduction | 학습량 개입 근거/경쟁설명 | 기존5epoch csv·loss코드 |
| 선택기/새모듈/전면대칭감사 | native2D 질문과 관계없음 | 제외 |
| 추가hard labels/모든지표비악화 | 이번 종료기준 아님 | 제외 |

가장 강한 반론: 작은 반복DEV의10px문턱에 맞춘 변경 아닌가? TRAIN 증거로 한 개입만 잠그고 모든 결과/손상을 보고한다. 그래도 독립 검증은 아니다. confidence 보존은 corrected좌표의 새 calibration을 뜻하지 않는다. 관찰적잔차는 원인 증명이 아니다.
