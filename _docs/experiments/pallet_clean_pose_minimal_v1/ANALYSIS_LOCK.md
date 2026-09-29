# 후속 분석 규칙

이 규칙은 기존 OCC 결과를 본 뒤 추가된 분석이다. 새 R0/OLD_REF/CLEAR GEO 대조 결과를 보기 전에 고정하며 과거 사전등록으로 소급하지 않는다.

## 비교와 구성 단순화

R0, OLD_REF217, RAW_CLEAR42, REF_CLEAR42, RAW_OCC42, REF_OCC42 각각의 D9/GEO를 모두 보고한다. 기존43 OCC RAW/REF의 D9/GEO도 누락 없이 보고한다. 가장 강한 단순 대조는 사후에 유리한 하나만 고르지 않고 R0_GEO 및 같은 seed RAW_GEO/CLEAR_GEO에 대한 모든 pairwise T/R 관계로 제시한다. T와 R를 합산한 새 목적점수나 합격 개선율을 만들지 않는다. 지표가 서로 대립하면 Pareto/trade-off로 남긴다.

기존217과clean78 비교는 이미지 집합·고유 개수·반복노출과 공통 post-affine support 계약도 달라 'clean 여부만의 효과'로 해석하지 않는다. seed43 OCC만으로 입력 가림 효과가 반복됐다고 말하지 않는다.

## 불확실성과 기록 의존성

전수128 및 자연99의 각 프레임을 유지한다. 실패는 기존 extended-real quantile 규칙을 사용하며 common-valid paired 변화와 전체 valid/분모를 별도 보고한다.

같은 frame pair를 유지해 recording cluster를 복원추출하는 2000회 percentile bootstrap(고정 RNG20260929)을 설명용으로 보고한다. 점/코너 독립 표본 검정이나 새 독립 평가의 증거로 쓰지 않는다. recording 개수가 작고 크기가 불균형함을 명시한다. 모든 leave-one-recording-out 결과 및 recording별 T/R median/P90과 paired 방향을 보존한다. 중앙값 차이와 paired차이의 중앙값을 분리한다. 신뢰구간이0을 넘는지 여부만으로 완료/성공을 결정하지 않는다.

## 사례 선택

각 핵심 비교에서 같은99장 중 (1) T/R가 둘 다 개선된 프레임을 T 감소량 순, (2) 둘 다 악화된 프레임을 T 증가량 순, (3) 최종 T 최대, (4) 최종 R 최대 순으로 각각 상위2개를 선택한다. 동률은 고정 frame ID 사전순이다. 실패 프레임은 별도 전수 기록한다. 좋은 예시만 남기지 않고 중복 사례 ID도 표시한다. 이 순서는 사례 설명용이며 평가 모집단이나 모델/하이퍼파라미터 선정에 사용하지 않는다.

원본 RGB·좌표·checkpoint는 공개하지 않는다. 공개 보고서에는 선택기준과 ID/오차/paired 변화, 숫자 그림을 넣는다. 필요하면 로컬 전용 HTML/overlay에서 동일 사례를 확인하되 Git 공개 범위와 분리한다. 6D 참조는 기존 annotation 기반 기하 재구성이지 독립 물리 측정이 아니다.
