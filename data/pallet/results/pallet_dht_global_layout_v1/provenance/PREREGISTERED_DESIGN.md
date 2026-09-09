# 팔레트 전체 배치 선택 — 실행 전 설계

2026-09-09. 사용자가 승인한 빠른 점·Deep Hough 결합 실험이다. 기존 joint/incidence/decoder 실험은 이미 두 분기를 연결했지만 안정적인 개선을 보이지 않았다. 이번에 추가로 검증할 요소는 동일 역할의 선 양 끝점을 함께 평가하고, 8점의 projective-cuboid 배치를 통째로 선택하는 것이다. 이 작은 추론 실험의 성공도 새로운 end-to-end 학습의 성공을 보장하지 않는다.

기존 decoder 캐시 2,879장(합성 train2,048/val512, 실사 DEV319), 원본 이미지·GT·예측을 보존한다. CNN 학습/forward는 0이다. 실사 GT로 계수나 탐색 폭을 조정하지 않는다. GT 감사에서 외곽 부근으로 보인 점은 국소 오차와 가림·번호 해석 불확실성이 남는 정성 검토 집합이며, 정확 GT 인증 집합으로 부르지 않는다.

각 의미 코너의 후보는 기존 8개 코너 위치와 해당 역할의 선 교점 상위8개(선 unary만으로 pruning)로 최대16개다. 센터8번은 바꾸지 않는다. 선은 기존 역할별 상위4 Hough mode를 사용하며 보존된4개 안에서 확률을 재정규화한다. 독립 선택은 코너별 incident line mixture를, 공동 선택은 각 모서리의 두 점이 같은 line mode를 만족하는 mixture를 사용한다. 두 line 항을 중복 가산하지 않는다. 이동 prior와 각 항은 평균으로 정규화한다.

공동 선택은 고정4순서·beam64의 근사 탐색과 원래 전체 배치/그 C4 회전4종/독립 해를 명시 후보로 사용한다. 영상에서 2D 평행을 강요하지 않고 단위 cuboid의 정규화 DLT 투영 잔차를 추가한다. 기하만으로 C4 번호를 구별할 수 없으며, 역할별 선 근거가 필요하다. 퇴화·혼합 depth 배치는 제외하되 원래 예측은 예외 fallback으로 남기고 이를 기록한다. beam은 전역 최적해 보장이 없다.

합성 train에서 SHA256(`pallet_dht_global_layout_v1:calibration:20260909:`+frame_id) 순으로256장을 고른다. clean 합성의 같은-ID matched visible 코너 평균 오차/원본 대각선만으로9개 계수 조합을 선택한다: point [.25,1,4] × geometry [0,.25,1]. 1e-12 이내 동점이면 point 큰 값, geometry 작은 값을 택한다. 선택 JSON과 소스 hash를 실사 실행 전에 고정한다. 임의 점 교란이나 실사 사례 최적화는 선택 목적에 넣지 않는다.

평가 arm은 baseline, 같은 후보·선택된 point 계수의 independent, 선택된 global, geometry=0인 global_shared_only다. 합성 val512와 실사319장 전체를 같은 box match/point mask로 평가한다. 공식은 same-ID 원본 픽셀 오차이며 centroid도 기존처럼 포함한다. 실사는 309 matched/2,738 observed points의 baseline P90 41.48732863482036px를 재현해야 한다. 누락80점은 그대로 표시한다. 해상도별·세션별 및 시각 검토 점 진단을 함께 제공한다.

사전 진행 기준은 모두 충족해야 한다: global의 전체 median/P90이 baseline 및 independent보다 나빠지지 않을 것; baseline>20px 점의 평균오차를10% 이상 줄일 것; baseline<=10px 점의 >10px 이탈이1% 이하이고 independent보다 많지 않을 것; global−baseline 및 global−independent의 세션 bootstrap95% 상한이0 미만일 것(13session/20,000회/seed20260909). 단일 이전 모델 캐시 기반이라 충족해도 반복 seed의 안정성 입증은 아니며 큰 학습에 진입할 근거 수준이다. 불충족이면 같은 실사에서 계수를 다시 찾지 않는다. 후보 부족·역할 선 오류·점 prior·검색/배치 점수 오선택을 가능한 범위에서 분리한다. GT oracle 후보 거리는 사후 가능성 진단이며 실제 추론 성능으로 보고하지 않는다.

구현 검사는 선 mode 공유, 좌표/affine, 원근 투영·퇴화, ID 회전 불변성, baseline/centroid 보존을 독립 fixture로 수행한다. 원본 SHA, calibration freeze, 정량 독립 감사, 실제 HTML screenshot 검토와 브라우저 표시 확인 후 완료를 Discord로 알린다. 결과를 보고 기준을 완화하지 않는다.
