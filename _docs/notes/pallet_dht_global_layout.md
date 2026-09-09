# 팔레트 전체 배치 선택 — 실행 전 설계

2026-09-09. 사용자가 승인한 빠른 점·Deep Hough 결합 실험이다. 기존 joint/incidence/decoder 실험은 이미 두 분기를 연결했지만 안정적인 개선을 보이지 않았다. 이번에 추가로 검증할 요소는 동일 역할의 선 양 끝점을 함께 평가하고, 8점의 projective-cuboid 배치를 통째로 선택하는 것이다. 이 작은 추론 실험의 성공도 새로운 end-to-end 학습의 성공을 보장하지 않는다.

기존 decoder 캐시 2,879장(합성 train2,048/val512, 실사 DEV319), 원본 이미지·GT·예측을 보존한다. CNN 학습/forward는 0이다. 실사 GT로 계수나 탐색 폭을 조정하지 않는다. GT 감사에서 외곽 부근으로 보인 점은 국소 오차와 가림·번호 해석 불확실성이 남는 정성 검토 집합이며, 정확 GT 인증 집합으로 부르지 않는다.

각 의미 코너의 후보는 기존 8개 코너 위치와 해당 역할의 선 교점 상위8개(선 unary만으로 pruning)로 최대16개다. 센터8번은 바꾸지 않는다. 선은 기존 역할별 상위4 Hough mode를 사용하며 보존된4개 안에서 확률을 재정규화한다. 독립 선택은 코너별 incident line mixture를, 공동 선택은 각 모서리의 두 점이 같은 line mode를 만족하는 mixture를 사용한다. 두 line 항을 중복 가산하지 않는다. 이동 prior와 각 항은 평균으로 정규화한다.

공동 선택은 고정4순서·beam64의 근사 탐색과 원래 전체 배치/그 C4 회전4종/독립 해를 명시 후보로 사용한다. 영상에서 2D 평행을 강요하지 않고 단위 cuboid의 정규화 DLT 투영 잔차를 추가한다. 기하만으로 C4 번호를 구별할 수 없으며, 역할별 선 근거가 필요하다. 퇴화·혼합 depth 배치는 제외하되 원래 예측은 예외 fallback으로 남기고 이를 기록한다. beam은 전역 최적해 보장이 없다.

합성 train에서 SHA256(`pallet_dht_global_layout_v1:calibration:20260909:`+frame_id) 순으로256장을 고른다. clean 합성의 같은-ID matched visible 코너 평균 오차/원본 대각선만으로9개 계수 조합을 선택한다: point [.25,1,4] × geometry [0,.25,1]. 1e-12 이내 동점이면 point 큰 값, geometry 작은 값을 택한다. 선택 JSON과 소스 hash를 실사 실행 전에 고정한다. 임의 점 교란이나 실사 사례 최적화는 선택 목적에 넣지 않는다.

평가 arm은 baseline, 같은 후보·선택된 point 계수의 independent, 선택된 global, geometry=0인 global_shared_only다. 합성 val512와 실사319장 전체를 같은 box match/point mask로 평가한다. 공식은 same-ID 원본 픽셀 오차이며 centroid도 기존처럼 포함한다. 실사는 309 matched/2,738 observed points의 baseline P90 41.48732863482036px를 재현해야 한다. 누락80점은 그대로 표시한다. 해상도별·세션별 및 시각 검토 점 진단을 함께 제공한다.

사전 진행 기준은 모두 충족해야 한다: global의 전체 median/P90이 baseline 및 independent보다 나빠지지 않을 것; baseline>20px 점의 평균오차를10% 이상 줄일 것; baseline<=10px 점의 >10px 이탈이1% 이하이고 independent보다 많지 않을 것; global−baseline 및 global−independent의 세션 bootstrap95% 상한이0 미만일 것(13session/20,000회/seed20260909). 단일 이전 모델 캐시 기반이라 충족해도 반복 seed의 안정성 입증은 아니며 큰 학습에 진입할 근거 수준이다. 불충족이면 같은 실사에서 계수를 다시 찾지 않는다. 후보 부족·역할 선 오류·점 prior·검색/배치 점수 오선택을 가능한 범위에서 분리한다. GT oracle 후보 거리는 사후 가능성 진단이며 실제 추론 성능으로 보고하지 않는다.

구현 검사는 선 mode 공유, 좌표/affine, 원근 투영·퇴화, ID 회전 불변성, baseline/centroid 보존을 독립 fixture로 수행한다. 원본 SHA, calibration freeze, 정량 독립 감사, 실제 HTML screenshot 검토와 브라우저 표시 확인 후 완료를 Discord로 알린다. 결과를 보고 기준을 완화하지 않는다.

## 결과 — 성능 개선 기준 불충족

합성256장 중 기존 mask의2,022코너로9개 조합을 비교해 `w_point=4,w_geom=1`을 선택했다. calibration29.87초, 실제512+319 캐시 선택 약40초, 신규 CNN 학습/forward0. 실사309매칭·2,738/2,818관측점, 누락80점과GT원본을 그대로 유지했다.

| 실사319장 | Median(px) | P90(px) | 평균(px) |
|---|---:|---:|---:|
| 기존 joint 모델 점 출력 | 6.897 | 41.487 | 22.454 |
| 같은 후보 독립 선택 | 13.341 | 105.697 | 38.239 |
| 전체 배치 선택 | 11.444 | 79.682 | 31.658 |
| 전체 선택, DLT잔차 가중치0 | 13.161 | 94.544 | 36.172 |

전체선택−기존의 frame평균오차차이는+9.190px,13session bootstrap95%CI[+5.784,+13.372]다. 전체선택−독립선택은−6.558px,CI[−10.651,−3.553]다. 공동모드와DLT가 독립선택의 악화를 줄였지만 기존 정확도를 회복하지 못했다. baseline<=10px 1,711점 중577점(33.72%)이10px를 넘었고, baseline>20px 499점의 평균도92.547→100.197px로악화했다. 따라서 등록기준 불충족, 추가 큰 학습에 진입할 개선 근거 없음이다.

합성val512에서도P90 7.522→32.994px,median2.039→6.933px로악화했다. 이번 실패를 실사 도메인 차이나 수동GT 오차만으로 설명할 수 없다. 실사 직접클릭 기록643점 P90도28.812→74.661px였다. 이전GT-only검토에서경계부근으로보인17점은median244.143→240.859px지만P90431.121→434.022px로큰오류가남는다. 이는 픽셀 정확도를 인증한 집합의 결과가 아니다.

## 선택 실패에서 확인한 점

사용자 사례 `eval_pallet07:1778652166837872128`에 원래8점의yaw90 번호배치가 명시 후보로 존재한다. 그 배치는GT same-ID중앙오차20.704px인데, 실제선택은278.061px이다(기존270.065px). 후보를 못 생성한 경우와 다르다. 낮을수록 좋은 점수에서 yaw90은16.191,선택배치는5.465였다. yaw90의 이동 prior불이익10.363이 크고, 역할선 항도yaw90 5.822 vs기존오배치5.758로 올바른번호를구별하지못한다. 네C4배치의DLT비용은모두약.00552로같다. 따라서 단순히 beam을 넓히거나 점 이동 prior만 없애면 이 사례가 해결된다고 결론 내릴 수 없다.

전체실사관측corner2,429개에서, 원래48교점+8점에10px이내후보가있는점은1,986개, K16pruning후1,791개다. 195개에서좋은후보를잃는다. 남은16후보에10px이내점이있어도선택오차가20px를넘는점은396개다. GT-nearest퍼코너후보는공동으로유효한배치라는보장이없는사후oracle이다. 이전고정19프레임의GT-nearest배치실제점수검사에서는19개모두현재선택보다GT에가까우면서더불리한점수를받았다. 이19개에서의직접증거는점수오선택이며, 전체탐색누락0을증명한것은아니다.

일부선택에서서로다른코너ID가동일영상점에놓인다. DLT는soft투영잔차와수치guard로쓰였고,오차0의물리적인8개모서리검증으로쓰이지않았다. 완전정면에선실제로앞뒤투영점이겹칠수있으므로모든중복을단순금지할수도없다. 생성한정면/원근기하검사는통과했지만실제정면성proxy144장의P90은25.969→53.086px다. 기존GT면적비proxy와재구성앙각8그룹을그대로재사용한설명진단이며측정yaw나완전정면별정확도인증이아니다.

이 결과는 고정 점수의 전체 배치 선택기를 채택하지 않을 근거다. 점·선 결합 자체가 불가능하다는 증거로 확대하지 않는다. 후속 학습을 검토한다면, 번호 재배치와 선 끝점의 정확성을 구별하는 배치 점수와 양호한 점 보존을 함께 검증해야 한다. 현재의 합성계수 선택도9개중최선일뿐baseline보다개선했다는뜻이아니다. 동일calibration256장·2022코너에서baseline정규화평균오차.00559238인데선택계수는.01695915로3.03배나쁘고9개모두baseline보다못하다. 각bank에baseline배치가있어도점수가이를보존하지못했다. 별도사후 [calibration대조](../../data/pallet/results/pallet_dht_global_layout_v1/CALIBRATION_BASELINE_DIAGNOSTIC.json)이며이미고정한선택/실사수치는변경하지않았다.

25생성기하tests PASS. 독립검사831장×3선택arm의저장점수2,493개재계산최대차1.07e−14,GT/분모/centroid/mask/선택좌표일치. 이것은 계산무결성PASS이며성능PASS가아니다. [결과JSON](../../data/pallet/results/pallet_dht_global_layout_v1/RESULTS.json),[사후점수진단](../../data/pallet/results/pallet_dht_global_layout_v1/POSTHOC_SEARCH_SCORE_DIAGNOSIS.json),[비교HTML](../../data/pallet/results/pallet_dht_global_layout_v1/index.html).

완료 전달: 2026-09-08T22:49:34.334104Z (한국07:49). Chrome창 `0x62003d9` 실제표시 확인, DiscordHTTP204. HTML319decode/1276overlay/1914후보QA와root5최종screenshots검토PASS. 원입력6471불변 및최종23artifact/12source SHA재검증. [완료인계](../handoff/2026-09-09_pallet_dht_global_layout_v1.md).
