# v5 실제 미사용 선 관측의 사후 proxy 일치 진단

기하 봉인과 점수화가 완료된 동일245장의 저장 원행만 한 번 읽었다. 새 모델·PnP·optimizer·ray·학습·RGB·GT 자세 점수 계산은0이다.

선 오차는 고정 N3 phase의 기존 geometric-proxy native endpoint 두 개에서 관측 무한직선까지의 법선 거리 RMS다. 저장된 reference matched=True이고 두 endpoint가 모두 알려진 경우에만 기존8px 기준으로 일치/불일치를 나눴다.
이 값은 실제 물리 경계 소유권·가시선의 정확한 GT·선 방향의 support·실제3D경계 중 관측된 비율을 인증하지 않는다.

| 집합 | proxy≤8px | proxy>8px | 미상 |
|---|---:|---:|---:|
| 동일 IMAGE_ROLE 관측선 | 865 | 132 | 17 |
| 실제 미사용 선 factor pool | 654 | 122 | 17 |
| 채택 코너가 소비한 선 | 211 | 10 | 0 |
| 추가 제외 | 0 | 0 | 0 |
| 승인 NEW의 최종 선 inlier | 636 | 101 | 8 |
| 기본 반환의 raw 후보 선 inlier | 0 | 0 | 0 |

선 factor 단위는 고유 semantic edge1개다. 같은 선의 여러 query 수를 독립 선 inlier 수로 세지 않았다. 채택한 allowed boundary corner의 incident edge는 선 pool에서 제외한 것을 독립 확인했다.
최종 선 inlier는 승인 NEW에서만 집계했다. rank 실패·다중해·기본 반환의 best-candidate 선/점 inlier는 별도 raw 진단이며 출력 성공으로 처리하지 않았다.

모든 방법의 점 pool/실제 fit/승인 inlier의≤8/>8/unknown, 실제 반환 W/D분기, 사후 T/R/ADD 차이, 저장된 raw/승인 rank·layout은 ROWS와 CHECKS에 남겼다.
불일치 선과 큰 자세 오차의 동반은 연관이다. 현 proxy로 실제 물리 경계 소유권 오류의 원인 비율을 단정하거나 설정을 다시 선택하지 않는다.

[고정 프로토콜](LINE_QUALITY_PROTOCOL.json) · [245 원행](LINE_QUALITY_ROWS.jsonl.gz) · [검산](LINE_QUALITY_CHECKS.json)
